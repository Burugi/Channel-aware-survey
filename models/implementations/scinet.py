import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseTimeSeriesModel
from layers.scinet import EncoderTree

class SCINetModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.hid_size = self.config.get('hidden_size', 1)
        
        # input_len을 고려하여 num_levels 설정
        max_levels = int(np.log2(self.input_len))
        default_levels = min(3, max_levels)
        self.num_levels = min(self.config.get('num_levels', default_levels), max_levels)
        
        self.kernel = self.config.get('kernel', 5)
        self.dropout = self.config.get('dropout', 0.5)
        self.positional_encoding = self.config.get('positional_encoding', True)
        self.RIN = self.config.get('RIN', False)
        
        if self.channel_independence:
            # CI mode: 각 feature별 독립적 처리
            self.feature_encoders = nn.ModuleList([
                EncoderTree(
                    in_planes=1,  # 단일 feature
                    num_levels=self.num_levels,
                    kernel_size=self.kernel,
                    dropout=self.dropout,
                    groups=1,
                    hidden_size=self.hid_size,
                    INN=True
                ) for _ in range(self.base_features)
            ])
            
            # Feature별 projection layer
            self.feature_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_len, self.input_len // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.input_len // 2, self.pred_len)
                ) for _ in range(self.base_features)
            ])
            
            # Feature별 RIN parameters
            if self.RIN:
                self.feature_affine_weights = nn.ParameterList([
                    nn.Parameter(torch.ones(1, 1, 1))
                    for _ in range(self.base_features)
                ])
                self.feature_affine_biases = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, 1, 1))
                    for _ in range(self.base_features)
                ])
        else:
            # CD mode: 모든 feature 통합 처리
            # groups 설정 수정
            self.in_planes = self.num_features
            config_groups = self.config.get('groups', 1)
            self.groups = 1  # default to 1 for safety
            if config_groups > 1:
                # Ensure groups divides in_planes evenly
                if self.in_planes % config_groups == 0:
                    self.groups = config_groups
            
            self.encoder = EncoderTree(
                in_planes=self.in_planes,
                num_levels=self.num_levels,
                kernel_size=self.kernel,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hid_size,
                INN=True
            )
            
            # 통합 projection layer (feature 수 명시적 처리)
            self.projector = nn.Sequential(
                nn.Linear(self.input_len, self.input_len // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.input_len // 2, self.pred_len)
            )
            
            # Feature dimension projection
            self.feature_proj = nn.Linear(self.num_features, self.base_features)
            
            # CD mode RIN parameters
            if self.RIN:
                self.affine_weight = nn.Parameter(torch.ones(1, 1, self.base_features))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.base_features))

    def _apply_RIN(self, x, feature_idx=None):
        """Reversible Instance Normalization 적용"""
        # 평균과 표준편차 계산
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Affine 변환 적용
        if self.channel_independence:
            weight = self.feature_affine_weights[feature_idx]
            bias = self.feature_affine_biases[feature_idx]
        else:
            weight = self.affine_weight
            bias = self.affine_bias
            
        x = x * weight + bias
        
        return x, means, stdev

    def _reverse_RIN(self, x, means, stdev, feature_idx=None):
        """RIN 역변환 적용"""
        if self.channel_independence:
            weight = self.feature_affine_weights[feature_idx]
            bias = self.feature_affine_biases[feature_idx]
        else:
            weight = self.affine_weight
            bias = self.affine_bias
            
        x = x - bias
        x = x / (weight + 1e-10)
        x = x * stdev
        x = x + means
        
        return x

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.shape[0]
        
        # RIN 적용
        if self.RIN:
            x, means, stdev = self._apply_RIN(x, feature_idx)
        
        # Encoder tree 통과
        encoded = self.feature_encoders[feature_idx](x)
        
        # Projection to prediction length
        encoded = encoded.permute(0, 2, 1)  # [batch, 1, seq_len]
        predictions = self.feature_projectors[feature_idx](encoded)
        predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, 1]
        
        # RIN 역변환
        if self.RIN:
            predictions = self._reverse_RIN(predictions, means, stdev, feature_idx)
        
        return predictions

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 통합 처리"""
        batch_size = x.shape[0]
        
        # RIN 적용 (원본 feature 수에 맞춰)
        if self.RIN:
            x, means, stdev = self._apply_RIN(x)
        
        # Encoder tree 통과
        encoded = self.encoder(x)
        
        # Time dimension projection
        encoded = encoded.permute(0, 2, 1)  # [batch, feat, seq_len]
        predictions = self.projector(encoded)  # [batch, feat, pred_len]
        
        # Feature dimension projection
        predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, feat]
        predictions = self.feature_proj(predictions)  # [batch, pred_len, base_features]
        
        # RIN 역변환 (base_features 수에 맞춰)
        if self.RIN:
            # means와 stdev를 base_features 크기에 맞게 조정
            means = self.feature_proj(means.squeeze(1)).unsqueeze(1)
            stdev = self.feature_proj(stdev.squeeze(1)).unsqueeze(1)
            predictions = self._reverse_RIN(predictions, means, stdev)
        
        return predictions

    def forward(self, x):
        """순전파"""
        if self.channel_independence:
            # Feature별 독립 처리
            outputs = []
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]  # [batch, seq_len, 1]
                feature_output = self._forward_single_feature(feature_input, i)
                outputs.append(feature_output)
            
            # 모든 feature의 예측을 결합
            predictions = torch.cat(outputs, dim=-1)  # [batch, pred_len, num_features]
        else:
            # 전체 feature 통합 처리
            predictions = self._forward_all_features(x)  # [batch, pred_len, num_features]
        
        return predictions