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
        
        # feature 차원 설정 수정
        if self.channel_independence:
            self.in_planes = 1
            self.groups = 1
            self.out_planes = 1
        else:
            self.in_planes = self.base_features  # num_features 대신 base_features 사용
            config_groups = self.config.get('groups', 1)
            if config_groups > self.in_planes:
                self.groups = self.in_planes
            else:
                self.groups = config_groups if self.in_planes % config_groups == 0 else 1
            self.out_planes = self.base_features
        
        if self.channel_independence:
            # Feature별 독립적인 모델 구성
            self.feature_encoders = nn.ModuleList([
                EncoderTree(
                    in_planes=self.in_planes,
                    num_levels=self.num_levels,
                    kernel_size=self.kernel,
                    dropout=self.dropout,
                    groups=1,
                    hidden_size=self.hid_size,
                    INN=True
                ) for _ in range(self.base_features)
            ])
            
            self.feature_projections = nn.ModuleList([
                nn.Linear(self.input_len, self.pred_len)
                for _ in range(self.base_features)
            ])
            
            # RIN parameters per feature
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
            # Channel dependence mode
            self.encoder = EncoderTree(
                in_planes=self.in_planes,
                num_levels=self.num_levels,
                kernel_size=self.kernel,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hid_size,
                INN=True
            )
            
            self.projection = nn.Linear(self.input_len, self.pred_len)
            
            if self.RIN:
                self.affine_weight = nn.Parameter(torch.ones(1, 1, self.in_planes))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.in_planes))
                
        # Positional encoding setup
        if self.positional_encoding:
            if self.channel_independence:
                self.feature_pe_hidden_sizes = [1 + (1 % 2) for _ in range(self.base_features)]
                self.feature_inv_timescales = nn.ParameterList([
                    self._create_inv_timescales(1) for _ in range(self.base_features)
                ])
            else:
                self.pe_hidden_size = self.in_planes + (self.in_planes % 2)
                self.inv_timescales = self._create_inv_timescales(self.in_planes)

    def _create_inv_timescales(self, dim):
        """Create inverse timescales for positional encoding"""
        num_timescales = (dim + (dim % 2)) // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        return nn.Parameter(inv_timescales, requires_grad=False)

    def _get_position_encoding(self, x, feature_idx=None):
        """위치 인코딩 생성"""
        if not self.positional_encoding:
            return 0
            
        max_length = x.size(1)
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        
        if self.channel_independence:
            inv_timescales = self.feature_inv_timescales[feature_idx]
            pe_hidden_size = self.feature_pe_hidden_sizes[feature_idx]
        else:
            inv_timescales = self.inv_timescales
            pe_hidden_size = self.pe_hidden_size
            
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, pe_hidden_size % 2))
        signal = signal.view(1, max_length, pe_hidden_size)
        
        return signal

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        # RIN 적용
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = x * self.feature_affine_weights[feature_idx] + self.feature_affine_biases[feature_idx]

        # Positional encoding
        if self.positional_encoding:
            pe = self._get_position_encoding(x, feature_idx)
            if pe.shape[2] > x.shape[2]:
                x = x + pe[:, :, :-1]
            else:
                x = x + pe

        # Encoder tree
        x = self.feature_encoders[feature_idx](x)
        
        # Projection
        x = x.permute(0, 2, 1)
        x = self.feature_projections[feature_idx](x)
        x = x.permute(0, 2, 1)

        # Reverse RIN
        if self.RIN:
            x = x - self.feature_affine_biases[feature_idx]
            x = x / (self.feature_affine_weights[feature_idx] + 1e-10)
            x = x * stdev
            x = x + means

        return x

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        # RIN 적용
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = x * self.affine_weight + self.affine_bias

        # Positional encoding
        if self.positional_encoding:
            pe = self._get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x = x + pe[:, :, :-1]
            else:
                x = x + pe

        # Encoder tree
        x = self.encoder(x)
        
        # Projection
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        # Reverse RIN
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        return x

    def forward(self, x):
        """순전파"""
        if self.channel_independence:
            outputs = []
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]
                out = self._forward_single_feature(feature_input, i)
                outputs.append(out)
            
            predictions = torch.cat(outputs, dim=-1)  # (batch_size, pred_len, base_features)
        else:
            predictions = self._forward_all_features(x)  # (batch_size, pred_len, base_features)
        
        return predictions