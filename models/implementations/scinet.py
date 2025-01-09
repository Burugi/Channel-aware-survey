import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.base_model import BaseTimeSeriesModel
from layers.scinet import EncoderTree

class SCINetModel(BaseTimeSeriesModel):
    """SCINet Model for Time Series Forecasting"""
    
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.hid_size = self.config.get('hidden_size', 1)
        self.num_levels = self.config.get('num_levels', 3)
        self.groups = self.config.get('groups', 1)
        self.kernel = self.config.get('kernel', 5)
        self.dropout = self.config.get('dropout', 0.5)
        self.positional_encoding = self.config.get('positional_encoding', True)
        self.RIN = self.config.get('RIN', False)
        self.channel_independence = self.config.get('channel_independence', False)
        
        # Channel independence에 따른 입출력 차원 설정
        self.enc_in = 1 if self.channel_independence else self.num_features
        self.dec_out = 1 if self.channel_independence else self.base_features
        
        # Encoder Tree
        self.encoder = EncoderTree(
            in_planes=self.enc_in,
            num_levels=self.num_levels,
            kernel_size=self.kernel,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hid_size,
            INN=True
        )
        
        # Projection layers
        self.projection_time = nn.Linear(self.input_len, self.pred_len)
        
        if not self.channel_independence and self.num_features > 1:
            self.projection_feat = nn.Linear(self.num_features, self.base_features)
        
        # Positional Encoding
        if self.positional_encoding:
            self.pe_hidden_size = self.enc_in + (self.enc_in % 2)
            num_timescales = self.pe_hidden_size // 2
            max_timescale = 10000.0
            min_timescale = 1.0
            
            log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
            inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
            )
            self.register_buffer('inv_timescales', inv_timescales)
        
        # RevIN
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.enc_in))
            
    def get_position_encoding(self, x):
        """위치 인코딩 생성"""
        if not self.positional_encoding:
            return 0
            
        max_length = x.size(1)
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
        
        return signal[:, :, :self.enc_in]
            
    def forward(self, x):
        """순전파"""
        original_batch_size = x.size(0)
        
        if self.channel_independence:
            # Reshape for channel independence
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
        
        # RevIN
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            x = x * self.affine_weight + self.affine_bias
            
        # Positional Encoding
        x = x + self.get_position_encoding(x)
        
        # Encoder
        res = x
        x = self.encoder(x)
        x = x + res  # Skip connection
        
        # Project time dimension
        x = x.transpose(1, 2)  # [batch, feature, time]
        x = self.projection_time(x)  # [batch, feature, pred_len]
        x = x.transpose(1, 2)  # [batch, pred_len, feature]
        
        if self.channel_independence:
            # Reshape back to original dimensions
            x = x.reshape(original_batch_size, self.base_features, self.pred_len, -1)
            x = x.squeeze(-1).permute(0, 2, 1)  # [batch, pred_len, num_features]
        else:
            # Project feature dimension if needed
            if self.num_features > 1:
                x = self.projection_feat(x)
        
        # Reverse RevIN
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means
        
        return x

    def training_step(self, batch):
        """단일 학습 스텝"""
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        
        # 순전파
        y_pred = self(x)
        
        if self.channel_independence:
            # Channel independence mode에서는 각 feature별 loss를 평균
            loss = 0
            feature_losses = {}
            for i in range(self.base_features):
                feature_loss = self.loss_fn(y_pred[..., i], y[..., i])
                loss += feature_loss
                feature_losses[f'feature_{i}'] = feature_loss.item()
            loss = loss / self.base_features
            return {'loss': loss.item(), 'feature_losses': feature_losses}
        else:
            # Channel dependence mode에서는 전체 feature에 대한 단일 loss
            loss = self.loss_fn(y_pred, y)
            return {'loss': loss.item()}