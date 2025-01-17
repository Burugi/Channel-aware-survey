import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.decomposition import SeriesDecomposition

## Dlienar의 individual과 channel independence는 비슷하지만 약간 다른 개념이다.

class DLinearModel(BaseTimeSeriesModel):
    """DLinear 모델 구현"""
    
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.kernel_size = self.config.get('kernel_size', 25)
        
        # Decomposition
        self.decomposition = SeriesDecomposition(self.kernel_size)
        
        if self.channel_independence:
            # 각 feature별 독립적인 Linear layers
            self.feature_linears = nn.ModuleList([
                nn.ModuleDict({
                    'seasonal': nn.Linear(self.input_len, self.pred_len),
                    'trend': nn.Linear(self.input_len, self.pred_len)
                })
                for _ in range(self.base_features)
            ])
        else:
            # Channel dependence mode
            self.Linear_Seasonal = nn.Linear(self.input_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        # [Batch, seq_len, 1] -> [Batch, 1, seq_len]
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        
        # Linear transformation
        seasonal_output = self.feature_linears[feature_idx]['seasonal'](seasonal_init)
        trend_output = self.feature_linears[feature_idx]['trend'](trend_init)
        
        # [Batch, 1, pred_len] -> [Batch, pred_len, 1]
        x = (seasonal_output + trend_output).permute(0, 2, 1)
        
        return x

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        # [Batch, seq_len, num_features] -> [Batch, num_features, seq_len]
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        
        # Linear transformation
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        # [Batch, num_features, pred_len] -> [Batch, pred_len, num_features]
        x = (seasonal_output + trend_output).permute(0, 2, 1)
        
        return x