import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.decomposition import SeriesDecomposition

class DLinearModel(BaseTimeSeriesModel):
    """Decomposition-Linear Model for Time Series Forecasting"""
    
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.kernel_size = self.config.get('kernel_size', 25)
        self.individual = self.config.get('individual', True)
        
        # Decomposition
        self.decomposition = SeriesDecomposition(self.kernel_size)
        
        # Individual mode에서는 각 채널별로 별도의 Linear layer 사용
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            # 출력 채널 수 설정 (single/multi mode 구분)
            out_channels = 1 if self.num_features == 1 else self.base_features
            
            for _ in range(out_channels):
                self.Linear_Seasonal.append(nn.Linear(self.input_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.input_len, self.pred_len))
                
        else:
            self.Linear_Seasonal = nn.Linear(self.input_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        
        # [Batch, seq_len, Channel] -> [Batch, Channel, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros(
                [batch_size, self.base_features if self.num_features > 1 else 1, self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros_like(seasonal_output)
            
            # 각 채널별로 예측 수행
            for i in range(seasonal_output.size(1)):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        # 최종 예측값 생성
        predictions = seasonal_output + trend_output
        
        # [Batch, Channel, pred_len] -> [Batch, pred_len, Channel]
        predictions = predictions.permute(0, 2, 1)
        
        return predictions