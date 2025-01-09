import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.decomposition import SeriesDecomposition

## Dlienar의 individual과 channel independence는 비슷하지만 약간 다른 개념이다.

class DLinearModel(BaseTimeSeriesModel):
    """Decomposition-Linear Model for Time Series Forecasting"""
    
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.kernel_size = self.config.get('kernel_size', 25)
        self.individual = self.config.get('individual', True)
        self.channel_independence = self.config.get('channel_independence', False)
        
        # Decomposition
        self.decomposition = SeriesDecomposition(self.kernel_size)
        
        # Channel independence나 individual mode에서는 각 채널별로 별도의 Linear layer 사용
        if self.channel_independence or self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            # 출력 채널 수 설정
            out_channels = self.base_features if not self.channel_independence else 1
            
            for _ in range(out_channels):
                self.Linear_Seasonal.append(nn.Linear(self.input_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.input_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.input_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        if self.channel_independence:
            # Reshape for channel independence
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
        
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)
        
        # [Batch, seq_len, Channel] -> [Batch, Channel, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        if self.channel_independence:
            # Channel independence mode
            seasonal_output = self.Linear_Seasonal[0](seasonal_init.squeeze(-2))
            trend_output = self.Linear_Trend[0](trend_init.squeeze(-2))
            
            seasonal_output = seasonal_output.unsqueeze(1)  # Add channel dimension back
            trend_output = trend_output.unsqueeze(1)
            
        elif self.individual:
            # Individual mode (각 채널별 독립적인 linear layer)
            seasonal_output = torch.zeros(
                [batch_size, self.base_features, self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros_like(seasonal_output)
            
            for i in range(self.base_features):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Channel dependence mode
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        # 최종 예측값 생성
        predictions = seasonal_output + trend_output
        
        if self.channel_independence:
            # Reshape back to original dimensions
            predictions = predictions.reshape(batch_size, self.base_features, self.pred_len)
        
        # [Batch, Channel, pred_len] -> [Batch, pred_len, Channel]
        predictions = predictions.permute(0, 2, 1)
        
        return predictions

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