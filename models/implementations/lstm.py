import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel

class LSTMModel(BaseTimeSeriesModel):
    """LSTM 기반 시계열 예측 모델"""
    
    def _build_model(self):
        """LSTM 모델 아키텍처 구축"""
        self.hidden_size = self.config.get('hidden_size', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1 if self.num_features == 1 else self.base_features)
        )
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(x)
        
        # 마지막 hidden state 사용
        last_hidden = lstm_out[:, -1:, :]
        
        # Autoregressive 예측
        predictions = []
        current_input = last_hidden
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.fc(current_input.squeeze(1)).unsqueeze(1)
            predictions.append(output)
            
            # 다음 예측을 위한 입력 준비
            if self.num_features == 1:  # single mode
                current_input = torch.zeros(batch_size, 1, 1).to(self.device)
                current_input[:, :, 0:1] = output
            else:  # multi mode
                current_input = torch.zeros(batch_size, 1, self.num_features).to(self.device)
                current_input[:, :, :self.base_features] = output
            
            current_input, _ = self.lstm(current_input)
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)
        
        return predictions