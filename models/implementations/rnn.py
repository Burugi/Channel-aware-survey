import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel

class RNNModel(BaseTimeSeriesModel):
    """RNN 기반 시계열 예측 모델"""
    
    def _build_model(self):
        """RNN 모델 아키텍처 구축"""
        self.hidden_size = self.config.get('hidden_size', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.nonlinearity = self.config.get('nonlinearity', 'tanh')
        
        if self.channel_independence:
            # 각 feature별 독립적인 RNN과 출력 레이어
            self.feature_rnns = nn.ModuleList([
                nn.RNN(
                    input_size=1,  # 단일 feature
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0,
                    batch_first=True,
                    nonlinearity=self.nonlinearity
                )
                for _ in range(self.base_features)
            ])
            
            self.feature_fcs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size // 2, 1)  # 단일 feature 출력
                )
                for _ in range(self.base_features)
            ])
        else:
            # 기존 CD mode 구현
            self.rnn = nn.RNN(
                input_size=self.num_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True,
                nonlinearity=self.nonlinearity
            )
            
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, self.base_features)
            )

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.size(0)
        
        # RNN 통과
        rnn_out, _ = self.feature_rnns[feature_idx](x)
        
        # 마지막 hidden state 사용
        last_hidden = rnn_out[:, -1:, :]
        
        # Autoregressive 예측
        predictions = []
        current_input = last_hidden
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.feature_fcs[feature_idx](current_input.squeeze(1))
            predictions.append(output.unsqueeze(1))  # (batch_size, 1, 1)
            
            # 다음 예측을 위한 입력 준비
            current_input, _ = self.feature_rnns[feature_idx](output.unsqueeze(-2))
            current_input = current_input[:, -1:, :]
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, 1)
        return predictions

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.size(0)
        
        # RNN 통과
        rnn_out, _ = self.rnn(x)
        
        # 마지막 hidden state 사용
        last_hidden = rnn_out[:, -1:, :]
        
        # Autoregressive 예측
        predictions = []
        current_input = last_hidden
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.fc(current_input.squeeze(1))
            predictions.append(output.unsqueeze(1))
            
            # 다음 예측을 위한 입력 준비
            next_input = torch.zeros(batch_size, 1, self.num_features).to(self.device)
            next_input[:, :, :self.base_features] = output.unsqueeze(1)
            
            current_input, _ = self.rnn(next_input)
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, num_features)
        return predictions