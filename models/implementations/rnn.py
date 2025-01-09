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
        self.channel_independence = self.config.get('channel_independence', False)
        
        # 입력/출력 차원 설정
        in_features = 1 if self.channel_independence else self.num_features
        out_features = 1 if self.channel_independence else self.base_features
        
        # RNN 레이어
        self.rnn = nn.RNN(
            input_size=in_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            nonlinearity=self.nonlinearity
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, out_features)
        )
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        if self.channel_independence:
            # Channel independence mode: reshape to (batch_size * num_features, seq_len, 1)
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
        
        # RNN 통과
        rnn_out, _ = self.rnn(x)
        
        # 마지막 hidden state 사용
        last_hidden = rnn_out[:, -1:, :]
        
        # Autoregressive 예측
        predictions = []
        current_input = last_hidden
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.fc(current_input.squeeze(1)).unsqueeze(1)
            predictions.append(output)
            
            # 다음 예측을 위한 입력 준비
            if self.channel_independence:
                next_input = output  # (batch_size * num_features, 1, 1)
            else:
                next_input = torch.zeros(batch_size, 1, self.num_features).to(self.device)
                next_input[:, :, :self.base_features] = output
            
            current_input, _ = self.rnn(next_input)
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size * num_features, pred_len, 1) or (batch_size, pred_len, output_size)
        
        if self.channel_independence:
            # Reshape back to (batch_size, pred_len, num_features)
            predictions = predictions.reshape(batch_size, self.base_features, self.pred_len, -1)
            predictions = predictions.squeeze(-1).permute(0, 2, 1)
        
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