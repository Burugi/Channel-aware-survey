import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(BaseTimeSeriesModel):
    """TCN 기반 시계열 예측 모델"""
    
    def _build_model(self):
        """TCN 모델 아키텍처 구축"""
        self.num_channels = self.config.get('num_channels', [32, 64, 128])
        self.kernel_size = self.config.get('kernel_size', 3)
        self.dropout = self.config.get('dropout', 0.2)
        self.channel_independence = self.config.get('channel_independence', False)
        
        # 입력/출력 차원 설정
        in_features = 1 if self.channel_independence else self.num_features
        out_features = 1 if self.channel_independence else self.base_features
        
        layers = []
        num_levels = len(self.num_channels)
        
        # TCN layers 구축
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_features if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, self.kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(self.kernel_size-1) * dilation_size,
                    dropout=self.dropout
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(self.num_channels[-1], self.num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_channels[-1] // 2, out_features)
        )
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        if self.channel_independence:
            # Channel independence mode: reshape to (batch_size * num_features, seq_len, 1)
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
            # TCN은 (batch, channel, seq_len) 형태의 입력을 기대하므로 변환
            x = x.transpose(1, 2)  # (batch_size * num_features, 1, seq_len)
        else:
            # Channel dependence mode
            x = x.transpose(1, 2)  # (batch_size, num_features, seq_len)
        
        # TCN 적용
        x = self.tcn(x)
        
        # 마지막 시점의 특성만 사용
        x = x[:, :, -1]
        
        # Autoregressive 예측
        predictions = []
        current_input = x.unsqueeze(-1)
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 값 예측
            output = self.fc(current_input.squeeze(-1))
            predictions.append(output.unsqueeze(1))
            
            # 다음 예측을 위한 입력 준비
            if self.channel_independence:
                next_input = output.unsqueeze(-1)  # (batch_size * num_features, 1, 1)
            else:
                next_input = torch.zeros(batch_size, self.num_features, 1).to(self.device)
                next_input[:, :self.base_features] = output.unsqueeze(-1)
            
            current_input = self.tcn(next_input)[:, :, -1:]
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size * num_features, pred_len, 1) or (batch_size, pred_len, output_size)
        
        if self.channel_independence:
            # Reshape back to (batch_size, pred_len, num_features)
            predictions = predictions.view(batch_size, self.base_features, self.pred_len, -1)
            predictions = predictions.squeeze(-1).permute(0, 2, 1)
        
        return predictions