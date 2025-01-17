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
        
        if self.channel_independence:
            # 각 feature별 독립적인 TCN layers와 출력 레이어
            self.feature_tcns = nn.ModuleList([
                self._build_tcn_layers(input_size=1)  # 단일 feature input
                for _ in range(self.base_features)
            ])
            
            self.feature_fcs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.num_channels[-1], self.num_channels[-1] // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.num_channels[-1] // 2, 1)  # 단일 feature 출력
                )
                for _ in range(self.base_features)
            ])
        else:
            # Channel dependence mode
            self.tcn = self._build_tcn_layers(input_size=self.num_features)
            self.fc = nn.Sequential(
                nn.Linear(self.num_channels[-1], self.num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.num_channels[-1] // 2, self.base_features)
            )
    
    def _build_tcn_layers(self, input_size):
        """TCN layers 구축"""
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(self.kernel_size-1) * dilation_size,
                    dropout=self.dropout
                )
            )
        
        return nn.Sequential(*layers)

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.size(0)
        
        # [Batch, seq_len, 1] -> [Batch, 1, seq_len]
        x = x.permute(0, 2, 1)
        
        # TCN 통과
        x = self.feature_tcns[feature_idx](x)
        
        # 마지막 시점의 특성 추출
        x = x[:, :, -1]
        
        # Autoregressive 예측
        predictions = []
        current_input = x.unsqueeze(-1)
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.feature_fcs[feature_idx](current_input.squeeze(-1))
            predictions.append(output.unsqueeze(1))  # (batch_size, 1, 1)
            
            # 다음 예측을 위한 입력 준비
            next_input = output.unsqueeze(-2)  # (batch_size, 1, 1)
            current_input = self.feature_tcns[feature_idx](next_input)[:, :, -1:]
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, 1)
        return predictions

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.size(0)
        
        # [Batch, seq_len, num_features] -> [Batch, num_features, seq_len]
        x = x.transpose(1, 2)
        
        # TCN 통과
        x = self.tcn(x)
        
        # 마지막 시점의 특성 추출
        x = x[:, :, -1]
        
        # Autoregressive 예측
        predictions = []
        current_input = x.unsqueeze(-1)
        
        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            output = self.fc(current_input.squeeze(-1))
            predictions.append(output.unsqueeze(1))
            
            # 다음 예측을 위한 입력 준비
            next_input = torch.zeros(batch_size, self.num_features, 1).to(self.device)
            next_input[:, :self.base_features] = output.unsqueeze(-1)
            
            current_input = self.tcn(next_input)[:, :, -1:]
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, num_features)
        return predictions