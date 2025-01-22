import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.loss_transfer import TransferLoss

class AdaRNNModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        self.use_bottleneck = self.config.get('use_bottleneck', False)
        self.bottleneck_width = self.config.get('bottleneck_width', 256)
        hidden_size_config = self.config.get('hidden_sizes', [64, 64])
        # List나 tuple을 리스트로 변환
        self.hidden_sizes = list(hidden_size_config) if isinstance(hidden_size_config, (list, tuple)) else [hidden_size_config, hidden_size_config]
        self.dropout = self.config.get('dropout', 0.1)
        self.model_type = self.config.get('model_type', 'AdaRNN')
        self.trans_loss = self.config.get('trans_loss', 'mmd')
        
        if self.channel_independence:
            # 각 feature별 독립적인 RNN layers
            self.feature_rnns = nn.ModuleList([
                nn.ModuleList([
                    nn.GRU(
                        input_size=1 if i == 0 else hidden,
                        hidden_size=hidden,
                        num_layers=1,
                        batch_first=True,
                        dropout=self.dropout
                    )
                    for i, hidden in enumerate(self.hidden_sizes)
                ])
                for _ in range(self.base_features)
            ])

            # 각 feature별 bottleneck과 출력 layers
            if self.use_bottleneck:
                self.feature_bottlenecks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_sizes[-1], self.bottleneck_width),
                        nn.Linear(self.bottleneck_width, self.bottleneck_width),
                        nn.BatchNorm1d(self.bottleneck_width),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                    )
                    for _ in range(self.base_features)
                ])
                self.feature_fcs = nn.ModuleList([
                    nn.Linear(self.bottleneck_width, 1)
                    for _ in range(self.base_features)
                ])
            else:
                self.feature_fcs = nn.ModuleList([
                    nn.Linear(self.hidden_sizes[-1], 1)
                    for _ in range(self.base_features)
                ])

            # AdaRNN 특화 layers (feature별)
            if self.model_type == 'AdaRNN':
                self.feature_gates = nn.ModuleList([
                    nn.ModuleList([
                        nn.Linear(self.input_len * self.hidden_sizes[i] * 2, self.input_len)
                        for i in range(len(self.hidden_sizes))
                    ])
                    for _ in range(self.base_features)
                ])
                
                self.feature_bn_lst = nn.ModuleList([
                    nn.ModuleList([
                        nn.BatchNorm1d(self.input_len)
                        for _ in range(len(self.hidden_sizes))
                    ])
                    for _ in range(self.base_features)
                ])
        else:
            # Channel dependence mode
            self.features = nn.ModuleList([
                nn.GRU(
                    input_size=self.num_features if i == 0 else self.hidden_sizes[i-1],
                    hidden_size=hidden,
                    num_layers=1,
                    batch_first=True,
                    dropout=self.dropout
                )
                for i, hidden in enumerate(self.hidden_sizes)
            ])

            if self.use_bottleneck:
                self.bottleneck = nn.Sequential(
                    nn.Linear(self.hidden_sizes[-1], self.bottleneck_width),
                    nn.Linear(self.bottleneck_width, self.bottleneck_width),
                    nn.BatchNorm1d(self.bottleneck_width),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
                self.fc = nn.Linear(self.bottleneck_width, self.base_features)
            else:
                self.fc_out = nn.Linear(self.hidden_sizes[-1], self.base_features)

            if self.model_type == 'AdaRNN':
                self.gate = nn.ModuleList([
                    nn.Linear(self.input_len * self.hidden_sizes[i] * 2, self.input_len)
                    for i in range(len(self.hidden_sizes))
                ])
                self.bn_lst = nn.ModuleList([
                    nn.BatchNorm1d(self.input_len)
                    for _ in range(len(self.hidden_sizes))
                ])

        self.softmax = torch.nn.Softmax(dim=0)
        self._init_weights()

    def _init_weights(self):
        """게이트 레이어 초기화"""
        if self.model_type == 'AdaRNN':
            if self.channel_independence:
                for feature_gate in self.feature_gates:
                    for gate in feature_gate:
                        gate.weight.data.normal_(0, 0.05)
                        gate.bias.data.fill_(0.0)
            else:
                for gate in self.gate:
                    gate.weight.data.normal_(0, 0.05)
                    gate.bias.data.fill_(0.0)

    def _process_gate_weight(self, out, index, feature_idx=None):
        """게이트 가중치 처리
        Args:
            out: RNN 출력 (batch_size, seq_len, hidden_size)
            index: RNN 레이어 인덱스
            feature_idx: feature 인덱스 (CI mode에서만 사용)
        """
        batch_size = out.shape[0] // 2  # source와 target 데이터가 연결되어 있음
        seq_len = out.shape[1]
        hidden_size = out.shape[2]
        
        # Source와 target 데이터 분리
        x_s = out[:batch_size]  # (batch_size/2, seq_len, hidden_size)
        x_t = out[batch_size:]  # (batch_size/2, seq_len, hidden_size)
        
        # 결합 및 reshape
        x_all = torch.cat([x_s, x_t], dim=2)  # (batch_size/2, seq_len, hidden_size*2)
        x_all = x_all.reshape(batch_size, -1)  # (batch_size/2, seq_len*hidden_size*2)

        if self.channel_independence:
            weight = torch.sigmoid(
                self.feature_bn_lst[feature_idx][index](
                    self.feature_gates[feature_idx][index](x_all)
                )
            )
        else:
            weight = torch.sigmoid(
                self.bn_lst[index](
                    self.gate[index](x_all)
                )
            )

        # 가중치 평균 및 정규화
        weight = weight.mean(dim=0)  # (seq_len,)
        return self.softmax(weight)  # (seq_len,)

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.size(0)
        
        # RNN layers 통과
        current_input = x
        hidden_states = []
        gate_weights = [] if self.model_type == 'AdaRNN' else None

        for i in range(len(self.hidden_sizes)):
            out, _ = self.feature_rnns[feature_idx][i](current_input)
            hidden_states.append(out)
            current_input = out
            
            if self.model_type == 'AdaRNN':
                gate_weight = self._process_gate_weight(out, i, feature_idx)
                gate_weights.append(gate_weight)

        # 마지막 hidden state 사용
        final_hidden = hidden_states[-1][:, -1]  # (batch_size, hidden_size)
        
        # 예측 생성
        predictions = []
        current_hidden = final_hidden

        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            if self.use_bottleneck:
                fea = self.feature_bottlenecks[feature_idx](current_hidden)
                pred = self.feature_fcs[feature_idx](fea)
            else:
                pred = self.feature_fcs[feature_idx](current_hidden)
            
            predictions.append(pred.unsqueeze(1))  # (batch_size, 1, 1)
            
            # 다음 예측을 위한 입력 준비
            pred_input = pred.unsqueeze(1)  # (batch_size, 1, 1)
            out, _ = self.feature_rnns[feature_idx][0](pred_input)
            current_hidden = out[:, -1]  # 마지막 hidden state

        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, 1)
        return predictions

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.size(0)
        
        # RNN layers 통과
        current_input = x
        hidden_states = []
        gate_weights = [] if self.model_type == 'AdaRNN' else None

        for i, rnn in enumerate(self.features):
            out, _ = rnn(current_input)
            hidden_states.append(out)
            current_input = out
            
            if self.model_type == 'AdaRNN':
                gate_weight = self._process_gate_weight(out, i)
                gate_weights.append(gate_weight)

        # 마지막 hidden state 사용
        final_hidden = hidden_states[-1][:, -1]  # (batch_size, hidden_size)
        
        # 예측 생성
        predictions = []
        current_hidden = final_hidden

        for _ in range(self.pred_len):
            # 현재 상태로 다음 시점 예측
            if self.use_bottleneck:
                fea = self.bottleneck(current_hidden)
                pred = self.fc(fea)
            else:
                pred = self.fc_out(current_hidden)
            
            predictions.append(pred.unsqueeze(1))  # (batch_size, 1, base_features)
            
            # 다음 예측을 위한 입력 준비
            next_input = torch.zeros(batch_size, 1, self.num_features).to(self.device)
            next_input[:, :, :self.base_features] = pred.unsqueeze(1)
            out, _ = self.features[0](next_input)
            current_hidden = out[:, -1]  # 마지막 hidden state

        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len, base_features)
        return predictions

    def forward(self, x):
        batch_size = x.size(0)

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