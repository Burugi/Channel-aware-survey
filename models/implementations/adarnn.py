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
        """게이트 가중치 처리"""
        batch_size = out.shape[0]
                
        x_s = out[0: batch_size//2]
        x_t = out[batch_size//2: batch_size]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)

        if self.channel_independence:
            weight = torch.sigmoid(
                self.feature_bn_lst[feature_idx][index](
                    self.feature_gates[feature_idx][index](x_all.float())
                )
            )
        else:
            weight = torch.sigmoid(
                self.bn_lst[index](
                    self.gate[index](x_all.float())
                )
            )

        weight = torch.mean(weight, dim=0)
        return self.softmax(weight).squeeze()

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if self.model_type == 'AdaRNN' else None

        # RNN layers 통과
        for i in range(len(self.hidden_sizes)):
            out, _ = self.feature_rnns[feature_idx][i](x_input.float())
            x_input = out
            out_lis.append(out)
            
            if self.model_type == 'AdaRNN':
                out_gate = self._process_gate_weight(x_input, i, feature_idx)
                out_weight_list.append(out_gate)

        # Bottleneck과 출력 레이어
        if self.use_bottleneck:
            fea_bottleneck = self.feature_bottlenecks[feature_idx](out[:, -1, :])
            predictions = self.feature_fcs[feature_idx](fea_bottleneck)
        else:
            predictions = self.feature_fcs[feature_idx](out[:, -1, :])

        predictions = predictions.unsqueeze(-1)  # Add feature dimension
        return predictions, out_lis, out_weight_list

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if self.model_type == 'AdaRNN' else None

        # RNN layers 통과
        for i, rnn in enumerate(self.features):
            out, _ = rnn(x_input.float())
            x_input = out
            out_lis.append(out)
            
            if self.model_type == 'AdaRNN':
                out_gate = self._process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)

        # Bottleneck과 출력 레이어
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(out[:, -1, :])
            predictions = self.fc(fea_bottleneck)
        else:
            predictions = self.fc_out(out[:, -1, :])

        return predictions, out_lis, out_weight_list

    def forward(self, x):
        batch_size = x.size(0)

        if self.channel_independence:
            outputs = []
            all_out_lis = []
            all_out_weight_list = []

            for i in range(self.base_features):
                feature_input = x[..., i:i+1]
                out, out_lis, out_weight_list = self._forward_single_feature(feature_input, i)
                outputs.append(out)
                all_out_lis.append(out_lis)
                if out_weight_list is not None:
                    all_out_weight_list.append(out_weight_list)

            predictions = torch.cat(outputs, dim=-1)
            return predictions

        else:
            predictions, out_lis, out_weight_list = self._forward_all_features(x)
            return predictions.unsqueeze(1)  # Add sequence dimension