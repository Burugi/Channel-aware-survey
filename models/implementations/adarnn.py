import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.loss_transfer import TransferLoss

class AdaRNNModel(BaseTimeSeriesModel):
    def _build_model(self):
        """AdaRNN 모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.use_bottleneck = self.config.get('use_bottleneck', False)
        self.bottleneck_width = self.config.get('bottleneck_width', 256)
        self.hidden_sizes = self.config.get('hidden_sizes', [64, 64])
        self.dropout = self.config.get('dropout', 0.1)
        self.model_type = self.config.get('model_type', 'AdaRNN')
        self.trans_loss = self.config.get('trans_loss', 'mmd')
        self.channel_independence = self.config.get('channel_independence', False)
        
        # 입력/출력 차원 설정
        self.seq_len = self.input_len
        in_size = 1 if self.channel_independence else self.num_features
        out_size = 1 if self.channel_independence else self.base_features
        
        # RNN 레이어 구성
        features = nn.ModuleList()
        current_in_size = in_size
        for hidden in self.hidden_sizes:
            rnn = nn.GRU(
                input_size=current_in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                dropout=self.dropout
            )
            features.append(rnn)
            current_in_size = hidden
        self.features = nn.Sequential(*features)

        # Bottleneck 레이어
        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.hidden_sizes[-1], self.bottleneck_width),
                nn.Linear(self.bottleneck_width, self.bottleneck_width),
                nn.BatchNorm1d(self.bottleneck_width),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            self.fc = nn.Linear(self.bottleneck_width, out_size)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(self.hidden_sizes[-1], out_size)

        # AdaRNN 특화 레이어
        if self.model_type == 'AdaRNN':
            gate = nn.ModuleList()
            for i in range(len(self.hidden_sizes)):
                gate_weight = nn.Linear(self.seq_len * self.hidden_sizes[i] * 2, self.seq_len)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for _ in range(len(self.hidden_sizes)):
                bnlst.append(nn.BatchNorm1d(self.seq_len))
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self._init_layers()
    
    def _init_layers(self):
        """게이트 레이어 초기화"""
        for i in range(len(self.hidden_sizes)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)
            
    def forward(self, x):
        """순전파"""
        original_batch_size = x.size(0)
        
        if self.channel_independence:
            # Reshape for channel independence
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
                
        # 예측값 생성
        out = self.gru_features(x, predict=True)
        fea = out[0]
        
        # Bottleneck 사용 여부에 따른 출력 생성
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            current_pred = self.fc(fea_bottleneck)
        else:
            current_pred = self.fc_out(fea[:, -1, :])
                
        # Autoregressive 예측
        predictions = [current_pred.unsqueeze(1)]
        
        if self.channel_independence:
            current_input = current_pred.unsqueeze(-1)  # (batch_size * num_features, 1, 1)
        else:
            current_input = x[:, -1:, :].clone()
        
        for _ in range(self.pred_len - 1):
            if not self.channel_independence:
                if self.num_features == 1:
                    current_input = predictions[-1]
                else:
                    current_input = torch.zeros(original_batch_size, 1, self.num_features, device=self.device)
                    current_input[:, :, :self.base_features] = predictions[-1]
            
            # 다음 시점 예측
            out = self.gru_features(current_input, predict=True)
            fea = out[0]
            
            if self.use_bottleneck:
                fea_bottleneck = self.bottleneck(fea[:, -1, :])
                next_pred = self.fc(fea_bottleneck)
            else:
                next_pred = self.fc_out(fea[:, -1, :])
            
            predictions.append(next_pred.unsqueeze(1))
        
        # 모든 예측을 결합
        predictions = torch.cat(predictions, dim=1)  # (batch_size * num_features, pred_len, 1) or (batch_size, pred_len, output_size)
        
        if self.channel_independence:
            # Reshape back to original dimensions
            expanded_size = original_batch_size * self.base_features * self.pred_len
            if predictions.numel() != expanded_size:
                predictions = predictions.squeeze(-1)  # Remove last dimension if it's 1
            predictions = predictions.reshape(original_batch_size, self.base_features, self.pred_len)
            predictions = predictions.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        
        return predictions

    def _process_gate_weight(self, out, index):
        """게이트 가중치 처리"""
        if self.channel_independence:
            # Channel independence mode에서는 원래 batch size를 사용
            batch_size = out.shape[0] // self.base_features
        else:
            batch_size = out.shape[0]
                
        x_s = out[0: batch_size//2]
        x_t = out[batch_size//2: batch_size]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        return self.softmax(weight).squeeze()

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

    def gru_features(self, x, predict=False):
        """GRU 특성 추출"""
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (self.model_type == 'AdaRNN' and not predict) else None
            
        for i in range(len(self.hidden_sizes)):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and not predict:
                out_gate = self._process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
                
        return out, out_lis, out_weight_list