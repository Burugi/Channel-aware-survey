import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseTimeSeriesModel

class SegRNNModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.d_model = self.config.get('d_model', 512)
        self.dropout = self.config.get('dropout', 0.1)
        self.seg_len = self.config.get('seg_len', 12)  # 세그먼트 길이
        
        # 세그먼트 수 계산
        self.seg_num_x = self.input_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        
        if self.channel_independence:
            # 각 feature별 독립적인 모델 구성
            self.feature_value_embeddings = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seg_len, self.d_model),
                    nn.ReLU()
                )
                for _ in range(self.base_features)
            ])
            
            self.feature_rnns = nn.ModuleList([
                nn.GRU(
                    input_size=self.d_model,
                    hidden_size=self.d_model,
                    num_layers=1,
                    bias=True,
                    batch_first=True,
                    bidirectional=False
                )
                for _ in range(self.base_features)
            ])
            
            self.feature_predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, self.seg_len)
                )
                for _ in range(self.base_features)
            ])
            
            # Feature별 positional embedding (공유 안함)
            self.feature_pos_embs = nn.ParameterList([
                nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
                for _ in range(self.base_features)
            ])
            
            self.feature_channel_embs = nn.ParameterList([
                nn.Parameter(torch.randn(1, self.d_model // 2))
                for _ in range(self.base_features)
            ])
            
        else:
            # Channel dependence mode
            self.value_embedding = nn.Sequential(
                nn.Linear(self.seg_len, self.d_model),
                nn.ReLU()
            )
            
            self.rnn = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=1,
                bias=True,
                batch_first=True,
                bidirectional=False
            )
            
            self.predictor = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
            
            # 공유되는 positional embedding
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.base_features, self.d_model // 2))

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.size(0)
        
        # 마지막 값 저장 (denormalization을 위해)
        seq_last = x[:, -1:, :].detach()
        
        # Permute
        x = (x - seq_last).permute(0, 2, 1)
        
        # Segment and embedding
        x = self.feature_value_embeddings[feature_idx](
            x.reshape(-1, self.seg_num_x, self.seg_len)
        )
        
        # RNN encoding
        _, hn = self.feature_rnns[feature_idx](x)
        
        # Position embedding 생성 및 결합
        pos_emb = torch.cat([
            self.feature_pos_embs[feature_idx].unsqueeze(0),
            self.feature_channel_embs[feature_idx].unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        
        # Decoding
        _, hy = self.feature_rnns[feature_idx](
            pos_emb,
            hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        )
        
        # Prediction
        y = self.feature_predictors[feature_idx](hy).view(-1, 1, self.pred_len)
        
        # Denormalization
        y = y.permute(0, 2, 1) + seq_last
        
        return y

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.size(0)
        
        # 마지막 값 저장 (denormalization을 위해)
        seq_last = x[:, -1:, :].detach()
        
        # Permute
        x = (x - seq_last).permute(0, 2, 1)
        
        # Segment and embedding
        x = self.value_embedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        
        # RNN encoding
        _, hn = self.rnn(x)
        
        # Position embedding 생성 및 결합
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.base_features, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        
        # Decoding
        _, hy = self.rnn(
            pos_emb,
            hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        )
        
        # Prediction
        y = self.predictor(hy).view(-1, self.base_features, self.pred_len)
        
        # Denormalization
        y = y.permute(0, 2, 1) + seq_last
        
        return y

    def forward(self, x):
        """순전파"""
        if self.channel_independence:
            outputs = []
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]
                out = self._forward_single_feature(feature_input, i)
                outputs.append(out)
            predictions = torch.cat(outputs, dim=-1)
        else:
            predictions = self._forward_all_features(x)
        
        return predictions