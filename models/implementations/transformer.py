import torch
import torch.nn as nn
import math
from models.base_model import BaseTimeSeriesModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(BaseTimeSeriesModel):
    """Transformer 기반 시계열 예측 모델"""
    
    def _build_model(self):
        """Transformer 모델 아키텍처 구축"""
        self.d_model = self.config.get('d_model', 128)
        self.nhead = self.config.get('nhead', 8)
        self.num_encoder_layers = self.config.get('num_encoder_layers', 4)
        self.num_decoder_layers = self.config.get('num_decoder_layers', 4)
        self.dim_feedforward = self.config.get('dim_feedforward', 512)
        self.dropout = self.config.get('dropout', 0.1)
        
        if self.channel_independence:
            # 각 feature별 독립적인 transformer 구성
            self.feature_transformers = nn.ModuleList([
                nn.Transformer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_encoder_layers=self.num_encoder_layers,
                    num_decoder_layers=self.num_decoder_layers,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    batch_first=True
                ) for _ in range(self.base_features)
            ])
            
            # 각 feature별 임베딩 레이어
            self.feature_src_embeddings = nn.ModuleList([
                nn.Linear(1, self.d_model)
                for _ in range(self.base_features)
            ])
            
            self.feature_tgt_embeddings = nn.ModuleList([
                nn.Linear(1, self.d_model)
                for _ in range(self.base_features)
            ])
            
            # 각 feature별 출력 레이어
            self.feature_output_layers = nn.ModuleList([
                nn.Linear(self.d_model, 1)
                for _ in range(self.base_features)
            ])
        else:
            # 단일 transformer로 모든 feature 처리
            self.transformer = nn.Transformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True
            )
            
            self.src_embedding = nn.Linear(self.num_features, self.d_model)
            self.tgt_embedding = nn.Linear(self.base_features, self.d_model)
            self.output_layer = nn.Linear(self.d_model, self.base_features)
        
        # Positional encoding은 공유
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Target mask 생성
        self.register_buffer(
            'target_mask',
            self.transformer.generate_square_subsequent_mask(self.pred_len) if not self.channel_independence
            else None
        )
        
    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        # 입력 임베딩
        src = self.feature_src_embeddings[feature_idx](x)
        src = self.pos_encoder(src)
        
        # 디코더 입력 초기화 (zero tensor)
        tgt = torch.zeros(x.size(0), self.pred_len, 1, device=self.device)
        tgt = self.feature_tgt_embeddings[feature_idx](tgt)
        tgt = self.pos_encoder(tgt)
        
        # Target mask 생성 (각 feature별로)
        tgt_mask = self.feature_transformers[feature_idx].generate_square_subsequent_mask(
            self.pred_len
        ).to(self.device)
        
        # Transformer 통과
        output = self.feature_transformers[feature_idx](
            src, tgt,
            tgt_mask=tgt_mask
        )
        
        # 출력 생성
        predictions = self.feature_output_layers[feature_idx](output)
        
        return predictions
    
    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        # 입력 임베딩
        src = self.src_embedding(x)
        src = self.pos_encoder(src)
        
        # 디코더 입력 초기화 (zero tensor)
        tgt = torch.zeros(
            x.size(0),
            self.pred_len,
            self.base_features,
            device=self.device
        )
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Transformer 통과
        output = self.transformer(src, tgt, tgt_mask=self.target_mask)
        predictions = self.output_layer(output)
        
        return predictions