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
        self.channel_independence = self.config.get('channel_independence', False)
        
        # 입력/출력 임베딩
        in_features = 1 if self.channel_independence else self.num_features
        out_features = 1 if self.channel_independence else self.base_features
        
        self.src_embedding = nn.Linear(in_features, self.d_model)
        self.tgt_embedding = nn.Linear(out_features, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(self.d_model, out_features)
        
        # Target mask 생성
        self.register_buffer(
            'target_mask',
            self.transformer.generate_square_subsequent_mask(self.pred_len)
        )
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        if self.channel_independence:
            # Channel independence mode: reshape to (batch_size * num_features, seq_len, 1)
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
        
        # 입력 임베딩
        src = self.src_embedding(x)
        src = self.pos_encoder(src)
        
        # 디코더 입력 초기화 (zero tensor)
        tgt = torch.zeros(
            src.size(0),  # batch_size or batch_size * num_features
            self.pred_len,
            1 if self.channel_independence else self.base_features,
            device=self.device
        )
        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Transformer 통과
        output = self.transformer(src, tgt, tgt_mask=self.target_mask)
        predictions = self.output_layer(output)
        
        if self.channel_independence:
            # Reshape back to (batch_size, pred_len, num_features)
            predictions = predictions.reshape(batch_size, self.base_features, self.pred_len, -1)
            predictions = predictions.squeeze(-1).permute(0, 2, 1)
        
        return predictions