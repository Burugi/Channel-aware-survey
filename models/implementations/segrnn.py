import torch
import torch.nn as nn
import math
from models.base_model import BaseTimeSeriesModel

class SegRNNModel(BaseTimeSeriesModel):
    """
    SegRNN: Segmented RNN for Long Sequence Modeling
    Paper: https://arxiv.org/abs/2308.11200.pdf
    """
    
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.d_model = self.config.get('d_model', 512)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Segment lengths should divide input/pred lengths
        self.seg_len = self._get_valid_seg_len()
        
        # Calculate segment numbers
        self.seg_num_x = self.input_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        
        # Determine input/output dimensions
        self.enc_in = 1 if self.num_features == 1 else self.base_features
        self.dec_out = 1 if self.num_features == 1 else self.base_features
        
        # Value embedding - transforms segment to d_model dimension
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        
        # GRU encoder
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Position and channel embeddings
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        
        # Output projection
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        
    def _get_valid_seg_len(self):
        """Get valid segment length that divides both input and prediction length"""
        seg_len = self.config.get('seg_len', 12)
        
        # Find the largest factor of both input_len and pred_len that's <= seg_len
        def get_factors(n):
            factors = []
            for i in range(1, n + 1):
                if n % i == 0:
                    factors.append(i)
            return factors
        
        input_factors = set(get_factors(self.input_len))
        pred_factors = set(get_factors(self.pred_len))
        common_factors = sorted(list(input_factors & pred_factors))
        
        # Find largest factor <= seg_len
        valid_seg_len = seg_len
        for factor in reversed(common_factors):
            if factor <= seg_len:
                valid_seg_len = factor
                break
                
        return valid_seg_len

    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        # Select features for processing if in multi mode
        if self.num_features > 1:
            x = x[:, :, :self.base_features]
        
        # Store last value for denormalization
        seq_last = x[:, -1:, :].detach()
        
        # Normalize and permute [batch, seq, channel] -> [batch, channel, seq]
        x = (x - seq_last).permute(0, 2, 1)
        
        # Reshape to segments
        # [batch, channel, seq] -> [batch*channel, seg_num, seg_len]
        x_reshaped = x.reshape(batch_size * self.enc_in, self.seg_num_x, self.seg_len)
        
        # Embed segments
        x = self.valueEmbedding(x_reshaped)  # [batch*channel, seg_num, d_model]
        
        # Encode with GRU
        _, hn = self.rnn(x)  # [1, batch*channel, d_model]
        
        # Prepare positional embeddings for decoder
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # [channel, seg_num_y, d_model//2]
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # [channel, seg_num_y, d_model//2]
        ], dim=-1)  # [channel, seg_num_y, d_model]
        
        # Reshape for batch processing
        pos_emb = pos_emb.view(-1, 1, self.d_model)  # [channel*seg_num_y, 1, d_model]
        pos_emb = pos_emb.repeat(batch_size, 1, 1)  # [batch*channel*seg_num_y, 1, d_model]
        
        # Decode with GRU
        hn_repeated = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        _, hy = self.rnn(pos_emb, hn_repeated)
        
        # Project to output segments
        y = self.predict(hy.transpose(0, 1))  # [batch*channel*seg_num_y, seg_len]
        
        # Reshape to final output
        y = y.reshape(batch_size, self.enc_in, -1)  # [batch, channel, pred_len]
        
        # Permute and denormalize
        y = y.permute(0, 2, 1) + seq_last  # [batch, pred_len, channel]
        
        return y