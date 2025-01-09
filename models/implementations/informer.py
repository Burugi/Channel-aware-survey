import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.attn import FullAttention, ProbAttention, AttentionLayer
from layers.embed_informer import DataEmbedding
from layers.encoder import Encoder, EncoderLayer, ConvLayer
from layers.decoder import Decoder, DecoderLayer

class InformerModel(BaseTimeSeriesModel):
    def _build_model(self):
        # 기본 설정값 정의
        d_model = self.config.get('d_model', 512)
        n_heads = self.config.get('n_heads', 8)
        e_layers = self.config.get('e_layers', 3)
        d_layers = self.config.get('d_layers', 2)
        d_ff = self.config.get('d_ff', 512)
        dropout = self.config.get('dropout', 0.0)
        attn = self.config.get('attn', 'prob')
        embed = self.config.get('embed', 'fixed')
        freq = self.config.get('freq', 'h')
        activation = self.config.get('activation', 'gelu')
        distil = self.config.get('distil', True)
        mix = self.config.get('mix', True)
        factor = self.config.get('factor', 5)
        output_attention = self.config.get('output_attention', False)

        # single/multi 모드에 따른 입출력 차원 조정
        self.enc_in = self.num_features
        self.dec_in = self.base_features if self.num_features > 1 else 1
        self.c_out = self.base_features if self.num_features > 1 else 1
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            self.enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in, d_model, embed, freq, dropout
        )
        
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers-1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.projection = nn.Linear(d_model, self.c_out, bias=True)

    def _generate_time_features(self, x):
        """시간 특성 생성"""
        batch_size, seq_len, _ = x.shape
        # 시간 특성으로 [month, day, weekday, hour, minute] 정보 사용
        time_features = torch.zeros((batch_size, seq_len, 5), device=x.device)
        return time_features

    def forward(self, x):
        # 입력 준비
        x_enc = x
        x_mark_enc = self._generate_time_features(x_enc)
        
        # Decoder 입력 준비 (zeros)
        x_dec = torch.zeros(
            (x_enc.shape[0], self.pred_len, self.dec_in),
            device=x_enc.device
        )
        x_mark_dec = self._generate_time_features(x_dec)
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)

        return dec_out