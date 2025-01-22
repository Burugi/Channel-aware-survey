import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class AutoformerModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        d_model = self.config.get('d_model', 512)
        n_heads = self.config.get('n_heads', 8)
        e_layers = self.config.get('e_layers', 2)
        d_layers = self.config.get('d_layers', 1)
        d_ff = self.config.get('d_ff', 2048)
        moving_avg = self.config.get('moving_avg', 25)
        factor = self.config.get('factor', 1)
        dropout = self.config.get('dropout', 0.1)
        embed = self.config.get('embed', 'fixed')
        freq = self.config.get('freq', 'h')
        activation = self.config.get('activation', 'gelu')
        output_attention = self.config.get('output_attention', False)

        if self.channel_independence:
            # 각 feature별 독립적인 모델 구성
            self.feature_decomps = nn.ModuleList([
                series_decomp(moving_avg) for _ in range(self.base_features)
            ])

            # Feature별 임베딩
            self.feature_enc_embeddings = nn.ModuleList([
                DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
                for _ in range(self.base_features)
            ])
            self.feature_dec_embeddings = nn.ModuleList([
                DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
                for _ in range(self.base_features)
            ])

            # Feature별 인코더
            encoder_layers = lambda: [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ]
            
            self.feature_encoders = nn.ModuleList([
                Encoder(
                    encoder_layers(),
                    norm_layer=my_Layernorm(d_model)
                ) for _ in range(self.base_features)
            ])

            # Feature별 디코더
            decoder_layers = lambda: [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    1,  # c_out for single feature
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(d_layers)
            ]
            
            self.feature_decoders = nn.ModuleList([
                Decoder(
                    decoder_layers(),
                    norm_layer=my_Layernorm(d_model),
                    projection=nn.Linear(d_model, 1, bias=True)
                ) for _ in range(self.base_features)
            ])

        else:
            # Channel dependence mode
            self.decomp = series_decomp(moving_avg)
            
            self.enc_embedding = DataEmbedding_wo_pos(
                self.num_features, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding_wo_pos(
                self.base_features, d_model, embed, freq, dropout)

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor, attention_dropout=dropout,
                                          output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
                norm_layer=my_Layernorm(d_model)
            )

            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, factor, attention_dropout=dropout,
                                          output_attention=False),
                            d_model, n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor, attention_dropout=dropout,
                                          output_attention=False),
                            d_model, n_heads),
                        d_model,
                        self.base_features,
                        d_ff,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                    ) for _ in range(d_layers)
                ],
                norm_layer=my_Layernorm(d_model),
                projection=nn.Linear(d_model, self.base_features, bias=True)
            )

    def _generate_time_features(self, x):
        """시간 특성 생성"""
        batch_size = x.size(0)
        seq_len = x.size(1)
        time_features = torch.zeros((batch_size, seq_len, 4), device=x.device)
        return time_features

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.shape[0]
        
        # 시간 특성 생성
        enc_time_mark = self._generate_time_features(x)
        
        # Decomposition
        seasonal_init, trend_init = self.feature_decomps[feature_idx](x)
        
        # Encoder
        enc_out = self.feature_enc_embeddings[feature_idx](x, enc_time_mark)
        enc_out, _ = self.feature_encoders[feature_idx](enc_out)
        
        # Decoder input initialization
        dec_input = torch.zeros((batch_size, self.pred_len, 1), device=x.device)
        dec_time_mark = self._generate_time_features(dec_input)
        
        # Trend initialization
        trend_init = torch.mean(x, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        
        # Decoder
        dec_out = self.feature_dec_embeddings[feature_idx](dec_input, dec_time_mark)
        seasonal_part, trend_part = self.feature_decoders[feature_idx](dec_out, enc_out, trend=trend_init)
        
        return seasonal_part + trend_part

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.shape[0]
        
        # 시간 특성 생성
        enc_time_mark = self._generate_time_features(x)
        
        # Decomposition
        seasonal_init, trend_init = self.decomp(x)
        
        # Encoder
        enc_out = self.enc_embedding(x, enc_time_mark)
        enc_out, _ = self.encoder(enc_out)
        
        # Decoder input initialization
        dec_input = torch.zeros((batch_size, self.pred_len, self.base_features), device=x.device)
        dec_time_mark = self._generate_time_features(dec_input)
        
        # Trend initialization
        trend_init = torch.mean(x, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        
        # Decoder
        dec_out = self.dec_embedding(dec_input, dec_time_mark)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
        
        return seasonal_part + trend_part

    def forward(self, x):
        """순전파"""
        if self.channel_independence:
            outputs = []
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]
                feature_output = self._forward_single_feature(feature_input, i)
                outputs.append(feature_output)
            predictions = torch.cat(outputs, dim=-1)
        else:
            predictions = self._forward_all_features(x)
        
        return predictions