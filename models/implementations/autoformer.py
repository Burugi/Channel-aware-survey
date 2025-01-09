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
        self.channel_independence = self.config.get('channel_independence', False)

        # Decomposition
        self.decomp = series_decomp(moving_avg)

        # Channel independence에 따른 입출력 차원 조정
        if self.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = self.num_features
            self.dec_in = self.base_features
            self.c_out = self.base_features
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=self.enc_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )
        
        self.dec_embedding = DataEmbedding_wo_pos(
            c_in=self.dec_in,
            d_model=d_model,
            embed_type=embed,
            freq=freq,
            dropout=dropout
        )

        # Encoder
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

        # Decoder
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
                    self.c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True)
        )

    def _generate_time_features(self, x):
        """시간 특성 생성"""
        # Channel independence 모드에서는 expand된 batch size 사용
        batch_size = x.size(0)
        seq_len = x.size(1)
        time_features = torch.zeros((batch_size, seq_len, 4), device=x.device)
        return time_features

    def forward(self, x):
        """순전파"""
        original_batch_size = x.size(0)
        
        if self.channel_independence:
            # Reshape for channel independence
            x = x.permute(0, 2, 1).contiguous()  # (batch_size, num_features, seq_len)
            x = x.reshape(-1, x.size(-1), 1)     # (batch_size * num_features, seq_len, 1)
        
        # 입력 처리
        x_enc = x
        x_mark_enc = self._generate_time_features(x_enc)  # expand된 batch size 사용

        # Decomposition 초기화
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # 평균값으로 초기화된 trend
        mean = torch.mean(x_enc, dim=1).unsqueeze(1)
        trend_init = torch.cat([trend_init[:, -self.pred_len:, :], 
                            mean.repeat(1, self.pred_len, 1)], dim=1)

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # Decoder 입력 준비
        dec_zeros = torch.zeros_like(seasonal_init[:, -self.pred_len:, :])
        seasonal_init = torch.cat([seasonal_init[:, -self.pred_len:, :], dec_zeros], dim=1)
        
        # x_mark_dec 생성 - expand된 batch size 사용
        x_mark_dec = self._generate_time_features(
            torch.zeros(
                (x_enc.shape[0], seasonal_init.shape[1], x_enc.shape[2]), 
                device=x_enc.device
            )
        )
        
        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
        
        # 최종 출력
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :]

        if self.channel_independence:
            # Reshape back to original dimensions
            dec_out = dec_out.reshape(original_batch_size, self.base_features, self.pred_len, -1)
            dec_out = dec_out.squeeze(-1).permute(0, 2, 1)
                
        return dec_out

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