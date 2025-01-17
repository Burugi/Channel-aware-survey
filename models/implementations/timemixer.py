import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp

class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k=5):
        super(DFTSeriesDecomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class TimeMixerModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.d_model = self.config.get('d_model', 512)
        self.d_ff = self.config.get('d_ff', 2048)
        self.n_layers = self.config.get('e_layers', 3)
        self.down_sampling_layers = self.config.get('down_sampling_layers', 2)
        self.down_sampling_window = self.config.get('down_sampling_window', 2)
        self.moving_avg = self.config.get('moving_avg', 25)
        self.embed_type = self.config.get('embed', 'timeF')
        self.freq = self.config.get('freq', 'h')
        self.dropout = self.config.get('dropout', 0.1)

        # Channel independence mode 설정
        enc_in = 1 if self.channel_independence else self.base_features
        
        # Embedding layer
        self.embedding = DataEmbedding_wo_pos(
            enc_in,
            self.d_model,
            self.embed_type,
            self.freq,
            self.dropout
        )

        # Decomposition layer
        self.decomp = series_decomp(self.moving_avg)

        # Normalize layers
        self.norm_layers = nn.ModuleList([
            Normalize(enc_in, affine=True)
            for _ in range(self.down_sampling_layers + 1)
        ])

        # Cross attention layers
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Linear(self.d_ff, self.d_model)
            ) for _ in range(self.n_layers)
        ])

        # Prediction layers
        self.predict_layers = nn.ModuleList([
            nn.Linear(
                self.input_len // (self.down_sampling_window ** i),
                self.pred_len
            ) for i in range(self.down_sampling_layers + 1)
        ])

        # Trend related layers
        self.trend = nn.Linear(self.input_len, self.input_len)
        self.trend_dec = nn.Linear(self.input_len, self.pred_len)

        # Output projection
        self.projection = nn.Linear(self.d_model, 1 if self.channel_independence else self.base_features)

        # Regression and residual layers
        self.out_res_layers = nn.ModuleList([
            nn.Linear(
                self.input_len // (self.down_sampling_window ** i),
                self.input_len // (self.down_sampling_window ** i)
            ) for i in range(self.down_sampling_layers + 1)
        ])

        self.regression_layers = nn.ModuleList([
            nn.Linear(
                self.input_len // (self.down_sampling_window ** i),
                self.pred_len
            ) for i in range(self.down_sampling_layers + 1)
        ])

    def _generate_time_features(self, B, T):
        """시간 특성 생성 (empty features)"""
        # B: batch size, T: sequence length
        time_features = torch.zeros((B, T, 4), device=self.device)
        return time_features

    def _process_inputs(self, x, feature_idx=None):
        # Down sampling using avg pool
        down_pool = nn.AvgPool1d(self.down_sampling_window)
        
        x = x.permute(0, 2, 1)  # [B,C,T]
        x_list = [x.permute(0, 2, 1)]  # Original scale
        
        x_current = x
        for _ in range(self.down_sampling_layers):
            x_current = down_pool(x_current)
            x_list.append(x_current.permute(0, 2, 1))
            
        return x_list

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        batch_size = x.size(0)

        # Multi-scale processing
        x_list = self._process_inputs(x)
        
        # Normalize inputs
        norm_x_list = []
        for i, x_scale in enumerate(x_list):
            x_norm = self.norm_layers[i](x_scale, 'norm')
            norm_x_list.append(x_norm)
        
        # Generate time features for each scale
        time_features_list = []
        for i in range(len(norm_x_list)):
            time_features = self._generate_time_features(
                batch_size,
                norm_x_list[i].size(1)
            )
            time_features_list.append(time_features)
        
        # Embedding
        enc_out_list = []
        for x_scale, time_feat in zip(norm_x_list, time_features_list):
            enc_out = self.embedding(x_scale, time_feat)
            enc_out_list.append(enc_out)

        # Process through layers
        for i in range(self.n_layers):
            season_list = []
            trend_list = []
            
            for enc_out in enc_out_list:
                # Decomposition
                season, trend = self.decomp(enc_out)
                
                # Cross attention
                season = self.cross_layers[i](season)
                trend = self.cross_layers[i](trend)
                
                # Store results
                season_list.append(season)
                trend_list.append(trend)
            
            # Update encoded list with processed season and trend
            enc_out_list = []
            for season, trend in zip(season_list, trend_list):
                enc_out_list.append(season + trend)

        # Prediction phase
        dec_out_list = []
        for i, (enc_out, x_norm) in enumerate(zip(enc_out_list, norm_x_list)):
            # Project to prediction length
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Output projection
            dec_out = self.projection(dec_out)
            
            # Residual connection
            x_res = x_norm.permute(0, 2, 1)
            x_res = self.out_res_layers[i](x_res)
            x_res = self.regression_layers[i](x_res).permute(0, 2, 1)
            
            dec_out = dec_out + x_res
            dec_out_list.append(dec_out)

        # Combine predictions from different scales
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        
        # Denormalize
        dec_out = self.norm_layers[0](dec_out, 'denorm')
        
        return dec_out

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        batch_size = x.size(0)

        # Multi-scale processing
        x_list = self._process_inputs(x)
        
        # Normalize inputs
        norm_x_list = []
        for i, x_scale in enumerate(x_list):
            x_norm = self.norm_layers[i](x_scale, 'norm')
            norm_x_list.append(x_norm)
        
        # Generate time features for each scale
        time_features_list = []
        for i in range(len(norm_x_list)):
            time_features = self._generate_time_features(
                batch_size,
                norm_x_list[i].size(1)
            )
            time_features_list.append(time_features)
        
        # Embedding
        enc_out_list = []
        for x_scale, time_feat in zip(norm_x_list, time_features_list):
            enc_out = self.embedding(x_scale, time_feat)
            enc_out_list.append(enc_out)

        # Process through layers
        for i in range(self.n_layers):
            season_list = []
            trend_list = []
            
            for enc_out in enc_out_list:
                # Decomposition
                season, trend = self.decomp(enc_out)
                
                # Cross attention
                season = self.cross_layers[i](season)
                trend = self.cross_layers[i](trend)
                
                # Store results
                season_list.append(season)
                trend_list.append(trend)
            
            # Update encoded list with processed season and trend
            enc_out_list = []
            for season, trend in zip(season_list, trend_list):
                enc_out_list.append(season + trend)

        # Prediction phase
        dec_out_list = []
        for i, (enc_out, x_norm) in enumerate(zip(enc_out_list, norm_x_list)):
            # Project to prediction length
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Output projection
            dec_out = self.projection(dec_out)
            
            # Residual connection
            x_res = x_norm.permute(0, 2, 1)
            x_res = self.out_res_layers[i](x_res)
            x_res = self.regression_layers[i](x_res).permute(0, 2, 1)
            
            dec_out = dec_out + x_res
            dec_out_list.append(dec_out)

        # Combine predictions from different scales
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        
        # Denormalize
        dec_out = self.norm_layers[0](dec_out, 'denorm')
        
        return dec_out

    def forward(self, x):
        if self.channel_independence:
            outputs = []
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]
                out = self._forward_single_feature(feature_input, i)
                outputs.append(out)
            predictions = torch.cat(outputs, dim=-1)
        else:
            predictions = self._forward_all_features(x)

        return predictions[:, -self.pred_len:, :]