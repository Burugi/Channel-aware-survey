import torch
import torch.nn as nn
from models.base_model import BaseTimeSeriesModel
from layers.timemixer import PastDecomposableMixing
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class TimeMixerModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        # 기본 설정값 정의
        self.d_model = self.config.get('d_model', 512)
        self.d_ff = self.config.get('d_ff', 2048)
        self.n_layers = self.config.get('n_layers', 3)
        self.top_k = self.config.get('top_k', 5)
        self.down_sampling_window = self.config.get('down_sampling_window', 2)
        self.down_sampling_layers = self.config.get('down_sampling_layers', 2)
        self.embed_type = self.config.get('embed', 'timeF')
        self.freq = self.config.get('freq', 'h')
        self.dropout = self.config.get('dropout', 0.1)
        self.down_sampling_method = self.config.get('down_sampling_method', 'avg')
        
        # Determine input/output dimensions
        self.enc_in = 1 if self.num_features == 1 else self.base_features
        self.dec_out = 1 if self.num_features == 1 else self.base_features
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.d_model, 
            self.embed_type, self.freq, self.dropout)
        
        # Decomposition blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=self.input_len,
                pred_len=self.pred_len,
                d_model=self.d_model,
                d_ff=self.d_ff,
                down_sampling_window=self.down_sampling_window,
                down_sampling_layers=self.down_sampling_layers,
                top_k=self.top_k
            ) for _ in range(self.n_layers)
        ])
        
        # Normalization layers
        self.normalize_layers = nn.ModuleList([
            Normalize(self.enc_in, affine=True)
            for _ in range(self.down_sampling_layers + 1)
        ])
        
        # Prediction layers
        self.predict_layers = nn.ModuleList([
            nn.Linear(
                self.input_len // (self.down_sampling_window ** i),
                self.pred_len
            ) for i in range(self.down_sampling_layers + 1)
        ])
        
        # Output projection
        self.projection = nn.Linear(self.d_model, self.dec_out)
        
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

    def _generate_time_features(self, x):
        """시간 특성 생성"""
        batch_size, seq_len, _ = x.shape
        # 기본적인 시간 특성 (hour, dayofweek, month, dayofmonth)으로 구성
        time_features = torch.zeros((batch_size, seq_len, 4), device=x.device)
        return time_features

    def _process_inputs(self, x):
        """Multi-scale input processing"""
        # Select features for processing
        if self.num_features > 1:
            x = x[:, :, :self.base_features]
            
        if self.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.down_sampling_window)
        else:
            return [x], None
            
        x = x.permute(0, 2, 1)  # [B,C,T]
        x_list = [x.permute(0, 2, 1)]  # Original scale
        
        x_current = x
        for _ in range(self.down_sampling_layers):
            x_current = down_pool(x_current)
            x_list.append(x_current.permute(0, 2, 1))
            
        return x_list

    def _out_projection(self, dec_out, i, out_res):
        """출력 투영"""
        # Project temporal dimension first
        dec_out = self.predict_layers[i](dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Project feature dimension
        dec_out = self.projection(dec_out)
        
        # Process residual connection
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        
        return dec_out + out_res

    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        # Generate time features for embedding
        time_features = self._generate_time_features(x)
        
        # Multi-scale processing
        x_scales = self._process_inputs(x)
        
        # Normalize inputs
        x_list = []
        for i, x_scale in enumerate(x_scales):
            x_norm = self.normalize_layers[i](x_scale, 'norm')
            x_list.append(x_norm)
            
        # Get time features for each scale
        time_features_list = []
        for i in range(len(x_list)):
            if i == 0:
                time_features_list.append(time_features)
            else:
                # Downsample time features for each scale
                downsampled_features = time_features[:, ::self.down_sampling_window**i, :]
                time_features_list.append(downsampled_features)
        
        # Embedding
        enc_out_list = []
        for x_scale, t_scale in zip(x_list, time_features_list):
            enc_out = self.enc_embedding(x_scale, t_scale)
            enc_out_list.append(enc_out)
        
        # Past Decomposable Mixing
        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)
        
        # Prediction
        dec_out_list = []
        for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list)):
            dec_out = self._out_projection(enc_out, i, out_res)
            dec_out_list.append(dec_out)
        
        # Combine predictions
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        
        return dec_out