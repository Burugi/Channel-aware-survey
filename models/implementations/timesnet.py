import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.base_model import BaseTimeSeriesModel
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs['input_len']
        self.pred_len = configs['pred_len']
        self.k = configs.get('top_k', 5)
        
        self.conv = nn.Sequential(
            Inception_Block_V1(configs['d_model'], configs['d_ff'],
                             num_kernels=configs.get('num_kernels', 6)),
            nn.GELU(),
            Inception_Block_V1(configs['d_ff'], configs['d_model'],
                             num_kernels=configs.get('num_kernels', 6))
        )

    def forward(self, x):
        B, T, N = x.size()
        
        # FFT 기반 주기성 분석
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.k)
        period_list = x.shape[1] // top_list.detach().cpu().numpy()
        period_weight = abs(xf).mean(-1)[:, top_list]

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 패딩 처리
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                
            # 주기별 처리
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # 잔차 연결
        res = res + x
        return res

class TimesNetModel(BaseTimeSeriesModel):
    def _build_model(self):
        """모델 아키텍처 구축"""
        self.d_model = self.config.get('d_model', 512)
        self.d_ff = self.config.get('d_ff', 2048)
        self.e_layers = self.config.get('e_layers', 2)
        self.embed = self.config.get('embed', 'timeF')
        self.freq = self.config.get('freq', 'h')
        self.dropout = self.config.get('dropout', 0.1)
        self.top_k = self.config.get('top_k', 5)
        self.num_kernels = self.config.get('num_kernels', 6)
        
        if self.channel_independence:
            # Feature별 독립적인 모델 구성
            self.feature_embeddings = nn.ModuleList([
                DataEmbedding(
                    1, self.d_model, self.embed, self.freq, self.dropout
                ) for _ in range(self.base_features)
            ])
            
            # Feature별 Times blocks
            block_config = {
                'input_len': self.input_len,
                'pred_len': self.pred_len,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'top_k': self.top_k,
                'num_kernels': self.num_kernels
            }
            
            self.feature_layers = nn.ModuleList([
                nn.ModuleList([
                    TimesBlock(block_config)
                    for _ in range(self.e_layers)
                ]) for _ in range(self.base_features)
            ])
            
            # Feature별 layer normalization
            self.feature_layer_norms = nn.ModuleList([
                nn.LayerNorm(self.d_model)
                for _ in range(self.base_features)
            ])
            
            # Feature별 prediction layers
            self.feature_predict_linear = nn.ModuleList([
                nn.Linear(self.input_len, self.pred_len + self.input_len)
                for _ in range(self.base_features)
            ])
            
            # Feature별 projection layers
            self.feature_projections = nn.ModuleList([
                nn.Linear(self.d_model, 1, bias=True)
                for _ in range(self.base_features)
            ])
        else:
            # Channel dependence mode
            self.enc_embedding = DataEmbedding(
                self.num_features, self.d_model, self.embed, self.freq, self.dropout
            )
            
            block_config = {
                'input_len': self.input_len,
                'pred_len': self.pred_len,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'top_k': self.top_k,
                'num_kernels': self.num_kernels
            }
            
            self.layer = self.e_layers
            self.layers = nn.ModuleList([
                TimesBlock(block_config)
                for _ in range(self.e_layers)
            ])
            
            self.layer_norm = nn.LayerNorm(self.d_model)
            self.predict_linear = nn.Linear(self.input_len, self.pred_len + self.input_len)
            self.projection = nn.Linear(self.d_model, self.base_features, bias=True)

    def _forward_single_feature(self, x, feature_idx):
        """CI mode: 단일 feature 처리"""
        # 시간 특성 생성
        time_features = self._generate_time_features(x)
        
        # 정규화
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        
        # Embedding
        enc_out = self.feature_embeddings[feature_idx](x, time_features)
        
        # Predict linear
        enc_out = self.feature_predict_linear[feature_idx](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Times blocks
        for layer in self.feature_layers[feature_idx]:
            enc_out = self.feature_layer_norms[feature_idx](layer(enc_out))
        
        # Projection
        dec_out = self.feature_projections[feature_idx](enc_out)
        
        # 역정규화
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        
        return dec_out

    def _forward_all_features(self, x):
        """CD mode: 전체 feature 처리"""
        # 시간 특성 생성
        time_features = self._generate_time_features(x)
        
        # 정규화
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        
        # Embedding
        enc_out = self.enc_embedding(x, time_features)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Times layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.layers[i](enc_out))
        
        # Projection
        dec_out = self.projection(enc_out)
        
        # 역정규화
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        
        return dec_out

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
        
        return predictions[:, -self.pred_len:, :]
        
    def _generate_time_features(self, x, original_batch_size=None):
        """시간 특성 생성"""
        if self.channel_independence:
            time_features = torch.zeros((x.size(0), x.size(1), 4), device=x.device)
        else:
            batch_size = original_batch_size if original_batch_size else x.size(0)
            seq_len = x.size(1)
            time_features = torch.zeros((batch_size, seq_len, 4), device=x.device)
        return time_features