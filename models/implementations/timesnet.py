import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.base_model import BaseTimeSeriesModel
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        # Dictionary에서 값을 가져오도록 수정
        self.seq_len = configs['input_len']  # seq_len을 input_len으로 변경
        self.pred_len = configs['pred_len']
        self.k = configs.get('top_k', 5)
        
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs['d_model'], configs['d_ff'],
                             num_kernels=configs.get('num_kernels', 6)),
            nn.GELU(),
            Inception_Block_V1(configs['d_ff'], configs['d_model'],
                             num_kernels=configs.get('num_kernels', 6))
        )


    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class TimesNetModel(BaseTimeSeriesModel):
    def _generate_time_features(self, x, original_batch_size=None):
        """시간 특성 생성"""
        if self.channel_independence:
            time_features = torch.zeros((x.size(0), x.size(1), 4), device=x.device)
        else:
            batch_size = original_batch_size if original_batch_size else x.size(0)
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
            
            # Generate time features with expanded batch size
            time_features = self._generate_time_features(x)
        else:
            # Generate time features with original batch size
            time_features = self._generate_time_features(x, original_batch_size)
        
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        
        # Embedding
        enc_out = self.enc_embedding(x, time_features)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TimesNet layers
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # Project back
        dec_out = self.projection(enc_out)
        
        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.input_len, 1))
        
        if self.channel_independence:
            # Reshape back to original dimensions
            dec_out = dec_out.reshape(original_batch_size, self.base_features, -1)
            dec_out = dec_out.permute(0, 2, 1)
        
        return dec_out[:, -self.pred_len:, :]

    def _build_model(self):
        """TimesNet 모델 아키텍처 구축"""
        self.channel_independence = self.config.get('channel_independence', False)
        self.task_name = 'long_term_forecast'
        
        # Channel independence에 따른 입출력 차원 설정
        self.enc_in = 1 if self.channel_independence else self.num_features
        self.dec_out = 1 if self.channel_independence else self.base_features
        
        # Base configurations
        self.config.update({
            'input_len': self.input_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'dec_out': self.dec_out
        })
        
        self.d_model = self.config.get('d_model', 512)
        self.d_ff = self.config.get('d_ff', 2048)
        self.e_layers = self.config.get('e_layers', 2)
        self.embed = self.config.get('embed', 'timeF')
        self.freq = self.config.get('freq', 'h')
        self.dropout = self.config.get('dropout', 0.1)
        
        # Build model components with correct input dimensions
        if self.channel_independence:
            self.enc_embedding = DataEmbedding(
                1, self.d_model, self.embed, self.freq, self.dropout)
        else:
            self.enc_embedding = DataEmbedding(
                self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
            
        self.model = nn.ModuleList([
            TimesBlock(self.config) for _ in range(self.e_layers)
        ])
        
        self.layer = self.e_layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(self.input_len, self.pred_len + self.input_len)
        self.projection = nn.Linear(self.d_model, self.dec_out, bias=True)
        
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