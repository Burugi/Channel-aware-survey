import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

class DFTSeriesDecomp(nn.Module):
    """Series decomposition block using DFT"""
    def __init__(self, top_k=5):
        super().__init__()
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

class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing season pattern"""
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** i),
                         seq_len // (down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                         seq_len // (down_sampling_window ** (i + 1)))
            ) for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing trend pattern"""
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                         seq_len // (down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** i),
                         seq_len // (down_sampling_window ** i))
            ) for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    """Past decomposable mixing module"""
    def __init__(self, seq_len, pred_len, d_model, d_ff, 
                 down_sampling_window, down_sampling_layers, top_k=5):
        super().__init__()
        
        self.decomposition = DFTSeriesDecomp(top_k)
        
        # Cross attention layer
        self.cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Season and trend mixing
        self.mixing_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers)
            
        self.mixing_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers)

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        
        # Decomposition
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decomposition(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        
        # Mixing
        out_season_list = self.mixing_season(season_list)
        out_trend_list = self.mixing_trend(trend_list)
        
        # Combine
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        
        return out_list