import torch
import torch.nn as nn
from layers.Embed import Emb, BiMambaBlock
from mamba_ssm import Mamba

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Encoder(nn.Module):
    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 创新点集成：基于 ICLR 2025 SOR-Mamba
        # 1. Channel Mixing: 使用 BiMambaBlock 消除变量顺序偏差
        # 2. d_conv=1: 移除局部卷积，避免对无序变量进行局部平滑
        self.channel_mixer = BiMambaBlock(
            d_model=d_model, 
            d_state=16,      
            d_conv=1,        
            expand=2,        
        )

        # Time Mixing: 使用标准 Mamba 捕捉时间序列依赖
        # 这里处理的是时间/特征维度，保留 d_conv=4 以捕捉局部时序模式
        self.time_mixer = Mamba(
            d_model=enc_in,  
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, x):
        # x shape: [B, enc_in, d_model]
        # enc_in 是变量数，d_model 是特征维度
        
        # 1. Channel Mixing (在 enc_in 维度上扫描)
        y_0 = self.channel_mixer(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        
        # 2. Time/Feature Mixing
        # 转置以处理 d_model 维度 (如果 d_model 代表时间特征)
        y_1 = y_0.permute(0, 2, 1) # [B, d_model, enc_in]
        y_1 = self.time_mixer(y_1)
        y_1 = y_1.permute(0, 2, 1) # [B, enc_in, d_model]
        
        # 残差连接
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.decompsition = series_decomp(13)
        # Embedding
        self.emb = Emb(configs.seq_len, configs.d_model)
        
        self.seasonal_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        x = x_enc.permute(0, 2, 1) # [B, V, L]
        
        # Embedding 层现在包含 BiMamba 增强
        x = self.emb(x) # [B, V, d_model]

        seasonal_init, trend_init = self.decompsition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        
        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]