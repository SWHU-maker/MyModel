import torch
import torch.nn as nn
from layers.Embed import Emb
from mamba_ssm import Mamba

# ==========================================
# 核心组件：BiMambaBlock (IClR 2025 SOR-Mamba)
# ==========================================
class BiMambaBlock(nn.Module):
    """
    双向 Mamba 模块
    通过正向和反向扫描，消除序列(特别是变量Channel)的顺序偏差
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Forward
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # Backward
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: [B, L, D]
        y_f = self.mamba_f(x)
        
        # Backward pass
        x_b = torch.flip(x, dims=[1])
        y_b = self.mamba_b(x_b)
        y_b = torch.flip(y_b, dims=[1])
        
        return y_f + y_b


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
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

        # 1. Channel Mixing (变量间交互) -> 使用 BiMamba + d_conv=1
        # 消除变量顺序偏差，移除局部卷积偏置
        self.channel_mixer = BiMambaBlock(
            d_model=d_model, 
            d_state=16,      
            d_conv=1,        
            expand=2,        
        )

        # 2. Time Mixing (时间步交互) -> 使用标准 Mamba
        # 保持 d_conv=4 以捕捉局部时序模式
        self.time_mixer = Mamba(
            d_model=enc_in,  
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, x):
        # x: [B, enc_in, d_model]
        
        # Channel Mixing (Bi-Directional)
        y_0 = self.channel_mixer(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        
        # Time Mixing
        y_1 = y_0.permute(0, 2, 1) # [B, d_model, enc_in]
        y_1 = self.time_mixer(y_1)
        y_1 = y_1.permute(0, 2, 1) # [B, enc_in, d_model]
        
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
        
        # Embedding 层
        self.emb = Emb(configs.seq_len, configs.d_model)
        
        # [关键修复]：在 Embedding 之后、Encoder 之前增加一层全局变量交互
        # 这对应之前 Embed.py 内部被移除的 variable_mixer
        # 它可以作为 "Pre-Encoder" 层，进一步提取 Patch 后的多变量关联特征
        self.pre_encoder_mixer = BiMambaBlock(
            d_model=configs.d_model,
            d_state=16,
            d_conv=1, # 同样针对变量维度，移除卷积
            expand=2
        )
        
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
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        x = x_enc.permute(0, 2, 1) # [B, V, L]
        x = self.emb(x)            # [B, V, d_model]

        # [关键修复]：显式调用全局变量交互层
        # 输入形状 [B, V, d_model]，在 V 维度上进行双向扫描
        x = self.pre_encoder_mixer(x)

        seasonal_init, trend_init = self.decompsition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
