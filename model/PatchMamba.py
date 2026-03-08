import torch
import torch.nn as nn
from layers.Embed import Emb
from mamba_ssm import Mamba

# ==========================================
# 核心组件：Adaptive BiMambaBlock
# 改进点：引入通道注意力机制，动态加权双向特征
# ==========================================
class BiMambaBlock(nn.Module):
    """
    自适应双向 Mamba 模块
    1. 双向扫描：消除 Channel 顺序偏差
    2. 通道注意力：动态融合正向和反向特征，不再是简单的相加
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        # Forward
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # Backward
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 改进：引入门控融合机制，而不是简单的相加
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) # 【新增】应用 dropout

    def forward(self, x):
        # x: [B, L, D]
        y_f = self.mamba_f(x)
        
        # Backward pass
        x_b = torch.flip(x, dims=[1])
        y_b = self.mamba_b(x_b)
        y_b = torch.flip(y_b, dims=[1])
        
        # 改进：动态融合
        # 拼接双向特征
        combined = torch.cat([y_f, y_b], dim=-1) # [B, L, 2*D]
        z = self.gate(combined) # [B, L, D] 融合系数
        
        # 加权融合：z * y_f + (1-z) * y_b
        y = z * y_f + (1 - z) * y_b
        
        return self.dropout(self.out_proj(y)) # 【修改】输出时经过 dropout


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
    def __init__(self, d_model, enc_in, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # 【新增】应用 dropout

        # 1. Channel Mixing (变量间交互)
        # 使用 Adaptive BiMamba，引入门控机制提升融合效果
        self.channel_mixer = BiMambaBlock(
            d_model=d_model, 
            d_state=16,      
            d_conv=1,        
            expand=2,        
            dropout=dropout  # 【修改】向下传递 dropout
        )

        # 2. Time Mixing (时间步交互)
        self.time_mixer = Mamba(
            d_model=enc_in,  
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, x):
        # x: [B, enc_in, d_model]
        
        # Channel Mixing
        y_0 = self.channel_mixer(x)
        y_0 = self.dropout(y_0) + x # 【修改】残差连接前做 dropout
        y_0 = self.norm1(y_0)
        
        # Time Mixing
        y_1 = y_0.permute(0, 2, 1) 
        y_1 = self.time_mixer(y_1)
        y_1 = y_1.permute(0, 2, 1)
        
        y_2 = self.dropout(y_1 * y_0) + x # 【修改】残差连接前做 dropout
        y_2 = self.norm2(y_2) # 【修复】原来重复使用了 self.norm1，这里修正为正确的 self.norm2

        return y_2


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # 【新增】智能获取运行脚本中名为 dropout 或 drop 的参数，若没有配置则默认给 0.1
        self.dropout = getattr(configs, 'dropout', getattr(configs, 'drop', 0.1))

        self.decompsition = series_decomp(13)
        
        # Embedding
        self.emb = Emb(configs.seq_len, configs.d_model, dropout=self.dropout) # 【修改】传入 dropout
        
        # 全局变量交互层 (Pre-Encoder)
        # 同样使用 Adaptive BiMamba
        self.pre_encoder_mixer = BiMambaBlock(
            d_model=configs.d_model,
            d_state=16,
            d_conv=1,
            expand=2,
            dropout=self.dropout # 【修改】传入 dropout
        )
        
        self.seasonal_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in, dropout=self.dropout) # 【修改】传入 dropout
            for i in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in, dropout=self.dropout) # 【修改】传入 dropout
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

        # 显式调用全局变量交互
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