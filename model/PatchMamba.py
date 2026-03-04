import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from layers.Embed import Emb

# ================= 创新组件 1: 自适应傅里叶频域滤波器 =================
class AdaptiveFourierFilter(nn.Module):
    """
    通过 FFT 转换到频域，可学习过滤破坏性的噪声，并应用全局复数滤波器提升信噪比。
    """
    def __init__(self, d_model):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, 1, d_model, 2, dtype=torch.float32) * 0.02)
        self.high_threshold = nn.Parameter(torch.tensor(0.5))
        self.low_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, L, D] (在这里 L 实际上代表变量序列长度 enc_in)
        x_freq = torch.fft.rfft(x, dim=1)
        
        amp = torch.abs(x_freq)
        mask = (amp > self.low_threshold) & (amp < self.high_threshold)
        x_freq = x_freq * mask.float()
        
        weight = torch.view_as_complex(self.complex_weight)
        x_filtered = x_freq * weight
        
        x_out = torch.fft.irfft(x_filtered, n=x.size(1), dim=1)
        return x_out + x 

# ================= 基础序列分解 =================
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

# ================= 核心 Mamba 编码器 (双向扫描 + 局部卷积) =================
class Encoder(nn.Module):
    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 加入频域自适应滤波器，清除输入特征的噪声
        self.freq_filter = AdaptiveFourierFilter(d_model)

        # 针对变量维度的双向 Mamba (原 Mamba1)
        self.mamba1_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba1_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        # 创新：引入局部因果卷积增强 Mamba1 对相邻特征间的局部关系感知
        self.local_conv1 = nn.Conv1d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in)

        # 针对时序嵌入维度的双向 Mamba (原 Mamba2)
        self.mamba2_fwd = Mamba(d_model=enc_in, d_state=16, d_conv=4, expand=2)
        self.mamba2_bwd = Mamba(d_model=enc_in, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        # x shape: [B, enc_in, d_model]
        
        # 1. 频域去噪滤波
        x_filtered = self.freq_filter(x)
        
        # 2. 变量维度 (Channel) 的双向 Mamba 处理
        y_fwd = self.mamba1_fwd(x_filtered)
        
        x_rev = torch.flip(x_filtered, dims=[1])
        y_bwd = self.mamba1_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        # 局部卷积提取
        x_conv = self.local_conv1(x_filtered)
        
        # 多路径特征聚合
        y_0 = y_fwd + y_bwd + x_conv + x
        y_0 = self.norm1(y_0)
        
        # 3. 时序嵌入维度 (Temporal) 的双向 Mamba 处理
        y_1_in = y_0.permute(0, 2, 1) # [B, d_model, enc_in]
        
        y_1_fwd = self.mamba2_fwd(y_1_in)
        
        y_1_rev = torch.flip(y_1_in, dims=[1])
        y_1_bwd = self.mamba2_bwd(y_1_rev)
        y_1_bwd = torch.flip(y_1_bwd, dims=[1])
        
        y_1 = y_1_fwd + y_1_bwd
        y_1 = y_1.permute(0, 2, 1) # 转置回 [B, enc_in, d_model]
        
        # 4. 交互式门控机制与残差连接
        y_2 = y_1 * y_0 + x
        y_2 = self.norm2(y_2)

        return y_2

# ================= 预测主模型 (完全保留原有 API 接口) =================
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
            for _ in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)

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
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]