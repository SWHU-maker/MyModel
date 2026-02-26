import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import Emb
from layers.RouterEmbed import RouterEmb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SimpleMambaBlock(nn.Module):
    """
    标准 Mamba 核心块，使用完整的选择性状态空间模型 (Selective SSM)
    参考: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = int(expand * d_model)
        self.dt_rank = max(d_model // 16, 1)  # 确保至少为1

        # 输入投影 (x 和残差)
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)

        # 一维因果卷积：捕捉局部时间依赖
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,  # 深度可分离卷积
            padding=d_conv - 1,
        )

        # 核心 SSM 参数投影
        # 将输入投影到 dt, B, C
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # A 参数初始化 (S4 经典初始化，重要！)
        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=self.inner_dim)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D 参数 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # 输出投影
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        (b, l, d) = x.shape
        
        # 1. 输入投影，分成两个分支
        x_and_res = self.in_proj(x)  # [B, L, 2*inner_dim]
        x, res = x_and_res.chunk(2, dim=-1)  # 各 [B, L, inner_dim]

        # 2. 因果卷积
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]  # 因果卷积，截断padding
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # 3. 选择性 SSM
        y = self.ssm(x)

        # 4. 门控融合 (与残差分支)
        y = y * F.silu(res)  # [B, L, inner_dim]

        # 5. 输出投影
        output = self.out_proj(y)  # [B, L, d_model]
        return output

    def ssm(self, x):
        """
        运行选择性状态空间模型
        x: [B, L, inner_dim]
        """
        (b, l, d_inner) = x.shape
        
        # 投影得到 dt, B, C
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # 计算 dt (离散化步长)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, inner_dim]
        
        # 获取 A 和 D
        A = -torch.exp(self.A_log.float())  # [inner_dim, d_state]
        D = self.D.float()

        # 执行选择性扫描
        y = self.selective_scan(x, dt, A, B, C, D)
        return y

    def selective_scan(self, x, dt, A, B, C, D):
        """
        标准选择性扫描实现 (并行版本)
        x: [B, L, inner_dim]
        dt: [B, L, inner_dim] - 离散化步长
        A: [inner_dim, d_state] - 状态矩阵 (负数)
        B: [B, L, d_state] - 输入相关矩阵
        C: [B, L, d_state] - 输出相关矩阵  
        D: [inner_dim] - 跳跃连接参数
        """
        (b, l, d_inner) = x.shape
        n = A.size(1)  # d_state
        
        # 1. 离散化: 连续参数 -> 离散参数
        # 使用零阶保持 (ZOH) 离散化
        dt = F.softplus(dt)  # 确保 dt > 0
        
        # 离散化 A: Ā = exp(dt * A)
        # A: [d_inner, n], dt: [B, L, d_inner] -> [B, L, d_inner, n]
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, L, d_inner, n]
        discretize_A = torch.exp(dt_A)  # [B, L, d_inner, n]
        
        # 离散化 B: B̄ = dt * B
        dt_B = dt.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, d_inner, n]
        
        # 2. 并行扫描 (Parallel Scan / Blelloch Scan)
        # 计算状态: h_t = Ā_t * h_{t-1} + B̄_t * x_t
        # 使用关联扫描算法实现并行化
        
        # 展开维度以便计算
        x_expanded = x.unsqueeze(-1)  # [B, L, d_inner, 1]
        
        # 初始化状态
        h = torch.zeros(b, d_inner, n, device=x.device, dtype=x.dtype)  # [B, d_inner, n]
        ys = []  # 存储输出
        
        # 序列扫描 (此处为清晰性使用循环，可用并行扫描优化)
        for t in range(l):
            # 状态更新: h = Ā_t * h + B̄_t * x_t
            h = discretize_A[:, t] * h + dt_B[:, t] * x_expanded[:, t, :, 0].unsqueeze(-1)
            
            # 输出: y_t = C_t · h_t (+ D * x_t 在外面处理)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # [B, d_inner]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        
        # 3. 跳跃连接 (D * x)
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaEncoder(nn.Module):
    """
    改进点：将原本的 MLP 替换为 Mamba 块。
    Mamba 能够更好地处理序列中的动态变化，而 MLP 过于僵硬。
    """
    def __init__(self, d_model, enc_in ,dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 创新点 1：使用 Mamba 提取序列特征
        self.mamba_layer = SimpleMambaBlock(d_model)

        # 创新点 2：保留一个轻量级的特征交互层
        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        # 路径 1: Mamba 路径 (捕捉复杂的选择性时序依赖)
        res = x
        y = self.mamba_layer(x)
        y = self.norm1(y + res)

        # 路径 2: 维度交互 (处理变量间关系)
        y_0 = y.permute(0, 2, 1)
        y_1 = self.ff2(y_0)
        y_1 = y_1.permute(0, 2, 1)
        
        # 创新点 3：门控融合 (类似 2025 年流行的 Gated Mamba)
        dec_out = self.norm2(y_1 * y + x)
        return dec_out



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


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.dropout_rate = configs.dropout

        self.decompsition = series_decomp(13)
        # Embedding
        # self.emb = Emb(configs.seq_len, configs.d_model )
        self.emb = RouterEmb(configs.seq_len, configs.d_model , configs.enc_in ,dropout=self.dropout_rate)
        self.seasonal_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in , self.dropout_rate)
            for i in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in , self.dropout_rate)
            for i in range(configs.e_layers)
        ])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
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
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print('我是PatchMamba')

        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]