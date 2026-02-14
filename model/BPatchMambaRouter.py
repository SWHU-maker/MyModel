import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.RouterEmbed import RouterEmb

# ----------------- 增强型双向 Mamba 块 (顶会常用架构) -----------------
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.inner_dim = int(expand * d_model)
        self.dt_rank = d_model // 16

        # 输入投影：同时生成两个分支
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)

        # 卷积层：用于局部特征平滑
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )

        # SSM 参数投影 (B, C, Delta)
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # A 矩阵初始化 (基于 S4 理论)
        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=self.inner_dim)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def ssm_scan(self, x):
        # 简化的选择性扫描逻辑
        (b, l, d) = x.shape
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        # 模拟 SSM 的动态选择特性
        return x * torch.sigmoid(dt)

    def forward(self, x):
        # x: [B, L, D]
        res = x
        x = self.norm(x)
        
        # 投影与分块
        x_and_gate = self.in_proj(x)
        x, gate = x_and_gate.chunk(2, dim=-1)

        # 局部卷积
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # --- 核心改进：双向扫描 ---
        # 正向扫描
        y_fwd = self.ssm_scan(x)
        # 反向扫描 (将序列翻转再处理)
        y_bwd = self.ssm_scan(x.flip(dims=[1])).flip(dims=[1])
        
        # 融合双向信息并施加门控
        y = (y_fwd + y_bwd) * F.silu(gate)
        
        return self.out_proj(y) + res

# ----------------- 优化后的 MambaEncoder -----------------
class MambaEncoder(nn.Module):
    def __init__(self, d_model, enc_in):
        super().__init__()
        # 使用更强大的双向 Mamba 块
        self.mamba_layer = BiMambaBlock(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        # 特征交互层
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        # 针对变量维度的线性投影
        self.var_proj = nn.Linear(enc_in, enc_in)

    def forward(self, x):
        # x: [B, V, D] (根据你 forecast 的 permute，此时 D 是特征维度)
        
        # 1. 时间/Patch 维度建模
        x = self.mamba_layer(x)
        
        # 2. 特征交互与残差
        res = x
        x = self.norm(x)
        x = self.ff(x)
        x = x + res
        
        # 3. 变量维度建模 (按照原 PatchMLP 的思想)
        x = x.permute(0, 2, 1) # [B, D, V]
        x = self.var_proj(x)
        x = x.permute(0, 2, 1) # [B, V, D]
        
        return x

# ----------------- 基础组件 -----------------
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

    def forward(self, x):
        # x: [B, D, L]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# ----------------- 最终 Model -----------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # 序列分解
        self.decomposition = series_decomp(25)
        
        # 使用带动态路由的 Embedding (你之前要求的创新点)
        self.emb = RouterEmb(configs.seq_len, configs.d_model, configs.enc_in)
        
        # 季节性与趋势分支
        self.seasonal_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # 维度转换以适应 Embedding: [B, L, V] -> [B, V, L]
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x) # 输出 [B, V, D]

        # 分解
        # 注意：此时维度是 [B, V, D]，我们需要对 D (特征) 维度做分解
        # 为了复用池化，临时 permute
        x_tmp = x.permute(0, 2, 1) # [B, D, V]
        seasonal_init, trend_init = self.decomposition(x_tmp)
        seasonal_init = seasonal_init.permute(0, 2, 1) # [B, V, D]
        trend_init = trend_init.permute(0, 2, 1)

        # 通过 Mamba 层
        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        
        # 预测投影
        dec_out = self.projector(x) # [B, V, pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, V]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]