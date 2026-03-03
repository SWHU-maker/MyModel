import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
# 如果你有这些本地层，请取消注释并确保路径正确
# from layers.Embed import Emb
# from layers.RouterEmbed import RouterEmb

# ----------------- 基础 Embedding 组件 -----------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# ----------------- 核心路由组件：基于数据性质进行筛选 -----------------
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

class PatchRouter(nn.Module):
    def __init__(self, c_in, seq_len, num_path=5):
        super().__init__()
        self.decomp = series_decomp(kernel_size=25) 
        
        self.input_dim = c_in * seq_len * 3
        
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_path)
        )
        
        self.w_noise = nn.Linear(self.input_dim, num_path)

    def forward(self, x):
        B, V, L = x.shape
        seasonal, trend = self.decomp(x)
        ratio = trend / (seasonal + 1e-6)
        
        feat_trend = trend.reshape(B, -1)
        feat_seasonal = seasonal.reshape(B, -1)
        feat_ratio = ratio.reshape(B, -1)
        routing_features = torch.cat([feat_trend, feat_seasonal, feat_ratio], dim=-1)

        logits = self.gate(routing_features)
        
        if self.training:
            noise_logits = self.w_noise(routing_features)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
            
        return F.softmax(logits, dim=-1) # [B, 5]

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model_inner = d_model 
        
        self.ff = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU()
        )
        
        self.combine = nn.Linear(d_model * patch_num, d_model)

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step) 
        x = self.ff(x) 
        x = x.reshape(B, V, -1) 
        x = self.combine(x) 
        return x

class RouterEmb(nn.Module):
    def __init__(self, seq_len, d_model, c_in, dropout=0.1, patch_len=[16, 12, 8, 6, 4]):
        super().__init__()
        patch_steps = [p // 2 for p in patch_len]
        
        self.experts = nn.ModuleList([
            EmbLayer(patch_len[i], patch_steps[i], seq_len, d_model) 
            for i in range(5)
        ])

        self.router = PatchRouter(c_in, seq_len, num_path=5)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weights = self.router(x)
        
        out = 0
        for i, expert in enumerate(self.experts):
            w = weights[:, i].view(-1, 1, 1)
            out += w * expert(x)
        
        out = out + self.position_embedding(out)
        return self.dropout(out)

# ----------------- 创新模块 1：自适应傅里叶滤波器 (来自 Affirm) -----------------
class AdaptiveFourierFilter(nn.Module):
    """
    通过 FFT 转换到频域，使用可学习的阈值过滤掉破坏性的高/低频噪声，
    并应用全局频域复数滤波器提升信噪比。
    """
    def __init__(self, d_model):
        super().__init__()
        # 学习频域的全局滤波器权重
        self.complex_weight = nn.Parameter(torch.randn(1, 1, d_model, 2, dtype=torch.float32) * 0.02)
        
        # 可学习的自适应高低频过滤阈值
        self.high_threshold = nn.Parameter(torch.tensor(0.5))
        self.low_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, L, D]
        # 1. 转换到频域
        x_freq = torch.fft.rfft(x, dim=1)
        
        # 2. 计算幅值并生成自适应掩码 (过滤极端高低频噪声)
        amp = torch.abs(x_freq)
        mask = (amp > self.low_threshold) & (amp < self.high_threshold)
        x_freq = x_freq * mask.float()
        
        # 3. 全局频域滤波 (复数乘法)
        weight = torch.view_as_complex(self.complex_weight)
        x_filtered = x_freq * weight
        
        # 4. 转换回时域
        x_out = torch.fft.irfft(x_filtered, n=x.size(1), dim=1)
        return x_out + x  # 残差连接以防丢失原始有效信息

# ----------------- 创新模块 2：交互式双卷积 & 时间微调 Mamba -----------------
class InteractiveDualMambaBlock(nn.Module):
    """
    结合 Affirm 的 Interactive Dual-Conv 和 TimePro 的 Time-Tune 的 Mamba 核心块。
    """
    def __init__(self, d_model, d_state=16, d_conv_1=2, d_conv_2=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner_dim = int(expand * d_model)
        self.dt_rank = max(d_model // 16, 1)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)

        # 创新点 (Affirm): 双尺度一维因果卷积交互 (捕捉局部短期波动和稍长期的规律)
        self.conv1d_1 = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv_1, 
                                  groups=self.inner_dim, padding=d_conv_1 - 1)
        self.conv1d_2 = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv_2, 
                                  groups=self.inner_dim, padding=d_conv_2 - 1)
        
        # 创新点 (TimePro): 动态时间微调 (Time-Tune)，用于在特征进入SSM前对齐不同变量间的延迟 (Multi-delay)
        self.time_tune_shift = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, groups=self.inner_dim)

        # 核心 SSM 参数投影
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=self.inner_dim)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x) 
        x, res = x_and_res.chunk(2, dim=-1) 

        # 1. 维度转换以适应 Conv1d
        x_conv = rearrange(x, "b l d -> b d l")
        
        # 2. Affirm: 双尺度因果卷积特征提取
        x1 = self.conv1d_1(x_conv)[:, :, :l]
        x2 = self.conv1d_2(x_conv)[:, :, :l]
        
        # 3. TimePro: 动态时间偏移补偿
        x_tune = self.time_tune_shift(x_conv)
        
        # 将双尺度特征与时间偏移特征进行交互融合
        x_interact = F.silu(x1) * F.silu(x2) + x_tune 
        
        # 还原维度
        x_fused = rearrange(x_interact, "b d l -> b l d")

        # 4. 选择性 SSM
        y = self.ssm(x_fused)

        # 5. 门控融合 (与残差分支)
        y = y * F.silu(res) 
        output = self.out_proj(y) 
        return output

    def ssm(self, x):
        (b, l, d_inner) = x.shape
        x_dbl = self.x_proj(x) 
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) 
        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()
        y = self.selective_scan(x, dt, A, B, C, D)
        return y

    def selective_scan(self, x, dt, A, B, C, D):
        (b, l, d_inner) = x.shape
        n = A.size(1) 
        
        dt = F.softplus(dt) 
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0) 
        discretize_A = torch.exp(dt_A) 
        dt_B = dt.unsqueeze(-1) * B.unsqueeze(2) 
        
        x_expanded = x.unsqueeze(-1) 
        h = torch.zeros(b, d_inner, n, device=x.device, dtype=x.dtype) 
        ys = [] 
        
        for t in range(l):
            h = discretize_A[:, t] * h + dt_B[:, t] * x_expanded[:, t, :, 0].unsqueeze(-1)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1) 
            ys.append(y)
        
        y = torch.stack(ys, dim=1) 
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y


class MambaEncoder(nn.Module):
    def __init__(self, d_model, enc_in, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 加入 Affirm 频域自适应滤波器
        self.freq_filter = AdaptiveFourierFilter(d_model)

        # 替换为包含双卷积和时间微调的增强版 Mamba
        self.mamba_layer = InteractiveDualMambaBlock(d_model)

        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 1. 频域去噪与特征提取 (Affirm)
        x_filtered = self.freq_filter(x)
        
        # 2. 双卷积 Mamba 路径 (Affirm + TimePro)
        res = x_filtered
        y = self.mamba_layer(x_filtered)
        y = self.norm1(y + res)

        # 3. 维度交互 (处理变量间关系)
        y_0 = y.permute(0, 2, 1)
        y_1 = self.ff2(y_0)
        y_1 = y_1.permute(0, 2, 1)
        
        # 4. 门控融合
        dec_out = self.norm2(y_1 * y + x)
        return dec_out

# ----------------- 主模型 -----------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.dropout_rate = configs.dropout

        self.decompsition = series_decomp(13)
        
        self.emb = RouterEmb(configs.seq_len, configs.d_model, configs.enc_in, dropout=self.dropout_rate)
        
        self.seasonal_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in, self.dropout_rate)
            for i in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            MambaEncoder(configs.d_model, configs.enc_in, self.dropout_rate)
            for i in range(configs.e_layers)
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