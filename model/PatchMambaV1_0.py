import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        # 引入前后 Padding 保证 Patch 能完全 Cover 序列的边界信息
        self.pad_left = patch_len // 2
        self.pad_right = patch_len - self.pad_left
        padded_seq_len = seq_len + self.pad_left + self.pad_right
        
        # 动态计算 pad 后的 patch 数量
        self.patch_num = (padded_seq_len - patch_len) // patch_step + 1
        
        # 为了保证最后能拼接回 d_model，我们将特征维度均分
        self.d_model_inner = d_model // self.patch_num 
        
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model_inner),
            nn.GELU()  # 引入 GELU 激活增强特征非线性
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model_inner * self.patch_num, d_model),
        )

    def forward(self, x):
        B, V, L = x.shape
        
        # [B, V, L] -> [B, V, L + patch_len]
        x = F.pad(x, (self.pad_left, self.pad_right), "constant", 0)
        
        # 切片: [B, V, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        
        x = self.ff(x)          # [B, V, patch_num, d_model_inner]
        x = self.flatten(x)     # [B, V, patch_num * d_model_inner]
        x = self.ff_1(x)        # [B, V, d_model]
        
        return x


class Emb(nn.Module):
    def __init__(self, seq_len, d_model, patch_len=[96, 48, 24, 12, 6, 3], k=4, noisy_gating=True):
        super().__init__()
        patch_step = patch_len
        self.num_experts = len(patch_len)
        self.k = min(k, self.num_experts) 
        self.noisy_gating = noisy_gating  

        # 实例化专家分支
        self.EmbLayers = nn.ModuleList([
            EmbLayer(patch_len[i], patch_step[i] // 2, seq_len, d_model) 
            for i in range(self.num_experts)
        ])

        # 时频域联合路由网络
        freq_len = seq_len // 2 + 1
        routing_dim = seq_len + freq_len
        
        self.w_gate = nn.Sequential(
            nn.Linear(routing_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_experts)
        )
        self.w_noise = nn.Linear(routing_dim, self.num_experts)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # 提取频域特征
        x_freq = torch.abs(torch.fft.rfft(x, dim=-1))
        
        # 拼接时域和频域特征
        routing_features = torch.cat([x, x_freq], dim=-1)
        
        clean_logits = self.w_gate(routing_features)
        
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(routing_features)
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(self.k, dim=-1)
        top_k_gates = F.softmax(top_logits, dim=-1)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_indices, top_k_gates)

        # 负载均衡损失 (Load Balancing Loss)
        if self.training:
            # 计算每个专家的平均选择率 f_i
            # mask: [B, V, num_experts]
            mask = torch.zeros_like(logits).scatter_(-1, top_indices, 1.0)
            f = mask.mean(dim=(0, 1))
            
            # 计算每个专家的平均门控概率 P_i
            # probs: [B, V, num_experts]
            probs = F.softmax(logits, dim=-1)
            P = probs.mean(dim=(0, 1))
            
            # Loss = N * sum(f_i * P_i)
            loss = self.num_experts * torch.sum(f * P)
        else:
            loss = 0.0

        return gates, loss

    def forward(self, x):
        # x: [B, V, L]
        gates, loss = self.noisy_top_k_gating(x, self.training)
        
        # 获取所有尺度的表征
        expert_outputs = [layer(x) for layer in self.EmbLayers]
        expert_outputs = torch.stack(expert_outputs, dim=-2) # [B, V, Experts, d_model]

        gates = gates.unsqueeze(-1)  # [B, V, Experts, 1]
        
        # 加权融合
        s_out = (expert_outputs * gates).sum(dim=-2)  # [B, V, d_model]
        
        return s_out, loss

class MultiScaleEmb(nn.Module):
    def __init__(self, seq_len, d_model, scales=[1, 2], patch_len=[96, 48, 24, 12, 6, 3], k=4, noisy_gating=True):
        super().__init__()
        self.scales = scales
        self.embs = nn.ModuleList([
            Emb(seq_len // s, d_model, patch_len, k, noisy_gating) 
            for s in scales
        ])
        
    def forward(self, x):
        # x: [B, V, L]
        outs = []
        losses = []
        for i, scale in enumerate(self.scales):
            if scale == 1:
                x_s = x
            else:
                # Downsample
                x_s = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            
            out, loss = self.embs[i](x_s)
            outs.append(out)
            losses.append(loss)
            
        return outs, losses




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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
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
        
        return self.out_proj(y)


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

        # 1. Channel Mixing (变量间交互)
        # 使用 Adaptive BiMamba，引入门控机制提升融合效果
        self.channel_mixer = BiMambaBlock(
            d_model=d_model, 
            d_state=16,      
            d_conv=1,        
            expand=2,        
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
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        
        # Time Mixing
        y_1 = y_0.permute(0, 2, 1) 
        y_1 = self.time_mixer(y_1)
        y_1 = y_1.permute(0, 2, 1)
        
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
        
        # Embedding - Modified for TimeMixer++ Multi-scale
        # 借鉴 TimeMixer++ 的多尺度混合机制
        self.scales = [1, 2] # Default scales
        self.emb = MultiScaleEmb(configs.seq_len, configs.d_model, scales=self.scales)
        
        # 全局变量交互层 (Pre-Encoder)
        # 同样使用 Adaptive BiMamba
        self.pre_encoder_mixer = BiMambaBlock(
            d_model=configs.d_model,
            d_state=16,
            d_conv=1,
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

        # Projector for each scale
        self.projectors = nn.ModuleList([
            nn.Linear(configs.d_model, configs.pred_len, bias=True)
            for _ in self.scales
        ])

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        x_in = x_enc.permute(0, 2, 1) # [B, V, L]
        
        # Multi-scale Embedding
        emb_outs, losses = self.emb(x_in)
        
        dec_out_sum = 0
        total_aux_loss = 0
        
        for i, (x, loss) in enumerate(zip(emb_outs, losses)):
            total_aux_loss += loss
            
            # 显式调用全局变量交互
            x = self.pre_encoder_mixer(x)

            seasonal_init, trend_init = self.decompsition(x)

            for mod in self.seasonal_layers:
                seasonal_init = mod(seasonal_init)
            for mod in self.trend_layers:
                trend_init = mod(trend_init)

            x = seasonal_init + trend_init
            
            # Project using scale-specific projector
            out = self.projectors[i](x) # [B, V, pred_len]
            out = out.permute(0, 2, 1) # [B, pred_len, V]
            
            if isinstance(dec_out_sum, int):
                dec_out_sum = out
            else:
                dec_out_sum = dec_out_sum + out
        
        dec_out = dec_out_sum
        
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, total_aux_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print("我是V1_0")
        dec_out, aux_loss = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :], aux_loss