import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ==========================================
# Part 1: Embedding Layers
# ==========================================

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step
        self.pad_left = patch_len // 2
        self.pad_right = patch_len - self.pad_left
        padded_seq_len = seq_len + self.pad_left + self.pad_right
        self.patch_num = (padded_seq_len - patch_len) // patch_step + 1
        self.d_model_inner = d_model // self.patch_num 
        
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model_inner),
            nn.GELU()
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.ff_1 = nn.Sequential(nn.Linear(self.d_model_inner * self.patch_num, d_model))

    def forward(self, x):
        # x: [B, V, L]
        x = F.pad(x, (self.pad_left, self.pad_right), "constant", 0)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)
        x = self.ff_1(x)
        return x

class Emb(nn.Module):
    def __init__(self, seq_len, d_model, patch_len=[96, 48, 24, 12, 6, 3], k=4, noisy_gating=True):
        super().__init__()
        self.num_experts = len(patch_len)
        self.k = min(k, self.num_experts) 
        self.noisy_gating = noisy_gating  
        self.EmbLayers = nn.ModuleList([
            EmbLayer(patch_len[i], patch_len[i] // 2, seq_len, d_model) 
            for i in range(self.num_experts)
        ])
        freq_len = seq_len // 2 + 1
        routing_dim = seq_len + freq_len
        self.w_gate = nn.Sequential(nn.Linear(routing_dim, 128), nn.GELU(), nn.Linear(128, self.num_experts))
        self.w_noise = nn.Linear(routing_dim, self.num_experts)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x_freq = torch.abs(torch.fft.rfft(x, dim=-1))
        routing_features = torch.cat([x, x_freq], dim=-1)
        clean_logits = self.w_gate(routing_features)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(routing_features)
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(self.k, dim=-1)
        top_k_gates = F.softmax(top_logits, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, top_indices, top_k_gates)
        
        loss = 0.0
        if self.training:
            mask = torch.zeros_like(logits).scatter_(-1, top_indices, 1.0)
            f = mask.mean(dim=(0, 1))
            P = F.softmax(logits, dim=-1).mean(dim=(0, 1))
            loss = self.num_experts * torch.sum(f * P)
        return gates, loss

    def forward(self, x):
        gates, loss = self.noisy_top_k_gating(x, self.training)
        expert_outputs = torch.stack([layer(x) for layer in self.EmbLayers], dim=-2)
        s_out = (expert_outputs * gates.unsqueeze(-1)).sum(dim=-2)
        return s_out, loss

class MultiScaleEmb(nn.Module):
    def __init__(self, seq_len, d_model, scales=[1, 2], patch_len=[96, 48, 24, 12, 6, 3], k=4):
        super().__init__()
        self.scales = scales
        self.embs = nn.ModuleList([Emb(seq_len // s, d_model, patch_len, k) for s in scales])
        
    def forward(self, x):
        outs, losses = [], []
        for i, scale in enumerate(self.scales):
            x_s = x if scale == 1 else F.avg_pool1d(x, kernel_size=scale, stride=scale)
            out, loss = self.embs[i](x_s)
            outs.append(out)
            losses.append(loss)
        return outs, losses

# ==========================================
# Part 2: Model Components
# ==========================================

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        y_f = self.mamba_f(x)
        y_b = torch.flip(self.mamba_b(torch.flip(x, dims=[1])), dims=[1])
        combined = torch.cat([y_f, y_b], dim=-1)
        z = self.gate(combined)
        y = z * y_f + (1 - z) * y_b
        return self.out_proj(y)

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 保持原始平滑逻辑：在特征维度末尾进行 padding 和平均
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
        self.channel_mixer = BiMambaBlock(d_model=d_model, d_state=16, d_conv=1, expand=2)
        self.time_mixer = Mamba(d_model=enc_in, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        # x: [B, enc_in, d_model]
        y_0 = self.channel_mixer(x)
        y_0 = self.norm1(y_0 + x)
        
        y_1 = y_0.permute(0, 2, 1) 
        y_1 = self.time_mixer(y_1)
        y_1 = y_1.permute(0, 2, 1)
        
        y_2 = self.norm2(y_1 * y_0 + x)
        return y_2

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.scales = [1, 2]
        
        self.emb = MultiScaleEmb(configs.seq_len, configs.d_model, scales=self.scales)
        self.decompsition = series_decomp(13)
        
        # 跨尺度交互层
        self.cross_scale_interact = BiMambaBlock(d_model=configs.d_model)
        
        self.seasonal_layers = nn.ModuleList([Encoder(configs.d_model, configs.enc_in) for _ in range(configs.e_layers)])
        self.trend_layers = nn.ModuleList([Encoder(configs.d_model, configs.enc_in) for _ in range(configs.e_layers)])
        
        self.projectors = nn.ModuleList([
            nn.Linear(configs.d_model, configs.pred_len) for _ in self.scales
        ])

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        x_in = x_enc.permute(0, 2, 1) # [B, V, L]
        emb_outs, losses = self.emb(x_in) 
        
        # 跨尺度交互：让 Scale 1 和 Scale 2 在分解前相互感知
        stacked_embs = torch.stack(emb_outs, dim=1) # [B, Scales, V, D]
        B, S, V, D = stacked_embs.shape
        interacted = self.cross_scale_interact(stacked_embs.view(B*V, S, D))
        interacted = interacted.view(B, V, S, D).permute(2, 0, 1, 3) # [S, B, V, D]

        dec_out_sum = 0
        for i in range(len(self.scales)):
            x = interacted[i] 
            
            # 原始分解逻辑：作用在 [B, V, d_model] 的最后一个维度上
            seasonal_init, trend_init = self.decompsition(x)

            for mod in self.seasonal_layers:
                seasonal_init = mod(seasonal_init)
            for mod in self.trend_layers:
                trend_init = mod(trend_init)

            # 投影回预测长度
            out = self.projectors[i](seasonal_init + trend_init).permute(0, 2, 1) # [B, pred_len, V]
            
            if isinstance(dec_out_sum, int):
                dec_out_sum = out
            else:
                dec_out_sum = dec_out_sum + out
        
        if self.use_norm:
            dec_out_sum = dec_out_sum * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        return dec_out_sum, sum(losses)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        print("我是V1_1")
        dec_out, aux_loss = self.forecast(x_enc)
        return dec_out, aux_loss
