import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return gates

    def forward(self, x):
        # x: [B, V, L]
        gates = self.noisy_top_k_gating(x, self.training)

        # 获取所有尺度的表征
        expert_outputs = [layer(x) for layer in self.EmbLayers]
        expert_outputs = torch.stack(expert_outputs, dim=-2) # [B, V, Experts, d_model]

        gates = gates.unsqueeze(-1)  # [B, V, Experts, 1]
        
        # 加权融合
        s_out = (expert_outputs * gates).sum(dim=-2)  # [B, V, d_model]
        
        return s_out