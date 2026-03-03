import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)

        x = self.ff_1(x)
        return x


class Emb(nn.Module):
    # 修改1：将 patch_len 的默认列表扩展到 6 个（例如 96, 48, 24, 12, 6, 3），你也可以在外部传参指定。
    # 修改2：将 k 设置为 4，代表每次从这 6 个尺度中选排名前 4 的。
    def __init__(self, seq_len, d_model, patch_len=[96, 48, 24, 12, 6, 3], k=4, noisy_gating=True):
        super().__init__()
        patch_step = patch_len
        self.num_experts = len(patch_len) # 专家数量，现在是 6
        self.k = min(k, self.num_experts) # Top-K 路由选择，也就是 k=4
        self.noisy_gating = noisy_gating  

        # 实例化 6 个 Expert 分支
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)
        self.EmbLayer_5 = EmbLayer(patch_len[4], patch_step[4] // 2, seq_len, d_model)
        self.EmbLayer_6 = EmbLayer(patch_len[5], patch_step[5] // 2, seq_len, d_model)

        # 路由网络：用于给 6 个尺度打分
        self.w_gate = nn.Linear(seq_len, self.num_experts)
        self.w_noise = nn.Linear(seq_len, self.num_experts)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        计算 6 个尺度的得分，并仅保留 Top 4 的权重
        """
        clean_logits = self.w_gate(x)
        
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = (F.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 关键：“6选4” 发生在这里，选出 logits 分数最高的 4 个分支
        top_logits, top_indices = logits.topk(self.k, dim=-1)
        
        # 对选出的 4 个分支进行 Softmax 归一化，使其权重加起来等于 1
        top_k_gates = F.softmax(top_logits, dim=-1)

        # 将这 4 个分支的权重按原索引填回，剩下落选的 2 个分支权重默认为 0
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_indices, top_k_gates)
        return gates

    def forward(self, x):
        # 1. 计算权重，gates.shape 变成了 [B, V, 6] (其中有 4 个非零值，2 个零值)
        gates = self.noisy_top_k_gating(x, self.training)

        # 2. 获取每个尺度的表征
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        s_x5 = self.EmbLayer_5(x)
        s_x6 = self.EmbLayer_6(x)

        # 3. 堆叠起来，expert_outputs.shape: [B, V, 6, d_model]
        expert_outputs = torch.stack([s_x1, s_x2, s_x3, s_x4, s_x5, s_x6], dim=-2)

        # 4. 根据 Gates 动态加权求和
        gates = gates.unsqueeze(-1)  # 变为 [B, V, 6, 1] 以便广播机制相乘
        
        # 落选的 2 个分支由于 gates 为 0，在相乘时结果即为 0，相当于只加和了排名前 4 的分支
        s_out = (expert_outputs * gates).sum(dim=-2)  # 输出仍为 [B, V, d_model]

        return s_out