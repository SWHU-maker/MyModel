import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class PatchRouter(nn.Module):
    def __init__(self, c_in, seq_len, num_path=5): # 路径数改为 5
        super().__init__()
        self.decomp = series_decomp(kernel_size=25) 
        
        # 输入维度保持不变 (trend, seasonal, ratio)
        self.input_dim = c_in * seq_len * 3
        
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_path)
        )
        
        self.w_noise = nn.Linear(self.input_dim, num_path)

    def forward(self, x):
        B, V, L = x.shape
        
        # 提取数据性质
        seasonal, trend = self.decomp(x)
        ratio = trend / (seasonal + 1e-6)
        
        # 构建特征向量
        feat_trend = trend.reshape(B, -1)
        feat_seasonal = seasonal.reshape(B, -1)
        feat_ratio = ratio.reshape(B, -1)
        routing_features = torch.cat([feat_trend, feat_seasonal, feat_ratio], dim=-1)

        # 计算路由 logits
        logits = self.gate(routing_features)
        
        if self.training:
            # 训练时加入噪声增强泛化性
            noise_logits = self.w_noise(routing_features)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
            
        # 直接返回 5 个 patch size 的概率分布
        return F.softmax(logits, dim=-1) # [B, 5]

# ----------------- 核心 Embedding 逻辑 -----------------

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model_inner = d_model 
        
        # 投影层：将分好的 patch 映射到 d_model 空间
        self.ff = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU()
        )
        
        # 将所有 patch 聚合的层
        self.combine = nn.Linear(d_model * patch_num, d_model)

    def forward(self, x):
        # x: [B, V, L]
        B, V, L = x.shape
        # 分 patch
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step) # [B, V, patch_num, patch_len]
        # 映射到特征空间
        x = self.ff(x) # [B, V, patch_num, d_model]
        # 展平 patch 维度进行聚合
        x = x.reshape(B, V, -1) # [B, V, patch_num * d_model]
        x = self.combine(x) # [B, V, d_model]
        return x


class RouterEmb(nn.Module):
    def __init__(self, seq_len, d_model, c_in, patch_len=[16, 12, 8, 6, 4]):
        super().__init__()
        # 这里的 patch_step 依然采用 patch_len 的一半作为重叠
        patch_steps = [p // 2 for p in patch_len]
        
        # 创新点：使用 ModuleList 管理 5 个不同尺度的专家
        self.experts = nn.ModuleList([
            EmbLayer(patch_len[i], patch_steps[i], seq_len, d_model) 
            for i in range(5)
        ])

        # 路由选择器，对应 5 个路径
        self.router = PatchRouter(c_in, seq_len, num_path=5)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [B, V, L]
        
        # 1. 获取 5 个专家的权重 [B, 5]
        weights = self.router(x)
        
        # 2. 依次计算 5 个专家的输出并进行加权聚合
        # 初始化输出张量
        out = 0
        for i, expert in enumerate(self.experts):
            # 提取第 i 个专家的权重并变换维度为 [B, 1, 1] 以便广播
            w = weights[:, i].view(-1, 1, 1)
            out += w * expert(x)
        
        # 3. 融合位置编码与 Dropout
        out = out + self.position_embedding(out)
        
        return self.dropout(out)