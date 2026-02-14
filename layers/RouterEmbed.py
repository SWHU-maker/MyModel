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
    """
    修改后的路由：真正利用原始数据的【趋势性】、【周期性】和【趋势/周期比例】
    """
    def __init__(self, c_in, seq_len, num_path):
        super().__init__()
        # 使用分解层处理原始数据
        self.decomp = series_decomp(kernel_size=25) 
        
        # 路由的特征输入：
        # 1. 趋势项特征 (c_in * seq_len)
        # 2. 周期项特征 (c_in * seq_len)
        # 3. 趋势/周期比例特征 (c_in * seq_len)
        self.input_dim = c_in * seq_len * 3
        
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_path)
        )
        
        # 噪声层：用于训练时的路径探索
        self.w_noise = nn.Linear(self.input_dim, num_path)

    def forward(self, x):
        # x: 原始数据 [B, V, L]
        B, V, L = x.shape
        
        # --- A. 提取数据性质 ---
        # 1. 提取周期性(res)和趋势性(trend)
        seasonal, trend = self.decomp(x)
        
        # 2. 计算趋势/周期比例 (加入 eps 防止除零)
        # 这里的 ratio 反映了数据的信噪比或波动特征
        ratio = trend / (seasonal + 1e-6)
        
        # --- B. 构建路由特征向量 ---
        # 将三个性质展平并拼接，作为决策依据
        feat_trend = trend.reshape(B, -1)     # [B, V*L]
        feat_seasonal = seasonal.reshape(B, -1) # [B, V*L]
        feat_ratio = ratio.reshape(B, -1)     # [B, V*L]
        
        routing_features = torch.cat([feat_trend, feat_seasonal, feat_ratio], dim=-1) # [B, V*L*3]

        # --- C. 计算路由权重 ---
        logits = self.gate(routing_features)
        
        if self.training:
            # 引入噪声分量 (Noisy Gating)
            noise_logits = self.w_noise(routing_features)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
            
        # 返回 Softmax 后的权重，决定 4 个 Patch Size 的占比
        return F.softmax(logits, dim=-1) # [B, 4]


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
    """
    带动态路由选择的 Embedding 层
    """
    def __init__(self, seq_len, d_model, c_in, patch_len=[16, 12, 8, 4]):
        super().__init__()
        patch_step = patch_len
        
        # 4 个并行的多尺度专家
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

        # 路由选择器：输入 c_in 以便处理原始数据
        self.router = PatchRouter(c_in, seq_len, num_path=4)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x 为输入的原始数据 [B, V, L]
        
        # 1. 关键步骤：利用原始数据的趋势/周期/比例 计算路由权重
        # weights 的大小为 [B, 4]
        weights = self.router(x)
        
        # 2. 计算 4 种不同 Patch 大小的特征表达
        s_x1 = self.EmbLayer_1(x) # [B, V, d_model]
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        
        # 3. 动态加权聚合
        # 根据数据性质，动态决定使用哪种尺度的特征更多
        out = s_x1 * weights[:, 0].view(-1, 1, 1) + \
              s_x2 * weights[:, 1].view(-1, 1, 1) + \
              s_x3 * weights[:, 2].view(-1, 1, 1) + \
              s_x4 * weights[:, 3].view(-1, 1, 1)
        
        # 4. 融合位置编码
        out = out + self.position_embedding(out)
        
        return self.dropout(out)