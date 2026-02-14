import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------- 基础 Embedding 组件 (保持不变) -----------------
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

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# ----------------- 路由辅助组件 (严格参照 Pathformer) -----------------

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
    严格参照 Pathformer 的 AMS 路由逻辑：
    输入分解项，输出动态 Patch 权重
    """
    def __init__(self, c_in, seq_len, num_path):
        super().__init__()
        self.decomp = series_decomp(25) # 较大的窗口提取更稳健的趋势
        
        # 路由输入维度：Trend(c_in*seq_len) + Seasonal(c_in*seq_len)
        # 我们在这里实现你要求的：趋势 + 周期 + 趋势/周期
        input_dim = c_in * seq_len * 3 
        
        self.gate = nn.Linear(input_dim, num_path)
        self.noise_gate = nn.Linear(input_dim, num_path)

    def forward(self, x):
        # x: [B, V, L]
        B, V, L = x.shape
        
        # 1. 序列分解
        seasonal, trend = self.decomp(x)
        
        # 2. 构建特征：趋势项 + 周期项 + 趋势/周期项
        # 为了数值稳定，除法项加入 epsilon
        ratio = trend / (seasonal + 1e-5)
        
        # 展平并拼接
        feat_t = trend.reshape(B, -1)
        feat_s = seasonal.reshape(B, -1)
        feat_r = ratio.reshape(B, -1)
        
        routing_input = torch.cat([feat_t, feat_s, feat_r], dim=-1) # [B, 3*V*L]
        
        # 3. Noisy Top-K Gating 逻辑
        logits = self.gate(routing_input)
        
        if self.training:
            # 引入噪声以增强训练时的路径探索性
            noise_scale = F.softplus(self.noise_gate(routing_input))
            logits = logits + torch.randn_like(logits) * noise_scale
            
        return F.softmax(logits, dim=-1) # [B, num_path]

# ----------------- 核心 Embedding 逻辑 -----------------

class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        # 注意：这里不再除以 patch_num，因为我们要加权聚合，保持各分支维度均为 d_model
        self.ff = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU()
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(d_model * patch_num, d_model),
        )

    def forward(self, x):
        # x: [B, V, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step) # [B, V, num, len]
        x = self.ff(x) # [B, V, num, d_model]
        x = torch.mean(x, dim=2) # 聚合 patch 维，保持维度对齐
        return x # [B, V, d_model]


class Emb(nn.Module):
    """
    修改后的动态 Patch 路由 Emb 层
    """
    def __init__(self, seq_len, d_model, c_in, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        
        # 4 个并行的 Patch 专家分支
        # 每个分支独立输出 d_model 维特征
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

        # 核心路由
        self.router = PatchRouter(c_in, seq_len, num_path=4)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [B, V, L]
        
        # 1. 路由权重计算
        weights = self.router(x) # [B, 4]
        
        # 2. 计算 4 个 Patch 尺度的 Embedding
        s_x1 = self.EmbLayer_1(x) # [B, V, d_model]
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        
        # 3. 动态加权聚合 (AMS Combine 逻辑)
        # weights[:, 0] 是第 1 个 patch 分支的权重
        out = s_x1 * weights[:, 0].view(-1, 1, 1) + \
              s_x2 * weights[:, 1].view(-1, 1, 1) + \
              s_x3 * weights[:, 2].view(-1, 1, 1) + \
              s_x4 * weights[:, 3].view(-1, 1, 1)
        
        # 4. 加入位置信息
        out = out + self.position_embedding(out)
        
        return self.dropout(out)

# 以下原有的 Embedding 类保持原样以防其他模型引用
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't': self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        return self.hour_embed(x[:, :, 3]) + self.weekday_embed(x[:, :, 2]) + \
               self.day_embed(x[:, :, 1]) + self.month_embed(x[:, :, 0]) + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)
    def forward(self, x):
        return self.embed(x)