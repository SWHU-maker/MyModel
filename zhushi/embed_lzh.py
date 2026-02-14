
import torch
import torch.nn as nn
import math

# 1. 位置编码：为序列中的每个时间点提供位置信息（用于非循环网络如Transformer）
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 初始化一个全零张量作为位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 位置编码是固定的，不需要梯度更新

        # 生成位置索引列向量 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算正弦/余弦函数的缩放频率项（分母部分）
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 偶数列使用正弦函数编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列使用余弦函数编码
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度，变为 [1, max_len, d_model]，方便后续与 Batch 数据相加
        pe = pe.unsqueeze(0)
        # 注册为 buffer，这样它会随模型保存但不会被优化器更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入序列 x 的实际长度截取对应长度的位置编码
        return self.pe[:, :x.size(1)]


# 2. 标记嵌入：通过一维卷积提取局部时间特征
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据 PyTorch 版本选择 padding 大小以保持长度一致
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 使用卷积核为 3 的一维卷积将输入通道 c_in 映射到 d_model 维度
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 对模型内部的卷积层进行 Kaiming 正态分布初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x 形状: [Batch, Length, Channel] -> 变换为 [Batch, Channel, Length] 进行卷积
        # 卷积后再转置回 [Batch, Length, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# 3. 固定编码：基于正余弦预定义位置编码的 Embedding 层
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # 初始化权重矩阵
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 使用 nn.Embedding 结构，但将其权重初始化为上述固定正余弦值
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # 查找嵌入表并从计算图中分离（不训练）
        return self.emb(x).detach()


# 4. 时间特征嵌入：将离散的时间特征（月、日、周、时、分）转换为向量
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # 定义各个时间维度的粒度大小
        minute_size = 4    # 15分钟一个刻度，一小时4个
        hour_size = 24
        weekday_size = 7
        day_size = 32      # 考虑到每月天数
        month_size = 13    # 1-12月

        # 根据 embed_type 选择使用固定编码还是可学习的 Embedding
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        # 如果频率是 't' (分钟)，初始化分钟嵌入
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        # 初始化其余维度的嵌入层
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long() # 转换为长整型用于索引
        # 根据输入特征矩阵的索引提取各个维度的嵌入向量，并相加融合
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # 返回融合后的时间嵌入
        return hour_x + weekday_x + day_x + month_x + minute_x


# 5. 时间频率线性嵌入：直接用线性层处理归一化后的时间特征（如 timeF 格式）
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # 根据时间频率确定输入维度
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # 使用简单的线性层进行映射
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# 6. 标准数据嵌入层：融合值嵌入、时间嵌入和位置嵌入
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 1. 数值映射层
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 2. 位置信息层
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 3. 时间特征层（根据 embed_type 选择类）
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 如果没有提供时间标记，只加位置编码；否则将三者相加
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# 7. iTransformer 风格的反转数据嵌入层：将变量作为 Token 处理
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # 对整个序列长度（c_in）进行线性映射
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x 形状: [Batch, Time, Variate] -> [Batch, Variate, Time]
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 如果有时间协变量，拼接到特征维度后一起嵌入
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        return self.dropout(x)


# 8. 单层补丁嵌入：PatchMLP 的核心组件
class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        # 计算切片后的补丁数量
        patch_num = int((seq_len - patch_len) / patch_step + 1)
        # 每个补丁分配的子维度
        self.d_model = d_model // patch_num
        # 局部特征提取：将补丁长度映射到子维度
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2) # 展平补丁特征维度

        # 全局映射：将所有补丁特征融合回 d_model 维度
        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):
        # x 形状: [B, V, L]
        # 使用 unfold 函数进行滑动窗口切片（即 Patching）
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        # x 形状变为: [B, V, patch_num, patch_len]
        x = self.ff(x)       # [B, V, patch_num, d_model_sub]
        x = self.flatten(x)  # [B, V, patch_num * d_model_sub]
        x = self.ff_1(x)     # [B, V, d_model]
        return x


# 9. 多尺度补丁嵌入：最终被 PatchMLP 模型调用的嵌入类
class Emb(nn.Module):
    def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len # 步长等于长度（或根据定义调整）
        d_model = d_model // 4 # 因为有4个尺度的输出，所以平分维度
        
        # 初始化四个不同尺度的补丁嵌入层
        # 这里的步长 patch_step // 2 意味着补丁之间有 50% 的重叠
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

    def forward(self, x):
        # 这里的 x 是经过变量转置后的 [Batch, Variate, Length]
        # 分别计算四个尺度的嵌入特征
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        # 在最后一个维度拼接所有尺度特征，恢复到原始 d_model 总维度
        s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        return s_out