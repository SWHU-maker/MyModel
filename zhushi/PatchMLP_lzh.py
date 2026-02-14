import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import Emb


# =============================================================================
# 1. 移动平均块 (moving_avg): 像一把“平滑尺”，把数据的细节抹平，只看大趋势
# =============================================================================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 使用平均池化操作：在窗口内取平均值。kernel_size通常是奇数（如13、25）
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 输入 x 形状: [Batch, Dim, Time] -> [32, 21, 96]
        
        # 填充开头：把序列第一个点复制并贴在最前面，防止池化后序列变短
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        # 填充末尾：把序列最后一个点复制并贴在最后面
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        
        # 拼接后的 x 形状: [32, 21, 96 + 填充长度]
        x = torch.cat([front, x, end], dim=-1)

        # 执行平均池化：计算每一个小窗口的平均值
        x = self.avg(x)
        # 返回结果形状恢复为: [32, 21, 96]
        return x


# =============================================================================
# 2. 序列分解块 (series_decomp): 数据的“手术刀”，将数据切成：趋势 + 周期
# =============================================================================
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # 内部包含一个上面定义的移动平均模块
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 1. 提取趋势 (moving_mean)：平滑后的数据
        moving_mean = self.moving_avg(x)
        # 2. 提取残差 (res)：原始数据减去趋势，剩下的就是周期性波动
        res = x - moving_mean
        # 返回：周期项, 趋势项
        return res, moving_mean


# =============================================================================
# 3. 编码器层 (Encoder): 核心大脑，负责学习不同维度间的复杂关系
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, d_model, enc_in):
        super().__init__()
        # 层归一化：让数据在每一层都保持良好的分布，防止训练炸掉
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ff1 (Feed Forward 1): 处理隐藏空间 (d_model，通常是1024)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model), # 线性层
            nn.GELU(),                   # 高级的激活函数，比ReLU更好用
            nn.Dropout(0.1)              # 随机丢弃10%的神经元，防止模型“死记硬背”
        )

        # ff2 (Feed Forward 2): 处理原始特征维度 (enc_in，你的weather是21)
        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 输入 x 形状: [32, 21, 1024]
        
        # 第一步：学习隐藏层内部的关系
        y_0 = self.ff1(x)              # 计算变换
        y_0 = y_0 + x                  # 残差连接：把原始信息加回来，防止深层网络退化
        y_0 = self.norm1(y_0)          # 归一化
        
        # 第二步：学习不同物理量（变量）之间的关系
        # 我们需要把维度转过来：[32, 21, 1024] -> [32, 1024, 21]
        y_1 = y_0.permute(0, 2, 1)     
        y_1 = self.ff2(y_1)            # 此时线性层处理的是那21个变量
        # 变回原本形状：[32, 1024, 21] -> [32, 21, 1024]
        y_1 = y_1.permute(0, 2, 1)     
        
        # 第三步：混合信息
        y_2 = y_1 * y_0 + x            # 门控机制：让模型决定哪些信息重要
        y_2 = self.norm1(y_2)

        return y_2


# =============================================================================
# 4. 主模型 (Model): 整个流程的指挥官
# =============================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 保存各种超参数
        self.seq_len = configs.seq_len      # 输入长度 (如96)
        self.pred_len = configs.pred_len    # 预测长度 (如96)
        self.use_norm = configs.use_norm    # 是否使用内部归一化开关

        # 分解模块：窗口设为13
        self.decompsition = series_decomp(13)
        # 嵌入层：把时间信息转换成高维向量 (升维：96 -> 1024)
        self.emb = Emb(configs.seq_len, configs.d_model)
        
        # 季节项处理层：多层计算
        self.seasonal_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])
        # 趋势项处理层：多层计算
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])

        # 投影层：收尾工作，把 1024 映射到预测的 96 步
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        # 1. 内部归一化：让模型处理非平稳数据更鲁棒
        if self.use_norm:
            # 计算输入 x_enc [32, 96, 21] 的均值和标准差
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        # 2. 变换维度送入 Embedding
        # [32, 96, 21] -> [32, 21, 96]
        x = x_enc.permute(0, 2, 1) 
        # 升维：[32, 21, 96] -> [32, 21, 1024]
        x = self.emb(x)

        # 3. 分解：拆出周期项和趋势项
        # seasonal_init, trend_init 形状均为 [32, 21, 1024]
        seasonal_init, trend_init = self.decompsition(x)

        # 4. 分别通过各自的“加工车间”
        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)
        
        # 5. 合并并输出
        x = seasonal_init + trend_init       # 合并两者
        # 投影：[32, 21, 1024] -> [32, 21, 96]
        dec_out = self.projector(x)
        # 转置回标准形状：[32, 21, 96] -> [32, 96, 21]
        dec_out = dec_out.permute(0, 2, 1)
        
        # 6. 反归一化：把数据还原回原始的物理量级（如把 0.1 变回 15.5度）
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 推理入口
        # x_enc: 历史数据 [32, 96, 21]
        dec_out = self.forecast(x_enc)
        # 只返回最后我们要的预测长度段
        return dec_out[:, -self.pred_len:, :]