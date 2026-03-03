# Embed.py

import torch
import torch.nn as nn
import math



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

    def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        d_model = d_model//4
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

    def forward(self, x):
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        return s_out