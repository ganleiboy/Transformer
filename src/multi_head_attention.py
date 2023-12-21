import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim和num_heads不能整除！"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads  # 需要确保二者能整除
        self.all_head_dim = self.head_dim * self.num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, _ = x.shape  # [batch_size, num_patches, embed_dim], [2, 16, 96]
        qkv = self.qkv(x).chunk(3, -1)  # tuple, [B, N, all_head_dim]*3
        q,k,v = map(self.transpose_multi_head, qkv)  # [B, num_heads, N, head_dim], [2, 4, 16, 24]

        attn = torch.matmul(q, k.transpose(-2, -1))  # q * k', [B, num_heads, N, N], [2, 4, 16, 16]
        attn = self.scale * attn
        attn = self.softmax(attn)
        atten_weight = attn  # 不同patch之间的attention score，[B, num_heads, N, N], [2, 4, 16, 16]
        # dropout
        attn = self.attention_dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim], [2, 4, 16, 24]
        out = out.transpose(2, 1)  # [B, N, num_heads, head_dim], [2, 16, 4, 24]
        out = out.reshape([B, N, -1])  # [2, 16, 96], 隐式地对不同head的结果进行了concat

        # proj
        out = self.proj(out)  # 对不同head的结果进行特征融合
        out = self.dropout(out)
        
        return out, atten_weight

    def transpose_multi_head(self, x):
        # input：[B, N, all_head_dim]
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_shape)  # [B, N, num_heads, head_dim], [2, 16, 4, 24]
        x = x.transpose(2, 1)  # [B, num_heads, N, head_dim]
        return x


if __name__=="__main__":
    x = torch.randn([2, 16, 96])  # [batch_size, num_patches, embed_dim]
    mha = MultiHeadAttention(embed_dim=96, num_heads=4)
    out = mha(x)
    print(out.shape)
