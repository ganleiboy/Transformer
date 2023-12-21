# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 实现DeiT。蒸馏学习的话，student模型的精度是可能超过teacher模型的。
# refer:https://blog.csdn.net/m0_37046057/article/details/125883742
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
from torchsummary import summary  # https://blog.csdn.net/m0_70813473/article/details/130204898


class PatchEmbedding(nn.Module):
    # 整个过程的输出和Transformer中的词嵌入是相同的，都是(bts, words_num, embed_dim)
    def __init__(self, image_size=224, patch_size=16, in_channel=3, embed_dim=768, dropout=0.) -> None:
        """
        # embed_dim是词嵌入维度，可以理解为使用的卷积核的组数
        # in_channel是输入图像的通道数
        """
        super().__init__()
        # patch嵌入，可学习
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=in_channel,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        # 位置嵌入，可学习. Parameter和Tensor的区别，https://blog.csdn.net/hxxjxw/article/details/107904012
        self.position_embedding = nn.Parameter(torch.Tensor(1, num_patches+2, embed_dim), requires_grad=True)
        nn.init.normal_(self.position_embedding, mean=0., std=0.02)
        
        # 分类token，可学习
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)

        # 蒸馏学习token，可学习
        self.distill_token = nn.Parameter(torch.Tensor(1, embed_dim), requires_grad=True)
        nn.init.normal_(self.distill_token, mean=0., std=0.02)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # x.shape=(bts, c, h, w), [2, 3, 224, 224]
        # 根据当前batchsize的维度创建分类token
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)  # 扩展为batchsize, [2, 1, 768]
        distill_tokens = self.distill_token.repeat(x.shape[0], 1, 1)  # 扩展为batchsize, [2, 1, 768]
        # 图像patch嵌入
        x = self.patch_embedding(x)  # [2, 768, 14, 14]
        x = torch.flatten(x, 2)  # x.shape=(bts, embed_dim, h'*w')
        x = x.permute(0, 2, 1)   # x.shape=(bts, h'*w', embed_dim), torch.Size([2, 196, 768])
        # 增加分类token和蒸馏学习token
        x = torch.cat((cls_tokens, distill_tokens, x), dim=1)  # 将分类和蒸馏学习token增加到最前面, [2, 198, 768]
        # 位置嵌入
        embeddings = x + self.position_embedding
        # dropout
        embeddings = self.dropout(embeddings)
        return embeddings


class FFN(nn.Module):
    # FFN是Encoder中的Feed Forward Network模块，输入输出尺寸不变
    def __init__(self, embed_dim, ffn_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*ffn_ratio))
        self.fc2 = nn.Linear(int(embed_dim*ffn_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.fc1(x)  # 先升维提取特征
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    # 多头自注意力机制
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
        # atten_weight = attn  # 不同patch之间的attention score，[B, num_heads, N, N], [2, 4, 16, 16]
        # dropout
        attn = self.attention_dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim], [2, 4, 16, 24]
        out = out.transpose(2, 1)  # [B, N, num_heads, head_dim], [2, 16, 4, 24]
        out = out.reshape([B, N, -1])  # [2, 16, 96], 隐式地对不同head的结果进行了concat

        # proj
        out = self.proj(out)  # 对不同head的结果进行特征融合
        out = self.dropout(out)
        
        return out

    def transpose_multi_head(self, x):
        # input：[B, N, all_head_dim]
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_shape)  # [B, N, num_heads, head_dim], [2, 16, 4, 24]
        x = x.transpose(2, 1)  # [B, num_heads, N, head_dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, qkv_bias=True, ffn_ratio=4.0, dropout=0., attention_dropout=0.):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias)
        self.ffn = FFN(embed_dim, ffn_ratio)  # 两个全连接占了计算量的大头！
        self.norm = nn.LayerNorm(embed_dim)  # 将对预期具有该特定大小的最后一个维度进行归一化
    
    def forward(self, x):
        h = x
        x = self.norm(x)  # 很多论文都说是norm前置效果更好一些
        x = self.attn(x)
        x += h

        h = x
        x = self.norm(x)
        x = self.ffn(x)
        x += h
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = [EncoderLayer(embed_dim) for i in range(depth)]
        self.encoders = nn.Sequential(*layer_list)  # encoder
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoders(x)
        x = self.norm(x)
        return x[:, 0, :], x[:, 1, :]  # 两个输出


class Head(nn.Module):
    def __init__(self, embed_dim, class_num):
        super().__init__()
        self.linear = nn.Linear(embed_dim, class_num)  # head
        self.softmax = nn.Softmax(dim=1)  # dim=1，按行计算
    
    def forward(self, x):
        return self.softmax(self.linear(x))


class Deit(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 ffn_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, dropout)
        self.encoder = Encoder(embed_dim, depth)
        self.head = Head(embed_dim, num_classes)
        self.head_distill = Head(embed_dim, num_classes)


    def forward(self, x):
        # input x: [2, 3, 224, 224]
        # 嵌入
        x = self.patch_embedding(x)  # [2, 198, 768]
        # encoder
        x, x_distill = self.encoder(x)  # [1, 768]，[1, 768], embed_dim类似于CNN中的channel
        # 分类
        x = self.head(x)
        x_distill = self.head_distill(x_distill)
        # 训练时是蒸馏学习，需要有两个输出分别计算loss；推理时二者1:1加权输出
        if self.training:
            return x, x_distill
        else:
            return (x + x_distill) / 2
        # x = self.classifier(x[:, 0, :])  # [2, 1000], 只使用第一个分类token进入分类，其他token不管
        # return x


if __name__ == "__main__":
    input = torch.randint(0, 255, [4, 3, 224, 224]).float()  # (n,c,h,w)，彩图
    model = Deit()
    summary(model, (3, 224, 224), batch_size=-1, device="cpu")  # 查看网络每一层的输出尺寸
    out = model(input)  # [2, 1000]
    print(out)
    print(out[0].shape)
    print("end.")
