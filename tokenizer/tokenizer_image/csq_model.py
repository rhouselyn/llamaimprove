from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True

    patch_size: int = 16  # 新增，从第二个脚本

    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


# Custom ZCA Whitening Module with accumulative stats
class ZCAWhitening(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer('sum_x', torch.zeros(dim))  # S: 数据总和
        self.register_buffer('sum_xxT', torch.zeros(dim, dim))  # Q: 外积总和
        self.register_buffer('whitening_matrix', torch.eye(dim))  # 初始单位矩阵
        self.register_buffer('mean', torch.zeros(dim))
        self.count = 0

    def update_stats(self, x):
        # x: (N, dim)
        batch_size = x.size(0)
        # 累加数据总和
        self.sum_x += x.sum(dim=0)
        # 累加外积总和
        self.sum_xxT += torch.mm(x.t(), x)
        # 更新样本数
        self.count += batch_size

    def compute_whitening_matrix(self):
        if self.count == 0:
            raise ValueError("No data accumulated for ZCA computation.")

        # 计算全局均值
        mean = self.sum_x / self.count
        # 计算全局协方差
        cov = (self.sum_xxT / self.count) - torch.outer(mean, mean)
        # SVD 分解
        U, S, _ = torch.svd(cov, some=False)
        # 正则化特征值
        S = S + self.eps
        # ZCA 白化矩阵
        whitening = torch.mm(U, torch.mm(torch.diag(1.0 / torch.sqrt(S)), U.t()))
        self.whitening_matrix = whitening
        # 保存均值用于推理
        self.mean.copy_(mean)

    def forward(self, x, train_mode=False):
        # x: (N, dim)
        if train_mode:
            self.update_stats(x)
            return x  # 第一阶段仅更新统计量
        else:
            # 去均值并白化
            x = x - self.mean
            x = torch.mm(x, self.whitening_matrix.t())
            return x


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(patch_size=config.patch_size, encoding_dim=14)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim,
                                        config.codebook_l2_norm, config.codebook_show_usage)
        self.quant_conv = nn.Conv2d(14, config.codebook_embed_dim, 1)  # 输入改为14
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        # 固定Encoder，不更新
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, x):
        h = self.encoder(x)
        # 不使用quant_conv，因为后续直接转index
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff  # 无diff/loss


# Modified Encoder with Conv + Flatten + Norm + Linear + ZCA (based on PatchMLPEncoder architecture, no sign)
class Encoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, encoding_dim=14, learnable=False):
        super().__init__()
        # Set seed for reproducible random initialization
        torch.manual_seed(64)

        self.patch_size = patch_size
        self.encoding_dim = encoding_dim
        self.learnable = learnable

        # Conv layer: Conv2d(3, 1, kernel_size=(2,2), stride=(2,2)) for each patch
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=(2,2), stride=(2,2), bias=False)

        # LayerNorm after flatten (on 64 dim)
        self.norm = nn.LayerNorm(64)

        # Linear layer: input 64 (1*8*8), output encoding_dim=14
        self.linear = nn.Linear(64, encoding_dim, bias=False)

        # ZCA Whitening with accumulative stats (unchanged)
        self.zca = ZCAWhitening(encoding_dim)

        # Initialize weights
        with torch.no_grad():
            # Conv: constant init (1.0/12)
            nn.init.xavier_uniform_(self.conv.weight)
            # Linear: xavier uniform init
            nn.init.xavier_uniform_(self.linear.weight)

        # Set conv fixed, linear learnable based on param
        self.conv.weight.requires_grad = False
        self.linear.weight.requires_grad = self.learnable

    def forward(self, x, train_mode=False):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size."

        # Unfold to patches: (B, C * patch_size^2, num_patches)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        # Reshape to (B * num_patches, C, patch_size, patch_size) for conv processing
        num_patches = patches.size(2)
        h = patches.view(B, C, self.patch_size, self.patch_size, num_patches)
        h = h.permute(0, 4, 1, 2, 3).reshape(-1, C, self.patch_size, self.patch_size)  # (B*num_patches, 3, 16, 16)

        # Apply conv: (B*num_patches, 1, 8, 8)
        h = self.conv(h)

        # Flatten to (B*num_patches, 64)
        h = h.flatten(1)

        # Apply LayerNorm
        h = self.norm(h)

        # Apply linear: to (B*num_patches, 14)
        h = self.linear(h)

        # Apply ZCA (with train_mode for accumulation)
        h = self.zca(h, train_mode=train_mode)

        # Reshape back
        h = h.view(B, H // self.patch_size, W // self.patch_size, self.encoding_dim)
        h = h.permute(0, 3, 1, 2).contiguous()
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type='group', dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # upsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # z: [B, 14, H, W]
        z = z.detach()  # 确保输入无梯度
        codebook_usage = 0
        with torch.no_grad():
            # 二进制 bits: (z >= 0).long() -> 0 or 1
            bits = (z >= 0).long()  # [B, 14, H, W]

            # 二进制转十进制 index（向量化，无循环）
            B, _, H, W = bits.shape
            bits = bits.permute(0, 2, 3, 1).contiguous().view(B * H * W, 14)  # [BHW, 14]
            powers = 2 ** torch.arange(14, device=bits.device)  # [14]
            index = torch.sum(bits * powers, dim=1)  # [BHW]
            index = index.clamp(0, self.n_e - 1).long()

        # 计算codebook usage（如果启用且训练中）
        if self.show_usage and self.training:
            cur_len = index.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = index
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # 用index查询embedding (有梯度)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[index]  # [BHW, 8]

        # reshape back to [B, 8, H, W]
        z_q = z_q.view(B, H, W, self.e_dim).permute(0, 3, 1, 2).contiguous()

        return z_q, (0, 0, 0, codebook_usage), (None, None, index)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(decoder_ch_mult=[1, 2, 2, 4], **kwargs))


def VQ_16(**kwargs):
    return VQModel(ModelArgs(**kwargs))


VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
