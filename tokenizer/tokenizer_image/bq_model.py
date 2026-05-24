from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    bits_dim: int = 14
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    patch_size: int = 16
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])

class ZCAWhitening(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.sum_x = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.sum_xxT = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)
        self.count = 0
        self.whitening_matrix = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def update_stats(self, x):
        x = x.view(-1, self.dim)
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += torch.mm(x.t(), x)
        self.count += x.size(0)

    def compute_whitening_matrix(self):
        self.mean.copy_(self.sum_x / self.count)
        cov = (self.sum_xxT / self.count) - torch.outer(self.mean, self.mean)
        cov += self.eps * torch.eye(self.dim, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
        self.whitening_matrix = torch.mm(eigenvectors, torch.diag(1.0 / sqrt_eigenvalues)) @ eigenvectors.t()

    def forward(self, x):
        x_flat = x.view(-1, self.dim)
        x_centered = x_flat - self.mean
        x_whitened = torch.mm(x_centered, self.whitening_matrix.t())
        return x_whitened.view_as(x)


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(patch_size=config.patch_size, encoding_dim=14)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim,
                                        config.codebook_l2_norm, config.codebook_show_usage)
        self.quant_conv = nn.Conv2d(14, config.codebook_embed_dim, 1)  # 输入改为14
        self.post_quant_conv = nn.Conv2d(config.bits_dim, config.z_channels, 1)

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
    def __init__(self, patch_size=16, encoding_dim=14, learnable=False):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_dim = encoding_dim
        self.conv = nn.Conv2d(3, 1, kernel_size=2, stride=2, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.norm = nn.LayerNorm(64)
        self.linear = nn.Linear(64, encoding_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.zca = ZCAWhitening(encoding_dim)
        self.learnable = learnable
        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, train_mode=False):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2).reshape(-1, C, self.patch_size, self.patch_size)
        conv_out = self.conv(patches).squeeze(1).flatten(1)
        norm_out = self.norm(conv_out)
        linear_out = self.linear(norm_out)
        if train_mode:
            self.zca.update_stats(linear_out)
        zca_out = self.zca(linear_out)
        return zca_out.view(B, -1, self.encoding_dim).permute(0, 2, 1).view(B, self.encoding_dim, H//self.patch_size, W//self.patch_size)



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

        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # z: [B, 14, H, W]
        z_q = torch.sign(z)

        # l2norm
        if self.l2_norm:
            z_q = F.normalize(z_q, p=2, dim=1)

        codebook_usage = 0
        # if self.show_usage and self.training:
        #     # 计算sign index，使用bits_dim
        #     z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)  # [B*H*W, 14]
        #     bits = (z_flattened >= 0).long()
        #     powers = 2 ** torch.arange(self.bits_dim, device=z.device)
        #     min_encoding_indices = torch.clamp((bits * powers).sum(dim=-1), 0, self.n_e - 1)
        #     cur_len = min_encoding_indices.shape[0]
        #     self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
        #     self.codebook_used[-cur_len:] = min_encoding_indices
        #     codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        return z_q, (0, 0, 0, codebook_usage), (None, None, None)


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
