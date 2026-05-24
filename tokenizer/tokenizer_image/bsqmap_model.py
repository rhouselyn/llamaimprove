# tokenizer/tokenizer_image/ste_model.py
# Modified: BSQ with Fixed Gaussian Projection & Double L2 Norm & vMF Usage

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Config ----------------------------
@dataclass
class ModelArgs:
    # BSQ Specific
    num_bits: int = 64  # 量化前的通道数 (Binary Bits)
    quantizer: str = "bsq"  # "bsq", "fsq", "vq"
    sample: bool = False

    # Common & Projection Target
    codebook_size: int = 16384  # 仅用于 legacy VQ，BSQ下忽略
    codebook_embed_dim: int = 8  # BSQ 投影后的目标维度 (Projected Output Dim)
    codebook_l2_norm: bool = True  # 是否对投影后的向量做 L2 Norm
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.0  # Sign loss weight
    entropy_loss_ratio: float = 0.1  # Balance (EMA) loss weight
    uniformity_weight: float = 1.0

    # FSQ/VQ Legacy
    fsq_bins: List[int] = field(default_factory=lambda: [2, 2, 3, 4, 4, 4, 4, 5])
    init_target: str = "gaussian"
    lloyd_steps: int = 2
    sphere_dim: int = 8

    # Backbone
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


# ---------------------------- Utils ----------------------------
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return nn.SyncBatchNorm(in_channels)


def compute_avg_min_distance(quantizer) -> float:
    # BSQ 下此指标仅作参考
    if isinstance(quantizer, BSQQuantizer):
        return 1.0
    if hasattr(quantizer, "get_codebook_vectors"):
        C = quantizer.get_codebook_vectors().float()
    else:
        C = quantizer.embedding.weight
        if getattr(quantizer, "l2_norm", False):
            C = F.normalize(C, p=2, dim=-1)
        C = C.float()
    D = torch.cdist(C, C)
    D.fill_diagonal_(float('inf'))
    return D.min(dim=1).values.mean().item()


def compute_isotropy_score(z: torch.Tensor, num_samples: int = 2048) -> float:
    """
    计算球面 Embedding 的空间利用率 (Isotropy Score / Effective Rank)。
    基于奇异值的熵 (vMF Density estimation approach)。

    Args:
        z: (B, C, H, W) Output embedding
        num_samples: 为了计算速度，随机采样的样本数
    Returns:
        score: 0.0 (Collapse) -> 1.0 (Uniform Isotropic)
    """
    B, C, H, W = z.shape
    # 展平为 [N, C]
    flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

    # 随机采样以加速计算 (SVD on 2048x8 is instantaneous)
    N = flat_z.shape[0]
    if N > num_samples:
        # 使用随机索引采样
        idx = torch.randperm(N, device=z.device)[:num_samples]
        flat_z = flat_z[idx]

    # 计算奇异值 (Singular Values)
    # 输入已经是 L2 Normalized 的，直接计算原点分布的奇异值
    try:
        S = torch.linalg.svdvals(flat_z.float())
    except:
        return 0.0  # 极少数情况 SVD 不收敛

    # 归一化奇异值，使其和为1 (类似概率分布)
    S_norm = S / (S.sum() + 1e-6)

    # 计算香农熵 (Shannon Entropy)
    entropy = -(S_norm * torch.log(S_norm + 1e-6)).sum()

    # 归一化熵值到 [0, 1]
    # 最大熵发生在所有奇异值相等时 (Isotropic), max_entropy = log(min(N, C)) -> log(C)
    max_entropy = math.log(C)

    score = entropy / max_entropy
    return score.item()


# ---------------------------- BSQ Quantizer ----------------------------
class BSQQuantizer(nn.Module):
    """
    Binary Spherical Quantization (BSQ) with Entropy Loss & EMA.

    Corrected Flow:
    1. L2 Norm (Project to Sphere)
    2. Soft Quantization (for Entropy Loss) & Hard Quantization (Sign)
    3. Entropy Losses (Conditional + Marginal with EMA)
    4. Projection to Target Dimension
    """

    def __init__(self,
                 num_bits: int,
                 embed_dim: int,
                 sign_weight: float = 1.0,  # 对应 Conditional Entropy (确定性)
                 balance_weight: float = 1.0,  # 对应 Marginal Entropy (多样性/最大熵)
                 ema_decay: float = 0.99,
                 l2_norm: bool = True,
                 temperature: float = 10.0,
                 sample: bool = False):  # 新增: 控制软量化的陡峭程度
        super().__init__()
        self.num_bits = num_bits
        self.embed_dim = embed_dim
        self.sign_weight = sign_weight
        self.balance_weight = balance_weight
        self.ema_decay = ema_decay
        self.l2_norm = l2_norm
        self.temperature = temperature  # 建议设为 sqrt(num_bits) 或更高，如 10.0
        self.sample = sample

        # Fixed Gaussian Projection Matrix
        self.register_buffer("proj_weight", torch.randn(embed_dim, num_bits))

        # EMA buffer for global probability tracking (initialized to 0.5)
        # 记录每个 bit 为 1 的概率，初始假设均匀分布
        self.register_buffer("running_prob", torch.ones(num_bits) * 0.5)

        # Helper for decoding
        self.register_buffer("pow2", 2 ** torch.arange(num_bits, dtype=torch.long))

    def forward(self, z: torch.Tensor):
        """
        z: (B, num_bits, H, W)
        """
        B, C, H, W = z.shape
        assert C == self.num_bits, f"BSQ expects input dim {self.num_bits}, got {C}"

        # -----------------------------------------------------------
        # 1. Spherical Normalization (Corrected Position)
        # -----------------------------------------------------------
        # 论文 4.1: "project v onto the unit sphere u = v / |v|"
        # 先进行归一化，再进行量化，确保是在球面上切分
        z_norm = F.normalize(z, p=2, dim=1)

        # -----------------------------------------------------------
        # 2. Quantization (Hard & Soft)
        # -----------------------------------------------------------
        # (A) Hard Quantization: u_hat = sign(u)
        z_sign = torch.sign(z_norm)
        # STE: Forward pass uses sign, Backward pass uses z_norm gradients
        z_q = z_norm + (z_sign - z_norm).detach()

        if self.training and self.sample:
            z_q = z_q + torch.randn_like(z_q)

        # -----------------------------------------------------------
        # 3. Entropy Losses (Paper Eq. 7, 8, 9)
        # -----------------------------------------------------------
        bsq_sign_loss = 0.0  # 对应 Conditional Entropy
        bsq_balance_loss = 0.0  # 对应 Marginal Entropy
        usage = 1.0

        if self.training:
            # Soft probabilities: q(c|u) ~ Sigmoid(u * temp)
            # z_norm 范围 [-1, 1], 乘 temp 增加陡峭度以便计算熵
            probs = torch.sigmoid(z_norm * self.temperature)
            eps = 1e-6

            # Loss Term 1: Conditional Entropy (Determinism)
            # 目标：最小化每个样本的不确定性 (让 probs 接近 0 或 1)
            # H(q(c|u)) = - sum(p log p)
            cond_entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
            loss_cond = cond_entropy.sum(dim=1).mean()  # Sum over bits, Mean over batch

            bsq_sign_loss = self.sign_weight * loss_cond

            # Loss Term 2: Marginal Entropy with EMA (Diversity)
            # 目标：最大化全局使用的熵 (让平均概率接近 0.5)

            # (a) 计算当前 Batch 的每个 bit 的平均概率
            batch_mean_prob = probs.mean(dim=[0, 2, 3])  # (num_bits,)

            # (b) EMA 更新逻辑 (关键点：保留梯度)
            # 我们构造一个 "proxy" 变量用于计算 Loss，它混合了历史均值和当前均值
            # 这样 Loss 对 current_batch_mean 有梯度，进而优化 z
            prob_ema = self.running_prob.detach() * self.ema_decay + \
                       batch_mean_prob * (1 - self.ema_decay)

            # (c) 更新实际 Buffer (不通过梯度)
            with torch.no_grad():
                self.running_prob.copy_(prob_ema)

            # (d) 计算边缘分布的熵并最大化 (即最小化负熵)
            marg_entropy = - (prob_ema * torch.log(prob_ema + eps) + (1 - prob_ema) * torch.log(1 - prob_ema + eps))
            loss_marg = - marg_entropy.sum()  # Negative sign to maximize entropy

            bsq_balance_loss = self.balance_weight * loss_marg

            # 计算 Usage 指标 (Isotropy)
            with torch.no_grad():
                # 检查最终输出空间的均匀度
                z_check = F.normalize(z_q, p=2, dim=1)  # 1/sqrt(L) * sign
                z_check_proj = F.linear(z_check.permute(0, 2, 3, 1), self.proj_weight)
                if self.l2_norm: z_check_proj = F.normalize(z_check_proj, p=2, dim=-1)
                usage = compute_isotropy_score(z_check_proj.permute(0, 3, 1, 2))

        # -----------------------------------------------------------
        # 4. Projections & Output
        # -----------------------------------------------------------
        # BSQ paper: u_hat = 1/sqrt(L) * sign(u)
        # z_q 当前是 +/- 1。normalize 后变成 +/- 1/sqrt(L)。
        z_q_norm = F.normalize(z_q, p=2, dim=1)

        # Fixed Projection
        z_perm = z_q_norm.permute(0, 2, 3, 1)
        z_proj = F.linear(z_perm, self.proj_weight)

        # Second L2 Norm (Output Spherical Embedding)
        if self.l2_norm:
            z_proj = F.normalize(z_proj, p=2, dim=-1)

        z_out = z_proj.permute(0, 3, 1, 2).contiguous()

        return z_out, (bsq_balance_loss, bsq_sign_loss, 0.0, usage), (None, None, None)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        device = indices.device
        # Indices -> Bits
        bits_bool = (indices.unsqueeze(1) & self.pow2.unsqueeze(0)) > 0
        z_bits = torch.where(bits_bool, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

        # 1. First L2 Norm (BSQ scaling)
        z_bits_norm = F.normalize(z_bits.float(), p=2, dim=1)

        # 2. Fixed Projection
        z_proj = F.linear(z_bits_norm, self.proj_weight)

        # 3. Second L2 Norm
        if self.l2_norm:
            z_proj = F.normalize(z_proj, p=2, dim=-1)

        if shape is not None:
            if channel_first:
                return z_proj.view(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).contiguous()
            else:
                return z_proj.view(shape)
        return z_proj


# ---------------------------- Encoder / Decoder (Standard) ----------------------------
# 保持不变
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = self.norm1(x);
        h = nonlinearity(h);
        h = self.conv1(h)
        h = self.norm2(h);
        h = nonlinearity(h);
        h = self.dropout(h);
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h);
        k = self.k(h);
        v = self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)
        k = k.reshape(b, c, hh * ww)
        w = torch.bmm(q, k) * (c ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(b, c, hh * ww)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv: self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, 2, 2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv: self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv: x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)
        self.conv_blocks = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block_out = ch * ch_mult[i_level]
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block = nn.Module()
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
            AttnBlock(block_in, norm_type=norm_type),
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
        ])
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0: h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1: h = block.downsample(h)
        for m in self.mid: h = m(h)
        h = self.norm_out(h);
        h = nonlinearity(h);
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type="group", dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
            AttnBlock(block_in, norm_type=norm_type),
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
        ])
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block = nn.Module()
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != 0: conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        h = self.conv_in(z)
        for m in self.mid: h = m(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0: h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1: h = block.upsample(h)
        h = self.norm_out(h);
        h = nonlinearity(h);
        h = self.conv_out(h)
        return h


# ---------------------------- VQModel wrapper ----------------------------
class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        if config.quantizer == "bsq":
            # BSQ with Fixed Gaussian Projection
            self.quantize = BSQQuantizer(
                num_bits=config.num_bits,
                embed_dim=config.codebook_embed_dim,
                sign_weight=config.commit_loss_beta,
                balance_weight=config.entropy_loss_ratio,
                ema_decay=0.99,
                l2_norm=config.codebook_l2_norm,
                sample=config.sample
            )
            # Quant conv maps 256 -> num_bits (14)
            self.quant_conv = nn.Conv2d(config.z_channels, config.num_bits, 1)
            # Post quant conv maps embed_dim (8) -> 256
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        elif config.quantizer == "fsq":
            # Placeholder for FSQ
            from tokenizer.tokenizer_image.ste_model import FSQQuantizer as FSQOrig
            self.quantize = FSQOrig(
                per_dim_bins=config.fsq_bins,
                init_target=config.init_target,
                lloyd_steps=config.lloyd_steps,
                sphere_dim=config.sphere_dim,
                l2_norm=config.codebook_l2_norm,
                show_usage=config.codebook_show_usage,
                uniformity_weight=config.uniformity_weight,
                entropy_loss_ratio=config.entropy_loss_ratio,
                beta=config.commit_loss_beta,
            )
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        else:
            # Legacy VQ
            from tokenizer.tokenizer_image.ste_model import VectorQuantizer as VQOrig
            self.quantize = VQOrig(config.codebook_size, config.codebook_embed_dim,
                                   config.commit_loss_beta, config.entropy_loss_ratio,
                                   config.codebook_l2_norm, config.codebook_show_usage)
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        q, codebook_loss, info = self.quantize(h)
        return q, codebook_loss, info

    def decode(self, q):
        q = self.post_quant_conv(q)
        return self.decoder(q)

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        return self.decode(quant_b)

    def forward(self, x):
        q, codebook_loss, _ = self.encode(x)
        dec = self.decode(q)
        return dec, codebook_loss


# Factory
def VQ_8(**kwargs):  return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))


def VQ_16(**kwargs): return VQModel(
    ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
