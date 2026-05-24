# tokenizer/tokenizer_image/ste_model.py
# Modified: BSQ with ICLR 2025 Gradient Rotation Trick & Random Group Entropy

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
    sample: bool = False  # 是否在训练时注入噪声

    # Common & Projection Target
    codebook_size: int = 16384  # 仅用于 legacy VQ，BSQ下忽略
    codebook_embed_dim: int = 8  # BSQ 投影后的目标维度 (Projected Output Dim)
    codebook_l2_norm: bool = True  # 是否对投影后的向量做 L2 Norm
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.0  # Sign loss weight
    entropy_loss_ratio: float = 0.1  # Balance (Entropy) loss weight
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
    计算球面 Embedding 的空间利用率 (Isotropy Score)。
    """
    B, C, H, W = z.shape
    flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)
    N = flat_z.shape[0]
    if N > num_samples:
        idx = torch.randperm(N, device=z.device)[:num_samples]
        flat_z = flat_z[idx]
    try:
        S = torch.linalg.svdvals(flat_z.float())
    except:
        return 0.0
    S_norm = S / (S.sum() + 1e-6)
    entropy = -(S_norm * torch.log(S_norm + 1e-6)).sum()
    max_entropy = math.log(C)
    score = entropy / max_entropy
    return score.item()


# ---------------------------- ICLR 2025 Rotation Trick ----------------------------
class RotationTrickSign(torch.autograd.Function):
    """
    Forward: Returns sign(z).
    Backward: Applies Householder rotation to the gradients to align e with q.
    Ref: 'Restructuring Vector Quantization with the Rotation Trick' (ICLR 2025)
    """

    @staticmethod
    def forward(ctx, z_norm):
        # z_norm is expected to be on the unit sphere (via L2 norm)
        z_q = torch.sign(z_norm)
        ctx.save_for_backward(z_norm, z_q)
        return z_q

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, C, H, W)
        e, q = ctx.saved_tensors

        # 1. Flatten and Prepare
        B, C, H, W = e.shape
        # Permute to (N, C) for easier matrix ops
        e_flat = e.permute(0, 2, 3, 1).reshape(-1, C)
        q_flat = q.permute(0, 2, 3, 1).reshape(-1, C)
        grad_flat = grad_output.permute(0, 2, 3, 1).reshape(-1, C)

        # 2. Normalize vectors for rotation calculation
        # Even though input e is normalized, we ensure safety.
        # q (sign) needs normalization to be a unit vector for the rotation logic.
        e_hat = F.normalize(e_flat, p=2, dim=1, eps=1e-6)
        q_hat = F.normalize(q_flat, p=2, dim=1, eps=1e-6)

        # 3. Householder Reflection Vector
        # r = (e + q) / ||e + q||
        middle = e_hat + q_hat
        r = F.normalize(middle, p=2, dim=1, eps=1e-6)

        # 4. Rotate Gradient
        # R = I - 2rr^T + 2q e^T (Rotation from e to q)
        # We need to rotate gradient back, effectively applying R (since R is orthogonal)
        # Formula: grad_new = grad - 2r(r^T grad) + 2e(q^T grad)

        r_dot_grad = (r * grad_flat).sum(dim=1, keepdim=True)
        q_dot_grad = (q_hat * grad_flat).sum(dim=1, keepdim=True)

        grad_rotated = grad_flat - 2 * r * r_dot_grad + 2 * e_hat * q_dot_grad

        # 5. No Scaling (Avoid Gradient Explosion)
        # We purposely skip the ||q||/||e|| scaling factor from the paper
        # because BSQ dim=64 would cause 8x gradient explosion.

        return grad_rotated.view(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------- BSQ Quantizer ----------------------------
class BSQQuantizer(nn.Module):
    """
    Binary Spherical Quantization (BSQ)
    Features:
    - RMSNorm + Gaussian Noise + L2Norm pipeline
    - Rotation Trick Gradient Estimator
    - Random Grouping Entropy Loss
    """

    def __init__(self,
                 num_bits: int,
                 embed_dim: int,
                 sign_weight: float = 1.0,
                 balance_weight: float = 1.0,
                 ema_decay: float = 0.99,
                 l2_norm: bool = True,
                 temperature: float = 10.0,
                 sample: bool = False):
        super().__init__()
        self.num_bits = num_bits
        self.embed_dim = embed_dim
        self.sign_weight = sign_weight
        self.balance_weight = balance_weight
        self.l2_norm = l2_norm
        self.temperature = temperature
        self.sample = sample  # True means inject noise during training

        # Fixed Gaussian Projection Matrix (bits -> output dim)
        self.register_buffer("proj_weight", torch.randn(embed_dim, num_bits))

        # Helper for decoding
        self.register_buffer("pow2", 2 ** torch.arange(num_bits, dtype=torch.long))

    def forward(self, z: torch.Tensor):
        """
        z: (B, num_bits, H, W)
        """
        B, C, H, W = z.shape
        assert C == self.num_bits, f"BSQ expects input dim {self.num_bits}, got {C}"

        # -----------------------------------------------------------
        # 1. RMS Norm
        # -----------------------------------------------------------
        # Calculate RMS along the channel dimension
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-6)
        z_rms = z / rms
        # Now z_rms has magnitude approx sqrt(num_bits) in Euclidean space
        # but unit RMS.

        # -----------------------------------------------------------
        # 2. Add Noise (Training Only)
        # -----------------------------------------------------------
        # Since z_rms is ~1.0 on average (element-wise),
        # Standard Gaussian Noise (mean=0, std=1) provides strong regularization (~16% flip chance)
        if self.training and self.sample:
            noise = torch.randn_like(z_rms)
            z_noisy = z_rms + noise
        else:
            z_noisy = z_rms

        # -----------------------------------------------------------
        # 3. L2 Norm (Project to Sphere)
        # -----------------------------------------------------------
        # Project onto the unit sphere before quantization
        z_sphere = F.normalize(z_noisy, p=2, dim=1)

        # -----------------------------------------------------------
        # 4. Quantization (Rotation Trick)
        # -----------------------------------------------------------
        # Forward: sign(z_sphere) -> values are -1 or +1
        # Backward: Rotation Trick
        z_q = RotationTrickSign.apply(z_sphere)

        # -----------------------------------------------------------
        # 5. Output Projection
        # -----------------------------------------------------------
        # z_q is +/- 1. We normalize it to project 1/sqrt(d)
        z_q_norm = F.normalize(z_q, p=2, dim=1)

        # Fixed Projection (num_bits -> embed_dim)
        z_perm = z_q_norm.permute(0, 2, 3, 1)
        z_proj = F.linear(z_perm, self.proj_weight)

        # Optional: Second L2 Norm (Output Spherical Embedding)
        if self.l2_norm:
            z_proj = F.normalize(z_proj, p=2, dim=-1)

        z_out = z_proj.permute(0, 3, 1, 2).contiguous()

        # -----------------------------------------------------------
        # 6. Entropy Losses (Random Grouping)
        # -----------------------------------------------------------
        bsq_sign_loss = 0.0
        bsq_balance_loss = 0.0
        usage = 1.0

        if self.training:
            # A. Calculate Probabilities (Soft)
            # Use z_sphere for cleaner gradients than z_noisy
            probs = torch.sigmoid(z_sphere * self.temperature)  # (B, C, H, W)
            eps = 1e-6

            # --- Loss 1: Sign / Conditional Entropy (Determinism) ---
            # Minimize uncertainty: - sum(p log p)
            cond_entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
            loss_cond = cond_entropy.mean()
            bsq_sign_loss = self.sign_weight * loss_cond

            # --- Loss 2: Balance / Marginal Entropy (Diversity with Random Grouping) ---
            # Goal: Maximize entropy of groups of bits to ensure uniform distribution

            # 1. Flatten spatial dims: (B, C, N) -> (B*N, C)
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, C)

            # 2. Random Permutation of Codebook Indices (Stochastic Grouping)
            perm_indices = torch.randperm(C, device=z.device)
            shuffled_probs = flat_probs[:, perm_indices]

            # 3. Grouping (Group Size = 2 for pairwise decorrelation efficiency)
            # Reshape to (Batch, Groups, 2)
            group_size = 2
            num_groups = C // group_size

            # Take mean over batch first?
            # We want the *dataset* statistic to be uniform, not necessarily every single sample.
            # So we average probs over batch first.
            avg_probs = shuffled_probs.mean(dim=0)  # (C,)

            # Reshape to groups: (Num_Groups, 2)
            # p represents prob of being 1.
            p_groups = avg_probs.view(num_groups, group_size)

            # 4. Compute Joint Distribution for each group (Size 2)
            # States: 00, 01, 10, 11
            p0 = p_groups[:, 0]
            p1 = p_groups[:, 1]

            # Probabilities of the 4 states (assuming independence within the calculated mean)
            # We want to force these to 0.25 each.
            prob_00 = (1 - p0) * (1 - p1)
            prob_01 = (1 - p0) * p1
            prob_10 = p0 * (1 - p1)
            prob_11 = p0 * p1

            # Stack: (Num_Groups, 4)
            joint_probs = torch.stack([prob_00, prob_01, prob_10, prob_11], dim=1)

            # 5. Maximize Entropy of this joint distribution
            # H = - sum(p log p)
            group_entropy = - (joint_probs * torch.log(joint_probs + eps)).sum(dim=1)  # (Num_Groups,)

            # Target is max entropy (log(4) = 2*log(2))
            # We minimize negative entropy
            loss_balance = - group_entropy.mean()

            bsq_balance_loss = self.balance_weight * loss_balance

            # Compute Usage Metric
            with torch.no_grad():
                z_check = F.normalize(z_q, p=2, dim=1)
                z_check_proj = F.linear(z_check.permute(0, 2, 3, 1), self.proj_weight)
                if self.l2_norm: z_check_proj = F.normalize(z_check_proj, p=2, dim=-1)
                usage = compute_isotropy_score(z_check_proj.permute(0, 3, 1, 2))

        return z_out, (bsq_balance_loss, bsq_sign_loss, 0.0, usage), (None, None, None)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        device = indices.device
        # Indices -> Bits
        bits_bool = (indices.unsqueeze(1) & self.pow2.unsqueeze(0)) > 0
        z_bits = torch.where(bits_bool, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

        # 1. First L2 Norm
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
            self.quantize = BSQQuantizer(
                num_bits=config.num_bits,
                embed_dim=config.codebook_embed_dim,
                sign_weight=config.commit_loss_beta,
                balance_weight=config.entropy_loss_ratio,
                ema_decay=0.99,
                l2_norm=config.codebook_l2_norm,
                sample=config.sample
            )
            self.quant_conv = nn.Conv2d(config.z_channels, config.num_bits, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        elif config.quantizer == "fsq":
            # (Assuming FSQOrig is available in the original file or imported)
            # Keeping structure compatible with your request
            from tokenizer.tokenizer_image.ste_model import FSQQuantizer as FSQOrig
            self.quantize = FSQOrig(
                per_dim_bins=config.fsq_bins, init_target=config.init_target,
                lloyd_steps=config.lloyd_steps, sphere_dim=config.sphere_dim,
                l2_norm=config.codebook_l2_norm, show_usage=config.codebook_show_usage,
                uniformity_weight=config.uniformity_weight, entropy_loss_ratio=config.entropy_loss_ratio,
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
def VQ_8(**kwargs): return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))


def VQ_16(**kwargs): return VQModel(
    ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
