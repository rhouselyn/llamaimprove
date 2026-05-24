# tokenizer/tokenizer_image/bsq_model.py
# Modified: Standard BSQ + History Buffer for Usage Calculation

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
    num_bits: int = 64  # BSQ 的维度，同时也是量化后的 Codebook 维度
    quantizer: str = "bsq"  # "bsq", "fsq", "vq"
    sample: bool = False

    # Common
    codebook_size: int = 16384  # Legacy VQ
    codebook_embed_dim: int = 8  # BSQ下忽略
    codebook_l2_norm: bool = True  # BSQ 输出是否保持在单位球面上
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.0
    entropy_loss_ratio: float = 0.1
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
        return nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return nn.SyncBatchNorm(in_channels)


def compute_avg_min_distance(quantizer) -> float:
    """
    兼容 train.py 调用的接口。BSQ 返回 0.0。
    """
    if isinstance(quantizer, BSQQuantizer):
        return 0.0

    if hasattr(quantizer, "get_codebook_vectors"):
        C = quantizer.get_codebook_vectors().float()
    elif hasattr(quantizer, "embedding"):
        C = quantizer.embedding.weight
        if getattr(quantizer, "l2_norm", False):
            C = F.normalize(C, p=2, dim=-1)
        C = C.float()
    else:
        return 0.0

    D = torch.cdist(C, C)
    D.fill_diagonal_(float('inf'))
    return D.min(dim=1).values.mean().item()


# ---------------------------- BSQ Quantizer (Modified) ----------------------------
class BSQQuantizer(nn.Module):
    """
    Standard Binary Spherical Quantization (BSQ).
    Features:
    - No Projection: Output dimension == num_bits.
    - History Buffer: Tracks usage over a sliding window of samples.
    """

    def __init__(self,
                 num_bits: int,
                 sign_weight: float = 1.0,
                 balance_weight: float = 1.0,
                 ema_decay: float = 0.99,
                 l2_norm: bool = True,
                 temperature: float = 10.0,
                 usage_buffer_size: int = 65536,
                 sample: bool = False):  # 2^16
        super().__init__()
        self.num_bits = num_bits
        self.sign_weight = sign_weight
        self.balance_weight = balance_weight
        self.ema_decay = ema_decay
        self.l2_norm = l2_norm
        self.temperature = temperature
        self.sample = sample  # <--- [新增] 保存参数

        # EMA buffer for global probability tracking
        self.register_buffer("running_prob", torch.ones(num_bits) * 0.5)
        # Powers of 2 for binary -> decimal conversion
        self.register_buffer("pow2", 2 ** torch.arange(num_bits, dtype=torch.long))

        # --- Usage Tracking Buffer ---
        # 存储最近 usage_buffer_size 个 Patch 的十进制 Code
        self.register_buffer("code_usage_buffer", torch.zeros(usage_buffer_size, dtype=torch.long))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.usage_buffer_size = usage_buffer_size

    @torch.no_grad()
    def _update_usage_buffer(self, code_dec: torch.Tensor):
        """
        将新的 codes 更新到环形 buffer 中
        """
        batch_codes = code_dec.view(-1)
        num_new = batch_codes.numel()

        # 如果新数据比 Buffer 还大，直接取最后一部分覆盖整个 Buffer
        if num_new >= self.usage_buffer_size:
            self.code_usage_buffer.copy_(batch_codes[-self.usage_buffer_size:])
            self.buffer_ptr.fill_(0)
            return

        ptr = self.buffer_ptr.item()
        remaining_space = self.usage_buffer_size - ptr

        if num_new <= remaining_space:
            # 可以直接放下
            self.code_usage_buffer[ptr: ptr + num_new] = batch_codes
            self.buffer_ptr.fill_((ptr + num_new) % self.usage_buffer_size)
        else:
            # 需要回绕 (Wrap around)
            # 1. 填满尾部
            self.code_usage_buffer[ptr:] = batch_codes[:remaining_space]
            # 2. 填头部
            self.code_usage_buffer[:num_new - remaining_space] = batch_codes[remaining_space:]
            self.buffer_ptr.fill_(num_new - remaining_space)

    def forward(self, z: torch.Tensor):
        """
        z: (B, num_bits, H, W)
        """
        B, C, H, W = z.shape
        assert C == self.num_bits, f"BSQ expects input dim {self.num_bits}, got {C}"

        # 1. Spherical Normalization (Input)
        z_norm = F.normalize(z, p=2, dim=1)

        # 2. Quantization (Hard & Soft)
        z_sign = torch.sign(z_norm)
        # STE
        z_q = z_norm + (z_sign - z_norm).detach()

        if self.training and self.sample:
            z_q = z_q + torch.randn_like(z_q)

        # 3. Entropy Losses
        bsq_sign_loss = 0.0
        bsq_balance_loss = 0.0
        usage = 0.0

        if self.training:
            # Soft probabilities
            probs = torch.sigmoid(z_norm * self.temperature)
            eps = 1e-6

            # Conditional Entropy
            cond_entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
            loss_cond = cond_entropy.sum(dim=1).mean()
            bsq_sign_loss = self.sign_weight * loss_cond

            # Marginal Entropy with EMA
            batch_mean_prob = probs.mean(dim=[0, 2, 3])
            prob_ema = self.running_prob.detach() * self.ema_decay + \
                       batch_mean_prob * (1 - self.ema_decay)
            with torch.no_grad():
                self.running_prob.copy_(prob_ema)

            marg_entropy = - (prob_ema * torch.log(prob_ema + eps) + (1 - prob_ema) * torch.log(1 - prob_ema + eps))
            loss_marg = - marg_entropy.sum()
            bsq_balance_loss = self.balance_weight * loss_marg

            # 4. Usage Calculation (History Buffer)
            with torch.no_grad():
                # (B, C, H, W) -> (B, H, W, C) -> Flatten
                flat_z = z_sign.permute(0, 2, 3, 1).reshape(-1, C)
                # Binary -> Decimal
                code_bin = (flat_z > 0).long()
                code_dec = (code_bin * self.pow2).sum(dim=-1)

                # Update buffer
                self._update_usage_buffer(code_dec)

                # Compute usage on the full buffer
                unique_count = torch.unique(self.code_usage_buffer).numel()
                total_possible = 2 ** self.num_bits
                usage = unique_count / float(total_possible)

        # 5. Output Preparation (No Projection)
        z_out = F.normalize(z_q, p=2, dim=1)

        return z_out, (bsq_balance_loss, bsq_sign_loss, 0.0, usage), (None, None, None)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        device = indices.device
        # Indices -> Bits
        # indices: integer codes
        bits_bool = (indices.unsqueeze(1) & self.pow2.unsqueeze(0)) > 0
        z_bits = torch.where(bits_bool, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

        # Normalize (Standard BSQ output)
        z_out = F.normalize(z_bits.float(), p=2, dim=1)

        if shape is not None:
            if channel_first:
                return z_out.view(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).contiguous()
            else:
                return z_out.view(shape)
        return z_out


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


# ---------------------------- VQModel wrapper (Modified) ----------------------------
class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        if config.quantizer == "bsq":
            # Standard BSQ (No Projection)
            self.quantize = BSQQuantizer(
                num_bits=config.num_bits,
                sign_weight=config.commit_loss_beta,
                balance_weight=config.entropy_loss_ratio,
                ema_decay=0.99,
                l2_norm=config.codebook_l2_norm,
                usage_buffer_size=65536,  # Default 2^16
                sample=config.sample  # <--- [新增] 传递配置
            )
            # Quant conv maps: 256 -> num_bits
            self.quant_conv = nn.Conv2d(config.z_channels, config.num_bits, 1)

            # Post quant conv maps: num_bits -> 256
            self.post_quant_conv = nn.Conv2d(config.num_bits, config.z_channels, 1)

        elif config.quantizer == "fsq":
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
