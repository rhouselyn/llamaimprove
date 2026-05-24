from dataclasses import dataclass, field
from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Config ----------------------------
@dataclass
class ModelArgs:
    num_bits: int = 128
    quantizer: str = "bsq"
    sample: bool = True
    codebook_embed_dim: int = 16
    codebook_l2_norm: bool = True

    # Annealing Config
    anneal_noise: bool = False
    anneal_start_epoch: int = 10
    anneal_end_epoch: int = 30
    noise_start_scale: float = 1.0
    noise_end_scale: float = 0.1

    # Projection Matrix Config
    # 当前推荐结构默认使用 learnable group-wise linear。
    learnable_proj: bool = True

    # Residual MLP Projection Config
    # 每组: group_bits -> group_embed_dim * hidden_mult -> group_embed_dim
    projector_hidden_mult: int = 4
    projector_res_scale_init: float = 1e-3

    # Backbone Config
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
    z_channels: int = 256
    num_res_blocks: int = 2
    dropout_p: float = 0.0


# ---------------------------- Utils ----------------------------
def compute_isotropy_score(z: torch.Tensor, num_samples: int = 2048) -> float:
    """计算球面 embedding 的空间利用率。"""
    B, C, H, W = z.shape
    flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)
    N = flat_z.shape[0]

    if N > num_samples:
        idx = torch.randperm(N, device=z.device)[:num_samples]
        flat_z = flat_z[idx]

    try:
        S = torch.linalg.svdvals(flat_z.float())
    except Exception:
        return 0.0

    S_norm = S / (S.sum() + 1e-6)
    entropy = -(S_norm * torch.log(S_norm + 1e-6)).sum()
    max_entropy = math.log(C)
    score = entropy / max_entropy
    return score.item()


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0, (
            f"in_channels={in_channels} must be divisible by num_heads={num_heads}"
        )

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_in = self.norm(x)
        b, c, hh, ww = h_in.shape
        n = hh * ww

        q = self.q(h_in).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = self.k(h_in).view(b, self.num_heads, self.head_dim, n)
        v = self.v(h_in).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        w = torch.matmul(q, k) * (self.head_dim ** -0.5)
        w = F.softmax(w, dim=-1)

        h_out = torch.matmul(w, v).permute(0, 1, 3, 2).reshape(b, c, hh, ww)
        h_out = self.proj_out(h_out)

        return x + h_out


class DCAE_Downsample(nn.Module):
    """
    保持原 DCAE 下采样逻辑不变：
    conv stride=2 分支 + pixel_unshuffle shortcut 分支。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv(x)

        s2c = F.pixel_unshuffle(x, 2)
        if s2c.shape[1] > h.shape[1]:
            n = s2c.shape[1] // h.shape[1]
            shortcut = sum(
                s2c[:, i * h.shape[1]:(i + 1) * h.shape[1]]
                for i in range(n)
            ) / n
        else:
            shortcut = s2c

        return h + shortcut


class DCAE_Upsample(nn.Module):
    """
    保持原 DCAE 上采样逻辑不变：
    nearest + conv 分支，cat 后 pixel_shuffle 的 shortcut 分支。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = self.conv(h)

        c2s_in = torch.cat([x] * 4, dim=1)
        shortcut = F.pixel_shuffle(c2s_in, 2)

        if shortcut.shape[1] != h.shape[1]:
            shortcut = F.interpolate(shortcut, size=h.shape[2:], mode="nearest")

        return h + shortcut


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class RotationTrickSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_norm):
        z_q = torch.sign(z_norm)
        ctx.save_for_backward(z_norm, z_q)
        return z_q

    @staticmethod
    def backward(ctx, grad_output):
        e, q = ctx.saved_tensors
        B, C, H, W = e.shape

        e_f = e.permute(0, 2, 3, 1).reshape(-1, C).float()
        q_f = q.permute(0, 2, 3, 1).reshape(-1, C).float()
        g_f = grad_output.permute(0, 2, 3, 1).reshape(-1, C).float()

        e_h = F.normalize(e_f, p=2, dim=1, eps=1e-6)
        q_h = F.normalize(q_f, p=2, dim=1, eps=1e-6)

        s = e_h + q_h
        s_norm = torch.linalg.norm(s, dim=1, keepdim=True)

        good = s_norm > 1e-3
        r = s / (s_norm + 1e-6)

        r_dot_g = (r * g_f).sum(dim=1, keepdim=True)
        q_dot_g = (q_h * g_f).sum(dim=1, keepdim=True)
        g_rot = g_f - 2.0 * r * r_dot_g + 2.0 * e_h * q_dot_g

        g_rot = torch.where(good, g_rot, g_f)

        eh_dot = (e_h * g_rot).sum(dim=1, keepdim=True)
        g_final = g_rot - e_h * eh_dot

        out = g_final.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out.to(grad_output.dtype)


# ---------------------------- Group-wise Residual Projector ----------------------------
class GroupResidualProjector(nn.Module):
    """
    推荐结构：
        每组 bits 先走 learnable linear，得到主路径；
        同一组 bits 再走一个小 residual MLP；
        两者相加后 concat 所有 group；
        最后做 global L2 norm。

    输入：
        bits_group: [N, num_groups, group_bits]

    输出：
        z: [N, codebook_embed_dim]
    """
    def __init__(
        self,
        num_groups: int,
        group_bits: int,
        group_embed_dim: int,
        l2_norm: bool = True,
        learnable_linear: bool = True,
        hidden_mult: int = 2,
        res_scale_init: float = 1e-3,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_bits = group_bits
        self.group_embed_dim = group_embed_dim
        self.embed_dim = num_groups * group_embed_dim
        self.l2_norm = l2_norm
        self.learnable_linear = learnable_linear

        hidden_dim = group_embed_dim * hidden_mult

        # 主路径：group-wise linear。
        # 权重形状保持和原版一致: [G, Eg, Bg]
        init_weight = torch.randn(num_groups, group_embed_dim, group_bits)
        init_weight = F.normalize(init_weight, p=2, dim=1)

        if learnable_linear:
            self.proj_weight = nn.Parameter(init_weight)
        else:
            self.register_buffer("proj_weight", init_weight)

        # residual 路径：每组一个小 MLP。
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(group_bits),
                nn.Linear(group_bits, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, group_embed_dim),
            )
            for _ in range(num_groups)
        ])

        # 小尺度 residual，避免一开始 MLP 破坏原 linear projection 几何。
        self.group_res_scale = nn.Parameter(
            torch.full((num_groups,), float(res_scale_init))
        )

    def forward(self, bits_group: torch.Tensor) -> torch.Tensor:
        """
        bits_group:
            [N, G, Bg]
        """
        n, g, bg = bits_group.shape
        assert g == self.num_groups, f"expected {self.num_groups} groups, got {g}"
        assert bg == self.group_bits, f"expected {self.group_bits} bits/group, got {bg}"

        # Linear 主路径: [N, G, Bg] x [G, Eg, Bg] -> [N, G, Eg]
        linear_out = torch.einsum(
            "ngb,geb->nge",
            bits_group,
            self.proj_weight,
        )

        # Residual MLP 路径。
        mlp_out = []
        for group_id, mlp in enumerate(self.group_mlps):
            mlp_out.append(mlp(bits_group[:, group_id, :]))
        mlp_out = torch.stack(mlp_out, dim=1)  # [N, G, Eg]

        res_scale = self.group_res_scale.view(1, self.num_groups, 1)
        z_group = linear_out + res_scale * mlp_out

        # Concat groups: [N, G, Eg] -> [N, embed_dim]
        z = z_group.reshape(n, self.embed_dim)

        # Global L2 norm，而不是 per-group L2 norm。
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1, eps=1e-6)

        return z


# ---------------------------- Group-wise BSQ Quantizer ----------------------------
class BSQQuantizer(nn.Module):
    """
    Group-wise BSQ + group-wise residual projector.

    原始版本：
        num_bits -> codebook_embed_dim
        例如 DCAE-32: 128 -> 32

    当前版本：
        分成 4 组，每组先做 learnable linear，再加一个小 residual MLP。
        最后 concat 所有组并做 global L2 norm。

        例如 DCAE-32:
            128 bits = 4 x 32 bits
             32 dims = 4 x 8 dims

            group 0: 32 bits -> linear 8 dims + residual MLP 8 dims
            group 1: 32 bits -> linear 8 dims + residual MLP 8 dims
            group 2: 32 bits -> linear 8 dims + residual MLP 8 dims
            group 3: 32 bits -> linear 8 dims + residual MLP 8 dims
            concat -> 32 dims -> global L2 norm
    """
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.num_bits = config.num_bits
        self.embed_dim = config.codebook_embed_dim
        self.l2_norm = config.codebook_l2_norm
        self.sample = config.sample
        self.learnable_proj = config.learnable_proj

        # 固定四组。
        self.num_groups = 4
        assert self.num_bits % self.num_groups == 0, (
            f"num_bits={self.num_bits} must be divisible by num_groups={self.num_groups}"
        )
        assert self.embed_dim % self.num_groups == 0, (
            f"codebook_embed_dim={self.embed_dim} must be divisible by num_groups={self.num_groups}"
        )

        self.group_bits = self.num_bits // self.num_groups
        self.group_embed_dim = self.embed_dim // self.num_groups

        # Annealing configs
        self.anneal_noise = config.anneal_noise
        self.anneal_start_epoch = config.anneal_start_epoch
        self.anneal_end_epoch = config.anneal_end_epoch
        self.noise_start_scale = config.noise_start_scale
        self.noise_end_scale = config.noise_end_scale
        self.current_epoch = 0

        # Group-wise learnable linear + group-wise residual MLP。
        self.projector = GroupResidualProjector(
            num_groups=self.num_groups,
            group_bits=self.group_bits,
            group_embed_dim=self.group_embed_dim,
            l2_norm=self.l2_norm,
            learnable_linear=self.learnable_proj,
            hidden_mult=config.projector_hidden_mult,
            res_scale_init=config.projector_res_scale_init,
        )

        # 用于 decode_code / get_codebook_entry。
        # packed integer 只在 group_bits <= 62 时安全；
        # DCAE-32: group_bits=32，可用。
        # DCAE-64: group_bits=128，单个 int64 放不下，需传 explicit bits。
        if self.group_bits <= 62:
            self.register_buffer(
                "pow2",
                2 ** torch.arange(self.group_bits, dtype=torch.long),
            )
        else:
            self.register_buffer("pow2", torch.empty(0, dtype=torch.long))

    @property
    def proj_weight(self):
        # 兼容旧代码里可能访问 model.quantize.proj_weight 的情况。
        return self.projector.proj_weight

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _project_groupwise_4d(self, z_bits: torch.Tensor) -> torch.Tensor:
        """
        z_bits:
            [B, num_bits, H, W]

        return:
            [B, codebook_embed_dim, H, W]
        """
        b, c, h, w = z_bits.shape
        assert c == self.num_bits, f"expected {self.num_bits} channels, got {c}"

        # [B, num_bits, H, W]
        # -> [B, G, group_bits, H, W]
        # -> [B, H, W, G, group_bits]
        z_group = z_bits.reshape(
            b,
            self.num_groups,
            self.group_bits,
            h,
            w,
        ).permute(0, 3, 4, 1, 2).contiguous()

        # [B, H, W, G, group_bits] -> [B*H*W, G, group_bits]
        z_group_flat = z_group.reshape(b * h * w, self.num_groups, self.group_bits)

        # [B*H*W, codebook_embed_dim]
        z_proj_flat = self.projector(z_group_flat)

        # [B*H*W, codebook_embed_dim] -> [B, codebook_embed_dim, H, W]
        z_out = z_proj_flat.reshape(b, h, w, self.embed_dim)
        z_out = z_out.permute(0, 3, 1, 2).contiguous()
        return z_out

    def _project_groupwise_flat(self, z_bits_flat: torch.Tensor) -> torch.Tensor:
        """
        z_bits_flat:
            [N, num_bits]

        return:
            [N, codebook_embed_dim]
        """
        n, c = z_bits_flat.shape
        assert c == self.num_bits, f"expected {self.num_bits} bits, got {c}"

        # [N, num_bits] -> [N, G, group_bits]
        z_group = z_bits_flat.reshape(n, self.num_groups, self.group_bits)

        # group-wise linear + group-wise residual MLP + global L2 norm
        z = self.projector(z_group)
        return z

    def forward(self, z):
        # z: [B, num_bits, H, W]

        # 1. RMS Norm
        rms = torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-6)
        z_rms = z / rms

        # 2. Add Noise with Optional Annealing
        if self.training and self.sample:
            noise_scale = 1.0

            if self.anneal_noise:
                if self.current_epoch < self.anneal_start_epoch:
                    noise_scale = self.noise_start_scale
                elif self.current_epoch >= self.anneal_end_epoch:
                    noise_scale = self.noise_end_scale
                else:
                    progress = (
                        self.current_epoch - self.anneal_start_epoch
                    ) / (
                        self.anneal_end_epoch - self.anneal_start_epoch
                    )
                    noise_scale = self.noise_end_scale + 0.5 * (
                        self.noise_start_scale - self.noise_end_scale
                    ) * (
                        1 + math.cos(math.pi * progress)
                    )

            noise = torch.randn_like(z_rms) * noise_scale
            z_noisy = z_rms + noise
        else:
            z_noisy = z_rms

        # 3. L2 Norm -> Sign Quantization
        z_sphere = F.normalize(z_noisy, p=2, dim=1)
        z_q = RotationTrickSign.apply(z_sphere)
        z_q_norm = F.normalize(z_q, p=2, dim=1)

        # 4. Group-wise linear + group-wise residual MLP + global L2 norm
        z_out = self._project_groupwise_4d(z_q_norm)

        # z_out = self._project_groupwise_4d(z_sphere)

        # 5. Compute Isotropy Score
        usage = 1.0
        if self.training:
            with torch.no_grad():
                usage = compute_isotropy_score(z_out)

        dummy_loss = torch.tensor(0.0, device=z.device)
        return z_out, (dummy_loss, dummy_loss, dummy_loss, usage)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        """
        支持三种 indices 输入：

        1. explicit full bits:
            indices shape: [..., num_bits]
            值可以是 {-1, +1} 或 {0, 1}

        2. explicit grouped bits:
            indices shape: [..., num_groups, group_bits]
            值可以是 {-1, +1} 或 {0, 1}

        3. packed group indices:
            indices shape: [..., num_groups]
            每组一个 int，适用于 group_bits <= 62。
            DCAE-32 下 group_bits=32，因此可用。

        原来的 single packed index:
            indices shape: [...]
            只在 num_bits <= 62 时支持。
            DCAE-32 的 num_bits=128，单个 int64 放不下，因此不建议使用。
        """
        device = indices.device

        if indices.dim() >= 1 and indices.shape[-1] == self.num_bits:
            # [..., num_bits] explicit full bits
            bits = torch.where(indices.reshape(-1, self.num_bits) > 0, 1.0, -1.0)

        elif indices.dim() >= 2 and indices.shape[-2:] == (self.num_groups, self.group_bits):
            # [..., G, group_bits] explicit grouped bits
            bits = torch.where(
                indices.reshape(-1, self.num_groups, self.group_bits) > 0,
                1.0,
                -1.0,
            ).reshape(-1, self.num_bits)

        elif indices.dim() >= 1 and indices.shape[-1] == self.num_groups:
            # [..., G] packed group indices
            if self.pow2.numel() != self.group_bits:
                raise ValueError(
                    "Packed group indices require group_bits <= 62. "
                    f"Got group_bits={self.group_bits}. "
                    "Please pass explicit bits with shape [..., num_bits] "
                    "or [..., num_groups, group_bits]."
                )

            packed = indices.reshape(-1, self.num_groups).to(torch.long)
            bits = torch.where(
                (packed.unsqueeze(-1) & self.pow2.view(1, 1, -1).to(device)) > 0,
                1.0,
                -1.0,
            ).reshape(-1, self.num_bits)

        else:
            # Original single packed index path.
            if self.num_bits > 62:
                raise ValueError(
                    "Single packed index cannot represent num_bits > 62 in int64. "
                    f"Got num_bits={self.num_bits}. "
                    "For DCAE-32 / DCAE-64, pass packed group indices with shape "
                    "[..., num_groups] or explicit bits."
                )

            pow2 = 2 ** torch.arange(self.num_bits, dtype=torch.long, device=device)
            packed = indices.reshape(-1).to(torch.long)
            bits = torch.where(
                (packed.unsqueeze(1) & pow2.unsqueeze(0)) > 0,
                1.0,
                -1.0,
            )

        bits = bits.to(device=device, dtype=torch.float32)
        bits_norm = F.normalize(bits, p=2, dim=1)

        z = self._project_groupwise_flat(bits_norm)

        if shape is not None:
            if channel_first:
                z = z.reshape(shape[0], shape[2], shape[3], shape[1])
                z = z.permute(0, 3, 1, 2).contiguous()
            else:
                z = z.view(shape)

        return z


# ---------------------------- Encoder / Decoder ----------------------------
class Encoder(nn.Module):
    """
    VQ-VAE-style Encoder stack，DCAE downsample 保持不变。

    结构：
    conv_in
    -> per-resolution:
       ResnetBlock x num_res_blocks
       只有最后一个 resolution 加 attention
       非最后层接 DCAE_Downsample
    -> mid:
       ResnetBlock
       Attention
       ResnetBlock
    -> norm_out + swish + conv_out
    """
    def __init__(self, config: ModelArgs):
        super().__init__()

        ch = 128
        self.num_resolutions = len(config.encoder_ch_mult)
        self.num_res_blocks = config.num_res_blocks

        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(config.encoder_ch_mult)
        self.conv_blocks = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()

            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * config.encoder_ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                res_block.append(
                    ResnetBlock(
                        block_in,
                        block_out,
                        dropout=config.dropout_p,
                    )
                )
                block_in = block_out

                # 硬编码：严格对齐 VQ-VAE，只在 encoder 最后一层 resolution 加 attention。
                if i_level == self.num_resolutions - 1:
                    attn_block.append(MultiHeadAttnBlock(block_in))

            conv_block.res = res_block
            conv_block.attn = attn_block

            if i_level != self.num_resolutions - 1:
                conv_block.downsample = DCAE_Downsample(block_in, block_in)

            self.conv_blocks.append(conv_block)

        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=config.dropout_p))
        self.mid.append(MultiHeadAttnBlock(block_in))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=config.dropout_p))

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)

        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)

                if len(block.attn) > 0:
                    h = block.attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        for mid_block in self.mid:
            h = mid_block(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    """
    VQ-VAE-style Decoder stack，DCAE upsample 保持不变。

    结构：
    conv_in
    -> mid:
       ResnetBlock
       Attention
       ResnetBlock
    -> reversed per-resolution:
       ResnetBlock x (num_res_blocks + 1)
       只有最低分辨率 level 加 attention
       非最后输出层接 DCAE_Upsample
    -> norm_out + swish + conv_out
    """
    def __init__(self, config: ModelArgs):
        super().__init__()

        ch = 128
        self.num_resolutions = len(config.decoder_ch_mult)
        self.num_res_blocks = config.num_res_blocks

        block_in = ch * config.decoder_ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=config.dropout_p))
        self.mid.append(MultiHeadAttnBlock(block_in))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=config.dropout_p))

        self.conv_blocks = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()

            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()

            block_out = ch * config.decoder_ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(
                        block_in,
                        block_out,
                        dropout=config.dropout_p,
                    )
                )
                block_in = block_out

                # 硬编码：严格对齐 VQ-VAE，只在 decoder 最低分辨率 level 加 attention。
                # 因为这里是 reversed(range(...))，所以 i_level == num_resolutions - 1
                # 对应 decoder 刚开始的 bottleneck level。
                if i_level == self.num_resolutions - 1:
                    attn_block.append(MultiHeadAttnBlock(block_in))

            conv_block.res = res_block
            conv_block.attn = attn_block

            if i_level != 0:
                conv_block.upsample = DCAE_Upsample(block_in, block_in)

            self.conv_blocks.append(conv_block)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        h = self.conv_in(z)

        for mid_block in self.mid:
            h = mid_block(h)

        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)

                if len(block.attn) > 0:
                    h = block.attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    @property
    def last_layer(self):
        return self.conv_out.weight


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.quantize = BSQQuantizer(config)

        self.quant_conv = nn.Conv2d(config.z_channels, config.num_bits, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, codebook_loss_tuple = self.quantize(h)
        return quant, codebook_loss_tuple

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x):
        q, codebook_loss_tuple = self.encode(x)
        dec = self.decode(q)
        return dec, codebook_loss_tuple


# ---------------------------- Factory ----------------------------
def DCAE_f32_Attn(**kwargs):
    # z_ch = kwargs.get("z_channels", 512)
    # embed_dim = 32
    #
    # # 清理 kwargs，防止传给 ModelArgs 时参数重复。
    # # DCAE-32 固定使用 embed_dim=32, num_bits=128。
    # kwargs.pop("z_channels", None)
    # kwargs.pop("codebook_embed_dim", None)
    # kwargs.pop("num_bits", None)
    #
    # return VQModel(
    #     ModelArgs(
    #         encoder_ch_mult=[1, 1, 2, 2, 4, 4],
    #         decoder_ch_mult=[1, 1, 2, 2, 4, 4],
    #         z_channels=z_ch,
    #         codebook_embed_dim=embed_dim,
    #         num_bits=embed_dim * 4,
    #         **kwargs,
    #     )
    # )

    z_ch = kwargs.get("z_channels", 512)
    embed_dim = 64

    kwargs.pop("z_channels", None)
    kwargs.pop("codebook_embed_dim", None)
    kwargs.pop("num_bits", None)

    return VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4, 4],
            z_channels=z_ch,
            codebook_embed_dim=embed_dim,
            num_bits=embed_dim * 4,
            **kwargs,
        )
    )


def DCAE_f64_Attn(**kwargs):
    z_ch = kwargs.get("z_channels", 1024)
    embed_dim = 256

    kwargs.pop("z_channels", None)
    kwargs.pop("codebook_embed_dim", None)
    kwargs.pop("num_bits", None)

    return VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4, 4, 8],
            decoder_ch_mult=[1, 1, 2, 2, 4, 4, 8],
            z_channels=z_ch,
            codebook_embed_dim=embed_dim,
            num_bits=embed_dim * 4,
            **kwargs,
        )
    )


def DCAE_f16_Attn(**kwargs):
    z_ch = kwargs.get("z_channels", 256)
    embed_dim = 8

    kwargs.pop("z_channels", None)
    kwargs.pop("codebook_embed_dim", None)
    kwargs.pop("num_bits", None)

    return VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4],
            z_channels=z_ch,
            codebook_embed_dim=embed_dim,
            num_bits=embed_dim * 4,
            **kwargs,
        )
    )


VQ_models = {
    "DCAE-16": DCAE_f16_Attn,
    "DCAE-32": DCAE_f32_Attn,
    "DCAE-64": DCAE_f64_Attn,
}
