import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Basic blocks
# =============================================================================

class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * norm * self.weight


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout_p: float = 0.0,
        norm_layer=RMSNorm2d,
    ):
        super().__init__()

        self.block = nn.Sequential(
            norm_layer(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            norm_layer(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttnBlock(nn.Module):
    """
    Single-head spatial self-attention, similar to the original VQ bottleneck attention.

    Input / output:
        [B, C, H, W]
    """

    def __init__(self, channels: int, norm_layer=RMSNorm2d):
        super().__init__()
        self.norm = norm_layer(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, height, width = q.shape
        n = height * width

        q = q.reshape(b, c, n).permute(0, 2, 1)   # [B, HW, C]
        k = k.reshape(b, c, n)                    # [B, C, HW]
        v = v.reshape(b, c, n)                    # [B, C, HW]

        attn = torch.bmm(q, k) * (c ** -0.5)      # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.permute(0, 2, 1))   # [B, C, HW]
        h = h.reshape(b, c, height, width)
        h = self.proj_out(h)

        return x + h


class MidBlock(nn.Module):
    """
    Bottleneck block, VQ-style:

        ResBlock
        AttnBlock
        ResBlock

    No extra attention is added outside this mid block.
    """

    def __init__(
        self,
        channels: int,
        dropout_p: float = 0.0,
        norm_layer=RMSNorm2d,
    ):
        super().__init__()

        self.block = nn.Sequential(
            ResBlock(channels, dropout_p=dropout_p, norm_layer=norm_layer),
            AttnBlock(channels, norm_layer=norm_layer),
            ResBlock(channels, dropout_p=dropout_p, norm_layer=norm_layer),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# Encoder / Decoder
# =============================================================================

class DCAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        z_channels: int = 8,
        num_res_blocks: int = 2,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        blocks = []
        curr_channels = base_channels

        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult

            if curr_channels != out_channels:
                blocks.append(nn.Conv2d(curr_channels, out_channels, kernel_size=1))
                curr_channels = out_channels

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(curr_channels, dropout_p=dropout_p))

            if level != len(channel_mult) - 1:
                next_channels = base_channels * channel_mult[level + 1]
                blocks.append(Downsample(curr_channels, next_channels))
                curr_channels = next_channels

        self.blocks = nn.Sequential(*blocks)
        self.mid = MidBlock(curr_channels, dropout_p=dropout_p)

        self.norm_out = RMSNorm2d(curr_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(curr_channels, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x


class DCAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        z_channels: int = 8,
        num_res_blocks: int = 2,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.out_channels = out_channels

        rev_mult = tuple(reversed(channel_mult))
        curr_channels = base_channels * rev_mult[0]

        self.conv_in = nn.Conv2d(
            z_channels,
            curr_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = MidBlock(curr_channels, dropout_p=dropout_p)

        blocks = []

        for level, mult in enumerate(rev_mult):
            block_out_channels = base_channels * mult

            if curr_channels != block_out_channels:
                blocks.append(
                    nn.Conv2d(
                        curr_channels,
                        block_out_channels,
                        kernel_size=1,
                    )
                )
                curr_channels = block_out_channels

            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        curr_channels,
                        dropout_p=dropout_p,
                    )
                )

            if level != len(rev_mult) - 1:
                next_channels = base_channels * rev_mult[level + 1]
                blocks.append(
                    Upsample(
                        curr_channels,
                        next_channels,
                    )
                )
                curr_channels = next_channels

        self.blocks = nn.Sequential(*blocks)

        self.norm_out = RMSNorm2d(curr_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_out = nn.Conv2d(
            curr_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid(z)
        z = self.blocks(z)
        z = self.norm_out(z)
        z = self.act(z)
        z = self.conv_out(z)
        return z

# =============================================================================
# BSQ quantizer
# =============================================================================

class BSQQuantizer(nn.Module):
    """
    Binary Spherical Quantizer.

    No integer index is produced or stored.

    Internal training representation:
        bits_st: float tensor in {-1, +1}, shape [B, num_bits, H, W]

    Export / storage representation:
        binary_code: bool / uint8 tensor in {0, 1}, shape [B, num_bits, H, W]
    """

    def __init__(
        self,
        num_bits: int = 128,
        embed_dim: int = 32,
        codebook_l2_norm: bool = True,
        sample: bool = False,
        anneal_noise: bool = False,
        noise_schedule: str = "cosine_decay",
        anneal_start_epoch: int = 10,
        anneal_end_epoch: int = 30,
        noise_start_scale: float = 1.0,
        noise_end_scale: float = 0.1,
        noise_peak_scale: float = 1.0,
        learnable_proj: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        assert noise_schedule in ["constant", "cosine_decay", "warmup_cosine_decay"]

        self.num_bits = num_bits
        self.embed_dim = embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.sample = sample

        self.anneal_noise = anneal_noise
        self.noise_schedule = noise_schedule
        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_end_epoch = anneal_end_epoch
        self.noise_start_scale = noise_start_scale
        self.noise_end_scale = noise_end_scale
        self.noise_peak_scale = noise_peak_scale

        self.learnable_proj = learnable_proj
        self.eps = eps

        self.register_buffer("current_epoch", torch.zeros((), dtype=torch.long), persistent=False)

        # z-space -> bit logits
        # shape: [num_bits, embed_dim]
        proj = torch.randn(num_bits, embed_dim) / math.sqrt(embed_dim)

        # bits -> z-space
        # shape: [embed_dim, num_bits]
        deproj = torch.randn(embed_dim, num_bits) / math.sqrt(num_bits)

        if learnable_proj:
            self.proj = nn.Parameter(proj)
            self.deproj = nn.Parameter(deproj)
        else:
            self.register_buffer("proj", F.normalize(proj, dim=1, eps=eps), persistent=True)
            self.register_buffer("deproj", F.normalize(deproj, dim=0, eps=eps), persistent=True)

    def set_epoch(self, epoch: int):
        self.current_epoch.fill_(int(epoch))

    def _get_proj(self) -> torch.Tensor:
        if self.learnable_proj:
            return self.proj
        return F.normalize(self.proj, dim=1, eps=self.eps)

    def _get_deproj(self) -> torch.Tensor:
        if self.learnable_proj:
            return self.deproj
        return F.normalize(self.deproj, dim=0, eps=self.eps)

    def _get_noise_scale(self) -> float:
        """
        Supported schedules:

        1. constant
           scale = noise_start_scale

        2. cosine_decay
           Before anneal_start_epoch:
               scale = noise_start_scale
           Between start and end:
               cosine decay from noise_start_scale to noise_end_scale
           After anneal_end_epoch:
               scale = noise_end_scale

        3. warmup_cosine_decay
           Before anneal_start_epoch:
               linearly warm up from noise_start_scale to noise_peak_scale
           Between start and end:
               cosine decay from noise_peak_scale to noise_end_scale
           After anneal_end_epoch:
               scale = noise_end_scale

        For tokenizer reconstruction, cosine_decay is usually safer.
        warmup_cosine_decay is useful when early bit collapse is observed.
        """
        if not self.sample:
            return 0.0

        if not self.anneal_noise:
            return float(self.noise_start_scale)

        epoch = int(self.current_epoch.item())

        if self.noise_schedule == "constant":
            return float(self.noise_start_scale)

        if self.anneal_end_epoch <= self.anneal_start_epoch:
            return float(self.noise_end_scale)

        if self.noise_schedule == "cosine_decay":
            if epoch < self.anneal_start_epoch:
                return float(self.noise_start_scale)

            if epoch >= self.anneal_end_epoch:
                return float(self.noise_end_scale)

            progress = (epoch - self.anneal_start_epoch) / (
                self.anneal_end_epoch - self.anneal_start_epoch
            )
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.noise_end_scale + (
                self.noise_start_scale - self.noise_end_scale
            ) * cosine
            return float(scale)

        # warmup_cosine_decay
        if epoch < self.anneal_start_epoch:
            if self.anneal_start_epoch <= 0:
                return float(self.noise_peak_scale)

            warmup_progress = epoch / max(self.anneal_start_epoch, 1)
            warmup_progress = min(max(warmup_progress, 0.0), 1.0)
            scale = self.noise_start_scale + (
                self.noise_peak_scale - self.noise_start_scale
            ) * warmup_progress
            return float(scale)

        if epoch >= self.anneal_end_epoch:
            return float(self.noise_end_scale)

        progress = (epoch - self.anneal_start_epoch) / (
            self.anneal_end_epoch - self.anneal_start_epoch
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = self.noise_end_scale + (
            self.noise_peak_scale - self.noise_end_scale
        ) * cosine
        return float(scale)

    @staticmethod
    def _ste_binarize(logits: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            hard bits in {-1, +1}

        Backward:
            identity gradient through logits
        """
        hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        return logits + (hard - logits).detach()

    @staticmethod
    def bits_to_binary_code(
        bits_pm_one: torch.Tensor,
        dtype: torch.dtype = torch.bool,
    ) -> torch.Tensor:
        code = bits_pm_one > 0

        if dtype == torch.bool:
            return code

        return code.to(dtype)

    @staticmethod
    def binary_code_to_bits(binary_code: torch.Tensor) -> torch.Tensor:
        return binary_code.float() * 2.0 - 1.0

    def project_to_bits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, embed_dim, H, W]

        Returns:
            logits: [B, num_bits, H, W]
        """
        proj = self._get_proj()
        logits = torch.einsum("bchw,nc->bnhw", z, proj)
        return logits

    def project_from_bits(self, bits_pm_one: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits_pm_one: [B, num_bits, H, W]

        Returns:
            z_q: [B, embed_dim, H, W]
        """
        deproj = self._get_deproj()
        z_q = torch.einsum("bnhw,cn->bchw", bits_pm_one, deproj)
        return z_q

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: continuous latent, [B, embed_dim, H, W]

        Returns:
            z_q_st:
                quantized latent with straight-through path,
                [B, embed_dim, H, W]

            loss_tuple:
                compatible with existing VQLoss:
                    codebook_loss, commit_loss, entropy_loss, usage

            binary_code:
                bool tensor, [B, num_bits, H, W]
        """
        if self.codebook_l2_norm:
            z_in = F.normalize(z, dim=1, eps=self.eps)
        else:
            z_in = z

        logits = self.project_to_bits(z_in)

        if self.training and self.sample:
            noise_scale = self._get_noise_scale()
            logits_for_bits = logits + torch.randn_like(logits) * noise_scale
        else:
            logits_for_bits = logits

        bits_st = self._ste_binarize(logits_for_bits)
        binary_code = self.bits_to_binary_code(bits_st, dtype=torch.bool)

        z_q = self.project_from_bits(bits_st)

        if self.codebook_l2_norm:
            z_q = F.normalize(z_q, dim=1, eps=self.eps)

        # Straight-through latent path.
        z_q_st = z + (z_q - z).detach()

        # BSQ-compatible losses.
        # codebook_loss trains projection / deprojection side.
        # commit_loss trains encoder output toward quantized latent.
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z_q.detach(), z)
        entropy_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        with torch.no_grad():
            prob_one = binary_code.float().mean(dim=(0, 2, 3))
            bit_balance = 1.0 - (prob_one - 0.5).abs().mean() * 2.0
            usage = bit_balance.clamp(0.0, 1.0)

        loss_tuple = (
            codebook_loss,
            commit_loss,
            entropy_loss,
            usage,
        )

        return z_q_st, loss_tuple, binary_code

    @torch.no_grad()
    def encode(self, z: torch.Tensor, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Args:
            z: [B, embed_dim, H, W]

        Returns:
            binary_code: [B, num_bits, H, W], values in {0, 1}
        """
        if self.codebook_l2_norm:
            z = F.normalize(z, dim=1, eps=self.eps)

        logits = self.project_to_bits(z)
        bits = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        binary_code = self.bits_to_binary_code(bits, dtype=dtype)
        return binary_code

    @torch.no_grad()
    def decode(self, binary_code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            binary_code: bool / uint8 / float tensor in {0, 1},
                         shape [B, num_bits, H, W]

        Returns:
            z_q: [B, embed_dim, H, W]
        """
        bits = self.binary_code_to_bits(binary_code)
        z_q = self.project_from_bits(bits)

        if self.codebook_l2_norm:
            z_q = F.normalize(z_q, dim=1, eps=self.eps)

        return z_q


# =============================================================================
# DCAE + BSQ model
# =============================================================================

class DCAE_BSQ(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        num_bits: int = 32,
        codebook_embed_dim: int = 8,
        codebook_l2_norm: bool = True,
        dropout_p: float = 0.0,
        sample: bool = False,
        anneal_noise: bool = False,
        noise_schedule: str = "cosine_decay",
        anneal_start_epoch: int = 10,
        anneal_end_epoch: int = 30,
        noise_start_scale: float = 1.0,
        noise_end_scale: float = 0.1,
        noise_peak_scale: float = 1.0,
        learnable_proj: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_bits = num_bits
        self.codebook_embed_dim = codebook_embed_dim
        self.channel_mult = channel_mult

        self.encoder = DCAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            z_channels=codebook_embed_dim,
            num_res_blocks=num_res_blocks,
            dropout_p=dropout_p,
        )

        self.quantize = BSQQuantizer(
            num_bits=num_bits,
            embed_dim=codebook_embed_dim,
            codebook_l2_norm=codebook_l2_norm,
            sample=sample,
            anneal_noise=anneal_noise,
            noise_schedule=noise_schedule,
            anneal_start_epoch=anneal_start_epoch,
            anneal_end_epoch=anneal_end_epoch,
            noise_start_scale=noise_start_scale,
            noise_end_scale=noise_end_scale,
            noise_peak_scale=noise_peak_scale,
            learnable_proj=learnable_proj,
        )

        self.decoder = DCAEDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            z_channels=codebook_embed_dim,
            num_res_blocks=num_res_blocks,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, codebook_loss, binary_code = self.quantize(z)
        x_rec = self.decoder(z_q)

        # Keep training API unchanged:
        #     recons_imgs, codebook_loss = vq_model(imgs)
        return x_rec, codebook_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Return binary code directly.

        Args:
            x: image tensor [B, 3, H, W]
            dtype: torch.bool or torch.uint8

        Returns:
            binary_code: [B, num_bits, h, w]
        """
        z = self.encoder(x)
        binary_code = self.quantize.encode(z, dtype=dtype)
        return binary_code

    @torch.no_grad()
    def decode_code(self, binary_code: torch.Tensor) -> torch.Tensor:
        """
        Decode binary code tensor.

        Args:
            binary_code: [B, num_bits, h, w], bool / uint8 / float in {0, 1}

        Returns:
            reconstructed image
        """
        z_q = self.quantize.decode(binary_code)
        x_rec = self.decoder(z_q)
        return x_rec

    @torch.no_grad()
    def reconstruct_from_code(self, binary_code: torch.Tensor) -> torch.Tensor:
        return self.decode_code(binary_code)


# =============================================================================
# Model configs
# =============================================================================

def DCAE_16(**kwargs):
    kwargs.setdefault("codebook_embed_dim", 8)
    kwargs.setdefault("num_bits", 32)

    return DCAE_BSQ(
        channel_mult=(1, 1, 2, 2, 4),
        base_channels=128,
        num_res_blocks=2,
        **kwargs,
    )


def DCAE_32(**kwargs):
    kwargs.setdefault("codebook_embed_dim", 32)
    kwargs.setdefault("num_bits", 128)

    return DCAE_BSQ(
        channel_mult=(1, 1, 2, 2, 4, 4),
        base_channels=128,
        num_res_blocks=2,
        **kwargs,
    )


def DCAE_64(**kwargs):
    kwargs.setdefault("codebook_embed_dim", 128)
    kwargs.setdefault("num_bits", 512)

    return DCAE_BSQ(
        channel_mult=(1, 1, 2, 2, 4, 4, 8),
        base_channels=128,
        num_res_blocks=2,
        **kwargs,
    )


VQ_models: Dict[str, nn.Module] = {
    "DCAE-16": DCAE_16,
    "DCAE-32": DCAE_32,
    "DCAE-64": DCAE_64,
}