import contextlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from tokenizer.tokenizer_image.bsqplus_model import VQ_models


@dataclass
class VideoTokenizerConfig:
    num_frames: int = 8
    compressed_channels: Optional[int] = None
    use_3d_smoother: bool = True
    smoother_layers: int = 2
    smoother_kernel: int = 3

    def __post_init__(self):
        if self.num_frames not in (4, 8, 16):
            raise ValueError(f"num_frames must be one of [4, 8, 16], got {self.num_frames}")


class DecoderFeatureSmoother3D(nn.Module):
    """Apply 3D conv on decoder feature volume: [B, C, T, H, W]."""

    def __init__(self, channels: int, layers: int = 2, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        blocks = []
        for _ in range(layers):
            blocks.extend([
                nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=pad),
                nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
                nn.SiLU(inplace=True),
            ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalResidualCompressor(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int, compressed_channels: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.delta_frames = num_frames - 1

        raw_channels = self.delta_frames * embed_dim
        if compressed_channels is None:
            compressed_channels = 2 * num_frames
        self.raw_channels = raw_channels
        self.compressed_channels = compressed_channels

        hidden = max(raw_channels // 2, compressed_channels)
        self.compress = nn.Sequential(
            nn.Linear(raw_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, compressed_channels),
        )
        self.expand = nn.Sequential(
            nn.Linear(compressed_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, raw_channels),
        )

    def encode(self, residuals: torch.Tensor) -> torch.Tensor:
        # residuals: [B, T-1, D, H, W]
        b, t, d, h, w = residuals.shape
        assert t == self.delta_frames and d == self.embed_dim
        flat = residuals.permute(0, 3, 4, 1, 2).contiguous().view(b, h, w, t * d)
        comp = self.compress(flat)
        return comp.permute(0, 3, 1, 2).contiguous()  # [B, Cc, H, W]

    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        # compressed: [B, Cc, H, W]
        b, c, h, w = compressed.shape
        assert c == self.compressed_channels
        flat = compressed.permute(0, 2, 3, 1).contiguous()
        raw = self.expand(flat).view(b, h, w, self.delta_frames, self.embed_dim)
        return raw.permute(0, 3, 4, 1, 2).contiguous()  # [B, T-1, D, H, W]


class BSQPlusVideoTokenizer(nn.Module):
    """
    Video tokenizer on top of a frozen image tokenizer.

    Encoder:
    - Encode each frame with frozen image tokenizer to q_t
    - Build residuals r_t = q_t - q_{t-1}
    - Compress concatenated residuals to a short temporal code
    - Final video token = concat([q_0, temporal_code])

    Decoder:
    - Decode temporal code to residual sequence
    - Recover frame latents by cumulative sum from q_0
    - Run image decoder weights, with optional inserted 3D conv in decoder feature space
    """

    def __init__(
        self,
        image_tokenizer: nn.Module,
        config: VideoTokenizerConfig,
        freeze_image_tokenizer: bool = True,
    ):
        super().__init__()
        self.image_tokenizer = image_tokenizer
        self.config = config
        self.freeze_image_tokenizer = freeze_image_tokenizer

        if freeze_image_tokenizer:
            for p in self.image_tokenizer.parameters():
                p.requires_grad = False
            self.image_tokenizer.eval()

        embed_dim = getattr(self.image_tokenizer.config, "codebook_embed_dim")
        self.embed_dim = embed_dim
        self.temporal = TemporalResidualCompressor(
            embed_dim=embed_dim,
            num_frames=config.num_frames,
            compressed_channels=config.compressed_channels,
        )

        self.decoder_feature_smoother = None
        if config.use_3d_smoother:
            decoder_mid_channels = self.image_tokenizer.decoder.conv_in.out_channels
            self.decoder_feature_smoother = DecoderFeatureSmoother3D(
                channels=decoder_mid_channels,
                layers=config.smoother_layers,
                kernel_size=config.smoother_kernel,
            )

    @property
    def compressed_channels(self) -> int:
        return self.temporal.compressed_channels

    @property
    def video_token_channels(self) -> int:
        return self.embed_dim + self.temporal.compressed_channels

    def _as_video_volume(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        # [B*T, C, H, W] -> [B, C, T, H, W]
        bt, c, h, w = x.shape
        assert bt == b * t
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def _as_frame_batch(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W] -> [B*T, C, H, W]
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)

    def _encode_frames(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # video: [B, T, 3, H, W]
        b, t, c, h, w = video.shape
        if t != self.config.num_frames:
            raise ValueError(f"Expected {self.config.num_frames} frames, got {t}")

        x = video.view(b * t, c, h, w)
        grad_ctx = torch.no_grad() if self.freeze_image_tokenizer else contextlib.nullcontext()
        with grad_ctx:
            q, codebook_loss, _ = self.image_tokenizer.encode(x)
        q = q.view(b, t, q.shape[1], q.shape[2], q.shape[3]).contiguous()
        return q, codebook_loss

    def encode_video(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        q, _ = self._encode_frames(video)
        base = q[:, 0]  # [B, D, h, w]
        residuals = q[:, 1:] - q[:, :-1]  # [B, T-1, D, h, w]
        temporal_code = self.temporal.encode(residuals)
        video_tokens = torch.cat([base, temporal_code], dim=1)
        return {
            "video_tokens": video_tokens,
            "base": base,
            "temporal_code": temporal_code,
            "frame_latents": q,
            "residuals": residuals,
        }

    def _decode_latents_with_decoder(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, T, D, h, w]
        b, t, d, h, w = latents.shape
        flat = latents.view(b * t, d, h, w)

        # Use frozen image decoder weights; insert 3D conv in decoder middle feature map.
        dec = self.image_tokenizer.decoder
        h2d = self.image_tokenizer.post_quant_conv(flat)
        h2d = dec.conv_in(h2d)

        for m in dec.mid:
            h2d = m(h2d)

        if self.decoder_feature_smoother is not None:
            h3d = self._as_video_volume(h2d, b=b, t=t)
            h3d = self.decoder_feature_smoother(h3d)
            h2d = self._as_frame_batch(h3d)

        for i_level, block in enumerate(dec.conv_blocks):
            for i_block in range(dec.num_res_blocks + 1):
                h2d = block.res[i_block](h2d)
                if len(block.attn) > 0:
                    h2d = block.attn[i_block](h2d)
            if i_level != dec.num_resolutions - 1:
                h2d = block.upsample(h2d)

        h2d = dec.norm_out(h2d)
        h2d = h2d * torch.sigmoid(h2d)
        h2d = dec.conv_out(h2d)

        return h2d.view(b, t, h2d.shape[1], h2d.shape[2], h2d.shape[3]).contiguous()

    def decode_video_tokens(self, video_tokens: torch.Tensor, return_latents: bool = False):
        base = video_tokens[:, : self.embed_dim]
        temporal_code = video_tokens[:, self.embed_dim :]

        pred_residuals = self.temporal.decode(temporal_code)  # [B, T-1, D, h, w]

        latents = [base]
        for i in range(self.config.num_frames - 1):
            latents.append(latents[-1] + pred_residuals[:, i])
        latents = torch.stack(latents, dim=1)  # [B, T, D, h, w]

        frames = self._decode_latents_with_decoder(latents)

        if return_latents:
            return frames, latents, pred_residuals
        return frames

    def forward(self, video: torch.Tensor, return_latents: bool = False):
        enc = self.encode_video(video)
        if return_latents:
            recon, recon_latents, pred_residuals = self.decode_video_tokens(enc["video_tokens"], return_latents=True)
            return recon, {
                "video_tokens": enc["video_tokens"],
                "q_latents": enc["frame_latents"],
                "gt_residuals": enc["residuals"],
                "recon_latents": recon_latents,
                "pred_residuals": pred_residuals,
            }
        recon = self.decode_video_tokens(enc["video_tokens"], return_latents=False)
        return recon, {"video_tokens": enc["video_tokens"]}


def load_image_tokenizer_from_ckpt(
    ckpt_path: str,
    vq_model: str = "VQ-16",
    quantizer: str = "bsq",
    num_bits: int = 14,
    codebook_embed_dim: int = 8,
    codebook_l2_norm: bool = True,
    codebook_size: int = 16384,
    commit_loss_beta: float = 0.0,
    entropy_loss_ratio: float = 0.1,
    dropout_p: float = 0.0,
    sample: bool = False,
    map_location: str = "cpu",
) -> nn.Module:
    model = VQ_models[vq_model](
        quantizer=quantizer,
        num_bits=num_bits,
        codebook_embed_dim=codebook_embed_dim,
        codebook_l2_norm=codebook_l2_norm,
        codebook_size=codebook_size,
        commit_loss_beta=commit_loss_beta,
        entropy_loss_ratio=entropy_loss_ratio,
        dropout_p=dropout_p,
        sample=sample,
    )
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model
