from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from tokenizer.tokenizer_image.bsqdc_evit_model import Decoder, DecoderConfig, Encoder, EncoderConfig


@dataclass
class ModelArgs:
    latent_channels: int = 32
    encoder_width_list: list[int] = field(default_factory=lambda: [128, 256, 512, 512, 1024, 1024])
    encoder_depth_list: list[int] = field(default_factory=lambda: [0, 4, 8, 2, 2, 2])
    encoder_block_type: list[str] = field(
        default_factory=lambda: ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
    )
    decoder_width_list: list[int] = field(default_factory=lambda: [128, 256, 512, 512, 1024, 1024])
    decoder_depth_list: list[int] = field(default_factory=lambda: [0, 5, 10, 2, 2, 2])
    decoder_block_type: list[str] = field(
        default_factory=lambda: ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
    )
    decoder_norm: list[str] = field(default_factory=lambda: ["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"])
    decoder_act: list[str] = field(default_factory=lambda: ["relu", "relu", "relu", "silu", "silu", "silu"])
    scaling_factor: Optional[float] = None


class NullQuantizer(nn.Module):
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)


class DCAEModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            EncoderConfig(
                latent_channels=config.latent_channels,
                width_list=tuple(config.encoder_width_list),
                depth_list=tuple(config.encoder_depth_list),
                block_type=list(config.encoder_block_type),
            )
        )
        self.decoder = Decoder(
            DecoderConfig(
                latent_channels=config.latent_channels,
                width_list=tuple(config.decoder_width_list),
                depth_list=tuple(config.decoder_depth_list),
                block_type=list(config.decoder_block_type),
                norm=list(config.decoder_norm),
                act=list(config.decoder_act),
            )
        )

        # Kept only so the existing train loop can call raw_model.quantize.set_epoch(epoch).
        self.quantize = NullQuantizer()

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (len(self.config.decoder_width_list) - 1)

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.config.scaling_factor is not None:
            z = z * self.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor):
        if self.config.scaling_factor is not None:
            z = z / self.config.scaling_factor
        return self.decoder(z)

    def decode_code(self, code_b, shape=None, channel_first=True):
        if shape is not None:
            if channel_first:
                code_b = code_b.reshape(shape[0], shape[1], shape[2], shape[3])
            else:
                code_b = code_b.reshape(shape).permute(0, 3, 1, 2).contiguous()
        return self.decode(code_b)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        dec = self.decode(z)
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        usage = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return dec, (zero, zero, zero, usage)


def _drop_unused_kwargs(kwargs: dict[str, Any]):
    unused_keys = [
        "num_bits",
        "quantizer",
        "sample",
        "codebook_embed_dim",
        "codebook_l2_norm",
        "anneal_noise",
        "anneal_start_epoch",
        "anneal_end_epoch",
        "noise_start_scale",
        "noise_end_scale",
        "learnable_proj",
        "projector_hidden_mult",
        "projector_res_scale_init",
        "z_channels",
        "dropout_p",
    ]
    for key in unused_keys:
        kwargs.pop(key, None)


def _make_model(
    latent_channels: int,
    widths: list[int],
    enc_depths: list[int],
    dec_depths: list[int],
    block_types: list[str],
    decoder_norm: list[str],
    decoder_act: list[str],
    scaling_factor: Optional[float] = None,
    **kwargs,
):
    _drop_unused_kwargs(kwargs)
    return DCAEModel(
        ModelArgs(
            latent_channels=latent_channels,
            encoder_width_list=widths,
            encoder_depth_list=enc_depths,
            encoder_block_type=block_types,
            decoder_width_list=widths,
            decoder_depth_list=dec_depths,
            decoder_block_type=block_types,
            decoder_norm=decoder_norm,
            decoder_act=decoder_act,
            scaling_factor=scaling_factor,
            **kwargs,
        )
    )


def DCAE_f16(**kwargs):
    # EfficientViT does not publish an official f16 checkpoint config. This keeps
    # the same DCAE block/downsample recipe and drops the final f32 stage.
    return _make_model(
        latent_channels=16,
        widths=[128, 256, 512, 512, 1024],
        enc_depths=[0, 4, 8, 2, 2],
        dec_depths=[0, 5, 10, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu"],
        **kwargs,
    )


def DCAE_f32c32(**kwargs):
    return _make_model(
        latent_channels=32,
        widths=[128, 256, 512, 512, 1024, 1024],
        enc_depths=[0, 4, 8, 2, 2, 2],
        dec_depths=[0, 5, 10, 2, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu", "silu"],
        **kwargs,
    )


def DCAE_f64c128(**kwargs):
    return _make_model(
        latent_channels=128,
        widths=[128, 256, 512, 512, 1024, 1024, 2048],
        enc_depths=[0, 4, 8, 2, 2, 2, 2],
        dec_depths=[0, 5, 10, 2, 2, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
        **kwargs,
    )


def DCAE_f128c512(**kwargs):
    return _make_model(
        latent_channels=512,
        widths=[128, 256, 512, 512, 1024, 1024, 2048, 2048],
        enc_depths=[0, 4, 8, 2, 2, 2, 2, 2],
        dec_depths=[0, 5, 10, 2, 2, 2, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu", "silu", "silu", "silu"],
        **kwargs,
    )


VQ_models = {
    "DCAE-16": DCAE_f16,
    "DCAE-32": DCAE_f32c32,
    "DCAE-64": DCAE_f64c128,
    "DCAE-128": DCAE_f128c512,
}
