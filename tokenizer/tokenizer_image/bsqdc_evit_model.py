from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.tokenizer_image.bsqdc_model import BSQQuantizer, ModelArgs


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class RMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x / torch.sqrt(torch.square(x).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name: Optional[str], num_features: Optional[int] = None) -> Optional[nn.Module]:
    if name is None:
        return None
    if name == "bn2d":
        return nn.BatchNorm2d(num_features)
    if name == "ln2d":
        return LayerNorm2d(num_features)
    if name == "trms2d":
        return RMSNorm2d(num_features)
    raise ValueError(f"norm {name} is not supported")


def build_act(name: Optional[str], inplace: bool = True) -> Optional[nn.Module]:
    if name is None:
        return None
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "hswish":
        return nn.Hardswish(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"act {name} is not supported")


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: Sequence[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module, shortcut: Optional[nn.Module]):
        super().__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.main(x)
        return self.main(x) + self.shortcut(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bias: tuple[bool, bool] = (True, False),
        norm: tuple[Optional[str], Optional[str]] = (None, "trms2d"),
        act_func: tuple[Optional[str], Optional[str]] = ("silu", None),
    ):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expand_ratio: float = 4,
        use_bias: tuple[bool, bool, bool] = (True, True, False),
        norm: tuple[Optional[str], Optional[str], Optional[str]] = (None, None, "trms2d"),
        act_func: tuple[Optional[str], Optional[str], Optional[str]] = ("silu", "silu", None),
    ):
        super().__init__()
        mid_channels = round(in_channels * expand_ratio)
        self.glu_act = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(in_channels, mid_channels * 2, 1, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, use_bias=use_bias[2], norm=norm[2], act_func=act_func[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(self.inverted_conv(x))
        x, gate = torch.chunk(x, 2, dim=1)
        return self.point_conv(x * self.glu_act(gate))


class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        norm: tuple[Optional[str], Optional[str]] = (None, "trms2d"),
        scales: tuple[int, ...] = (),
        eps: float = 1e-15,
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        heads = int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1, use_bias=False, norm=norm[0], act_func=None)
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 * total_dim, 3 * total_dim, scale, padding=scale // 2, groups=3 * total_dim, bias=False),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=False),
                )
                for scale in scales
            ]
        )
        self.kernel_func = nn.ReLU(inplace=False)
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, 1, use_bias=False, norm=norm[1], act_func=None)

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        b, _, h, w = qkv.shape
        if qkv.dtype in (torch.float16, torch.bfloat16):
            qkv = qkv.float()
        qkv = qkv.reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        out = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = torch.cat([qkv] + [op(qkv) for op in self.aggreg], dim=1)
        return self.proj(self.relu_linear_att(qkv).to(qkv.dtype))


class EfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, norm: str = "trms2d", act_func: str = "silu", scales: tuple[int, ...] = ()):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(in_channels, in_channels, norm=(None, norm), scales=scales),
            IdentityLayer(),
        )
        self.local_module = ResidualBlock(
            GLUMBConv(in_channels, in_channels, norm=(None, None, norm), act_func=(act_func, act_func, None)),
            IdentityLayer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_module(self.context_module(x))


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, factor: int):
        super().__init__()
        assert out_channels % (factor ** 2) == 0
        self.factor = factor
        self.conv = ConvLayer(in_channels, out_channels // (factor ** 2), kernel_size, use_bias=True, norm=None, act_func=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.conv(x), self.factor)


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        assert in_channels * factor ** 2 % out_channels == 0
        self.out_channels = out_channels
        self.factor = factor
        self.group_size = in_channels * factor ** 2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        b, _, h, w = x.shape
        return x.view(b, self.out_channels, self.group_size, h, w).mean(dim=2)


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, factor: int):
        super().__init__()
        self.factor = factor
        self.conv = ConvLayer(in_channels, out_channels * factor ** 2, kernel_size, use_bias=True, norm=None, act_func=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), self.factor)


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        assert out_channels * factor ** 2 % in_channels == 0
        self.factor = factor
        self.repeats = out_channels * factor ** 2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x.repeat_interleave(self.repeats, dim=1), self.factor)


@dataclass
class EncoderConfig:
    in_channels: int = 3
    latent_channels: int = 256
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (0, 4, 8, 2, 2, 2)
    block_type: Any = field(default_factory=lambda: ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"])
    norm: str = "trms2d"
    act: str = "silu"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None


@dataclass
class DecoderConfig:
    in_channels: int = 3
    latent_channels: int = 256
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (0, 5, 10, 2, 2, 2)
    block_type: Any = field(default_factory=lambda: ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"])
    norm: Any = field(default_factory=lambda: ["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"])
    act: Any = field(default_factory=lambda: ["relu", "relu", "relu", "silu", "silu", "silu"])
    out_norm: str = "trms2d"
    out_act: str = "relu"


def build_block(block_type: str, channels: int, norm: Optional[str], act: Optional[str]) -> nn.Module:
    if block_type == "ResBlock":
        return ResidualBlock(
            ResBlock(channels, channels, use_bias=(True, False), norm=(None, norm), act_func=(act, None)),
            IdentityLayer(),
        )
    if block_type == "EViT_GLU":
        return EfficientViTBlock(channels, norm=norm, act_func=act, scales=())
    if block_type == "EViTS5_GLU":
        return EfficientViTBlock(channels, norm=norm, act_func=act, scales=(5,))
    raise ValueError(f"block_type {block_type} is not supported")


def build_stage_main(width: int, depth: int, block_type: str, norm: str, act: str) -> list[nn.Module]:
    return [build_block(block_type, width, norm, act) for _ in range(depth)]


def build_downsample_block(in_channels: int, out_channels: int) -> nn.Module:
    return ResidualBlock(
        ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, kernel_size=3, factor=2),
        PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, factor=2),
    )


def build_upsample_block(in_channels: int, out_channels: int) -> nn.Module:
    return ResidualBlock(
        ConvPixelShuffleUpSampleLayer(in_channels, out_channels, kernel_size=3, factor=2),
        ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, factor=2),
    )


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        num_stages = len(cfg.width_list)
        assert len(cfg.depth_list) == num_stages
        assert isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages

        self.project_in = (
            ConvLayer(cfg.in_channels, cfg.width_list[0], 3, use_bias=True, norm=None, act_func=None)
            if cfg.depth_list[0] > 0
            else ConvPixelUnshuffleDownSampleLayer(cfg.in_channels, cfg.width_list[1], 3, factor=2)
        )

        stages = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            stage = build_stage_main(width, depth, cfg.block_type[stage_id], cfg.norm, cfg.act)
            if stage_id < num_stages - 1 and depth > 0:
                stage.append(build_downsample_block(width, cfg.width_list[stage_id + 1]))
            stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(stages)

        self.project_out = ResidualBlock(
            OpSequential([
                build_norm(cfg.out_norm, cfg.width_list[-1]),
                build_act(cfg.out_act),
                ConvLayer(cfg.width_list[-1], cfg.latent_channels, 3, use_bias=True, norm=None, act_func=None),
            ]),
            PixelUnshuffleChannelAveragingDownSampleLayer(cfg.width_list[-1], cfg.latent_channels, factor=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) > 0:
                x = stage(x)
        return self.project_out(x)


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        num_stages = len(cfg.width_list)
        assert len(cfg.depth_list) == num_stages
        assert isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages

        self.project_in = ResidualBlock(
            ConvLayer(cfg.latent_channels, cfg.width_list[-1], 3, use_bias=True, norm=None, act_func=None),
            ChannelDuplicatingPixelUnshuffleUpSampleLayer(cfg.latent_channels, cfg.width_list[-1], factor=1),
        )

        stages = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                stage.append(build_upsample_block(cfg.width_list[stage_id + 1], width))
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(build_stage_main(width, depth, cfg.block_type[stage_id], norm, act))
            stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(stages)

        out_in = cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1]
        self.project_out = OpSequential([
            build_norm(cfg.out_norm, out_in),
            build_act(cfg.out_act),
            ConvPixelShuffleUpSampleLayer(out_in, cfg.in_channels, 3, factor=2)
            if cfg.depth_list[0] == 0
            else ConvLayer(out_in, cfg.in_channels, 3, use_bias=True, norm=None, act_func=None),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) > 0:
                x = stage(x)
        return self.project_out(x)

    @property
    def last_layer(self):
        for module in reversed(list(self.project_out.modules())):
            if isinstance(module, nn.Conv2d):
                return module.weight
        raise RuntimeError("decoder has no Conv2d output layer")


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            EncoderConfig(
                latent_channels=config.z_channels,
                width_list=tuple(config.encoder_ch_mult),
                depth_list=tuple(config.encoder_depth_list),
                block_type=list(config.encoder_block_type),
            )
        )
        self.decoder = Decoder(
            DecoderConfig(
                latent_channels=config.z_channels,
                width_list=tuple(config.decoder_ch_mult),
                depth_list=tuple(config.decoder_depth_list),
                block_type=list(config.decoder_block_type),
                norm=list(config.decoder_norm),
                act=list(config.decoder_act),
            )
        )
        self.quantize = BSQQuantizer(config)
        self.quant_conv = nn.Conv2d(config.z_channels, config.num_bits, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.quant_conv(self.encoder(x))
        quant, codebook_loss_tuple = self.quantize(h)
        return quant, codebook_loss_tuple

    def decode(self, quant):
        return self.decoder(self.post_quant_conv(quant))

    def decode_code(self, code_b, shape=None, channel_first=True):
        return self.decode(self.quantize.get_codebook_entry(code_b, shape, channel_first))

    def forward(self, x):
        q, codebook_loss_tuple = self.encode(x)
        return self.decode(q), codebook_loss_tuple


def _make_model(
    z_channels: int,
    embed_dim: int,
    widths: list[int],
    enc_depths: list[int],
    dec_depths: list[int],
    block_types: list[str],
    decoder_norm: list[str],
    decoder_act: list[str],
    **kwargs,
):
    kwargs.pop("z_channels", None)
    kwargs.pop("codebook_embed_dim", None)
    kwargs.pop("num_bits", None)

    config = ModelArgs(
        encoder_ch_mult=widths,
        decoder_ch_mult=widths,
        z_channels=z_channels,
        codebook_embed_dim=embed_dim,
        num_bits=embed_dim * 4,
        **kwargs,
    )
    config.encoder_depth_list = enc_depths
    config.decoder_depth_list = dec_depths
    config.encoder_block_type = block_types
    config.decoder_block_type = block_types
    config.decoder_norm = decoder_norm
    config.decoder_act = decoder_act
    return VQModel(config)


def DCAE_f16_Attn(**kwargs):
    return _make_model(
        z_channels=kwargs.get("z_channels", 256),
        embed_dim=8,
        widths=[128, 256, 512, 512, 1024],
        enc_depths=[0, 4, 8, 2, 2],
        dec_depths=[0, 5, 10, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu"],
        **kwargs,
    )


def DCAE_f32_Attn(**kwargs):
    return _make_model(
        z_channels=kwargs.get("z_channels", 512),
        embed_dim=64,
        # embed_dim=32,

        widths=[128, 256, 512, 512, 1024, 1024],
        enc_depths=[0, 4, 8, 2, 2, 2],
        dec_depths=[0, 5, 10, 2, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu", "silu"],
        **kwargs,
    )


def DCAE_f64_Attn(**kwargs):
    return _make_model(
        z_channels=kwargs.get("z_channels", 1024),
        embed_dim=256,
        widths=[128, 256, 512, 512, 1024, 1024, 2048],
        enc_depths=[0, 4, 8, 2, 2, 2, 2],
        dec_depths=[0, 5, 10, 2, 2, 2, 2],
        block_types=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
        decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d"],
        decoder_act=["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
        **kwargs,
    )


VQ_models = {
    "DCAE-16": DCAE_f16_Attn,
    "DCAE-32": DCAE_f32_Attn,
    "DCAE-64": DCAE_f64_Attn,
}
