# tokenizer/tokenizer_image/ste_model.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Configuration & Utils
# ==============================================================================

@dataclass
class ModelArgs:
    # Architecture Params
    # Encoder downsample steps: 1->1->2->4->8->16 (assuming input 256)
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256  # The dimension of the latent space / tokens
    dropout_p: float = 0.0

    # TiTok (Global) Settings
    num_global_tokens: int = 64
    min_global_tokens: int = 8

    # VidCom (Spatial) Settings
    p_min: float = 0.5  # Min probability mass to keep during training
    p_max: float = 1.0  # Max probability mass to keep

    # Decoder Settings
    bridge_depth: int = 6  # Number of Transformer layers in the decoder bridge


def nonlinearity(x):
    # Swish / Silu
    return x * torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    # Standard GroupNorm for VQGAN/SD
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# ==============================================================================
# 2. Basic Building Blocks (MLP, Attention)
# ==============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, mask=None):
        # x_q: (B, N_q, C)
        # x_kv: (B, N_kv, C)
        # mask: (B, N_kv) -> True indicates padding/ignore
        B, N_q, C = x_q.shape
        N_kv = x_kv.shape[1]

        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 2, 1, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)

        if mask is not None:
            # Expand mask for heads and query dim: (B, 1, 1, N_kv)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            # Masked fill with -inf where mask is True
            attn = attn.masked_fill(mask_expanded, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==============================================================================
# 3. Compressor (Continuous / Dynamic)
# ==============================================================================

class ContinuousCompressor(nn.Module):
    def __init__(self, dim, num_global_tokens=64, min_global_tokens=8, p_min=0.5, p_max=1.0):
        super().__init__()
        self.dim = dim
        self.num_global_tokens = num_global_tokens
        self.min_global_tokens = min_global_tokens
        self.p_min = p_min
        self.p_max = p_max

        # --- Global Branch ---
        # Learnable tokens that will query the image
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, dim))
        self.global_cross_attn = CrossAttention(dim, num_heads=8)

        # --- Spatial Branch ---
        # No fixed query vector. We use dynamic global mean.

    def forward(self, z):
        """
        z: (B, C, H, W) -> Encoder features
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, L, C)

        # ------------------------------------------------------------------
        # Part 1: Global Tokens (TiTok style)
        # ------------------------------------------------------------------
        g_tokens = self.global_tokens.expand(B, -1, -1)
        # Global tokens gather info from all spatial tokens
        g_out = self.global_cross_attn(g_tokens, z_flat)

        # Random Sub-selection for training robustness
        if self.training:
            m = torch.randint(self.min_global_tokens, self.num_global_tokens + 1, (1,)).item()
        else:
            m = self.num_global_tokens

        g_selected = g_out[:, :m, :]  # (B, m, C)

        # ------------------------------------------------------------------
        # Part 2: Spatial Tokens (Top-P with Dynamic Query)
        # ------------------------------------------------------------------
        # Calculate Importance Query: Mean of currently selected global tokens
        # The logic: "What is important?" -> "What relates to the global semantic gist."
        global_summary = g_selected.mean(dim=1)  # (B, C)

        # Dot Product Importance
        # (B, L, C) @ (B, C, 1) -> (B, L, 1)
        scores = torch.bmm(z_flat, global_summary.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Logic: We want to find tokens that are *most* relevant.
        # Standard Dot Product: High value = Similar.
        # Requirement: "Negate then Softmax" (per user prompt) implies selecting distinct/complementary features
        # OR usually implies Distance. Assuming prompt requirement is strict:
        scores = -scores

        probs = F.softmax(scores, dim=-1)  # (B, L)

        # Determine P threshold
        if self.training:
            target_p = torch.empty(1).uniform_(self.p_min, self.p_max).item()
        else:
            target_p = 0.95

        # Sort probabilities to perform Nucleus Sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Mask: Keep tokens until cumulative prob > target_p
        mask_sorted = cumulative_probs <= target_p
        # Ensure at least one token is kept
        mask_sorted[:, 0] = True

        # Gather the actual features using the sorted indices
        # We expand sorted_indices to match C dimension
        z_sorted = torch.gather(z_flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, C))

        # ------------------------------------------------------------------
        # Part 3: Concatenation for Decoder
        # ------------------------------------------------------------------
        # We need to construct a mask for the CrossAttention in the decoder.
        # In my CrossAttention impl, True = Ignore/Pad.

        # Global tokens are always kept -> False (don't ignore)
        mask_global = torch.zeros((B, m), dtype=torch.bool, device=z.device)

        # Spatial tokens: mask_sorted is True for Keep. So we invert it for the Padding Mask.
        mask_spatial = ~mask_sorted

        # Concat Features
        context = torch.cat([g_selected, z_sorted], dim=1)  # (B, m + L, C)

        # Concat Masks
        padding_mask = torch.cat([mask_global, mask_spatial], dim=1)

        return context, padding_mask, target_p, m


# ==============================================================================
# 4. Transformer Bridge (The Unpacker)
# ==============================================================================

class TransformerDecoderBlock(nn.Module):
    """
    Decoder Layer: Norm -> Cross -> Norm -> Self -> Norm -> FFN
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, context, padding_mask=None):
        # 1. Cross Attention (Query=Grid, Key/Value=Context)
        # Inject info from compressed representation to the grid
        x = x + self.cross_attn(self.norm1(x), context, mask=padding_mask)

        # 2. Self Attention (Spatial reasoning on the grid)
        x = x + self.self_attn(self.norm2(x))

        # 3. FFN
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerBridge(nn.Module):
    """
    Deep Transformer stack acting as the bridge between Compressed Tokens and CNN Decoder.
    """

    def __init__(self, dim, depth=6, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context, padding_mask=None):
        for block in self.blocks:
            x = block(x, context, padding_mask)
        x = self.norm(x)
        return x


# ==============================================================================
# 5. CNN Components (ResNet)
# ==============================================================================

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
        h = self.norm1(x)
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


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # VQGAN downsample: pad + conv stride 2
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ==============================================================================
# 6. Main Classes: Encoder, Decoder, ContinuousModel
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type='group', dropout=0.0, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        self.conv_blocks = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block_out = ch * ch_mult[i_level]
            res_block = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
            conv_block = nn.Module()
            conv_block.res = res_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in)
            self.conv_blocks.append(conv_block)

        # Mid Block
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
            # Note: No attention here in encoder, pure CNN for speed/texture
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
        ])

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.conv_blocks:
            for res in block.res: h = res(h)
            if hasattr(block, 'downsample'): h = block.downsample(h)

        for m in self.mid: h = m(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type="group", dropout=0.0, out_channels=3,
                 bridge_depth=6):
        super().__init__()
        self.z_channels = z_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # --- 1. Learnable Grid Initialization ---
        # Fixed spatial size for the bottleneck (16x16)
        self.grid_h = 16
        self.grid_w = 16
        # Learnable positional queries
        self.grid_embed = nn.Parameter(torch.randn(1, self.grid_h * self.grid_w, z_channels))

        # --- 2. Transformer Bridge (Unpacker) ---
        # Decoupled deep transformer stack
        self.bridge = TransformerBridge(
            dim=z_channels,
            depth=bridge_depth,
            num_heads=8,
            mlp_ratio=4.
        )

        block_in = ch * ch_mult[self.num_resolutions - 1]

        # Conv Adapter
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # --- 3. Standard ResNet Renderer ---
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type),
        ])

        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            res_block = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
            conv_block = nn.Module()
            conv_block.res = res_block
            if i_level != 0:
                conv_block.upsample = Upsample(block_in)
            self.conv_blocks.append(conv_block)

        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, context, padding_mask=None):
        """
        context: (B, m+L, C) - Compressed Tokens
        padding_mask: (B, m+L) - True = Ignore
        """
        B = context.shape[0]

        # 1. Initialize Grid Queries
        x = self.grid_embed.expand(B, -1, -1)  # (B, 256, C)

        # 2. Transformer Bridge: Unpack context into grid
        # Queries (Grid) attend to Keys/Values (Context)
        x = self.bridge(x, context, padding_mask)  # (B, 256, C)

        # 3. Reshape to Spatial Image format for CNN
        x = x.permute(0, 2, 1).reshape(B, self.z_channels, self.grid_h, self.grid_w)

        # 4. Standard CNN Decoding
        x = self.conv_in(x)
        for m in self.mid: x = m(x)

        for i_level, block in enumerate(self.conv_blocks):
            for res in block.res: x = res(x)
            if hasattr(block, 'upsample'): x = block.upsample(x)

        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x


class ContinuousModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # 1. Encoder (CNN)
        self.encoder = Encoder(
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p
        )

        # 2. Compressor (Continuous)
        self.compressor = ContinuousCompressor(
            dim=config.z_channels,
            num_global_tokens=config.num_global_tokens,
            min_global_tokens=config.min_global_tokens,
            p_min=config.p_min,
            p_max=config.p_max
        )

        # 3. Decoder (Transformer Bridge + CNN)
        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
            bridge_depth=config.bridge_depth
        )

    def forward(self, x):
        """
        x: (B, 3, 256, 256)
        """
        # Encode -> (B, 256, 16, 16)
        z = self.encoder(x)

        # Compress -> Context (B, VarLen, 256) + Mask
        context, padding_mask, p_val, m_val = self.compressor(z)

        # Decode -> Recons (B, 3, 256, 256)
        dec = self.decoder(context, padding_mask)

        # Return reconstruction and stats for logging
        return dec, (p_val, m_val)


def Continuous_Base(**kwargs):
    # Factory function for standard config
    config = ModelArgs(
        encoder_ch_mult=[1, 2, 2, 4],  # Results in 256 down to 16
        decoder_ch_mult=[1, 2, 2, 4],
        z_channels=256,
        **kwargs
    )
    return ContinuousModel(config)


Continuous_models = {'Continuous-Base': Continuous_Base}
