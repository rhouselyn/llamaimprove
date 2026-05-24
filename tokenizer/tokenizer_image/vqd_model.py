# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim,
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
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
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        # Stage 控制
        self.detach_quant = False                       # 进入 stage2 后由外部设置
        self.stage2_fixed = False                       # ### NEW: 是否已固定 top-1/4 子集
        self.register_buffer("code_counts", torch.zeros(n_e, dtype=torch.long))  # ### NEW: 每 epoch 的累计计数
        self._usage_subset: Optional[torch.Tensor] = None                        # ### NEW: 上一 batch 的候选集合（索引）
        self.fixed_topk_indices: Optional[torch.Tensor] = None                   # ### NEW: stage2 的固定子集

        # Embedding
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1 / math.sqrt(self.n_e), 1 / math.sqrt(self.n_e))
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)

        # Usage 统计（原有）
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536, dtype=torch.long)))

    # ---------- NEW: epoch 钩子 ----------
    @torch.no_grad()
    def on_new_epoch(self):
        """每个 epoch 开始时调用，清零累计计数并重置 usage 子集。"""
        self.code_counts.zero_()
        self._usage_subset = None

    # ---------- NEW: 固定 stage2 的 top-1/4 codebook 子集 ----------
    @torch.no_grad()
    def fix_topk_codebook(self):
        """在进入 stage2 前调用：从累计的 code_counts 中选 top-1/4，之后量化仅在此子集上进行。"""
        k = max(1, self.n_e // 4)
        # 如果计数全 0，torch.topk 仍然会返回前 k 个索引（按默认顺序），可接受
        topk = torch.topk(self.code_counts.to(torch.float32), k, largest=True).indices
        self.fixed_topk_indices = topk.clone()
        self.stage2_fixed = True

    # ---------- NEW: 在每个 batch 结束时更新下一批的 usage 子集 ----------
    @torch.no_grad()
    def _update_usage_subset_after_batch(self):
        """基于累计计数选出 top-1/4，作为下一 batch 的 usage 统计集合。"""
        if self.stage2_fixed:
            # stage2 不再维护 usage 子集
            self._usage_subset = None
            return
        k = max(1, self.n_e // 4)
        self._usage_subset = torch.topk(self.code_counts.to(torch.float32), k, largest=True).indices

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 归一化
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        # 计算到 embedding 的距离（只影响 codebook，不影响 z）
        z_flattened_det = z_flattened.detach()

        # --- 支持 stage2 固定子集：只在 fixed_topk_indices 上计算 ---
        if self.stage2_fixed and self.fixed_topk_indices is not None:
            embedding_sub = embedding.index_select(0, self.fixed_topk_indices)  # (k, e_dim)
            # d_sub shape: (BHW, k)
            d_sub = torch.sum(z_flattened_det ** 2, dim=1, keepdim=True) \
                    + torch.sum(embedding_sub ** 2, dim=1) \
                    - 2 * torch.einsum('bd,dn->bn', z_flattened_det,
                                       torch.einsum('nd->dn', embedding_sub))
            min_encoding_indices_local = torch.argmin(d_sub, dim=1)  # in [0, k)
            min_encoding_indices = self.fixed_topk_indices[min_encoding_indices_local]  # map to global index
            # 供 code 熵损失使用
            affinity = -d_sub
        else:
            # 全 codebook 参与最近邻
            d = torch.sum(z_flattened_det ** 2, dim=1, keepdim=True) \
                + torch.sum(embedding ** 2, dim=1) \
                - 2 * torch.einsum('bd,dn->bn', z_flattened_det,
                                   torch.einsum('nd->dn', embedding))
            min_encoding_indices = torch.argmin(d, dim=1)
            affinity = -d  # for entropy over assignments

        # 取量化向量
        z_q = embedding[min_encoding_indices].view(z.shape)

        # 统计 usage（两种模式）
        perplexity = None
        min_encodings = None
        codebook_usage = 0.0

        # --- NEW: stage1 的 usage 用上一批的 top-1/4 子集；stage2 恢复旧逻辑 ---
        if not self.stage2_fixed:
            if self._usage_subset is not None and self._usage_subset.numel() > 0:
                # unique 使用的 codes
                used_codes = torch.unique(min_encoding_indices)
                # 交集计数
                # 为避免 O(n*m) 操作，建一个 mask
                mask = torch.zeros(self.n_e, dtype=torch.bool, device=used_codes.device)
                mask[self._usage_subset] = True
                in_subset = mask[used_codes]
                used_in_subset = int(in_subset.sum().item())
                codebook_usage = used_in_subset / float(self._usage_subset.numel())
            else:
                codebook_usage = 0.0
        else:
            # 恢复原来的滑动窗口 unique 比率
            if self.show_usage:
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / float(self.n_e)

        # 更新累计计数（epoch 内）
        with torch.no_grad():
            binc = torch.bincount(min_encoding_indices, minlength=self.n_e)
            self.code_counts.add_(binc.to(self.code_counts.dtype))
            # 为下一 batch 准备 usage 子集（仅 stage1）
            if not self.stage2_fixed:
                self._update_usage_subset_after_batch()

        # Loss 项
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        if self.detach_quant:
            # Stage2：与原逻辑一致，量化向量不反传，且 codebook 相关损失置 0
            z_q = z_q.detach()
            if self.training:
                vq_loss = 0.0
                commit_loss = 0.0
                entropy_loss = 0.0
        else:
            # Stage1：code 侧（只更新 codebook）——使用 assignment 熵 + embedding 均匀（repulsion）
            if self.training:
                # 对 code 的熵（只依赖 z_detached）——不影响 z
                vq_loss = compute_entropy_loss(affinity)

                # 对 embedding 的均匀性（uniform/repulsion）——只更新 codebook
                commit_loss = compute_pairwise_repulsion_loss(self.embedding.weight)

                # 对 z 的“发散/均匀性”——更新 z（配合 AE）
                entropy_loss = compute_pairwise_repulsion_loss(z_flattened)

            # 不使用 STE，保持 AE 路径：z_q 走纯 passthrough
            z_q = z  # 与你原始代码保持一致

        # reshape 回 (B, C, H, W)
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


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
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_)  # b,c,hw
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def nonlinearity(x):
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
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


def compute_pairwise_repulsion_loss(z: torch.Tensor, tau: float = 0.1, variant: str = 'basic') -> torch.Tensor:
    """
    计算 Pairwise Repulsion Loss，鼓励向量在单位超球面上均匀分布。
    """
    z_norm = F.normalize(z, p=2, dim=-1)
    num_vectors = z_norm.shape[0]
    sim_matrix = torch.mm(z_norm, z_norm.t())
    eye = torch.eye(num_vectors, device=sim_matrix.device)
    sim_matrix = sim_matrix - eye * sim_matrix.diag()

    if variant == 'basic':
        loss = torch.mean(sim_matrix)
    elif variant == 'relu':
        loss = torch.mean(F.relu(sim_matrix))
    elif variant == 'exponential':
        loss = torch.mean(torch.exp(sim_matrix / tau)) - 1
    elif variant == 'softplus':
        loss = torch.mean(torch.log(1 + torch.exp(sim_matrix / tau)))
    else:
        raise ValueError(f"Unknown variant: {variant}. Supported: 'basic', 'relu', 'exponential', 'softplus'.")
    return loss


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))


def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}
