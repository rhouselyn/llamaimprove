# Modified from original CSQ and VQ train.py
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from tqdm import tqdm
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import argparse
import os
import time
from glob import glob
from copy import deepcopy
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.csq_loss import VQLoss  # 原CSQ损失

from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


@dataclass
class ModelArgs:
    bits_dim: int = 14
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = False
    codebook_show_usage: bool = True
    patch_size: int = 16
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])


class L2NormalizeParam(nn.Module):
    def forward(self, weight):
        return F.normalize(weight, p=2, dim=1)


class ZCAWhitening(nn.Module):  # 修改：PCA -> ZCA，用于临时积累stats
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.sum_x = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.sum_xxT = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)
        self.count = 0
        self.whitening_matrix = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def update_stats(self, x):
        x = x.view(-1, self.dim)
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += torch.mm(x.t(), x)
        self.count += x.size(0)

    def compute_whitening_matrix(self):
        self.mean.copy_(self.sum_x / self.count)  # 假设有mean更新，类似原代码
        cov = (self.sum_xxT / self.count) - torch.outer(self.mean, self.mean)
        cov += self.eps * torch.eye(self.dim, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
        diag = torch.diag(1.0 / sqrt_eigenvalues)
        new_matrix = eigenvectors @ diag @ eigenvectors.t()  # 修改：ZCA = U @ D^{-1/2} @ U.T
        self.whitening_matrix.copy_(new_matrix)  # 用.copy_()更新

    def forward(self, x):
        x_flat = x.view(-1, self.dim)
        x_centered = x_flat - self.mean
        x_whitened = torch.mm(x_centered, self.whitening_matrix)  # 修改：移除.t()
        return x_whitened.view_as(x)


class CSQEncoder(nn.Module):  # 修改：移除self.zca，linear bias=True，添加fuse_pca
    def __init__(self, patch_size=16, encoding_dim=14, learnable=False):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_dim = encoding_dim
        self.conv = nn.Conv2d(3, 1, kernel_size=4, stride=4, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.norm = nn.LayerNorm(16)
        self.linear = nn.Linear(16, encoding_dim, bias=True)  # 修改：bias=True，初始bias=0
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # 初始bias=0，等价原bias=False
        self.fused = False  # 标志是否融合PCA
        self.learnable = learnable
        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def fuse_pca(self, whitening_matrix, mean):
        # 融合推导：new_weight = whitening_matrix.t() @ self.linear.weight
        # new_bias = - (mean @ whitening_matrix)
        new_weight = whitening_matrix.t() @ self.linear.weight
        new_bias = - (mean @ whitening_matrix)
        self.linear.weight.data.copy_(new_weight)
        self.linear.bias.data.copy_(new_bias)
        self.fused = True

    def forward(self, x, train_mode=False):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2).reshape(-1, C, self.patch_size, self.patch_size)
        conv_out = self.conv(patches).squeeze(1).flatten(1)
        norm_out = self.norm(conv_out)
        linear_out = self.linear(norm_out)
        # 修改：移除zca调用，直接返回linear_out（融合后即whitened）
        return linear_out.view(B, -1, self.encoding_dim).permute(0, 2, 1).view(B, self.encoding_dim, H//self.patch_size, W//self.patch_size)


class StdEncoder(nn.Module):
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


class VectorQuantizer(nn.Module):  # CSQ quantize (from csq_model, no loss)
    def __init__(self, n_e, e_dim, bits_dim, l2_norm=True, show_usage=True):  # 新增bits_dim参数
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.bits_dim = bits_dim  # 新增：sign bits维度
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(self.n_e * 4))  # 修改：扩大buffer，避免overflow

    def forward(self, z):
        B, C, H, W = z.shape  # [B, bits_dim=14, H, W]
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.bits_dim)  # [-1, 14]
        # 计算sign index，使用bits_dim
        bits = (z_flattened >= 0).long()
        powers = 2 ** torch.arange(self.bits_dim, device=z.device)
        min_encoding_indices = torch.clamp((bits * powers).sum(dim=-1), 0, self.n_e - 1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1) if self.l2_norm else self.embedding.weight
        z_q = embedding[min_encoding_indices].view(B, H, W, self.e_dim)  # [B, H, W, 8]，使用e_dim重塑
        codebook_usage = self.compute_usage(min_encoding_indices) if self.show_usage and self.training else 0
        # 移除straight-through，因为dim不匹配（14 != 8），且encoder frozen无需梯度
        z_q = torch.einsum('b h w c -> b c h w', z_q)
        return z_q, (0, 0, 0, codebook_usage), (None, None, min_encoding_indices)

    def compute_usage(self, min_encoding_indices):
        cur_len = min_encoding_indices.shape[0]
        self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
        self.codebook_used[-cur_len:] = min_encoding_indices
        return len(torch.unique(self.codebook_used)) / self.n_e


class IntegratedModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.csq_encoder = CSQEncoder(patch_size=config.patch_size, encoding_dim=config.bits_dim)  # encoding_dim = bits_dim=14
        self.std_encoder = StdEncoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.fc_classifier = nn.Linear(config.codebook_embed_dim, config.codebook_size)
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, config.bits_dim, config.codebook_l2_norm, config.codebook_show_usage)  # 传入bits_dim
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        self.decoder = Decoder(z_channels=config.z_channels, ch_mult=config.decoder_ch_mult, dropout=config.dropout_p)
        if self.quantize.l2_norm:
            parametrize.register_parametrization(self.fc_classifier, "weight", L2NormalizeParam())
        self.register_buffer("initialized", torch.zeros(config.codebook_size, dtype=torch.bool))  # 新增：追踪codebook初始化

    def accumulate_and_fuse_pca(self, loader, device, rank, world_size, checkpoint_dir, logger):
        dim = self.csq_encoder.encoding_dim
        temp_pca = ZCAWhitening(dim).to(device)  # 修改：PCA -> ZCA
        progress = tqdm(loader, desc="Accumulating ZCA stats") if rank == 0 else loader
        for x, _ in progress:
            x = x.to(device)
            with torch.no_grad():
                z = self.csq_encoder(x)  # [B,14,H/16,W/16]
                linear_out = z.permute(0, 2, 3, 1).reshape(-1, dim)  # [-1,14]
                temp_pca.update_stats(linear_out)

        dist.all_reduce(temp_pca.sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(temp_pca.sum_xxT, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([temp_pca.count], device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        temp_pca.count = int(count_tensor.item())

        if rank == 0:
            temp_pca.compute_whitening_matrix()
            self.csq_encoder.fuse_pca(temp_pca.whitening_matrix, temp_pca.mean)
            logger.info("Computed and fused ZCA into CSQEncoder linear layer.")

        dist.barrier()
        for param in self.csq_encoder.parameters():
            dist.broadcast(param.data, src=0)

        if rank == 0 and checkpoint_dir is not None:  # 新增检查：避免非rank0或None时执行保存
            pca_save = {"sum_x": temp_pca.sum_x.cpu(), "sum_xxT": temp_pca.sum_xxT.cpu(), "count": temp_pca.count,
                        "whitening_matrix": temp_pca.whitening_matrix.cpu(), "mean": temp_pca.mean.cpu()}
            torch.save(pca_save, f"{checkpoint_dir}/pca.pt")
            logger.info(f"Saved ZCA checkpoint to {checkpoint_dir}/pca.pt")

    def get_csq_index(self, x):
        z = self.csq_encoder(x)  # [B, 14, 16, 16]
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, 14)
        bits = (z_flattened >= 0).long()
        powers = 2 ** torch.arange(14, device=x.device)
        index = torch.clamp((bits * powers).sum(dim=-1), 0, self.quantize.n_e - 1)
        return index.view(x.shape[0], -1)  # [B, 256]

    def decode_forward(self, x):
        z = self.csq_encoder(x)  # [B, 14, 16, 16]
        z_q, diff, (_, _, indices) = self.quantize(z)  # z_q [B, 8, 16, 16], diff=(0,0,0,usage)
        z_q = self.post_quant_conv(z_q)
        recons = self.decoder(z_q)
        return recons, diff, indices


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
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def nonlinearity(x):
    # swish
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
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def joint_train_step(model, vq_loss, index_loss_fn, x, joint_optimizer, optimizer_disc, scaler, scaler_disc, args, train_steps, device, logger):
    B, _, H, W = x.shape  # 假设 H=W=256
    patch_h, patch_w = H // 16, W // 16  # 16x16 patches

    h = model.module.std_encoder(x)  # [B, 256, 16, 16]
    quant_h = model.module.quant_conv(h)  # [B, 8, 16, 16]
    if model.module.quantize.l2_norm:
        quant_h = F.normalize(quant_h, p=2, dim=1)  # 新增：norm for consistency

    flattened_h = quant_h.permute(0, 2, 3, 1).reshape(-1, model.module.quantize.e_dim)  # [-1, 8]

    if train_steps < args.ce_start:
        # 纯AE模式
        z_q = quant_h
        index_loss = 0.0
        codebook_usage = 0.0
        min_encoding_indices = None  # 无需
    else:
        index = model.module.get_csq_index(x)  # [B, 256]
        min_encoding_indices = index.view(-1)
        # 初始化阶段：分布式累积新code
        local_sum = torch.zeros(model.module.quantize.n_e, model.module.quantize.e_dim, device=device)
        local_count = torch.zeros(model.module.quantize.n_e, device=device)
        unique_indices = torch.unique(min_encoding_indices)
        for idx in unique_indices:
            mask = min_encoding_indices == idx
            local_sum[idx] += flattened_h[mask].sum(0)
            local_count[idx] += mask.sum()
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        with torch.no_grad():
            for idx in unique_indices:
                if not model.module.initialized[idx] and local_count[idx] > 0:
                    mean_h = local_sum[idx] / local_count[idx]
                    if model.module.quantize.l2_norm:
                        mean_h = F.normalize(mean_h, p=2, dim=0)
                    model.module.fc_classifier.weight.data[idx] = mean_h
                    model.module.initialized[idx] = True
        # 检查是否全部初始化
        all_init_tensor = torch.tensor([model.module.initialized.all()], device=device)
        dist.all_reduce(all_init_tensor, op=dist.ReduceOp.MIN)
        all_init = bool(all_init_tensor.item())
        if all_init:
            logits = model.module.fc_classifier(flattened_h)  # [-1, 16384]
            index_loss = index_loss_fn(logits, min_encoding_indices)
            true_indices = min_encoding_indices
            true_codes = model.module.fc_classifier.weight[true_indices]  # [-1, 8]
            z_q = true_codes.view(B, patch_h, patch_w, model.module.quantize.e_dim).permute(0, 3, 1, 2)  # [B, 8, 16, 16]
            codebook_usage = model.module.quantize.compute_usage(min_encoding_indices) if model.module.quantize.show_usage and model.training else 0
        else:
            z_q = quant_h
            index_loss = 0.0
            codebook_usage = 0.0

    post_quant_z = model.module.post_quant_conv(z_q)
    recons = model.module.decoder(post_quant_z)

    diff_zero = (0, 0, 0, codebook_usage)
    loss_gen = vq_loss(diff_zero, x, recons, optimizer_idx=0, global_step=train_steps+1, last_layer=model.module.decoder.last_layer, logger=logger, log_every=args.log_every)
    total_gen_loss = loss_gen + index_loss

    joint_optimizer.zero_grad()
    scaler.scale(total_gen_loss).backward()
    if args.max_grad_norm != 0.0:
        scaler.unscale_(joint_optimizer)
        torch.nn.utils.clip_grad_norm_(chain(model.module.std_encoder.parameters(), model.module.quant_conv.parameters(),
                                             model.module.fc_classifier.parameters(), model.module.post_quant_conv.parameters(),
                                             model.module.decoder.parameters()), args.max_grad_norm)
    scaler.step(joint_optimizer)
    scaler.update()

    loss_disc = vq_loss(diff_zero, x, recons, optimizer_idx=1, global_step=train_steps+1, logger=logger, log_every=args.log_every)
    optimizer_disc.zero_grad()
    scaler_disc.scale(loss_disc).backward()
    if args.max_grad_norm != 0.0:
        scaler_disc.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
    scaler_disc.step(optimizer_disc)
    scaler_disc.update()

    # 修改：返回三个独立loss
    return loss_gen.item(), index_loss, loss_disc.item()


def main(args):
    assert torch.cuda.is_available(), "Training requires GPU."
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-Integrated_Model"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None)  # dummy logger
        checkpoint_dir = None  # 新增：为非rank0设置None，避免未定义错误

    logger.info(f"{args}")

    shared_seed = args.global_seed
    torch.manual_seed(shared_seed)
    torch.cuda.manual_seed_all(shared_seed)

    model = IntegratedModel(ModelArgs())  # 创建模型
    model = model.to(device)

    if rank == 0:
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    unique_seed = args.global_seed * world_size + rank
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)

    if args.pca_ckpt:
        if rank == 0:
            pca_checkpoint = torch.load(args.pca_ckpt, map_location="cpu")
            whitening_matrix = pca_checkpoint["whitening_matrix"].to(device)
            mean = pca_checkpoint["mean"].to(device)
            model.csq_encoder.fuse_pca(whitening_matrix, mean)
            logger.info("Loaded and fused ZCA checkpoint into CSQEncoder linear layer.")
        dist.barrier()
        for param in model.csq_encoder.parameters():
            dist.broadcast(param.data, src=0)
        logger.info("Broadcasted fused CSQEncoder parameters.")
    else:
        logger.info("Accumulating ZCA stats in distributed mode with no augmentation...")
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = build_dataset(args, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=args.global_batch_size // world_size, sampler=sampler,
                            num_workers=args.num_workers, drop_last=False)
        model.accumulate_and_fuse_pca(loader, device, rank, world_size, checkpoint_dir, logger)

    index_loss_fn = nn.CrossEntropyLoss()

    vq_loss = None

    if rank == 0:
        vq_loss = VQLoss(disc_start=args.disc_start, disc_weight=args.disc_weight, disc_type=args.disc_type,
                         disc_loss=args.disc_loss,
                         gen_adv_loss=args.gen_loss, image_size=args.image_size,
                         perceptual_weight=args.perceptual_weight,
                         reconstruction_weight=args.reconstruction_weight,
                         reconstruction_loss=args.reconstruction_loss).to(device)
        logger.info("Rank 0: VQLoss initialized and VGG downloaded if needed.")

    dist.barrier()

    if rank != 0:
        vq_loss = VQLoss(disc_start=args.disc_start, disc_weight=args.disc_weight, disc_type=args.disc_type,
                         disc_loss=args.disc_loss,
                         gen_adv_loss=args.gen_loss, image_size=args.image_size,
                         perceptual_weight=args.perceptual_weight,
                         reconstruction_weight=args.reconstruction_weight,
                         reconstruction_loss=args.reconstruction_loss).to(device)
        logger.info(f"Rank {rank}: VQLoss initialized using existing cache.")

    dist.barrier()

    if rank == 0:
        vq_loss_state_dict = vq_loss.state_dict()
        broadcast_list = list(vq_loss_state_dict.values())
    else:
        broadcast_list = [torch.empty_like(param) for param in vq_loss.state_dict().values()]

    dist.broadcast_object_list(broadcast_list, src=0)

    state_dict_keys = list(vq_loss.state_dict().keys())
    updated_state_dict = {key: broadcast_list[i] for i, key in enumerate(state_dict_keys)}
    vq_loss.load_state_dict(updated_state_dict)

    joint_optimizer = torch.optim.Adam(chain(model.std_encoder.parameters(), model.quant_conv.parameters(),
                                             model.fc_classifier.parameters(), model.post_quant_conv.parameters(),
                                             model.decoder.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    model = DDP(model, device_ids=[device])
    vq_loss = DDP(vq_loss, device_ids=[device])

    train_steps = 0
    if args.resume:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.module.load_state_dict(checkpoint["model"], strict=True)
            joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            vq_loss.module.discriminator.load_state_dict(checkpoint["discriminator"])
            train_steps = checkpoint["steps"]

        dist.barrier()

        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        for param in vq_loss.parameters():
            dist.broadcast(param.data, src=0)

        if rank == 0:
            logger.info("Checkpoint loaded and broadcasted successfully.")

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == 'fp16')
    scaler_disc = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == 'fp16')

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.global_batch_size // world_size, sampler=sampler, num_workers=args.num_workers, drop_last=True)

    log_steps = 0
    running_gen_loss = 0.0  # 新增
    running_ce_loss = 0.0   # 新增
    running_disc_loss = 0.0 # 新增
    running_total_loss = 0.0  # 可选：保留总loss
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(dtype=ptdtype):
                gen_l, ce_l, disc_l = joint_train_step(model, vq_loss, index_loss_fn, x, joint_optimizer, optimizer_disc, scaler, scaler_disc, args, train_steps, device, logger)

            # 新增：累加三个loss
            running_gen_loss += gen_l
            running_ce_loss += ce_l
            running_disc_loss += disc_l
            running_total_loss += (gen_l + ce_l + disc_l)  # 可选总loss
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # 新增：分布式平均三个loss
                avg_gen = torch.tensor(running_gen_loss / log_steps, device=device)
                avg_ce = torch.tensor(running_ce_loss / log_steps, device=device)
                avg_disc = torch.tensor(running_disc_loss / log_steps, device=device)
                dist.all_reduce(avg_gen, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_ce, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_disc, op=dist.ReduceOp.SUM)
                avg_gen = avg_gen.item() / world_size
                avg_ce = avg_ce.item() / world_size
                avg_disc = avg_disc.item() / world_size

                # 修改log：新增CE
                logger.info(f"(step={train_steps:07d}) Gen: {avg_gen:.4f}, CE: {avg_ce:.4f}, Disc: {avg_disc:.4f}, Steps/Sec: {steps_per_sec:.2f}")

                # 重置
                running_gen_loss = 0.0
                running_ce_loss = 0.0
                running_disc_loss = 0.0
                running_total_loss = 0.0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "joint_optimizer": joint_optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--results-dir", type=str, default="/mnt/afs/zhengmingkai/raozf/llamagen/results_integrated")
    parser.add_argument("--pca-ckpt", type=str, default=None, help="Path to precomputed PCA checkpoint")
    parser.add_argument("--ce_start", type=int, default=1000)  # 新增：CE loss介入step
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    # 其他原参数如disc_start, disc_weight 等同原CSQ parser
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--disc-start", type=int, default=100000)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-type", type=str, default='patchgan')
    parser.add_argument("--disc-loss", type=str, default='hinge')
    parser.add_argument("--gen-loss", type=str, default='hinge')
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default='l2')
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='aoss')
    args = parser.parse_args()
    main(args)
