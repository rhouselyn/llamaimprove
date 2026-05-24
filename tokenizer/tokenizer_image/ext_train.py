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
# 假设您有ZCAWhitening, Normalize, nonlinearity 等从csq_model.py
# 这里简化定义必要类，实际复制从csq_model.py和vq_model.py

from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    bits_dim: int = 14
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    patch_size: int = 16
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])

class ZCAWhitening(nn.Module):
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
        self.mean.copy_(self.sum_x / self.count)
        cov = (self.sum_xxT / self.count) - torch.outer(self.mean, self.mean)
        cov += self.eps * torch.eye(self.dim, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
        self.whitening_matrix = torch.mm(eigenvectors, torch.diag(1.0 / sqrt_eigenvalues)) @ eigenvectors.t()

    def forward(self, x):
        x_flat = x.view(-1, self.dim)
        x_centered = x_flat - self.mean
        x_whitened = torch.mm(x_centered, self.whitening_matrix.t())
        return x_whitened.view_as(x)

class CSQEncoder(nn.Module):  # 简化CSQ encoder
    def __init__(self, patch_size=16, encoding_dim=14, learnable=False):
        super().__init__()
        self.patch_size = patch_size
        self.encoding_dim = encoding_dim
        self.conv = nn.Conv2d(3, 1, kernel_size=2, stride=2, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.norm = nn.LayerNorm(64)
        self.linear = nn.Linear(64, encoding_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.zca = ZCAWhitening(encoding_dim)
        self.learnable = learnable
        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, train_mode=False):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2).reshape(-1, C, self.patch_size, self.patch_size)
        conv_out = self.conv(patches).squeeze(1).flatten(1)
        norm_out = self.norm(conv_out)
        linear_out = self.linear(norm_out)
        if train_mode:
            self.zca.update_stats(linear_out)
        zca_out = self.zca(linear_out)
        return zca_out.view(B, -1, self.encoding_dim).permute(0, 2, 1).view(B, self.encoding_dim, H//self.patch_size, W//self.patch_size)


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
            # downsample
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
            self.register_buffer("codebook_used", torch.zeros(65536))

    def forward(self, z):
        B, C, H, W = z.shape  # [B, bits_dim=14, H, W]
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.bits_dim)  # [-1, 14]
        if self.l2_norm:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        # 计算sign index，使用bits_dim
        bits = (z_flattened >= 0).long()
        powers = 2 ** torch.arange(self.bits_dim, device=z.device)
        min_encoding_indices = torch.clamp((bits * powers).sum(dim=-1), 0, self.n_e - 1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1) if self.l2_norm else self.embedding.weight
        z_q = embedding[min_encoding_indices].view(B, H, W, self.e_dim)  # [B, H, W, 8]，使用e_dim重塑
        codebook_usage = 0
        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
        # 移除straight-through，因为dim不匹配（14 != 8），且encoder frozen无需梯度
        z_q = torch.einsum('b h w c -> b c h w', z_q)
        return z_q, (0, 0, 0, codebook_usage), (None, None, min_encoding_indices)

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

    def get_csq_index(self, x):
        z = self.csq_encoder(x)  # [B, 14, 16, 16]
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, 14)
        bits = (z_flattened >= 0).long()
        powers = 2 ** torch.arange(14, device=x.device)
        index = torch.clamp((bits * powers).sum(dim=-1), 0, self.quantize.n_e - 1)
        return index.view(x.shape[0], -1)  # [B, 256]

    def classify_forward(self, x):
        h = self.std_encoder(x)  # [B, 256, 16, 16]
        h = self.quant_conv(h)  # [B, 8, 16, 16]
        h = h.permute(0, 2, 3, 1).reshape(-1, 8)  # [B*256, 8]
        logits = self.fc_classifier(h)  # [B*256, 16384]
        return logits

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


def main(args):
    assert torch.cuda.is_available(), "Training requires GPU."
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # 【修改1: 立即创建 logger 和 experiment_dir】
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

    logger.info(f"{args}")  # 现在安全

    # 【修改2: 先共享种子 init 模型】
    shared_seed = args.global_seed
    torch.manual_seed(shared_seed)
    torch.cuda.manual_seed_all(shared_seed)

    model = IntegratedModel(ModelArgs())  # 创建模型
    model = model.to(device)

    # 【修改3: 参数日志用 if rank==0 包裹】
    if rank == 0:
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 现在设置独特种子...
    unique_seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)

    # Handle ZCA（积累用独特seed，数据多样，但all_reduce同步）
    if args.zca_ckpt:
        if rank == 0:
            zca_checkpoint = torch.load(args.zca_ckpt, map_location="cpu")
        dist.barrier()
        # Broadcast ZCA params
        zca = model.csq_encoder.zca
        if rank == 0:
            sum_x = zca_checkpoint["sum_x"].to(device)
            sum_xxT = zca_checkpoint["sum_xxT"].to(device)
            count = zca_checkpoint["count"]
            whitening_matrix = zca_checkpoint["whitening_matrix"].to(device)
            mean = zca_checkpoint["mean"].to(device)
        else:
            sum_x = torch.zeros(zca.dim, device=device)
            sum_xxT = torch.zeros(zca.dim, zca.dim, device=device)
            whitening_matrix = torch.zeros(zca.dim, zca.dim, device=device)
            mean = torch.zeros(zca.dim, device=device)
            count = 0
        dist.broadcast(sum_x, src=0)
        dist.broadcast(sum_xxT, src=0)
        count_tensor = torch.tensor([count], device=device)
        dist.broadcast(count_tensor, src=0)
        dist.broadcast(whitening_matrix, src=0)
        dist.broadcast(mean, src=0)
        zca.sum_x.copy_(sum_x)
        zca.sum_xxT.copy_(sum_xxT)
        zca.count = int(count_tensor.item())
        zca.whitening_matrix.copy_(whitening_matrix)
        zca.mean.copy_(mean)
        logger.info("Loaded and broadcasted ZCA checkpoint.")
    else:
        logger.info("Accumulating ZCA stats in distributed mode with no augmentation...")
        # 使用无增强的transform：Resize + CenterCrop + ToTensor + Normalize
        transform = transforms.Compose([
            transforms.Resize(args.image_size),  # 先缩放较小边
            transforms.CenterCrop(args.image_size),  # 中心裁剪到image_size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = build_dataset(args, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank,
                                     shuffle=False)  # shuffle=False for deterministic stats
        loader = DataLoader(dataset, batch_size=args.global_batch_size // dist.get_world_size(), sampler=sampler,
                            num_workers=args.num_workers, drop_last=False)

        zca = model.csq_encoder.zca
        # 所有ranks积累本地stats
        progress = tqdm(loader, desc="Accumulating ZCA stats") if rank == 0 else loader  # 只rank 0显示进度条
        for x, _ in progress:
            x = x.to(device)
            with torch.no_grad():
                _ = model.csq_encoder(x, train_mode=True)  # 更新本地sum_x, sum_xxT, count

        # 全局合并stats
        dist.all_reduce(zca.sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(zca.sum_xxT, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([zca.count], device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        zca.count = int(count_tensor.item())

        # 只rank 0计算矩阵
        if rank == 0:
            zca.compute_whitening_matrix()

        # 广播矩阵和mean给所有ranks
        dist.broadcast(zca.whitening_matrix, src=0)
        dist.broadcast(zca.mean, src=0)

        # 保存（只rank 0）
        if rank == 0:
            zca_save = {"sum_x": zca.sum_x.cpu(), "sum_xxT": zca.sum_xxT.cpu(), "count": zca.count,
                        "whitening_matrix": zca.whitening_matrix.cpu(), "mean": zca.mean.cpu()}
            torch.save(zca_save, f"{checkpoint_dir}/zca.pt")
            logger.info(f"Saved ZCA checkpoint to {checkpoint_dir}/zca.pt")

    index_loss_fn = nn.CrossEntropyLoss()

    vq_loss = None  # 先为空

    if rank == 0:
        # 只 rank 0 先创建 vq_loss（会触发 VGG 下载，如果 ~/.cache 中没有）
        vq_loss = VQLoss(disc_start=args.disc_start, disc_weight=args.disc_weight, disc_type=args.disc_type,
                         disc_loss=args.disc_loss,
                         gen_adv_loss=args.gen_loss, image_size=args.image_size,
                         perceptual_weight=args.perceptual_weight,
                         reconstruction_weight=args.reconstruction_weight,
                         reconstruction_loss=args.reconstruction_loss).to(device)
        logger.info("Rank 0: VQLoss initialized and VGG downloaded if needed.")

    dist.barrier()  # 同步！其他 rank 等待 rank 0 下载完成（缓存已更新）

    if rank != 0:
        # 其他 rank 现在创建 vq_loss（检查缓存，已存在，不会下载）
        vq_loss = VQLoss(disc_start=args.disc_start, disc_weight=args.disc_weight, disc_type=args.disc_type,
                         disc_loss=args.disc_loss,
                         gen_adv_loss=args.gen_loss, image_size=args.image_size,
                         perceptual_weight=args.perceptual_weight,
                         reconstruction_weight=args.reconstruction_weight,
                         reconstruction_loss=args.reconstruction_loss).to(device)
        logger.info(f"Rank {rank}: VQLoss initialized using existing cache.")

    dist.barrier()  # 再同步，确保所有 rank 都有 vq_loss

    # 现在广播 state_dict（所有 rank 都有 vq_loss）
    if rank == 0:
        vq_loss_state_dict = vq_loss.state_dict()
        broadcast_list = list(vq_loss_state_dict.values())
    else:
        broadcast_list = [torch.empty_like(param) for param in vq_loss.state_dict().values()]

    dist.broadcast_object_list(broadcast_list, src=0)

    # 所有 ranks 加载广播的 state_dict（确保权重一致，即使其他 rank 的 VGG 可能略有不同）
    state_dict_keys = list(vq_loss.state_dict().keys())
    updated_state_dict = {key: broadcast_list[i] for i, key in enumerate(state_dict_keys)}
    vq_loss.load_state_dict(updated_state_dict)

    # 优化器: 分阶段切换
    distill_optimizer = torch.optim.Adam([*model.std_encoder.parameters(), *model.quant_conv.parameters(), *model.fc_classifier.parameters()], lr=args.lr, betas=(args.beta1, args.beta2))
    decoder_optimizer = torch.optim.Adam(chain(model.decoder.parameters(), model.quantize.embedding.parameters()),
                                         lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # DDP wrap
    model = DDP(model, device_ids=[device])
    vq_loss = DDP(vq_loss, device_ids=[device])

    # resume 处理（现在安全！加载到 .module）
    train_steps = 0
    if args.resume:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu")

            # 加载模型权重（包括decoder）
            model.module.load_state_dict(checkpoint["model"], strict=True)  # strict=True 确保所有key匹配

            # 可选：加载优化器状态（如果要继续训练）
            distill_optimizer.load_state_dict(checkpoint["distill_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            vq_loss.module.discriminator.load_state_dict(checkpoint["discriminator"])  # 注意：vq_loss还未DDP，这里假设加载前

            # 可选：恢复训练步数
            train_steps = checkpoint["steps"]

        dist.barrier()  # 同步所有ranks

        # 广播模型参数到所有ranks（确保一致）
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # 广播vq_loss（discriminator）
        for param in vq_loss.parameters():
            dist.broadcast(param.data, src=0)

        if rank == 0:
            logger.info("Checkpoint loaded and broadcasted successfully.")

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == 'fp16')
    scaler_disc = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == 'fp16')

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # 数据loader (训练用random crop)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, dist.get_world_size(), rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.global_batch_size // dist.get_world_size(), sampler=sampler, num_workers=args.num_workers, drop_last=True)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(dtype=ptdtype):
                if epoch < args.distill_epochs:  # 阶段1: 蒸馏标准encoder
                    # 先计算原 index（基于未掩码的 x）
                    index = model.module.get_csq_index(x)  # [B, 256]

                    # 应用 patch 级随机掩码
                    B, C, H, W = x.shape
                    patch_size = model.module.csq_encoder.patch_size  # 假设 16
                    num_patches_h = H // patch_size
                    num_patches_w = W // patch_size
                    mask_ratio = 0.15  # 掩码比例，可调整
                    num_masked = int(mask_ratio * num_patches_h * num_patches_w)

                    # 为每个 batch 生成随机掩码索引
                    masked_indices = torch.randperm(num_patches_h * num_patches_w, device=device)[:num_masked]
                    masked_h = masked_indices // num_patches_w
                    masked_w = masked_indices % num_patches_w

                    # 创建 masked_x，用高斯噪声替换 masked patch
                    masked_x = x.clone()
                    noise = torch.randn((B, C, patch_size, patch_size), device=device) * 0.1  # 高斯噪声，std=0.1
                    for i in range(num_masked):
                        h_start = masked_h[i] * patch_size
                        w_start = masked_w[i] * patch_size
                        masked_x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size] = noise

                    # 使用 masked_x 计算 logits，但目标是原 index
                    logits = model.module.classify_forward(masked_x)  # [B*256, 16384]
                    index_loss = index_loss_fn(logits, index.view(-1))
                    distill_optimizer.zero_grad()
                    scaler.scale(index_loss).backward()
                    if args.max_grad_norm != 0.0:
                        scaler.unscale_(distill_optimizer)
                        torch.nn.utils.clip_grad_norm_([*model.module.std_encoder.parameters(), *model.module.quant_conv.parameters(), *model.module.fc_classifier.parameters()], args.max_grad_norm)
                    scaler.step(distill_optimizer)
                    scaler.update()
                    running_loss += index_loss.item()
                else:
                    if epoch == args.distill_epochs and train_steps % len(loader) == 0:  # 只在epoch开始切换一次
                        with torch.no_grad():
                            model.module.quantize.embedding.weight.copy_(model.module.fc_classifier.weight)  # 无.T，shape匹配
                        model.module.quantize.embedding.weight.requires_grad = True  # 解冻 codebook，使其继续训练
                        logger.info("Switched to decoder training with new codebook (unfrozen).")

                    # 阶段2: 训decoder
                    recons, diff, _ = model.module.decode_forward(x)
                    loss_gen = vq_loss(diff, x, recons, optimizer_idx=0, global_step=train_steps+1, last_layer=model.module.decoder.last_layer, logger=logger, log_every=args.log_every)
                    decoder_optimizer.zero_grad()
                    scaler.scale(loss_gen).backward()
                    if args.max_grad_norm != 0.0:
                        scaler.unscale_(decoder_optimizer)
                        torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), args.max_grad_norm)
                    scaler.step(decoder_optimizer)
                    scaler.update()

                    loss_disc = vq_loss(diff, x, recons, optimizer_idx=1, global_step=train_steps+1, logger=logger, log_every=args.log_every)
                    optimizer_disc.zero_grad()
                    scaler_disc.scale(loss_disc).backward()
                    if args.max_grad_norm != 0.0:
                        scaler_disc.unscale_(optimizer_disc)
                        torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
                    scaler_disc.step(optimizer_disc)
                    scaler_disc.update()

                    running_loss += loss_gen.item() + loss_disc.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "distill_optimizer": distill_optimizer.state_dict(),
                        "decoder_optimizer": decoder_optimizer.state_dict(),
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
    parser.add_argument("--zca-ckpt", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/results_integrated/004-Integrated_Model/checkpoints/zca.pt', help="Path to precomputed ZCA checkpoint")
    parser.add_argument("--distill-epochs", type=int, default=10, help="Epochs for distilling std encoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100000)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    # 其他原参数如disc_start, disc_weight 等同原CSQ parser
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--disc-start", type=int, default=20000)
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
