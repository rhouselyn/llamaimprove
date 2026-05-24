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
from utils.ema import requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_loss import VQLoss  # 原CSQ损失

from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])


class L2NormalizeParam(nn.Module):
    def forward(self, weight):
        return F.normalize(weight, p=2, dim=1)


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
    def __init__(self, n_e, e_dim, bits_dim, l2_norm=True, show_usage=True):  # 保留bits_dim但不使用
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.bits_dim = bits_dim  # 保留但不使用
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(self.n_e * 4))  # 修改：扩大buffer，避免overflow

    def compute_usage(self, min_encoding_indices):
        cur_len = min_encoding_indices.shape[0]
        self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
        self.codebook_used[-cur_len:] = min_encoding_indices
        return len(torch.unique(self.codebook_used)) / self.n_e


class IntegratedModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.std_encoder = StdEncoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.fc_classifier = nn.Linear(config.codebook_embed_dim, config.codebook_size)
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 14, config.codebook_l2_norm, config.codebook_show_usage)  # 保留usage，bits_dim随意
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        self.decoder = Decoder(z_channels=config.z_channels, ch_mult=config.decoder_ch_mult, dropout=config.dropout_p)
        if config.codebook_l2_norm:
            parametrize.register_parametrization(self.fc_classifier, "weight", L2NormalizeParam())


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


def train_step(model, vq_loss, index_loss_fn, x, joint_optimizer, optimizer_disc, scaler, scaler_disc, args, train_steps, device, logger, fixed_encoder, fixed_quant_conv, fixed_codebook):
    B, _, H, W = x.shape  # 假设 H=W=256
    patch_h, patch_w = H // 16, W // 16  # 16x16 patches

    with torch.no_grad():
        h = fixed_encoder(x)
        quant_h = fixed_quant_conv(h)
        if args.codebook_l2_norm:
            quant_h = F.normalize(quant_h, p=2, dim=1)
        flattened_h = quant_h.permute(0, 2, 3, 1).reshape(-1, 8)
        # 计算dist (squared euclidean)
        d = torch.cdist(flattened_h, fixed_codebook, p=2) ** 2
        min_encoding_indices = torch.argmin(d, dim=1)

    codebook_usage = model.module.quantize.compute_usage(min_encoding_indices) if model.module.quantize.show_usage and model.training else 0

    h = model.module.std_encoder(x)  # [B, 256, 16, 16]
    quant_h = model.module.quant_conv(h)  # [B, 8, 16, 16]
    if model.module.quantize.l2_norm:
        quant_h = F.normalize(quant_h, p=2, dim=1)  # 新增：norm for consistency

    flattened_h = quant_h.permute(0, 2, 3, 1).reshape(-1, model.module.quantize.e_dim)  # [-1, 8]
    logits = model.module.fc_classifier(flattened_h)  # [-1, 16384]
    index_loss = index_loss_fn(logits, min_encoding_indices)

    # 新增：用 fc.weight[true_index] 作为 true_codes 生成 recons
    true_indices = min_encoding_indices
    true_codes = model.module.fc_classifier.weight[true_indices]  # [-1, 8]，weight已norm如果l2_norm
    z_q = true_codes.view(B, patch_h, patch_w, model.module.quantize.e_dim).permute(0, 3, 1, 2)  # [B, 8, 16, 16]

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

    # 返回三个独立loss
    return loss_gen.item(), index_loss.item(), loss_disc.item()


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

    fixed_encoder = None
    fixed_quant_conv = None
    fixed_codebook = None

    train_steps = 0
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        load_dict = {}
        for k, v in checkpoint["model"].items():
            if k.startswith('encoder.') or k.startswith('quant_conv.') or k.startswith('quantize.embedding.') or k.startswith('decoder.') or k.startswith('post_quant_conv.'):
                if k.startswith('quantize.embedding.'):
                    load_dict['fc_classifier.weight'] = checkpoint["model"]["quantize.embedding.weight"]
                else:
                    load_dict[k] = v
        model.load_state_dict(load_dict, strict=False)
        model.fc_classifier.bias.data.zero_()  # 设置bias=0

        # 创建fixed
        fixed_encoder = StdEncoder(in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256).to(device)
        fixed_quant_conv = nn.Conv2d(256, 8, 1).to(device)
        fixed_codebook = checkpoint["model"]["quantize.embedding.weight"].to(device)
        if args.codebook_l2_norm:
            fixed_codebook = F.normalize(fixed_codebook, p=2, dim=1)

        load_fixed_encoder = {k.replace('encoder.', ''): v for k, v in load_dict.items() if k.startswith('encoder.')}
        fixed_encoder.load_state_dict(load_fixed_encoder)
        load_fixed_quant_conv = {k.replace('quant_conv.', ''): v for k, v in load_dict.items() if k.startswith('quant_conv.')}
        fixed_quant_conv.load_state_dict(load_fixed_quant_conv)

        requires_grad(fixed_encoder, False)
        requires_grad(fixed_quant_conv, False)

        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        train_steps = checkpoint["steps"]

        if rank == 0:
            logger.info(f"Loaded VQ checkpoint from {args.vq_ckpt} and broadcasted.")

        dist.barrier()
        for param_group in joint_optimizer.param_groups:
            for param in param_group['params']:
                dist.broadcast(param, src=0)
        for param in vq_loss.parameters():
            dist.broadcast(param, src=0)
        for param in fixed_encoder.parameters():
            dist.broadcast(param, src=0)
        for param in fixed_quant_conv.parameters():
            dist.broadcast(param, src=0)
        dist.broadcast(fixed_codebook, src=0)

    model = DDP(model, device_ids=[device])
    vq_loss = DDP(vq_loss, device_ids=[device])

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
    running_gen_loss = 0.0
    running_ce_loss = 0.0
    running_disc_loss = 0.0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, _ in loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(dtype=ptdtype):
                gen_l, ce_l, disc_l = train_step(model, vq_loss, index_loss_fn, x, joint_optimizer, optimizer_disc, scaler, scaler_disc, args, train_steps, device, logger, fixed_encoder, fixed_quant_conv, fixed_codebook)

            running_gen_loss += gen_l
            running_ce_loss += ce_l
            running_disc_loss += disc_l
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_gen = torch.tensor(running_gen_loss / log_steps, device=device)
                avg_ce = torch.tensor(running_ce_loss / log_steps, device=device)
                avg_disc = torch.tensor(running_disc_loss / log_steps, device=device)
                dist.all_reduce(avg_gen, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_ce, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_disc, op=dist.ReduceOp.SUM)
                avg_gen = avg_gen.item() / world_size
                avg_ce = avg_ce.item() / world_size
                avg_disc = avg_disc.item() / world_size

                logger.info(f"(step={train_steps:07d}) Gen: {avg_gen:.4f}, CE: {avg_ce:.4f}, Disc: {avg_disc:.4f}, Steps/Sec: {steps_per_sec:.2f}")

                running_gen_loss = 0.0
                running_ce_loss = 0.0
                running_disc_loss = 0.0
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
    parser.add_argument("--vq-ckpt", type=str, default="/mnt/afs/zhengmingkai/raozf/llamagen/results_tokenizer_image/008-VQ-16/checkpoints/0400000.pt")
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--results-dir", type=str, default="/mnt/afs/zhengmingkai/raozf/llamagen/results_integrated")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20000)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
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
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True)
    parser.add_argument("--codebook-show-usage", action='store_true', default=True)
    args = parser.parse_args()
    main(args)
