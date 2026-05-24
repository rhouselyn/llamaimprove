# train.py
# Modified: Supports Hybrid Continuous Model logging

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from collections import defaultdict

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset

# Import
from tokenizer.tokenizer_image.hybrid_model import Continuous_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings

warnings.filterwarnings('ignore')


def main(args):
    # Standard DDP Init
    assert torch.cuda.is_available()
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        name = f"{experiment_index:03d}-Cont-Hybrid-G{args.num_global_tokens}"
        experiment_dir = f"{args.results_dir}/{name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Hybrid Continuous Model Training: {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None

    # Model
    model = Continuous_models[args.model_type](
        num_global_tokens=args.num_global_tokens,
        p_min=args.p_min,
        p_max=args.p_max
    ).to(device)

    if rank == 0:
        logger.info(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        from copy import deepcopy
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

    # Loss: L1/L2 + Perceptual + Discriminator
    vq_loss = VQLoss(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_type='patchgan',
        disc_loss='hinge',
        gen_adv_loss='hinge',
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=1.0,
        reconstruction_loss='l1',
        codebook_weight=0.0,  # No quantization loss
    ).to(device)

    # Opt
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True,
                        drop_last=True)

    model = DDP(model, device_ids=[device])
    vq_loss = DDP(vq_loss, device_ids=[device])

    # Tracking
    p_stats = defaultdict(list)
    running_gen, running_disc = 0.0, 0.0
    log_steps, train_steps = 0, 0
    start_time = time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0: logger.info(f"Epoch {epoch} Start")

        for x, _ in loader:
            imgs = x.to(device, non_blocking=True)

            # --- Generator ---
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16):
                recons, (p_val, m_val) = model(imgs)

                # Pass dummy codebook loss
                loss_gen = vq_loss((0., 0., 0., 0.), imgs, recons, optimizer_idx=0, global_step=train_steps,
                                   last_layer=model.module.decoder.last_layer, logger=None, log_every=args.log_every)

            scaler.scale(loss_gen).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if args.ema: update_ema(ema, model.module)

            # --- Discriminator ---
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16):
                loss_disc = vq_loss((0.,), imgs, recons, optimizer_idx=1, global_step=train_steps, logger=None)

            scaler_disc.scale(loss_disc).backward()
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            # --- Stats ---
            train_steps += 1
            log_steps += 1
            running_gen += loss_gen.item()
            running_disc += loss_disc.item()

            # Bucket P-value for analysis
            if p_val < 0.6:
                p_stats['<0.6'].append(loss_gen.item())
            elif p_val < 0.8:
                p_stats['0.6-0.8'].append(loss_gen.item())
            else:
                p_stats['>0.8'].append(loss_gen.item())

            if train_steps % args.log_every == 0:
                end_time = time.time()
                spd = log_steps / (end_time - start_time)

                # Format bucket stats
                stats_str = " | ".join([f"L({k}): {sum(v) / len(v):.3f}" for k, v in sorted(p_stats.items()) if v])
                p_stats = defaultdict(list)

                if rank == 0:
                    logger.info(
                        f"Step {train_steps}: L_Gen {running_gen / log_steps:.4f}, L_Disc {running_disc / log_steps:.4f}, "
                        f"P={p_val:.2f}, M={m_val}, {spd:.1f} it/s | {stats_str}"
                    )

                running_gen, running_disc = 0.0, 0.0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and rank == 0:
                torch.save(model.module.state_dict(), f"{checkpoint_dir}/step_{train_steps:06d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="Continuous-Base")
    parser.add_argument("--num-global-tokens", type=int, default=64)
    parser.add_argument("--p-min", type=float, default=0.5)
    parser.add_argument("--p-max", type=float, default=1.0)

    # Standard paths
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--results-dir", type=str, default='./results_continuous')

    # Training hyperparams
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--mixed-precision", type=str, default='bf16')
    parser.add_argument("--disc-start", type=int, default=5001)
    parser.add_argument("--disc-weight", type=float, default=0.8)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()
    main(args)
