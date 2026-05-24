# train.py
# Modified for BSQ training with bit-count configuration and EMA loss.

import torch

# Optimization for A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.bsq_model import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings

warnings.filterwarnings('ignore')


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Experiment logging
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # Include bit info in name
        model_string_name = f"{args.vq_model}-{args.quantizer}-{args.num_bits}bits".replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}" if args.cloud_save_path else None
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints" if cloud_results_dir else None
        if cloud_checkpoint_dir:
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        cloud_checkpoint_dir = None
        experiment_dir = None

    logger.info(f"{args}")

    # Create model
    # Note: we pass num_bits to the factory. If quantizer is 'bsq', codebook_size is ignored.
    vq_model = VQ_models[args.vq_model](
        quantizer=args.quantizer,
        num_bits=args.num_bits,  # Defines dimension for BSQ
        codebook_size=args.codebook_size,  # Only for legacy VQ
        codebook_embed_dim=args.num_bits,  # Ensure embed dim matches bits
        commit_loss_beta=args.commit_loss_beta,  # Weight for Sign Loss (close to -1,1)
        entropy_loss_ratio=args.entropy_loss_ratio,  # Weight for Balance Loss (EMA)
        dropout_p=args.dropout_p,
    )

    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    logger.info(f"Quantizer Type: {args.quantizer}, Bits: {args.num_bits}")

    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    # Discriminator / Perceptual Loss
    vq_loss = VQLoss(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
    ).to(device)

    # Optimizers
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Resume
    train_steps = 0
    start_epoch = 0
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])

        if not args.finetune:
            train_steps = checkpoint["steps"]
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        logger.info(f"Resumed from {args.vq_ckpt}, steps={train_steps}")

    if args.compile:
        vq_model = torch.compile(vq_model)

    vq_model = DDP(vq_model, device_ids=[device])
    vq_model.train()
    if args.ema: ema.eval()
    vq_loss = DDP(vq_loss, device_ids=[device])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    log_steps = 0
    running_loss = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch} start...")
        for x, _ in loader:
            imgs = x.to(device, non_blocking=True)

            # Generator Step
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                recons_imgs, codebook_loss = vq_model(imgs)
                # Note: codebook_loss tuple structure is handled inside VQLoss if generic,
                # but typically VQLoss expects a specific scalar or sum.
                # In ste_model.py for BSQ we return: (balance_loss, sign_loss, 0.0, usage)
                # Ensure VQLoss handles this tuple correctly.
                # Standard VQ logic sums up codebook_loss[0] usually.
                # Here we pass the tuple to vq_loss() wrapper which typically internally sums them
                # or expects the user to have summed them in the model.
                # Let's assume VQLoss (which wasn't provided but imported) handles the tuple
                # or we rely on the `codebook_weight` applied to the sum of losses.

                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps,
                                   last_layer=vq_model.module.decoder.last_layer,
                                   logger=logger, log_every=args.log_every)

            scaler.scale(loss_gen).backward()
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # Discriminator Step
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            running_loss += loss_gen.item() + loss_disc.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss)
                avg_loss = avg_loss.item() / dist.get_world_size()

                if rank == 0:
                    # Log Codebook Usage (passed in codebook_loss tuple index 3)
                    # BSQ usage calculation is in the model
                    _, _, _, usage = codebook_loss
                    if isinstance(usage, torch.Tensor): usage = usage.item()

                    logger.info(
                        f"(step={train_steps}) Loss: {avg_loss:.4f}, Spd: {steps_per_sec:.2f}, "
                        f"Bit Usage: {usage:.4f}"
                    )

                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save Checkpoint
            if train_steps % args.ckpt_every == 0 and rank == 0:
                state_dict = vq_model.module._orig_mod.state_dict() if args.compile else vq_model.module.state_dict()
                ckpt = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "discriminator": vq_loss.module.discriminator.state_dict(),
                    "optimizer_disc": optimizer_disc.state_dict(),
                    "steps": train_steps,
                    "args": args
                }
                if args.ema: ckpt["ema"] = ema.state_dict()
                torch.save(ckpt, f"{checkpoint_dir}/{train_steps:07d}.pt")
                logger.info(f"Saved checkpoint {train_steps}")

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Modified/New Arguments
    parser.add_argument("--quantizer", type=str, default="bsq", choices=["bsq", "fsq", "vq"])
    parser.add_argument("--num-bits", type=int, default=64, help="Number of bits for BSQ (defines embed dim)")
    parser.add_argument("--sample", action='store_true', help="Add standard Gaussian noise to quantized bits during training") # <--- [新增]

    # Existing Args
    # parser.add_argument("--data-path", type=str,
    #                     default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/whl/llamagen/ILSVRC/Data/CLS-LOC/train')
    parser.add_argument("--data-face-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, required=False)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--ema", action='store_true')

    # codebook-size will be ignored for BSQ, but kept for arg parsing compatibility
    parser.add_argument("--codebook-size", type=int, default=16384)

    # Weights for BSQ losses mapped to existing arg names for convenience
    parser.add_argument("--commit-loss-beta", type=float, default=0.0, help="Weight for Sign Loss (Encourage -1/1)")
    parser.add_argument("--entropy-loss-ratio", type=float, default=1.0, help="Weight for EMA Balance Loss")

    parser.add_argument("--codebook-l2-norm", action='store_true', default=False, help="Usually False for BSQ")
    parser.add_argument("--codebook-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default='l2')
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-start", type=int, default=20000)
    parser.add_argument("--disc-type", type=str, default='patchgan')
    parser.add_argument("--disc-loss", type=str, default='hinge')
    parser.add_argument("--gen-loss", type=str, default='hinge')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--results-dir", type=str, default="results_bsq")
    # parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--dataset", type=str, default='imagenet')
    # parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # Changed Batch Size Default
    parser.add_argument("--global-batch-size", type=int, default=128)

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)  # Slightly more frequent for large batch
    parser.add_argument("--mixed-precision", type=str, default='bf16')

    args = parser.parse_args()
    main(args)
