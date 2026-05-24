# train.py
# Modified: Implemented Two-Stage Training (Fix Encoder & Enable Sample at Stage 2)

import torch

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
from tokenizer.tokenizer_image.bsqmap_model import VQ_models, compute_avg_min_distance
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings

warnings.filterwarnings('ignore')


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Logging setup
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # Name reflects bits -> dim structure
        model_string_name = f"{args.vq_model}-{args.quantizer}-{args.num_bits}b2{args.codebook_embed_dim}d".replace("/",
                                                                                                                    "-")
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

    # Model Initialization
    # [Stage 1 Config] Force sample=False initially regardless of args, unless resuming from a later stage (handled later)
    initial_sample_state = False

    vq_model = VQ_models[args.vq_model](
        quantizer=args.quantizer,
        num_bits=args.num_bits,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        codebook_size=args.codebook_size,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        sample=initial_sample_state  # Start with False
    )

    if args.quantizer == "bsq" and rank == 0:
        logger.info(f"BSQ Mode Active: Using FIXED Gaussian Projection with Double L2 Norm.")
        logger.info(f"Projection: {args.num_bits} bits -> {args.codebook_embed_dim} dim")

    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")

    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    # Losses
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

    # Initial Optimizer (Training everything)
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

    # Resume Logic
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
        logger.info(f"Resumed from {args.vq_ckpt}, steps={train_steps}, epoch={start_epoch}")

    if args.compile:
        vq_model = torch.compile(vq_model)

    # [CRITICAL] find_unused_parameters=True is required because we will freeze the encoder later.
    # If this is False, DDP will hang when encoder gradients stop being produced in Stage 2.
    vq_model = DDP(vq_model, device_ids=[device], find_unused_parameters=True)
    vq_model.train()

    if args.ema: ema.eval()
    vq_loss = DDP(vq_loss, device_ids=[device], find_unused_parameters=False)  # Discriminator usually trains fully
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    log_steps = 0
    running_loss = 0
    start_time = time.time()

    # Helper flag to ensure we only switch once if resuming exactly at the boundary
    stage2_initialized = False

    # Check if we resumed directly into Stage 2
    if start_epoch >= args.stage2_start_epoch:
        logger.info(f"Resuming directly into Stage 2 (Epoch {start_epoch})")
        # Trigger the switch logic immediately before loop
        pass  # Logic handled inside loop check for robustness

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        # ==========================================
        # STAGE 2 TRANSITION LOGIC
        # ==========================================
        if epoch == args.stage2_start_epoch and not stage2_initialized:
            stage2_initialized = True
            logger.info(f"\n{'=' * 40}")
            logger.info(f" एंटering Stage 2 at Epoch {epoch}!")
            logger.info(f" Actions: 1. Enable Sampling (Noise Injection)")
            logger.info(f"          2. Freeze Encoder & QuantConv")
            logger.info(f"          3. Re-initialize Generator Optimizer")
            logger.info(f"{'=' * 40}\n")

            # 1. Enable Sample
            # Access underlying model from DDP
            raw_model = vq_model.module
            if hasattr(raw_model, 'sample'):
                raw_model.sample = True
            else:
                # Fallback if sample is inside a sub-module like quantizer
                logger.warning("Could not find 'sample' attribute directly on model, checking submodules...")
                # Add specific logic here if your model structure differs, e.g. raw_model.quantizer.sample = True

            # Update EMA model as well if it exists
            if args.ema:
                if hasattr(ema, 'sample'):
                    ema.sample = True

            # 2. Freeze Encoder
            # Assuming standard naming: 'encoder' and optionally 'quant_conv'
            # We want to train 'decoder', 'post_quant_conv', and 'quantizer' (if learnable)

            if hasattr(raw_model, 'encoder'):
                for p in raw_model.encoder.parameters():
                    p.requires_grad = False
                logger.info("Frozen: Encoder")

            if hasattr(raw_model, 'quant_conv'):
                for p in raw_model.quant_conv.parameters():
                    p.requires_grad = False
                logger.info("Frozen: QuantConv (Pre-Quant)")

            # 3. Re-initialize Optimizer
            # This is critical. Creating a new optimizer with only requires_grad=True parameters
            # prevents the optimizer from trying to update frozen params (which can cause errors or bugs).
            # Note: Decoder momentum is reset, which is acceptable for this stage transition.

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vq_model.parameters()),
                lr=args.lr,
                betas=(args.beta1, args.beta2)
            )
            logger.info("Optimizer re-initialized with only trainable parameters.")
            logger.info(f"Trainable params: {sum(p.numel() for p in vq_model.parameters() if p.requires_grad):,}")

        # Ensure 'sample' is correct if we resumed past stage 2 but didn't hit the == transition
        if epoch > args.stage2_start_epoch and not stage2_initialized:
            # If we resumed at epoch 25, we need to ensure state is correct
            raw_model = vq_model.module
            if hasattr(raw_model, 'sample') and not raw_model.sample:
                logger.info(f"Resumed past Stage 2 start. Forcing Sample=True and Freezing Encoder.")
                raw_model.sample = True
                if args.ema: ema.sample = True
                if hasattr(raw_model, 'encoder'):
                    for p in raw_model.encoder.parameters(): p.requires_grad = False
                if hasattr(raw_model, 'quant_conv'):
                    for p in raw_model.quant_conv.parameters(): p.requires_grad = False
                # Re-init optimizer if we just resumed
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, vq_model.parameters()),
                    lr=args.lr, betas=(args.beta1, args.beta2)
                )
                stage2_initialized = True

        logger.info(
            f"Epoch {epoch} start... [Sample Mode: {vq_model.module.sample if hasattr(vq_model.module, 'sample') else 'Unknown'}]")

        for x, _ in loader:
            imgs = x.to(device, non_blocking=True)

            # Generator Step
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                recons_imgs, codebook_loss = vq_model(imgs)
                # codebook_loss is tuple: (balance, sign, 0.0, usage)
                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps,
                                   last_layer=vq_model.module.decoder.last_layer,
                                   logger=logger, log_every=args.log_every)

            scaler.scale(loss_gen).backward()
            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                # Only clip grads for active parameters
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, vq_model.parameters()),
                    args.max_grad_norm
                )
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
                    _, _, _, usage = codebook_loss
                    if isinstance(usage, torch.Tensor): usage = usage.item()
                    logger.info(
                        f"(step={train_steps}) Loss: {avg_loss:.4f}, Spd: {steps_per_sec:.2f}, "
                        f"Usage: {usage:.4f}"
                    )

                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save Checkpoint
            if train_steps % args.ckpt_every == 0 and rank == 0:
                state_dict = vq_model.module._orig_mod.state_dict() if args.compile else vq_model.module.state_dict()
                ckpt = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),  # Will save the active optimizer (full or partial)
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

    # BSQ Arguments
    parser.add_argument("--quantizer", type=str, default="bsq", choices=["bsq", "fsq", "vq"])
    parser.add_argument("--num-bits", type=int, default=14, help="Number of bits for quantizer (Intermediate)")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Output projection dimension (Spherical)")

    # New Arguments for Two-Stage Training
    parser.add_argument("--sample", action='store_true', help="Ignored in stage 1, auto-enabled in stage 2")
    parser.add_argument("--stage2-start-epoch", type=int, default=20,
                        help="Epoch to freeze encoder and enable sampling")

    # Existing Args
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--data-face-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, required=False)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--ema", action='store_true')

    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="L2 norm for output projection")

    # Weightsf
    parser.add_argument("--commit-loss-beta", type=float, default=0.0, help="Weight for Sign Loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=1.0, help="Weight for EMA Balance Loss")
    parser.add_argument("--codebook-weight", type=float, default=1.0)

    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default='l2')
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-start", type=int, default=40000)
    parser.add_argument("--disc-type", type=str, default='patchgan')
    parser.add_argument("--disc-loss", type=str, default='hinge')
    parser.add_argument("--gen-loss", type=str, default='hinge')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_bsq_spherical")
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--mixed-precision", type=str, default='bf16')

    args = parser.parse_args()
    main(args)
