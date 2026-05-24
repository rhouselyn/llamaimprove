import os
import time
import argparse
import warnings
from glob import glob
from copy import deepcopy

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset

from tokenizer.tokenizer_image.bsqdc_wrong import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

warnings.filterwarnings("ignore")


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Logging setup
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = (
            f"{args.vq_model}-{args.num_bits}b-{args.codebook_embed_dim}d-"
            f"{args.image_size}px"
        ).replace("/", "-")

        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        cloud_results_dir = (
            f"{args.cloud_save_path}/{time_record}"
            if args.cloud_save_path
            else None
        )
        cloud_checkpoint_dir = (
            f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
            if cloud_results_dir
            else None
        )

        if cloud_checkpoint_dir:
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        cloud_checkpoint_dir = None
        experiment_dir = None

    logger.info(f"{args}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    vq_model = VQ_models[args.vq_model](
        image_size=args.image_size,
        num_bits=args.num_bits,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        dropout_p=args.dropout_p,
        sample=args.sample,
        anneal_noise=args.anneal_noise,
        noise_schedule=args.noise_schedule,
        anneal_start_epoch=args.anneal_start_epoch,
        anneal_end_epoch=args.anneal_end_epoch,
        noise_start_scale=args.noise_start_scale,
        noise_end_scale=args.noise_end_scale,
        noise_peak_scale=args.noise_peak_scale,
        learnable_proj=args.learnable_proj,
    )

    if rank == 0:
        logger.info(f"DC-AE BSQ Mode Active: {args.vq_model}")
        logger.info(
            f"Compression Info: {args.num_bits} bits -> "
            f"{args.codebook_embed_dim} projected dim"
        )
        logger.info(f"Learnable projection: {args.learnable_proj}")

        if args.sample:
            logger.info("Training noise injection: ON")
            if args.anneal_noise:
                logger.info(
                    f"Noise annealing ON: schedule={args.noise_schedule}, "
                    f"epoch {args.anneal_start_epoch} -> {args.anneal_end_epoch}, "
                    f"start={args.noise_start_scale}, "
                    f"peak={args.noise_peak_scale}, "
                    f"end={args.noise_end_scale}"
                )
            else:
                logger.info(f"Noise annealing OFF: fixed scale={args.noise_start_scale}")
        else:
            logger.info("Training noise injection: OFF")

    logger.info(
        f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}"
    )

    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
    else:
        ema = None

    vq_model = vq_model.to(device)

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    optimizer = torch.optim.Adam(
        vq_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    optimizer_disc = torch.optim.Adam(
        vq_loss.discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )

    dataset = build_dataset(args, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    train_steps = 0
    start_epoch = 0

    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)

        vq_model.load_state_dict(checkpoint["model"])

        if args.ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])

        if not args.finetune:
            train_steps = checkpoint["steps"]
            start_epoch = int(train_steps / len(loader))

        logger.info(f"Resumed from {args.vq_ckpt}, steps={train_steps}")

    # -------------------------------------------------------------------------
    # Compile + DDP
    # -------------------------------------------------------------------------
    if args.compile:
        vq_model = torch.compile(vq_model)

    vq_model = DDP(vq_model, device_ids=[device])
    vq_model.train()

    vq_loss = DDP(vq_loss, device_ids=[device])
    vq_loss.train()

    ptdtype = {
        "none": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]

    log_steps = 0
    running_loss = 0.0
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        raw_model = vq_model.module._orig_mod if args.compile else vq_model.module
        raw_model.quantize.set_epoch(epoch)

        for x, _ in loader:
            imgs = x.to(device, non_blocking=True)

            # -----------------------------------------------------------------
            # Generator step
            # -----------------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(args.mixed_precision != "none"),
                dtype=ptdtype,
            ):
                recons_imgs, codebook_loss = vq_model(imgs)

                last_layer = raw_model.decoder.last_layer

                loss_gen = vq_loss(
                    codebook_loss,
                    imgs,
                    recons_imgs,
                    optimizer_idx=0,
                    global_step=train_steps,
                    last_layer=last_layer,
                    logger=logger,
                    log_every=args.log_every,
                )

            scaler.scale(loss_gen).backward()

            if args.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    vq_model.parameters(),
                    args.max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            if args.ema:
                update_ema(ema, raw_model)

            # -----------------------------------------------------------------
            # Discriminator step
            # -----------------------------------------------------------------
            optimizer_disc.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(args.mixed_precision != "none"),
                dtype=ptdtype,
            ):
                loss_disc = vq_loss(
                    codebook_loss,
                    imgs,
                    recons_imgs.detach(),
                    optimizer_idx=1,
                    global_step=train_steps,
                    logger=logger,
                    log_every=args.log_every,
                )

            scaler_disc.scale(loss_disc).backward()

            if args.max_grad_norm:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(
                    vq_loss.module.discriminator.parameters(),
                    args.max_grad_norm,
                )

            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            running_loss += loss_gen.item() + loss_disc.item()
            log_steps += 1
            train_steps += 1

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()

                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                if rank == 0:
                    usage = codebook_loss[3] if isinstance(codebook_loss, tuple) else 0.0

                    if torch.is_tensor(usage):
                        usage_value = usage.item()
                    else:
                        usage_value = float(usage)

                    logger.info(
                        f"(epoch={epoch}, step={train_steps}) "
                        f"Loss: {running_loss / log_steps:.4f}, "
                        f"Spd: {steps_per_sec:.2f}, "
                        f"Usage: {usage_value:.4f}"
                    )

                running_loss = 0.0
                log_steps = 0
                start_time = time.time()

            # -----------------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------------
            if train_steps % args.ckpt_every == 0 and rank == 0:
                state_dict = raw_model.state_dict()

                ckpt = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "discriminator": vq_loss.module.discriminator.state_dict(),
                    "optimizer_disc": optimizer_disc.state_dict(),
                    "steps": train_steps,
                    "args": args,
                }

                if args.ema:
                    ckpt["ema"] = ema.state_dict()

                save_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(ckpt, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

                if cloud_checkpoint_dir:
                    cloud_save_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(ckpt, cloud_save_path)
                    logger.info(f"Saved cloud checkpoint to {cloud_save_path}")

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # DCAE BSQ parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--vq-model",
        type=str,
        default="DCAE-32",
        choices=["DCAE-16", "DCAE-32", "DCAE-64"],
    )

    parser.add_argument(
        "--num-bits",
        type=int,
        default=128,
        help="DCAE-16 default: 32, DCAE-32 default: 128, DCAE-64 default: 512",
    )

    parser.add_argument(
        "--codebook-embed_dim",
        dest="codebook_embed_dim",
        type=int,
        default=32,
        help="DCAE-16 default: 8, DCAE-32 default: 32, DCAE-64 default: 128",
    )

    # Noise injection / annealing
    parser.add_argument("--sample", action="store_true", help="Enable training noise injection")
    parser.add_argument("--anneal-noise", action="store_true", help="Enable noise annealing")

    parser.add_argument(
        "--noise-schedule",
        type=str,
        default="cosine_decay",
        choices=["constant", "cosine_decay", "warmup_cosine_decay"],
    )

    parser.add_argument("--anneal-start-epoch", type=int, default=10)
    parser.add_argument("--anneal-end-epoch", type=int, default=30)
    parser.add_argument("--noise-start-scale", type=float, default=1.0)
    parser.add_argument("--noise-end-scale", type=float, default=0.1)
    parser.add_argument("--noise-peak-scale", type=float, default=1.0)

    # Projection matrix control
    parser.add_argument(
        "--learnable-proj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True. Use --no-learnable-proj for fixed normalized projection.",
    )

    # -------------------------------------------------------------------------
    # Basic parameters
    # -------------------------------------------------------------------------
    parser.add_argument("--dataset", type=str, default="aoss")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt",
    )
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")

    parser.add_argument("--results-dir", type=str, default="results_dcae_training")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
    )

    # -------------------------------------------------------------------------
    # Loss parameters
    # -------------------------------------------------------------------------
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default="l2")
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--codebook-weight", type=float, default=1.0)

    parser.add_argument("--disc-start", type=int, default=20000)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-type", type=str, default="patchgan")
    parser.add_argument("--disc-loss", type=str, default="hinge")
    parser.add_argument("--gen-loss", type=str, default="hinge")

    # -------------------------------------------------------------------------
    # Training components
    # -------------------------------------------------------------------------
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--cloud-save-path", type=str, default=None)

    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--codebook-l2-norm", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)

    args = parser.parse_args()

    # =========================================================================
    # Dynamic scaling based on global_batch_size = 256
    # =========================================================================
    BASE_BATCH_SIZE = 256

    if args.global_batch_size != BASE_BATCH_SIZE:
        scale_ratio = args.global_batch_size / BASE_BATCH_SIZE
        step_ratio = BASE_BATCH_SIZE / args.global_batch_size

        print("=" * 60)
        print(
            f"Detected global_batch_size={args.global_batch_size}, "
            f"baseline={BASE_BATCH_SIZE}."
        )
        print("Auto-scaling parameters:")

        original_lr = args.lr
        args.lr = args.lr * scale_ratio
        print(f"  [LR] lr: {original_lr} -> {args.lr}")

        original_disc_start = args.disc_start
        args.disc_start = int(args.disc_start * step_ratio)
        print(f"  [Step] disc_start: {original_disc_start} -> {args.disc_start}")

        original_ckpt_every = args.ckpt_every
        args.ckpt_every = int(args.ckpt_every * step_ratio)
        print(f"  [Step] ckpt_every: {original_ckpt_every} -> {args.ckpt_every}")

        original_log_every = args.log_every
        args.log_every = int(args.log_every * step_ratio)
        print(f"  [Step] log_every: {original_log_every} -> {args.log_every}")

        print("=" * 60)

    main(args)