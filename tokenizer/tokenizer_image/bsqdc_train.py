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

from tokenizer.tokenizer_image.bsqdc_model import VQ_models  # 这里记得确认你自己的 import 路径是否一致
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings

warnings.filterwarnings('ignore')


def save_checkpoint(
    raw_model,
    optimizer,
    vq_loss,
    optimizer_disc,
    ema,
    args,
    checkpoint_dir,
    logger,
    train_steps,
    epoch_done,
):
    """
    epoch_done: 已完成的 epoch 数，例如跑完 epoch index=9 后传入 10。
    """
    state_dict = raw_model.state_dict()
    ckpt = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "discriminator": vq_loss.module.discriminator.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "steps": train_steps,
        "epoch": epoch_done,
        "args": args,
    }
    if args.ema:
        ckpt["ema"] = ema.state_dict()

    save_path = f"{checkpoint_dir}/epoch_{epoch_done:04d}.pt"
    torch.save(ckpt, save_path)
    logger.info(f"Saved checkpoint to {save_path} at epoch={epoch_done}, step={train_steps}")


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
        model_string_name = f"{args.vq_model}-{args.num_bits}b-{args.image_size}px".replace("/", "-")
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

    # --- 模型初始化 ---
    vq_model = VQ_models[args.vq_model](
        codebook_l2_norm=args.codebook_l2_norm,
        dropout_p=args.dropout_p,
        sample=args.sample,
        # 传入 Annealing 参数
        anneal_noise=args.anneal_noise,
        anneal_start_epoch=args.anneal_start_epoch,
        anneal_end_epoch=args.anneal_end_epoch,
        noise_start_scale=args.noise_start_scale,
        noise_end_scale=args.noise_end_scale,
        # 传入投影矩阵控制参数
        learnable_proj=args.learnable_proj,
    )

    if rank == 0:
        logger.info(f"🚀 DC-AE Mode Active: {args.vq_model}")
        logger.info(f"Compression Info: {args.num_bits} bits -> {args.codebook_embed_dim} projected dim")
        if args.anneal_noise:
            logger.info(
                f"Noise Annealing ON: Epoch {args.anneal_start_epoch} -> {args.anneal_end_epoch}, "
                f"Scale {args.noise_start_scale} -> {args.noise_end_scale}"
            )
        proj_mode = "Learnable & Unconstrained" if args.learnable_proj else "Fixed & L2 Constrained"
        logger.info(f"Projection Matrix Mode: {proj_mode}")

    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")

    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
    else:
        ema = None
    vq_model = vq_model.to(device)

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
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
        drop_last=True,
    )

    steps_per_epoch = len(loader)

    # 现在 discriminator 以 epoch 为标准：
    # disc_start_epoch=4 表示从 epoch index=4 开始启用，也就是完成 0,1,2,3 四个 epoch 后启用。
    # 仍然传给 VQLoss 一个 step 阈值，以兼容 VQLoss 内部原本的 global_step 判断。
    disc_start_step = int(args.disc_start_epoch * steps_per_epoch)

    if rank == 0:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(
            f"Discriminator will start at epoch index >= {args.disc_start_epoch} "
            f"(global_step >= {disc_start_step})."
        )
        logger.info(f"Checkpoint will be saved every {args.ckpt_every_epoch} epochs.")

    # Losses
    vq_loss = VQLoss(
        disc_start=disc_start_step,
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
            train_steps = checkpoint.get("steps", 0)
            # 新版 checkpoint 会保存已完成 epoch 数；旧版 checkpoint 没有 epoch 字段时，退回用 step 推算。
            start_epoch = checkpoint.get("epoch", int(train_steps / steps_per_epoch))
        logger.info(f"Resumed from {args.vq_ckpt}, start_epoch={start_epoch}, steps={train_steps}")

    if args.compile:
        vq_model = torch.compile(vq_model)

    vq_model = DDP(vq_model, device_ids=[device])
    vq_model.train()
    vq_loss = DDP(vq_loss, device_ids=[device])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    log_steps = 0
    running_loss = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        # --- 将当前 epoch 传递给 quantizer (兼容 compile 和 ddp) ---
        raw_model = vq_model.module._orig_mod if args.compile else vq_model.module
        raw_model.quantize.set_epoch(epoch)

        disc_active = epoch >= args.disc_start_epoch
        if rank == 0:
            logger.info(
                f"Starting epoch {epoch}/{args.epochs - 1} "
                f"(epoch_done={epoch}, global_step={train_steps}, disc_active={disc_active})"
            )

        for x, _ in loader:
            imgs = x.to(device, non_blocking=True)

            # Generator Step
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                recons_imgs, codebook_loss = vq_model(imgs)

                # 兼容性处理 last_layer
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
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.ema:
                update_ema(ema, raw_model)

            # Discriminator Step
            # 现在 discriminator 以 epoch 为标准启用。
            # 启用前完全跳过 D 的 backward / step；VQLoss 的 gen adv loss 也由 disc_start_step 控制。
            loss_disc_item = 0.0
            if disc_active:
                optimizer_disc.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    loss_disc = vq_loss(
                        codebook_loss,
                        imgs,
                        recons_imgs,
                        optimizer_idx=1,
                        global_step=train_steps,
                        logger=logger,
                        log_every=args.log_every,
                    )
                scaler_disc.scale(loss_disc).backward()
                if args.max_grad_norm:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                loss_disc_item = loss_disc.item()

            running_loss += loss_gen.item() + loss_disc_item
            log_steps += 1
            train_steps += 1

            # Log 仍然按 step 触发。
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                if rank == 0:
                    usage = codebook_loss[3] if isinstance(codebook_loss, tuple) else 0.0
                    logger.info(
                        f"(epoch={epoch}, step={train_steps}) "
                        f"Loss: {running_loss / log_steps:.4f}, "
                        f"Spd: {steps_per_sec:.2f}, "
                        f"Usage: {usage:.4f}, "
                        f"DiscActive: {disc_active}"
                    )
                running_loss = 0
                log_steps = 0
                start_time = time.time()

        # Save Checkpoint
        # 现在 checkpoint 以 epoch 为标准，在每 ckpt_every_epoch 个 epoch 结束后保存一次。
        epoch_done = epoch + 1
        if rank == 0 and (
                epoch_done % args.ckpt_every_epoch == 0 or epoch_done == args.epochs
        ):            save_checkpoint(
                raw_model=raw_model,
                optimizer=optimizer,
                vq_loss=vq_loss,
                optimizer_disc=optimizer_disc,
                ema=ema,
                args=args,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                train_steps=train_steps,
                epoch_done=epoch_done,
            )

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DCAE 相关参数
    parser.add_argument("--vq-model", type=str, default="DCAE-32", choices=["DCAE-32", "DCAE-64", "DCAE-16"])
    parser.add_argument("--num-bits", type=int, default=128, help="DCAE-32建议128, DCAE-64建议256")
    parser.add_argument("--codebook-embed_dim", type=int, default=32, help="DCAE建议16或32")
    parser.add_argument("--quantizer", type=str, default="bsq", choices=["bsq"])

    # 噪声退火参数
    parser.add_argument("--anneal-noise", action='store_true', help="Enable cosine noise annealing")
    parser.add_argument("--anneal-start-epoch", type=int, default=10, help="Epoch to start noise annealing")
    parser.add_argument("--anneal-end-epoch", type=int, default=30, help="Epoch to end noise annealing")
    parser.add_argument("--noise-start-scale", type=float, default=1.0, help="Initial noise multiplier")
    parser.add_argument("--noise-end-scale", type=float, default=0.1, help="noise multiplier")

    # --- 新增：投影矩阵控制参数 ---
    parser.add_argument(
        "--learnable-proj",
        action='store_true',
        help="投影矩阵仅作随机初始化，且作为参数参与训练（不固定且不受圆上约束）。不加此参数时则默认固定并限制在单位圆上。",
    )

    # 基础参数
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument(
        "--data-path",
        type=str,
        default='/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt',
    )
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--results-dir", type=str, default="results_dcae_training")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)  # 这里的默认值被视为对应 batch_size=256 时的标准值
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=['none', 'fp16', 'bf16'])

    # 损失权重
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default='l2')

    # 现在 discriminator 以 epoch 为标准启用。
    # default=4 表示从日志里的 epoch=4 开始启用，也就是先完整跑完 epoch=0,1,2,3。
    parser.add_argument("--disc-start-epoch", type=int, default=4)

    # 现在 checkpoint 以 epoch 为标准保存。
    parser.add_argument("--ckpt-every-epoch", type=int, default=10)

    # 兼容旧命令行参数：保留但不再使用。
    parser.add_argument(
        "--disc-start",
        type=int,
        default=None,
        help="Deprecated: discriminator is now controlled by --disc-start-epoch.",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=None,
        help="Deprecated: checkpoint saving is now controlled by --ckpt-every-epoch.",
    )

    # 训练组件
    parser.add_argument("--ema", action='store_true', default=True)
    parser.add_argument("--sample", action='store_true', help="Training noise injection")
    parser.add_argument("--compile", action='store_true', help="Use torch.compile")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cloud-save-path", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-type", type=str, default='patchgan')
    parser.add_argument("--disc-loss", type=str, default='hinge')
    parser.add_argument("--gen-loss", type=str, default='hinge')
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)

    args = parser.parse_args()

    # =========================================================================
    # 动态参数调整模块 (以 global_batch_size = 256 作为基准)
    # =========================================================================
    BASE_BATCH_SIZE = 256

    if args.global_batch_size != BASE_BATCH_SIZE:
        # scale_ratio = 128 / 256 = 0.5
        # step_ratio  = 256 / 128 = 2.0
        scale_ratio = args.global_batch_size / BASE_BATCH_SIZE
        step_ratio = BASE_BATCH_SIZE / args.global_batch_size

        print("=" * 60)
        print(f"⚠️ 提示: 检测到 global_batch_size={args.global_batch_size}，偏离基准值 {BASE_BATCH_SIZE}。")
        print(f"🔄 正在根据当前 batch size 自动动态缩放参数：")

        # 1. 学习率按线性比例缩放 (Linear Scaling Rule)
        original_lr = args.lr
        args.lr = args.lr * scale_ratio
        print(f"  [LR] 学习率: {original_lr} -> {args.lr}")

        # 2. 只有 log_every 仍然是 step 触发，所以继续按反比例缩放。
        # discriminator 和 checkpoint 已经改成 epoch 触发，不再随 batch size 缩放。
        original_log_every = args.log_every
        args.log_every = int(args.log_every * step_ratio)
        print(f"  [Step] 日志打印频率 (log_every): {original_log_every} -> {args.log_every}")

        print(f"  [Epoch] 判别器启动 epoch (disc_start_epoch): {args.disc_start_epoch}，不随 batch size 缩放")
        print(f"  [Epoch] 模型保存频率 (ckpt_every_epoch): {args.ckpt_every_epoch}，不随 batch size 缩放")
        print("=" * 60)

    if args.disc_start is not None:
        print(
            f"⚠️ --disc-start={args.disc_start} 已弃用且不会生效；"
            f"当前使用 --disc-start-epoch={args.disc_start_epoch}。"
        )
    if args.ckpt_every is not None:
        print(
            f"⚠️ --ckpt-every={args.ckpt_every} 已弃用且不会生效；"
            f"当前使用 --ckpt-every-epoch={args.ckpt_every_epoch}。"
        )

    main(args)
