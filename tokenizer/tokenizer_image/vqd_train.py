# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
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
from copy import deepcopy

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq2_model import VQ_models  # 使用你修改后的文件
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings
warnings.filterwarnings('ignore')


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}" if args.cloud_save_path else None
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints" if cloud_results_dir else None
        if cloud_checkpoint_dir is not None:
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        cloud_checkpoint_dir = None

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # ---------- Build dataset & loader first（以便 resume 时能用到长度计算） ----------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
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
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # ---------- create and load model ----------
    # Stage1：codebook 放大（若非从 ckpt 恢复）
    codebook_size_for_build = args.codebook_size
    if args.vq_ckpt is None:
        codebook_size_for_build = args.codebook_size * args.expand_codebook_factor  # ### NEW

    vq_model = VQ_models[args.vq_model](
        codebook_size=codebook_size_for_build,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

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
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    detach_quant = False

    # ---------- optimizer & (optional) resume ----------
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])

        # restore quantizer runtime state（向后兼容）
        q = vq_model.quantize
        if "detach_quant" in checkpoint:
            detach_quant = checkpoint["detach_quant"]
            q.detach_quant = detach_quant
        if "stage2_fixed" in checkpoint:
            q.stage2_fixed = checkpoint["stage2_fixed"]
        if "fixed_topk_indices" in checkpoint and checkpoint["fixed_topk_indices"] is not None:
            q.fixed_topk_indices = checkpoint["fixed_topk_indices"].to(device)
        if "code_counts" in checkpoint and checkpoint["code_counts"] is not None:
            q.code_counts = checkpoint["code_counts"].to(device)

        # freeze if needed
        if detach_quant:
            requires_grad(vq_model.encoder, False)
            requires_grad(vq_model.quant_conv, False)
            vq_model.quantize.embedding.requires_grad = False
            if args.ema:
                requires_grad(ema.encoder, False)
                requires_grad(ema.quant_conv, False)
                ema.quantize.embedding.requires_grad = False

        # Setup optimizer based on stage
        if detach_quant:
            optimizer_params = list(vq_model.decoder.parameters()) + list(vq_model.post_quant_conv.parameters())
        else:
            optimizer_params = vq_model.parameters()
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])

        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            # 依据 dataset 长度估计起始 epoch
            iters_per_epoch = int(len(dataset) / args.global_batch_size)
            iters_per_epoch = max(1, iters_per_epoch)
            start_epoch = int(train_steps / iters_per_epoch)
            train_steps = int(start_epoch * iters_per_epoch)
        else:
            train_steps = 0
            start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}, detach_quant={detach_quant}")
    else:
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model)  # requires PyTorch 2.0

    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu])
    vq_model.train()
    if args.ema:
        ema.eval()
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0.0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        # ---------- NEW: epoch 钩子（仅 stage1 生效） ----------
        if not detach_quant:
            # 注意：DDP 包裹后，需访问 module
            vq_model.module.quantize.on_new_epoch()

        # ---------- Enter stage2 if needed ----------
        if epoch == args.stage1_epochs and not detach_quant:
            # 1) 固定 top-1/4 子集
            vq_model.module.quantize.fix_topk_codebook()
            if args.ema:
                ema.quantize.fix_topk_codebook()
            logger.info("Fixed top-1/4 codebook subset for stage 2.")

            # 2) 进入 stage2：与现有逻辑保持一致
            detach_quant = True
            vq_model.module.quantize.detach_quant = True
            requires_grad(vq_model.module.encoder, False)
            requires_grad(vq_model.module.quant_conv, False)
            vq_model.module.quantize.embedding.requires_grad = False
            if args.ema:
                ema.quantize.detach_quant = True
                requires_grad(ema.encoder, False)
                requires_grad(ema.quant_conv, False)
                ema.quantize.embedding.requires_grad = False

            optimizer = torch.optim.Adam(
                list(vq_model.module.decoder.parameters()) + list(vq_model.module.post_quant_conv.parameters()),
                lr=args.lr, betas=(args.beta1, args.beta2)
            )
            logger.info("Entering stage 2: freeze encoder/quantizer, detach z_q, optimize decoder/post_quant_conv.")

        # ---------- Train loop ----------
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=ptdtype):
                recons_imgs, codebook_loss = vq_model(imgs)
                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps + 1,
                                   last_layer=vq_model.module.decoder.last_layer,
                                   logger=logger, log_every=args.log_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training
            optimizer_disc.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps + 1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            running_loss += loss_gen.item() + loss_disc.item()
            log_steps += 1
            train_steps += 1

            # logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0.0
                log_steps = 0
                start_time = time.time()

            # save ckpt
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "detach_quant": detach_quant,
                        "args": args,
                        # --- NEW: 保存 quantizer 的运行时信息以便恢复 ---
                        "stage2_fixed": vq_model.module.quantize.stage2_fixed,
                        "fixed_topk_indices": vq_model.module.quantize.fixed_topk_indices,
                        "code_counts": vq_model.module.quantize.code_counts,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()

                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    if cloud_checkpoint_dir is not None:
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    vq_model.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--data-face-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, required=False, help='cloud disk path, optional')
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True)
    parser.add_argument("--codebook-weight", type=float, default=1.0)
    parser.add_argument("--entropy-loss-ratio", type=float, default=1.0)
    parser.add_argument("--commit-loss-beta", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-loss", type=str, default='l2')
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-start", type=int, default=20000)
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan')
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge')
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge')
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")

    parser.add_argument("--dataset", type=str, default='aoss',
                        choices=['imagenet', 'aoss', 'imagenet_code', 'coco', 'openimage', 'pexels',
                                 't2i_image', 't2i', 't2i_code'])
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    # ---------- NEW ----------
    parser.add_argument("--expand-codebook-factor", type=int, default=4,
                        help="stage1 将 codebook size 放大倍数（默认 4）。从 ckpt 恢复时不强制放大。")

    args = parser.parse_args()
    main(args)
