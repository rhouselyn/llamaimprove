import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import time
import argparse
from glob import glob
from copy import deepcopy

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.csq_model import VQModel, ModelArgs  # 调整为前缀tokenizer.tokenizer_image
from tokenizer.tokenizer_image.csq_loss import VQLoss  # 修改后的VQLoss

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
    # seed = args.global_seed * dist.get_world_size() + rank
    seed = 64
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = "CSQ_Model"
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQModel(
        ModelArgs(
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
            codebook_l2_norm=args.codebook_l2_norm,
            codebook_show_usage=True,
            patch_size=16,
            decoder_ch_mult=[1, 1, 2, 2, 4],
            z_channels=256,
            dropout_p=args.dropout_p,
        )
    )
    logger.info(f"CSQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"CSQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    vq_model = vq_model.to(device)

    # Setup data:
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

    # Handle ZCA only if not resuming from a full VQ checkpoint (which includes ZCA)
    if args.vq_ckpt is None:
        if args.zca_ckpt is not None:
            # Load precomputed ZCA checkpoint on rank 0 and broadcast
            if rank == 0:
                zca_checkpoint = torch.load(args.zca_ckpt, map_location="cpu")
                zca = vq_model.encoder.zca
                zca.sum_x.copy_(zca_checkpoint["sum_x"].to(device))
                zca.sum_xxT.copy_(zca_checkpoint["sum_xxT"].to(device))
                zca.count = zca_checkpoint["count"]
                zca.whitening_matrix.copy_(zca_checkpoint["whitening_matrix"].to(device))
                zca.mean.copy_(zca_checkpoint["mean"].to(device))
                logger.info(f"Loaded ZCA checkpoint from {args.zca_ckpt}")
            dist.barrier()

            # Broadcast ZCA parameters to all processes
            zca = vq_model.encoder.zca
            dist.broadcast(zca.sum_x, src=0)
            dist.broadcast(zca.sum_xxT, src=0)
            count_tensor = torch.tensor([zca.count], device=device, dtype=torch.long)
            dist.broadcast(count_tensor, src=0)
            zca.count = int(count_tensor.item())
            dist.broadcast(zca.whitening_matrix, src=0)
            dist.broadcast(zca.mean, src=0)

            logger.info(f"ZCA parameters loaded and synchronized.")
        else:
            # ZCA accumulation phase: Treat as encoder pre-training, over one full epoch
            logger.info("Starting ZCA stats accumulation over one full epoch...")
            sampler.set_epoch(0)  # Use epoch 0 for accumulation
            vq_model.encoder.eval()  # Encoder in eval mode for accumulation
            accum_steps = 0
            total_samples = 0
            accum_start_time = time.time()
            for x, y in loader:
                x = x.to(device)
                with torch.no_grad():
                    _ = vq_model.encoder(x, train_mode=True)  # Update ZCA stats
                accum_steps += 1
                total_samples += x.size(0) * dist.get_world_size()  # Approximate global samples

                if accum_steps % args.log_every == 0:
                    accum_time = time.time() - accum_start_time
                    steps_per_sec = accum_steps / accum_time
                    logger.info(f"(accum_step={accum_steps:07d}) Accumulated samples: {total_samples:,}, Steps/Sec: {steps_per_sec:.2f}")

            # Sync ZCA stats across processes
            zca = vq_model.encoder.zca
            sum_x_tensor = zca.sum_x.clone()
            sum_xxT_tensor = zca.sum_xxT.clone()
            count_tensor = torch.tensor([zca.count], device=device, dtype=torch.long)

            dist.all_reduce(sum_x_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_xxT_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            zca.sum_x.copy_(sum_x_tensor)
            zca.sum_xxT.copy_(sum_xxT_tensor)
            zca.count = int(count_tensor.item())

            # Compute whitening matrix on rank 0 and broadcast
            if rank == 0:
                if zca.count > 0:
                    zca.compute_whitening_matrix()
                    logger.info("ZCA whitening matrix computed.")
                else:
                    logger.warning("No samples accumulated for ZCA; skipping computation.")
            dist.barrier()

            # Broadcast whitening_matrix and mean to all processes
            dist.broadcast(zca.whitening_matrix, src=0)
            dist.broadcast(zca.mean, src=0)

            logger.info(f"ZCA accumulation completed. Total samples accumulated (global): {zca.count:,}")

            # Save ZCA checkpoint after accumulation
            if rank == 0:
                zca_checkpoint = {
                    "sum_x": zca.sum_x.cpu(),
                    "sum_xxT": zca.sum_xxT.cpu(),
                    "count": zca.count,
                    "whitening_matrix": zca.whitening_matrix.cpu(),
                    "mean": zca.mean.cpu()
                }
                if not args.no_local_save:
                    zca_save_path = f"{checkpoint_dir}/zca.pt"
                    torch.save(zca_checkpoint, zca_save_path)
                    logger.info(f"Saved ZCA checkpoint to {zca_save_path}")

                cloud_zca_path = f"{cloud_checkpoint_dir}/zca.pt"
                torch.save(zca_checkpoint, cloud_zca_path)
                logger.info(f"Saved ZCA checkpoint in cloud to {cloud_zca_path}")
            dist.barrier()

    # Now fix encoder parameters (already set requires_grad=False in model init)
    # 禁用encoder的参数梯度计算，以修复DDP reduction错误
    for param in vq_model.encoder.parameters():
        param.requires_grad = False

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
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    # Setup optimizer (only for codebook and decoder)
    params_to_optimize = list(vq_model.decoder.parameters()) + list(vq_model.quantize.embedding.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(
                args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model)  # requires PyTorch 2.0

    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                recons_imgs, diff = vq_model(imgs)  # diff = (0,0,0,usage)
                loss_gen = vq_loss(diff, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps + 1,
                                   last_layer=vq_model.module.decoder.last_layer,
                                   logger=logger, log_every=args.log_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module,
                           decay=0.9999)  # 假设使用渐进decay

            # discriminator training
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(diff, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps + 1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            # # Log loss values:
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
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
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
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, required=False,
                        help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true',
                        help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--zca-ckpt", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/tokenizer/tokenizer_image/results_tokenizer_image/020-CSQ_Model/checkpoints/zca.pt', help="Path to precomputed ZCA checkpoint")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0,
                        help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000,
                        help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan',
                        help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge',
                        help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge',
                        help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")

    # 在 if __name__ == "__main__": 的 parser 部分添加
    parser.add_argument("--dataset", type=str, default='aoss',
                        choices=['imagenet', 'aoss', 'imagenet_code', 'coco', 'openimage', 'pexels',
                                 't2i_image', 't2i', 't2i_code'])
    parser.add_argument("--aoss-bucket", type=str, default="imagenet",
                        help="AOSS bucket name (only for aoss_imagenet dataset)")

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)  # 调低学习率
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100000)

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)
