import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import numpy as np
import math

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from autoregressive.models.gpt_continuous import ContinuousTransformer, GPT_continuous_models


class DCAEDirectCodeDataset(Dataset):
    def __init__(self, code_dir, label_dir):
        self.code_dir = code_dir
        self.label_dir = label_dir
        self.num_files = len([f for f in os.listdir(code_dir) if f.endswith('.npy')])

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        codes = np.load(os.path.join(self.code_dir, f"{idx}.npy"))
        aug_idx = torch.randint(low=0, high=codes.shape[0], size=(1,)).item()
        codes = codes[aug_idx]

        if codes.ndim == 3:
            codes = codes.transpose(1, 2, 0)
            codes = codes.reshape(-1, codes.shape[2])

        codes = codes.astype(np.float32) * 2.0 - 1.0

        labels = np.load(os.path.join(self.label_dir, f"{idx}.npy"))
        return torch.from_numpy(codes), torch.from_numpy(labels)


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
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
        checkpoint_dir = None
        cloud_checkpoint_dir = None

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.block_size is None:
        spatial_size = args.image_size // args.downsample_size
        args.block_size = spatial_size * spatial_size

    grid_size = int(math.sqrt(args.block_size))
    assert grid_size * grid_size == args.block_size, \
        f"block_size={args.block_size} must be a perfect square for 2D RoPE, got grid_size={grid_size}"

    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p

    model = GPT_continuous_models[args.gpt_model](
        vocab_size=1,
        block_size=args.block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        codebook_dim=args.codebook_dim,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Continuous Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"codebook_dim={args.codebook_dim}, block_size={args.block_size}, grid_size={grid_size}x{grid_size}")

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    code_dir = f"{args.code_path}/imagenet{args.image_size}_codes_dcae_direct"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(code_dir) and os.path.exists(label_dir), \
        f"Code dir {code_dir} or label dir {label_dir} does not exist. Please run extract_codes_dcae_direct.py first."

    dataset = DCAEDirectCodeDataset(code_dir, label_dir)

    if args.eval_every > 0:
        val_split = int(len(dataset) * 0.01)
        train_split = len(dataset) - val_split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_split, val_split],
            generator=torch.Generator().manual_seed(args.global_seed)
        )
        logger.info(f"Train/Val split: {train_split}/{val_split}")
    else:
        train_dataset = dataset
        val_dataset = None

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.code_path})")

    val_loader = None
    if val_dataset is not None and args.eval_every > 0:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Validation dataset contains {len(val_dataset):,} images")

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model)

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()
    if args.ema:
        ema.eval()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_codes = x.reshape(x.shape[0], -1, args.codebook_dim)
            c_indices = y.reshape(-1)
            assert z_codes.shape[0] == c_indices.shape[0]
            with torch.cuda.amp.autocast(dtype=ptdtype):
                _, loss = model(cond_idx=c_indices, idx=z_codes[:, :-1, :], targets=z_codes)
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
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

        if val_loader is not None and (epoch + 1) % args.eval_every == 0:
            eval_model = ema if args.ema else (model.module._orig_mod if not args.no_compile else model.module)
            eval_model.eval()
            val_loss = 0.0
            val_steps = 0
            sign_acc = 0.0
            total_elements = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    z_codes = x.reshape(x.shape[0], -1, args.codebook_dim)
                    c_indices = y.reshape(-1)
                    with torch.cuda.amp.autocast(dtype=ptdtype):
                        output, loss = eval_model(cond_idx=c_indices, idx=z_codes[:, :-1, :], targets=z_codes)
                    val_loss += loss.item()
                    val_steps += 1
                    pred_sign = torch.sign(output)
                    target_sign = z_codes[:, 1:, :]
                    sign_acc += (pred_sign == target_sign).float().sum().item()
                    total_elements += target_sign.numel()

            val_loss_avg = torch.tensor(val_loss / max(val_steps, 1), device=device)
            dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
            val_loss_avg = val_loss_avg.item() / dist.get_world_size()

            sign_acc_total = torch.tensor(sign_acc, device=device)
            total_elements_total = torch.tensor(total_elements, device=device)
            dist.all_reduce(sign_acc_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_elements_total, op=dist.ReduceOp.SUM)
            sign_acc_pct = sign_acc_total.item() / max(total_elements_total.item(), 1) * 100

            logger.info(f"(epoch={epoch:04d}) Val Loss: {val_loss_avg:.4f}, Sign Accuracy: {sign_acc_pct:.2f}%")
            eval_model.train()

    model.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_continuous_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--codebook-dim", type=int, default=128, help="dimension of binary code per spatial position from DCAE")
    parser.add_argument("--block-size", type=int, default=None,
                        help="total sequence length. If None, auto-computed as (image_size/downsample_size)^2")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, default=32, help="downsample factor of the DCAE tokenizer")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=10, help="evaluate every N epochs (0 to disable)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)
