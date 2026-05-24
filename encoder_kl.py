# train_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import time
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import re

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset

import warnings

warnings.filterwarnings('ignore')

def get_experiment_dir(base_dir, name="encoder_training"):
    """
    自动创建带编号的实验目录
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                existing_dirs.append(d)
    max_num = -1
    pattern = re.compile(rf"^{re.escape(name)}(_(\d+))?$")
    for dir_name in existing_dirs:
        match = pattern.match(dir_name)
        if match:
            if match.group(2) is None:
                max_num = max(max_num, 0)
            else:
                num = int(match.group(2))
                max_num = max(max_num, num)
    if max_num == -1:
        new_dir = os.path.join(base_dir, name)
    else:
        new_dir = os.path.join(base_dir, f"{name}_{max_num + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

class PatchMLPEncoder(nn.Module):
    """
    使用 MLP 处理 RGB patch 的编码器
    """
    def __init__(self, conv_kernel=2, hidden_dim=12, learnable=True):
        super().__init__()
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=conv_kernel, stride=conv_kernel, bias=False)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 12.0)
            nn.init.xavier_uniform_(self.linear.weight)
        self.conv.weight.requires_grad = learnable
        self.linear.weight.requires_grad = learnable
        self.learnable = learnable
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        x = x.flatten(1)
        return x

class BinaryKLLoss(nn.Module):
    """
    改进版损失函数：
    1. 对编码器输出应用 STE：x -> x + (binary - x).detach()，其中 binary = sign(x)
    2. 计算每个 patch 的二值均值
    3. 使 patch 均值服从高斯分布 N(0, 1/16)
    4. 使用 STE 后的 binary_values 计算 p_positive 和 p_negative，保留梯度
    """
    def __init__(self, weight_mean=1.0, weight_binomial=1.0, encoding_size=64):
        super().__init__()
        self.weight_mean = weight_mean
        self.weight_binomial = weight_binomial
        self.encoding_size = encoding_size

    def forward(self, encodings):
        """
        encodings: [batch_size * n_patches, encoding_size]
        encoding_size = 64
        """
        # 1. 应用 STE
        binary_values = torch.where(encodings > 0,
                                   torch.ones_like(encodings),
                                   -torch.ones_like(encodings))
        encodings_ste = encodings + (binary_values - encodings).detach()

        # 2. 计算每个 patch 的均值
        patch_means = encodings_ste.mean(dim=1)  # [batch_size * n_patches]

        # 3. patch 均值 KL 散度：N(μ, σ²) || N(0, 1/16)
        mean = patch_means.mean()
        var = patch_means.var(unbiased=False)
        target_var = 1.0 / 16.0  # σ² = 1/16
        target_mean = 0.0
        kl_div_mean = 0.5 * (
            var / target_var +
            (mean - target_mean) ** 2 / target_var -
            1 +
            torch.log(torch.tensor(target_var) / (var + 1e-8))
        )

        # 4. 使用 binary_values 计算 p_positive 和 p_negative
        total_elements = encodings.numel()
        sum_binary = binary_values.sum()
        p_positive = (sum_binary + total_elements) / (2.0 * total_elements)
        p_negative = (total_elements - sum_binary) / (2.0 * total_elements)
        target_p = 0.5
        kl_div_binomial = 0.5 * (
            p_positive * torch.log(p_positive / target_p + 1e-8) +
            p_negative * torch.log(p_negative / target_p + 1e-8)
        )

        # 5. 总损失
        total_loss = self.weight_mean * kl_div_mean + self.weight_binomial * kl_div_binomial

        # 6. 统计信息
        stats_dict = {
            'total_loss': total_loss.item(),
            'kl_div_mean': kl_div_mean.item(),
            'kl_div_binomial': kl_div_binomial.item(),
            'patch_mean_mean': patch_means.mean().item(),
            'patch_mean_std': patch_means.std().item(),
            'p_positive': p_positive.item(),
            'p_negative': p_negative.item(),
            'avg_abs_value': torch.abs(encodings_ste).mean().item(),  # 应为 1
        }

        return total_loss, stats_dict

def analyze_distribution(encoder, data_loader, device, n_batches=10,
                        phase="before", save_data=None):
    """
    分析编码器输出的分布特性
    """
    encoder.eval()
    all_patch_means = []
    all_patch_outputs = []
    patch_size = 16
    encoding_size = 64

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            images = images.to(device)
            batch_size = images.shape[0]
            image_size = images.shape[-1]
            n_patches_per_side = image_size // patch_size
            n_patches_per_image = n_patches_per_side ** 2

            for b in range(batch_size):
                image = images[b:b + 1]
                patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
                patches = patches.squeeze(0).transpose(0, 1)
                patches = patches.reshape(n_patches_per_image, 3, patch_size, patch_size)
                encodings = encoder(patches)  # [n_patches, encoding_size]

                # 应用 STE
                binary_values = np.where(encodings.cpu().numpy() > 0, 1, -1)
                patch_means = binary_values.mean(axis=1)
                all_patch_means.extend(patch_means)
                all_patch_outputs.append(binary_values)

    all_patch_means = np.array(all_patch_means)
    all_patch_outputs = np.concatenate(all_patch_outputs, axis=0).flatten()

    # 打印统计信息
    print(f"\n{'=' * 80}")
    print(f"Distribution Analysis - {phase.upper()}")
    print(f"{'=' * 80}")
    print(f"Total patches: {len(all_patch_means):,}")
    print(f"\nPatch Mean Statistics:")
    print(f"  Mean: {all_patch_means.mean():.4f} (target: 0.0)")
    print(f"  Std:  {all_patch_means.std():.4f} (target: 0.25)")
    print(f"  Min:  {all_patch_means.min():.4f}")
    print(f"  Max:  {all_patch_means.max():.4f}")
    p_positive = (all_patch_outputs > 0).mean()
    p_negative = (all_patch_outputs < 0).mean()
    print(f"\nOutput Value Statistics:")
    print(f"  P(X>0): {p_positive:.6f} (target: 0.5)")
    print(f"  P(X<0): {p_negative:.6f} (target: 0.5)")
    print(f"  Values: {np.unique(all_patch_outputs)} (target: [-1, 1])")

    if save_data is not None:
        save_data[phase] = {
            'patch_means': all_patch_means,
            'patch_outputs': all_patch_outputs,
            'encoding_size': encoding_size
        }

    return all_patch_means, all_patch_outputs

def save_plot_data(before_data, after_data, save_path):
    """保存绘图数据到文件"""
    plot_data = {
        'before': before_data,
        'after': after_data,
        'metadata': {
            'encoding_size': before_data['encoding_size'],
            'target_p': 0.5,
            'target_mean_std': 0.25
        }
    }
    with open(save_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to: {save_path}")

def plot_comparison(before_data, after_data, save_path, epoch_info=""):
    """
    对比训练前后的分布
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    phases = ['before', 'after']
    colors = ['steelblue', 'coral']
    phase_labels = ['Before Training', epoch_info if epoch_info else 'After Training']

    for idx, (phase, color, label) in enumerate(zip(phases, colors, phase_labels)):
        data = before_data if phase == 'before' else after_data
        patch_means = data['patch_means']
        patch_outputs = data['patch_outputs']
        encoding_size = data['encoding_size']
        p_positive = (patch_outputs > 0).mean()
        p_negative = (patch_outputs < 0).mean()

        # 第一行: Patch 均值分布
        ax1 = fig.add_subplot(gs[0, idx * 2:idx * 2 + 2])
        ax1.hist(patch_means, bins=50, density=True, alpha=0.7,
                 color=color, edgecolor='black', label=f'Patch Means ({label})')
        x_gauss = np.linspace(patch_means.min(), patch_means.max(), 500)
        target_gauss = stats.norm.pdf(x_gauss, 0, 0.25)
        ax1.plot(x_gauss, target_gauss, 'g-', linewidth=2.5,
                 label='Target N(0, 0.25²)', alpha=0.9)
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax1.axvline(patch_means.mean(), color='red', linestyle='-',
                    linewidth=2, label=f'Mean: {patch_means.mean():.2f}')
        ax1.set_xlabel('Patch Mean Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'Patch Mean Distribution - {label}', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        textstr = (f'Mean={patch_means.mean():.4f}\n'
                   f'Std={patch_means.std():.4f}\n'
                   f'Target: μ=0, σ=0.25')
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 第二行: 输出值分布
        ax2 = fig.add_subplot(gs[1, idx * 2:idx * 2 + 2])
        values, counts = np.unique(patch_outputs, return_counts=True)
        counts = counts / counts.sum()
        ax2.bar(values, counts, width=0.4, alpha=0.7, color=color,
                edgecolor='black', label=f'Outputs ({label})')
        ax2.axhline(0.5, color='green', linestyle='--', linewidth=2,
                    label='Target P=0.5', alpha=0.5)
        ax2.set_xlabel('Output Value', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title(f'Output Distribution - {label}', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        textstr = (f'P(X>0)={p_positive:.4f}\n'
                   f'P(X<0)={p_negative:.4f}\n'
                   f'Target P=0.5')
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison figure saved to: {save_path}")
    plt.close()

def train_one_epoch(encoder, criterion, optimizer, data_loader, device,
                    epoch, logger, scaler=None):
    """训练一个epoch"""
    encoder.train()
    patch_size = 16
    image_size = 256
    n_patches_per_side = image_size // patch_size
    n_patches_per_image = n_patches_per_side ** 2
    running_loss = 0.0
    running_stats = {}
    log_interval = 100
    start_time = None
    total_samples = 0

    for batch_idx, (images, _) in enumerate(data_loader):
        if start_time is None:
            start_time = time.time()
        images = images.to(device)
        batch_size = images.shape[0]
        total_samples += batch_size

        all_patches = []
        for b in range(batch_size):
            image = images[b:b + 1]
            patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
            patches = patches.squeeze(0).transpose(0, 1)
            patches = patches.reshape(n_patches_per_image, 3, patch_size, patch_size)
            all_patches.append(patches)
        all_patches = torch.cat(all_patches, dim=0)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            encodings = encoder(all_patches)
            loss, stats_dict = criterion(encodings)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        for key, value in stats_dict.items():
            running_stats[key] = running_stats.get(key, 0.0) + value

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_stats = {k: v / log_interval for k, v in running_stats.items()}
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(data_loader)}] "
                f"Loss: {avg_loss:.6f} | "
                f"KL_mean: {avg_stats['kl_div_mean']:.6f} | "
                f"KL_binomial: {avg_stats['kl_div_binomial']:.6f} | "
                f"Patch Mean: {avg_stats['patch_mean_mean']:.4f} | "
                f"Patch Std: {avg_stats['patch_mean_std']:.4f} | "
                f"P_pos: {avg_stats['p_positive']:.4f} | "
                f"P_neg: {avg_stats['p_negative']:.4f} | "
                f"Samples/s: {samples_per_sec:.1f}"
            )
            running_loss = 0.0
            running_stats = {}

def main(args):
    assert torch.cuda.is_available(), "Training requires GPU"
    init_distributed_mode(args)
    if hasattr(args, 'distributed') and args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        args.gpu = 0
        args.distributed = False

    device = args.gpu if hasattr(args, 'gpu') else 0
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        experiment_dir = get_experiment_dir(args.results_dir, "encoder_training")
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Running in {'distributed' if args.distributed else 'single-GPU'} mode")
        logger.info(f"Using MLP encoder with Tanh activation (hidden_dim={args.hidden_dim})")
        logger.info(f"Using STE loss with patch mean KL and binomial KL (with gradient-preserving p_positive/p_negative)")
    else:
        experiment_dir = None
        logger = create_logger(None)

    logger.info(f"Args: {args}")

    encoder = PatchMLPEncoder(
        conv_kernel=2,
        hidden_dim=args.hidden_dim,
        learnable=True
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder Parameters: {total_params:,}")
    logger.info(f"Conv weight shape: {encoder.conv.weight.shape}")
    logger.info(f"Linear weight shape: {encoder.linear.weight.shape}")

    criterion = BinaryKLLoss(
        weight_mean=args.weight_mean,
        weight_binomial=args.weight_binomial,
        encoding_size=64
    ).to(device)

    logger.info(f"Loss weights - Mean KL: {args.weight_mean}, Binomial KL: {args.weight_binomial}")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = build_dataset(args, transform=transform)

    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(dataset):,} images")

    encoder_module = encoder
    if args.distributed:
        encoder = DDP(encoder, device_ids=[device])

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("BEFORE TRAINING - Distribution Analysis")
        logger.info("=" * 80)
        before_data = {}
        analyze_distribution(
            encoder_module,
            loader,
            device,
            n_batches=10,
            phase="before",
            save_data=before_data
        )

    if args.distributed:
        dist.barrier()

    logger.info("\n" + "=" * 80)
    logger.info("START TRAINING")
    logger.info("=" * 80)

    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_one_epoch(
            encoder.module if args.distributed else encoder,
            criterion,
            optimizer,
            loader,
            device,
            epoch + 1,
            logger,
            scaler if args.mixed_precision == 'fp16' else None
        )

        if rank == 0:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch + 1} - Distribution Analysis")
            logger.info(f"{'=' * 80}")
            current_epoch_data = {}
            analyze_distribution(
                encoder.module if args.distributed else encoder,
                loader,
                device,
                n_batches=10,
                phase=f"epoch_{epoch + 1}",
                save_data=current_epoch_data
            )
            epoch_plot_data_path = os.path.join(
                experiment_dir,
                f"plot_data_epoch_{epoch + 1}.pkl"
            )
            save_plot_data(
                before_data['before'],
                current_epoch_data[f"epoch_{epoch + 1}"],
                epoch_plot_data_path
            )
            epoch_plot_path = os.path.join(
                experiment_dir,
                f"distribution_epoch_{epoch + 1}.png"
            )
            plot_comparison(
                before_data['before'],
                current_epoch_data[f"epoch_{epoch + 1}"],
                epoch_plot_path,
                epoch_info=f"After Epoch {epoch + 1}"
            )
            logger.info(f"Epoch {epoch + 1} analysis completed and saved.")

        if args.distributed:
            dist.barrier()

    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL ANALYSIS - After All Training")
        logger.info("=" * 80)
        final_data = {}
        analyze_distribution(
            encoder.module if args.distributed else encoder,
            loader,
            device,
            n_batches=10,
            phase="final",
            save_data=final_data
        )
        final_plot_data_path = os.path.join(experiment_dir, "plot_data_final.pkl")
        save_plot_data(
            before_data['before'],
            final_data['final'],
            final_plot_data_path
        )
        final_plot_path = os.path.join(experiment_dir, "distribution_final.png")
        plot_comparison(
            before_data['before'],
            final_data['final'],
            final_plot_path,
            epoch_info=f"After Training (Final)"
        )
        checkpoint_path = os.path.join(experiment_dir, "encoder_final.pt")
        torch.save({
            'model': (encoder.module if args.distributed else encoder).state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'epoch': args.epochs
        }, checkpoint_path)
        logger.info(f"Model saved to: {checkpoint_path}")
        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info(f"Total {args.epochs} comparison plots saved")
        logger.info(f"Results saved in: {experiment_dir}")
        logger.info("=" * 80)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--mixed-precision", type=str, default='bf16',
                        choices=["none", "fp16", "bf16"])
    parser.add_argument("--hidden-dim", type=int, default=12,
                        help="Hidden dimension for MLP encoder")
    parser.add_argument("--weight-mean", type=float, default=1.0,
                        help="Weight for patch mean KL divergence loss")
    parser.add_argument("--weight-binomial", type=float, default=1.0,
                        help="Weight for binomial KL divergence loss")
    parser.add_argument("--results-dir", type=str, default="results_encoder")
    args = parser.parse_args()
    main(args)
