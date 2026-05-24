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

    Args:
        base_dir: 基础目录路径
        name: 实验名称前缀

    Returns:
        新的实验目录路径
    """
    os.makedirs(base_dir, exist_ok=True)

    # 获取所有已存在的实验目录
    existing_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                existing_dirs.append(d)

    # 查找最大编号
    max_num = -1
    pattern = re.compile(rf"^{re.escape(name)}(_(\d+))?$")

    for dir_name in existing_dirs:
        match = pattern.match(dir_name)
        if match:
            if match.group(2) is None:  # 匹配 "encoder_training"
                max_num = max(max_num, 0)
            else:  # 匹配 "encoder_training_1", "encoder_training_2" 等
                num = int(match.group(2))
                max_num = max(max_num, num)

    # 创建新目录
    if max_num == -1:
        new_dir = os.path.join(base_dir, name)
    else:
        new_dir = os.path.join(base_dir, f"{name}_{max_num + 1}")

    os.makedirs(new_dir, exist_ok=True)
    return new_dir


class PatchMLPEncoder(nn.Module):
    """
    使用 MLP 处理 RGB patch 的编码器

    结构：
    1. Conv2d(3, hidden_dim, 2, 2) - 每个 2×2 区域输出 hidden_dim 维特征
    2. Tanh 激活
    3. Linear(hidden_dim, 1) - 映射到最终编码值

    对于 16×16 patch：
    - 经过 2×2 conv 后得到 8×8 的 feature map
    - 每个位置有 hidden_dim 个特征
    - Flatten 后得到 64 × hidden_dim 维向量
    - 通过 Linear 层映射到 64 维编码
    """

    def __init__(self, conv_kernel=2, hidden_dim=12, learnable=True):
        super().__init__()

        # 第一层：卷积提取特征
        # 输入: [batch, 3, 16, 16]
        # 输出: [batch, hidden_dim, 8, 8]
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=conv_kernel, stride=conv_kernel, bias=False)

        # Tanh 激活函数
        self.activation = nn.Tanh()

        # 第二层：线性映射到最终编码
        # 每个 8×8 位置的 hidden_dim 维特征独立映射到 1 维
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

        # 初始化权重
        with torch.no_grad():
            # Conv 层初始化为 1/12（保持与原始版本一致）
            self.conv.weight.fill_(1.0 / 12.0)
            # Linear 层使用 Xavier 初始化
            nn.init.xavier_uniform_(self.linear.weight)

        # 设置是否可学习
        self.conv.weight.requires_grad = learnable
        self.linear.weight.requires_grad = learnable
        self.learnable = learnable
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        输入: [batch_size, 3, patch_size, patch_size]  # RGB patches
        输出: [batch_size, encoding_size]  # encoding_size = (patch_size/2)^2
        """
        batch_size = x.shape[0]

        # 1. 卷积提取特征
        x = self.conv(x)  # [batch_size, hidden_dim, 8, 8]

        # 2. Tanh 激活
        x = self.activation(x)  # [batch_size, hidden_dim, 8, 8]

        # 3. 重排维度以便线性层处理
        # [batch_size, hidden_dim, 8, 8] -> [batch_size, 8, 8, hidden_dim]
        x = x.permute(0, 2, 3, 1)

        # 4. 线性映射：每个位置的 hidden_dim 维特征 -> 1 维
        x = self.linear(x)  # [batch_size, 8, 8, 1]

        # 5. 去除最后一维并展平
        x = x.squeeze(-1)  # [batch_size, 8, 8]
        x = x.flatten(1)  # [batch_size, 64]

        return x


class GaussianKLLoss(nn.Module):
    """
    改进版损失函数：
    1. 原始KL散度损失（编码分布与标准高斯的KL散度）
    2. 新增：基于STE的正值个数估计的KL散度
    """

    def __init__(self, weight_original=1.0, weight_count=1.0):
        super().__init__()
        self.weight_original = weight_original  # 原始KL散度权重
        self.weight_count = weight_count  # 正值个数KL散度权重

    def forward(self, encodings):
        """
        encodings: [batch_size * n_patches, encoding_size]
        """
        # ===== 1. 原始KL散度：编码分布 vs 标准高斯 =====
        mean = encodings.mean(dim=0)  # [encoding_size]
        var = encodings.var(dim=0, unbiased=False)  # [encoding_size]

        # KL散度: D_KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl_div_original = 0.5 * (var + mean ** 2 - 1 - torch.log(var + 1e-8))
        kl_loss_original = kl_div_original.mean()

        # ===== 2. 基于STE的个数估计KL散度 =====
        encoding_size = encodings.shape[1]
        binary_values = torch.where(encodings > 0, torch.ones_like(encodings), -torch.ones_like(encodings))
        binary_approx = encodings + (binary_values - encodings).detach()
        approx_sum = binary_approx.sum(dim=1)
        approx_pos_count = (approx_sum + encoding_size) / 2.0
        approx_neg_count = (encoding_size - approx_sum) / 2.0
        all_approx_counts = torch.cat([approx_pos_count, approx_neg_count])  # [2*n_patches]

        # 计算估计个数分布与目标高斯分布的KL散度
        # 目标：N(32, 16)，因为编码大小64，期望32个正值，方差=64*0.5*0.5=16
        count_mean = all_approx_counts.mean()
        count_var = all_approx_counts.var(unbiased=False)

        target_mean = encoding_size / 2.0  # 32.0
        target_var = encoding_size * 0.25  # 16.0

        # KL散度: D_KL(N(μ1, σ1²) || N(μ2, σ2²))
        # = 0.5 * (σ1²/σ2² + (μ1-μ2)²/σ2² - 1 + log(σ2²/σ1²))
        kl_loss_count = 0.5 * (
                count_var / target_var +
                (count_mean - target_mean) ** 2 / target_var -
                1 +
                torch.log(torch.tensor(target_var) / (count_var + 1e-8))
        )

        # ===== 3. 总损失 =====
        total_loss = (self.weight_original * kl_loss_original +
                      self.weight_count * kl_loss_count)

        # ===== 统计信息 =====
        pos_counts = (binary_values > 0).sum(dim=1).float()  # 真实正值个数

        stats_dict = {
            'total_loss': total_loss.item(),
            'kl_loss_original': kl_loss_original.item(),
            'kl_loss_count': kl_loss_count.item(),
            # 真实统计
            'pos_mean': pos_counts.mean().item(),
            'pos_std': pos_counts.std().item(),
            'pos_min': pos_counts.min().item(),
            'pos_max': pos_counts.max().item(),
            # 估计统计
            'estimated_count_mean': count_mean.item(),
            'estimated_count_std': torch.sqrt(count_var).item(),
            'estimated_pos_mean': approx_pos_count.mean().item(),
            'estimated_neg_mean': approx_neg_count.mean().item(),
            # 分布统计
            'output_mean': mean.mean().item(),
            'output_var': var.mean().item(),
            'output_std': torch.sqrt(var).mean().item(),
            # 整体统计
            'p_positive': (encodings > 0).float().mean().item(),
            'avg_abs_value': torch.abs(encodings).mean().item(),
        }

        return total_loss, stats_dict


def analyze_distribution(encoder, data_loader, device, n_batches=10,
                         phase="before", save_data=None):
    """
    分析编码器输出的分布特性（增强版，包含估计个数分析）
    """
    encoder.eval()

    all_positive_counts = []
    all_patch_outputs = []
    all_estimated_counts = []  # 新增

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

                # 真实正值个数
                positive_counts = (encodings > 0).sum(dim=1).cpu().numpy()
                all_positive_counts.extend(positive_counts)

                # 保存输出值
                all_patch_outputs.append(encodings.cpu().numpy())

                # 估计个数
                estimated_positive = positive_counts
                estimated_negative = encoding_size - positive_counts
                all_estimated_counts.extend(estimated_positive)
                all_estimated_counts.extend(estimated_negative)

    all_positive_counts = np.array(all_positive_counts)
    all_patch_outputs = np.concatenate(all_patch_outputs, axis=0).flatten()
    all_estimated_counts = np.array(all_estimated_counts)

    # 打印统计信息
    print(f"\n{'=' * 80}")
    print(f"Distribution Analysis - {phase.upper()}")
    print(f"{'=' * 80}")
    print(f"Total patches: {len(all_positive_counts):,}")
    print(f"\nTrue Positive Count Statistics:")
    print(f"  Mean: {all_positive_counts.mean():.4f} (target: 32.0)")
    print(f"  Std:  {all_positive_counts.std():.4f} (target: 4.0)")
    print(f"  Min:  {all_positive_counts.min()}")
    print(f"  Max:  {all_positive_counts.max()}")

    print(f"\nEstimated Count Statistics:")
    print(f"  Mean: {all_estimated_counts.mean():.4f} (target: 32.0)")
    print(f"  Std:  {all_estimated_counts.std():.4f} (target: 4.0)")

    p_positive = (all_patch_outputs > 0).mean()
    print(f"\nOutput Value Statistics:")
    print(f"  P(X>0): {p_positive:.6f} (target: 0.5)")
    print(f"  Mean:   {all_patch_outputs.mean():.6f} (target: 0.0)")
    print(f"  Std:    {all_patch_outputs.std():.6f} (target: 1.0)")
    print(f"  |x|_avg: {np.abs(all_patch_outputs).mean():.6f}")

    if save_data is not None:
        save_data[phase] = {
            'positive_counts': all_positive_counts,
            'patch_outputs': all_patch_outputs,
            'estimated_counts': all_estimated_counts,
            'encoding_size': encoding_size
        }

    return all_positive_counts, all_patch_outputs, all_estimated_counts


def save_plot_data(before_data, after_data, save_path):
    """保存绘图数据到文件"""
    plot_data = {
        'before': before_data,
        'after': after_data,
        'metadata': {
            'encoding_size': before_data['encoding_size'],
            'target_p': 0.5,
            'log_2': np.log(2),
            'sqrt_pi_over_2': np.sqrt(np.pi / 2)
        }
    }

    with open(save_path, 'wb') as f:
        pickle.dump(plot_data, f)

    print(f"Plot data saved to: {save_path}")


def load_and_plot(data_path, save_path):
    """从文件加载数据并绘图"""
    with open(data_path, 'rb') as f:
        plot_data = pickle.load(f)

    before_data = plot_data['before']
    after_data = plot_data['after']

    plot_comparison(before_data, after_data, save_path)


def plot_comparison(before_data, after_data, save_path, epoch_info=""):
    """
    对比训练前后的分布（增强版，包含估计个数分布）

    Args:
        before_data: 训练前数据
        after_data: 训练后/当前epoch数据
        save_path: 保存路径
        epoch_info: epoch信息字符串（如 "Epoch 1"）
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    phases = ['before', 'after']
    colors = ['steelblue', 'coral']
    phase_labels = ['Before Training', epoch_info if epoch_info else 'After Training']

    for idx, (phase, color, label) in enumerate(zip(phases, colors, phase_labels)):
        data = before_data if phase == 'before' else after_data
        positive_counts = data['positive_counts']
        patch_outputs = data['patch_outputs']
        estimated_counts = data['estimated_counts']
        encoding_size = data['encoding_size']

        p_positive = (patch_outputs > 0).mean()

        # 第一行: 真实正值个数分布
        ax1 = fig.add_subplot(gs[0, idx * 2:idx * 2 + 2])

        counts, bins, _ = ax1.hist(
            positive_counts,
            bins=np.arange(0, encoding_size + 2) - 0.5,
            density=True,
            alpha=0.7,
            color=color,
            edgecolor='black',
            label=f'Observed ({label})'
        )

        x_binom = np.arange(0, encoding_size + 1)
        binom_pmf = stats.binom.pmf(x_binom, encoding_size, 0.5)
        ax1.plot(x_binom, binom_pmf, 'go-', linewidth=2, markersize=4,
                 label=f'Target Binomial(64, 0.5)', alpha=0.8)

        actual_binom_pmf = stats.binom.pmf(x_binom, encoding_size, p_positive)
        ax1.plot(x_binom, actual_binom_pmf, 'ro--', linewidth=2, markersize=3,
                 label=f'Actual Binomial(64, {p_positive:.3f})', alpha=0.8)

        ax1.axvline(32, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax1.axvline(positive_counts.mean(), color='red', linestyle='-',
                    linewidth=2, label=f'Mean: {positive_counts.mean():.2f}')

        ax1.set_xlabel('True Number of Positive Values per Patch', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'True Positive Count Distribution - {label}',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # 第二行: 估计个数分布
        ax2 = fig.add_subplot(gs[1, idx * 2:idx * 2 + 2])

        ax2.hist(estimated_counts, bins=100, density=True, alpha=0.7,
                 color=color, edgecolor='black',
                 label=f'Estimated Counts ({label})')

        # 目标高斯分布 N(32, 16)
        x_gauss = np.linspace(estimated_counts.min(), estimated_counts.max(), 500)
        target_gauss = stats.norm.pdf(x_gauss, 32, 4)
        ax2.plot(x_gauss, target_gauss, 'g-', linewidth=2.5,
                 label='Target N(32, 4²)', alpha=0.9)

        ax2.axvline(32, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax2.axvline(estimated_counts.mean(), color='red', linestyle='-',
                    linewidth=2, label=f'Mean: {estimated_counts.mean():.2f}')

        ax2.set_xlabel('Estimated Count (positive + negative)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'Estimated Count Distribution - {label}',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        textstr = (f'Mean={estimated_counts.mean():.2f}\n'
                   f'Std={estimated_counts.std():.2f}\n'
                   f'Target: μ=32, σ=4')
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 第三行: 输出值分布
        ax3 = fig.add_subplot(gs[2, idx * 2:idx * 2 + 2])

        ax3.hist(patch_outputs, bins=150, density=True, alpha=0.7,
                 color=color, edgecolor='black', label=f'Raw Outputs ({label})')

        x_range = np.linspace(patch_outputs.min(), patch_outputs.max(), 500)
        standard_gaussian = stats.norm.pdf(x_range, 0, 1)
        ax3.plot(x_range, standard_gaussian, 'g-', linewidth=2.5,
                 label='Standard Gaussian N(0,1)', alpha=0.9)

        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('Output Value', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title(f'Output Distribution - {label}',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)

        ks_stat, ks_p = stats.kstest(patch_outputs, 'norm', args=(0, 1))
        textstr = (f'K-S Test:\nstat={ks_stat:.4f}\np={ks_p:.2e}\n'
                   f'Mean={patch_outputs.mean():.4f}\n'
                   f'Std={patch_outputs.std():.4f}')
        ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 第四行: Q-Q图
        ax4 = fig.add_subplot(gs[3, idx * 2:idx * 2 + 2])

        sample_size = min(5000, len(patch_outputs))
        sample_indices = np.random.choice(len(patch_outputs), sample_size, replace=False)
        sample_data = patch_outputs[sample_indices]

        stats.probplot(sample_data, dist="norm", plot=ax4)
        ax4.set_title(f'Q-Q Plot - {label}', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)

        theoretical_quantiles = np.array([x[0] for x in ax4.lines[0].get_data()])
        sample_quantiles = np.array([x[1] for x in ax4.lines[0].get_data()])
        r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
        ax4.text(0.05, 0.95, f'R² = {r_squared:.6f}', transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

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

        # 收集所有patches
        all_patches = []
        for b in range(batch_size):
            image = images[b:b + 1]
            patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
            patches = patches.squeeze(0).transpose(0, 1)
            patches = patches.reshape(n_patches_per_image, 3, patch_size, patch_size)
            all_patches.append(patches)

        all_patches = torch.cat(all_patches, dim=0)

        # 前向传播
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            encodings = encoder(all_patches)
            loss, stats_dict = criterion(encodings)

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 累积统计
        running_loss += loss.item()
        for key, value in stats_dict.items():
            running_stats[key] = running_stats.get(key, 0.0) + value

        # 日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_stats = {k: v / log_interval for k, v in running_stats.items()}

            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed

            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(data_loader)}] "
                f"Loss: {avg_loss:.6f} | "
                f"KL_orig: {avg_stats['kl_loss_original']:.6f} | "
                f"KL_count: {avg_stats['kl_loss_count']:.6f} | "
                f"Mean: {avg_stats['output_mean']:.4f} | "
                f"Std: {avg_stats['output_std']:.4f} | "
                f"+True: {avg_stats['pos_mean']:.2f} | "
                f"+Est: {avg_stats['estimated_count_mean']:.2f} | "
                f"P_pos: {avg_stats['p_positive']:.4f} | "
                f"Samples/s: {samples_per_sec:.1f}"
            )

            running_loss = 0.0
            running_stats = {}


def main(args):
    """主训练函数"""
    assert torch.cuda.is_available(), "Training requires GPU"

    # Setup DDP
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

    # Setup experiment folder（使用自动编号）
    if rank == 0:
        experiment_dir = get_experiment_dir(args.results_dir, "encoder_training")
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Running in {'distributed' if args.distributed else 'single-GPU'} mode")
        logger.info(f"Using MLP encoder with Tanh activation (hidden_dim={args.hidden_dim})")
        logger.info(f"Using enhanced loss with estimated count KL divergence")
    else:
        experiment_dir = None
        logger = create_logger(None)

    logger.info(f"Args: {args}")

    # Create model (使用 MLP 编码器)
    encoder = PatchMLPEncoder(
        conv_kernel=2,
        hidden_dim=args.hidden_dim,
        learnable=True
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder Parameters: {total_params:,}")
    logger.info(f"Conv weight shape: {encoder.conv.weight.shape}")  # [hidden_dim, 3, 2, 2]
    logger.info(f"Linear weight shape: {encoder.linear.weight.shape}")  # [1, hidden_dim]

    # Create loss（增强版）
    criterion = GaussianKLLoss(
        weight_original=args.weight_original,
        weight_count=args.weight_count
    ).to(device)

    logger.info(f"Loss weights - Original KL: {args.weight_original}, Count KL: {args.weight_count}")

    # Setup optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # Setup data
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

    # ===== 关键修改：先保存原始模型用于分析 =====
    encoder_module = encoder  # 保存原始模型引用，用于分析

    # 然后再包装成 DDP
    if args.distributed:
        encoder = DDP(encoder, device_ids=[device])

    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # ===== 训练前分析（使用原始模型，只在 rank 0 执行）=====
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("BEFORE TRAINING - Distribution Analysis")
        logger.info("=" * 80)

        before_data = {}
        # 使用原始模型（非 DDP）进行分析
        analyze_distribution(
            encoder_module,
            loader,
            device,
            n_batches=10,
            phase="before",
            save_data=before_data
        )

    # ===== 同步所有进程 =====
    if args.distributed:
        dist.barrier()

    # ===== 训练循环 =====
    logger.info("\n" + "=" * 80)
    logger.info("START TRAINING")
    logger.info("=" * 80)

    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练一个epoch（这里 encoder 已经是 DDP 包装后的）
        train_one_epoch(
            encoder.module if args.distributed else encoder,  # 传递原始模型
            criterion,
            optimizer,
            loader,
            device,
            epoch + 1,
            logger,
            scaler if args.mixed_precision == 'fp16' else None
        )

        # ===== 每个epoch结束后分析并保存对比图（只在 rank 0）=====
        if rank == 0:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch + 1} - Distribution Analysis")
            logger.info(f"{'=' * 80}")

            # 分析当前epoch的分布
            current_epoch_data = {}
            # 使用原始模型（非 DDP）
            analyze_distribution(
                encoder.module if args.distributed else encoder,
                loader,
                device,
                n_batches=10,
                phase=f"epoch_{epoch + 1}",
                save_data=current_epoch_data
            )

            # 保存当前epoch的绘图数据
            epoch_plot_data_path = os.path.join(
                experiment_dir,
                f"plot_data_epoch_{epoch + 1}.pkl"
            )
            save_plot_data(
                before_data['before'],
                current_epoch_data[f"epoch_{epoch + 1}"],
                epoch_plot_data_path
            )

            # 绘制对比图
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

        # ===== 同步所有进程 =====
        if args.distributed:
            dist.barrier()

    # ===== 训练结束后的最终分析 =====
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL ANALYSIS - After All Training")
        logger.info("=" * 80)

        # 最终分析
        final_data = {}
        analyze_distribution(
            encoder.module if args.distributed else encoder,
            loader,
            device,
            n_batches=10,
            phase="final",
            save_data=final_data
        )

        # 保存最终绘图数据
        final_plot_data_path = os.path.join(experiment_dir, "plot_data_final.pkl")
        save_plot_data(
            before_data['before'],
            final_data['final'],
            final_plot_data_path
        )

        # 绘制最终对比图
        final_plot_path = os.path.join(experiment_dir, "distribution_final.png")
        plot_comparison(
            before_data['before'],
            final_data['final'],
            final_plot_path,
            epoch_info=f"After Training (Final)"
        )

        # 保存模型
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

    # Data args
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)

    # Training args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--mixed-precision", type=str, default='bf16',
                        choices=["none", "fp16", "bf16"])

    # Model args (新增)
    parser.add_argument("--hidden-dim", type=int, default=12,
                        help="Hidden dimension for MLP encoder")

    # Loss weights
    parser.add_argument("--weight-original", type=float, default=1.0,
                        help="Weight for original KL divergence loss")
    parser.add_argument("--weight-count", type=float, default=1.0,
                        help="Weight for estimated count KL divergence loss")

    # Save args
    parser.add_argument("--results-dir", type=str, default="results_encoder")

    args = parser.parse_args()
    main(args)
