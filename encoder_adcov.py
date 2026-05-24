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
    自动创建带编号的experiment目录

    Args:
        base_dir: 基础目录路径
        name: 实验名称前缀

    Returns:
        新的experiment目录路径
    """
    os.makedirs(base_dir, exist_ok=True)

    # 获取所有已存在的experiment目录
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


class ImprovedGaussianKLLoss(nn.Module):
    """
    改进版损失函数：
    1. Anderson-Darling损失（每个维度独立计算AD统计量，平均）或原KL损失（根据epoch切换）
    2. Cov 损失（协方差矩阵非对角元素平方均值，鼓励维度独立）
    3. 新增：基于STE的归一化维度均值KL损失（mean-1/2 归一化版本，KL到 N(0, 0.25)），只在第一个epoch作用
    """
    def __init__(self, weight_original=1.0, weight_count=0.5, weight_cov=0.5, distributed=False):
        super().__init__()
        self.weight_original = weight_original  # 原KL或AD损失权重
        self.weight_count = weight_count  # 归一化KL权重
        self.weight_cov = weight_cov  # Cov损失权重
        self.distributed = distributed
        self.encoding_size = 64
        self.target_mean = 0.0
        self.target_var = 0.5
        self.target_count_mean = 0.0
        self.target_count_var = 0.25  # 1/4

    def forward(self, encodings, current_epoch=None, total_epochs=None):
        """
        encodings: [batch_size * n_patches, encoding_size]
        """
        N, D = encodings.shape

        # 计算均值和方差（用于原KL和标准化）
        mean = encodings.mean(dim=0)  # [D]
        var = encodings.var(dim=0, unbiased=False)  # [D]

        # ===== 1. 原KL损失或Anderson-Darling损失（根据epoch切换）=====
        kl_loss_original = torch.tensor(0.0, device=encodings.device)
        ad_loss = torch.tensor(0.0, device=encodings.device)

        if current_epoch == 1:
            # 原KL损失: D_KL(N(μ, σ²) || N(0, target_var))
            kl_div_original = (torch.log(self.target_var / (var + 1e-8)) + (var + mean ** 2) / self.target_var - 1) / 2
            kl_loss_original = kl_div_original.mean()
        else:
            # Anderson-Darling损失
            # 每个维度标准化
            mean_k = mean.unsqueeze(0)  # [1, D]
            std = torch.sqrt(var + 1e-8).unsqueeze(0)  # [1, D]
            x_std = (encodings - mean_k) / std  # [N, D]

            # 排序每个维度
            x_sorted, _ = torch.sort(x_std, dim=0)  # [N, D]

            # 标准正态CDF
            cdf = 0.5 * (1 + torch.erf(x_sorted / torch.sqrt(torch.tensor(2.0, device=encodings.device))))  # [N, D]
            cdf = torch.clamp(cdf, 1e-8, 1 - 1e-8)

            # 翻转CDF
            flipped_cdf = 1 - torch.flip(cdf, dims=[0])  # [N, D]

            # Log
            log_cdf = torch.log(cdf)  # [N, D]
            log_flipped = torch.log(flipped_cdf)  # [N, D]

            # i = arange(1, N+1)
            i = torch.arange(1, N + 1, dtype=encodings.dtype, device=encodings.device).unsqueeze(1)  # [N, 1]

            # 加权和
            weighted = (2 * i - 1) * (log_cdf + log_flipped)  # [N, D]
            sum_weighted = torch.sum(weighted, dim=0) / N  # [D]

            # AD统计量（每个维度）
            ad_stats = -N - sum_weighted  # [D]

            # 平均AD损失
            ad_loss = ad_stats.mean()

        # ===== 2. Cov损失：非对角协方差接近0 =====
        cov_matrix = torch.cov(encodings.T)  # [encoding_size, encoding_size]
        off_diag = cov_matrix - torch.diag_embed(cov_matrix.diagonal())
        cov_loss = torch.mean(off_diag ** 2)

        # ===== 3. 基于STE的归一化KL损失（只在第一个epoch作用）=====
        kl_loss_count = torch.tensor(0.0, device=encodings.device)
        count_mean = torch.tensor(0.0, device=encodings.device)
        count_var = torch.tensor(0.0, device=encodings.device)
        if current_epoch == 1:
            binary_values = torch.where(encodings > 0, torch.ones_like(encodings), -torch.ones_like(encodings))
            binary_approx = encodings + (binary_values - encodings).detach()
            approx_mean = binary_approx.mean(dim=1)  # [N]
            approx_shifted = approx_mean / 2  # 等价于 (prop - 0.5)
            approx_normalized = torch.sqrt(torch.tensor(self.encoding_size, device=encodings.device)) * approx_shifted  # 归一化
            count_mean = approx_normalized.mean()
            count_var = approx_normalized.var(unbiased=False)

            # KL散度: D_KL(N(μ1, σ1²) || N(μ2, σ2²))
            kl_loss_count = 0.5 * (
                count_var / self.target_count_var +
                (count_mean - self.target_count_mean) ** 2 / self.target_count_var -
                1 +
                torch.log(torch.tensor(self.target_count_var, device=encodings.device) / (count_var + 1e-8))
            )

        # ===== 总损失 =====
        total_loss = (self.weight_original * (kl_loss_original + ad_loss) +
                      self.weight_count * kl_loss_count +
                      self.weight_cov * cov_loss)

        # ===== 统计信息 =====
        binary_values = torch.where(encodings > 0, torch.ones_like(encodings), -torch.ones_like(encodings))
        pos_counts = (binary_values > 0).sum(dim=1).float()  # 真实正值个数

        stats_dict = {
            'total_loss': total_loss.item(),
            'kl_loss_original': kl_loss_original.item(),
            'ad_loss': ad_loss.item(),
            'kl_loss_count': kl_loss_count.item(),
            'cov_loss': cov_loss.item(),
            # 真实统计
            'pos_mean': pos_counts.mean().item(),
            'pos_std': pos_counts.std().item(),
            'pos_min': pos_counts.min().item(),
            'pos_max': pos_counts.max().item(),
            # 估计统计
            'estimated_count_mean': count_mean.item(),
            'estimated_count_std': torch.sqrt(count_var).item(),
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

            # 矢量化提取patches
            patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)  # [B, 3*16*16, n_patches_per_image]
            patches = patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)  # [B * n_patches, 3, 16, 16]

            encodings = encoder(patches)  # [B * n_patches, encoding_size]

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
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

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

        # 第二行: 输出值分布
        ax3 = fig.add_subplot(gs[1, idx * 2:idx * 2 + 2])

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

        # 第三行: Q-Q图
        ax4 = fig.add_subplot(gs[2, idx * 2:idx * 2 + 2])

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
                    epoch, logger, scaler=None, total_epochs=None):
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

        # 矢量化提取所有patches
        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)  # [B, 3*16*16, n_patches_per_image]
        patches = patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)  # [B * n_patches, 3, 16, 16]

        # 前向传播
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            encodings = encoder(patches)
            loss, stats_dict = criterion(encodings, current_epoch=epoch, total_epochs=total_epochs)

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
                f"AD_loss: {avg_stats['ad_loss']:.6f} | "
                f"KL_count: {avg_stats['kl_loss_count']:.6f} | "
                f"Cov_loss: {avg_stats['cov_loss']:.6f} | "
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

    # Create loss（增强版，使用新的 ImprovedGaussianKLLoss）
    criterion = ImprovedGaussianKLLoss(
        weight_original=args.weight_original,
        weight_count=args.weight_count,
        weight_cov=args.weight_cov,
        distributed=args.distributed
    ).to(device)

    logger.info(f"Loss weights - Original KL/AD: {args.weight_original}, Count KL: {args.weight_count}, Cov: {args.weight_cov}")

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
            scaler if args.mixed_precision == 'fp16' else None,
            total_epochs=args.epochs
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
    parser.add_argument("--data-path", type=str, default='imagenet_train_filelist.txt')
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
                        help="Weight for original KL/AD loss")
    parser.add_argument("--weight-count", type=float, default=0.5,
                        help="Weight for normalized count KL divergence loss")
    parser.add_argument("--weight-cov", type=float, default=0.5,
                        help="Weight for covariance loss")

    # Save args
    parser.add_argument("--results-dir", type=str, default="results_encoder")

    args = parser.parse_args()
    main(args)
