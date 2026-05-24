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
import torch.distributed as dist  # 保持原分布式导入
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import re

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset

import warnings

warnings.filterwarnings('ignore')

# 修改此处：使用不同的别名导入 distributions 模块，避免覆盖 dist
import torch.distributions as tdist  # 修改为 tdist (torch distributions)

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


class ImprovedLaplaceKLLoss(nn.Module):
    """
    改进版损失函数：
    1. KL损失（每个维度独立计算Laplace KL，平均）
    2. 新增：基于STE的归一化维度均值KL损失（mean-1/2 归一化版本，KL到 N(0, 0.25)），只在第一个epoch作用
    3. MI损失（通过MINE估计，加入总损失，鼓励独立性）
    4. 新增：尾部惩罚（对正值计数极端偏差的平方惩罚，加大尾部抑制）
    """

    def __init__(self, weight_original=1.0, weight_count=0.5, weight_cov=0.5, weight_tail=1.0, distributed=False, mine_model=None):
        super().__init__()
        self.weight_original = weight_original  # 原KL损失权重
        self.weight_count = weight_count  # 归一化KL权重
        self.weight_cov = weight_cov  # MI损失权重（复用，原为cov）
        self.weight_tail = weight_tail  # 尾部惩罚权重
        self.distributed = distributed
        self.mine_model = mine_model  # Pre-trained MINE model for MI loss
        self.encoding_size = 64
        self.target_mean = 0.0
        self.target_b = 1.0
        self.target_count_mean = 0.0
        self.target_count_var = 0.25  # 1/4
        self.target_binom_mean = 32.0  # Binomial mean
        self.tail_threshold = 28.0  # 7*std (std=4 for Binom(64,0.5))
        # 新增：目标二项分布参数
        self.target_p = 0.5  # Bernoulli概率p=0.5，确保P(pos>0)=0.5
        self.binom_dist = tdist.Binomial(total_count=self.encoding_size, probs=self.target_p)  # 修改为 tdist

    def kl_laplace(self, mu1, b1, mu2, b2):
        d = torch.abs(mu1 - mu2)
        return torch.log(b2 / b1) - 1 + (d + b1 * torch.exp(-d / b1)) / b2

    def forward(self, encodings, current_epoch=None, total_epochs=None):
        """
        encodings: [batch_size * n_patches, encoding_size]
        """
        N, D = encodings.shape

        # 计算每个维度的mu和b（用于原KL）
        mu = torch.median(encodings, dim=0).values  # [D]
        b = torch.mean(torch.abs(encodings - mu.unsqueeze(0)), dim=0)  # [D]
        b = torch.clamp(b, min=1e-6)  # 避免除零

        # ===== 1. 原KL损失: D_KL(Lap(μ, b) || Lap(0, target_b)) =====
        kl_div_original = self.kl_laplace(mu, b, self.target_mean, self.target_b)
        kl_loss_original = kl_div_original.mean()

        # ===== 2. 基于STE的归一化KL损失（或Binom NLL，根据epoch切换） =====
        # 修改为0/1 binary (数学等价于原-1/1，但更便于count)
        binary_values = (encodings > 0).float()  # 0 or 1
        binary_approx = encodings - encodings.detach() + binary_values.detach()  # STE: forward=0/1, backward grad=1 (从encodings)

        # 计算近似正值个数 (forward为整数, backward流动)
        count_approx = binary_approx.sum(dim=1)  # [N], forward=pos_count (0~64), backward=sum(grad=1)

        # 始终使用Binom NLL作为kl_loss_count
        log_probs = self.binom_dist.log_prob(count_approx)
        kl_loss_count = -log_probs.mean()  # NLL，minimize to fit Binom PMF

        # 额外计算高斯KL（仅用于log显示，不用于总损失）
        approx_shifted = count_approx / (2 * D)  # 等价于 (prop - 0.5)，但用0/1后 count/D = prop, shifted=prop-0.5
        approx_normalized = torch.sqrt(
            torch.tensor(self.encoding_size, device=encodings.device)) * approx_shifted  # 归一化到std=1 (var=0.25目标)
        count_mean = approx_normalized.mean()
        count_var = approx_normalized.var(unbiased=False)

        # KL散度: D_KL(N(μ1, σ1²) || N(μ2, σ2²))
        kl_loss_gaussian = 0.5 * (
                count_var / self.target_count_var +
                (count_mean - self.target_count_mean) ** 2 / self.target_count_var -
                1 +
                torch.log(torch.tensor(self.target_count_var, device=encodings.device) / (count_var + 1e-8))
        )

        # ===== 3. MINE-based MI loss (加入总损失) =====
        mi_value = torch.tensor(0.0, device=encodings.device)
        if self.mine_model is not None:
            # 先对维度（列）打乱
            perm = torch.randperm(D, device=encodings.device)
            encodings_perm = binary_approx[:, perm]
            # 分割成左右32维
            left = encodings_perm[:, :32]  # [N, 32]
            right = encodings_perm[:, 32:]  # [N, 32]
            # 对右边的batch维度（行）打乱
            right_shuffle = right[torch.randperm(right.size(0))]
            pred_xy = self.mine_model(left, right)
            pred_x_y = self.mine_model(left, right_shuffle)
            mi_value = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))

        # ===== 4. 尾部惩罚：对极端count的偏差平方惩罚（只惩罚超出阈值部分） =====
        deviation = torch.abs(count_approx - self.target_binom_mean)
        tail_penalty = torch.mean(torch.clamp(deviation - self.tail_threshold, min=0.0) ** 2)

        # ===== 总损失（包括MI损失，条件化加入尾部惩罚，不包括cov_loss） =====
        total_loss = (self.weight_original * kl_loss_original +
                      self.weight_count * kl_loss_count +
                      self.weight_cov * mi_value)
        if current_epoch is not None and current_epoch >= 3:
            total_loss += self.weight_tail * tail_penalty

        # ===== 统计信息（添加tail_penalty，移除cov_loss，保留mi_value） =====
        pos_counts = (binary_values > 0).sum(dim=1).float()  # 真实正值个数 (基于0/1)

        stats_dict = {
            'total_loss': total_loss.item(),
            'kl_loss_original': kl_loss_original.item(),
            'kl_loss_count': kl_loss_count.item(),
            'kl_loss_gaussian': kl_loss_gaussian.item(),  # 新增：高斯KL，用于log显示
            'mi_value': mi_value.item(),  # MINE MI estimate
            'tail_penalty': tail_penalty.item(),  # 尾部惩罚值
            # 真实统计
            'pos_mean': pos_counts.mean().item(),
            'pos_std': pos_counts.std().item(),
            'pos_min': pos_counts.min().item(),
            'pos_max': pos_counts.max().item(),
            # 估计统计 (基于count_approx)
            'estimated_count_mean': count_approx.mean().item(),
            'estimated_count_std': count_approx.std().item(),
            # 分布统计
            'output_mean': mu.mean().item(),
            'output_b': b.mean().item(),
            'output_var': encodings.var(dim=0, unbiased=False).mean().item(),
            'output_std': torch.sqrt(encodings.var(dim=0, unbiased=False)).mean().item(),
            # 整体统计
            'p_positive': (encodings > 0).float().mean().item(),
            'avg_abs_value': torch.abs(encodings).mean().item(),
        }

        return total_loss, stats_dict


def analyze_distribution(encoder, data_loader, device, n_batches=10,
                         phase="before", save_data=None, mine_model=None):
    """
    分析编码器输出的分布特性（增强版，包含估计个数分析）
    """
    encoder.eval()

    all_positive_counts = []
    all_patch_outputs = []
    all_estimated_counts = []

    patch_size = 16
    encoding_size = 64
    tail_threshold = 28.0  # 与损失中一致

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            images = images.to(device)

            # 矢量化提取patches
            patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
            patches = patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)

            encodings = encoder(patches)

            # 真实正值个数
            positive_counts = (encodings > 0).sum(dim=1).cpu().numpy()
            all_positive_counts.extend(positive_counts)

            # 保存输出值 (二维，用于后续计算)
            all_patch_outputs.append(encodings.cpu().numpy())

            # 估计个数
            estimated_positive = positive_counts
            estimated_negative = encoding_size - positive_counts
            all_estimated_counts.extend(estimated_positive)
            all_estimated_counts.extend(estimated_negative)

    all_positive_counts = np.array(all_positive_counts)
    all_encodings_2d = np.concatenate(all_patch_outputs, axis=0)  # [total_patches, 64]，用于 cov/mi
    all_patch_outputs_np = all_encodings_2d.flatten()  # [total_patches * 64]，用于 hist/KS 等一维分布
    all_estimated_counts = np.array(all_estimated_counts)

    # 计算Laplace参数估计
    estimated_mu = np.median(all_patch_outputs_np)
    estimated_b = np.mean(np.abs(all_patch_outputs_np - estimated_mu))

    # ===== 计算 cov_loss（只在这里计算，使用二维） =====
    all_encodings_torch = torch.from_numpy(all_encodings_2d).to(device)  # [N, D]
    cov_matrix = torch.cov(all_encodings_torch.T)  # [encoding_size, encoding_size]
    off_diag = cov_matrix - torch.diag_embed(cov_matrix.diagonal())
    cov_loss = torch.mean(off_diag ** 2).item()

    # ===== 计算整体 mi_value（作为观察，使用二维） =====
    mi_value = 0.0
    if mine_model is not None:
        N, D = all_encodings_torch.shape
        binary_values = (all_encodings_torch > 0).float()  # 无STE，因为是观察
        perm = torch.randperm(D, device=device)
        encodings_perm = binary_values[:, perm]
        left = encodings_perm[:, :32]
        right = encodings_perm[:, 32:]
        right_shuffle = right[torch.randperm(right.size(0))]
        pred_xy = mine_model(left, right)
        pred_x_y = mine_model(left, right_shuffle)
        mi_value = (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))).item()

    # ===== 计算尾部比例（P(|count - 32| > threshold)） =====
    tail_ratio = np.mean(np.abs(all_positive_counts - 32) > tail_threshold)

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
    print(f"  Tail ratio (|count-32| > {tail_threshold}): {tail_ratio:.4f}")

    print(f"\nEstimated Count Statistics:")
    print(f"  Mean: {all_estimated_counts.mean():.4f} (target: 32.0)")
    print(f"  Std:  {all_estimated_counts.std():.4f} (target: 4.0)")

    p_positive = (all_patch_outputs_np > 0).mean()
    print(f"\nOutput Value Statistics:")
    print(f"  P(X>0): {p_positive:.6f} (target: 0.5)")
    print(f"  Mean:   {all_patch_outputs_np.mean():.6f} (target: 0.0)")
    print(f"  Median: {estimated_mu:.6f} (target: 0.0)")
    print(f"  Scale b: {estimated_b:.6f} (target: 1.0)")
    print(f"  Std:    {all_patch_outputs_np.std():.6f} (target: {np.sqrt(2) * 1.0:.6f})")
    print(f"  |x|_avg: {np.abs(all_patch_outputs_np).mean():.6f} (target: 1.0)")

    print(f"\nIndependence Metrics:")
    print(f"  Cov loss: {cov_loss:.6f}")
    print(f"  MI value: {mi_value:.6f}")

    if save_data is not None:
        save_data[phase] = {
            'positive_counts': all_positive_counts,
            'patch_outputs': all_patch_outputs_np,  # 一维，用于 plot
            'estimated_counts': all_estimated_counts,
            'encoding_size': encoding_size
        }


def save_plot_data(before_data, after_data, save_path):
    """保存绘图数据到文件"""
    plot_data = {
        'before': before_data,
        'after': after_data,
        'metadata': {
            'encoding_size': before_data['encoding_size'],
        }
    }
    with open(save_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to: {save_path}")


def plot_comparison(before_data, after_data, save_path, epoch_info=""):
    """
    对比训练前后的分布（增强版，包含估计个数分布）
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
        encoding_size = data['encoding_size']
        p_positive = (patch_outputs > 0).mean()
        estimated_b = np.mean(np.abs(patch_outputs - np.median(patch_outputs)))

        # 第一行: 真实正值个数分布
        ax1 = fig.add_subplot(gs[0, idx * 2:idx * 2 + 2])
        ax1.hist(positive_counts, bins=np.arange(0, encoding_size + 2) - 0.5, density=True, alpha=0.7, color=color,
                 edgecolor='black', label=f'Observed ({label})')
        x_binom = np.arange(0, encoding_size + 1)
        ax1.plot(x_binom, stats.binom.pmf(x_binom, encoding_size, 0.5), 'go-', linewidth=2, markersize=4,
                 label=f'Target Binomial(64, 0.5)', alpha=0.9)
        ax1.plot(x_binom, stats.binom.pmf(x_binom, encoding_size, p_positive), 'ro--', linewidth=2, markersize=3,
                 label=f'Actual Binomial(64, {p_positive:.3f})', alpha=0.8)
        ax1.axvline(32, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax1.axvline(positive_counts.mean(), color='red', linestyle='-', linewidth=2,
                    label=f'Mean: {positive_counts.mean():.2f}')
        ax1.set_xlabel('True Number of Positive Values per Patch', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'True Positive Count Distribution - {label}', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # 第二行: 输出值分布
        ax3 = fig.add_subplot(gs[1, idx * 2:idx * 2 + 2])
        ax3.hist(patch_outputs, bins='auto', density=True, alpha=0.7, color=color, edgecolor='blue',
                 label=f'Raw Outputs ({label})')
        x_range = np.linspace(patch_outputs.min(), patch_outputs.max(), 500)
        ax3.plot(x_range, stats.laplace.pdf(x_range, loc=0, scale=1.0), 'g-', linewidth=2.5,
                 label='Target Laplace(0,1.0)', alpha=0.9)
        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('Output Value', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title(f'Output Distribution - {label}', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        ks_stat, ks_p = stats.kstest(patch_outputs, 'laplace', args=(0, 1.0))
        textstr = (f'K-S Test:\nstat={ks_stat:.4f}\np={ks_p:.2e}\n'
                   f'Mean={patch_outputs.mean():.4f}\n'
                   f'Scale={estimated_b:.4f}\n'
                   f'Std={np.std(patch_outputs):.4f}')
        ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 第三行: Q-Q图
        ax4 = fig.add_subplot(gs[2, idx * 2:idx * 2 + 2])
        sample_size = min(5000, len(patch_outputs))
        sample_indices = np.random.choice(len(patch_outputs), sample_size, replace=False)
        sample_data = np.sort(patch_outputs[sample_indices])
        stats.probplot(sample_data, dist=stats.laplace(loc=0, scale=1.0), plot=ax4)
        ax4.set_title(f'Q-Q Plot - {label}', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)
        theoretical_quantiles = np.array([x[0] for x in ax4.lines[0].get_data()])
        sample_quantiles = np.array([x[1] for x in ax4.lines[0].get_data()])
        r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
        ax4.text(0.05, 0.95, f'R² = {r_squared:.6f}', transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison figure saved to: {save_path}")
    plt.close()


def train_one_epoch(encoder, criterion, optimizer, data_loader, device,
                    epoch, logger, scaler=None, total_epochs=None):
    """
    训练一个epoch，并返回整个epoch的平均损失和统计信息
    """
    encoder.train()

    patch_size = 16
    log_interval = 100
    num_batches = len(data_loader)

    # 用于日志记录的运行统计
    running_loss = 0.0
    running_stats = {}

    # 用于整个epoch的累积统计
    epoch_total_loss = 0.0
    epoch_total_stats = {}

    start_time = time.time()  # 整体epoch时间（可选保留）
    total_samples = 0

    # Interval级别计时，用于更准确的samples/s
    interval_start_time = time.time()
    interval_samples = 0

    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to(device)
        batch_size = images.shape[0]
        total_samples += batch_size
        interval_samples += batch_size

        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        patches = patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            encodings = encoder(patches)
            loss, stats_dict = criterion(encodings, current_epoch=epoch, total_epochs=total_epochs)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 累积统计
        loss_item = loss.item()
        running_loss += loss_item
        epoch_total_loss += loss_item
        for key, value in stats_dict.items():
            running_stats[key] = running_stats.get(key, 0.0) + value
            epoch_total_stats[key] = epoch_total_stats.get(key, 0.0) + value

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_stats = {k: v / log_interval for k, v in running_stats.items()}
            interval_elapsed = time.time() - interval_start_time
            samples_per_sec = interval_samples / interval_elapsed if interval_elapsed > 0 else 0

            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {avg_loss:.6f} | "
                f"KL_orig: {avg_stats['kl_loss_original']:.6f} | "
                f"KL_count: {avg_stats['kl_loss_count']:.6f} | "
                f"MI_value: {avg_stats['mi_value']:.6f} | "
                f"Tail_penalty: {avg_stats['tail_penalty']:.6f} | "
                f"Mean: {avg_stats['output_mean']:.4f} | "
                f"B: {avg_stats['output_b']:.4f} | "
                f"Std: {avg_stats['output_std']:.4f} | "
                f"+True: {avg_stats['pos_mean']:.2f} | "
                f"P_pos: {avg_stats['p_positive']:.4f} | "
                f"Samples/s: {samples_per_sec:.1f}"
            )
            # 重置interval
            interval_start_time = time.time()
            interval_samples = 0
            running_loss = 0.0
            running_stats = {}

    # 计算整个epoch的平均值
    avg_epoch_loss = epoch_total_loss / num_batches
    avg_epoch_stats = {k: v / num_batches for k, v in epoch_total_stats.items()}

    logger.info(f"\n--- Epoch [{epoch}] Summary ---")
    logger.info(f"    Average Loss: {avg_epoch_loss:.6f}")
    for key, val in avg_epoch_stats.items():
        logger.info(f"    Avg {key.replace('_', ' ').title()}: {val:.4f}")
    logger.info("---------------------------\n")

    return avg_epoch_loss, avg_epoch_stats


def save_checkpoint(model_state, optimizer_state, epoch, args_state, path):
    """
    保存模型检查点
    """
    torch.save({
        'model': model_state,
        'optimizer': optimizer_state,
        'epoch': epoch,
        'args': args_state,
    }, path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 20)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2


def main(args):
    """主训练函数"""
    assert torch.cuda.is_available(), "Training requires GPU"

    init_distributed_mode(args)

    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        args.gpu = 0

    device = args.gpu
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        experiment_dir = get_experiment_dir(args.results_dir, "encoder_training")
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Running in {'distributed' if args.distributed else 'single-GPU'} mode")
    else:
        experiment_dir = None
        logger = create_logger(None)

    logger.info(f"Args: {args}")

    encoder = PatchMLPEncoder(hidden_dim=args.hidden_dim, learnable=True).to(device)

    # Load pre-trained MINE model for observation
    mine_model = Net().to(device)
    mine_model.load_state_dict(torch.load('./mine_weights.pth'))
    mine_model.eval()
    for param in mine_model.parameters():
        param.requires_grad = False

    criterion = ImprovedLaplaceKLLoss(
        weight_original=args.weight_original,
        weight_count=args.weight_count,
        weight_cov=args.weight_cov,
        weight_tail=args.weight_tail,
        distributed=args.distributed,
        mine_model=mine_model
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.global_seed) if args.distributed else None
    loader = DataLoader(dataset, batch_size=int(args.global_batch_size // world_size), shuffle=(sampler is None),
                        sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Dataset contains {len(dataset):,} images")

    encoder_module = encoder
    if args.distributed:
        encoder = DDP(encoder, device_ids=[device])
        encoder_module = encoder.module

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    best_loss = float('inf')

    if rank == 0:
        logger.info("\n" + "=" * 80 + "\nBEFORE TRAINING - Distribution Analysis\n" + "=" * 80)
        before_data = {}
        analyze_distribution(encoder_module, loader, device, n_batches=10, phase="before", save_data=before_data, mine_model=mine_model)

    if args.distributed:
        dist.barrier()

    logger.info("\n" + "=" * 80 + "\nSTART TRAINING\n" + "=" * 80)
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            sampler.set_epoch(epoch)

        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        avg_epoch_loss, _ = train_one_epoch(encoder_module, criterion, optimizer, loader, device, epoch, logger, scaler,
                                            total_epochs=args.epochs)

        scheduler.step()

        if rank == 0:
            logger.info(f"\n{'=' * 80}\nEpoch {epoch} - Distribution Analysis\n{'=' * 80}")
            current_epoch_data = {}
            analyze_distribution(encoder_module, loader, device, n_batches=10, phase=f"epoch_{epoch}",
                                 save_data=current_epoch_data, mine_model=mine_model)

            epoch_plot_path = os.path.join(experiment_dir, f"distribution_epoch_{epoch}.png")
            plot_comparison(before_data['before'], current_epoch_data[f"epoch_{epoch}"], epoch_plot_path,
                            epoch_info=f"After Epoch {epoch}")
            logger.info(f"Epoch {epoch} analysis completed and plot saved.")

            # ===== 保存模型 =====
            # 1. 保存 'last' 模型
            last_checkpoint_path = os.path.join(experiment_dir, "encoder_last.pt")
            save_checkpoint(encoder_module.state_dict(), optimizer.state_dict(), epoch, args, last_checkpoint_path)
            logger.info(f"Saved last model checkpoint to: {last_checkpoint_path}")

            # 2. 检查并保存 'best' 模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_checkpoint_path = os.path.join(experiment_dir, "encoder_best.pt")
                save_checkpoint(encoder_module.state_dict(), optimizer.state_dict(), epoch, args, best_checkpoint_path)
                logger.info(f"*** New best model saved with loss {best_loss:.6f} to: {best_checkpoint_path} ***")

        if args.distributed:
            dist.barrier()

    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info(f" 'last' model and plots from all epochs saved.")
        logger.info(f"The 'best' model was saved at the epoch with the lowest training loss: {best_loss:.6f}")
        logger.info(f"Results saved in: {experiment_dir}")
        logger.info("=" * 80)

    if args.distributed:
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
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    # Model args
    parser.add_argument("--hidden-dim", type=int, default=12, help="Hidden dimension for MLP encoder")
    # Loss weights
    parser.add_argument("--weight-original", type=float, default=1.0, help="Weight for original KL loss")
    parser.add_argument("--weight-count", type=float, default=1.0,
                        help="Weight for normalized count KL divergence loss")
    parser.add_argument("--weight-cov", type=float, default=1.0, help="Weight for MI loss (via MINE)")
    parser.add_argument("--weight-tail", type=float, default=1.0, help="Weight for tail penalty")
    # Save args
    parser.add_argument("--results-dir", type=str, default="results_encoder")
    args = parser.parse_args()
    main(args)
