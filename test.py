import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class PatchConvEncoder(nn.Module):
    """使用2x2不重叠卷积处理patch"""

    def __init__(self, conv_kernel=2, conv_weight_value=1 / 64):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, stride=conv_kernel, bias=False)

        # 设置卷积核所有权重为1/64
        with torch.no_grad():
            self.conv.weight.fill_(conv_weight_value)

        # 冻结权重
        self.conv.weight.requires_grad = False

    def forward(self, x):
        """
        输入: [batch_size, 1, patch_size, patch_size]  # e.g., [B, 1, 16, 16]
        输出: [batch_size, encoding_size]  # e.g., [B, 64]
        """
        x = self.conv(x)  # [batch_size, 1, 8, 8]
        x = x.flatten(1)  # [batch_size, 64]
        return x


def test_patch_conv_distribution(n_trials=10, batch_size=64, image_size=256, patch_size=16):
    """
    测试图像块卷积处理后的分布

    Args:
        n_trials: 重复次数
        batch_size: 批次大小
        image_size: 图像大小 (256x256)
        patch_size: 块大小 (16x16)
    """
    print("=" * 80)
    print(f"Testing Patch Convolution Encoder")
    print(f"Image size: {image_size}x{image_size}, Patch size: {patch_size}x{patch_size}")
    print(f"Convolution: 2x2 kernel with stride 2, weight = 1/64")
    print(f"Output encoding per patch: {(patch_size // 2) ** 2} values")
    print(f"Number of patches per image: {(image_size // patch_size) ** 2}")
    print(f"Batch size: {batch_size}, Trials: {n_trials}")
    print("=" * 80)

    # 计算相关参数
    n_patches_per_side = image_size // patch_size  # 16
    n_patches_per_image = n_patches_per_side ** 2  # 256
    encoding_size = (patch_size // 2) ** 2  # 8*8 = 64

    # 创建卷积编码器
    encoder = PatchConvEncoder(conv_kernel=2, conv_weight_value=1 / 64)
    encoder.eval()

    # 存储所有patch的正值个数统计
    all_positive_counts = []  # 每个patch中值>0的个数 (0-64)
    all_patch_outputs = []  # 所有patch的输出值

    for trial in range(n_trials):
        print(f"\n{'=' * 80}")
        print(f"Trial {trial + 1}/{n_trials}")
        print(f"{'=' * 80}")

        # 设置随机种子
        torch.manual_seed(trial * 100)

        # 生成高斯分布的图像: [batch_size, 1, 256, 256]
        images = torch.randn(batch_size, 1, image_size, image_size)

        print(f"Generated images shape: {images.shape}")

        # 处理每个batch中的每张图像
        for batch_idx in range(batch_size):
            image = images[batch_idx:batch_idx + 1]  # [1, 1, 256, 256]

            # 将图像划分为patches
            # Unfold: [1, 1, 256, 256] -> [1, 1*patch_size*patch_size, n_patches]
            patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
            # 重塑: [1, 256, 256] -> [256, 1, 16, 16]
            patches = patches.squeeze(0).transpose(0, 1).reshape(n_patches_per_image, 1, patch_size, patch_size)

            # 通过卷积编码器处理所有patches
            with torch.no_grad():
                patch_encodings = encoder(patches)  # [256, 64]

            # 统计每个patch中值>0的个数
            positive_counts = (patch_encodings > 0).sum(dim=1).cpu().numpy()  # [256]
            all_positive_counts.extend(positive_counts)

            # 保存所有输出值用于分布分析
            all_patch_outputs.append(patch_encodings.cpu().numpy())

        print(f"Processed {batch_size} images, {batch_size * n_patches_per_image} patches")
        print(f"Collected {len(all_positive_counts)} positive count samples so far")

    # 转换为numpy数组
    all_positive_counts = np.array(all_positive_counts)
    all_patch_outputs = np.concatenate(all_patch_outputs, axis=0).flatten()

    print("\n" + "=" * 80)
    print(f"Total patches analyzed: {len(all_positive_counts):,}")
    print(f"Total output values: {len(all_patch_outputs):,}")
    print("=" * 80)

    # 统计分析 - 正值个数分布
    print("\nPositive Count Statistics (per patch, 0-64):")
    print("-" * 80)
    print(f"Mean:               {all_positive_counts.mean():.4f}")
    print(f"Std:                {all_positive_counts.std():.4f}")
    print(f"Min:                {all_positive_counts.min()}")
    print(f"Max:                {all_positive_counts.max()}")
    print(f"Median:             {np.median(all_positive_counts):.4f}")

    # 理论期望：如果是标准正态分布，P(X>0)=0.5，期望个数=64*0.5=32
    theoretical_mean = encoding_size * 0.5
    print(f"Theoretical mean:   {theoretical_mean:.4f} (assuming P(X>0)=0.5)")
    print(f"Deviation:          {all_positive_counts.mean() - theoretical_mean:.4f}")

    # 统计分析 - 输出值分布
    print("\nOutput Value Statistics:")
    print("-" * 80)
    print(f"Mean:               {all_patch_outputs.mean():.6f}")
    print(f"Std:                {all_patch_outputs.std():.6f}")
    print(f"Min:                {all_patch_outputs.min():.6f}")
    print(f"Max:                {all_patch_outputs.max():.6f}")
    print(f"P(output > 0):      {(all_patch_outputs > 0).mean():.6f}")

    # 标准化输出值以对齐标准正态分布
    outputs_standardized = (all_patch_outputs - all_patch_outputs.mean()) / all_patch_outputs.std()

    print("\nStandardized Output Statistics:")
    print("-" * 80)
    print(f"Mean:               {outputs_standardized.mean():.6f}")
    print(f"Std:                {outputs_standardized.std():.6f}")
    print(f"Min:                {outputs_standardized.min():.6f}")
    print(f"Max:                {outputs_standardized.max():.6f}")

    # 正态性检验
    sample_size = min(5000, len(all_patch_outputs))
    sample_indices = np.random.choice(len(all_patch_outputs), sample_size, replace=False)
    sample_data = all_patch_outputs[sample_indices]

    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
    print(f"\nShapiro-Wilk Test (on {sample_size} samples):")
    print(f"  statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")

    # 二项分布拟合（正值个数应该服从二项分布）
    print("\nBinomial Distribution Test:")
    print("-" * 80)
    p_positive = (all_patch_outputs > 0).mean()
    print(f"Estimated p (P(X>0)): {p_positive:.6f}")
    print(f"Theoretical binomial: Binomial(n={encoding_size}, p={p_positive:.4f})")
    print(f"Expected mean:        {encoding_size * p_positive:.4f}")
    print(f"Expected std:         {np.sqrt(encoding_size * p_positive * (1 - p_positive)):.4f}")

    # 可视化
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 正值个数分布直方图 vs 二项分布
    ax1 = fig.add_subplot(gs[0, :2])
    counts, bins, patches_plot = ax1.hist(all_positive_counts, bins=np.arange(0, encoding_size + 2) - 0.5,
                                          density=True, alpha=0.7, color='steelblue',
                                          edgecolor='black', label='Observed')

    # 理论二项分布
    x_binom = np.arange(0, encoding_size + 1)
    binom_pmf = stats.binom.pmf(x_binom, encoding_size, p_positive)
    ax1.plot(x_binom, binom_pmf, 'ro-', linewidth=2, markersize=4,
             label=f'Binomial(n={encoding_size}, p={p_positive:.3f})', alpha=0.8)

    ax1.axvline(theoretical_mean, color='green', linestyle='--', linewidth=2,
                label=f'Theoretical (p=0.5): {theoretical_mean:.1f}')
    ax1.axvline(all_positive_counts.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Observed mean: {all_positive_counts.mean():.2f}')

    ax1.set_xlabel('Number of Positive Values per Patch (0-64)', fontsize=13)
    ax1.set_ylabel('Probability Density', fontsize=13)
    ax1.set_title('Distribution of Positive Counts vs Binomial Distribution',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-1, encoding_size + 1)

    # 2. 累积分布对比
    ax2 = fig.add_subplot(gs[0, 2])
    sorted_counts = np.sort(all_positive_counts)
    empirical_cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    ax2.plot(sorted_counts, empirical_cdf, 'b-', linewidth=2,
             label='Empirical CDF', alpha=0.7)

    # 理论二项分布CDF
    theoretical_cdf = stats.binom.cdf(sorted_counts, encoding_size, p_positive)
    ax2.plot(sorted_counts, theoretical_cdf, 'r--', linewidth=2,
             label='Binomial CDF', alpha=0.7)

    ax2.set_xlabel('Positive Count', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('CDF Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # 3. 标准化输出值分布 vs 标准高斯分布（对齐版本）
    ax3 = fig.add_subplot(gs[1, :2])

    # 使用标准化后的输出值绘制直方图
    counts_hist, bins_hist, _ = ax3.hist(outputs_standardized, bins=150, density=True, alpha=0.7,
                                         color='steelblue', edgecolor='black', label='Standardized Output Values')

    # 标准高斯分布（用于对齐比较）
    x_range = np.linspace(outputs_standardized.min(), outputs_standardized.max(), 500)
    standard_gaussian = stats.norm.pdf(x_range, 0, 1)
    ax3.plot(x_range, standard_gaussian, 'r-', linewidth=2.5,
             label='Standard Gaussian (μ=0, σ=1)', alpha=0.9)

    # 原始数据的拟合高斯（标准化后应该是μ=0, σ=1）
    mu_std, sigma_std = outputs_standardized.mean(), outputs_standardized.std()
    fitted_gaussian = stats.norm.pdf(x_range, mu_std, sigma_std)
    ax3.plot(x_range, fitted_gaussian, 'g--', linewidth=2,
             label=f'Fitted Gaussian (μ={mu_std:.4f}, σ={sigma_std:.4f})', alpha=0.8)

    ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Standardized Value', fontsize=13)
    ax3.set_ylabel('Density', fontsize=13)
    ax3.set_title('Standardized Output Distribution vs Standard Gaussian',
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(-5, 5)  # 聚焦在±5σ范围内

    # 添加文本框显示KL散度或其他距离度量
    # 计算Kolmogorov-Smirnov统计量
    ks_stat, ks_p = stats.kstest(outputs_standardized, 'norm', args=(0, 1))
    textstr = f'K-S Test:\nstatistic={ks_stat:.4f}\np-value={ks_p:.4e}'
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 4. Q-Q图（标准化输出值）
    ax4 = fig.add_subplot(gs[1, 2])
    sample_standardized = outputs_standardized[sample_indices]
    stats.probplot(sample_standardized, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot\n(Standardized Values)', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3)

    # 添加R²值
    theoretical_quantiles = np.array([x[0] for x in ax4.lines[0].get_data()])
    sample_quantiles = np.array([x[1] for x in ax4.lines[0].get_data()])
    r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
    ax4.text(0.05, 0.95, f'R² = {r_squared:.6f}', transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 5. 每个trial的平均正值个数
    ax5 = fig.add_subplot(gs[2, 0])
    samples_per_trial = batch_size * n_patches_per_image
    mean_positive_per_trial = []

    for i in range(n_trials):
        start_idx = i * samples_per_trial
        end_idx = start_idx + samples_per_trial
        trial_mean = all_positive_counts[start_idx:end_idx].mean()
        mean_positive_per_trial.append(trial_mean)

    mean_positive_per_trial = np.array(mean_positive_per_trial)

    ax5.plot(range(1, n_trials + 1), mean_positive_per_trial, 'o-',
             markersize=8, linewidth=2, color='steelblue')
    ax5.axhline(theoretical_mean, color='green', linestyle='--',
                linewidth=2, label=f'Theoretical: {theoretical_mean:.1f}')
    ax5.axhline(mean_positive_per_trial.mean(), color='red', linestyle='-',
                linewidth=2, label=f'Mean: {mean_positive_per_trial.mean():.2f}')

    ax5.set_xlabel('Trial Number', fontsize=12)
    ax5.set_ylabel('Mean Positive Count', fontsize=12)
    ax5.set_title('Mean Positive Count per Trial', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.set_xticks(range(1, min(n_trials + 1, 21)))  # 最多显示20个刻度

    # 6. 标准化前后的分布对比
    ax6 = fig.add_subplot(gs[2, 1])

    # 原始输出值的直方图（归一化显示）
    mu_orig, sigma_orig = all_patch_outputs.mean(), all_patch_outputs.std()
    x_range_orig = np.linspace(all_patch_outputs.min(), all_patch_outputs.max(), 200)

    ax6.hist(all_patch_outputs, bins=100, density=True, alpha=0.5,
             color='coral', edgecolor='black', label='Original Output')
    ax6.plot(x_range_orig, stats.norm.pdf(x_range_orig, mu_orig, sigma_orig),
             'r-', linewidth=2, label=f'Original Fit\nμ={mu_orig:.4f}, σ={sigma_orig:.4f}')

    ax6.set_xlabel('Output Value', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12)
    ax6.set_title('Original Output Distribution', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    # 7. 残差分析（标准化值与理论分位数的差异）
    ax7 = fig.add_subplot(gs[2, 2])

    # 对标准化数据排序并计算理论分位数
    sorted_std = np.sort(outputs_standardized)
    n_points = len(sorted_std)
    theoretical_quantiles_full = stats.norm.ppf(np.linspace(0.001, 0.999, n_points))

    # 计算残差（采样以提高性能）
    sample_step = max(1, n_points // 10000)
    residuals = sorted_std[::sample_step] - theoretical_quantiles_full[::sample_step]

    ax7.scatter(theoretical_quantiles_full[::sample_step], residuals,
                alpha=0.3, s=1, color='steelblue')
    ax7.axhline(0, color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax7.set_ylabel('Residuals', fontsize=12)
    ax7.set_title('Q-Q Plot Residuals', fontsize=13, fontweight='bold')
    ax7.grid(alpha=0.3)

    # 添加残差统计
    residual_std = np.std(residuals)
    ax7.text(0.05, 0.95, f'Residual Std: {residual_std:.4f}',
             transform=ax7.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig('patch_conv_distribution_analysis.png', dpi=300, bbox_inches='tight')

    print("\n" + "=" * 80)
    print("Figure saved as 'patch_conv_distribution_analysis.png'")
    print("=" * 80)

    # 额外的统计测试
    print("\nAdditional Statistical Tests:")
    print("-" * 80)
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  statistic={ks_stat:.6f}, p-value={ks_p:.6e}")
    print(f"Q-Q Plot R²: {r_squared:.6f}")
    print(f"Residual Std: {residual_std:.6f}")

    return all_positive_counts, all_patch_outputs, outputs_standardized


# 运行测试
if __name__ == "__main__":
    positive_counts, patch_outputs, standardized_outputs = test_patch_conv_distribution(
        n_trials=1000,
        batch_size=128,
        image_size=256,
        patch_size=16
    )

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
#
#
# class PatchConvEncoder(nn.Module):
#     """使用2x2不重叠卷积处理patch - 随机高斯初始化"""
#
#     def __init__(self, conv_kernel=2, init_std=1.0):
#         super().__init__()
#         self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, stride=conv_kernel, bias=False)
#
#         # 随机高斯初始化：N(0, init_std²)
#         with torch.no_grad():
#             nn.init.normal_(self.conv.weight, mean=0.0, std=init_std)
#
#         # 冻结权重
#         self.conv.weight.requires_grad = False
#
#         # 打印权重统计信息
#         weight_np = self.conv.weight.data.numpy().flatten()
#         print(f"\nConvolution Weight Statistics:")
#         print(f"  Shape: {self.conv.weight.shape}")
#         print(f"  Mean: {weight_np.mean():.6f}")
#         print(f"  Std:  {weight_np.std():.6f}")
#         print(f"  Min:  {weight_np.min():.6f}")
#         print(f"  Max:  {weight_np.max():.6f}")
#         print(f"  Weights: {weight_np}")
#
#     def forward(self, x):
#         """
#         输入: [batch_size, 1, patch_size, patch_size]  # e.g., [B, 1, 16, 16]
#         输出: [batch_size, encoding_size]  # e.g., [B, 64]
#         """
#         x = self.conv(x)  # [batch_size, 1, 8, 8]
#         x = x.flatten(1)  # [batch_size, 64]
#         return x
#
#
# def test_patch_conv_distribution(n_trials=10, batch_size=64, image_size=256,
#                                  patch_size=16, conv_init_std=1.0, seed=42):
#     """
#     测试图像块卷积处理后的分布（随机高斯初始化权重）
#
#     Args:
#         n_trials: 重复次数
#         batch_size: 批次大小
#         image_size: 图像大小 (256x256)
#         patch_size: 块大小 (16x16)
#         conv_init_std: 卷积权重初始化的标准差
#         seed: 随机种子
#     """
#     print("=" * 80)
#     print(f"Testing Patch Convolution Encoder with Random Gaussian Weights")
#     print(f"Image size: {image_size}x{image_size}, Patch size: {patch_size}x{patch_size}")
#     print(f"Convolution: 2x2 kernel with stride 2")
#     print(f"Weight initialization: N(0, {conv_init_std}²)")
#     print(f"Output encoding per patch: {(patch_size // 2) ** 2} values")
#     print(f"Number of patches per image: {(image_size // patch_size) ** 2}")
#     print(f"Batch size: {batch_size}, Trials: {n_trials}")
#     print("=" * 80)
#
#     # 计算相关参数
#     n_patches_per_side = image_size // patch_size  # 16
#     n_patches_per_image = n_patches_per_side ** 2  # 256
#     encoding_size = (patch_size // 2) ** 2  # 8*8 = 64
#
#     # 设置随机种子并创建卷积编码器
#     torch.manual_seed(seed)
#     encoder = PatchConvEncoder(conv_kernel=2, init_std=conv_init_std)
#     encoder.eval()
#
#     # 理论方差计算
#     weight_var = conv_init_std ** 2
#     # 输出方差 = 4个输入值 × 权重方差（假设输入方差=1）
#     theoretical_output_var = 4 * weight_var * 1.0  # 输入N(0,1)
#     theoretical_output_std = np.sqrt(theoretical_output_var)
#
#     print(f"\nTheoretical Analysis:")
#     print(f"  Input: N(0, 1)")
#     print(f"  Weight: N(0, {conv_init_std}²)")
#     print(f"  Expected output variance: {theoretical_output_var:.6f}")
#     print(f"  Expected output std: {theoretical_output_std:.6f}")
#
#     # 存储所有patch的正值个数统计
#     all_positive_counts = []  # 每个patch中值>0的个数 (0-64)
#     all_patch_outputs = []  # 所有patch的输出值
#
#     for trial in range(n_trials):
#         print(f"\n{'=' * 80}")
#         print(f"Trial {trial + 1}/{n_trials}")
#         print(f"{'=' * 80}")
#
#         # 设置随机种子（不同于权重初始化）
#         torch.manual_seed(seed + trial * 100)
#
#         # 生成高斯分布的图像: [batch_size, 1, 256, 256]
#         images = torch.randn(batch_size, 1, image_size, image_size)
#
#         print(f"Generated images shape: {images.shape}")
#
#         # 处理每个batch中的每张图像
#         for batch_idx in range(batch_size):
#             image = images[batch_idx:batch_idx + 1]  # [1, 1, 256, 256]
#
#             # 将图像划分为patches
#             patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
#             patches = patches.squeeze(0).transpose(0, 1).reshape(n_patches_per_image, 1, patch_size, patch_size)
#
#             # 通过卷积编码器处理所有patches
#             with torch.no_grad():
#                 patch_encodings = encoder(patches)  # [256, 64]
#
#             # 统计每个patch中值>0的个数
#             positive_counts = (patch_encodings > 0).sum(dim=1).cpu().numpy()  # [256]
#             all_positive_counts.extend(positive_counts)
#
#             # 保存所有输出值用于分布分析
#             all_patch_outputs.append(patch_encodings.cpu().numpy())
#
#         print(f"Processed {batch_size} images, {batch_size * n_patches_per_image} patches")
#         print(f"Collected {len(all_positive_counts)} positive count samples so far")
#
#     # 转换为numpy数组
#     all_positive_counts = np.array(all_positive_counts)
#     all_patch_outputs = np.concatenate(all_patch_outputs, axis=0).flatten()
#
#     print("\n" + "=" * 80)
#     print(f"Total patches analyzed: {len(all_positive_counts):,}")
#     print(f"Total output values: {len(all_patch_outputs):,}")
#     print("=" * 80)
#
#     # 统计分析 - 正值个数分布
#     print("\nPositive Count Statistics (per patch, 0-64):")
#     print("-" * 80)
#     print(f"Mean:               {all_positive_counts.mean():.4f}")
#     print(f"Std:                {all_positive_counts.std():.4f}")
#     print(f"Min:                {all_positive_counts.min()}")
#     print(f"Max:                {all_positive_counts.max()}")
#     print(f"Median:             {np.median(all_positive_counts):.4f}")
#
#     # 理论期望
#     p_positive = (all_patch_outputs > 0).mean()
#     theoretical_mean = encoding_size * 0.5  # 假设对称分布
#     print(f"Theoretical mean (p=0.5):   {theoretical_mean:.4f}")
#     print(f"Observed P(>0):             {p_positive:.4f}")
#     print(f"Observed mean:              {all_positive_counts.mean():.4f}")
#     print(f"Deviation from theoretical: {all_positive_counts.mean() - theoretical_mean:.4f}")
#
#     # 统计分析 - 输出值分布
#     print("\nOutput Value Statistics:")
#     print("-" * 80)
#     print(f"Mean:               {all_patch_outputs.mean():.6f}")
#     print(f"Std (observed):     {all_patch_outputs.std():.6f}")
#     print(f"Std (theoretical):  {theoretical_output_std:.6f}")
#     print(f"Std ratio (obs/theo): {all_patch_outputs.std() / theoretical_output_std:.4f}")
#     print(f"Min:                {all_patch_outputs.min():.6f}")
#     print(f"Max:                {all_patch_outputs.max():.6f}")
#     print(f"P(output > 0):      {p_positive:.6f}")
#
#     # 正态性检验
#     sample_size = min(5000, len(all_patch_outputs))
#     sample_indices = np.random.choice(len(all_patch_outputs), sample_size, replace=False)
#     sample_data = all_patch_outputs[sample_indices]
#
#     shapiro_stat, shapiro_p = stats.shapiro(sample_data)
#     print(f"\nShapiro-Wilk Test (on {sample_size} samples):")
#     print(f"  statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
#     if shapiro_p > 0.05:
#         print("  -> Distribution appears normal (p > 0.05)")
#     else:
#         print("  -> Distribution may not be strictly normal (p < 0.05)")
#
#     # 二项分布拟合
#     print("\nBinomial Distribution Test:")
#     print("-" * 80)
#     print(f"Estimated p (P(X>0)): {p_positive:.6f}")
#     print(f"Theoretical binomial: Binomial(n={encoding_size}, p={p_positive:.4f})")
#     print(f"Expected mean:        {encoding_size * p_positive:.4f}")
#     print(f"Expected std:         {np.sqrt(encoding_size * p_positive * (1 - p_positive)):.4f}")
#
#     # 可视化
#     fig = plt.figure(figsize=(18, 12))
#     gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
#
#     # 1. 正值个数分布直方图 vs 二项分布
#     ax1 = fig.add_subplot(gs[0, :2])
#     counts, bins, patches_plot = ax1.hist(all_positive_counts, bins=np.arange(0, encoding_size + 2) - 0.5,
#                                           density=True, alpha=0.7, color='steelblue',
#                                           edgecolor='black', label='Observed')
#
#     # 理论二项分布
#     x_binom = np.arange(0, encoding_size + 1)
#     binom_pmf = stats.binom.pmf(x_binom, encoding_size, p_positive)
#     ax1.plot(x_binom, binom_pmf, 'ro-', linewidth=2, markersize=4,
#              label=f'Binomial(n={encoding_size}, p={p_positive:.3f})', alpha=0.8)
#
#     ax1.axvline(theoretical_mean, color='green', linestyle='--', linewidth=2,
#                 label=f'Theoretical (p=0.5): {theoretical_mean:.1f}')
#     ax1.axvline(all_positive_counts.mean(), color='red', linestyle='-', linewidth=2,
#                 label=f'Observed mean: {all_positive_counts.mean():.2f}')
#
#     ax1.set_xlabel('Number of Positive Values per Patch (0-64)', fontsize=13)
#     ax1.set_ylabel('Probability Density', fontsize=13)
#     ax1.set_title(f'Distribution of Positive Counts vs Binomial\n(Random Gaussian Weights: N(0, {conv_init_std}²))',
#                   fontsize=14, fontweight='bold')
#     ax1.legend(fontsize=11)
#     ax1.grid(alpha=0.3)
#     ax1.set_xlim(-1, encoding_size + 1)
#
#     # 2. 累积分布对比
#     ax2 = fig.add_subplot(gs[0, 2])
#     sorted_counts = np.sort(all_positive_counts)
#     empirical_cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
#
#     ax2.plot(sorted_counts, empirical_cdf, 'b-', linewidth=2,
#              label='Empirical CDF', alpha=0.7)
#
#     theoretical_cdf = stats.binom.cdf(sorted_counts, encoding_size, p_positive)
#     ax2.plot(sorted_counts, theoretical_cdf, 'r--', linewidth=2,
#              label='Binomial CDF', alpha=0.7)
#
#     ax2.set_xlabel('Positive Count', fontsize=12)
#     ax2.set_ylabel('Cumulative Probability', fontsize=12)
#     ax2.set_title('CDF Comparison', fontsize=13, fontweight='bold')
#     ax2.legend(fontsize=10)
#     ax2.grid(alpha=0.3)
#
#     # 3. 输出值分布 vs 高斯分布
#     ax3 = fig.add_subplot(gs[1, :2])
#     ax3.hist(all_patch_outputs, bins=100, density=True, alpha=0.7,
#              color='steelblue', edgecolor='black', label='Output Values')
#
#     # 拟合高斯分布
#     mu, sigma = all_patch_outputs.mean(), all_patch_outputs.std()
#     x_range = np.linspace(all_patch_outputs.min(), all_patch_outputs.max(), 200)
#     ax3.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
#              'r-', linewidth=2.5, label=f'Fitted Gaussian\nμ={mu:.4f}, σ={sigma:.4f}')
#
#     # 理论高斯（基于理论标准差）
#     ax3.plot(x_range, stats.norm.pdf(x_range, 0, theoretical_output_std),
#              'g--', linewidth=2, alpha=0.7,
#              label=f'Theoretical Gaussian\nμ=0, σ={theoretical_output_std:.4f}')
#
#     # 标准高斯
#     ax3.plot(x_range, stats.norm.pdf(x_range, 0, 1),
#              'orange', linestyle=':', linewidth=2, alpha=0.5, label='Standard Gaussian\nμ=0, σ=1')
#
#     ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
#     ax3.set_xlabel('Output Value', fontsize=13)
#     ax3.set_ylabel('Density', fontsize=13)
#     ax3.set_title('Output Value Distribution vs Gaussian', fontsize=14, fontweight='bold')
#     ax3.legend(fontsize=10)
#     ax3.grid(alpha=0.3)
#
#     # 4. Q-Q图（输出值）
#     ax4 = fig.add_subplot(gs[1, 2])
#     stats.probplot(sample_data, dist="norm", plot=ax4)
#     ax4.set_title('Q-Q Plot\n(Output Values)', fontsize=13, fontweight='bold')
#     ax4.grid(alpha=0.3)
#
#     # 5. 每个trial的平均正值个数
#     ax5 = fig.add_subplot(gs[2, 0])
#     samples_per_trial = batch_size * n_patches_per_image
#     mean_positive_per_trial = []
#
#     for i in range(n_trials):
#         start_idx = i * samples_per_trial
#         end_idx = start_idx + samples_per_trial
#         trial_mean = all_positive_counts[start_idx:end_idx].mean()
#         mean_positive_per_trial.append(trial_mean)
#
#     mean_positive_per_trial = np.array(mean_positive_per_trial)
#
#     ax5.plot(range(1, n_trials + 1), mean_positive_per_trial, 'o-',
#              markersize=8, linewidth=2, color='steelblue')
#     ax5.axhline(theoretical_mean, color='green', linestyle='--',
#                 linewidth=2, label=f'Theoretical: {theoretical_mean:.1f}')
#     ax5.axhline(mean_positive_per_trial.mean(), color='red', linestyle='-',
#                 linewidth=2, label=f'Mean: {mean_positive_per_trial.mean():.2f}')
#
#     ax5.set_xlabel('Trial Number', fontsize=12)
#     ax5.set_ylabel('Mean Positive Count', fontsize=12)
#     ax5.set_title('Mean Positive Count per Trial', fontsize=13, fontweight='bold')
#     ax5.legend(fontsize=10)
#     ax5.grid(alpha=0.3)
#     ax5.set_xticks(range(1, n_trials + 1))
#
#     # 6. 正值个数的箱线图
#     ax6 = fig.add_subplot(gs[2, 1])
#     bp = ax6.boxplot([all_positive_counts], widths=0.6, patch_artist=True,
#                      boxprops=dict(facecolor='lightblue', color='black'),
#                      medianprops=dict(color='red', linewidth=2.5),
#                      whiskerprops=dict(color='black', linewidth=1.5),
#                      capprops=dict(color='black', linewidth=1.5))
#
#     ax6.axhline(theoretical_mean, color='green', linestyle='--',
#                 linewidth=2, label=f'Theoretical: {theoretical_mean:.1f}')
#     ax6.set_ylabel('Positive Count per Patch', fontsize=12)
#     ax6.set_title('Box Plot of Positive Counts', fontsize=13, fontweight='bold')
#     ax6.set_xticklabels(['All Patches'])
#     ax6.legend(fontsize=10)
#     ax6.grid(axis='y', alpha=0.3)
#
#     # 7. 卡方拟合优度检验可视化
#     ax7 = fig.add_subplot(gs[2, 2])
#
#     # 计算观察频率和期望频率
#     observed_freq, _ = np.histogram(all_positive_counts, bins=np.arange(0, encoding_size + 2) - 0.5)
#     expected_freq = binom_pmf * len(all_positive_counts)
#
#     x_pos = np.arange(len(observed_freq))
#     width = 0.35
#
#     ax7.bar(x_pos - width / 2, observed_freq, width, label='Observed',
#             alpha=0.7, color='steelblue', edgecolor='black')
#     ax7.bar(x_pos + width / 2, expected_freq, width, label='Expected (Binomial)',
#             alpha=0.7, color='coral', edgecolor='black')
#
#     ax7.set_xlabel('Positive Count', fontsize=12)
#     ax7.set_ylabel('Frequency', fontsize=12)
#     ax7.set_title('Observed vs Expected Frequencies', fontsize=13, fontweight='bold')
#     ax7.legend(fontsize=10)
#     ax7.grid(axis='y', alpha=0.3)
#
#     # 卡方检验
#     mask = expected_freq >= 5
#     if mask.sum() > 0:
#         try:
#             obs = observed_freq[mask]
#             exp = expected_freq[mask]
#             exp = exp * obs.sum() / exp.sum()
#
#             chi2_stat, chi2_p = stats.chisquare(obs, exp)
#             ax7.text(0.95, 0.95, f'χ² test:\nχ²={chi2_stat:.2f}\np={chi2_p:.4f}',
#                      transform=ax7.transAxes, fontsize=10,
#                      verticalalignment='top', horizontalalignment='right',
#                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#         except ValueError as e:
#             print(f"\nWarning: Chi-square test skipped: {str(e)[:100]}...")
#             ax7.text(0.95, 0.95, 'χ² test:\n(numerical issue)',
#                      transform=ax7.transAxes, fontsize=10,
#                      verticalalignment='top', horizontalalignment='right',
#                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
#
#     plt.savefig('patch_conv_random_weights_analysis.png', dpi=300, bbox_inches='tight')
#
#     print("\n" + "=" * 80)
#     print("Figure saved as 'patch_conv_random_weights_analysis.png'")
#     print("=" * 80)
#
#     return all_positive_counts, all_patch_outputs, encoder
#
#
# # 运行测试
# if __name__ == "__main__":
#     # 测试不同的初始化标准差
#     for init_std in [1.0, 0.5, 2.0]:
#         print(f"\n\n{'#' * 80}")
#         print(f"# Testing with initialization std = {init_std}")
#         print(f"{'#' * 80}\n")
#
#         positive_counts, patch_outputs, model = test_patch_conv_distribution(
#             n_trials=10,
#             batch_size=64,
#             image_size=256,
#             patch_size=16,
#             conv_init_std=init_std,
#             seed=42
#         )
