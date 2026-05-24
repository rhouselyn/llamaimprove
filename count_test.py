import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
from dataset.augmentation import random_crop_arr

# PatchMLPEncoder 类定义
class PatchMLPEncoder(nn.Module):
    """
    使用 MLP 处理 RGB patch 的编码器

    结构：
    1. Conv2d(3, hidden_dim, kernel_size=(4,2), stride=(4,2)) - 每个 4×2 区域输出 hidden_dim 维特征
    2. Tanh 激活
    3. Linear(hidden_dim, 1) - 映射到最终编码值
    4. LayerNorm - 在最终输出上应用Layer Normalization

    对于 16×16 patch：
    - 经过 (4,2) conv 后得到 4×8 的 feature map
    - 每个位置有 hidden_dim 个特征
    - Flatten 后得到 32 × hidden_dim 维向量
    - 通过 Linear 层映射到 32 维编码
    - 应用 LayerNorm
    """

    def __init__(self, hidden_dim=24, learnable=True):
        super().__init__()

        # 第一层：卷积提取特征
        # 输入: [batch, 3, 16, 16]
        # 输出: [batch, hidden_dim, 4, 8]
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=(4,2), stride=(4,2), bias=False)

        # Tanh 激活函数
        self.activation = nn.ReLU()

        # 第二层：线性映射到最终编码
        # 每个 4×8 位置的 hidden_dim 维特征独立映射到 1 维
        self.linear = nn.Linear(hidden_dim, 1, bias=False)

        # Layer Normalization
        self.norm = nn.LayerNorm(32, elementwise_affine=False)  # encoding_size = 32

        # 初始化权重
        with torch.no_grad():
            # Conv 层初始化为 1/24（因为 3*4*2=24）
            self.conv.weight.fill_(1.0 / 24.0)
            # Linear 层使用 Xavier 初始化
            nn.init.xavier_uniform_(self.linear.weight)

        # 设置是否可学习
        self.conv.weight.requires_grad = learnable
        self.linear.weight.requires_grad = learnable
        if not learnable:
            for p in self.norm.parameters():
                p.requires_grad = False
        self.learnable = learnable
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        输入: [batch_size, 3, patch_size, patch_size]  # RGB patches
        输出: [batch_size, encoding_size]  # encoding_size = (4*8)=32
        """
        batch_size = x.shape[0]

        # 1. 卷积提取特征
        x = self.conv(x)  # [batch_size, hidden_dim, 4, 8]

        # 2. Tanh 激活
        x = self.activation(x)  # [batch_size, hidden_dim, 4, 8]

        # 3. 重排维度以便线性层处理
        # [batch_size, hidden_dim, 4, 8] -> [batch_size, 4, 8, hidden_dim]
        x = x.permute(0, 2, 3, 1)

        # 4. 线性映射：每个位置的 hidden_dim 维特征 -> 1 维
        x = self.linear(x)  # [batch_size, 4, 8, 1]

        # 5. 去除最后一维并展平
        x = x.squeeze(-1)  # [batch_size, 4, 8]
        x = x.flatten(1)  # [batch_size, 32]

        # 6. 应用 LayerNorm
        x = self.norm(x)

        return x


def setup_logger(log_dir=None):
    """简单的logger设置，不依赖分布式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'test.log')) if log_dir else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def analyze_and_plot_positive_counts(encoder, data_loader, device, n_batches=10,
                                     save_path='positive_count_comparison.png', logger=None):
    """
    分析前 n_batches 个 batch 的正值 count 分布，并与标准 Binomial(64, 0.5) PMF 对比绘制图片。
    """
    encoder.eval()
    all_positive_counts = []
    all_patch_outputs = []
    total_patches = 0

    patch_size = 16
    encoding_size = 64

    if logger:
        logger.info(f"Analyzing first {n_batches} batches...")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            images = images.to(device, non_blocking=True)
            batch_patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
            batch_patches = batch_patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)

            encodings = encoder(batch_patches)
            positive_counts = (encodings > 0).sum(dim=1).cpu().numpy()

            all_positive_counts.extend(positive_counts)
            all_patch_outputs.append(encodings.cpu().numpy())
            total_patches += len(positive_counts)

            if logger:
                logger.info(f"Batch {batch_idx + 1}/{n_batches}: processed {len(positive_counts)} patches")

    all_positive_counts = np.array(all_positive_counts)
    all_patch_outputs_np = np.concatenate(all_patch_outputs, axis=0).flatten()
    p_positive = (all_patch_outputs_np > 0).mean()

    # 打印详细统计信息
    print(f"\n{'=' * 80}")
    print("正值个数分布分析 (Positive Count Distribution Analysis)")
    print(f"{'=' * 80}")
    print(f"总patch数量: {len(all_positive_counts):,}")
    print(f"平均值: {all_positive_counts.mean():.4f} (目标: 32.0)")
    print(f"标准差: {all_positive_counts.std():.4f} (目标: 4.0)")
    print(f"最小值: {all_positive_counts.min()}")
    print(f"最大值: {all_positive_counts.max()}")
    print(f"P(X>0): {p_positive:.6f} (目标: 0.5)")

    # 计算与目标分布的KL散度
    target_pmf = stats.binom.pmf(np.arange(encoding_size + 1), encoding_size, 0.5)
    observed_pmf, _ = np.histogram(all_positive_counts, bins=np.arange(encoding_size + 2) - 0.5, density=True)
    # 移除错误的归一化行，直接使用 observed_pmf 作为概率（已由 density=True 归一化）

    kl_div = stats.entropy(observed_pmf[:encoding_size + 1], target_pmf)
    chi2_stat = \
    stats.chisquare(observed_pmf[:encoding_size + 1] * len(all_positive_counts), target_pmf * len(all_positive_counts))[
        0]

    print(f"KL散度 (vs Binom(64,0.5)): {kl_div:.6f}")
    print(f"卡方统计量: {chi2_stat:.6f}")

    # 绘制对比图
    plt.figure(figsize=(14, 8))

    # 主图：直方图 + PMF
    plt.subplot(1, 2, 1)
    plt.hist(all_positive_counts, bins=np.arange(0, encoding_size + 2) - 0.5,
             density=True, alpha=0.7, color='steelblue', edgecolor='black',
             label=f'观测值 (Observed, n={len(all_positive_counts):,} patches)')

    x_binom = np.arange(0, encoding_size + 1)
    plt.plot(x_binom, stats.binom.pmf(x_binom, encoding_size, 0.5), 'go-', linewidth=3, markersize=6,
             label='目标分布 Binomial(64, 0.5)', alpha=0.9)
    plt.plot(x_binom, stats.binom.pmf(x_binom, encoding_size, p_positive), 'ro--', linewidth=2.5, markersize=4,
             label=f'实际分布 Binomial(64, {p_positive:.3f})', alpha=0.8)

    plt.axvline(32, color='green', linestyle='--', linewidth=2, alpha=0.7, label='目标均值 32')
    plt.axvline(all_positive_counts.mean(), color='red', linestyle='-', linewidth=2,
                label=f'观测均值 {all_positive_counts.mean():.1f}')

    plt.xlabel('每个Patch的正值个数 (Number of Positive Values per Patch)', fontsize=12)
    plt.ylabel('概率密度 (Probability Density)', fontsize=12)
    plt.title('正值个数分布对比 (Positive Count Distribution Comparison)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 统计信息文本框
    textstr = f'''统计信息:
均值: {all_positive_counts.mean():.2f} (目标: 32.0)
标准差: {all_positive_counts.std():.2f} (目标: 4.0)
P(X>0): {p_positive:.4f}
总patch数: {len(all_positive_counts):,}
KL散度: {kl_div:.6f}'''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # 子图：观测PMF vs 目标PMF
    plt.subplot(1, 2, 2)
    observed_pmf_smooth = np.convolve(observed_pmf[:encoding_size + 1], np.ones(3) / 3, mode='same')
    plt.plot(x_binom, target_pmf, 'g-', linewidth=3, label='目标PMF Binomial(64, 0.5)', alpha=0.9)
    plt.plot(x_binom, observed_pmf_smooth, 'b-', linewidth=2, label='观测PMF', alpha=0.8, marker='o', markersize=3)
    plt.fill_between(x_binom, observed_pmf[:encoding_size + 1], alpha=0.3, color='steelblue')

    plt.xlabel('正值个数 k (Positive Count k)', fontsize=12)
    plt.ylabel('概率质量函数 PMF', fontsize=12)
    plt.title('观测PMF vs 目标PMF对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 添加误差条
    observed_counts, _ = np.histogram(all_positive_counts, bins=np.arange(encoding_size + 2) - 0.5)
    observed_pmf_error = np.sqrt(observed_counts) / len(all_positive_counts)
    plt.errorbar(x_binom, observed_pmf[:encoding_size + 1], yerr=observed_pmf_error[:encoding_size + 1],
                 fmt='none', ecolor='blue', alpha=0.5, capsize=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {save_path}")
    plt.close()

    return {
        'positive_counts': all_positive_counts,
        'p_positive': p_positive,
        'kl_divergence': kl_div,
        'total_patches': total_patches
    }


def main(args):
    """主测试函数"""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建实验目录
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = setup_logger(save_dir)
    else:
        logger = setup_logger()

    if logger:
        logger.info(f"加载模型权重: {args.checkpoint_path}")

    # 加载模型
    try:
        # 添加白名单以支持 argparse.Namespace (用于 PyTorch 2.6+ 的 weights_only=True)
        import torch.serialization
        torch.serialization.add_safe_globals([argparse.Namespace])

        encoder = PatchMLPEncoder(hidden_dim=args.hidden_dim, learnable=False).to(device)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        # 处理可能来自DDP的检查点
        state_dict = checkpoint['model']
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        encoder.load_state_dict(state_dict)
        encoder.eval()
        if logger:
            logger.info("模型加载成功")
        print("模型加载成功")

    except Exception as e:
        print(f"加载模型失败: {e}")
        if logger:
            logger.error(f"加载模型失败: {e}")
        return

    # 数据变换（与训练一致）
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 构建数据集和DataLoader
    try:
        from dataset.build import build_dataset  # 假设你的 build_dataset 在这个路径
        dataset = build_dataset(args, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.global_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        if logger:
            logger.info(f"数据集加载成功，共 {len(dataset):,} 张图片")
        print(f"数据集加载成功，共 {len(dataset):,} 张图片")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        if logger:
            logger.error(f"数据集加载失败: {e}")
        return

    # 分析并绘图
    save_path = args.save_path if args.save_path else 'positive_count_comparison.png'
    results = analyze_and_plot_positive_counts(
        encoder, loader, device,
        n_batches=args.n_batches,
        save_path=save_path,
        logger=logger
    )

    if logger:
        logger.info(f"分析完成，结果已保存至 {save_path}")
    print(f"\n{'=' * 60}")
    print("测试总结:")
    print(f"处理了 {results['total_patches']:,} 个patches")
    print(f"正值比例 P(X>0): {results['p_positive']:.6f}")
    print(f"与目标分布的KL散度: {results['kl_divergence']:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试编码器的正值个数分布")
    # 模型参数
    parser.add_argument("--hidden-dim", type=int, default=12, help="MLP编码器的隐藏维度")
    # 数据参数
    parser.add_argument("--data-path", type=str, default='imagenet_train_filelist.txt')
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    # 加载检查点
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="训练好的编码器检查点路径 (如 encoder_best.pt)")
    # 保存路径
    parser.add_argument("--save-path", type=str, default=None,
                        help="保存图片的路径 (默认: positive_count_comparison.png)")
    # 测试参数
    parser.add_argument("--n-batches", type=int, default=10,
                        help="分析的batch数量 (默认: 10)")
    # DataLoader参数
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
