import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import re
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import torch.distributed as dist  # 添加这个导入，确保 dist 可用

from utils.logger import create_logger  # 假设这个模块存在，如果没有，可以替换为print
from dataset.augmentation import random_crop_arr  # 假设存在
from dataset.build import build_dataset  # 假设存在

import warnings

warnings.filterwarnings('ignore')


def get_experiment_dir(base_dir, name="encoder_analysis"):
    """
    自动创建带编号的experiment目录
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_num = -1
    pattern = re.compile(rf"^{re.escape(name)}(_(\d+))?$")
    for dir_name in existing_dirs:
        match = pattern.match(dir_name)
        if match:
            num = 0 if match.group(2) is None else int(match.group(2))
            max_num = max(max_num, num)
    new_dir = os.path.join(base_dir, name if max_num == -1 else f"{name}_{max_num + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

# class PatchMLPEncoder(nn.Module):
#     """
#     修改后的编码器：直接展平 patch 到 768 维，然后 Linear 到 14 维，应用 sign。
#     Linear 使用 Xavier uniform 初始化。
#     """
#
#     def __init__(self, hidden_dim=14, learnable=True):
#         super().__init__()
#
#         # 线性层：从 768 维 (3*16*16) 映射到 14 维
#         self.linear = nn.Linear(768, hidden_dim, bias=False)
#
#         # 初始化权重使用 Xavier uniform
#         nn.init.xavier_uniform_(self.linear.weight)
#
#         # 根据 learnable 设置是否可学习
#         self.linear.weight.requires_grad = learnable
#         self.learnable = learnable
#         self.hidden_dim = hidden_dim
#
#     def forward(self, x):
#         """
#         输入: [batch_size, 3, 16, 16]  # RGB patches
#         输出: [batch_size, 14]  # 最终 14 维编码 (-1/1)
#         """
#         # 直接展平到 768 维
#         x = x.flatten(1)  # [batch_size, 768]
#
#         # 线性映射到 14 维
#         x = self.linear(x)  # [batch_size, 14]
#
#         # 应用 sign 函数
#         x = torch.sign(x)
#
#         return x


class PatchMLPEncoder(nn.Module):
    """
    使用 MLP 处理 RGB patch 的编码器

    修改后的结构：
    1. Conv2d(3, 1, kernel_size=(2,2), stride=(2,2)) - 每个 2×2 区域输出 1 维特征，总输出 8×8=64 维
    2. Flatten 后得到 64 维向量
    3. Linear(64, 32) - 映射到最终 32 维编码
    4. Sign 函数 - 应用 torch.sign 到输出
    """

    def __init__(self, hidden_dim=32, learnable=True):
        super().__init__()

        # 第一层：卷积提取特征
        # 输入: [batch, 3, 16, 16]
        # 输出: [batch, 1, 8, 8]
        self.conv = nn.Conv2d(3, 1, kernel_size=(2,2), stride=(2,2), bias=False)

        # 第二层：线性映射到最终编码
        # 输入: 64 维，输出: 32 维
        self.linear = nn.Linear(64, 32, bias=False)

        self.norm = nn.LayerNorm(64)

        # 初始化权重
        with torch.no_grad():
            # conv 层使用固定常数初始化（所有权重设为相同常数，这里假设为1.0；根据输入通道3，可调整为特定逻辑，但默认统一常数）
            nn.init.constant_(self.conv.weight, 1.0/12)
            # Linear 层使用高斯初始化（正态分布，mean=0, std=1）
            # nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
            nn.init.xavier_uniform_(self.linear.weight)
        # 设置conv不可学习（固定），linear根据learnable
        self.conv.weight.requires_grad = False
        self.linear.weight.requires_grad = learnable
        self.learnable = learnable
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        输入: [batch_size, 3, patch_size, patch_size]  # RGB patches
        输出: [batch_size, 32]  # 最终 32 维编码
        """
        # 1. 卷积提取特征
        x = self.conv(x)  # [batch_size, 1, 8, 8]

        # 2. 展平到 64 维
        x = x.flatten(1)  # [batch_size, 64]

        x = self.norm(x)

        # 3. 线性映射
        x = self.linear(x)  # [batch_size, 32]

        # 4. 应用 sign 函数
        x = torch.sign(x)

        return x

# class PatchMLPEncoder(nn.Module):
#     """
#     修改后的编码器：随机初始化，不训练。
#     - Conv2d + LeakyReLU + Linear 投影到16维
#     - 加高斯噪声
#     - sign转为-1/1
#     - 不应用LayerNorm或其他
#     """
#
#     def __init__(self, hidden_dim=24, noise_std=1.0):
#         super().__init__()
#         self.conv = nn.Conv2d(3, 1, kernel_size=(4, 4), stride=(4, 4), bias=False)
#         self.activation = nn.LeakyReLU()
#         self.linear = nn.Linear(16, 16, bias=False)
#         nn.init.xavier_uniform_(self.conv.weight)
#         nn.init.xavier_uniform_(self.linear.weight)
#         # 不需要grad
#         for param in self.parameters():
#             param.requires_grad = False
#         self.noise_std = noise_std
#         self.hidden_dim = hidden_dim
#         # 用 register_buffer 注册噪声，这样 to(device) 时会自动移动
#         self.register_buffer('noise', torch.randn(16) * self.noise_std)
#
#     def forward(self, x):
#         """
#         输入: [batch_size, 3, 16, 16] (patches)
#         输出: [batch_size, 16] (binary -1/1)
#         """
#         x = self.conv(x)  # [batch, 1, 4, 4]
#         x = x.permute(0, 2, 3, 1)  # [batch, 4, 4, 1]
#         x = x.squeeze(-1).flatten(1)  # [batch, 16]
#         x = self.linear(x)  # [batch, 16]
#         # L2 归一化替换原 LayerNorm
#         # norm = torch.norm(x, p=2, dim=1, keepdim=True)
#         # x = x / (norm + 1e-8)  # 添加 epsilon 避免除零
#
#         # 加随机高斯噪声
#         # x = x + self.noise
#
#         # sign转为-1/1 (sign(0)=0，但噪声使之罕见；clamp确保范围)
#         binary = torch.sign(x).clamp(min=-1, max=1)
#         return binary



def analyze_distribution(encoder, data_loader, device, n_batches=1000, experiment_dir="."):
    """
    分析随机初始化encoder的sum分布，与理论Binomial对比
    """
    encoder.eval()
    all_sums = []
    patch_size = 16
    encoding_size = 32

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            images = images.to(device)
            patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
            patches = patches.permute(0, 2, 1).reshape(-1, 3, patch_size, patch_size)
            binary = encoder(patches)  # [n_patches, 32] (-1/1)
            sums = binary.sum(dim=1).cpu().numpy()  # [n_patches]
            all_sums.extend(sums)

    all_sums = np.array(all_sums)

    # 计算observed histogram (归一化为density)
    bins = np.arange(-32.5, 32.6, 1)  # 为了bar居中在整数点
    observed_hist, _ = np.histogram(all_sums, bins=bins, density=True)

    # 理论分布: S = 2*K - 32, K ~ Binom(32, 0.5)
    theoretical_probs = np.zeros(33)  # index 0: -32, 1: -30, ..., 32: 32
    for k in range(encoding_size + 1):
        s = 2 * k - encoding_size
        idx = (s + 32) // 2  # 0 to 32
        theoretical_probs[idx] = stats.binom.pmf(k, encoding_size, 0.5)

    # 离散点
    discrete_points = np.arange(-32, 33, 2)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.4
    ax.bar(discrete_points - width / 2, observed_hist[::2], width=width, alpha=0.7, color='steelblue',
           label='Observed (Random Encoder + Noise + Sign)')
    ax.bar(discrete_points + width / 2, theoretical_probs, width=width, alpha=0.7, color='green',
           label='Theoretical Binomial(32, 0.5) Shifted')
    ax.set_xlabel('Sum per Patch (-32 to 32)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Comparison of Sum Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    save_path = os.path.join(experiment_dir, "sum_distribution_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to: {save_path}")
    plt.close()

class PerImageNormalize(object):
    """
    自定义变换：对每个图像单独计算mean和std，然后标准化到mean=0, std=1。
    输入：tensor (C, H, W)，输出：标准化后的tensor。
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, tensor):
        # 计算每个通道的mean和std
        mean = tensor.mean(dim=[1, 2], keepdim=True)  # [C, 1, 1]
        std = tensor.std(dim=[1, 2], keepdim=True)    # [C, 1, 1]
        # 标准化：(tensor - mean) / (std + epsilon)
        normalized = (tensor - mean) / (std + self.epsilon)
        return normalized

def main(args):
    # 新增：固定种子
    seed = 0  # 可以从 args 添加 --seed 参数来配置 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 添加分布式初始化
    dist.init_process_group(backend='gloo')  # 假设用 GPU；如果 CPU，用 'gloo'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = get_experiment_dir(args.results_dir, "encoder_analysis")
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Args: {args}")

    encoder = PatchMLPEncoder(hidden_dim=args.hidden_dim).to(device)

    transform = transforms.Compose([
        transforms.Resize(256),  # 先 resize 到短边 256
        transforms.CenterCrop(256),  # 中心裁剪到 256x256（或用 RandomCrop 如果需要随机性）

        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        PerImageNormalize()  # 替换原Normalize，实现per-image高斯式标准化

    ])
    dataset = build_dataset(args, transform=transform)
    loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True)
    logger.info(f"Dataset contains {len(dataset):,} images")

    logger.info("\n" + "=" * 80 + "\nRANDOM INITIALIZED ENCODER - Sum Distribution Analysis\n" + "=" * 80)
    analyze_distribution(encoder, loader, device, n_batches=1000, experiment_dir=experiment_dir)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis completed!")
    logger.info(f"Results saved in: {experiment_dir}")
    logger.info("=" * 80)

    # 添加销毁进程组，避免异常退出警告
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--data-path", type=str, default='imagenet_val_filelist.txt')
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    # Model args
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension for MLP encoder")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Std of Gaussian noise added before sign")
    # Training args (simplified, no epochs)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    # Save args
    parser.add_argument("--results-dir", type=str, default="results_encoder")
    args = parser.parse_args()
    main(args)
