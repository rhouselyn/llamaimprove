import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest  # For Gaussian fitting and KS test
import math
from collections import Counter
from torchvision import transforms
from dataset.augmentation import random_crop_arr

# Assuming dataset.augmentation and dataset.build are available
# If not, you'll need to provide or mock these dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from torch.utils.data import DataLoader


# Custom PCA Whitening Module with accumulative stats
class PCAWhitening(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_cov', torch.eye(dim))
        self.register_buffer('whitening_matrix', torch.eye(dim))  # Initial identity
        self.count = 0

    def update_stats(self, x):
        # x: (N, dim)
        batch_mean = x.mean(dim=0)
        batch_cov = torch.mm((x - batch_mean).t(), (x - batch_mean)) / (x.size(0) - 1)

        self.count += x.size(0)
        if self.count == x.size(0):  # First batch
            self.running_mean = batch_mean
            self.running_cov = batch_cov
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * batch_cov

    def compute_whitening_matrix(self):
        # Compute PCA matrix from running_cov
        U, S, Vt = torch.svd(self.running_cov, some=False)
        S = torch.clamp(S, min=self.eps)
        whitening = torch.mm(U, torch.diag(1.0 / torch.sqrt(S)))
        self.whitening_matrix = whitening

    def forward(self, x, train_mode=False):
        # x: (N, dim)
        if train_mode:
            self.update_stats(x)
            return x  # During accumulation, return original (or optionally whiten with current)
        else:
            # Subtract mean and apply whitening
            x = x - self.running_mean
            x = torch.mm(x, self.whitening_matrix)  # Apply PCA
            return x


# Modified Encoder with Conv + Flatten + Norm + Linear + PCA (based on PatchMLPEncoder architecture, no sign)
class Encoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, encoding_dim=14, learnable=False):
        super().__init__()
        # Set seed for reproducible random initialization
        torch.manual_seed(64)

        self.patch_size = patch_size
        self.encoding_dim = encoding_dim
        self.learnable = learnable

        # Conv layer: Conv2d(3, 2, kernel_size=(4,4), stride=(4,4)) for each patch
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=(4,4), stride=(4,4), bias=False)

        # LayerNorm after flatten (on 32 dim)
        self.norm = nn.LayerNorm(32)

        # Linear layer: input 32 (2*4*4), output encoding_dim
        self.linear = nn.Linear(32, encoding_dim, bias=True)  # 修改：bias=True，初始bias=0

        # Initialize weights
        with torch.no_grad():
            # Conv: xavier init
            nn.init.xavier_uniform_(self.conv.weight)
            # Linear: xavier init
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)  # 初始bias=0，等价原bias=False

        self.fused = False  # 标志是否融合PCA

        # Set conv fixed, linear learnable based on param
        self.conv.weight.requires_grad = False
        self.linear.weight.requires_grad = self.learnable
        self.linear.bias.requires_grad = self.learnable  # bias也根据learnable

    def fuse_pca(self, whitening_matrix, mean):
        # 融合推导：new_weight = whitening_matrix.t() @ self.linear.weight
        # new_bias = - (mean @ whitening_matrix)
        new_weight = whitening_matrix.t() @ self.linear.weight
        new_bias = - (mean @ whitening_matrix)
        self.linear.weight.data.copy_(new_weight)
        self.linear.bias.data.copy_(new_bias)
        self.fused = True

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size."

        # Unfold to patches: (B, C * patch_size^2, num_patches)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        # Reshape to (B * num_patches, C, patch_size, patch_size) for conv processing
        num_patches = patches.size(2)
        h = patches.view(B, C, self.patch_size, self.patch_size, num_patches)
        h = h.permute(0, 4, 1, 2, 3).reshape(-1, C, self.patch_size, self.patch_size)  # (B*num_patches, 3, 16, 16)

        # Apply conv: (B*num_patches, 2, 4, 4)
        h = self.conv(h)

        # Flatten to (B*num_patches, 32)
        h = h.flatten(1)

        # Apply LayerNorm
        h = self.norm(h)

        # Apply linear: to (B*num_patches, encoding_dim)
        h = self.linear(h)

        # 修改：移除pca调用，直接返回linear_out（融合后即whitened）
        h = h.view(B, H // self.patch_size, W // self.patch_size, self.encoding_dim)
        h = h.permute(0, 3, 1, 2).contiguous()
        return h


def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."
    device = 'cuda'
    torch.cuda.set_device(0)

    # Create Encoder instance
    encoder = Encoder(
        in_channels=3,
        patch_size=16,
        encoding_dim=args.encoding_dim,  # Use args.encoding_dim (default 16)
        learnable=False
    )
    encoder.to(device)
    encoder.eval()

    # Setup data: remove normalize_per_image since BatchNorm handles it
    crop_size = args.image_size
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 修改：临时PCA用于积累stats
    temp_pca = PCAWhitening(args.encoding_dim).to(device)

    # First pass: Accumulate PCA stats (用linear输出更新temp_pca)
    print("First pass: Accumulating PCA stats...")
    batch_count = 0
    for x, _ in loader:
        if args.num_batches > 0 and batch_count >= args.num_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            # 计算到linear输入前的conv和norm
            B, C, H, W = x.shape
            patches = F.unfold(x, kernel_size=encoder.patch_size, stride=encoder.patch_size)
            num_patches = patches.size(2)
            h = patches.view(B, C, encoder.patch_size, encoder.patch_size, num_patches)
            h = h.permute(0, 4, 1, 2, 3).reshape(-1, C, encoder.patch_size, encoder.patch_size)
            h = encoder.conv(h)
            h = h.flatten(1)
            norm_out = encoder.norm(h)
            # 计算linear_out
            linear_out = encoder.linear(norm_out)  # [B*num_patches, encoding_dim]
            # 更新temp_pca
            temp_pca.update_stats(linear_out)
        batch_count += 1

    # Compute whitening matrix after accumulation
    temp_pca.compute_whitening_matrix()
    print("PCA whitening matrix computed.")

    # 修改：直接fuse到encoder.linear
    encoder.fuse_pca(temp_pca.whitening_matrix, temp_pca.running_mean)
    print("Fused PCA into Encoder linear layer.")

    # Second pass: Collect sampled z_e vectors (whitened) for visualization and usage
    all_vectors = []
    # Counter for codebook usage (binary sign-based discretization to encoding_dim-bit decimal codes)
    usage_counter = Counter()
    codebook_size = 1 << args.encoding_dim  # 2^encoding_dim
    powers = 2 ** np.arange(args.encoding_dim)  # Precompute powers of 2 for binary to decimal conversion
    print("Second pass: Collecting whitened vectors and codebook usage...")

    batch_count = 0
    for x, _ in loader:  # Ignore labels
        if args.num_batches > 0 and batch_count >= args.num_batches:
            break

        x = x.to(device)
        with torch.no_grad():
            # Get z_e from encoder (已融合，无需train_mode)
            z_e = encoder(x)
            print(f"z_e shape from encoder: {z_e.shape}")  # Debug: should be (batch, encoding_dim, 16, 16) for 256x256 image
            # Reshape to (batch, num_tokens, embed_dim) then flatten to (total_tokens, embed_dim)
            z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1,
                                                       args.encoding_dim).detach().cpu().numpy()  # (batch * num_tokens, encoding_dim)

        # Sample for continuous vectors (for plots)
        if len(all_vectors) < args.sample_size:
            num_to_sample = min(args.sample_size - len(all_vectors), len(z_e_flat))
            if num_to_sample > 0:
                sample_idx = random.sample(range(len(z_e_flat)), num_to_sample)
                sampled = z_e_flat[sample_idx]
                all_vectors.extend(sampled.tolist())

        # Always update codebook usage from full batch (discretize by sign to binary, convert to decimal)
        z_binary = (z_e_flat > 0).astype(int)  # (num_tokens, encoding_dim), 0 or 1
        codes = np.dot(z_binary, powers).astype(int)  # Vectorized: each row dot powers -> decimal code
        usage_counter.update(codes)

        batch_count += 1

    all_vectors = np.array(all_vectors)  # (sample_size, encoding_dim)
    print(f"Collected {len(all_vectors)} vectors for visualization.")
    print(f"Unique codes mapped: {len(usage_counter)} / {codebook_size}")
    if len(usage_counter) == codebook_size:
        print("All possible {args.encoding_dim}-bit binary combinations are mapped!")
    else:
        print("Not all possible {args.encoding_dim}-bit binary combinations are mapped.")

    # Save codebook utilization to txt (used_indices / total_indices)
    usage_file = os.path.join(args.output_path, "codebook_usage.txt")
    utilization = len(usage_counter) / codebook_size
    with open(usage_file, "w") as f:
        f.write(f"Utilization: {len(usage_counter)} / {codebook_size} ({utilization * 100:.4f}%)\n")
    print(f"Saved codebook utilization to {usage_file}")

    # Visualize: scatter plots for paired dims in one figure (dynamic for arbitrary even dim)
    pairs = [(i, i + 1) for i in range(0, args.encoding_dim, 2)]
    num_pairs = len(pairs)
    rows = math.ceil(num_pairs / 2)
    fig, axs = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axs = axs.flatten() if rows > 1 else [axs]
    for i, (dim1, dim2) in enumerate(pairs):
        ax = axs[i]
        ax.scatter(all_vectors[:, dim1], all_vectors[:, dim2], alpha=0.5, s=1)
        ax.set_xlabel(f"Dim {dim1}")
        ax.set_ylabel(f"Dim {dim2}")
        ax.set_title(f"Scatter of Dims {dim1}-{dim2}")
    plt.tight_layout()
    output_file = os.path.join(args.output_path, "vector_distribution.png")
    plt.savefig(output_file)
    print(f"Saved distribution plot to {output_file}")

    # Analyze each dimension's closeness to Gaussian distribution
    print("\nAnalyzing each dimension's distribution...")
    rows_gauss = math.ceil(args.encoding_dim / 4)
    fig_gauss, axs_gauss = plt.subplots(rows_gauss, 4, figsize=(20, 5 * rows_gauss))
    axs_gauss = axs_gauss.flatten()
    for dim in range(args.encoding_dim):
        data = all_vectors[:, dim]
        mean = np.mean(data)
        std = np.std(data)
        standardized = (data - mean) / std if std != 0 else data  # Standardize to N(0,1)

        # KS test against standard normal
        ks_stat, p_value = kstest(standardized, 'norm')

        # Print results
        print(f"Dim {dim}: Mean={mean:.4f}, Std={std:.4f}, KS Stat={ks_stat:.4f}, p-value={p_value:.4f}")
        if p_value > 0.05:
            print(f"  - Likely Gaussian (p > 0.05)")
        else:
            print(f"  - Not Gaussian (p <= 0.05)")

        # Plot histogram + Gaussian PDF
        ax = axs_gauss[dim]
        ax.hist(standardized, bins=50, density=True, alpha=0.6, color='b', label='Data Hist')
        x = np.linspace(-4, 4, 100)
        ax.plot(x, norm.pdf(x), 'r-', lw=2, label='N(0,1) PDF')
        ax.set_title(f"Dim {dim}: KS={ks_stat:.4f}, p={p_value:.4f}")
        ax.set_xlabel("Standardized Value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    gauss_output_file = os.path.join(args.output_path, "gaussian_fit_per_dim.png")
    plt.savefig(gauss_output_file)
    print(f"Saved Gaussian fit plot to {gauss_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of vectors to sample for visualization")
    parser.add_argument("--output-path", type=str, default='./', help="Path to save the plot")
    parser.add_argument("--encoding-dim", type=int, default=24, help="Encoder output dimension")
    parser.add_argument("--num-batches", type=int, default=-1, help="Number of batches to process, -1 for all")
    parser.add_argument("--aoss-bucket", type=str, default="imagenet",
                        help="AOSS bucket name (only for aoss_imagenet dataset)")
    args = parser.parse_args()
    main(args)
