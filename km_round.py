#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

def compute_uniformity_loss(z: torch.Tensor, t: float = 3.0) -> torch.Tensor:
    """
    基于高斯势核的均匀性损失：
    loss = log( mean_{i<j} exp( -t * ||x_i - x_j||^2 ) )
    先对输入向量逐行 L2 归一化到单位球面再计算。
    """
    z_norm = F.normalize(z, p=2, dim=-1)
    sq_pdist = torch.pdist(z_norm, p=2).pow(2)  # pairwise squared L2 distances
    loss = sq_pdist.mul(-t).exp().mean().log()
    return loss

def optimize_random_codebook(
    N: int = 4096,
    dim: int = 6,
    steps: int = 1000,
    lr: float = 1e-2,
    t: float = 3.0,
    seed: int = 42,
    device: str | torch.device = None,
):
    """
    生成随机高斯初始化向量并在单位球面上优化 uniformity loss。
    返回训练后的 (N, dim) torch.Tensor（在 CPU 上）。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gaussian init + normalize to unit sphere
    codebook = torch.randn(N, dim, device=device)
    codebook = F.normalize(codebook, p=2, dim=1)
    codebook.requires_grad_(True)

    optimizer = torch.optim.Adam([codebook], lr=lr)

    for step in range(1, steps + 1):
        loss = compute_uniformity_loss(codebook, t=t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # project back to unit sphere
        with torch.no_grad():
            codebook.data = F.normalize(codebook.data, p=2, dim=1)

        if step % 100 == 0 or step == 1 or step == steps:
            print(f"[Step {step:4d}] loss = {loss.item():.6f}")

    return codebook.detach().cpu()

def per_dimension_kmeans_report(X: np.ndarray, n_clusters: int = 4, seed: int = 42):
    """
    对每个维度做 KMeans 聚 4 类。
    按簇中心从小到大输出：centers、stds、sizes。
    """
    N, dim = X.shape
    print("\n=== Per-Dimension KMeans(4) summary ===")
    for d in range(dim):
        vals = X[:, d].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(vals)
        centers = kmeans.cluster_centers_.flatten()

        # 按中心值排序，便于一致展示
        order = np.argsort(centers)
        centers_sorted = centers[order]

        stds_sorted = []
        sizes_sorted = []
        for idx in order:
            cluster_vals = vals[labels == idx].reshape(-1)
            # 使用无偏标准差（样本标准差）
            std = cluster_vals.std(ddof=1) if cluster_vals.size > 1 else 0.0
            stds_sorted.append(std)
            sizes_sorted.append(cluster_vals.size)

        centers_str = ", ".join(f"{c:.6f}" for c in centers_sorted)
        stds_str    = ", ".join(f"{s:.6f}" for s in stds_sorted)
        sizes_str   = ", ".join(str(n) for n in sizes_sorted)

        print(f"Dim {d}:")
        print(f"  centers = [{centers_str}]")
        print(f"  stds    = [{stds_str}]")
        print(f"  sizes   = [{sizes_str}]")

def main():
    # 配置
    N = 4096
    dim = 6
    steps = 1000
    lr = 1e-2
    t = 3.0
    seed = 42

    codebook = optimize_random_codebook(
        N=N, dim=dim, steps=steps, lr=lr, t=t, seed=seed
    )

    X = codebook.numpy()
    per_dimension_kmeans_report(X, n_clusters=4, seed=seed)

if __name__ == "__main__":
    main()
