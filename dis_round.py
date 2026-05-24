import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def compute_uniformity_loss(z: torch.Tensor, t: float = 3) -> torch.Tensor:
    # 归一化到单位球面
    z_norm = F.normalize(z, p=2, dim=-1)
    # 两两距离平方
    sq_pdist = torch.pdist(z_norm, p=2).pow(2)
    # 均匀性损失
    loss = sq_pdist.mul(-t).exp().mean().log()
    return loss

def compute_full_metrics(points: torch.Tensor, batch_size: int = 1024):
    N = points.size(0)
    global_min = float('inf')
    min_dists_sum = 0.0
    all_dists_sum = 0.0
    total_pairs = N * (N - 1)

    for i in range(0, N, batch_size):
        batch1 = points[i:i + batch_size]
        batch_min = float('inf')
        batch_min_dists = None
        batch_dists_sum = 0.0
        batch_count = 0

        for j in range(0, N, batch_size):
            batch2 = points[j:j + batch_size]
            dists = torch.cdist(batch1, batch2)
            if i == j:
                dists.fill_diagonal_(float('inf'))
                finite_mask = torch.isfinite(dists)
                batch_dists_sum += dists[finite_mask].sum().item()
                batch_count += finite_mask.sum().item()
            else:
                batch_dists_sum += dists.sum().item()
                batch_count += dists.numel()

            batch_min = min(batch_min, dists.min().item())
            row_mins = dists.min(dim=1)[0]
            if batch_min_dists is None:
                batch_min_dists = row_mins
            else:
                batch_min_dists = torch.min(batch_min_dists, row_mins)

        global_min = min(global_min, batch_min)
        min_dists_sum += batch_min_dists.sum().item()
        all_dists_sum += batch_dists_sum

    mean_min_dist = min_dists_sum / N
    mean_dist = all_dists_sum / total_pairs
    return global_min, mean_min_dist, mean_dist

def main():
    # —— 这里按你的要求设置每个维度的离散“个数” ——
    per_dim_bins = [2, 2, 2, 4, 4, 4, 5, 6]   # 8个维度
    dim = len(per_dim_bins)

    # 输出目录
    output_dir = 'random_sphere_codebook'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Generating initial grid vectors...")
    # 为每个维度构造离散取值集合：
    # - 2个值：[-1, 1]
    # - 4个值：[-1, -0.5, 0.5, 1]（与原始代码一致）
    # - 其他个数：在[-1,1]上用等距 linspace
    value_sets = []
    for k in per_dim_bins:
        if k == 2:
            vals = torch.tensor([-1.0, 1.0], device=device)
        elif k == 4:
            vals = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)
        else:
            vals = torch.linspace(-1.0, 1.0, steps=k, device=device)
        value_sets.append(vals)

    # 生成笛卡尔积网格，并 reshape 成 (N, dim)
    grids = torch.meshgrid(*value_sets, indexing='ij')
    codebook = torch.stack(grids, dim=-1).reshape(-1, dim)
    N = codebook.size(0)
    print(f"Total vectors N = {N}")  # 期望 15360

    # 归一化并开启梯度
    codebook = F.normalize(codebook, p=2, dim=1)
    codebook.requires_grad_(True)

    optimizer = torch.optim.Adam([codebook], lr=0.01)

    print("Starting optimization...")
    for step in range(100000):
        loss = compute_uniformity_loss(codebook, t=3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保持在单位球面
        with torch.no_grad():
            codebook.data = F.normalize(codebook.data, p=2, dim=1)

        if (step + 1) % 100 == 0:
            global_min, mean_min, mean_dist = compute_full_metrics(codebook)
            print(f"Step {step + 1}: Loss = {loss.item():.4f}")
            print(f"Global min pairwise dist: {global_min:.4f}")
            print(f"Mean min pairwise dist: {mean_min:.4f}")
            print(f"Mean pairwise dist: {mean_dist:.4f}")

            # —— 这里按各维的“个数”分别做KMeans，输出该维的离散值（簇中心） ——
            for d, k in enumerate(per_dim_bins):
                dim_vals = codebook[:, d].detach().cpu().numpy().reshape(-1, 1)
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(dim_vals)
                centers = sorted(kmeans.cluster_centers_.flatten())
                print(f"Dim {d} ({k} values): " + ", ".join(f"{c:.4f}" for c in centers))

    # 保存结果
    torch.save(codebook, os.path.join(output_dir, 'codebook_8d_optimized.pth'))
    print(f"Optimized codebook saved to {output_dir}/codebook_8d_optimized.pth")

    # PCA到2D仅用于可视化
    pca_vis = PCA(n_components=2)
    codebook_np = codebook.detach().cpu().numpy()
    points_2d = pca_vis.fit_transform(codebook_np)

    plt.figure(figsize=(8, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.5)
    plt.title('2D PCA Projection of 8D Optimized Vectors on Unit Sphere (N=15360)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'optimized_8d_projection.png'))
    plt.close()
    print(f"Visualization saved to {output_dir}/optimized_8d_projection.png")

if __name__ == "__main__":
    main()
