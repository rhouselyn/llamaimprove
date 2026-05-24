import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def rand_on_sphere(d: int, N: int):
    """
    生成在单位球面上的均匀随机向量（基于Matlab思路，但仅表面而非内部）。

    Args:
        d (int): 维度。
        N (int): 向量数量。

    Returns:
        torch.Tensor: 形状 (N, d) 的单位向量，在球面上均匀分布。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成随机方向（高斯随机）
    random_directions = torch.randn(d, N, device=device)

    # 归一化到单位球面（固定半径=1，去掉Matlab的随机半径部分）
    points = F.normalize(random_directions.t(), p=2, dim=1)  # 转置后归一化，形状 (N, d)

    return points


def compute_full_metrics(points: torch.Tensor, batch_size: int = 1024):
    """
    Batched computation of full set metrics:
    - Global min pairwise dist
    - Mean min pairwise dist (average of each point's min dist to others)
    - Mean pairwise dist (average of all pairwise dists)
    """
    N = points.size(0)
    device = points.device
    global_min = float('inf')
    min_dists_sum = 0.0  # For mean min
    all_dists_sum = 0.0  # For mean dist
    total_pairs = N * (N - 1)  # Total non-self pairs

    for i in range(0, N, batch_size):
        batch1 = points[i:i + batch_size]
        batch_min = float('inf')  # Per batch global min contrib
        batch_min_dists = None  # Per point min in batch
        batch_dists_sum = 0.0
        batch_count = 0

        for j in range(0, N, batch_size):
            batch2 = points[j:j + batch_size]
            dists = torch.cdist(batch1, batch2)

            if i == j:  # Mask diagonal for same batch
                dists.fill_diagonal_(float('inf'))
                # For sum, exclude inf (diagonal)
                finite_mask = torch.isfinite(dists)
                batch_dists_sum += dists[finite_mask].sum().item()
                batch_count += finite_mask.sum().item()
            else:
                # All are finite
                batch_dists_sum += dists.sum().item()
                batch_count += dists.numel()

            # Update global min
            batch_min = min(batch_min, dists.min().item())

            # For mean min: min per row across cross-batches
            row_mins = dists.min(dim=1)[0]
            if batch_min_dists is None:
                batch_min_dists = row_mins
            else:
                batch_min_dists = torch.min(batch_min_dists, row_mins)

        # Update globals
        global_min = min(global_min, batch_min)

        # Sum for mean min
        min_dists_sum += batch_min_dists.sum().item()

        # Sum for mean dist
        all_dists_sum += batch_dists_sum

    mean_min_dist = min_dists_sum / N
    mean_dist = all_dists_sum / total_pairs

    return global_min, mean_min_dist, mean_dist


# 主函数
def main():
    N = 2 ** 14  # 16384
    dim = 8
    output_dir = 'random_sphere_codebook'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating random vectors on unit sphere...")
    codebook = rand_on_sphere(dim, N)  # 生成原始球面向量

    # PCA whitening: 使用sklearn PCA with whiten=True，保持原维度
    pca = PCA(n_components=dim, whiten=True)  # whiten=True: 去相关 + 单位方差
    codebook_np = codebook.cpu().numpy()
    whitened = pca.fit_transform(codebook_np)  # 形状 (N, dim)

    # 转回tensor，并L2 normalize (norm2) 得到最终结果
    codebook_final = torch.tensor(whitened, dtype=torch.float32, device=codebook.device)
    codebook_final = F.normalize(codebook_final, p=2, dim=1)  # 单位球面

    torch.save(codebook_final, os.path.join(output_dir, 'codebook_8d_whitened.pth'))
    print(f"Whitened codebook saved to {output_dir}/codebook_8d_whitened.pth")

    # 全集量化 pairwise 参数（基于最终结果）
    global_min, mean_min, mean_dist = compute_full_metrics(codebook_final)
    print(f"True global min pairwise dist (full set): {global_min:.4f}")
    print(f"Mean min pairwise dist (full set): {mean_min:.4f}")
    print(f"Mean pairwise dist (full set): {mean_dist:.4f}")

    # PCA 降维到2D 可视化最终结果
    pca_vis = PCA(n_components=2)
    points_2d = pca_vis.fit_transform(whitened)  # 用whitened可视化（已去相关）

    plt.figure(figsize=(8, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.5)
    plt.title('2D PCA Projection of 8D Whitened Random Vectors on Unit Sphere (N=16384)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'whitened_8d_projection.png'))
    plt.close()
    print(f"Visualization saved to {output_dir}/whitened_8d_projection.png")


if __name__ == "__main__":
    main()
