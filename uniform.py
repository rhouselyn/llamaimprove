import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA


# 损失函数（不变，使用pair sampling）
def compute_gaussian_uniformity_loss(z: torch.Tensor, t: float = 5, num_pairs: int = 100000) -> torch.Tensor:
    z_norm = F.normalize(z, p=2, dim=-1)
    N = z.size(0)
    # 随机采样pairs (i, j) where i < j
    indices = torch.randint(0, N, (2 * num_pairs,), device=z.device)
    i, j = indices[:num_pairs], indices[num_pairs:]
    mask = i != j  # 避免self-pair
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    i, j = i[mask], j[mask]
    sq_dists = torch.norm(z_norm[i] - z_norm[j], p=2, dim=-1).pow(2)
    loss = sq_dists.mul(-t).exp().mean().log()
    return loss


def compute_riesz_energy_uniformity_loss(mu: torch.Tensor, s: float = 5.0, num_pairs: int = 100000) -> torch.Tensor:
    mu_norm = F.normalize(mu, p=2, dim=-1)
    N = mu.size(0)
    # 随机采样pairs (i, j) where i != j
    indices = torch.randint(0, N, (2 * num_pairs,), device=mu.device)
    i, j = indices[:num_pairs], indices[num_pairs:]
    mask = i != j
    if mask.sum() == 0:
        return torch.tensor(0.0, device=mu.device)
    i, j = i[mask], j[mask]
    cosines = torch.sum(mu_norm[i] * mu_norm[j], dim=-1)
    distances = 1 - cosines + 1e-8
    loss = torch.pow(distances, -s / 2).mean()
    return loss


# 修改量化指标：添加subsample和return_metrics参数
def compute_metrics(z: torch.Tensor, subsample: int = None, return_metrics: bool = True):
    z_norm = F.normalize(z, p=2, dim=-1).detach().cpu().numpy()

    mean_min_dist = mean_dist = mean_cosine = None
    if return_metrics:
        if subsample is not None:
            idx = np.random.choice(z_norm.shape[0], min(subsample, z_norm.shape[0]), replace=False)
            z_sub = z_norm[idx]
        else:
            z_sub = z_norm

        # 计算距离矩阵
        dists = np.linalg.norm(z_sub[:, None] - z_sub[None, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        mean_min_dist = np.mean(min_dists)
        valid_dists = dists[dists < np.inf]
        mean_dist = np.mean(valid_dists) if len(valid_dists) > 0 else 0.0

        # 计算平均余弦相似度
        cosines = np.dot(z_sub, z_sub.T)
        np.fill_diagonal(cosines, 0)
        mean_cosine = np.mean(cosines) if z_sub.shape[0] > 1 else 0.0

    # 为可视化用完整点做PCA（如果dim>3）
    if z_norm.shape[1] > 3:
        pca = PCA(n_components=3)
        z_3d = pca.fit_transform(z_norm)
    else:
        z_3d = z_norm

    return mean_min_dist, mean_dist, mean_cosine, z_3d


# 主脚本：使用subsample for loop metrics, full for print/vis
def compare_uniformity_methods(N=2 ** 14, dim=8, epochs=100, lr=0.01, output_dir='uniformity_comparison',
                               num_pairs=100000, subsample_metrics=4096, initial_z=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'Gaussian': compute_gaussian_uniformity_loss,
        'Riesz': compute_riesz_energy_uniformity_loss,
    }

    # 初始化结果和优化器
    results = {name: {'losses': [], 'mean_min_dists': [], 'mean_dists': [], 'mean_cosines': []} for name in methods}
    zs = {name: None for name in methods}
    optimizers = {name: None for name in methods}

    # 使用提供的initial_z或生成
    if initial_z is None:
        initial_z = torch.randn(N, dim, device=device) * 0.1

    # 计算初始metrics (full)
    initial_mean_min, initial_mean, initial_mean_cos, initial_z_3d = compute_metrics(initial_z, subsample=None)
    print(
        f"Initial: Mean Min Dist = {initial_mean_min:.4f}, Mean Dist = {initial_mean:.4f}, Mean Cosine = {initial_mean_cos:.4f}")

    # 保存初始可视化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(initial_z_3d[:, 0], initial_z_3d[:, 1], initial_z_3d[:, 2], s=0.1)
    ax.set_title('Initial Distribution')
    plt.savefig(os.path.join(output_dir, 'initial_3d.png'))
    plt.close()

    # 为每个方法 clone 初始 z 并设置 optimizer
    for name in methods:
        zs[name] = initial_z.clone().requires_grad_(True)
        optimizers[name] = optim.Adam([zs[name]], lr=lr)

    print(f"\nRunning Gaussian and Riesz in parallel (t=3, s=3, num_pairs={num_pairs})...")
    print("Using subsample for loop metrics, full for prints and vis.")
    for epoch in range(epochs):
        epoch_losses = {}
        for name, loss_fn in methods.items():
            optimizers[name].zero_grad()
            try:
                if name == 'Gaussian':
                    loss = loss_fn(zs[name], t=3, num_pairs=num_pairs)
                elif name == 'Riesz':
                    loss = loss_fn(zs[name], s=3.0, num_pairs=num_pairs)
                loss.backward()
                optimizers[name].step()
                epoch_losses[name] = loss.item()
            except RuntimeError as e:
                print(f"Warning: {name} epoch {epoch + 1} error: {e}. Skipping update.")
                epoch_losses[name] = float('nan')

        # 量化每个方法（使用subsample）
        for name in methods:
            mean_min, mean, mean_cos, _ = compute_metrics(zs[name], subsample=subsample_metrics)
            results[name]['losses'].append(epoch_losses.get(name, float('nan')))
            results[name]['mean_min_dists'].append(mean_min)
            results[name]['mean_dists'].append(mean)
            results[name]['mean_cosines'].append(mean_cos)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}:")
            for name in methods:
                # 为打印重新计算full metrics
                full_mean_min, full_mean, full_mean_cos, _ = compute_metrics(zs[name], subsample=None)
                print(
                    f"  {name}: Loss = {results[name]['losses'][-1]:.4f}, "
                    f"Mean Min Dist = {full_mean_min:.4f}, "
                    f"Mean Dist = {full_mean:.4f}, "
                    f"Mean Cosine = {full_mean_cos:.4f}")

        # 减少可视化：只在1, 50, 100 epoch绘制（使用完整点 for z_3d）
        if epoch + 1 in [1, 50, 100]:
            fig = plt.figure(figsize=(12, 6))
            for i, name in enumerate(methods, 1):
                _, _, _, z_3d = compute_metrics(zs[name], return_metrics=False)  # no metrics, just z_3d on full
                ax = fig.add_subplot(1, 2, i, projection='3d')
                ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=0.5)
                ax.set_title(f'{name} (param=3) Epoch {epoch + 1}')
            plt.suptitle(f'Gaussian vs Riesz Comparison at Epoch {epoch + 1}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_epoch_{epoch + 1}_3d.png'))
            plt.close()

    # 每个方法的收敛曲线（使用sampled metrics）
    for name in methods:
        plt.figure(figsize=(8, 4))
        plt.plot(results[name]['losses'], label='Loss')
        plt.plot(results[name]['mean_min_dists'], label='Mean Min Dist')
        plt.plot(results[name]['mean_dists'], label='Mean Dist')
        plt.plot(results[name]['mean_cosines'], label='Mean Cosine')
        plt.legend()
        plt.title(f'{name} (param=3) Convergence')
        plt.savefig(os.path.join(output_dir, f'{name}_convergence.png'))
        plt.close()

    # 跨方法比较图
    plt.figure(figsize=(8, 4))
    for name in methods:
        plt.plot(results[name]['mean_min_dists'], label=f'{name} Mean Min Dist')
    plt.legend()
    plt.title('Mean Min Dist Comparison (param=3)')
    plt.savefig(os.path.join(output_dir, 'mean_min_dist_comparison.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    for name in methods:
        plt.plot(results[name]['mean_dists'], label=f'{name} Mean Dist')
    plt.legend()
    plt.title('Mean Dist Comparison (param=3)')
    plt.savefig(os.path.join(output_dir, 'mean_dist_comparison.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    for name in methods:
        plt.plot(results[name]['mean_cosines'], label=f'{name} Mean Cosine')
    plt.legend()
    plt.title('Mean Cosine Comparison (param=3)')
    plt.savefig(os.path.join(output_dir, 'mean_cosine_comparison.png'))
    plt.close()

    # 打印最终指标 (full)
    print("\nFinal Results:")
    for name in methods:
        final_mean_min, final_mean, final_mean_cos, _ = compute_metrics(zs[name], subsample=None)
        print(f"{name} (param=3): "
              f"Final Loss = {results[name]['losses'][-1]:.4f}, "
              f"Mean Min Dist = {final_mean_min:.4f}, "
              f"Mean Dist = {final_mean:.4f}, "
              f"Mean Cosine = {final_mean_cos:.4f}")

    print("\nComparison complete. All images saved in:", output_dir)


# 运行两个对比
if __name__ == "__main__":
    N = 2 ** 14
    dim = 8
    epochs = 100
    lr = 0.01
    num_pairs = 100000
    subsample_metrics = 4096  # 用于loop中的metrics采样

    # 第一个：完全高斯初始化 (clustered)
    print("=== Full Gaussian Initialization ===")
    full_gauss_init = torch.randn(N, dim) * 0.1  # 无device，稍后移
    compare_uniformity_methods(N=N, dim=dim, epochs=epochs, lr=lr, output_dir='full_gaussian',
                               num_pairs=num_pairs, subsample_metrics=subsample_metrics, initial_z=full_gauss_init)

    # 第二个：部分均匀 + 部分重合高斯
    print("\n=== Partial Uniform + Overlapping Gaussian Initialization ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uniform_half = torch.randn(N // 2, dim, device=device)
    uniform_half = F.normalize(uniform_half, p=2, dim=-1)
    # 部分重合：随机选择uniform_half中的点，加小高斯噪声，然后normalize
    indices = torch.randint(0, N // 2, (N // 2,), device=device)
    clustered_half = uniform_half[indices] + torch.randn(N // 2, dim, device=device) * 0.1
    clustered_half = F.normalize(clustered_half, p=2, dim=-1)
    partial_init = torch.cat([uniform_half, clustered_half], dim=0)

    compare_uniformity_methods(N=N, dim=dim, epochs=epochs, lr=lr, output_dir='partial_uniform',
                               num_pairs=num_pairs, subsample_metrics=subsample_metrics, initial_z=partial_init)
