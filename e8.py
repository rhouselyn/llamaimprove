import math
import itertools
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 从原代码复制的E8生成函数
def generate_e8_shell_norm8(dim=8):
    points = []
    sqrt_norm = math.sqrt(8)  # 用于后续归一化，但实际我们用norm计算

    # 1. 整数坐标类型：所有 xi 为整数，sum xi 偶，sum xi^2 = 8
    coord_range = range(-3, 4)
    for coords in itertools.product(coord_range, repeat=dim):
        coords = list(coords)
        if all(abs(x - round(x)) < 1e-6 for x in coords):  # 全整数（浮点安全）
            sum_coords = sum(coords)
            if sum_coords % 2 != 0: continue  # sum 须偶
            norm_sq = sum(x ** 2 for x in coords)
            if norm_sq == 8:
                points.append(coords)

    # 2. 半整数坐标类型：所有 xi = k + 0.5, k 整数，sum xi 偶整数，sum xi^2 = 8
    half_coord_range = [k + 0.5 for k in range(-3, 3)]  # -2.5, -1.5, -0.5, 0.5, 1.5, 2.5
    for coords in itertools.product(half_coord_range, repeat=dim):
        coords = list(coords)
        if all(abs(abs(x) - (math.floor(abs(x)) + 0.5)) < 1e-6 for x in coords):  # 全半整数
            sum_coords = sum(coords)
            if not math.isclose(sum_coords % 2, 0, abs_tol=1e-6): continue  # sum 偶整数
            norm_sq = sum(x ** 2 for x in coords)
            if math.isclose(norm_sq, 8, abs_tol=1e-6):
                points.append(coords)

    # 去重并转换为 tuple 以唯一化
    points = [tuple(round(x, 6) for x in p) for p in points]  # 浮点精度处理
    points = list(set(points))
    assert len(points) == 17520, f"Expected 17520 points, got {len(points)}"

    # 转换为 tensor 并归一化到单位球面
    points_tensor = torch.tensor(points, dtype=torch.float32)
    points_tensor = points_tensor / points_tensor.norm(p=2, dim=-1, keepdim=True)  # L2 归一化

    return points_tensor

# 生成随机单位向量（均匀分布在8维球面）
def generate_random_unit_vectors(num_points, dim=8):
    # 高斯随机，然后归一化（Marsaglia方法）
    rand_vectors = np.random.randn(num_points, dim)
    norms = np.linalg.norm(rand_vectors, axis=1, keepdims=True)
    rand_vectors = rand_vectors / norms
    return torch.tensor(rand_vectors, dtype=torch.float32)

# 计算点间余弦相似度分布（作为均匀性代理：均匀分布应有更低的平均sim，更宽的分布）
def compute_cosine_sim_distribution(points):
    points_norm = points / points.norm(p=2, dim=-1, keepdim=True)
    sim_matrix = torch.mm(points_norm, points_norm.t())
    # 排除自相似
    sim_matrix.fill_diagonal_(0)
    sim_flat = sim_matrix.flatten()
    return sim_flat[sim_flat != 0].cpu().numpy()  # 只取非零（但实际全非对角）

# 主测试脚本
if __name__ == "__main__":
    dim = 8
    print("生成 E8 shell 点（norm^2=8，归一化到单位球）...")
    e8_points = generate_e8_shell_norm8(dim=dim)
    num_points = len(e8_points)
    print(f"E8 点数: {num_points}")

    print("生成相同数量的随机单位向量（作为均匀分布对照）...")
    random_points = generate_random_unit_vectors(num_points, dim=dim)

    # 1. 计算余弦相似度分布并可视化（histogram）
    print("计算余弦相似度分布...")
    e8_sims = compute_cosine_sim_distribution(e8_points)
    random_sims = compute_cosine_sim_distribution(random_points)

    plt.figure(figsize=(12, 6))
    sns.histplot(e8_sims, bins=100, kde=True, color='blue', label='E8 Points')
    sns.histplot(random_sims, bins=100, kde=True, color='red', label='Random Uniform')
    plt.title('Cosine Similarity Distribution (Pairwise, excluding self)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('cosine_sim_distribution.png')
    plt.show()
    print("余弦相似度分布图保存为 'cosine_sim_distribution.png'")
    print("解释: 如果E8更均匀，分布应更宽、峰值更低（sim更接近0）；如果簇聚，sim更高。")

    # 2. 降维可视化：用PCA到3D
    print("进行 PCA 降维到 3D...")
    pca = PCA(n_components=3)
    e8_pca = pca.fit_transform(e8_points.cpu().numpy())
    random_pca = pca.transform(random_points.cpu().numpy())  # 用相同PCA

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(e8_pca[:, 0], e8_pca[:, 1], e8_pca[:, 2], s=1, c='blue', alpha=0.5)
    ax1.set_title('E8 Points (PCA 3D)')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(random_pca[:, 0], random_pca[:, 1], random_pca[:, 2], s=1, c='red', alpha=0.5)
    ax2.set_title('Random Uniform (PCA 3D)')
    plt.savefig('pca_3d.png')
    plt.show()
    print("PCA 3D 图保存为 'pca_3d.png'")

    # 3. 降维可视化：用t-SNE到2D（更适合非线性结构）
    print("进行 t-SNE 降维到 2D（可能较慢）...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    e8_tsne = tsne.fit_transform(e8_points.cpu().numpy())
    random_tsne = tsne.fit_transform(random_points.cpu().numpy())  # 独立fit，因为分布不同

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(e8_tsne[:, 0], e8_tsne[:, 1], s=1, c='blue', alpha=0.5)
    plt.title('E8 Points (t-SNE 2D)')
    plt.subplot(122)
    plt.scatter(random_tsne[:, 0], random_tsne[:, 1], s=1, c='red', alpha=0.5)
    plt.title('Random Uniform (t-SNE 2D)')
    plt.savefig('tsne_2d.png')
    plt.show()
    print("t-SNE 2D 图保存为 'tsne_2d.png'")
    print("解释: 在降维图中，如果E8点显示出网格/簇聚结构，而随机点更均匀填充空间，则E8不是完全均匀的。E8是格子，所以预期有结构化分布，但作为codebook初始化，它在高维是'quasi-uniform'（准均匀），比随机更好用于量化。")

    # 额外统计
    print("\n额外统计:")
    print(f"E8 平均余弦sim: {np.mean(e8_sims):.4f}")
    print(f"Random 平均余弦sim: {np.mean(random_sims):.4f}")
    print(f"E8 sim 标准差: {np.std(e8_sims):.4f}")
    print(f"Random sim 标准差: {np.std(random_sims):.4f}")
    print("如果E8的平均sim更低/std更高，表明更均匀（点更分散）。但在高维，随机均匀的平均sim接近0。")
