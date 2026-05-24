import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
DIM = 8
LEVELS = 5
BATCH_SIZE = 4096  # 每次优化的采样数
STEPS = 100000  # 训练步数
LR = 0.01  # 学习率


# ================= 定义优化模型 =================
class OptimizedFSQ(nn.Module):
    def __init__(self, dim, levels):
        super().__init__()
        self.dim = dim
        self.levels = levels

        # 初始化：使用标准高斯分位点作为起点，这已经是一个很好的起点了
        # 我们让每一维略微有一点随机扰动，打破完全对称，防止 [1,1] 和 [2,2] 重叠
        with torch.no_grad():
            # 基础高斯分位点
            probs = (torch.arange(levels) * 2 + 1) / (2 * levels)
            base = math_erfinv(2 * probs - 1) * np.sqrt(2)

            # 复制到每个维度
            init_val = base.unsqueeze(0).repeat(dim, 1)

            # 加入微小噪声，打破维度的对称性
            init_val += torch.randn_like(init_val) * 0.05

        self.values = nn.Parameter(init_val)

    def get_sorted_values(self):
        # 确保输出时是排序的（FSQ量化需要有序）
        v, _ = torch.sort(self.values, dim=1)
        return v

    def sample_vectors(self, n_samples):
        # 1. 随机生成索引 (N, Dim)
        indices = torch.randint(0, self.levels, (n_samples, self.dim), device=self.values.device)

        # 2. 查表获取数值
        # values: (Dim, Levels)
        # 我们需要根据 indices 获取对应的 value
        # 扩展 values 到 (1, Dim, Levels) -> (N, Dim, Levels)
        v_sorted = self.get_sorted_values()
        v_exp = v_sorted.unsqueeze(0).expand(n_samples, -1, -1)
        ind_exp = indices.unsqueeze(-1)

        # Gather: (N, Dim)
        vecs = torch.gather(v_exp, 2, ind_exp).squeeze(-1)
        return vecs


# 辅助数学函数
def math_erfinv(x):
    return torch.erfinv(x.clamp(-0.999, 0.999))


# ================= 损失函数 =================
def uniformity_loss_func(x, t=2):
    """
    RBF Kernel Uniformity Loss (同 Contrastive Loss)
    计算球面上点对的斥力。
    """
    x = F.normalize(x, p=2, dim=-1)
    # 计算成对距离的平方 (a-b)^2 = a^2 + b^2 - 2ab = 2 - 2(a.b)
    # 也可以直接算 Cosine Sim
    cov = torch.mm(x, x.t())  # (B, B)
    # 对角线是自己跟自己，距离为0，相似度为1，要去掉
    cov_no_diag = cov[~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)].view(cov.shape[0], -1)

    # 最小化 log(mean(exp(sim * t))) -> 也就是让相似度越小越好
    loss = (cov_no_diag * t).exp().mean().log()
    return loss


def gaussian_structure_loss(values):
    """
    保持每维度的数值大体上是高斯分布的尺度 (Mean~0, Std~1)
    不要让数值跑到 100 去，也不要缩成 0.001
    """
    mu = values.mean(dim=1)
    std = values.std(dim=1)
    loss = mu.pow(2).mean() + (std - 1.0).pow(2).mean()
    return loss


# ================= 主程序 =================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OptimizedFSQ(DIM, LEVELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始优化 FSQ Levels: {DIM}维 x {LEVELS}值...")

    history = []

    for step in range(STEPS):
        optimizer.zero_grad()

        # 1. 采样向量
        vecs = model.sample_vectors(BATCH_SIZE)

        # 2. 计算 Loss
        # 主要目标：归一化后的均匀性
        u_loss = uniformity_loss_func(vecs)

        # 次要目标：原始数值的合理性 (保持高斯尺度)
        # 获取排序后的值来计算统计量
        current_vals = model.get_sorted_values()
        g_loss = gaussian_structure_loss(current_vals)

        # 防止数值过于靠近0 (会导致归一化不稳定)
        # 惩罚绝对值小于 0.05 的数值
        zero_penalty = (0.05 - current_vals.abs()).relu().mean() * 10.0

        total_loss = u_loss + 0.1 * g_loss + zero_penalty

        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: Total={total_loss.item():.4f} | Unif={u_loss.item():.4f} | Gauss={g_loss.item():.4f}")
            history.append(u_loss.item())

    print("优化完成！")

    # ================= 结果展示与深度评估 =================
    final_vals = model.get_sorted_values().detach().cpu()

    print("\n" + "=" * 50)
    print(" 最终生成的 Levels (可直接复制到 ModelArgs)")
    print("=" * 50)
    print("fsq_levels = [")
    for i in range(DIM):
        vals_str = ", ".join([f"{v:.4f}" for v in final_vals[i]])
        print(f"    [{vals_str}],  # Dim {i}")
    print("]")
    print("=" * 50)

    # ================= 深度测试 =================
    print("\n正在进行最终质量评估...")

    # 为了准确评估 Min Distance，我们不能只随机采一点点
    # 对于 5^8 = 390,625 这种规模，我们可以尝试遍历采样更多，或者直接大批量采样
    # 这里采样 50,000 个向量来做成对距离计算（50000^2 矩阵太大，分块计算或用 KNN）

    N_TEST = 20000
    test_vecs = model.sample_vectors(N_TEST).detach().cpu().float()  # (N, D)
    test_vecs_norm = F.normalize(test_vecs, p=2, dim=-1)  # 投影到球面

    print(f"采样了 {N_TEST} 个向量进行统计...")

    # 1. 计算 Cosine Similarity
    # 为了避免 OOM，我们只计算一部分的最近邻
    # 使用 torch.cdist 计算欧氏距离，然后转为角距离
    # 球面欧氏距离 d = sqrt(2 - 2cos(theta))
    # 所以 min_d 越大越好

    # 计算所有点对距离的一小部分（例如前 2000 个点相对于所有点的最近距离）
    # 这能很好的估计全局的拥挤程度
    N_QUERY = 2000
    queries = test_vecs_norm[:N_QUERY]
    database = test_vecs_norm

    # 计算点积
    sim_matrix = torch.mm(queries, database.t())  # (2000, 20000)

    # 把自己跟自己的距离(1.0) 设为 -1，方便找 max 相似度（即除了自己以外的最近邻）
    mask = torch.arange(N_QUERY).unsqueeze(1) == torch.arange(N_TEST).unsqueeze(0)
    sim_matrix.masked_fill_(mask, -1.0)

    # 找到每个点的最近邻（相似度最大）
    max_sim, _ = sim_matrix.max(dim=1)  # (N_QUERY,)

    # 转换为角度 (弧度 和 角度)
    # cos(theta) = sim => theta = acos(sim)
    # 相似度越接近 1，角度越接近 0。我们要角度尽可能大。
    min_angles_rad = torch.acos(max_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    min_angles_deg = torch.rad2deg(min_angles_rad)

    avg_min_angle = min_angles_deg.mean().item()
    worst_min_angle = min_angles_deg.min().item()

    print(f"\n[Uniformity Metric - Separation]")
    print(f"平均最近邻角度 (Avg Min Angle): {avg_min_angle:.4f}° (越大越好)")
    print(f"最差最近邻角度 (Worst Min Angle): {worst_min_angle:.4f}° (越大越好)")

    # 2. 统计各向同性 (Isotropy) - 检查协方差矩阵特征值
    # 理想情况下，特征值应该全部相等 (球形)
    cov = torch.cov(test_vecs_norm.T)
    eigvals = torch.linalg.eigvalsh(cov)
    eig_ratio = eigvals.max() / eigvals.min()
    print(f"\n[Uniformity Metric - Isotropy]")
    print(f"PCA 特征值比率 (Max/Min Eigval): {eig_ratio:.4f} (越接近 1.0 越圆)")

    # 3. 计算 Min Euclidean Distance
    # d^2 = 2 - 2*cos(theta)
    min_dist_sq = 2.0 - 2.0 * max_sim
    min_dist = min_dist_sq.clamp(min=0).sqrt()
    print(f"\n[Distance Metric]")
    print(f"平均最小欧氏距离 (Avg Min Dist): {min_dist.mean().item():.4f}")

    # ================= 绘图 =================
    plt.figure(figsize=(15, 5))

    # 1. Scalar Levels
    plt.subplot(1, 3, 1)
    for i in range(DIM):
        plt.plot(final_vals[i].numpy(), marker='o', markersize=3, label=f'D{i}')
    plt.title("Optimized Levels")
    plt.grid(alpha=0.3)

    # 2. 最近邻角度分布
    plt.subplot(1, 3, 2)
    plt.hist(min_angles_deg.numpy(), bins=50, color='green', alpha=0.7)
    plt.title(f"Min Angular Dist (Avg={avg_min_angle:.1f}°)")
    plt.xlabel("Degrees")

    # 3. 总体相似度分布 (随机对)
    # 随机取一些对计算全局分布
    rand_idx = torch.randperm(N_TEST)[:5000]
    rand_vecs = test_vecs_norm[rand_idx]
    sim_rand = torch.mm(rand_vecs, rand_vecs.t())
    tri_idx = torch.triu_indices(5000, 5000, 1)
    # 随机抽样 50000 个点对
    perm_tri = torch.randperm(tri_idx.shape[1])[:50000]
    flat_sims = sim_rand[tri_idx[0, perm_tri], tri_idx[1, perm_tri]].numpy()

    plt.subplot(1, 3, 3)
    plt.hist(flat_sims, bins=100, color='blue', alpha=0.7, density=True)
    plt.title("Global Cosine Similarity")
    plt.xlabel("Cos Sim (Ideal ~0)")

    plt.tight_layout()
    plt.savefig("fsq_optimization_result.png")
    print("\n结果图已保存为 fsq_optimization_result.png")


if __name__ == "__main__":
    main()