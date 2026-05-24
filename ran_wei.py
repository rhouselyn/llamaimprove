import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# 修复 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LinearProjectionChaos(nn.Module):
    """
    基于方案二：线性代数投影法
    利用 N=16 的向量控制两个固定的随机基底矩阵，生成混乱平面
    """

    def __init__(self, latent_dim=16, output_dim=256, seed=999):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # 使用确定的种子生成固定的随机基底矩阵 (Basis Matrices)
        # 我们将其注册为 buffer，这样它们会随模型移动但不会被视为训练参数
        torch.manual_seed(seed)

        # A 矩阵: [256, 16]
        basis_a = torch.randn(output_dim, latent_dim)
        # B 矩阵: [16, 256]
        basis_b = torch.randn(latent_dim, output_dim)

        self.register_buffer('basis_a', basis_a)
        self.register_buffer('basis_b', basis_b)

    def forward(self, z, alpha=100.0):
        """
        z: [1, 16] 的隐向量
        alpha: 混乱强度系数
        """
        # 1. 核心线性投影: A * diag(z) * B
        # res 维度: [256, 256]
        res = (self.basis_a * z) @ self.basis_b

        # 2. 引入非线性破碎 (Shattering)
        # 使用 torch.sin 产生高频波动，然后用 % 1.0 进行截断产生断层
        # 这里的 % 1.0 等同于原代码想实现的取模逻辑
        chaos_matrix = torch.sin(res * alpha) % 1.0

        return chaos_matrix

def normalize_instance(matrix):
    """标准归一化，增强对比度"""
    return (matrix - matrix.mean()) / (matrix.std() + 1e-8)


# --- 参数配置 ---
TARGET_DIM = 256
LATENT_DIM = 16
# ALPHA 越大，像素间的起伏越剧烈，越不像平滑的函数
ALPHA = 150.0
SAVE_PATH = 'linear_chaos_weight.png'

# 初始化模型
model = LinearProjectionChaos(latent_dim=LATENT_DIM, output_dim=TARGET_DIM)

# 生成隐向量 z (N=16)
# 改变这个向量，生成的整个平面会随之改变，但保持“混乱”的风格
random_z = torch.randn(1, LATENT_DIM)

with torch.no_grad():
    # 生成 256x256 平面
    weight_matrix = model(random_z, alpha=ALPHA)
    # 归一化处理
    weight_matrix = normalize_instance(weight_matrix)

# --- 绘图与保存 (沿用你的风格) ---
plt.figure(figsize=(8, 8))
# 使用 'magma' 展示高频破碎感，'icefire' 也是不错的选择
sns.heatmap(weight_matrix.cpu().numpy(), cmap='magma', cbar=False,
            xticklabels=False, yticklabels=False, square=True)

plt.title(f"Linear Projection Chaos (N={LATENT_DIM}, Alpha={ALPHA})")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"生成的平面维度: {weight_matrix.shape}")
print(f"输入控制向量: {random_z.numpy()}")
print(f"混沌参数已保存至: {os.path.abspath(SAVE_PATH)}")