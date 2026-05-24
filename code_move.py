import torch
import numpy as np
import matplotlib.pyplot as plt
from tokenizer.tokenizer_image.vq_model import VQ_models  # 假设你有这个模块，根据原代码路径导入
from sklearn.decomposition import PCA  # 用于PCA降维

# 定义参数（从原args中复制必要部分）
class Args:
    vq_model = "VQ-16"
    codebook_size = 16384
    codebook_embed_dim = 8
    commit_loss_beta = 0.25
    entropy_loss_ratio = 0.0
    dropout_p = 0.0
    codebook_l2_norm = True  # 假设默认True，根据你的args

args = Args()
torch.manual_seed(0)
# 步骤1: 创建初始VQ模型，获取初始codebook
initial_vq_model = VQ_models[args.vq_model](
    codebook_size=args.codebook_size,
    codebook_embed_dim=args.codebook_embed_dim,
    commit_loss_beta=args.commit_loss_beta,
    entropy_loss_ratio=args.entropy_loss_ratio,
    dropout_p=args.dropout_p,
)
initial_codebook = initial_vq_model.quantize.embedding.weight.detach().cpu().numpy()  # (16384, 8)

# 步骤2: 加载训练后的checkpoint，获取最终codebook
checkpoint_path = "/mnt/afs/zhengmingkai/raozf/llamagen/results_tokenizer_image/008-VQ-16/checkpoints/0400000.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
final_model_state = checkpoint["model"]  # 或 checkpoint["ema"] 如果使用EMA
final_vq_model = VQ_models[args.vq_model](
    codebook_size=args.codebook_size,
    codebook_embed_dim=args.codebook_embed_dim,
    commit_loss_beta=args.commit_loss_beta,
    entropy_loss_ratio=args.entropy_loss_ratio,
    dropout_p=args.dropout_p,
)
final_vq_model.load_state_dict(final_model_state)
final_codebook = final_vq_model.quantize.embedding.weight.detach().cpu().numpy()  # (16384, 8)

# 步骤3: 计算每个code的初始 vs 最终角度
norm_initial = np.linalg.norm(initial_codebook, axis=1)
norm_final = np.linalg.norm(final_codebook, axis=1)
dot_products = np.sum(initial_codebook * final_codebook, axis=1)
cos_sim = dot_products / (norm_initial * norm_final + 1e-8)  # 避免除零
angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)  # 转换为度，(16384,)

# 步骤4: 计算角度统计指标
avg_angle = np.mean(angles)
max_angle = np.max(angles)
min_angle = np.min(angles)

print(f"平均角度 (度): {avg_angle:.4f}")
print(f"最大角度 (度): {max_angle:.4f}")
print(f"最小角度 (度): {min_angle:.4f}")

# 步骤5: 绘制角度分布直方图
plt.figure(figsize=(8, 5))
plt.hist(angles, bins=50, color='green', alpha=0.7)
plt.title('Angle Distribution (Degrees) between Initial and Final Codes')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.savefig("angle_hist.png")
plt.show()  # 如果在交互环境，会显示；否则只保存文件
print("角度分布图形已保存为 angle_hist.png")

# 步骤6: 验证均匀分布（初始和最终codebook）
def check_uniformity(codebook, name):
    # 量化指标: 随机采样1000个向量，计算所有对的平均cos sim（期望接近0）
    np.random.seed(42)  # 固定种子
    sample_indices = np.random.choice(codebook.shape[0], min(1000, codebook.shape[0]), replace=False)
    sampled = codebook[sample_indices]
    norm_sampled = np.linalg.norm(sampled, axis=1, keepdims=True)
    cos_matrix = np.dot(sampled, sampled.T) / (norm_sampled * norm_sampled.T + 1e-8)
    np.fill_diagonal(cos_matrix, 0)  # 忽略自相似
    avg_cos_sim = np.mean(cos_matrix)
    print(f"{name} codebook 的平均余弦相似度 (采样): {avg_cos_sim:.4f} (越接近0越均匀)")

    # 图形: PCA降到2D并绘制散点图
    pca = PCA(n_components=2)
    projected = pca.fit_transform(codebook)
    plt.figure(figsize=(6, 6))
    plt.scatter(projected[:, 0], projected[:, 1], s=1, alpha=0.5)
    plt.title(f'{name} Codebook PCA Projection (2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(f"{name.lower()}_pca_scatter.png")
    plt.show()
    print(f"{name} codebook 的2D投影图形已保存为 {name.lower()}_pca_scatter.png")

check_uniformity(initial_codebook, "Initial")
check_uniformity(final_codebook, "Final")
