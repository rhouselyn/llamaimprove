import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import random

# 从训练脚本导入必要的部分
from tokenizer.tokenizer_image.vq2_model import VQ_models
from dataset.build import build_dataset


def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置cudnn为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def denormalize(tensor):
    """将归一化的tensor转换回[0, 1]范围"""
    return tensor * 0.5 + 0.5


def save_comparison_grid(original, reconstructed, save_path, n_images=8):
    """保存原图和重建图的对比网格"""
    # 限制显示数量
    original = original[:n_images]
    reconstructed = reconstructed[:n_images]

    # 反归一化
    original = denormalize(original)
    reconstructed = denormalize(reconstructed)

    # 创建对比图：交替显示原图和重建图
    comparison = []
    for orig, recon in zip(original, reconstructed):
        comparison.extend([orig, recon])

    # 创建网格 (2行，每行n_images张图片)
    grid = make_grid(comparison, nrow=2, padding=2, normalize=False)

    # 转换为numpy用于显示
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # 保存图片
    plt.figure(figsize=(20, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('Original (left) vs Reconstructed (right)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid to {save_path}")


def save_individual_images(original, reconstructed, save_dir, n_images=8):
    """保存单独的原图和重建图"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'reconstructed'), exist_ok=True)

    # 限制数量
    original = original[:n_images]
    reconstructed = reconstructed[:n_images]

    # 反归一化
    original = denormalize(original)
    reconstructed = denormalize(reconstructed)

    # 保存每张图片
    for idx, (orig, recon) in enumerate(zip(original, reconstructed)):
        # 转换为PIL Image
        orig_img = transforms.ToPILImage()(orig.cpu())
        recon_img = transforms.ToPILImage()(recon.cpu())

        # 保存
        orig_img.save(os.path.join(save_dir, 'original', f'{idx:04d}.png'))
        recon_img.save(os.path.join(save_dir, 'reconstructed', f'{idx:04d}.png'))

    print(f"Saved {n_images} individual images to {save_dir}")


def main(args):
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading VQ model...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    ).to(device)

    # 加载checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if args.use_ema and "ema" in checkpoint:
        vq_model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights.")
    else:
        vq_model.load_state_dict(checkpoint["model"])
        print("Loaded model weights.")
    vq_model.eval()

    # 准备数据集
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)

    # 随机选择图片（使用固定种子确保可复现）
    print(f"Randomly selecting {args.n_samples} images with seed {args.seed}...")
    indices = np.random.choice(len(dataset), size=args.n_samples, replace=False)
    print(f"Selected indices: {indices.tolist()}")

    original_images = []
    reconstructed_images = []

    # 生成重建图片
    print("Generating reconstructions...")
    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)  # 添加batch维度

            # 重建
            recon, _ = vq_model(img)

            original_images.append(img.squeeze(0))
            reconstructed_images.append(recon.squeeze(0))

    # 转换为tensor
    original_tensor = torch.stack(original_images)
    reconstructed_tensor = torch.stack(reconstructed_images)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存对比网格图
    grid_path = os.path.join(args.output_dir, f'reconstruction_comparison_seed{args.seed}.png')
    save_comparison_grid(original_tensor, reconstructed_tensor, grid_path, n_images=args.n_samples)

    # 如果需要，保存单独的图片
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, f'individual_seed{args.seed}')
        save_individual_images(original_tensor, reconstructed_tensor,
                               individual_dir,
                               n_images=args.n_samples)

    # 计算并打印重建误差
    mse = torch.mean((original_tensor - reconstructed_tensor) ** 2).item()
    print(f"\nMean Squared Error: {mse:.6f}")

    # 保存选择的索引
    indices_path = os.path.join(args.output_dir, f'selected_indices_seed{args.seed}.txt')
    with open(indices_path, 'w') as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Selected indices: {indices.tolist()}\n")
        f.write(f"MSE: {mse:.6f}\n")

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize VQ-VAE reconstructions")

    # 模型参数
    parser.add_argument("--ckpt-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/tokenizer/tokenizer_image/results_tokenizer_image/075-VQ-16/checkpoints/0200000.pt',
                        help="Path to the trained VQ checkpoint")
    parser.add_argument("--use-ema", action='store_true',
                        help="Use EMA weights if available")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()),
                        default="VQ-16", help="VQ model architecture")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--commit-loss-beta", type=float, default=0.25)
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)

    # 数据集参数
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_val_filelist.txt',
                        help="Path to dataset")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)

    # 生成参数
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="./visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--save-individual", action='store_true',
                        help="Save individual images in addition to grid")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)
