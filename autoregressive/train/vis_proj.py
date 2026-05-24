import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import argparse
import os
import sys
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest  # For Gaussian fitting and KS test
from collections import Counter  # Added for codebook usage
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models
from torch.utils.data import DataLoader
from torchvision import transforms

def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."
    device = 'cuda'
    torch.cuda.set_device(0)

    # Create and load VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    # Handle compiled model if applicable
    if 'model' in checkpoint:
        vq_model.load_state_dict(checkpoint["model"])
    else:
        vq_model.load_state_dict(checkpoint)
    del checkpoint

    # Setup data: simple center crop, no augmentation/ten_crop
    crop_size = args.image_size
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # Adjustable for memory
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Collect sampled z_e vectors
    all_vectors = []
    # Added: Counter for codebook usage (binary sign-based discretization)
    usage_counter = Counter()
    codebook_size = 1 << args.codebook_embed_dim  # 2^embed_dim (e.g., 256 for dim=8)
    powers = 2 ** np.arange(args.codebook_embed_dim)  # Precompute powers of 2
    print("Collecting vectors and codebook usage...")

    for x, _ in loader:  # Ignore labels
        x = x.to(device)
        with torch.no_grad():
            # Directly get z_e from encoder (shape: batch, embed_dim, h/8, w/8)
            z_e = vq_model.encoder(x)
            print(f"z_e shape from encoder: {z_e.shape}")  # Debug: should be (batch, 8, 32, 32) for 256x256 image
            # Reshape to (batch, num_tokens, embed_dim)
            z_e = z_e.permute(0, 2, 3, 1).reshape(z_e.size(0), -1, z_e.size(1))
        z_e = z_e.detach().cpu().numpy()  # (batch, num_tokens, 8)
        z_e_flat = z_e.reshape(-1, args.codebook_embed_dim)  # (batch*num_tokens, 8)

        # Added: Always update codebook usage from full batch (discretize by sign to binary, convert to decimal)
        z_binary = (z_e_flat > 0).astype(int)  # (num_tokens, embed_dim), 0 or 1
        codes = np.dot(z_binary, powers).astype(int)  # Vectorized: each row dot powers -> decimal code
        usage_counter.update(codes)

        # Sample if too many (for visualization only)
        if len(all_vectors) < args.sample_size:
            num_to_sample = min(args.sample_size - len(all_vectors), len(z_e_flat))
            if num_to_sample > 0:
                sample_idx = random.sample(range(len(z_e_flat)), num_to_sample)
                sampled = z_e_flat[sample_idx]
                all_vectors.extend(sampled.tolist())
        if len(all_vectors) >= args.sample_size:
            break  # Stop early if enough samples

    all_vectors = np.array(all_vectors)  # (sample_size, 8)
    print(f"Collected {len(all_vectors)} vectors.")

    # Added: Print unique codes and save codebook usage to txt (decimal: count)
    print(f"Unique codes mapped: {len(usage_counter)} / {codebook_size}")
    if len(usage_counter) == codebook_size:
        print("All possible binary combinations are mapped!")
    else:
        print("Not all possible binary combinations are mapped.")
    usage_file = os.path.join(args.output_path, "codebook_usage.txt")
    with open(usage_file, "w") as f:
        for i in range(codebook_size):
            f.write(f"{i}: {usage_counter[i]}\n")
    print(f"Saved codebook usage to {usage_file}")

    # Visualize: 4 scatter plots for paired dims in one figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    pairs = [(0,1), (2,3), (4,5), (6,7)]
    for i, (dim1, dim2) in enumerate(pairs):
        ax = axs[i//2, i%2]
        ax.scatter(all_vectors[:, dim1], all_vectors[:, dim2], alpha=0.5, s=1)
        ax.set_xlabel(f"Dim {dim1}")
        ax.set_ylabel(f"Dim {dim2}")
        ax.set_title(f"Scatter of Dims {dim1}-{dim2}")
    plt.tight_layout()
    output_file = os.path.join(args.output_path, "vector_distribution.png")
    plt.savefig(output_file)
    print(f"Saved distribution plot to {output_file}")

    # New: Analyze each dimension's closeness to Gaussian distribution
    print("\nAnalyzing each dimension's distribution...")
    fig_gauss, axs_gauss = plt.subplots(2, 4, figsize=(20, 10))
    axs_gauss = axs_gauss.flatten()
    for dim in range(args.codebook_embed_dim):
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
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='/mnt/afs/zhengmingkai/raozf/llamagen/results_tokenizer_image/008-VQ-16/checkpoints/0400000.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of vectors to sample for visualization")
    parser.add_argument("--output-path", type=str, default='./', help="Path to save the plot")
    parser.add_argument("--aoss-bucket", type=str, default="imagenet",
                        help="AOSS bucket name (only for aoss_imagenet dataset)")
    args = parser.parse_args()
    main(args)
