import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.special import gamma  # For theoretical bounds
import torch.nn.functional as F


def load_codebook(checkpoint_path, l2_norm=True):
    # Load checkpoint with weights_only=False as specified
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['model']
    codebook = state_dict['quantize.embedding.weight']  # [16384, 8]
    if l2_norm:
        codebook = F.normalize(codebook, p=2, dim=-1)
    return codebook  # [K, d]


def generate_uniform_samples(K, d, device):
    # Generate K uniform points on S^{d-1} by normalizing Gaussian samples
    samples = torch.randn(K, d, device=device)
    samples = F.normalize(samples, p=2, dim=-1)
    return samples


def compute_min_nn_distance(points):
    # Compute min pairwise Euclidean distance (on unit sphere)
    dists = torch.cdist(points, points)  # [K, K]
    dists.fill_diagonal_(float('inf'))  # Ignore self-dist
    min_dist = dists.min()
    return min_dist.item()


def compute_hyperspherical_energy(points, alpha):
    # E = sum_{i<j} 1 / ||x_i - x_j||^alpha
    dists = torch.cdist(points, points)  # [K, K]
    dists = dists + torch.eye(len(points), device=points.device)  # Avoid zero div
    energy = (1.0 / dists.pow(alpha)).triu(diagonal=1).sum()
    return energy.item()


def compute_rayleigh_stat(points):
    # Rayleigh test: ||sum x_i / K||, close to 0 for uniform
    mean_vector = points.mean(dim=0)
    stat = torch.norm(mean_vector).item()
    return stat


def theoretical_min_dist_bound(K, d):
    # Approximate lower bound for min dist: ~ (Gamma((d+1)/2) / Gamma(d/2))^{1/(d-1)} * (pi K)^{-1/(2(d-1))}
    # Simplified from packing bounds
    vol_factor = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
    bound = (4 * vol_factor / K) ** (1 / d)
    return bound


def visualize_projection(points, uniform_samples, method='pca', dim=2, output_path='projection.png'):
    if method == 'pca':
        reducer = PCA(n_components=dim)
    elif method == 'tsne':
        reducer = TSNE(n_components=dim, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    points_np = points.cpu().numpy()
    uniform_np = uniform_samples.cpu().numpy()

    reduced_points = reducer.fit_transform(points_np)
    reduced_uniform = reducer.fit_transform(uniform_np)  # Separate fit for fair comparison

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    if dim == 2:
        axs[0].scatter(reduced_points[:, 0], reduced_points[:, 1], s=1)
        axs[1].scatter(reduced_uniform[:, 0], reduced_uniform[:, 1], s=1)
    else:  # dim=3
        axs[0] = fig.add_subplot(121, projection='3d')
        axs[1] = fig.add_subplot(122, projection='3d')
        axs[0].scatter(reduced_points[:, 0], reduced_points[:, 1], reduced_points[:, 2], s=1)
        axs[1].scatter(reduced_uniform[:, 0], reduced_uniform[:, 1], reduced_uniform[:, 2], s=1)

    axs[0].set_title('Codebook Projection')
    axs[1].set_title('Uniform Samples Projection')
    plt.savefig(output_path)
    plt.close()


def visualize_angle_histogram(points, d, output_path='angle_hist.png'):
    # Compute pairwise angles: acos(x_i · x_j)
    dots = torch.mm(points, points.t())  # [K, K]
    dots = torch.clamp(dots, -1.0, 1.0)
    angles = torch.acos(dots).flatten().cpu().numpy()
    angles = angles[angles > 0]  # Ignore zero/self

    # Theoretical PDF for uniform: p(theta) ∝ sin^{d-2}(theta)
    theta = np.linspace(0, np.pi, 100)
    pdf = np.sin(theta) ** (d - 2)
    pdf /= pdf.sum() * (theta[1] - theta[0])  # Normalize

    plt.figure(figsize=(8, 6))
    plt.hist(angles, bins=50, density=True, alpha=0.7, label='Empirical')
    plt.plot(theta, pdf, 'r-', label='Theoretical Uniform')
    plt.title('Pairwise Angle Distribution')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load codebook
    print("Loading codebook from checkpoint...")
    codebook = load_codebook(args.checkpoint_path, l2_norm=args.l2_norm).to(device)
    K, d = codebook.shape

    # Print codebook information
    print("=" * 60)
    print("CODEBOOK INFORMATION")
    print("=" * 60)
    print(f"Codebook size: {K} (number of embeddings)")
    print(f"Embedding dimension: {d}")
    print(f"Codebook shape: {codebook.shape}")
    print(f"L2 normalization applied: {args.l2_norm}")
    print("=" * 60)
    print()

    # Generate uniform reference
    print("Generating uniform reference samples...")
    uniform_samples = generate_uniform_samples(K, d, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Quantify
    print("Quantifying uniformity...")
    print("-" * 60)
    min_dist_code = compute_min_nn_distance(codebook)
    min_dist_uniform = compute_min_nn_distance(uniform_samples)
    theo_bound = theoretical_min_dist_bound(K, d)
    print(f"Min NN Distance (Codebook): {min_dist_code:.4f}")
    print(f"Min NN Distance (Uniform):  {min_dist_uniform:.4f}")
    print(f"Theoretical bound:          ~{theo_bound:.4f}")
    print()

    alpha = d - 2  # For energy
    energy_code = compute_hyperspherical_energy(codebook, alpha)
    energy_uniform = compute_hyperspherical_energy(uniform_samples, alpha)
    print(f"Hyperspherical Energy (Codebook): {energy_code:.2e}")
    print(f"Hyperspherical Energy (Uniform):  {energy_uniform:.2e}")
    print()

    rayleigh_code = compute_rayleigh_stat(codebook)
    rayleigh_uniform = compute_rayleigh_stat(uniform_samples)
    print(f"Rayleigh Stat (Codebook): {rayleigh_code:.4f}")
    print(f"Rayleigh Stat (Uniform):  {rayleigh_uniform:.4f}")
    print(f"(Closer to 0 indicates more uniform distribution)")
    print("-" * 60)
    print()

    # Visualize
    print("Generating visualizations...")
    visualize_projection(codebook, uniform_samples, method='pca', dim=2,
                         output_path=os.path.join(args.output_dir, 'pca_2d.png'))
    visualize_projection(codebook, uniform_samples, method='tsne', dim=2,
                         output_path=os.path.join(args.output_dir, 'tsne_2d.png'))
    visualize_projection(codebook, uniform_samples, method='pca', dim=3,
                         output_path=os.path.join(args.output_dir, 'pca_3d.png'))
    visualize_angle_histogram(codebook, d, output_path=os.path.join(args.output_dir, 'angle_hist.png'))
    print(f"✓ All visualizations saved to {args.output_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test codebook uniformity from checkpoint")
    parser.add_argument("--checkpoint_path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/tokenizer/tokenizer_image/results_tokenizer_image/075-VQ-16/checkpoints/0100000.pt')
    parser.add_argument("--output_dir", type=str, default="./uniformity_results", help="Directory to save results")
    parser.add_argument("--l2_norm", action='store_true', default=True,
                        help="Whether to normalize codebook (default True)")
    args = parser.parse_args()
    main(args)
