# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import binom

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models


#################################################################################
#                                  Main Function                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Always run in single GPU mode
    device_id = 0  # Use GPU 0; change if needed
    device = torch.device(f'cuda:{device_id}')
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device_id)
    print("Running on single GPU.")

    # Hardcode the VQ checkpoint path
    args.vq_ckpt = 'weights/vq_ds16_c2i.pt'

    # Setup output folder (for plot only)
    os.makedirs(args.code_path, exist_ok=True)

    # Create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup data
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        num_aug = 10
    else:
        crop_size = args.image_size
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        num_aug = 2  # Original + flip

    dataset = build_dataset(args, transform=transform)
    sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Compute binary dim (bits needed for indices: ceil(log2(codebook_size)))
    dim = math.ceil(math.log2(args.codebook_size))

    # Local counter for popcounts (0 to dim)
    popcount_counts = torch.zeros(dim + 1, dtype=torch.long, device=device)

    for x, y in loader:
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x_all)

        # Process for popcount stats
        indices_flat = indices.flatten().cpu().numpy()
        for idx in indices_flat:
            bin_str = bin(idx)[2:].zfill(dim)  # Fixed-length binary string
            popcount = bin_str.count('1')  # Number of 1s
            popcount_counts[popcount] += 1

    # Handle plotting
    popcount_counts = popcount_counts.cpu().numpy()
    total_patches = popcount_counts.sum()
    if total_patches > 0:
        observed_probs = popcount_counts / total_patches

        # Expected binomial probabilities (Bernoulli p=0.5)
        k_values = np.arange(dim + 1)
        expected_probs = binom.pmf(k_values, n=dim, p=0.5)

        # Plot
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(k_values - width / 2, observed_probs, width, label='Observed')
        ax.bar(k_values + width / 2, expected_probs, width, label=f'Expected (Binomial n={dim}, p=0.5)')
        ax.set_xlabel('Number of 1s in Binary Index (Popcount)')
        ax.set_ylabel('Probability')
        ax.set_title('Distribution of Popcounts vs. Binomial Expectation')
        ax.set_xticks(k_values)
        ax.legend()
        plot_path = os.path.join(args.code_path, f'{args.dataset}{args.image_size}_popcount_dist.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    else:
        print("No patches processed; skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='imagenet_train_filelist.txt')
    parser.add_argument("--code-path", type=str, default='./')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='weights/vq_ds16_c2i.pt',
                        help="ckpt path for vq model")  # This will be overridden
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using ten crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")

    args = parser.parse_args()
    main(args)
