import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import time
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import re

# BSQ class from the first script (simplified, without MultiScaleBSQ)
class BSQ(nn.Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.,
        frac_per_sample_entropy=1.,
        has_projections=None,
        projection_has_bias=True,
        soft_clamp_input_value=None,
        cosine_sim_project_in=False,
        cosine_sim_project_in_scale=None,
        channel_first=None,
        experimental_softplus_entropy_loss=False,
        entropy_loss_offset=5.,
        spherical=True,
        force_quantization_f32=True,
        inv_temperature=100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        preserve_norm=False,
        new_quant=False,
        mask_out=False,
        use_out_phi=False,
        use_out_phi_res=False,
    ):
        super().__init__()

        from math import log2
        import torch.distributed as dist

        def exists(v):
            return v is not None

        def default(*args):
            for arg in args:
                if exists(arg):
                    return arg() if callable(arg) else arg
            return None

        def l2norm(t):
            return F.normalize(t, dim=-1)

        class CosineSimLinear(nn.Module):
            def __init__(self, dim_in, dim_out, scale=1.):
                super().__init__()
                self.scale = scale
                self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

            def forward(self, x):
                x = F.normalize(x, dim=-1)
                w = F.normalize(self.weight, dim=0)
                return (x @ w) * self.scale

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = lambda dim_in, dim_out: CosineSimLinear(dim_in, dim_out, scale=cosine_sim_project_in)
        else:
            project_in_klass = lambda dim_in, dim_out: nn.Linear(dim_in, dim_out, bias=projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias=projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True)

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.channel_first = channel_first

        self.activation = straight_through_activation

        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.preserve_norm = preserve_norm
        self.new_quant = new_quant
        self.mask_out = mask_out

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        self.codebook_scale = codebook_scale

        self.commitment_loss_weight = commitment_loss_weight

        self.soft_clamp_input_value = soft_clamp_input_value

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    def indices_to_codes(self, indices, label_type='int_label', project_out=True):
        import torch

        def rearrange(t, einops_pattern, **kwargs):
            from einops import rearrange as einops_rearrange
            return einops_rearrange(t, einops_pattern, **kwargs)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = self.channel_first if self.channel_first is not None else is_img_or_video

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        if label_type == 'int_label':
            bits = ((indices[..., None].int() & self.mask) != 0).float()
        else:
            bits = indices

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes)

        codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize_new(self, z):
        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = torch.where(z > 0, 1, -1) * q_scale
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)
        per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        avg_prob = prob.mean(dim=list(range(prob.ndim - 1)))
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(self, x, return_loss_breakdown=False, mask=None, entropy_weight=0.1):
        import torch.distributed.nn as dist_nn

        def maybe_distributed_mean(t):
            if not dist.is_initialized() or dist.get_world_size() == 1:
                return t
            dist_nn.all_reduce(t)
            t = t / dist.get_world_size()
            return t

        is_img_or_video = x.ndim >= 4
        should_transpose = self.channel_first if self.channel_first is not None else is_img_or_video

        if should_transpose:
            x = x.permute(0, 2, 3, 1) if x.ndim == 4 else x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.project_in(x)

        x = x.reshape(x.shape[0], x.shape[1], self.num_codebooks, -1)

        x = l2norm(x)

        force_f32 = self.force_quantization_f32
        with torch.cuda.amp.autocast(enabled=not force_f32):
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            if self.new_quant:
                quantized = self.quantize_new(x)

            bit_indices = (quantized > 0).int()
            entropy_penalty = persample_entropy = cb_entropy = self.zero
            commit_loss = self.zero

            if force_f32:
                x = x.type(orig_dtype)

        x = quantized
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.project_out(x)

        if should_transpose:
            x = x.reshape_as(input=x)  # Simplified, adjust as needed

        if not self.keep_num_codebooks_dim:
            bit_indices = bit_indices.squeeze(-2)

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature) * entropy_weight

        ret = (x, None, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, (persample_entropy, cb_entropy, commit_loss)


def get_experiment_dir(base_dir, name="bsq_analysis"):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_num = -1
    pattern = re.compile(rf"^{re.escape(name)}(_(\d+))?$")
    for dir_name in existing_dirs:
        match = pattern.match(dir_name)
        if match:
            num = 0 if match.group(2) is None else int(match.group(2))
            max_num = max(max_num, num)
    new_dir = os.path.join(base_dir, name if max_num == -1 else f"{name}_{max_num + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def analyze_distribution(bsq, data_loader, device, n_batches=10, phase="analysis", save_data=None):
    bsq.eval()

    all_positive_counts = []
    all_patch_outputs = []
    all_codes = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            images = images.to(device)

            # Assume images are [B, C, H, W], flatten to patches if needed, but for BSQ, assume input is flattened latents
            # Here, simplify: assume data_loader provides latents or adjust transform to output suitable input for BSQ
            # For demo, we use images as is, but in practice, pass through encoder to get latents
            # Placeholder: flatten images to [B*H*W/16/16, dim] assuming dim matches BSQ input
            # Note: In real use, replace with actual latent extraction
            latents = images.view(images.size(0), -1, bsq.dim)  # Adjust based on actual input

            quantized, _, bit_indices, _ = bsq(latents)

            positive_counts = (bit_indices > 0).sum(dim=-1).cpu().numpy()
            all_positive_counts.extend(positive_counts)

            all_patch_outputs.append(quantized.cpu().numpy())
            all_codes.append(bit_indices.cpu().numpy())

    all_positive_counts = np.array(all_positive_counts)
    all_patch_outputs = np.concatenate(all_patch_outputs, axis=0).flatten()
    all_codes = np.concatenate(all_codes, axis=0).flatten()

    print(f"\n{'=' * 80}")
    print(f"BSQ Distribution Analysis - {phase.upper()}")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(all_positive_counts):,}")
    print(f"\nPositive Bit Counts:")
    print(f"  Mean: {all_positive_counts.mean():.4f} (target: {bsq.codebook_dim / 2:.1f})")
    print(f"  Std:  {all_positive_counts.std():.4f}")
    print(f"  Min:  {all_positive_counts.min()}")
    print(f"  Max:  {all_positive_counts.max()}")

    p_positive = (all_codes > 0).mean()
    print(f"\nCode Statistics:")
    print(f"  P(Positive): {p_positive:.6f} (target: 0.5)")
    print(f"  Mean: {all_patch_outputs.mean():.6f} (target: 0.0)")
    print(f"  Std:  {all_patch_outputs.std():.6f} (target: 1.0 / sqrt(dim))")

    if save_data is not None:
        save_data[phase] = {
            'positive_counts': all_positive_counts,
            'patch_outputs': all_patch_outputs,
            'codes': all_codes,
            'codebook_dim': bsq.codebook_dim
        }

    return all_positive_counts, all_patch_outputs, all_codes


def plot_comparison(before_data, after_data, save_path, epoch_info=""):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    phases = ['before', 'after']
    for i, phase in enumerate(phases):
        data = before_data if phase == 'before' else after_data
        axs[0, i].hist(data['positive_counts'], bins=50)
        axs[0, i].set_title(f'Positive Counts - {phase}')
        axs[1, i].hist(data['codes'], bins=3)
        axs[1, i].set_title(f'Code Distribution - {phase}')
    plt.savefig(save_path)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = get_experiment_dir(args.results_dir)

    # Load VAE checkpoint (assuming it contains BSQ as quantize)
    print(f"Loading VAE from {args.vae_ckpt}")
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')

    # Extract BSQ (assume vae_ckpt['model'] has 'quantize' as BSQ, adjust keys as per actual ckpt)
    bsq_state = {k.replace('quantize.', ''): v for k, v in vae_ckpt.get('model', vae_ckpt).items() if 'quantize' in k}
    bsq = BSQ(dim=64, codebook_size=2**18)  # Adjust params based on weights/infinity_vae_d64.pth
    bsq.load_state_dict(bsq_state, strict=False)
    bsq.to(device)
    bsq.eval()

    # Data setup (from train_encoder.py)
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    from dataset.build import build_dataset  # Assume available or implement
    dataset = build_dataset(args, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Analysis
    analysis_data = {}
    analyze_distribution(bsq, loader, device, n_batches=10, phase="bsq", save_data=analysis_data)

    # Save plot (dummy before/after for demo)
    plot_path = os.path.join(experiment_dir, "bsq_distribution.png")
    plot_comparison(analysis_data['bsq'], analysis_data['bsq'], plot_path)  # Use same for before/after

    print(f"Analysis completed. Results in: {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, default="weights/infinity_vae_d64.pth")
    parser.add_argument("--data-path", type=str, default="imagenet_train_filelist.txt")
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--results-dir", type=str, default="results_bsq")
    args = parser.parse_args()
    main(args)
