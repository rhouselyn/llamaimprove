import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.utils as vutils

import os
import time
import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 修复导入和加速设置 ---
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- 2. 导入自定义工具库 ---
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset


#################################################################################
#                           RoPE & Positional Utils                             #
#################################################################################

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", 1.0 / (max_freq ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))))

    def forward(self, x, pos):
        # x: [B, H, Seq, D], pos: [B, Seq, 2]
        pos_y, pos_x = pos[..., 0], pos[..., 1]
        freqs_y = torch.einsum('bi,j->bij', pos_y, self.inv_freq)
        freqs_x = torch.einsum('bi,j->bij', pos_x, self.inv_freq)
        return torch.cat([freqs_y, freqs_x], dim=-1)


def apply_rotary_pos_emb(t, freqs):
    # t: [B, H, L, D], freqs: [B, L, D/2]
    freqs = freqs.unsqueeze(1)  # Broadcast head
    t_left, t_right = t.chunk(2, dim=-1)
    freqs_y, freqs_x = freqs.chunk(2, dim=-1)

    def rotate_half(x, theta):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos, sin = theta.cos(), theta.sin()
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

    return torch.cat([rotate_half(t_left, freqs_y), rotate_half(t_right, freqs_x)], dim=-1)


class RoPECrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x_q, x_kv, coords_q, coords_k):
        # x_q: [B, 1, D], x_kv: [B, N, D]
        B, Lq, _ = x_q.shape
        _, Lk, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE scaling factor to make rotation sensitive to image grid
        rope_scale = 32.0
        freqs_q = self.rope(q, coords_q * rope_scale)
        freqs_k = self.rope(k, coords_k * rope_scale)

        q = apply_rotary_pos_emb(q, freqs_q)
        k = apply_rotary_pos_emb(k, freqs_k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.out_proj(out)


#################################################################################
#                              Core Model Components                            #
#################################################################################

class Encoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Simple Pyramid to get a good feature map
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, feature_dim, 3, 1, 1), nn.BatchNorm2d(feature_dim), nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)  # Output: 32x32 if input 128
        )

    def forward(self, x):
        return self.net(x)


class SaccadePredictor(nn.Module):
    """
    Mental Saccade: Given current thought (z_t) and movement (delta),
    predict next thought (z_{t+1}).
    """

    def __init__(self, feature_dim=256):
        super().__init__()
        self.delta_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # Concat [z, delta_emb]
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim)  # Predict next z
        )

    def forward(self, z_current, delta_coords):
        # delta_coords: [B, 2] (value range approx -1.0 to 1.0)
        d_emb = self.delta_encoder(delta_coords)
        combined = torch.cat([z_current, d_emb], dim=-1)
        z_pred = self.predictor(combined)
        return z_pred


class FeatureDiscriminator(nn.Module):
    """
    GAN Discriminator: Distinguish between Real Feature vs Predicted Feature.
    Working in latent space.
    """

    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # Output logic (Real=1, Fake=0)
        )

    def forward(self, z):
        return self.net(z)


class FoveatedDecoder(nn.Module):
    """
    Reconstructs image from a single vector z, guided by a center coordinate.
    Uses coordinate injection to allow spatially varying reconstruction quality.
    """

    def __init__(self, feature_dim=256, output_res=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_res = output_res

        # Learnable base canvas (small resolution)
        self.init_res = 16
        self.base_canvas = nn.Parameter(torch.randn(1, feature_dim // 2, self.init_res, self.init_res))

        # Injection MLP: z -> modulation parameters
        self.style_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2)
        )

        # Main decoder body
        self.up_blocks = nn.ModuleList([
            # 16 -> 32
            nn.Sequential(nn.ConvTranspose2d(feature_dim, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)),
            # 32 -> 64
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)),
            # 64 -> 128
            nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        ])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z, coords):
        B = z.shape[0]

        # 1. Prepare Base Feature Map
        # Expand base canvas
        feat = self.base_canvas.expand(B, -1, -1, -1)  # [B, D/2, 16, 16]

        # 2. Inject Spatial Bias (The Foveation Hint)
        # Create a coordinate grid for the initial resolution
        y = torch.linspace(0, 1, self.init_res, device=z.device)
        x = torch.linspace(0, 1, self.init_res, device=z.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [16, 16]

        # Calculate relative distance to the fovea center
        # coords: [B, 2] -> [B, 1, 1]
        cy = coords[:, 1].view(B, 1, 1)
        cx = coords[:, 0].view(B, 1, 1)

        dist_y = grid_y.unsqueeze(0) - cy
        dist_x = grid_x.unsqueeze(0) - cx

        # Concatenate distance map (2 channels) + radial distance (1 channel)
        dist_r = torch.sqrt(dist_y ** 2 + dist_x ** 2)
        pos_feat = torch.stack([dist_y, dist_x, dist_r], dim=1)  # [B, 3, 16, 16]

        # Project pos_feat to remaining channels
        # Note: simplistic projection, could be conv
        pos_emb = pos_feat.repeat(1, (self.feature_dim // 2) // 3 + 1, 1, 1)
        pos_emb = pos_emb[:, :self.feature_dim // 2, :, :]

        # Combine: Base + Position
        x = torch.cat([feat, pos_emb], dim=1)  # [B, D, 16, 16]

        # 3. Modulate with Z (Global Context)
        style = self.style_mlp(z).view(B, self.feature_dim, 1, 1)
        x = x * (1 + style) + style  # Simple affine modulation

        # 4. Upsample
        for block in self.up_blocks:
            x = block(x)

        return self.to_rgb(x)


class ActiveVisionGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = 256

        # 1. Vision System
        self.encoder = Encoder(self.feature_dim)
        self.attn_extractor = RoPECrossAttention(self.feature_dim)

        # 2. Mental Simulation System
        self.predictor = SaccadePredictor(self.feature_dim)

        # 3. Reconstruction System
        self.decoder = FoveatedDecoder(self.feature_dim, args.image_size)

        # 4. Critic (Discriminator)
        self.discriminator = FeatureDiscriminator(self.feature_dim)

        # Buffers for grid
        enc_res = 32  # 128 / 4
        y = torch.linspace(0, 1, enc_res)
        x = torch.linspace(0, 1, enc_res)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('grid_coords', torch.stack((mesh_y, mesh_x), dim=-1).reshape(-1, 2).unsqueeze(0))

    def get_features(self, img, coords):
        """Extract feature z at specific coords."""
        B = img.size(0)
        # Encode
        f_map = self.encoder(img)  # [B, 256, 32, 32]
        f_flat = f_map.flatten(2).transpose(1, 2)  # [B, 1024, 256]

        # Attention Extraction
        coords_q = coords.unsqueeze(1)  # [B, 1, 2]
        coords_k = self.grid_coords.expand(B, -1, -1)

        # Query vector: sampled from feature map at coord (bilinear) + positional awareness via RoPE
        # To give the Query some content content, we bilinear sample the map first
        # Map coords to [-1, 1]
        grid = (coords.view(B, 1, 1, 2) * 2) - 1
        q_content = F.grid_sample(f_map, grid, align_corners=True).view(B, self.feature_dim, 1).transpose(1, 2)

        z = self.attn_extractor(x_q=q_content, x_kv=f_flat, coords_q=coords_q, coords_k=coords_k)
        return z.squeeze(1), f_map

    def forward_generator(self, img, c_start, delta):
        """
        Forward pass for Generator/Predictor optimization.
        1. Extract z_start at c_start.
        2. Reconstruct image from z_start (Constraint: Foveated Recon).
        3. Predict z_pred for c_end = c_start + delta.
        4. Extract z_real at c_end (Ground Truth).
        """
        c_end = torch.clamp(c_start + delta, 0, 1)

        # 1. Observation
        z_start, f_map = self.get_features(img, c_start)

        # 2. Reconstruction (Task: Reconstruct whole image from initial glimpse)
        img_recon = self.decoder(z_start, c_start)

        # 3. Prediction (Mental Saccade)
        z_pred = self.predictor(z_start, delta)

        # 4. Ground Truth Verification
        # Need to extract z_end using the SAME feature map to ensure consistency
        B = img.size(0)
        f_flat = f_map.flatten(2).transpose(1, 2)
        coords_q_end = c_end.unsqueeze(1)
        coords_k = self.grid_coords.expand(B, -1, -1)

        grid_end = (c_end.view(B, 1, 1, 2) * 2) - 1
        q_content_end = F.grid_sample(f_map, grid_end, align_corners=True).view(B, self.feature_dim, 1).transpose(1, 2)

        z_real_end = self.attn_extractor(x_q=q_content_end, x_kv=f_flat, coords_q=coords_q_end,
                                         coords_k=coords_k).squeeze(1)

        return img_recon, z_pred, z_real_end, c_end


#################################################################################
#                               Loss Functions                                  #
#################################################################################

class WeightedFovealLoss(nn.Module):
    def __init__(self, size, sigma=0.2, background_weight=0.1):
        super().__init__()
        self.sigma = sigma
        self.bg_weight = background_weight
        y = torch.linspace(0, 1, size)
        x = torch.linspace(0, 1, size)
        self.register_buffer('my', y.view(1, 1, size, 1))
        self.register_buffer('mx', x.view(1, 1, 1, size))

    def forward(self, pred, target, center):
        # center: [B, 2]
        cx = center[:, 0].view(-1, 1, 1, 1)
        cy = center[:, 1].view(-1, 1, 1, 1)

        dist_sq = (self.mx - cx) ** 2 + (self.my - cy) ** 2
        # Gaussian weight: 1.0 at center, falling off
        weight = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        # Add background awareness (don't let it be zero)
        weight = weight + self.bg_weight

        diff = (pred - target).abs()  # L1 Loss
        weighted_loss = (diff * weight).mean()
        return weighted_loss


def gan_hinge_loss(score_real, score_fake):
    loss_real = torch.relu(1 - score_real).mean()
    loss_fake = torch.relu(1 + score_fake).mean()
    return loss_real + loss_fake


#################################################################################
#                               Training Loop                                   #
#################################################################################

def main(args):
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # --- Logging Setup ---
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        logger = create_logger(args.results_dir)
        logger.info(args)
    else:
        logger = None

    # --- Model Init ---
    model = ActiveVisionGAN(args).to(device)

    # Split parameters for GAN training
    # Generator part: Encoder, Extractor, Predictor, Decoder
    gen_params = list(model.encoder.parameters()) + \
                 list(model.attn_extractor.parameters()) + \
                 list(model.predictor.parameters()) + \
                 list(model.decoder.parameters())

    # Discriminator part
    disc_params = list(model.discriminator.parameters())

    model = DDP(model, device_ids=[device], find_unused_parameters=True)  # needed for split forward passes

    # Two Optimizers
    opt_g = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(disc_params, lr=args.lr, betas=(0.5, 0.9))

    criterion_recon = WeightedFovealLoss(args.image_size, sigma=args.sigma).to(device)

    # --- Data ---
    transform = transforms.Compose([
        transforms.Lambda(lambda p: random_crop_arr(p, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                        drop_last=True)

    # --- Train ---
    scaler = torch.cuda.amp.GradScaler()  # Use scaler for mixed precision

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0: logger.info(f"Start Epoch {epoch}")

        for step, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            B = imgs.size(0)

            # 1. Random Coords & Shifts
            c_start = torch.rand(B, 2, device=device)
            # Shift magnitude: small to medium jumps, not full image
            shift = torch.randn(B, 2, device=device) * 0.3

            # --- Train Discriminator ---
            opt_d.zero_grad()
            with torch.cuda.amp.autocast():
                # We need to run the generation process to get fake features
                with torch.no_grad():
                    # Call model's generator path
                    # We access module because DDP wraps it
                    _, z_pred_fake, z_real_end, _ = model.module.forward_generator(imgs, c_start, shift)

                # Discriminate
                d_real = model.module.discriminator(z_real_end.detach())
                d_fake = model.module.discriminator(z_pred_fake.detach())

                loss_d = gan_hinge_loss(d_real, d_fake)

            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            # --- Train Generator / Predictor / Encoder ---
            opt_g.zero_grad()
            with torch.cuda.amp.autocast():
                img_recon, z_pred, z_real_end, c_end = model.module.forward_generator(imgs, c_start, shift)

                # 1. Reconstruction Loss (Foveated)
                # Ensure the starting view reconstruction is good
                loss_recon = criterion_recon(img_recon, imgs, c_start)

                # 2. Prediction Consistency Loss (Direct Feature Matching)
                # We want z_pred to match z_real_end
                loss_pred_mse = F.mse_loss(z_pred, z_real_end)

                # 3. GAN Generator Loss (Fool the discriminator)
                d_fake_g = model.module.discriminator(z_pred)
                loss_g_adv = -d_fake_g.mean()

                total_g_loss = 10.0 * loss_recon + 1.0 * loss_pred_mse + 0.1 * loss_g_adv

            scaler.scale(total_g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            # === Logging & Step-based Visualization (Overwritable) ===
            if step % 100 == 0 and rank == 0:
                logger.info(
                    f"Ep {epoch} | D: {loss_d.item():.4f} | G_Rec: {loss_recon.item():.4f} | G_Pred: {loss_pred_mse.item():.4f} | G_Adv: {loss_g_adv.item():.4f}")

                # --- VISUALIZATION: Save 'current_vis_step.png' (Overwrite every 100 steps) ---
                with torch.no_grad():
                    # Generate simple visualization using first 4 images of current batch
                    vis_recon, _, _, _ = model.module.forward_generator(imgs[:4], c_start[:4], shift[:4])
                    vis_list = []
                    for i in range(4):
                        orig = imgs[i].cpu()
                        recon = vis_recon[i].cpu()
                        vis_list.extend([orig, recon])

                    # Save to fixed filename to allow overwriting
                    vutils.save_image(torch.stack(vis_list), f"{args.results_dir}/current_vis_step.png", nrow=2)

        # === Epoch End Visualization (Permanent) ===
        if rank == 0:
            with torch.no_grad():
                # Use the last batch of the epoch for archival visualization
                vis_recon, _, _, _ = model.module.forward_generator(imgs[:4], c_start[:4], shift[:4])

                vis_list = []
                for i in range(4):
                    orig = imgs[i].cpu()
                    recon = vis_recon[i].cpu()
                    vis_list.extend([orig, recon])

                # Save with Epoch ID (Permanent)
                vutils.save_image(torch.stack(vis_list), f"{args.results_dir}/epoch_{epoch}_vis.png", nrow=2)

                # Save Checkpoint
                torch.save(model.module.state_dict(), f"{args.results_dir}/last.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--dataset", type=str, default='aoss', choices=['imagenet', 'aoss', 'coco'])
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma", type=float, default=0.2, help="Foveal clarity radius")
    parser.add_argument("--results-dir", type=str, default="results_gan_active")

    args = parser.parse_args()
    main(args)
