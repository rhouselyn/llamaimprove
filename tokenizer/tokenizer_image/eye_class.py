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
import argparse
import numpy as np

# --- 1. 基础设置 ---
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from utils.logger import create_logger
    from utils.distributed import init_distributed_mode
    from dataset.augmentation import random_crop_arr
    from dataset.build import build_dataset
except ImportError:
    print("Warning: Custom utils not found, using dummy implementations.")


    def create_logger(path):
        import logging;
        logging.basicConfig(level=logging.INFO);
        return logging.getLogger()


    def init_distributed_mode(args):
        pass


    def random_crop_arr(img, size):
        return img


    def build_dataset(args, transform):
        # Mock 数据集返回 Label
        return torch.utils.data.TensorDataset(torch.randn(100, 3, 128, 128), torch.randint(0, 1000, (100,)))


#################################################################################
#                           RoPE & Positional Utils                             #
#################################################################################

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", 1.0 / (max_freq ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))))

    def forward(self, x, pos):
        # pos: [B, SeqLen, 2]
        pos_y, pos_x = pos[..., 0], pos[..., 1]
        freqs_y = torch.einsum('bi,j->bij', pos_y, self.inv_freq)
        freqs_x = torch.einsum('bi,j->bij', pos_x, self.inv_freq)
        return torch.cat([freqs_y, freqs_x], dim=-1)


def apply_rotary_pos_emb(t, freqs):
    # t: [B, SeqLen, Heads, Dim]
    freqs = freqs.unsqueeze(2)
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
    def __init__(self, in_dim=256, match_dim=128, out_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        assert match_dim % num_heads == 0
        assert out_dim % num_heads == 0
        self.head_match_dim = match_dim // num_heads
        self.head_v_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, match_dim)
        self.k_proj = nn.Linear(in_dim, match_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.rope = RotaryEmbedding(self.head_match_dim)

    def forward(self, x_q, x_kv, coords_q, coords_k):
        B, Lq, _ = x_q.shape
        _, Lk, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, Lq, self.num_heads, self.head_match_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, Lk, self.num_heads, self.head_match_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Lk, self.num_heads, self.head_v_dim).transpose(1, 2)

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
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, feature_dim, 3, 1, 1), nn.BatchNorm2d(feature_dim), nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class LatentNavigator(nn.Module):
    def __init__(self, feature_dim=256, num_freqs=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer('freq_bands', 2.0 ** torch.arange(num_freqs))
        pos_dim = 2 + 2 * 2 * num_freqs
        in_dim = feature_dim + pos_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, feature_dim)
        )

    def positional_encoding(self, coords):
        orig_shape = coords.shape
        x = coords.flatten(0, -2)
        spectrum = x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
        sin_emb = torch.sin(spectrum)
        cos_emb = torch.cos(spectrum)
        emb = torch.cat([x, sin_emb.flatten(1), cos_emb.flatten(1)], dim=-1)
        return emb.view(*orig_shape[:-1], -1)

    def forward(self, z, delta):
        pos_emb = self.positional_encoding(delta)
        if delta.dim() == 4:
            B, H, W, _ = delta.shape
            z_in = z.view(B, 1, 1, -1).expand(-1, H, W, -1)
        else:
            z_in = z
        mlp_in = torch.cat([z_in, pos_emb], dim=-1)
        return self.net(mlp_in)


class FeatureDiscriminator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.net(z)


class ActiveVisionGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = 256

        self.encoder = Encoder(self.feature_dim)
        self.attn_extractor = RoPECrossAttention(
            in_dim=self.feature_dim, match_dim=128, out_dim=self.feature_dim
        )
        self.navigator = LatentNavigator(self.feature_dim)
        self.discriminator = FeatureDiscriminator(self.feature_dim)

        # ImageNet 分类头
        self.classifier = nn.Linear(self.feature_dim, 1000)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(self.feature_dim, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, 3, 1, 1), nn.Sigmoid()
        )

        enc_res = args.image_size // 4
        y = torch.linspace(0, 1, enc_res)
        x = torch.linspace(0, 1, enc_res)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('grid_coords', torch.stack((mesh_y, mesh_x), dim=-1).reshape(-1, 2).unsqueeze(0))

    def get_features_parallel(self, img, coords_list):
        B = img.size(0)
        f_map = self.encoder(img)
        f_flat = f_map.flatten(2).transpose(1, 2)

        coords_q = torch.stack(coords_list, dim=1)
        grid = (coords_q.view(B, -1, 1, 2) * 2) - 1
        q_content = F.grid_sample(f_map, grid, align_corners=True).squeeze(-1).transpose(1, 2)

        coords_k = self.grid_coords.expand(B, -1, -1)
        z_seq = self.attn_extractor(x_q=q_content, x_kv=f_flat, coords_q=coords_q, coords_k=coords_k)

        return z_seq, f_map

    def forward_generator(self, img, c_start, delta_eye):
        B = img.size(0)
        c_end = torch.clamp(c_start + delta_eye, 0, 1)

        z_both, f_map = self.get_features_parallel(img, [c_start, c_end])

        z_start = z_both[:, 0]
        z_real_end = z_both[:, 1]

        logits_start = self.classifier(z_start)
        logits_end = self.classifier(z_real_end)

        z_pred = self.navigator(z_start, delta_eye)

        logits_pred = self.classifier(z_pred)

        recon_res = 16
        y = torch.linspace(0, 1, recon_res, device=img.device)
        x = torch.linspace(0, 1, recon_res, device=img.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        pixel_grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        delta_pixel = pixel_grid - c_start.view(B, 1, 1, 2)
        recon_feats = self.navigator(z_start, delta_pixel)
        recon_feats = recon_feats.permute(0, 3, 1, 2)
        img_recon = self.to_rgb(F.interpolate(recon_feats, size=img.shape[-1], mode='bilinear', align_corners=False))

        return img_recon, z_pred, z_real_end, logits_start, logits_end, logits_pred


#################################################################################
#                       Loss & Visualization Utils                              #
#################################################################################

# [修改] 使用 UnifiedFovealLoss 替代原有的 HardWindowLoss
class UnifiedFovealLoss(nn.Module):
    """
    统一的视网膜凹Loss (Unified Foveal Loss)
    支持 'hard', 'soft', 'hybrid' 三种模式。
    Hybrid模式: 核心区(box内)权重1.0，周边区(box外)高斯衰减+保底权重。
    """

    def __init__(self, image_size, ratio=16.0, sigma=0.2, background_weight=0.1, mode='hybrid'):
        super().__init__()
        self.image_size = image_size
        self.ratio = ratio
        self.sigma = sigma
        self.bg_weight = background_weight
        self.mode = mode

        self.half_win = int((image_size / ratio) / 2) if ratio > 0 else 0

        # 硬窗口计算用的坐标
        y = torch.arange(image_size)
        x = torch.arange(image_size)
        self.register_buffer('grid_y', y.view(1, 1, image_size, 1))
        self.register_buffer('grid_x', x.view(1, 1, 1, image_size))

        # 软/混合模式用的归一化坐标
        self.register_buffer('norm_y', torch.linspace(0, 1, image_size).view(1, 1, image_size, 1))
        self.register_buffer('norm_x', torch.linspace(0, 1, image_size).view(1, 1, 1, image_size))

    def forward(self, pred, target, center):
        B = pred.size(0)
        diff = (pred - target).abs()  # L1 Loss

        # 1. 计算 Hard Mask (如果需要)
        if self.mode in ['hard', 'hybrid']:
            cy_idx = (center[:, 0] * self.image_size).long().view(B, 1, 1, 1)
            cx_idx = (center[:, 1] * self.image_size).long().view(B, 1, 1, 1)

            mask_y = (self.grid_y >= (cy_idx - self.half_win)) & (self.grid_y <= (cy_idx + self.half_win))
            mask_x = (self.grid_x >= (cx_idx - self.half_win)) & (self.grid_x <= (cx_idx + self.half_win))
            box_mask = (mask_y & mask_x).float()

        # 2. 计算权重
        if self.mode == 'hard':
            weighted_loss = (diff * box_mask).sum() / (box_mask.sum() + 1e-6)
            return weighted_loss

        elif self.mode == 'soft':
            cy_norm = center[:, 0].view(B, 1, 1, 1)
            cx_norm = center[:, 1].view(B, 1, 1, 1)
            dist_sq = (self.norm_x - cx_norm) ** 2 + (self.norm_y - cy_norm) ** 2
            weight_map = torch.exp(-dist_sq / (2 * self.sigma ** 2)) + self.bg_weight
            weighted_loss = (diff * weight_map).mean()
            return weighted_loss

        elif self.mode == 'hybrid':
            # 混合模式: Box内固定1.0, Box外高斯衰减
            cy_norm = center[:, 0].view(B, 1, 1, 1)
            cx_norm = center[:, 1].view(B, 1, 1, 1)
            dist_sq = (self.norm_x - cx_norm) ** 2 + (self.norm_y - cy_norm) ** 2
            soft_decay = torch.exp(-dist_sq / (2 * self.sigma ** 2)) + self.bg_weight

            # 使用 torch.where 进行融合
            final_weight = torch.where(box_mask > 0.5, torch.tensor(1.0, device=pred.device), soft_decay)
            weighted_loss = (diff * final_weight).mean()
            return weighted_loss

        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")


def gan_hinge_loss(score_real, score_fake):
    loss_real = torch.relu(1 - score_real).mean()
    loss_fake = torch.relu(1 + score_fake).mean()
    return loss_real + loss_fake


def draw_gaze_marker(img_tensor, center, color=(1.0, 0.0, 0.0), marker_size=10):
    _, H, W = img_tensor.shape
    cy = int(center[0] * H)
    cx = int(center[1] * W)
    cy = max(0, min(H - 1, cy))
    cx = max(0, min(W - 1, cx))
    half = marker_size // 2
    y_min, y_max = max(0, cy - half), min(H, cy + half)
    x_min, x_max = max(0, cx - half), min(W, cx + half)
    for c in range(3):
        img_tensor[c, y_min:y_max, cx] = color[c]
        img_tensor[c, cy, x_min:x_max] = color[c]
    return img_tensor


#################################################################################
#                               Training Loop                                   #
#################################################################################

def main(args):
    init_distributed_mode(args)
    rank = dist.get_rank() if dist.is_initialized() else 0
    device_count = torch.cuda.device_count()
    device = rank % device_count if device_count > 0 else 'cpu'
    if isinstance(device, int):
        torch.cuda.set_device(device)

    save_dir = args.results_dir
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        existing_runs = [d for d in os.listdir(args.results_dir) if
                         os.path.isdir(os.path.join(args.results_dir, d)) and d.startswith("run_")]
        run_ids = []
        for d in existing_runs:
            try:
                run_ids.append(int(d.split("_")[-1]))
            except ValueError:
                pass
        next_id = max(run_ids) + 1 if run_ids else 1
        save_dir = os.path.join(args.results_dir, f"run_{next_id}")
        os.makedirs(save_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(args)
        logger.info(f"Using Reconstruction Loss Mode: {args.loss_mode}")
    else:
        logger = None

    model = ActiveVisionGAN(args).to(device)

    gen_params = list(model.encoder.parameters()) + \
                 list(model.attn_extractor.parameters()) + \
                 list(model.navigator.parameters()) + \
                 list(model.to_rgb.parameters()) + \
                 list(model.classifier.parameters())
    disc_params = list(model.discriminator.parameters())

    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        model_module = model.module
    else:
        model_module = model

    opt_g = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(disc_params, lr=args.lr, betas=(0.5, 0.9))

    # [修改] 使用 UnifiedFovealLoss 初始化
    criterion_recon = UnifiedFovealLoss(
        image_size=args.image_size,
        ratio=args.window_ratio,
        mode=args.loss_mode,
        background_weight=0.1
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss().to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda p: random_crop_arr(p, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = build_dataset(args, transform=transform)

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(args.epochs):
        if dist.is_initialized():
            sampler.set_epoch(epoch)
        if rank == 0 and logger: logger.info(f"Start Epoch {epoch}")

        for step, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            B = imgs.size(0)

            c_start = torch.rand(B, 2, device=device)
            shift = torch.randn(B, 2, device=device) * 0.3

            # --- Train Discriminator ---
            opt_d.zero_grad()
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    _, z_pred_fake, z_real_end, _, _, _ = model_module.forward_generator(imgs, c_start, shift)

                d_real = model_module.discriminator(z_real_end.detach())
                d_fake = model_module.discriminator(z_pred_fake.detach())
                loss_d = gan_hinge_loss(d_real, d_fake)

            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            # --- Train Generator ---
            opt_g.zero_grad()
            with torch.amp.autocast('cuda'):
                img_recon, z_pred, z_real_end, logit_start, logit_end, logit_pred = \
                    model_module.forward_generator(imgs, c_start, shift)

                loss_recon = criterion_recon(img_recon, imgs, c_start)
                loss_pred_mse = F.mse_loss(z_pred, z_real_end)
                d_fake_g = model_module.discriminator(z_pred)
                loss_g_adv = -d_fake_g.mean()

                loss_cls_start = criterion_cls(logit_start, labels)
                loss_cls_end = criterion_cls(logit_end, labels)
                loss_cls_pred = criterion_cls(logit_pred, labels)
                loss_cls_total = (loss_cls_start + loss_cls_end + loss_cls_pred) / 3.0

                total_g_loss = 10.0 * loss_recon + \
                               1.0 * loss_pred_mse + \
                               0.1 * loss_g_adv + \
                               1.0 * loss_cls_total

            scaler.scale(total_g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            if step % 100 == 0 and rank == 0 and logger:
                logger.info(
                    f"Ep {epoch} | "
                    f"Rec: {loss_recon.item():.3f} | "
                    f"MSE: {loss_pred_mse.item():.3f} | "
                    f"Adv: {loss_g_adv.item():.3f} | "
                    f"Cls: {loss_cls_total.item():.3f} | "
                    f"Total: {total_g_loss.item():.3f}"
                )

                with torch.no_grad():
                    vis_recon, _, _, _, _, _ = model_module.forward_generator(imgs[:4], c_start[:4], shift[:4])
                    vis_list = []
                    cpu_imgs = imgs[:4].cpu().clone()
                    cpu_recon = vis_recon[:4].cpu()
                    cpu_coords = c_start[:4].cpu()
                    for i in range(4):
                        orig_marked = draw_gaze_marker(cpu_imgs[i], cpu_coords[i])
                        vis_list.extend([orig_marked, cpu_recon[i]])
                    vutils.save_image(torch.stack(vis_list), f"{save_dir}/current_vis_step.png", nrow=2)

        if rank == 0:
            torch.save(model_module.state_dict(), f"{save_dir}/last.pt")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default='/mnt/afs/zhengmingkai/raozf/llamagen/imagenet_train_filelist.txt')
    parser.add_argument("--dataset", type=str, default='aoss', choices=['imagenet', 'aoss'])
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--results-dir", type=str, default="results_gan_active")

    # [新增] Loss配置参数
    parser.add_argument("--window-ratio", type=float, default=16.0)
    parser.add_argument("--loss-mode", type=str, default="hybrid", choices=['hard', 'soft', 'hybrid'],
                        help="hard: old window mask; soft: gaussian only; hybrid: window(1.0) + gaussian decay")

    args = parser.parse_args()
    main(args)
