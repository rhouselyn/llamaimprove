import argparse
import os
import random
from glob import glob
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tokenizer.tokenizer_image.bsqplus_video_model import (
    BSQPlusVideoTokenizer,
    VideoTokenizerConfig,
    load_image_tokenizer_from_ckpt,
)


class VideoFolderDataset(Dataset):
    """
    Expected structure:
    root/
      clip_0001/*.jpg
      clip_0002/*.png
      ...
    """

    IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    def __init__(self, root: str, num_frames: int, image_size: int = 256, random_sample: bool = True):
        self.root = root
        self.num_frames = num_frames
        self.random_sample = random_sample
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.clips: List[List[str]] = []
        clip_dirs = sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)])
        for d in clip_dirs:
            frames = []
            for ext in self.IMG_EXT:
                frames.extend(glob(os.path.join(d, ext)))
            frames = sorted(frames)
            if len(frames) >= num_frames:
                self.clips.append(frames)

        if len(self.clips) == 0:
            raise RuntimeError(f"No valid clips found in {root}. Need >= {num_frames} frames per clip.")

    def __len__(self):
        return len(self.clips)

    def _pick_indices(self, n: int):
        if n == self.num_frames:
            return list(range(n))

        if self.random_sample:
            start = random.randint(0, n - self.num_frames)
            return list(range(start, start + self.num_frames))

        # Uniform deterministic sampling for validation/inference
        step = (n - 1) / (self.num_frames - 1)
        return [round(i * step) for i in range(self.num_frames)]

    def __getitem__(self, idx):
        frame_paths = self.clips[idx]
        indices = self._pick_indices(len(frame_paths))
        imgs = []
        for i in indices:
            with Image.open(frame_paths[i]).convert("RGB") as im:
                imgs.append(self.transform(im))
        video = torch.stack(imgs, dim=0)  # [T, 3, H, W]
        return video


def parse_args():
    parser = argparse.ArgumentParser("Train temporal residual video tokenizer on top of BSQ+ image tokenizer")

    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--image_tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/video_tokenizer")

    parser.add_argument("--vq_model", type=str, default="VQ-16", choices=["VQ-8", "VQ-16"])
    parser.add_argument("--quantizer", type=str, default="bsq")
    parser.add_argument("--num_bits", type=int, default=14)
    parser.add_argument("--codebook_embed_dim", type=int, default=8)
    parser.add_argument("--codebook_l2_norm", action="store_true", default=True)
    parser.add_argument("--codebook_size", type=int, default=16384)
    parser.add_argument("--commit_loss_beta", type=float, default=0.0)
    parser.add_argument("--entropy_loss_ratio", type=float, default=0.1)
    parser.add_argument("--dropout_p", type=float, default=0.0)

    parser.add_argument("--num_frames", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--compressed_channels", type=int, default=None)
    parser.add_argument("--disable_3d_smoother", action="store_true")
    parser.add_argument("--smoother_layers", type=int, default=2)
    parser.add_argument("--smoother_kernel", type=int, default=3)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--lambda_latent", type=float, default=0.5)
    parser.add_argument("--lambda_residual", type=float, default=0.2)
    parser.add_argument("--lambda_temporal_smooth", type=float, default=0.05)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)

    return parser.parse_args()


def build_dtype(mixed_precision: str):
    return {
        "none": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[mixed_precision]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scaler, dtype, device, args, epoch):
    model.train()
    total = 0.0

    for step, videos in enumerate(loader, start=1):
        videos = videos.to(device, non_blocking=True)  # [B, T, 3, H, W]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            recon, aux = model(videos, return_latents=True)

            frame_loss = F.l1_loss(recon, videos)
            latent_loss = F.mse_loss(aux["recon_latents"], aux["q_latents"].detach())
            residual_loss = F.mse_loss(aux["pred_residuals"], aux["gt_residuals"].detach())

            if aux["pred_residuals"].shape[1] > 1:
                temporal_smooth = (aux["pred_residuals"][:, 1:] - aux["pred_residuals"][:, :-1]).abs().mean()
            else:
                temporal_smooth = torch.zeros_like(frame_loss)

            loss = (
                frame_loss
                + args.lambda_latent * latent_loss
                + args.lambda_residual * residual_loss
                + args.lambda_temporal_smooth * temporal_smooth
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()

        if step % args.log_every == 0:
            avg = total / step
            print(
                f"[Epoch {epoch}] step={step}/{len(loader)} "
                f"loss={avg:.4f} frame={frame_loss.item():.4f} "
                f"latent={latent_loss.item():.4f} residual={residual_loss.item():.4f}"
            )

    return total / max(1, len(loader))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = build_dtype(args.mixed_precision)

    dataset = VideoFolderDataset(
        root=args.video_root,
        num_frames=args.num_frames,
        image_size=args.image_size,
        random_sample=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    image_tokenizer = load_image_tokenizer_from_ckpt(
        ckpt_path=args.image_tokenizer_ckpt,
        vq_model=args.vq_model,
        quantizer=args.quantizer,
        num_bits=args.num_bits,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        codebook_size=args.codebook_size,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        sample=False,
        map_location="cpu",
    ).to(device)

    cfg = VideoTokenizerConfig(
        num_frames=args.num_frames,
        compressed_channels=args.compressed_channels,
        use_3d_smoother=not args.disable_3d_smoother,
        smoother_layers=args.smoother_layers,
        smoother_kernel=args.smoother_kernel,
    )
    model = BSQPlusVideoTokenizer(
        image_tokenizer=image_tokenizer,
        config=cfg,
        freeze_image_tokenizer=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    print(f"Dataset clips: {len(dataset)}")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Video token channels: {model.video_token_channels} (base={model.embed_dim}, temporal={model.compressed_channels})")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scaler, dtype, device, args, epoch)
        print(f"Epoch {epoch} done. avg_loss={loss:.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"video_tokenizer_epoch{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "video_cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
