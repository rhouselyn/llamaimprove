import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time

# 假设你有类似 VideoFolder 的 dataset，返回 (B, F, 3, H, W)
from dataset.video_dataset import VideoDataset
from video_tokenizer import VideoTokenizer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = VideoTokenizer(base_vq_name=args.vq_model, frames=args.frames).to(device)

    # 仅优化新加的参数
    trainable_params = list(model.compressor.parameters()) + \
                       list(model.decompressor.parameters()) + \
                       list(model.latent_smoother.parameters())

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # 模拟 DataLoader
    dataset = VideoDataset(args.data_path, frames=args.frames, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    os.makedirs(args.save_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()

    model.train()
    # image_vq 的部分在 init 里已经 eval 和 requires_grad=False 了

    for epoch in range(args.epochs):
        for step, (video, _) in enumerate(loader):
            video = video.to(device)  # (B, F, 3, H, W)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                recon_video = model(video)

                # 视频重建损失 (可以加入 perceptual loss / 3D GAN loss 等，这里演示最基础的 L2/L1)
                loss_mse = F.mse_loss(recon_video, video)
                loss_l1 = F.l1_loss(recon_video, video)
                loss = loss_mse + 0.5 * loss_l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.log_every == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}] Loss: {loss.item():.4f}")

        if epoch % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"video_tokenizer_epoch_{epoch}.pt")
            torch.save({
                'compressor': model.compressor.state_dict(),
                'decompressor': model.decompressor.state_dict(),
                'latent_smoother': model.latent_smoother.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-model", type=str, default="DCAE-32")
    parser.add_argument("--frames", type=int, default=16, help="视频处理的帧数")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints_video")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)

    args = parser.parse_args()
    train(args)