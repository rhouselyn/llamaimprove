import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.tokenizer_image.bsqdc_model import VQ_models, ModelArgs  # 请替换为你的实际路径


class TemporalCompressor(nn.Module):
    def __init__(self, in_channels, out_channels, frames):
        super().__init__()
        # 压缩时间上的残差特征，使用 3D 卷积平滑空间和时间
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, in_channels // 2),
            nn.SiLU(),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=(frames - 1, 1, 1), padding=0)
            # 聚合时间维度为1
        )

    def forward(self, x):
        # x: (B, C, F-1, H, W)
        return self.conv(x).squeeze(2)  # (B, out_C, H, W)


class TemporalDecompressor(nn.Module):
    def __init__(self, in_channels, out_channels, frames):
        super().__init__()
        self.frames = frames
        self.expand = nn.Conv2d(in_channels, out_channels * (frames - 1), kernel_size=1)
        # 解压后使用 3D 卷积增加相邻帧的连贯性
        self.smooth = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        # x: (B, in_C, H, W)
        B, _, H, W = x.shape
        x = self.expand(x)  # (B, out_C * (F-1), H, W)
        x = x.view(B, -1, self.frames - 1, H, W)  # (B, out_C, F-1, H, W)
        x = self.smooth(x)
        return x


class Latent3DSmoother(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = F.silu(self.norm(self.conv1(x)))
        x = self.conv2(x)
        return res + x


class VideoTokenizer(nn.Module):
    def __init__(self, base_vq_name='DCAE-32', frames=16):
        super().__init__()
        self.frames = frames
        # 1. 加载冻结的图像 VQ 模型
        self.image_vq = VQ_models[base_vq_name](sample=False)
        for param in self.image_vq.parameters():
            param.requires_grad = False
        self.image_vq.eval()

        self.embed_dim = self.image_vq.quantize.embed_dim
        self.time_dim = 2 * frames  # 按你的要求：时间残差压缩到 2*F 的维度

        # 2. 视频残差压缩网络
        # 残差包含：角度 \theta (1维) + 正交方向 v (embed_dim 维) = embed_dim + 1
        self.compressor = TemporalCompressor(self.embed_dim + 1, self.time_dim, frames)
        self.decompressor = TemporalDecompressor(self.time_dim, self.embed_dim + 1, frames)

        # 3. 潜空间时间平滑模块
        self.latent_smoother = Latent3DSmoother(self.embed_dim)

    def get_spherical_residual(self, x, y):
        """计算球面上 x 到 y 的残差：角度和正交方向"""
        # x, y: (B, C, H, W), 已 L2 归一化
        cos_theta = (x * y).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta) / math.pi  # 归一化到 [0, 1]

        # 计算正交方向 v
        v_unnorm = y - x * cos_theta
        v = F.normalize(v_unnorm, p=2, dim=1, eps=1e-6)  # 范围 [-1, 1]

        return torch.cat([theta, v], dim=1)  # (B, C+1, H, W)

    def apply_spherical_residual(self, x, residual):
        """利用上一帧 x 和残差还原下一帧 y"""
        theta_norm, v = residual[:, 0:1, ...], residual[:, 1:, ...]
        theta = theta_norm * math.pi
        v = F.normalize(v, p=2, dim=1, eps=1e-6)

        y = x * torch.cos(theta) + v * torch.sin(theta)
        return F.normalize(y, p=2, dim=1)

    def encode_video(self, video):
        # video: (B, F, 3, H, W)
        B, F, C, H, W = video.shape
        video_flat = video.view(B * F, C, H, W)

        with torch.no_grad():
            z = self.image_vq.encoder(video_flat)
            z = self.image_vq.quant_conv(z)
            # 获取球面 L2 归一化且量化的 embedding
            z_q, _ = self.image_vq.quantize(z)

        _, C_emb, H_emb, W_emb = z_q.shape
        z_q = z_q.view(B, F, C_emb, H_emb, W_emb)

        first_frame_emb = z_q[:, 0, ...]

        # 计算逐帧残差
        residuals = []
        for i in range(1, F):
            res = self.get_spherical_residual(z_q[:, i - 1, ...], z_q[:, i, ...])
            residuals.append(res)

        residuals = torch.stack(residuals, dim=2)  # (B, C+1, F-1, H_emb, W_emb)

        # 压缩时间残差
        time_embedding = self.compressor(residuals)  # (B, time_dim, H_emb, W_emb)

        # 拼接首帧和时间特征
        video_embedding = torch.cat([first_frame_emb, time_embedding], dim=1)
        return video_embedding  # (B, embed_dim + 2F, H_emb, W_emb)

    def decode_video(self, video_embedding):
        B, C_total, H_emb, W_emb = video_embedding.shape
        first_frame_emb = video_embedding[:, :self.embed_dim, ...]
        time_embedding = video_embedding[:, self.embed_dim:, ...]

        # 解压时间残差
        residuals_rec = self.decompressor(time_embedding)  # (B, C+1, F-1, H_emb, W_emb)

        # 逐帧还原
        frames_emb = [first_frame_emb]
        current_frame = first_frame_emb
        for i in range(self.frames - 1):
            next_frame = self.apply_spherical_residual(current_frame, residuals_rec[:, :, i, ...])
            frames_emb.append(next_frame)
            current_frame = next_frame

        frames_emb = torch.stack(frames_emb, dim=2)  # (B, embed_dim, F, H_emb, W_emb)

        # 使用 3D 卷积平滑潜空间
        frames_emb = self.latent_smoother(frames_emb)

        # 使用固定的图像 Decoder 逐帧解码
        frames_emb_flat = frames_emb.transpose(1, 2).reshape(B * self.frames, self.embed_dim, H_emb, W_emb)

        # 注意这里要还原回原始 z_channels 的维度才能送入原 decoder
        post_q = self.image_vq.post_quant_conv(frames_emb_flat)
        decoded_flat = self.image_vq.decoder(post_q)

        decoded_video = decoded_flat.view(B, self.frames, 3, decoded_flat.shape[2], decoded_flat.shape[3])
        return decoded_video

    def forward(self, video):
        video_emb = self.encode_video(video)
        recon_video = self.decode_video(video_emb)
        return recon_video