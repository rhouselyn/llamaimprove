import argparse
import os
import random
from typing import Iterable, Tuple

# 必须放在 import tensorflow 之前
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import requests
import tensorflow.compat.v1 as tf
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# 从训练脚本导入必要的部分
from tokenizer.tokenizer_image.bsqdc_model import VQ_models


# Inception 配置
INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"
FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


class CustomImageDataset(Dataset):
    """
    无论图片是平铺在根目录，还是放在子文件夹中，这个 Dataset 都能加载。
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        print(f"Scanning images in {root}...")
        for dp, dn, filenames in os.walk(root):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    self.images.append(os.path.join(dp, f))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No valid images found in {root}")

        self.images.sort()
        print(f"Found {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            img = Image.new("RGB", (256, 256), color="black")

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def create_tf_session(use_tf_gpu: bool = False):
    """
    默认让 TensorFlow 只走 CPU，避免在 H20/H100 等机器上因 PTX/驱动不匹配而报错。
    只有显式传 --tf-use-gpu 才让 TF 用 GPU。
    """
    config = tf.ConfigProto(allow_soft_placement=True)

    if use_tf_gpu:
        config.gpu_options.allow_growth = True
        print("TensorFlow will use GPU for FID.")
    else:
        config.device_count["GPU"] = 0
        print("TensorFlow will use CPU for FID.")

    return tf.Session(config=config)


class Evaluator:
    def __init__(self, session, batch_size=64, softmax_batch_size=512):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size

        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations([np.zeros([1, 256, 256, 3], dtype=np.float32)])

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        preds, spatial_preds = [], []

        for batch in tqdm(batches, desc="Computing Activations"):
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features],
                {self.image_input: batch},
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))

        return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []

        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))

        preds = np.concatenate(softmax_out, axis=0)
        scores = []

        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part + 1e-12) - np.log(np.expand_dims(np.mean(part, 0), 0) + 1e-12))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))

        return float(np.mean(scores))


def _download_inception_model():
    if not os.path.exists(INCEPTION_V3_PATH):
        print(f"Downloading InceptionV3 to {INCEPTION_V3_PATH}...")
        with requests.get(INCEPTION_V3_URL, stream=True) as r:
            r.raise_for_status()
            with open(INCEPTION_V3_PATH + ".tmp", "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading Inception"):
                    if chunk:
                        f.write(chunk)
            os.rename(INCEPTION_V3_PATH + ".tmp", INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2 ** 32)}_{random.randrange(2 ** 32)}"

    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    return pool3, spatial[..., :7]


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2 ** 32)}_{random.randrange(2 ** 32)}"

    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    (matmul,) = tf.import_graph_def(
        graph_def,
        return_elements=["softmax/logits/MatMul"],
        name=prefix,
    )
    logits = tf.matmul(input_batch, matmul.inputs[1])
    return tf.nn.softmax(logits)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")

    # 初始化 VQ 模型
    vq_model = VQ_models[args.vq_model](
        dropout_p=args.dropout_p,
    ).to(device)

    # 加载权重
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["ema"] if args.use_ema and "ema" in checkpoint else checkpoint["model"]
    vq_model.load_state_dict(state_dict)
    vq_model.eval()
    print(f"Loaded weights from {args.ckpt_path} (EMA: {args.use_ema})")

    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_dataset = CustomImageDataset(args.data_path, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # TensorFlow Session 初始化
    tf_session = create_tf_session(use_tf_gpu=args.tf_use_gpu)
    evaluator = Evaluator(tf_session)

    try:
        evaluator.warmup()
    except Exception as e:
        raise RuntimeError(
            "TensorFlow evaluator warmup failed. "
            "If you are on H20/H100 and see PTX/CUDA errors, do NOT pass --tf-use-gpu. "
            "Use the default CPU FID path instead."
        ) from e

    # 1. 计算真图特征
    def get_real_batches():
        for x, _ in val_loader:
            yield (x.permute(0, 2, 3, 1).numpy() * 127.5 + 127.5).astype(np.float32)

    print("\nStep 1/2: Computing reference activations...")
    ref_acts = evaluator.compute_activations(get_real_batches())
    np.savez(args.ref_npz, pool=ref_acts[0], spatial=ref_acts[1])

    # 2. 计算重建图特征
    def get_recon_batches():
        for x, _ in val_loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                recon, _ = vq_model(x)
            yield (recon.permute(0, 2, 3, 1).cpu().numpy() * 127.5 + 127.5).astype(np.float32)

    print("\nStep 2/2: Computing sample (reconstructed) activations...")
    sample_acts = evaluator.compute_activations(get_recon_batches())
    np.savez(args.sample_npz, pool=sample_acts[0], spatial=sample_acts[1])

    # 3. 计算最终指标
    print("\nAnalyzing metrics...")
    ref_stats = evaluator.compute_statistics(ref_acts[0])
    ref_stats_spatial = evaluator.compute_statistics(ref_acts[1])
    sample_stats = evaluator.compute_statistics(sample_acts[0])
    sample_stats_spatial = evaluator.compute_statistics(sample_acts[1])

    inception_score = evaluator.compute_inception_score(sample_acts[0])
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)

    result_str = (
        f"Inception Score: {inception_score:.4f}\n"
        f"FID: {fid:.4f}\n"
        f"sFID: {sfid:.4f}"
    )

    print("\n" + "=" * 30)
    print(result_str)
    print("=" * 30)

    with open(args.sample_npz.replace(".npz", ".txt"), "w") as f:
        f.write(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 模型路径参数
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights")

    # 核心路径参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/afs/zhengmingkai/whl/llamagen/ILSVRC/Data/CLS-LOC/val",
        help="Path to ImageNet validation set",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")

    # 模型架构参数
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--quantizer", type=str, default="bsq", choices=["bsq", "fsq", "vq"])
    parser.add_argument("--num-bits", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--commit-loss-beta", type=float, default=0.25)
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)

    # 运行配置
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--ref-npz", type=str, default="ref_val_stats.npz")
    parser.add_argument("--sample-npz", type=str, default="recon_val_stats.npz")

    # TensorFlow FID 配置
    parser.add_argument(
        "--tf-use-gpu",
        action="store_true",
        help="Use GPU for TensorFlow FID evaluator. Default is CPU-only for stability.",
    )

    args = parser.parse_args()
    main(args)