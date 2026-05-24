import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import requests
import random
# 从训练脚本导入必要的部分
from tokenizer.tokenizer_image.bsqdc_model import VQ_models
from dataset.build import build_dataset

# 从评估脚本导入必要的类和函数
import tensorflow.compat.v1 as tf
from scipy import linalg
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


class InvalidFIDException(Exception):
    pass


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
        assert mu1.shape == mu2.shape, f"Mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


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
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i: i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i: i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    import random
    prefix = f"{random.randrange(2 ** 32)}_{random.randrange(2 ** 32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2 ** 32)}_{random.randrange(2 ** 32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                shape = [s for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def main(args):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_model = VQ_models[args.vq_model](
        # codebook_size=args.codebook_size,
        # codebook_embed_dim=args.codebook_embed_dim,
        # commit_loss_beta=args.commit_loss_beta,
        # entropy_loss_ratio=args.entropy_loss_ratio,
        # dropout_p=args.dropout_p,
    ).to(device)

    # 加载checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if args.use_ema and "ema" in checkpoint:
        vq_model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights.")
    else:
        vq_model.load_state_dict(checkpoint["model"])
        print("Loaded model weights.")
    vq_model.eval()

    # 准备val数据集
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    val_dataset = build_dataset(args, transform=transform)  # 移除is_train参数
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # 创建Evaluator
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))
    evaluator.warmup()

    # 生成参考激活 (真实图像)
    def get_image_batches(loader):
        for x, _ in loader:
            x = x.permute(0, 2, 3, 1).numpy() * 127.5 + 127.5
            yield x

    print("Computing reference activations...")
    ref_acts = evaluator.compute_activations(get_image_batches(val_loader))
    np.savez(args.ref_npz, arr_0=ref_acts[0], arr_1=ref_acts[1])

    # 生成样本激活 (重建图像)
    def get_recon_batches(loader, model):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                recon, _ = model(x)
            recon = recon.permute(0, 2, 3, 1).cpu().numpy() * 127.5 + 127.5
            yield recon

    print("Computing sample (reconstructed) activations...")
    sample_acts = evaluator.compute_activations(get_recon_batches(val_loader, vq_model))
    np.savez(args.sample_npz, arr_0=sample_acts[0], arr_1=sample_acts[1])

    # 计算评估指标
    print("Computing statistics...")
    ref_stats, ref_stats_spatial = evaluator.compute_statistics(ref_acts[0]), evaluator.compute_statistics(ref_acts[1])
    sample_stats, sample_stats_spatial = evaluator.compute_statistics(sample_acts[0]), evaluator.compute_statistics(
        sample_acts[1])

    print("Computing evaluations...")
    IS = evaluator.compute_inception_score(sample_acts[0])
    FID = sample_stats.frechet_distance(ref_stats)
    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    print("Inception Score:", IS)
    print("FID:", FID)
    print("sFID:", sFID)

    # 保存结果到txt
    txt_path = args.sample_npz.replace('.npz', '.txt')
    with open(txt_path, 'w') as f:
        print("Inception Score:", IS, file=f)
        print("FID:", FID, file=f)
        print("sFID:", sFID, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the trained VQ checkpoint")
    parser.add_argument("--use-ema", action='store_true', help="Use EMA weights if available")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="DCAE-32")
    parser.add_argument("--codebook-size", type=int, default=16384)
    # parser.add_argument("--codebook-embed-dim", type=int, default=8)
    # parser.add_argument("--commit-loss-beta", type=float, default=0.25)
    # parser.add_argument("--entropy-loss-ratio", type=float, default=0.0)
    # parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default='aoss')
    parser.add_argument("--aoss-bucket", type=str, default="imagenet")
    parser.add_argument("--data-path", type=str, default='/mnt/afs/zhengmingkai/whl/llamagen/imagenet_val_filelist.txt')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--ref-npz", type=str, default="ref_batch.npz")
    parser.add_argument("--sample-npz", type=str, default="sample_batch.npz")
    args = parser.parse_args()
    main(args)
