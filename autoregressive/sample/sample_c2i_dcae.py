import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from tokenizer.tokenizer_image.bsqdc_evit_model import VQModel
from tokenizer.tokenizer_image.bsqdc_model import ModelArgs
from autoregressive.models.gpt_continuous import ContinuousTransformer, ContinuousModelArgs, GPT_continuous_models


def build_dcae_decoder_from_ckpt(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    num_bits = state_dict["quant_conv.weight"].shape[0]
    z_channels = state_dict["quant_conv.weight"].shape[1]
    codebook_embed_dim = state_dict["post_quant_conv.weight"].shape[1]

    ckpt_args = checkpoint.get("args", None)
    model_name = getattr(ckpt_args, "vq_model", None) if ckpt_args else None

    if model_name == "DCAE-16":
        encoder_widths = [128, 256, 512, 512, 1024]
        decoder_widths = [128, 256, 512, 512, 1024]
        encoder_depth_list = [0, 4, 8, 2, 2]
        decoder_depth_list = [0, 5, 10, 2, 2]
        encoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"]
        decoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"]
        decoder_norm = ["bn2d", "bn2d", "bn2d", "trms2d", "trms2d"]
        decoder_act = ["relu", "relu", "relu", "silu", "silu"]
    elif model_name == "DCAE-64":
        encoder_widths = [128, 256, 512, 512, 1024, 1024, 2048]
        decoder_widths = [128, 256, 512, 512, 1024, 1024, 2048]
        encoder_depth_list = [0, 4, 8, 2, 2, 2, 2]
        decoder_depth_list = [0, 5, 10, 2, 2, 2, 2]
        encoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
        decoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
        decoder_norm = ["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d"]
        decoder_act = ["relu", "relu", "relu", "silu", "silu", "silu", "silu"]
    else:
        encoder_widths = [128, 256, 512, 512, 1024, 1024]
        decoder_widths = [128, 256, 512, 512, 1024, 1024]
        encoder_depth_list = [0, 4, 8, 2, 2, 2]
        decoder_depth_list = [0, 5, 10, 2, 2, 2]
        encoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
        decoder_block_type = ["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"]
        decoder_norm = ["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"]
        decoder_act = ["relu", "relu", "relu", "silu", "silu", "silu"]

    sample = getattr(ckpt_args, "sample", True) if ckpt_args else True
    codebook_l2_norm = getattr(ckpt_args, "codebook_l2_norm", True) if ckpt_args else True
    learnable_proj = getattr(ckpt_args, "learnable_proj", True) if ckpt_args else True
    anneal_noise = getattr(ckpt_args, "anneal_noise", False) if ckpt_args else False
    dropout_p = getattr(ckpt_args, "dropout_p", 0.0) if ckpt_args else 0.0

    config = ModelArgs(
        num_bits=num_bits,
        codebook_embed_dim=codebook_embed_dim,
        codebook_l2_norm=codebook_l2_norm,
        z_channels=z_channels,
        encoder_ch_mult=encoder_widths,
        decoder_ch_mult=decoder_widths,
        sample=sample,
        learnable_proj=learnable_proj,
        anneal_noise=anneal_noise,
        dropout_p=dropout_p,
    )
    config.encoder_depth_list = encoder_depth_list
    config.decoder_depth_list = decoder_depth_list
    config.encoder_block_type = encoder_block_type
    config.decoder_block_type = decoder_block_type
    config.decoder_norm = decoder_norm
    config.decoder_act = decoder_act

    model = VQModel(config)
    use_ema = "ema" in checkpoint
    if use_ema:
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    downsample_factor = 2 ** (len(encoder_widths) - 1)
    return model, downsample_factor, num_bits


def decode_binary_to_image(dcae_model, binary_codes, downsample_factor, num_bits, image_size):
    B = binary_codes.shape[0]
    spatial_size = image_size // downsample_factor

    z_q = binary_codes.float()
    h = z_q.reshape(B, num_bits, spatial_size, spatial_size)
    h = h * 2.0 - 1.0

    with torch.no_grad():
        dec_input = dcae_model.post_quant_conv(h)
        samples = dcae_model.decoder(dec_input)

    return samples


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@torch.no_grad()
def generate_continuous(model, cond_idx, block_size, codebook_dim, cfg_scale=1.0, temperature=1.0):
    model.eval()
    model.setup_caches(
        max_batch_size=cond_idx.shape[0] * 2 if cfg_scale > 1.0 else cond_idx.shape[0],
        max_seq_length=block_size + model.cls_token_num,
        dtype=next(model.parameters()).dtype
    )

    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond_idx) * model.num_classes
        cond_combined = torch.cat([cond_idx, cond_null])
    else:
        cond_combined = cond_idx

    bs = cond_combined.shape[0]
    input_pos = torch.arange(0, model.cls_token_num, device=cond_combined.device)
    _, _ = model(None, cond_combined, input_pos=input_pos)

    generated = []
    for i in range(block_size):
        if i == 0:
            x_prev = torch.zeros(bs, 1, codebook_dim, device=cond_combined.device)
        else:
            x_prev = token.clone().detach().unsqueeze(1)

        input_pos = torch.tensor([model.cls_token_num + i], device=cond_combined.device)
        output, _ = model(x_prev, None, input_pos=input_pos)

        if temperature != 1.0:
            output = output / temperature

        if cfg_scale > 1.0:
            cond_output = output[:bs // 2]
            uncond_output = output[bs // 2:]
            output_cfg = uncond_output + cfg_scale * (cond_output - uncond_output)
            token = torch.sign(output_cfg)
            generated.append(token)
        else:
            token = torch.sign(output[:, -1:, :])
            generated.append(token.squeeze(1))

    return torch.stack(generated, dim=1)


def main(args):
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    dcae_model, downsample_factor, num_bits = build_dcae_decoder_from_ckpt(args.vq_ckpt, device)
    print(f"DCAE decoder loaded: downsample={downsample_factor}, num_bits={num_bits}")

    spatial_size = args.image_size // downsample_factor
    block_size = spatial_size * spatial_size
    codebook_dim = num_bits

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    gpt_model = GPT_continuous_models[args.gpt_model](
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        codebook_dim=codebook_dim,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    else:
        model_weight = checkpoint
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"GPT model loaded from {args.gpt_ckpt}")

    model_string_name = args.gpt_model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-temp-{args.temperature}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    total = 0
    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)

        binary_codes = generate_continuous(
            gpt_model, c_indices, block_size, codebook_dim,
            cfg_scale=args.cfg_scale, temperature=args.temperature,
        )

        samples = decode_binary_to_image(dcae_model, binary_codes, downsample_factor, num_bits, args.image_size)

        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Sampling done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_continuous_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for DCAE model")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-size-eval", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
