import argparse
import os
import sys
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="GPT checkpoint path")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="DCAE checkpoint path")
    parser.add_argument("--gpt-model", type=str, default="GPT-B")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--ref-batch", type=str, default=None,
                        help="Path to reference batch .npz for FID computation")
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Generating samples with DCAE continuous model")
    print("=" * 60)

    sample_cmd = [
        "torchrun",
        f"--nnodes=1",
        f"--nproc_per_node={args.num_gpus}",
        "--node_rank=0",
        "--master_port=12350",
        "autoregressive/sample/sample_c2i_dcae.py",
        "--gpt-model", args.gpt_model,
        "--gpt-ckpt", args.gpt_ckpt,
        "--vq-ckpt", args.vq_ckpt,
        "--image-size", str(args.image_size),
        "--num-classes", str(args.num_classes),
        "--cfg-scale", str(args.cfg_scale),
        "--temperature", str(args.temperature),
        "--num-fid-samples", str(args.num_fid_samples),
        "--per-proc-batch-size", str(args.per_proc_batch_size),
        "--sample-dir", args.sample_dir,
        "--global-seed", str(args.global_seed),
    ]
    print(f"Running: {' '.join(sample_cmd)}")
    subprocess.run(sample_cmd, check=True)

    import glob
    sample_dirs = sorted(glob.glob(f"{args.sample_dir}/*"))
    if not sample_dirs:
        print("ERROR: No sample directory found!")
        return
    latest_sample_dir = sample_dirs[-1]
    sample_npz = f"{latest_sample_dir}.npz"

    if not os.path.exists(sample_npz):
        print(f"ERROR: Sample npz not found at {sample_npz}")
        return

    print()
    print("=" * 60)
    print("Step 2: Computing FID/IS metrics")
    print("=" * 60)

    if args.ref_batch and os.path.exists(args.ref_batch):
        eval_cmd = [
            sys.executable, "evaluations/c2i/evaluator.py",
            args.ref_batch, sample_npz,
        ]
        print(f"Running: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)
    else:
        print("WARNING: No reference batch provided. Skipping FID/IS computation.")
        print(f"Sample npz saved at: {sample_npz}")
        print("To compute FID/IS, prepare a reference batch and run:")
        print(f"  python evaluations/c2i/evaluator.py <ref_batch.npz> {sample_npz}")

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print(f"Samples saved at: {latest_sample_dir}")
    print(f"NPZ file: {sample_npz}")
    print("=" * 60)


if __name__ == "__main__":
    main()
