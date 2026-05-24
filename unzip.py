import os
import zipfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

ZIP_FILE = "../imagenet/imagenet-object-localization-challenge.zip"
TARGET_DIR = os.getcwd()   # 当前目录

def unzip_worker(args):
    zip_path, extract_to, member = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(member, extract_to)
    except Exception:
        return member
    return None

def extract_only(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ 找不到压缩包: {zip_path}")

    print(f"📦 开始解压: {zip_path}")
    print(f"📂 解压到: {TARGET_DIR}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        print("📑 读取文件列表（可能较慢）...")
        members = zf.namelist()
        print(f"📑 文件总数: {len(members)}")

        cores = min(cpu_count(), 16)
        print(f"⚙️  使用 {cores} 个进程并行解压")

        args = [(zip_path, TARGET_DIR, m) for m in members]

        with Pool(cores) as p:
            list(
                tqdm(
                    p.imap_unordered(unzip_worker, args),
                    total=len(args),
                    desc="解压进度"
                )
            )

    print("✅ 解压完成")

if __name__ == "__main__":
    extract_only(ZIP_FILE)
