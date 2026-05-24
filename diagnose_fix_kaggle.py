import os
import sys
import shutil

# ================= 配置区域 =================
# 你指定的 kaggle.json 路径
KAGGLE_JSON_PATH = '/mnt/afs/zhengmingkai/whl/llamagen/kaggle.json'
# 你想要的下载目录
DOWNLOAD_DIR = '/mnt/afs/zhengmingkai/whl/imagenet'
# ===========================================

# --- 关键修复：在导入 kaggle 之前先处理环境 ---
print(f"Checking credentials at: {KAGGLE_JSON_PATH}")

if not os.path.exists(KAGGLE_JSON_PATH):
    print(f"Error: 文件不存在 -> {KAGGLE_JSON_PATH}")
    sys.exit(1)

# 1. 临时规避权限问题 (针对 AFS/NFS 无法 chmod 600 的情况)
# Kaggle 强制要求 json 文件权限为 600。在网络存储上 chmod 可能会失败或被忽略。
# 我们把 key 复制到本地临时目录来欺骗它。
user_home = os.path.expanduser("~")
temp_config_dir = os.path.join(user_home, ".kaggle_temp_fix")

if not os.path.exists(temp_config_dir):
    os.makedirs(temp_config_dir, exist_ok=True)

temp_json_path = os.path.join(temp_config_dir, "kaggle.json")
shutil.copy(KAGGLE_JSON_PATH, temp_json_path)

# 尝试修改临时文件的权限为 600 (Owner Read/Write Only)
try:
    os.chmod(temp_json_path, 0o600)
    print(f"权限已修正 (local temp): {temp_json_path}")
except Exception as e:
    print(f"警告: 无法修改临时文件权限: {e}")

# 2. 强制设置环境变量 (必须在 import kaggle 之前!)
os.environ['KAGGLE_CONFIG_DIR'] = temp_config_dir

print(f"环境变量 KAGGLE_CONFIG_DIR 已设置为: {os.environ['KAGGLE_CONFIG_DIR']}")

# --- 环境设置完毕，现在才开始导入其他库 ---
import zipfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: 找不到 kaggle 库，请运行: pip install kaggle")
    sys.exit(1)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_and_extract():
    ensure_dir(DOWNLOAD_DIR)

    print("\n[Step 1] 连接 Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()  # 这次应该能找到 temp_config_dir 下的 json 了
        print("认证成功！")
    except Exception as e:
        print(f"认证失败: {e}")
        print(f"请检查 json 内容是否正确，或者网络是否需要代理。")
        return

    print(f"\n[Step 2] 开始下载 ILSVRC 2012 到 {DOWNLOAD_DIR} ...")
    try:
        # 调用 API 下载
        api.competition_download_files(
            'imagenet-object-localization-challenge',
            path=DOWNLOAD_DIR,
            quiet=False
        )
    except Exception as e:
        print(f"下载出错: {e}")
        return

    zip_file = os.path.join(DOWNLOAD_DIR, "imagenet-object-localization-challenge.zip")
    if not os.path.exists(zip_file):
        print("下载似乎完成了，但没找到 zip 文件，请检查目录。")
        return

    print(f"\n[Step 3] 多进程解压: {zip_file}")
    extract_parallel(zip_file, DOWNLOAD_DIR)

    print("\n[Step 4] 整理目录结构...")
    restructure_dataset(DOWNLOAD_DIR)

    # 清理临时 key
    shutil.rmtree(temp_config_dir, ignore_errors=True)
    print("\nDone! 脚本运行结束。")


def unzip_worker(args):
    zip_path, extract_to, member = args
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract(member, extract_to)


def extract_parallel(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [m for m in zf.namelist() if m.startswith("ILSVRC/Data/CLS-LOC")]
        if not members: members = zf.namelist()  # fallback

        # 限制最大进程数防止内存溢出，一般 16-32 够了，除非你 CPU 核心非常多
        cores = min(cpu_count(), 32)
        print(f"使用 {cores} 个核心进行解压...")

        args = [(zip_path, extract_to, m) for m in members]
        with Pool(cores) as p:
            list(tqdm(p.imap_unordered(unzip_worker, args), total=len(args)))


def restructure_dataset(base_dir):
    # 处理 train
    src_train = os.path.join(base_dir, "ILSVRC/Data/CLS-LOC/train")
    dst_train = os.path.join(base_dir, "train")
    if os.path.exists(src_train):
        if os.path.exists(dst_train): shutil.rmtree(dst_train)
        shutil.move(src_train, dst_train)

    # 处理 val
    src_val = os.path.join(base_dir, "ILSVRC/Data/CLS-LOC/val")
    dst_val = os.path.join(base_dir, "val")
    if os.path.exists(src_val):
        if os.path.exists(dst_val): shutil.rmtree(dst_val)
        shutil.move(src_val, dst_val)

    shutil.rmtree(os.path.join(base_dir, "ILSVRC"), ignore_errors=True)


if __name__ == "__main__":
    download_and_extract()
