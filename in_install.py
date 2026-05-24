import os
import sys
import shutil
import zipfile
import subprocess
import threading
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 你的配置 =================
KAGGLE_JSON_ORIGIN = '/mnt/afs/zhengmingkai/whl/llamagen/kaggle.json'
TARGET_DIR = '/mnt/afs/zhengmingkai/whl/imagenet'
REQUIRED_GB = 350


# ===========================================

def force_print(msg):
    """强制刷新打印，确保云端日志能实时看到"""
    print(msg, flush=True)


def check_disk_space(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    stat = os.statvfs(path)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    force_print(f"📊 [磁盘检查] 当前剩余空间: {free_gb:.1f} GB")

    if free_gb < REQUIRED_GB:
        force_print(f"⚠️  警告: 剩余空间 ({free_gb:.1f} GB) 少于建议的 {REQUIRED_GB}GB。")
        # 云端任务严禁使用 input()，改为直接报错退出以保护环境
        force_print("🛑 错误：磁盘空间不足，为防止解压失败，脚本自动停止。")
        sys.exit(1)


def setup_auth():
    """配置认证环境"""
    if not os.path.exists(KAGGLE_JSON_ORIGIN):
        force_print(f"❌ 错误: 找不到 kaggle.json 凭证于 {KAGGLE_JSON_ORIGIN}")
        sys.exit(1)

    user_home = os.path.expanduser("~")
    temp_dir = os.path.join(user_home, ".kaggle_temp_auth")
    os.makedirs(temp_dir, exist_ok=True)
    temp_json = os.path.join(temp_dir, "kaggle.json")

    shutil.copy(KAGGLE_JSON_ORIGIN, temp_json)
    os.chmod(temp_json, 0o600)
    os.environ['KAGGLE_CONFIG_DIR'] = temp_dir
    force_print(f"🔑 [认证] 凭证已就绪: {temp_dir}")
    return temp_dir


def monitor_download(filepath, stop_event):
    """后台线程：监控文件大小变化，解决日志看起来卡死的问题"""
    force_print(f"⏳ [监控启动] 正在等待文件创建: {os.path.basename(filepath)} ...")

    start_time = time.time()
    while not stop_event.is_set():
        if os.path.exists(filepath):
            try:
                size_bytes = os.path.getsize(filepath)
                size_gb = size_bytes / (1024 ** 3)
                elapsed = time.time() - start_time
                # 每30秒打印一次进度
                force_print(f"⬇️  [下载中] 已耗时 {elapsed / 60:.1f}分 | 当前大小: {size_gb:.2f} GB")
            except Exception:
                pass
        else:
            # 文件还没生成，打印等待信息
            pass

        time.sleep(30)  # 30秒刷新一次


def download_data():
    """下载函数（带监控）"""
    force_print("\n🚀 [Step 1/3] 开始下载 ImageNet (ILSVRC 2012)...")

    # 1. 智能查找 kaggle 命令路径
    kaggle_bin = os.path.join(sys.prefix, 'bin', 'kaggle')
    if not os.path.exists(kaggle_bin):
        force_print(f"⚠️  警告: 在 {kaggle_bin} 未找到 kaggle，尝试使用系统默认路径 'kaggle'")
        kaggle_bin = "kaggle"
    else:
        force_print(f"🔍 [Debug] 使用 Kaggle 路径: {kaggle_bin}")

    zip_filename = "imagenet-object-localization-challenge.zip"
    zip_path = os.path.join(TARGET_DIR, zip_filename)

    # 2. 启动监控线程
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_download, args=(zip_path, stop_event))
    monitor_thread.daemon = True  # 设置为守护线程，主程序挂了它也会自动挂
    monitor_thread.start()

    # 3. 执行下载命令
    cmd = [
        kaggle_bin, "competitions", "download",
        "-c", "imagenet-object-localization-challenge",
        "-p", TARGET_DIR
    ]

    try:
        # 这里的 stdout 可能会被云平台缓存，但我们的监控线程会绕过它直接打印
        result = subprocess.run(cmd, check=True)

        # 下载完成，停止监控
        stop_event.set()
        monitor_thread.join(timeout=2)

        if result.returncode == 0:
            force_print(f"\n✅ 下载成功完成！最终文件位置: {zip_path}")

    except subprocess.CalledProcessError as e:
        stop_event.set()
        force_print(f"\n❌ 下载命令执行失败 (Exit Code: {e.returncode})。")
        force_print("💡 请检查：网络连接、磁盘空间或 Kaggle API Token 是否有效。")
        sys.exit(1)
    except Exception as e:
        stop_event.set()
        force_print(f"\n❌ 发生未知错误: {str(e)}")
        sys.exit(1)

    return zip_path


def unzip_worker(args):
    """解压单个文件的 Worker"""
    zip_path, extract_to, member = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(member, extract_to)
    except Exception:
        return member  # 返回失败的文件名
    return None


def extract_and_organize(zip_path):
    force_print(f"\n📦 [Step 2/3] 准备解压...")

    if not os.path.exists(zip_path):
        force_print(f"❌ 错误: 找不到压缩文件 {zip_path}")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            force_print("📑 正在读取压缩包文件列表 (这可能需要几分钟)...")
            all_files = zf.namelist()
            target_members = [f for f in all_files if "ILSVRC/Data/CLS-LOC" in f]

            if not target_members:
                force_print("⚠️  未发现标准路径，将进行全量解压...")
                target_members = all_files

            force_print(f"📑 待解压文件总数: {len(target_members)}")

            # 使用进程池加速
            cores = min(cpu_count(), 16)
            args = [(zip_path, TARGET_DIR, m) for m in target_members]

            force_print(f"⚙️  启动 {cores} 个进程并行解压...")
            with Pool(cores) as p:
                # 注意：tqdm 在某些云端日志中可能会乱码，这里只做简单的进度显示
                list(tqdm(p.imap_unordered(unzip_worker, args), total=len(args), desc="解压进度"))

    except zipfile.BadZipFile:
        force_print("❌ 错误: 压缩包已损坏 (Bad Zip File)。建议删除该文件并重新运行脚本。")
        sys.exit(1)

    force_print("\n📂 [Step 3/3] 正在整理目录结构...")
    src_root = os.path.join(TARGET_DIR, "ILSVRC", "Data", "CLS-LOC")

    for folder in ["train", "val"]:
        src = os.path.join(src_root, folder)
        dst = os.path.join(TARGET_DIR, folder)
        if os.path.exists(src):
            if os.path.exists(dst): shutil.rmtree(dst)
            shutil.move(src, dst)
            force_print(f" ✨ {folder} 文件夹已整理至顶层")

    # 清理无用的中间目录
    shutil.rmtree(os.path.join(TARGET_DIR, "ILSVRC"), ignore_errors=True)
    force_print("🧹 临时目录清理完毕。")


if __name__ == "__main__":
    # 确认空间
    check_disk_space(TARGET_DIR)

    temp_auth_dir = None
    try:
        temp_auth_dir = setup_auth()

        # 1. 下载 (带实时监控)
        zip_file = download_data()

        # 2. 解压与整理
        if os.path.exists(zip_file):
            extract_and_organize(zip_file)

        force_print("\n✅ 所有任务已完成！ImageNet 数据集现在可以使用了。")

    except KeyboardInterrupt:
        force_print("\n\n🛑 用户手动停止了任务。")
    finally:
        if temp_auth_dir and os.path.exists(temp_auth_dir):
            shutil.rmtree(temp_auth_dir)
