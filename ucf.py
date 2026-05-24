import os
import subprocess
import shutil


def run_command(cmd, step_name):
    print(f"\n{'=' * 50}")
    print(f"⏳ 正在执行: {step_name}")
    print(f"💻 命令: {cmd}")
    print(f"{'=' * 50}")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ {step_name} 失败！(退出码: {result.returncode})")
        print("请检查网络，或尝试手动在终端运行上述命令。")
        exit(1)
    else:
        print(f"✅ {step_name} 成功！\n")


def check_unrar():
    """检查系统是否安装了 unrar"""
    if not shutil.which("unrar"):
        print("\n⚠️ 注意：你的系统似乎没有安装 `unrar` 命令，解压过程可能会失败。")
        print("如果稍后解压报错，请先在终端执行以下命令安装：")
        print("对于 CentOS/TencentOS:  sudo yum install epel-release -y && sudo yum install unrar -y")
        print("对于 Ubuntu/Debian:    sudo apt-get install unrar -y\n")


def download_and_extract_ucf101(target_dir):
    os.makedirs(target_dir, exist_ok=True)

    # 官方使用的是 RAR 格式
    video_rar = os.path.join(target_dir, "UCF101.rar")
    splits_zip = os.path.join(target_dir, "UCF101TrainTestSplits-RecognitionTask.zip")

    # 清理刚才下载失败的残留文件（避免冲突）
    bad_zip = os.path.join(target_dir, "UCF101.zip")
    if os.path.exists(bad_zip):
        os.remove(bad_zip)

    # 官方直接下载链接（忽略证书验证保证稳定）
    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    splits_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    # wget 命令
    cmd_download_video = f'wget -c -O "{video_rar}" --no-check-certificate "{video_url}"'
    cmd_download_splits = f'wget -c -O "{splits_zip}" --no-check-certificate "{splits_url}"'

    # 解压命令（rar 使用 unrar x，zip 使用 unzip）
    # unrar 的 -o+ 表示覆盖已存在文件，-inul 表示静默模式避免刷屏
    cmd_unrar_video = f'unrar x -o+ -inul "{video_rar}" "{target_dir}/"'
    cmd_unzip_splits = f'unzip -q -o "{splits_zip}" -d "{target_dir}"'

    check_unrar()

    # 按顺序执行
    run_command(cmd_download_video, "下载 UCF101 官方视频数据 (约 6.5GB，支持断点续传)")
    run_command(cmd_unrar_video, "解压 UCF101 视频 RAR 文件 (耗时较长，请耐心等待...)")

    run_command(cmd_download_splits, "下载训练/测试集标注分割文件")
    run_command(cmd_unzip_splits, "解压标注分割文件")

    print(f"🎉 全部流程执行完毕！数据集准备就绪: {target_dir}")


if __name__ == "__main__":
    path = "/mnt/afs/zhengmingkai/whl/ucf"
    download_and_extract_ucf101(path)
