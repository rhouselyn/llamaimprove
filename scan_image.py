import os
from PIL import Image
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

# 配置路径
DATA_DIR = "/mnt/afs/zhengmingkai/whl/llamagen/ILSVRC/Data/CLS-LOC/train"
LOG_FILE = "corrupted_images.txt"


def check_image(img_path):
    """检查单张图片是否能被正常打开并验证"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # 检查文件是否损坏，不解码像素，速度快
        return None
    except Exception:
        return str(img_path)


def main():
    # 递归获取所有图片路径
    print(f"正在索引目录: {DATA_DIR} ...")
    image_paths = list(Path(DATA_DIR).rglob("*.JPEG"))
    print(f"共发现 {len(image_paths)} 张图片。")

    # 使用所有 CPU 核心并行扫描
    num_workers = cpu_count()
    print(f"启动 {num_workers} 个并行进程进行扫描...")

    corrupted_list = []

    with Pool(num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(pool.imap(check_image, image_paths), total=len(image_paths)))

        # 收集结果
        corrupted_list = [r for r in results if r is not None]

    # 输出并保存结果
    if corrupted_list:
        print(f"\n扫描完成！共发现 {len(corrupted_list)} 张坏图。")
        with open(LOG_FILE, "w") as f:
            for path in corrupted_list:
                f.write(path + "\n")
        print(f"坏图列表已保存至: {LOG_FILE}")

        # 可选：询问是否直接删除
        # for path in corrupted_list: os.remove(path)
    else:
        print("\n扫描完成，未发现损坏图片。")


if __name__ == "__main__":
    main()
