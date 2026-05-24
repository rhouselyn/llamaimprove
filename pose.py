import cv2
import numpy as np
import os
import time
from rtmlib import Wholebody, draw_skeleton


def main():
    # 1. 配置参数
    device = "cpu"  # "cpu", "cuda", "mps"
    backend = "onnxruntime"  # "opencv", "onnxruntime", "openvino"

    img_path = "demo.jpg"

    # 2. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(f"无法读取图片: {img_path}。请确认文件位于: {current_dir}")

    # 3. 初始化模型（并计时）
    print(f"正在加载模型 (Backend: {backend}, Device: {device})...")
    t_start_load = time.time()

    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",
        backend=backend,
        device=device,
    )

    t_end_load = time.time()
    load_time = t_end_load - t_start_load
    print(f"✅ 模型加载完成，耗时: {load_time:.4f} 秒")

    # 4. 推理（并计时）
    print("正在进行推理...")
    t_start_infer = time.time()

    keypoints, scores = wholebody(img)

    t_end_infer = time.time()
    infer_time = t_end_infer - t_start_infer
    print(f"⚡ 推理完成，耗时: {infer_time:.4f} 秒")

    # 5. 可视化绘制
    img_show = img.copy()
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

    # 6. 保存图片
    output_filename = "result.jpg"
    cv2.imwrite(output_filename, img_show)

    output_abs_path = os.path.abspath(output_filename)
    print(f"💾 结果已保存至: {output_abs_path}")


if __name__ == "__main__":
    main()
