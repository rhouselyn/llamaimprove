import requests
import time
import cv2
import json
import os
from tqdm import tqdm

# ===== 1. 配置信息 =====
base_url = "http://127.0.0.1:8000"
video_path = "demo.mp4"  # 输入视频
output_path = "result.mp4"  # 保存视频
proxies = {"http": None, "https": None}


def test_and_visualize():
    try:
        # ----- A. 提交并获取结果 (Job 模式) -----
        print("1. 正在提交推理任务...")
        with open(video_path, 'rb') as f:
            files = {'file': (video_path, f, 'video/mp4')}
            resp = requests.post(f"{base_url}/pose_start", files=files, proxies=proxies)

        job_id = resp.json()["job_id"]

        while True:
            status_resp = requests.get(f"{base_url}/pose_progress/{job_id}", proxies=proxies).json()
            if status_resp["status"] == "done": break
            if status_resp["status"] == "error":
                print(f"服务器报错: {status_resp['error']}")
                return
            time.sleep(1)

        result_data = requests.get(f"{base_url}/pose_result/{job_id}", proxies=proxies).json()
        print(f"✅ 推理数据下载完成，准备渲染视频...")

        # ----- B. 可视化处理模块 -----
        # 1. 将结果转为字典，方便按帧快速索引
        res_map = {item['frame']: item['targets'] for item in result_data}

        # 2. 打开原始视频获取参数
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. 初始化写入器 (使用 MP4V 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 4. 使用 tqdm 创建进度条
        with tqdm(total=total_frames, desc="视频渲染中", unit="frame") as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 检查当前帧是否有检测结果
                if frame_idx in res_map:
                    for target in res_map[frame_idx]:
                        bbox = target['bbox']  # [x1, y1, x2, y2]
                        angle = target['angle']
                        tid = target['id']

                        # 绘制矩形框 (绿色)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                        # 绘制 ID 和 角度文本 (红色)
                        label = f"ID:{tid} Angle:{angle}"
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 播放预览
                cv2.imshow("Gemini Enterprise - Pose Viewer", frame)

                # 写入视频文件
                out.write(frame)

                # 更新进度
                pbar.update(1)
                frame_idx += 1

                # 按 'q' 可以提前退出预览
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 5. 资源释放
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n🎉 视频处理完成！已保存至: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"程序运行异常: {e}")


if __name__ == "__main__":
    test_and_visualize()
