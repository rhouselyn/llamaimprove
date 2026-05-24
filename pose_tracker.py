import cv2
import torch
import numpy as np
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import time


# class FastPoseTracker:
#     def __init__(self, model_path: str, device: str = 'cuda'):
#         # 1. 预加载模型到 GPU，使用半精度 (FP16) 提升一倍速度
#         self.model = YOLO(model_path).to(device)
#         if device == 'cuda':
#             self.model.model.half()
#         self.device = device
#
#     def _stream_reader(self, video_path: str, frame_queue: Queue, stride: int):
#         """独立线程：负责视频解码"""
#         cap = cv2.VideoCapture(video_path)
#         frame_idx = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_idx % stride == 0:
#                 # 预先在 CPU 上缩小，减少后续传输到 GPU 的带宽压力
#                 # 这里的 640 是为了降低传输负载，YOLO 内部还会处理 imgsz
#                 frame = cv2.resize(frame, (640, int(frame.shape[0] * (640 / frame.shape[1]))))
#                 frame_queue.put((frame_idx, frame))
#             frame_idx += 1
#         cap.release()
#         frame_queue.put((None, None))  # 结束标志
#
#     def process(self, video_path: str, frame_stride: int = 3):
#         frame_queue = Queue(maxsize=30)  # 缓冲队列
#         reader_thread = Thread(target=self._stream_reader, args=(video_path, frame_queue, frame_stride))
#         reader_thread.start()
#
#         results_list = []
#
#         while True:
#             idx, frame = frame_queue.get()
#             if frame is None:
#                 break
#
#             # 2. 推理优化：
#             # stream=True 使用生成器模式减少内存占用
#             # persist=True 保持 ID
#             # imgsz 控制推理分辨率
#             # half=True 开启半精度推理
#             results = self.model.track(
#                 source=frame,
#                 persist=True,
#                 classes=[0],
#                 imgsz=640,
#                 conf=0.4,  # 过滤低置信度
#                 verbose=False,
#                 half=(self.device == 'cuda')
#             )
#
#             res = results[0]
#             if res.boxes is None or res.boxes.id is None:
#                 results_list.append({"frame": idx, "targets": []})
#                 continue
#
#             # 3. 向量化提取数据 (摒弃 for 循环)
#             # 直接在 GPU/Tensor 层面操作，最后一次性转为 numpy
#             ids = res.boxes.id.int().cpu().numpy()
#             boxes = res.boxes.xyxy.cpu().numpy()
#             kpts = res.keypoints.data  # [N, 17, 3] (x, y, conf)
#
#             # 计算肩膀中心 (index 5, 6) 和 髋部中心 (index 11, 12)
#             # 使用 Tensor 操作避免 Python 循环加速
#             sh_c = (kpts[:, 5, :2] + kpts[:, 6, :2]) / 2.0
#             hp_c = (kpts[:, 11, :2] + kpts[:, 12, :2]) / 2.0
#
#             # 向量化计算角度
#             diff = sh_c - hp_c
#             angles = torch.atan2(diff[:, 0], -diff[:, 1]) * (180.0 / np.pi)
#             angles = angles.cpu().numpy()
#
#             targets = []
#             for i in range(len(ids)):
#                 targets.append({
#                     "id": int(ids[i]),
#                     "bbox": boxes[i].tolist(),
#                     "angle": round(float(angles[i]), 2)
#                 })
#
#             results_list.append({"frame": idx, "targets": targets})
#
#         reader_thread.join()
#         return results_list

from ultralytics import YOLO
import cv2
import math
from typing import List, Dict, Union, Callable, Optional, Any

ProgressCb = Callable[[dict], None]


class PoseTracker:
    def __init__(self):
        # 注意：重新自行创建model用于测试
        model = YOLO(r"C:\Users\14815\Desktop\便携常驻\DynoTop\trace\models\yolov8s-pose.pt")
        self.model = model

    def get_pose_data(
        self,
        video_input: Union[str, cv2.VideoCapture],
        on_progress: Optional[ProgressCb] = None,
        progress_every_n_frames: int = 10,
        frame_stride: int = 3,          # 抽帧：每 frame_stride 帧处理 1 帧
        resize_width: Optional[int] = 640,  # 加速推理
    ) -> List[Dict[str, Any]]:

        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor.trackers = None

        if isinstance(video_input, str):
            cap = cv2.VideoCapture(video_input)
            should_close = True
        else:
            cap = video_input
            should_close = False

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

        results_list: List[Dict[str, Any]] = []
        frame_idx = 0
        processed = 0

        def emit(status: str, extra: dict):
            if not on_progress:
                return
            pct = None
            if total > 0:
                # 注意：抽帧时，进度仍然按原始帧序估算更直观
                pct = min(100.0, (frame_idx / total) * 100.0)
            payload = {
                "status": status,
                "pct": pct,
                "frame": frame_idx,
                "total": total if total > 0 else None,
                "fps": fps if fps > 0 else None,
                **extra,
            }
            on_progress(payload)

        emit("start", {})

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 抽帧：跳过不处理的帧，但仍然计入 frame_idx（进度自然）
            if frame_stride > 1 and (frame_idx % frame_stride != 0):
                frame_idx += 1
                continue

            if resize_width and resize_width > 0:
                h, w = frame.shape[:2]
                if w > resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

            # 关键：track(persist=True) 用于跨帧 ID 跟踪
            results = self.model.track(frame, persist=True, classes=[0], verbose=False)

            frame_data = {"frame": frame_idx, "targets": []}

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = results[0].keypoints.data.cpu().numpy()

                for i in range(len(track_ids)):
                    kpts = keypoints[i]
                    sh_c = [(kpts[5][0] + kpts[6][0]) / 2, (kpts[5][1] + kpts[6][1]) / 2]
                    hp_c = [(kpts[11][0] + kpts[12][0]) / 2, (kpts[11][1] + kpts[12][1]) / 2]
                    angle = math.degrees(math.atan2(sh_c[0] - hp_c[0], hp_c[1] - sh_c[1]))

                    frame_data["targets"].append({
                        "id": int(track_ids[i]),
                        "bbox": boxes[i].tolist(),
                        "angle": round(angle, 2)
                    })

            results_list.append(frame_data)
            processed += 1

            if on_progress and processed % progress_every_n_frames == 0:
                emit("processing", {"processed": processed})

            frame_idx += 1

        emit("done", {"processed": processed})

        if should_close:
            cap.release()

        return results_list


# --- 使用示例 ---
if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = YOLO(r"C:\Users\14815\Desktop\便携常驻\DynoTop\trace\models\yolov8s-pose.pt")

    tracker = PoseTracker()  # 建议用 n 或 s 模型跑生产
    start_time = time.time()
    data = tracker.get_pose_data("demo.mp4")
    print(f"处理完成，耗时: {time.time() - start_time:.2f}s")
