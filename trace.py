from ultralytics import YOLO
import cv2
import math
from typing import List, Dict, Union


class PoseTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def get_pose_data(self, video_input: Union[str, cv2.VideoCapture]) -> List[Dict]:
        """
        纯粹的分析方法
        输入：视频路径(str) 或 cv2.VideoCapture 对象
        输出：[{'frame': 0, 'targets': [{'id': 1, 'bbox': [x1,y1,x2,y2], 'angle': 0.0}]}]
        """
        # 判断输入类型：如果是路径则打开，如果是对象则直接使用
        if isinstance(video_input, str):
            cap = cv2.VideoCapture(video_input)
            should_close = True
        else:
            cap = video_input
            should_close = False

        results_list = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, classes=[0], verbose=False)
            frame_data = {"frame": frame_idx, "targets": []}

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = results[0].keypoints.data.cpu().numpy()

                for i in range(len(track_ids)):
                    kpts = keypoints[i]
                    # 计算肩膀中心和髋部中心
                    sh_c = [(kpts[5][0] + kpts[6][0]) / 2, (kpts[5][1] + kpts[6][1]) / 2]
                    hp_c = [(kpts[11][0] + kpts[12][0]) / 2, (kpts[11][1] + kpts[12][1]) / 2]

                    # 角度计算：左正右负，垂直为0
                    angle = math.degrees(math.atan2(sh_c[0] - hp_c[0], hp_c[1] - sh_c[1]))

                    frame_data["targets"].append({
                        "id": int(track_ids[i]),
                        "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                        "angle": round(angle, 2)
                    })

            results_list.append(frame_data)
            frame_idx += 1

        if should_close:
            cap.release()

        return results_list


# --- 外部调用逻辑示例 ---
if __name__ == "__main__":
    tracker = PoseTracker('yolov8s-pose.pt')

    # 方式 A：直接传路径
    # data = tracker.get_pose_data("test.mp4")

    # 方式 B：传视频对象（方便你在外部预先处理视频，如设置分辨率等）
    my_video = cv2.VideoCapture("E:/ultralytics/ultralytics-8.3.55/view.mp4")
    data = tracker.get_pose_data(my_video)
    my_video.release()  # 外部打开，外部负责关闭
    print(data)