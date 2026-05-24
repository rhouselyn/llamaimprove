import os
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from ultralytics import YOLO
from app.pose_tracker import PoseTracker

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/models/yolov8s-pose.pt")
TMP_DIR = os.getenv("TMP_DIR", "/tmp")

# ---- 生产建议：单 GPU 单任务（避免 track persist 状态串 & GPU OOM）
INFER_LOCK = threading.Lock()

# ---- 预加载模型：避免每个请求重复加载权重（节省大量时间）
# 注意：因为我们用 INFER_LOCK 串行推理，复用同一个 YOLO 实例是安全的（不会并发串状态）。
MODEL = YOLO(MODEL_PATH)

# ---- job 存储（轻量兜底；如果你确认平台长连接稳定，可不用 job 接口）
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SEC = int(os.getenv("JOB_TTL_SEC", "3600"))  # 1小时自动过期


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _cleanup_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired = [k for k, v in JOBS.items() if now - v.get("ts", now) > JOB_TTL_SEC]
        for k in expired:
            JOBS.pop(k, None)


@app.get("/health")
def health():
    return {"status": "ok"}


# ========== A) SSE：单请求实时返回进度 + 最终结果 ==========
@app.post("/pose_sse")
async def pose_sse(
    file: UploadFile = File(...),
    frame_stride: int = 1,
    resize_width: Optional[int] = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{suffix}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    def gen():
        progress: Dict[str, Any] = {"status": "queued", "pct": None, "frame": 0, "total": None}
        done_flag = {"done": False}
        result_box: Dict[str, Any] = {"result": None, "error": None}

        def on_progress(p: dict):
            progress.update(p)

        def worker():
            try:
                with INFER_LOCK:
                    tracker = PoseTracker(MODEL)
                    data = tracker.get_pose_data(
                        tmp_path,
                        on_progress=on_progress,
                        progress_every_n_frames=10,
                        frame_stride=max(1, int(frame_stride)),
                        resize_width=resize_width,
                    )
                result_box["result"] = data
            except Exception as e:
                result_box["error"] = str(e)
            finally:
                done_flag["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        try:
            yield _sse("start", {"msg": "started"})
            last_emit = 0.0
            # 主线程持续推送进度
            while not done_flag["done"]:
                now = time.time()
                # 每 0.2 秒推一次（可按需调整）
                if now - last_emit >= 0.2:
                    yield _sse("progress", progress)
                    last_emit = now
                time.sleep(0.05)

            if result_box["error"]:
                yield _sse("error", {"message": result_box["error"], "progress": progress})
            else:
                yield _sse("progress", progress)
                yield _sse("done", {"result": result_box["result"]})

        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# ========== B) Job 模式（兜底：start -> progress -> result） ==========
@app.post("/pose_start")
async def pose_start(
    file: UploadFile = File(...),
    frame_stride: int = 1,
    resize_width: Optional[int] = None,
):
    _cleanup_jobs()

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    os.makedirs(TMP_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex
    tmp_path = os.path.join(TMP_DIR, f"{job_id}{suffix}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "ts": time.time(),
            "status": "queued",
            "progress": {"status": "queued", "pct": None, "frame": 0, "total": None},
            "result": None,
            "error": None,
        }

    def on_progress(p: dict):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = p
                JOBS[job_id]["status"] = p.get("status", "processing")

    def worker():
        try:
            with INFER_LOCK:
                tracker = PoseTracker(MODEL)
                data = tracker.get_pose_data(
                    tmp_path,
                    on_progress=on_progress,
                    progress_every_n_frames=10,
                    frame_stride=max(1, int(frame_stride)),
                    resize_width=resize_width,
                )
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["result"] = data
                    JOBS[job_id]["status"] = "done"
        except Exception as e:
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["error"] = str(e)
                    JOBS[job_id]["status"] = "error"
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/pose_progress/{job_id}")
def pose_progress(job_id: str):
    _cleanup_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {"status": job["status"], "progress": job["progress"], "error": job["error"]}


@app.get("/pose_result/{job_id}")
def pose_result(job_id: str):
    _cleanup_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] == "done":
            return JSONResponse(content=job["result"])
        if job["status"] == "error":
            raise HTTPException(status_code=500, detail=job["error"] or "unknown error")
        raise HTTPException(status_code=425, detail="result not ready")
