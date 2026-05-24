FROM ultralytics/ultralytics:latest

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models/yolov8s-pose.pt /models/yolov8s-pose.pt

ENV MODEL_PATH=/models/yolov8s-pose.pt
ENV TMP_DIR=/tmp
ENV JOB_TTL_SEC=3600

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
