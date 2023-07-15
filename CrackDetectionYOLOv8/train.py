from ultralytics import YOLO
import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = REPO_PATH + '/crack.v2i.yolov8/data.yaml'

# Load a model
model = YOLO('yolov8l-seg.pt')

# Train the model
model.train(data=DATA_YAML, epochs=10, imgsz=640)