from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "data", "yolo_text_detection", "data.yaml")

model = YOLO("yolo11n.pt")   # hoặc yolo11s.pt / yolo11m.pt

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    cache=True,
    project="runs/text_detection",
    name="yolo11_text",
    patience=20,
    plots=True
)

print("Train xong.")