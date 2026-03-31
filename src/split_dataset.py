import os
import random
import shutil

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_ROOT = os.path.join(BASE_DIR, "data", "yolo_text_detection")

ALL_IMAGE_DIR = os.path.join(YOLO_ROOT, "all_images")
ALL_LABEL_DIR = os.path.join(YOLO_ROOT, "all_labels")

TRAIN_IMAGE_DIR = os.path.join(YOLO_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(YOLO_ROOT, "train", "labels")
VAL_IMAGE_DIR = os.path.join(YOLO_ROOT, "val", "images")
VAL_LABEL_DIR = os.path.join(YOLO_ROOT, "val", "labels")

for folder in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR]:
    os.makedirs(folder, exist_ok=True)

image_files = [
    f for f in os.listdir(ALL_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_pair(file_list, image_dst, label_dst):
    for img_name in file_list:
        stem = os.path.splitext(img_name)[0]
        label_name = stem + ".txt"

        img_src = os.path.join(ALL_IMAGE_DIR, img_name)
        label_src = os.path.join(ALL_LABEL_DIR, label_name)

        if not os.path.exists(label_src):
            continue

        shutil.copy(img_src, os.path.join(image_dst, img_name))
        shutil.copy(label_src, os.path.join(label_dst, label_name))

copy_pair(train_files, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
copy_pair(val_files, VAL_IMAGE_DIR, VAL_LABEL_DIR)

print("Tổng ảnh:", len(image_files))
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Đã chia train/val xong.")