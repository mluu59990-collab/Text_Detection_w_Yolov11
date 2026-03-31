import os
import cv2
import shutil
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_ROOT = os.path.join(BASE_DIR, "data", "raw", "img")
XML_PATH = os.path.join(BASE_DIR, "data", "raw", "annotation_xml", "words.xml")

YOLO_ROOT = os.path.join(BASE_DIR, "data", "yolo_text_detection")
ALL_IMAGES_DIR = os.path.join(YOLO_ROOT, "all_images")
ALL_LABELS_DIR = os.path.join(YOLO_ROOT, "all_labels")

os.makedirs(ALL_IMAGES_DIR, exist_ok=True)
os.makedirs(ALL_LABELS_DIR, exist_ok=True)


def yolo_box_from_xywh(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return x_center, y_center, bw, bh


print("Đang đọc XML:", XML_PATH)
tree = ET.parse(XML_PATH)
root = tree.getroot()

image_count = 0
box_count = 0
missing_count = 0

for image_node in root.findall("image"):
    image_name_node = image_node.find("imageName")
    resolution_node = image_node.find("resolution")
    tagged_rectangles_node = image_node.find("taggedRectangles")

    if image_name_node is None or resolution_node is None or tagged_rectangles_node is None:
        continue

    rel_image_path = image_name_node.text.strip()   # ví dụ: apanar_06.08.2002/IMG_1261.JPG
    src_image_path = os.path.join(IMG_ROOT, rel_image_path)

    if not os.path.exists(src_image_path):
        print(f"[WARNING] Không tìm thấy ảnh: {src_image_path}")
        missing_count += 1
        continue

    img_w = int(float(resolution_node.attrib["x"]))
    img_h = int(float(resolution_node.attrib["y"]))

    image_filename = os.path.basename(rel_image_path)
    dst_image_path = os.path.join(ALL_IMAGES_DIR, image_filename)
    txt_name = os.path.splitext(image_filename)[0] + ".txt"
    txt_path = os.path.join(ALL_LABELS_DIR, txt_name)

    # copy ảnh
    shutil.copy(src_image_path, dst_image_path)

    valid_boxes = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for rect in tagged_rectangles_node.findall("taggedRectangle"):
            x = float(rect.attrib["x"])
            y = float(rect.attrib["y"])
            w = float(rect.attrib["width"])
            h = float(rect.attrib["height"])

            # bỏ box lỗi
            if w <= 0 or h <= 0:
                continue

            # ép box vào trong ảnh
            x = max(0, x)
            y = max(0, y)
            if x + w > img_w:
                w = img_w - x
            if y + h > img_h:
                h = img_h - y

            if w <= 0 or h <= 0:
                continue

            x_center, y_center, bw, bh = yolo_box_from_xywh(x, y, w, h, img_w, img_h)

            # class text = 0
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            valid_boxes += 1
            box_count += 1

    # nếu ảnh không có box hợp lệ thì xóa file rỗng và ảnh copy
    if valid_boxes == 0:
        if os.path.exists(txt_path):
            os.remove(txt_path)
        if os.path.exists(dst_image_path):
            os.remove(dst_image_path)
        continue

    image_count += 1

print(f"Đã xử lý {image_count} ảnh.")
print(f"Tổng số box: {box_count}")
print(f"Số ảnh thiếu file thật: {missing_count}")
print("Ảnh YOLO ở:", ALL_IMAGES_DIR)
print("Label YOLO ở:", ALL_LABELS_DIR)