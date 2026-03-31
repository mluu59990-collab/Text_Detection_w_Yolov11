import os
import shutil
import xml.etree.ElementTree as ET

# =========================
# 1. KHAI BÁO ĐƯỜNG DẪN
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_XML_DIR = os.path.join(BASE_DIR, "data", "raw", "annotations_xml")

OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "data", "yolo_text_detection", "all_images")
OUTPUT_LABEL_DIR = os.path.join(BASE_DIR, "data", "yolo_text_detection", "all_labels")

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)


# =========================
# 2. HÀM CHUYỂN VOC -> YOLO
# =========================
def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


# =========================
# 3. CHUYỂN TOÀN BỘ XML
# =========================
xml_files = [f for f in os.listdir(RAW_XML_DIR) if f.endswith(".xml")]

print(f"Tìm thấy {len(xml_files)} file XML")

for xml_file in xml_files:
    xml_path = os.path.join(RAW_XML_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Lấy kích thước ảnh
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    # Tên ảnh tương ứng
    filename_tag = root.find("filename")
    if filename_tag is not None:
        img_name = filename_tag.text.strip()
    else:
        img_name = xml_file.replace(".xml", ".jpg")

    img_src_path = os.path.join(RAW_IMAGE_DIR, img_name)

    if not os.path.exists(img_src_path):
        print(f"[WARNING] Không tìm thấy ảnh tương ứng: {img_src_path}")
        continue

    # Copy ảnh sang all_images
    shutil.copy(img_src_path, os.path.join(OUTPUT_IMAGE_DIR, img_name))

    # Tạo file txt YOLO
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

    with open(txt_path, "w", encoding="utf-8") as f:
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()

            # Vì chỉ có 1 class text
            class_id = 0

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Ép box nằm trong ảnh
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w - 1, xmax)
            ymax = min(img_h - 1, ymax)

            x_center, y_center, width, height = voc_to_yolo(
                xmin, ymin, xmax, ymax, img_w, img_h
            )

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Đã convert xong XML -> YOLO")
print("Ảnh lưu tại:", OUTPUT_IMAGE_DIR)
print("Label lưu tại:", OUTPUT_LABEL_DIR)