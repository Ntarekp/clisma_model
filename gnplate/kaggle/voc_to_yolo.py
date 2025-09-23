import os
import xml.etree.ElementTree as ET
import shutil

# Paths
ANNOT_DIR = "car-plate-detection/annotations"
IMAGE_DIR = "car-plate-detection/images"
YOLO_IMAGES = "dataset/images/train"
YOLO_LABELS = "dataset/labels/train"

os.makedirs(YOLO_IMAGES, exist_ok=True)
os.makedirs(YOLO_LABELS, exist_ok=True)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

for file in os.listdir(ANNOT_DIR):
    if not file.endswith(".xml"):
        continue
    in_file = open(os.path.join(ANNOT_DIR, file))
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    out_file = open(os.path.join(YOLO_LABELS, file.replace(".xml", ".txt")), "w")

    for obj in root.iter("object"):
        cls_id = 0  # only 1 class: license_plate
        xmlbox = obj.find("bndbox")
        b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
             float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

    # Copy image to dataset/images/train
    img_file = file.replace(".xml", ".png")
    shutil.copy(os.path.join(IMAGE_DIR, img_file),
                os.path.join(YOLO_IMAGES, img_file))
