import os
from ultralytics import YOLO
import easyocr
import cv2

# === PATHS ===
MODEL_PATH = "C:/Users/rca/Documents/clisma/runs/detect/train/weights/best.pt"  # update if needed
SOURCE_PATH = "C:/Users/rca/Documents/clisma/gnplate/kaggle/car-plate-detection/images"
OUTPUT_DIR = "C:/Users/rca/Documents/clisma/ocr_results"
TEXT_FILE = os.path.join(OUTPUT_DIR, "detected_plates.txt")

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
reader = easyocr.Reader(['en'])
model = YOLO(MODEL_PATH)

# === RUN DETECTION ===
results = model.predict(source=SOURCE_PATH, save=True, save_crop=True, project=OUTPUT_DIR, name="predict")

# === OCR AND SAVE ===
with open(TEXT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        for i, box in enumerate(r.boxes):
            # Get crop image path
            crop_dir = os.path.join(OUTPUT_DIR, "predict", "crops", "plate")
            crop_files = os.listdir(crop_dir)

            for crop_file in crop_files:
                crop_path = os.path.join(crop_dir, crop_file)
                img = cv2.imread(crop_path)

                if img is None:
                    continue

                # Run OCR
                ocr_result = reader.readtext(img, detail=0)

                if ocr_result:
                    plate_text = ocr_result[0]
                    print(f"[OCR] {crop_file}: {plate_text}")
                    f.write(f"{crop_file}: {plate_text}\n")

print(f"\n✅ OCR results saved to: {TEXT_FILE}")
