import cv2
import os
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# -------------------
# Paths & Setup
# -------------------
VEHICLE_MODEL = "yolov8n.pt"  # pretrained COCO model
PLATE_MODEL = r"C:/Users/rca/Documents/clisma/runs/detect/train/weights/best.pt"
OUTPUT_DIR = r"C:/Users/rca/Documents/clisma/gnplate/results"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
TEXT_LOG = os.path.join(OUTPUT_DIR, "plates_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Load models
vehicle_model = YOLO(VEHICLE_MODEL)
plate_model = YOLO(PLATE_MODEL)

# Initialize OCR
reader = easyocr.Reader(['en'])

# Open camera
cap = cv2.VideoCapture(0)

# Matplotlib interactive mode
plt.ion()
fig, ax = plt.subplots()

plate_counter = 0  # to save unique crop names

with open(TEXT_LOG, "w") as log:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------
        # Stage 1: Vehicle Detection
        # Classes (COCO): 2=car, 3=motorbike, 5=bus, 7=truck
        # -------------------
        vehicle_results = vehicle_model(frame, classes=[2, 3, 5, 7], verbose=False)

        for v_result in vehicle_results:
            for vbox in v_result.boxes:
                x1, y1, x2, y2 = map(int, vbox.xyxy[0])
                vehicle_crop = frame[y1:y2, x1:x2]

                # -------------------
                # Stage 2: Plate Detection inside vehicle
                # -------------------
                plate_results = plate_model(vehicle_crop, verbose=False)

                for p_result in plate_results:
                    for pbox in p_result.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        # -------------------
                        # Stage 3: OCR
                        # -------------------
                        ocr_result = reader.readtext(plate_crop)
                        for (_, text, conf) in ocr_result:
                            if conf > 0.4:
                                plate_counter += 1
                                log.write(f"{text}\n")
                                print(f"Detected Plate: {text}")

                                # Save cropped plate image
                                crop_path = os.path.join(CROPS_DIR, f"plate_{plate_counter}.jpg")
                                cv2.imwrite(crop_path, plate_crop)

                                # Draw plate bbox
                                cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), (0, 255, 0), 2)
                                cv2.putText(vehicle_crop, text, (px1, py1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw vehicle bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Vehicle", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # -------------------
        # Display frame with matplotlib
        # -------------------
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        plt.draw()
        plt.pause(0.001)

cap.release()
plt.close(fig)
print("Camera stream ended. Results saved to:", TEXT_LOG)
