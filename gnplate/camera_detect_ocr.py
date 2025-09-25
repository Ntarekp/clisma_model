import cv2
import os
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = r"C:/Users/rca/Documents/clisma/runs/detect/train/weights/best.pt"
OUTPUT_DIR = r"C:/Users/rca/Documents/clisma/gnplate/results"
TEXT_LOG = os.path.join(OUTPUT_DIR, "plates_log.txt")

# Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize OCR
reader = easyocr.Reader(['en'])

# Open default camera
cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode for matplotlib
fig, ax = plt.subplots()

with open(TEXT_LOG, "w") as log:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                # OCR
                ocr_result = reader.readtext(plate_crop)
                for (_, text, conf) in ocr_result:
                    if conf > 0.4:  # filter low confidence
                        log.write(f"{text}\n")
                        print(f"Detected Plate: {text}")

                # Draw bbox & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame with matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        plt.draw()
        plt.pause(0.001)  # replaces cv2.waitKey

cap.release()
plt.close(fig)
print("Camera stream ended. Results saved to:", TEXT_LOG)
