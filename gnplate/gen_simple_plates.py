from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, random

OUTPUT_DIR = "dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images", "train")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

def random_plate():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"R{random.choice(letters)}{random.choice(letters)} {random.randint(100,999)} {random.choice(letters)}"

def generate_plate(index):
    plate_text = random_plate()
    img_w, img_h = 640, 480

    # Random background color
    bg_color = tuple(random.randint(150, 255) for _ in range(3))
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    d = ImageDraw.Draw(img)

    # Random plate size & position
    plate_w = random.randint(200, 350)
    plate_h = random.randint(80, 120)
    x0 = random.randint(50, img_w - plate_w - 50)
    y0 = random.randint(50, img_h - plate_h - 50)
    x1, y1 = x0 + plate_w, y0 + plate_h

    # Draw plate rectangle
    d.rectangle([x0, y0, x1, y1], outline="black", width=5, fill="white")

    # Load random font (fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", random.randint(35, 50))
    except:
        font = ImageFont.load_default()

    # Text size & position
    bbox = d.textbbox((0,0), plate_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    d.text((x0 + (plate_w - tw)//2, y0 + (plate_h - th)//2), plate_text, fill="black", font=font)

    # Apply random blur/noise
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(1))

    # Save image
    img_path = os.path.join(IMG_DIR, f"plate_{index}.jpg")
    img.save(img_path)

    # YOLO label (normalized)
    x_center = (x0 + plate_w/2) / img_w
    y_center = (y0 + plate_h/2) / img_h
    w = plate_w / img_w
    h = plate_h / img_h
    lbl_path = os.path.join(LBL_DIR, f"plate_{index}.txt")
    with open(lbl_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Generate dataset
for i in range(200):  # 200 samples
    generate_plate(i)

print("✅ Generated 200 varied training images + labels in dataset/")
