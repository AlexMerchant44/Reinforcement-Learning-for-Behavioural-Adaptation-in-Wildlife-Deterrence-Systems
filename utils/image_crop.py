from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = "yolo11n.pt"
# Input actual path to raw dataset
DATA_DIR = Path("data")
RAW_ROOT   = DATA_DIR / "Uncropped"
CROP_ROOT  = DATA_DIR / "Cropped"
CLASSES    = ["Magpie", "Crow", "None"]  # folder names

# Script to create cropped image dataset using Yolo11n

model = YOLO(MODEL_PATH)

for cls in CLASSES:
    in_dir = RAW_ROOT / cls
    out_dir = CROP_ROOT / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    # Make a new directory for the output crops, don't throw an error if it already exists

    img_paths = sorted([p for p in in_dir.glob("*") 
                    if p.suffix.lower() == ".jpg"])
    # Collect all .jpg files and return them as a sorted list of Path objects

    for img_path in img_paths:
        # Run yolon11 on image, turn off console output, select first result as yolo results list for batch processing
        # Save original image for cropping
        results = model(str(img_path), verbose=False)
        r = results[0]
        img = r.orig_img

        if r.boxes is None or len(r.boxes) == 0:
            # No detection, skip to next image
            print(f"No detection in {img_path}")
            continue

        # Pick the box with highest confidence, regardless of class
        best_box = None
        best_conf = -1.0
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        # Unpack best box coordinates from tuple and convert to integers from float
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 15% padding around the bird, image classifiers perform better with contextual information
        # max(w, h) so skinny objects still get sufficient padding
        w = x2 - x1
        h = y2 - y1
        pad = int(0.15 * max(w, h))

        # Clamp coordinates to prevent negative pixel indices
        H, W, _ = img.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), crop)

print("Done cropping.")
