from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = "yolo11n.pt"
RAW_ROOT   = Path("Uncropped")
CROP_ROOT  = Path("Cropped")
CLASSES    = ["Crow", "Magpie"]  # folder names
#MAX_IMAGES = 5  # <-- process only this many images per class

# Script to create cropped image dataset using Yolo11n

model = YOLO(MODEL_PATH)

for cls in CLASSES:
    in_dir = RAW_ROOT / cls
    out_dir = CROP_ROOT / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in in_dir.glob("*") 
                    if p.suffix.lower() == ".jpg"])
    
    print(f"Processing {cls}: {len(img_paths)} images")

    #img_paths = img_paths[:MAX_IMAGES]

    for img_path in img_paths:
        results = model(str(img_path), verbose=False)
        r = results[0]
        img = r.orig_img  # numpy array, BGR

        if r.boxes is None or len(r.boxes) == 0:
            # no detection – optional: skip or do a central crop
            print(f"No detection in {img_path}")
            continue

        # pick the box with highest confidence, regardless of class
        best_box = None
        best_conf = -1.0
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # padding around the bird
        w = x2 - x1
        h = y2 - y1
        pad = int(0.15 * max(w, h))

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
