import sys
from pathlib import Path
import cv2

# -------- Paths & imports --------

# Folder containing THIS test script
CURRENT_DIR = Path(__file__).resolve().parent

# Assume detector.py lives one level up (project root)
PROJECT_ROOT = CURRENT_DIR.parent

# Make sure Python can see detector.py
sys.path.insert(0, str(PROJECT_ROOT))

import detector  # now it should work

# --- 1. Correct path ---
img_path = PROJECT_ROOT / "dataset_examples" / "Raw" / "Crow+Magpie.JPG"

# --- 2. Load image ---
frame = cv2.imread(str(img_path))
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

if frame_bgr is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# --- 3. Run detector ---
species, frame_with_boxes = detector.detect_and_classify(frame_bgr)

scale = 0.1   # 50% size
h, w = frame_with_boxes.shape[:2]
frame_small = cv2.resize(frame_with_boxes, (int(w*scale), int(h*scale)))

print("Detected species:", species)

# --- 4. Show the result ---
cv2.imshow("Result", frame_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

