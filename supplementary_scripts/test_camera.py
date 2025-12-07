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

import camera

frame = camera.get_frame()

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()