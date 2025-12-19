import cv2
from pathlib import Path
import src.perception.detector as detector

ROOT = Path(__file__).resolve().parents[1]
IMG_PATH = ROOT / "data" / "dataset_examples" / "Raw" / "None01.JPG"
frame = cv2.imread(str(IMG_PATH))
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
species, conf, frame_bgr = detector.detect_and_classify(frame_bgr)
print(f"Species Detected: {species} (conf={conf:.2f})")

scale = 0.1
h, w = frame_bgr.shape[:2]
frame_small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))

cv2.imshow("Result", frame_small)
cv2.waitKey(0)
cv2.destroyAllWindows()