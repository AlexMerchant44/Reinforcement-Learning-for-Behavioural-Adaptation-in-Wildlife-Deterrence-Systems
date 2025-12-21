from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
from pathlib import Path

CLASS_NAMES = ["Crow", "Magpie", "None"]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "image_classifier.pth"
YOLO_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"

# Load YOLO and classifier
yolo = YOLO(YOLO_PATH)
classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
classifier.eval()

# Perform same nonrandom transformations as done in training/validation
to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
])


def detect_and_classify(frame_bgr):
    """
    Runs YOLO + classifier on frame_bgr.

    Returns:
      species (str: 'Crow' / 'Magpie' / 'None')
      confidence (float: 0-1)
      frame_bgr (BGR image with boxes drawn)
    """
    results = yolo(frame_bgr, verbose=False)
    r = results[0] # take first result as yolo returns list object for batch processing

    # No detections, return species 'None', confidence of 0 and the frame
    if not r.boxes or len(r.boxes) == 0:
        return "None", 0.0, frame_bgr

    detected = []  # list of (species, confidence)

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        yolo_conf = float(box.conf.item())  # YOLO confidence

        # Same 15% padding as in training
        w = x2 - x1
        h = y2 - y1
        pad = int(0.15 * max(w, h))

        H, W, _ = frame_bgr.shape
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        # crop to classify
        crop = frame_bgr[y1p:y2p, x1p:x2p]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb) # classifier expects PIL RGB image (ImageFolder did this in train_classifier.py)
        x = to_tensor(pil).unsqueeze(0)

        # classify crop
        with torch.no_grad():
            out = classifier(x)
            cls_idx = torch.argmax(out, dim=1).item() # get predicted species id
            species = CLASS_NAMES[cls_idx] # get species string from species id

        detected.append((species, yolo_conf))

        # draw box + label + confidence
        label = f"{species} {yolo_conf:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        
    # Priority: Crow > Magpie
    # unpack all instances of 'Crow' (s) with their corresponding confidence (c)
    crow_confs = [c for s, c in detected if s == "Crow"]
    if crow_confs:
        return "Crow", max(crow_confs), frame_bgr

    magpie_confs = [c for s, c in detected if s == "Magpie"]
    if magpie_confs:
        return "Magpie", max(magpie_confs), frame_bgr

    return "None", 0.0, frame_bgr
