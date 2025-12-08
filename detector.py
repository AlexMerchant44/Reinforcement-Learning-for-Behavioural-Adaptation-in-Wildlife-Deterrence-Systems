from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
from pathlib import Path

CLASS_NAMES = ["Crow", "Magpie"]

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "image_classifier.pth"
YOLO_PATH = HERE / "models" / "yolo11n.pt"

# Load YOLO and classifier
yolo = YOLO(YOLO_PATH)
classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
classifier.eval()

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
    Runs Yolo11n and classifier on frame_bgr
    Returns:
      species (str: 'Crow'/'Magpie'/None),
      frame_with_boxes (BGR)
    """
    results = yolo(frame_bgr, verbose=False)
    r = results[0]

    if not r.boxes or len(r.boxes) == 0:
        return None, frame_bgr

    detected_species = set()   # keep unique species seen

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # crop to classify
        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        x = to_tensor(pil).unsqueeze(0)

        # classify crop
        with torch.no_grad():
            out = classifier(x)
            cls_idx = torch.argmax(out, dim=1).item()
            species = CLASS_NAMES[cls_idx]

        detected_species.add(species)

        # draw box + label
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame_bgr, species, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # --- Apply priority rules ---
    if "Crow" in detected_species:
        return "Crow", frame_bgr

    if "Magpie" in detected_species:
        return "Magpie", frame_bgr

    return None, frame_bgr