from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image
import cv2

CLASS_NAMES = ["Crow", "Magpie"]

# load YOLO and classifier once
yolo = YOLO("models/yolo11n.pt")

classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
classifier.load_state_dict(torch.load("models/image_classifier.pth", map_location="cpu"))
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
    Returns:
      detected (bool),
      species (str: 'Crow'/'Magpie'/None),
      frame_with_box (BGR)
    """
    results = yolo(frame_bgr, verbose=False)
    r = results[0]

    if not r.boxes or len(r.boxes) == 0:
        return False, None, frame_bgr

    # pick highest-conf box
    best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    crop = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    x = to_tensor(pil).unsqueeze(0)

    with torch.no_grad():
        out = classifier(x)
        cls_idx = torch.argmax(out, dim=1).item()
        species = CLASS_NAMES[cls_idx]

    # draw box for debug
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return True, species, frame_bgr
