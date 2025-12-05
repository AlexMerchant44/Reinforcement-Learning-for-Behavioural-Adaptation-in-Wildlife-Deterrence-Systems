import random
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import cv2
from ultralytics import YOLO

# Script to test full detection and classification pipeline


# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
RAW_DIR = Path("Uncropped")
CLASS_NAMES = ["Crow", "Magpie"]
CLASSIFIER_WEIGHTS = "image_classifier.pth"
YOLO_WEIGHTS = "yolo11n.pt"
device = torch.device("cpu")
yolo_model = YOLO(YOLO_WEIGHTS)


# --------------------------------------------------------
# Build classifier model
# --------------------------------------------------------
def load_classifier(weights_path):
    model = models.resnet18(weights=None)  # no pretrained weights now
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Crow, Magpie
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


classifier = load_classifier(CLASSIFIER_WEIGHTS)


# --------------------------------------------------------
# Classifier preprocessing
# --------------------------------------------------------
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------------
# Utility: pick a random image
# --------------------------------------------------------
def get_random_image():
    allowed = ["Crow", "Magpie"]

    species_dirs = [RAW_DIR / name for name in allowed if (RAW_DIR / name).is_dir()]

    chosen_species = random.choice(species_dirs)
    images = list(chosen_species.glob("*"))
    images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    chosen_image = random.choice(images)
    return chosen_species.name, chosen_image


# --------------------------------------------------------
# Utility: run YOLO and extract best crop
# --------------------------------------------------------
def get_yolo_crop(image_path):
    results = yolo_model(str(image_path), verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        print(" YOLO found no detections.")
        return None, None

    # pick highest-confidence box
    best_box = None
    best_conf = -1

    for box in r.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # read original image (OpenCV default BGR)
    img = cv2.imread(str(image_path))
    H, W, _ = img.shape

    # add padding
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        print(" Invalid crop created.")
        return None, None

    return crop, best_conf


# --------------------------------------------------------
# Utility: classify a crop
# --------------------------------------------------------
def classify_crop(crop_bgr):
    # convert BGR → RGB → PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    x = classifier_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(x)
        probs = F.softmax(out, dim=1)[0]

    cls_idx = torch.argmax(probs).item()
    cls_name = CLASS_NAMES[cls_idx]
    confidence = probs[cls_idx].item()

    return cls_name, confidence


# --------------------------------------------------------
# MAIN TEST
# --------------------------------------------------------
if __name__ == "__main__":
    # pick image
    folder, img_path = get_random_image()
    print(f"\n Selected image from folder: {folder}")
    print(f"Image path: {img_path}")

    # run YOLO
    crop, yolo_conf = get_yolo_crop(img_path)
    if crop is None:
        exit()

    print(f"YOLO detected a bird with confidence {yolo_conf:.3f}")
    # classify crop
    cls_name, cls_conf = classify_crop(crop)
    print(f"\n CLASSIFIER RESULT: {cls_name} ({cls_conf:.2%} confidence)\n")

    # show crop
    cv2.imshow("YOLO Crop", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
