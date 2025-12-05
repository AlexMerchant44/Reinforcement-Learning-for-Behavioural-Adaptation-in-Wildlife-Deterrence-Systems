import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from pathlib import Path

# Script to train image classifier

def main():

    # -------- CONFIG --------
    DATA_DIR = Path("Cropped")  # <- change if needed
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    VAL_SPLIT = 0.2       # 20% for validation
    LR = 1e-4

    # -------- DEVICE --------
    device = torch.device("cpu")
    print("Using device:", device)

    # -------- DATA TRANSFORMS --------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -------- DATASETS / LOADERS --------
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    num_classes = len(full_dataset.classes)
    print("Classes:", full_dataset.classes)

    n_total = len(full_dataset)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Validation should not have augmentations
    val_dataset.dataset.transform = val_transform

    # *** Multiple workers work now because we're inside main() ***
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # -------- MODEL --------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------- TRAINING LOOP --------
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        train_loss = running_loss / n_train
        train_acc = running_corrects.double() / n_train

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)

        val_loss = val_loss / n_val
        val_acc = val_corrects.double() / n_val

        print(f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "image_classifier.pth")
            print(f"  ↳ New best model saved with val_acc={best_val_acc:.4f}")

    print("\nTraining done. Best val acc:", best_val_acc.item())
    print("Weights saved to image_classifier.pth")


if __name__ == "__main__":
    main()
