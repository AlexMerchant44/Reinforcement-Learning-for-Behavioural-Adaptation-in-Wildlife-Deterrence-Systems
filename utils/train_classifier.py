import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from pathlib import Path

# Script to train image classifier

def main():

    DATA_DIR = Path("data/Cropped")
    BATCH_SIZE = 32 # Number of images per gradient update, balances gradient estimates with GPU usage
    NUM_EPOCHS = 10 # Number of full passes over dataset, should converge before this
    VAL_SPLIT = 0.2
    LR = 1e-4 # Step size for gradient descent updates, good for pre-trained ResNet18

    device = torch.device("cpu")
    print("Using device:", device)

    # Resize to 224x224 (what resnet was trained on), flip some images horizontally (increase diversity)
    # Rotate up to +-10 degrees
    # Randomly adjust colours to simulate lighting variation
    # Convert to PyTorch tensor, scales pixel values from 0-255 to 0-1 
    # Normalise each channel to match ImageNet stats (match training data)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Deterministic validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Check number of classes
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    num_classes = len(full_dataset.classes)
    print("Classes:", full_dataset.classes)

    # Load the dataset for the two sets but with different transforms
    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_base   = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    n_total = len(train_base)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_val

    # Split dataset pseudo randomly by creating random list of integers and taking the first n_train
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    # Build subsets of datasets to use different transforms on the same base dataset
    train_dataset = Subset(train_base, train_idx)
    val_dataset   = Subset(val_base, val_idx)

    # Multiple workers work now because we're inside main(). No need for shuffling on validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Use pretrained resnet18 weights, change last fully connected layer to have 3 outputs
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Define loss function, CRL as its multi class classification
    # Define optimizer, adam works well for fine tuning pre trained models
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        model.train()
        running_loss = 0.0 # for epoch train loss
        running_corrects = 0 # for epoch train accuracy

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Ensure data/model live on same CPU/GPU

            optimizer.zero_grad() # clear gradients from previous batch
            outputs = model(inputs) # outputs are logits with shape BATCH_SIZE, num_classes
            loss = criterion(outputs, labels) # computes cross-entropy loss
            loss.backward() # compute gradient of loss
            optimizer.step() # update model parameters using adam

            _, preds = torch.max(outputs, 1) # takes argmax of class predictions
            running_loss += loss.item() * inputs.size(0) # multiple scalar loss by batch size
            running_corrects += torch.sum(preds == labels) # counts correct predictions for each item in the batch

        train_loss = running_loss / n_train # mean training loss per sample
        train_acc = running_corrects.double() / n_train # mean training accuracy, double() to avoid floating point division

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        # Validation phase, dont compute gradients.
        # Same logic as training phase just without gradients and backward pass
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "image_classifier.pth")
            print(f"New best model saved with val_acc={best_val_acc:.4f}")

    print("\nTraining done. Best val acc:", best_val_acc.item())
    print("Weights saved to image_classifier.pth")


if __name__ == "__main__":
    main()
