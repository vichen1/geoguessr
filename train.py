import os
import time
import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
DATA_DIR = "data"          # folder with training jpg files
MODEL_PATH = "model.pt"
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 2
IMAGE_SIZE = 224

CLASSES = [
    "Los_Angeles",
    "San_Diego",
    "SLO",
    "Bakersfield",
    "Riverside",
    "Anaheim",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for cls_name, i in CLASS_TO_IDX.items()}


# =========================
# Dataset
# =========================
class SoCalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.image_dir, filename)

        image = Image.open(filepath).convert("RGB")

        # Label is everything before the first dash
        # Example: Los_Angeles-abc123.jpg -> Los_Angeles
        label_str = filename.split("-")[0]

        if label_str not in CLASS_TO_IDX:
            raise ValueError(f"Unknown label '{label_str}' in filename '{filename}'")

        label = CLASS_TO_IDX[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# Model
# =========================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer for 6 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))

    return model


# =========================
# Training / Evaluation
# =========================
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_count += labels.size(0)

        train_loss = running_loss / running_count
        train_acc = running_correct / running_count

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history, elapsed


def plot_training_curve(history, save_path="training_curve.png"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# Main
# =========================
def main():
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    full_dataset = SoCalDataset(DATA_DIR, transform=None)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Assign transforms after split
    train_subset.dataset = copy.deepcopy(full_dataset)
    train_subset.dataset.transform = train_transform

    val_subset.dataset = copy.deepcopy(full_dataset)
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Optional: class distribution check
    label_counts = Counter()
    for filename in full_dataset.image_files:
        label_str = filename.split("-")[0]
        label_counts[label_str] += 1
    print("Class distribution:", label_counts)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, history, elapsed = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        NUM_EPOCHS
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model weights to {MODEL_PATH}")

    plot_training_curve(history, save_path="training_curve.png")
    print("Saved training curve to training_curve.png")


if __name__ == "__main__":
    main()