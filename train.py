# train.py
import os
import random
from collections import Counter
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from pipeline.preprocessing.transforms import get_transforms
from config import TRAIN_DIR, VAL_DIR


# -----------------------
# Global config & device
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_EPOCHS = 15           # more than 4 to let it actually fine-tune
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2

MODEL_DIR = "model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_fakeness_best.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_fakeness_last.pt")

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # If you want strict determinism (slower), uncomment:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(42)


# -----------------------
# Model: ResNet-50
# -----------------------
def build_model(num_classes: int) -> nn.Module:
    # Use default pre-trained weights (ImageNet)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last residual block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classification head with dropout + linear
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


# -----------------------
# Training & Validation
# -----------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Train", leave=False)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Val", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# -----------------------
# Main: put ALL DataLoader stuff here (Windows-safe)
# -----------------------
def main() -> None:
    print("CWD:", os.getcwd())

    # Transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Datasets
    image_datasets = {
        "train": datasets.ImageFolder(TRAIN_DIR, transform=train_transform),
        "val": datasets.ImageFolder(VAL_DIR, transform=val_transform),
    }

    print("Train classes:", image_datasets["train"].class_to_idx)
    print("Val   classes:", image_datasets["val"].class_to_idx)
    print("Train samples:", len(image_datasets["train"].samples))
    print("Val   samples:", len(image_datasets["val"].samples))

    # Class distribution (for imbalance)
    train_targets = image_datasets["train"].targets
    class_counts = Counter(train_targets)
    print("Train class counts:", class_counts)

    # Build class weights for CrossEntropyLoss
    class_sample_counts = [class_counts[i] for i in range(NUM_CLASSES)]
    class_sample_counts = torch.tensor(class_sample_counts, dtype=torch.float)
    class_weights = 1.0 / class_sample_counts  # inverse frequency
    class_weights = class_weights / class_weights.sum()  # normalize (optional)
    class_weights = class_weights.to(DEVICE)
    print("Class weights:", class_weights.tolist())

    # DataLoaders
    num_workers = max(1, os.cpu_count() // 2)  # reduce if issues; or set to 0
    pin_memory = DEVICE.type == "cuda"

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    # Model, criterion, optimizer, scheduler
    model = build_model(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion,
            scaler,
            DEVICE,
        )

        val_loss, val_acc = evaluate(
            model,
            dataloaders["val"],
            criterion,
            DEVICE,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Step scheduler with validation accuracy
        scheduler.step(val_acc)
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> New best model saved (val_acc={best_val_acc:.4f})")

    # Save last epoch model as well
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"\nTraining complete.")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Last model path: {LAST_MODEL_PATH}")


if __name__ == "__main__":
    # On Windows, this guard is REQUIRED when using DataLoader(num_workers>0)
    main()
