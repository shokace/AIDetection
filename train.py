# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.preprocessing.transforms import get_transforms
from config import TRAIN_DIR, VAL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("model", exist_ok=True)

batch_size = 32
num_epochs = 5
num_classes = 2

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

train_transform = get_transforms(train=True)
val_transform = get_transforms(train=False)

image_datasets = {
    'train': datasets.ImageFolder(TRAIN_DIR, transform=train_transform),
    'val': datasets.ImageFolder(VAL_DIR, transform=val_transform),
}

print("CWD:", os.getcwd())
print("Train classes:", image_datasets['train'].class_to_idx)
print("Val classes:", image_datasets['val'].class_to_idx)
print("Train samples:", len(image_datasets['train'].samples))
print("Val samples:", len(image_datasets['val'].samples))

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
}

# Load ResNet-50
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze base layers

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloaders['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in dataloaders['val']:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)
    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), "model/resnet50_fakeness.pt")
print("Model saved")