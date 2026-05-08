import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time, json
from pathlib import Path

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "data"
OUTPUT_DIR = Path("weights")
IMG_SIZE   = 224
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

train_ds = datasets.ImageFolder("data/train", transform=train_transform)
val_ds   = datasets.ImageFolder("data/val",   transform=val_transform)
num_classes = len(train_ds.classes)
print(f"Classes: {num_classes}")
print(f"Device: {DEVICE}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

best_acc = 0.0

for epoch in range(1, 21):
    # train
    model.train()
    correct, total = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/20 train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += len(imgs)
    train_acc = correct / total

    # val
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(imgs)
    val_acc = correct / total
    scheduler.step()

    print(f"Epoch {epoch}/20 | Train acc {train_acc:.3%} | Val acc {val_acc:.3%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_DIR / "resnet101_best.pth")
        print(f"  ✓ New best saved ({best_acc:.3%})")

print(f"\nResNet-101 Best Val Accuracy: {best_acc:.3%}")
print("Saved → weights/resnet101_best.pth")
