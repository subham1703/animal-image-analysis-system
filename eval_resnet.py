import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

from pipeline import CLASS_NAMES

# load model
ds = datasets.ImageFolder("data/test", transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
]))
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

model = models.resnet101(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("weights/resnet101_best.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# measure latency
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(dummy)
    times.append((time.perf_counter() - t0) * 1000)
avg_latency = float(np.mean(times))

# run predictions
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

print(f"\n{'='*50}")
print(f"  ResNet-101 Test Results")
print(f"{'='*50}")
print(f"  Accuracy  : {acc:.2%}")
print(f"  Precision : {prec:.2%}")
print(f"  Recall    : {rec:.2%}")
print(f"  F1-score  : {f1:.2%}")
print(f"  Latency   : {avg_latency:.1f} ms")
print(f"  Params    : 44.5M")
print(f"{'='*50}")

results = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "avg_latency_ms": round(avg_latency, 2),
    "parameters_M": 44.5
}
Path("outputs").mkdir(exist_ok=True)
with open("outputs/resnet101_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved → outputs/resnet101_metrics.json")