import json
import os
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from torchvision import datasets, transforms

from pipeline import AnimalClassifier, CLASS_NAMES, DEVICE, IMG_SIZE, NORM_MEAN, NORM_STD

# ── setup ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
SAMPLE_DIR = OUTPUT_DIR / "sample_predictions"
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

WEIGHTS    = "weights/efficientnet_b0_best.pth"
DATA_DIR   = "data/test"

print("Loading classifier...")
classifier = AnimalClassifier(
    weights_path=WEIGHTS,
    num_classes=len(CLASS_NAMES),
    class_names=CLASS_NAMES,
    device=DEVICE,
)

# ── phase 1: collect predictions ─────────────────────────────────────────────
print("\nRunning predictions on test set...")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

ds         = datasets.ImageFolder(DATA_DIR)
name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

y_true, y_pred, confidences, image_paths = [], [], [], []
latencies = []

for img_path, label_idx in ds.samples:
    true_name = ds.classes[label_idx]
    if true_name not in name_to_idx:
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = classifier.predict(img_rgb)

    y_true.append(name_to_idx[true_name])
    y_pred.append(result["class_idx"])
    confidences.append(result["confidence"])
    image_paths.append(img_path)
    latencies.append(result["latency_ms"])

print(f"Done! Evaluated {len(y_true)} images.")

# ── phase 2: confusion matrix ─────────────────────────────────────────────────
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(28, 24))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 6},
)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix — EfficientNet-B0 (90 classes)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved → outputs/confusion_matrix.png")

# ── phase 3: classification report ───────────────────────────────────────────
print("\nGenerating classification report...")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)
print("Saved → outputs/classification_report.txt")

# ── phase 6: metrics summary ─────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
avg_lat = float(np.mean(latencies))

metrics = {
    "accuracy":       round(acc, 4),
    "precision":      round(prec, 4),
    "recall":         round(rec, 4),
    "f1_score":       round(f1, 4),
    "avg_latency_ms": round(avg_lat, 2),
    "total_images":   len(y_true),
}
print(f"\n{'='*40}")
print(f"  Accuracy  : {acc:.2%}")
print(f"  Precision : {prec:.2%}")
print(f"  Recall    : {rec:.2%}")
print(f"  F1-score  : {f1:.2%}")
print(f"  Avg latency: {avg_lat:.1f} ms")
print(f"{'='*40}")

with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved → outputs/metrics_summary.json")

# ── phase 4: sample predictions ──────────────────────────────────────────────
print("\nSaving sample predictions...")
correct   = [(p, t, pr, c) for p, t, pr, c in zip(image_paths, y_true, y_pred, confidences) if t == pr]
incorrect = [(p, t, pr, c) for p, t, pr, c in zip(image_paths, y_true, y_pred, confidences) if t != pr]

random.shuffle(correct)
random.shuffle(incorrect)

def save_samples(samples, filename, title):
    n = min(10, len(samples))
    if n == 0:
        return
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for i in range(n):
        path, true_idx, pred_idx, conf = samples[i]
        img = Image.open(path).resize((150, 150))
        axes[i].imshow(img)
        true_name = CLASS_NAMES[true_idx].capitalize()
        pred_name = CLASS_NAMES[pred_idx].capitalize()
        color = "green" if true_idx == pred_idx else "red"
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.1%}",
            fontsize=8, color=color
        )
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SAMPLE_DIR / filename, dpi=120)
    plt.close()
    print(f"Saved → outputs/sample_predictions/{filename}")

save_samples(correct,   "correct_predictions.png",   "Correct Predictions ✓")
save_samples(incorrect, "incorrect_predictions.png",  "Incorrect Predictions ✗")

print("\nAll evaluation outputs saved to outputs/ folder!")
print("You can now add these to your research paper.")