import cv2
import numpy as np
import json
from pathlib import Path
from pipeline import AnimalClassifier, CLASS_NAMES, DEVICE
import matplotlib.pyplot as plt
import random

# ── setup ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = "weights/efficientnet_b0_best.pth"
TEST_DIR = Path("data/test")

classifier = AnimalClassifier(
    weights_path=WEIGHTS,
    num_classes=len(CLASS_NAMES),
    class_names=CLASS_NAMES,
    device=DEVICE,
)

# ── load test images ──────────────────────────────────────────────────────────
from torchvision import datasets
ds = datasets.ImageFolder("data/test")
name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

# pick 100 random images
samples = random.sample(ds.samples, 100)

# ── perturbation functions ────────────────────────────────────────────────────
def add_blur(img, ksize=15):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def add_noise(img, std=50):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def low_light(img, factor=0.2):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def add_occlusion(img, frac=0.3):
    img = img.copy()
    h, w = img.shape[:2]
    oh, ow = int(h * frac), int(w * frac)
    y0 = np.random.randint(0, h - oh)
    x0 = np.random.randint(0, w - ow)
    img[y0:y0+oh, x0:x0+ow] = 128
    return img

def get_prediction(img_rgb):
    result = classifier.predict(img_rgb)
    return result["class_idx"], result["confidence"]

# ── run tests ────────────────────────────────────────────────────────────────
conditions = {
    "Clean":      lambda x: x,
    "Blur":       lambda x: add_blur(x, 15),
    "Noise":      lambda x: add_noise(x, 50),
    "Low Light":  lambda x: low_light(x, 0.2),
    "Occlusion":  lambda x: add_occlusion(x, 0.3),
}

results = {name: {"correct": 0, "total": 0, "confidences": []} 
           for name in conditions}

print("Running robustness tests on 100 images...")
print(f"{'Condition':<15} {'Accuracy':>10} {'Avg Confidence':>15}")
print("-" * 45)

for img_path, label_idx in samples:
    true_name = ds.classes[label_idx]
    if true_name not in name_to_idx:
        continue
    true_idx = name_to_idx[true_name]

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for cond_name, perturb_fn in conditions.items():
        perturbed = perturb_fn(img_rgb.copy())
        pred_idx, conf = get_prediction(perturbed)
        results[cond_name]["total"] += 1
        results[cond_name]["confidences"].append(conf)
        if pred_idx == true_idx:
            results[cond_name]["correct"] += 1

# ── print results ─────────────────────────────────────────────────────────────
final = {}
for cond, data in results.items():
    acc  = data["correct"] / data["total"] if data["total"] > 0 else 0
    conf = np.mean(data["confidences"])
    final[cond] = {"accuracy": round(acc, 4), "avg_confidence": round(float(conf), 4)}
    print(f"{cond:<15} {acc:>10.2%} {conf:>15.2%}")

# ── save results ──────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "robustness_results.json", "w") as f:
    json.dump(final, f, indent=2)

# ── bar chart ─────────────────────────────────────────────────────────────────
conditions_list = list(final.keys())
accuracies      = [final[c]["accuracy"] * 100 for c in conditions_list]
colors          = ["#22c55e", "#f97316", "#ef4444", "#3b82f6", "#a855f7"]

plt.figure(figsize=(10, 6))
bars = plt.bar(conditions_list, accuracies, color=colors, width=0.5, edgecolor="white")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Model Robustness Under Different Conditions", fontsize=14, fontweight="bold")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
plt.axhline(y=92.59, color="gray", linestyle="--", label="Baseline accuracy (92.59%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "robustness_chart.png", dpi=150)
plt.close()
print(f"\nSaved → outputs/robustness/robustness_chart.png")
print("Saved → outputs/robustness/robustness_results.json")
print("\nRobustness testing complete!")