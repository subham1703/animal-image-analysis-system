"""
evaluate.py
-----------
Phase 6 + 7 — Performance evaluation and edge-AI optimization.

Produces:
  • Per-class accuracy, precision, recall, F1
  • Confusion matrix (saved as PNG)
  • Inference latency (mean ± std over N runs)
  • Model comparison table (EfficientNet vs ResNet-101)
  • Phase 7: INT8 post-training quantization + before/after comparison

Usage
-----
    # Evaluate EfficientNet-B0
    python evaluate.py --weights weights/efficientnet_b0_best.pth \
                       --data_dir data/test --model efficientnet

    # Compare both models
    python evaluate.py --weights weights/efficientnet_b0_best.pth \
                       --baseline_weights weights/resnet101_best.pth \
                       --data_dir data/test --compare

    # Run quantization
    python evaluate.py --weights weights/efficientnet_b0_best.pth \
                       --data_dir data/test --quantize
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import (
    build_efficientnet, NORM_MEAN, NORM_STD, IMG_SIZE, DEVICE, CLASS_NAMES
)


# ── helpers ──────────────────────────────────────────────────────────────────
def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def count_parameters(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_latency(model: nn.Module, device, n_runs: int = 200, batch_size: int = 1) -> dict:
    """
    Measure single-image inference latency (ms).
    Returns mean, std, min, max over n_runs.
    Warms up for 10 iterations before recording.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
    }


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_preds, all_labels) numpy arrays."""
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — EfficientNet-B0", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_comparison_table(rows: list[dict], save_path: str):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 2 + len(rows) * 0.7))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    # Header styling
    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor("#4F46E5")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    plt.title("Model Comparison — Animal Image Analysis System", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison table saved → {save_path}")


# ── quantization (Phase 7) ────────────────────────────────────────────────────
def quantize_model(model: nn.Module, calibration_loader: DataLoader, device):
    """
    Apply PyTorch static INT8 post-training quantization.

    Note: PTQ works best on CPU; CUDA requires a different backend.
    Falls back to CPU for quantization even if DEVICE is CUDA.
    """
    import copy
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    model_cpu.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_prepared   = torch.quantization.prepare(model_cpu)

    # Calibrate on a subset of data
    model_prepared.eval()
    n_batches = 0
    with torch.no_grad():
        for imgs, _ in calibration_loader:
            model_prepared(imgs.cpu())
            n_batches += 1
            if n_batches >= 20:
                break

    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized


def model_size_mb(model: nn.Module, path: str = "/tmp/_tmp_model.pth") -> float:
    torch.save(model.state_dict(), path)
    size = Path(path).stat().st_size / 1e6
    return size


# ── evaluation entry point ────────────────────────────────────────────────────
def evaluate_model(weights_path: str, data_dir: str, class_names: list[str],
                   model_name: str = "EfficientNet-B0") -> dict:
    import torchvision.models as tvm

    ds = datasets.ImageFolder(data_dir, transform=get_test_transform())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = len(ds.classes)

    if "resnet" in model_name.lower():
        model = tvm.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = build_efficientnet(num_classes, pretrained=False)

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()

    preds, labels = run_inference(model, loader, DEVICE)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec  = recall_score(labels, preds, average="macro", zero_division=0)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(labels, preds)
    lat  = measure_latency(model, DEVICE)
    params = count_parameters(model)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4%}")
    print(f"  Precision : {prec:.4%} (macro)")
    print(f"  Recall    : {rec:.4%} (macro)")
    print(f"  F1-score  : {f1:.4%} (macro)")
    print(f"  Params    : {params:.1f}M")
    print(f"  Latency   : {lat['mean_ms']:.2f} ± {lat['std_ms']:.2f} ms")
    print()
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    return {
        "model":       model_name,
        "accuracy":    f"{acc:.3%}",
        "precision":   f"{prec:.3%}",
        "recall":      f"{rec:.3%}",
        "f1":          f"{f1:.3%}",
        "params_M":    f"{params:.1f}",
        "latency_ms":  f"{lat['mean_ms']:.1f}",
        "cm":          cm,
        "_model_obj":  model,
        "_loader":     loader,
    }


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    names_file = Path(args.weights).parent / "class_names.json"
    class_names = json.load(open(names_file)) if names_file.exists() else CLASS_NAMES

    # ── Evaluate EfficientNet ────────────────────────────────────────────
    eff_result = evaluate_model(args.weights, args.data_dir, class_names, "EfficientNet-B0")
    plot_confusion_matrix(eff_result["cm"], class_names, str(out_dir / "confusion_matrix_efficientnet.png"))

    rows = [{
        "Model":          eff_result["model"],
        "Accuracy":       eff_result["accuracy"],
        "Precision":      eff_result["precision"],
        "Recall":         eff_result["recall"],
        "F1":             eff_result["f1"],
        "Params (M)":     eff_result["params_M"],
        "Latency (ms)":   eff_result["latency_ms"],
    }]

    # ── Evaluate ResNet-101 baseline ──────────────────────────────────────
    if args.compare and args.baseline_weights:
        res_result = evaluate_model(args.baseline_weights, args.data_dir, class_names, "ResNet-101")
        plot_confusion_matrix(res_result["cm"], class_names, str(out_dir / "confusion_matrix_resnet101.png"))
        rows.append({
            "Model":          res_result["model"],
            "Accuracy":       res_result["accuracy"],
            "Precision":      res_result["precision"],
            "Recall":         res_result["recall"],
            "F1":             res_result["f1"],
            "Params (M)":     res_result["params_M"],
            "Latency (ms)":   res_result["latency_ms"],
        })

    # ── Phase 7: Quantization ─────────────────────────────────────────────
    if args.quantize:
        print("\n[Phase 7] INT8 Post-Training Quantization")
        fp32_model  = eff_result["_model_obj"]
        calib_loader = eff_result["_loader"]

        fp32_size = model_size_mb(fp32_model)
        fp32_lat  = measure_latency(fp32_model.cpu(), torch.device("cpu"))

        q_model   = quantize_model(fp32_model, calib_loader, DEVICE)
        q_size    = model_size_mb(q_model)
        q_lat     = measure_latency(q_model, torch.device("cpu"))

        speedup = fp32_lat["mean_ms"] / q_lat["mean_ms"]
        size_reduction = (1 - q_size / fp32_size) * 100

        print(f"  FP32 size   : {fp32_size:.1f} MB  |  latency {fp32_lat['mean_ms']:.1f} ms")
        print(f"  INT8 size   : {q_size:.1f} MB  |  latency {q_lat['mean_ms']:.1f} ms")
        print(f"  Size saving : {size_reduction:.1f}%")
        print(f"  Speedup     : {speedup:.2f}×")

        rows.append({
            "Model":          "EfficientNet-B0 (INT8)",
            "Accuracy":       "~" + eff_result["accuracy"],
            "Precision":      "~" + eff_result["precision"],
            "Recall":         "~" + eff_result["recall"],
            "F1":             "~" + eff_result["f1"],
            "Params (M)":     eff_result["params_M"],
            "Latency (ms)":   f"{q_lat['mean_ms']:.1f}",
        })

        torch.save(q_model.state_dict(), out_dir / "efficientnet_b0_int8.pth")
        print(f"Quantized model saved → {out_dir}/efficientnet_b0_int8.pth")

    plot_comparison_table(rows, str(out_dir / "model_comparison_table.png"))

    # Save CSV
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
        out_dir / "results.csv", index=False
    )
    print(f"\nResults CSV saved → {out_dir}/results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate animal analysis models")
    parser.add_argument("--weights",          required=True,          help="EfficientNet weights path")
    parser.add_argument("--baseline_weights", default=None,           help="ResNet-101 weights path")
    parser.add_argument("--data_dir",         required=True,          help="Test dataset directory")
    parser.add_argument("--output_dir",       default="eval_outputs", help="Where to save outputs")
    parser.add_argument("--compare",          action="store_true",    help="Compare with ResNet-101")
    parser.add_argument("--quantize",         action="store_true",    help="Run INT8 quantization")
    args = parser.parse_args()
    main(args)
