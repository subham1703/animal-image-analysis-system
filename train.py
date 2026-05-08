"""
train.py
--------
Fine-tune EfficientNet-B0 on an animal image dataset.

Expected dataset layout (ImageFolder format):
    data/
      train/
        butterfly/  cat/  chicken/  cow/  dog/
        elephant/  horse/  sheep/  spider/  squirrel/
      val/
        (same structure)
      test/
        (same structure)

Dataset: Animals-10 (https://www.kaggle.com/datasets/alessiocorrado99/animals10)
         ~28 000 images, 10 classes.

Usage
-----
    python train.py --data_dir ./data --epochs 20 --batch_size 32 --lr 1e-4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pipeline import build_efficientnet, NORM_MEAN, NORM_STD, IMG_SIZE, DEVICE


# ── augmentation transforms ─────────────────────────────────────────────────
def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(20),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])


# ── training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, correct / total


# ── main ─────────────────────────────────────────────────────────────────────
def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── datasets ────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=get_transforms("val"))

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")

    # Save class mapping for inference
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(train_ds.classes, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # ── model ────────────────────────────────────────────────────────────
    model     = build_efficientnet(num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training ─────────────────────────────────────────────────────────
    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc  = 0.0
    best_path = output_dir / "efficientnet_b0_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.3%} | "
            f"Val loss {vl_loss:.4f} acc {vl_acc:.3%} | "
            f"{elapsed:.0f}s"
        )

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved ({best_acc:.3%})")

    torch.save(model.state_dict(), output_dir / "efficientnet_b0_last.pth")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3%}")
    print(f"Weights saved to: {best_path}")


# ── ResNet-101 baseline training (Phase 6 comparison) ───────────────────────
def train_resnet101_baseline(args):
    """Train ResNet-101 baseline for comparison table."""
    import torchvision.models as tvm
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=get_transforms("val"))
    num_classes = len(train_ds.classes)

    model = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"[ResNet101] Epoch {epoch:3d}/{args.epochs} | "
              f"Val loss {vl_loss:.4f} acc {vl_acc:.3%} | {time.time()-t0:.0f}s")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), output_dir / "resnet101_best.pth")

    print(f"ResNet-101 best val accuracy: {best_acc:.3%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 animal classifier")
    parser.add_argument("--data_dir",   default="data",    help="Root dataset directory")
    parser.add_argument("--output_dir", default="weights", help="Where to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--baseline",   action="store_true",
                        help="Also train ResNet-101 baseline")
    args = parser.parse_args()

    main(args)
    if args.baseline:
        train_resnet101_baseline(args)
