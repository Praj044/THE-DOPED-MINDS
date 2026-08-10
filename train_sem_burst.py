"""
Trains SimpleCNN from scratch on a pre-built burst-aware split (two separate
folders: train/ and val/, each root/<class>/*.png). Unlike train_sem.py, this
script does NOT re-split anything internally -- the split was already fixed
and verified leak-free by make_burst_split.py, and this script trusts it as-is.

Outputs:
    <output>                  -- best checkpoint (state_dict), by val accuracy
    <output>.class_mapping.json   -- {"classes": [...], "class_to_idx": {...}}
    <output>.manifest.json        -- seed, paths, hyperparams, per-class counts,
                                      per-epoch train/val loss+accuracy history

Usage:
    python train_sem_burst.py --train_dir /path/burst_split/train \
        --val_dir /path/burst_split/val --epochs 30 --batch_size 16 \
        --lr 0.001 --seed 42 --output best_sem_model_burst.pth
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import WM811KDataset
from model import SimpleCNN


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(train_dir, val_dir):
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = WM811KDataset(train_dir, transform=train_transform)
    val_ds = WM811KDataset(val_dir, transform=val_transform)

    if train_ds.classes != val_ds.classes:
        raise RuntimeError(
            f"Class mismatch between train_dir and val_dir!\n"
            f"  train classes: {train_ds.classes}\n"
            f"  val classes:   {val_ds.classes}\n"
            f"Refusing to train with misaligned label indices. Fix the folders "
            f"so both sides have the exact same class subfolders."
        )

    # Sanity: also confirm no filename appears in both splits (defense in
    # depth on top of the leak check already run by make_burst_split.py).
    train_files = set(os.path.basename(p) for p in train_ds.samples)
    val_files = set(os.path.basename(p) for p in val_ds.samples)
    overlap = train_files & val_files
    if overlap:
        raise RuntimeError(
            f"Found {len(overlap)} filename(s) present in BOTH train_dir and val_dir: "
            f"{list(overlap)[:10]}... Refusing to train -- this would leak validation "
            f"images into training."
        )

    return train_ds, val_ds


def per_class_counts(dataset):
    counts = {cls: 0 for cls in dataset.classes}
    for path in dataset.samples:
        cls = os.path.basename(os.path.dirname(path))
        counts[cls] += 1
    return counts


def train(args):
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets(args.train_dir, args.val_dir)
    classes = train_ds.classes
    print(f"Classes ({len(classes)}): {classes}")

    train_counts = per_class_counts(train_ds)
    val_counts = per_class_counts(val_ds)
    print(f"Train per-class counts: {train_counts}")
    print(f"Val per-class counts:   {val_counts}")

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects / len(train_ds)

        model.eval()
        val_loss_total, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss_total += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss = val_loss_total / len(val_ds)
        val_acc = val_corrects / len(val_ds)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f}) to {args.output}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val acc: {best_val_acc:.4f}")

    class_mapping = {"classes": classes, "class_to_idx": train_ds.class_to_idx}
    with open(args.output + ".class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    manifest = {
        "seed": args.seed,
        "train_dir": os.path.abspath(args.train_dir),
        "val_dir": os.path.abspath(args.val_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "classes": classes,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    with open(args.output + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved class mapping to {args.output}.class_mapping.json")
    print(f"Saved training manifest to {args.output}.manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN from scratch on a pre-built burst-aware split.")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="best_sem_model_burst.pth")
    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        raise SystemExit(f"train_dir not found: {args.train_dir}")
    if not os.path.exists(args.val_dir):
        raise SystemExit(f"val_dir not found: {args.val_dir}")

    train(args)
