"""
Loads a PyTorch checkpoint and its exported ONNX model, runs BOTH over the
same set of real images with identical preprocessing, and reports:
  - prediction agreement (exact match on argmax class)
  - which images/classes disagree, if any
  - average per-image latency for each backend
  - file sizes for both artifacts

Runs ONNX purely through onnxruntime -- no PyTorch/training-script code is
required for that half, verifying ONNX inference works standalone.

Usage:
    python compare_pytorch_onnx.py \
        --data_dir /path/burst_split/val \
        --pytorch_model best_sem_model_burst.pth \
        --onnx_model model_sem_burst.onnx \
        --class_mapping best_sem_model_burst.pth.class_mapping.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN


def build_transform():
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def list_images(data_dir, classes):
    items = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            fp = os.path.join(cls_dir, f)
            if os.path.isfile(fp):
                items.append((fp, cls))
    return items


def main(args):
    with open(args.class_mapping) as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    print(f"Classes: {classes}")

    items = list_images(args.data_dir, classes)
    print(f"Found {len(items)} images under {args.data_dir}")
    if not items:
        raise SystemExit("No images found -- check --data_dir.")

    transform = build_transform()

    # --- Load PyTorch model ---
    device = torch.device("cpu")
    pt_model = SimpleCNN(num_classes=len(classes))
    pt_model.load_state_dict(torch.load(args.pytorch_model, map_location=device))
    pt_model.eval()

    # --- Load ONNX model (standalone, no dependency on training code) ---
    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    # --- Run inference for every image, both backends ---
    pt_preds, onnx_preds, true_labels, filenames = [], [], [], []
    pt_latencies, onnx_latencies = [], []

    for fp, true_cls in items:
        img = Image.open(fp).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # (1,3,96,96)
        np_input = tensor.numpy().astype(np.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            pt_out = pt_model(tensor)
        pt_pred = int(torch.argmax(pt_out, dim=1).item())
        t1 = time.perf_counter()
        pt_latencies.append(t1 - t0)

        t0 = time.perf_counter()
        onnx_out = ort_session.run(None, {input_name: np_input})[0]
        onnx_pred = int(np.argmax(onnx_out, axis=1)[0])
        t1 = time.perf_counter()
        onnx_latencies.append(t1 - t0)

        pt_preds.append(pt_pred)
        onnx_preds.append(onnx_pred)
        true_labels.append(true_cls)
        filenames.append(fp)

    # --- Agreement ---
    matches = sum(1 for a, b in zip(pt_preds, onnx_preds) if a == b)
    total = len(items)
    agreement_pct = 100.0 * matches / total
    mismatches = [
        (filenames[i], true_labels[i], classes[pt_preds[i]], classes[onnx_preds[i]])
        for i in range(total) if pt_preds[i] != onnx_preds[i]
    ]

    # --- Accuracy of each backend vs ground truth (sanity, same preprocessing) ---
    pt_correct = sum(1 for i in range(total) if classes[pt_preds[i]] == true_labels[i])
    onnx_correct = sum(1 for i in range(total) if classes[onnx_preds[i]] == true_labels[i])

    # --- Sizes ---
    pt_size = os.path.getsize(args.pytorch_model)
    onnx_size = os.path.getsize(args.onnx_model)
    onnx_data_path = args.onnx_model + ".data"
    onnx_data_size = os.path.getsize(onnx_data_path) if os.path.exists(onnx_data_path) else 0
    onnx_total_size = onnx_size + onnx_data_size

    # --- Latency (skip first 5 as warmup for both) ---
    warmup = min(5, total - 1) if total > 1 else 0
    pt_avg_ms = (sum(pt_latencies[warmup:]) / len(pt_latencies[warmup:])) * 1000
    onnx_avg_ms = (sum(onnx_latencies[warmup:]) / len(onnx_latencies[warmup:])) * 1000
    speedup = pt_avg_ms / onnx_avg_ms if onnx_avg_ms > 0 else float("inf")

    print("\n" + "=" * 60)
    print("PYTORCH vs ONNX COMPARISON")
    print("=" * 60)
    print(f"Total images evaluated: {total}")
    print(f"PyTorch checkpoint size: {pt_size} bytes ({pt_size/1024:.1f} KB)")
    print(f"ONNX graph file size:    {onnx_size} bytes ({onnx_size/1024:.1f} KB)")
    if onnx_data_size:
        print(f"ONNX external weights (.onnx.data): {onnx_data_size} bytes ({onnx_data_size/1024:.1f} KB)")
    print(f"ONNX TOTAL size (graph + weights): {onnx_total_size} bytes ({onnx_total_size/1024:.1f} KB)")
    print(f"PyTorch accuracy vs ground truth: {pt_correct}/{total} = {100*pt_correct/total:.2f}%")
    print(f"ONNX accuracy vs ground truth:    {onnx_correct}/{total} = {100*onnx_correct/total:.2f}%")
    print(f"\nPrediction agreement (PyTorch vs ONNX): {matches}/{total} = {agreement_pct:.2f}%")
    print(f"Mismatched predictions: {len(mismatches)}")
    if mismatches:
        print("\nMismatch details (file, true_class, pytorch_pred, onnx_pred):")
        for fn, true_cls, pt_cls, onnx_cls in mismatches:
            print(f"  {os.path.basename(fn):<30} true={true_cls:<15} pt={pt_cls:<15} onnx={onnx_cls}")
    print(f"\nAverage PyTorch latency (post-warmup, {len(pt_latencies)-warmup} images): {pt_avg_ms:.4f} ms/image")
    print(f"Average ONNX latency    (post-warmup, {len(onnx_latencies)-warmup} images): {onnx_avg_ms:.4f} ms/image")
    print(f"ONNX speedup vs PyTorch: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX predictions and latency on real images.")
    parser.add_argument("--data_dir", type=str, required=True, help="Image-folder dataset to test on")
    parser.add_argument("--pytorch_model", type=str, required=True)
    parser.add_argument("--onnx_model", type=str, required=True)
    parser.add_argument("--class_mapping", type=str, required=True)
    args = parser.parse_args()

    for p, name in [(args.data_dir, "data_dir"), (args.pytorch_model, "pytorch_model"),
                     (args.onnx_model, "onnx_model"), (args.class_mapping, "class_mapping")]:
        if not os.path.exists(p):
            raise SystemExit(f"{name} not found: {p}")

    main(args)
