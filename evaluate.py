"""
Evaluate a trained checkpoint on a held-out folder of labeled images and
report accuracy, per-class precision/recall/F1, and a confusion matrix.

This did not exist in the original project -- there was no record anywhere
of how well either model actually performs. Run this against a test split
that was NOT used for training/validation before making a deployment call.

Usage:
    python evaluate.py --data_dir path/to/test_set --model_path best_sem_model.pth \
        --dataset sem
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SimpleCNN
from data_loader import WM811KDataset
from detector import WM811K_CLASSES, SEM_CLASSES


def evaluate(data_dir, model_path, dataset, batch_size=32):
    classes = SEM_CLASSES if dataset == "sem" else WM811K_CLASSES
    num_classes = len(classes)

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = WM811KDataset(data_dir, transform=transform)
    if ds.classes != classes:
        print(f"WARNING: folder classes {ds.classes} != expected {classes}. "
              f"Metrics below will use the folder's own class indices; double check "
              f"this test set matches the model you're evaluating.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n = len(ds.classes)
    confusion = [[0] * n for _ in range(n)]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
                confusion[true_idx][pred_idx] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(n))
    accuracy = correct / total if total else 0.0

    print(f"\nEvaluated {total} images from {data_dir}")
    print(f"Overall accuracy: {accuracy:.4f}\n")

    print(f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    precisions, recalls, f1s = [], [], []
    for i, cls in enumerate(ds.classes):
        tp = confusion[i][i]
        support = sum(confusion[i])
        pred_total = sum(confusion[r][i] for r in range(n))
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(f"{cls:<16}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{support:>10}")
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"\nMacro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{c[:8]:>10}" for c in ds.classes)
    print(" " * 16 + header)
    for i, cls in enumerate(ds.classes):
        row = "".join(f"{v:>10}" for v in confusion[i])
        print(f"{cls:<16}{row}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a held-out test set.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to a held-out image-folder test set (root/<class>/*.png)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="sem", choices=["sem", "wm811k"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Path not found: {args.data_dir}")
    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model not found: {args.model_path}")

    evaluate(args.data_dir, args.model_path, args.dataset, args.batch_size)
