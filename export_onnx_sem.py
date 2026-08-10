"""
Exports a SEM SimpleCNN checkpoint to ONNX. Unlike the original version of
this script, num_classes is NOT taken on faith from a --num_classes flag --
it's derived from the class_mapping.json that train_sem_burst.py saves next
to its checkpoint, so the exported model's class order is provably tied to
what the checkpoint was actually trained with, not assumed to match some
hardcoded default.

Usage:
    python export_onnx_sem.py --model_path best_sem_model_burst.pth \
        --class_mapping best_sem_model_burst.pth.class_mapping.json \
        --onnx_path model_sem_burst.onnx
"""
import argparse
import json
import os
import sys

import torch
import torch.onnx

from model import SimpleCNN

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def export_model(model_path, onnx_path, class_mapping_path):
    if not os.path.exists(model_path):
        raise SystemExit(f"Error: Model not found at {model_path}")
    if not os.path.exists(class_mapping_path):
        raise SystemExit(
            f"Error: class mapping not found at {class_mapping_path}. "
            f"This file is produced by train_sem_burst.py alongside the checkpoint -- "
            f"refusing to guess num_classes/class order without it."
        )

    with open(class_mapping_path) as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    num_classes = len(classes)
    print(f"Loaded class mapping from {class_mapping_path}: {classes}")

    device = torch.device("cpu")
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} ({num_classes} classes)")

    dummy_input = torch.randn(1, 3, 96, 96)

    print(f"Exporting to {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Export complete!")
    except Exception as e:
        raise SystemExit(f"Error during export: {e}")

    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully (onnx.checker.check_model passed).")
    except ImportError:
        print("ONNX library not installed, skipping structural verification.")
    except Exception as e:
        print(f"Verification failed: {e}")

    # Sidecar file: ties the exported .onnx file to the exact class order it
    # was exported with, so downstream ONNX-only consumers (no access to the
    # original .pth manifest) still don't have to assume anything.
    sidecar_path = onnx_path + ".class_mapping.json"
    with open(sidecar_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote class mapping sidecar to {sidecar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a SEM checkpoint to ONNX using its class_mapping.json.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--class_mapping", type=str, required=True,
                         help="Path to the *.class_mapping.json saved by train_sem_burst.py")
    parser.add_argument("--onnx_path", type=str, default="model_sem_burst.onnx")
    args = parser.parse_args()

    export_model(args.model_path, args.onnx_path, args.class_mapping)
