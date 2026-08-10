"""
Proves ONNX inference works standalone: this script imports NOTHING from
model.py, detector.py, or any training code. Only onnxruntime, numpy, and
PIL. This is what a deployment environment would actually run.

Usage:
    python onnx_only_inference.py --onnx_model model_sem_burst.onnx \
        --class_mapping model_sem_burst.onnx.class_mapping.json \
        --image /path/to/some_image.png
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess(image_input):
    """Accepts either a file path (str) or an already-opened PIL Image
    (e.g. from Streamlit's file_uploader, which gives us an in-memory
    object rather than a path). Same transform either way."""
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input
    img = img.convert("RGB").resize((96, 96), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, 0-1
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr.astype(np.float32)


def main(args):
    with open(args.class_mapping) as f:
        classes = json.load(f)["classes"]

    session = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    x = preprocess(args.image)
    out = session.run(None, {input_name: x})[0]
    pred_idx = int(np.argmax(out, axis=1)[0])
    probs = np.exp(out[0]) / np.sum(np.exp(out[0]))

    print(f"Predicted class: {classes[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", type=str, required=True)
    parser.add_argument("--class_mapping", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    for p in (args.onnx_model, args.class_mapping, args.image):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")

    main(args)
