"""
SEM Defect Classification -- Streamlit demo.

Uses ONLY model_sem_burst.onnx via ONNX Runtime. Does NOT import torch and
does NOT load best_sem_model_burst.pth -- this app is meant to demonstrate
the ONNX deployment path verified in Phase 3, independent of the training
stack.

Run:
    streamlit run app.py
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Reuse the exact preprocessing already verified against PyTorch in Phase 3
# (compare_pytorch_onnx.py showed 100% prediction agreement using this same
# function via onnx_only_inference.py). Importing it here avoids duplicating
# and risking preprocessing drift between the two.
from onnx_only_inference import preprocess

# Resolve paths relative to this file's own directory, not the process's
# current working directory -- so `streamlit run app.py` works the same way
# regardless of where it's launched from (e.g. from a parent folder, or via
# a launcher script that cd's elsewhere first).
APP_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = APP_DIR / "model_sem_burst.onnx"
CLASS_MAPPING_PATH = APP_DIR / "model_sem_burst.onnx.class_mapping.json"
# model_sem_burst.onnx.data is loaded implicitly by ONNX Runtime -- it looks
# for it next to the .onnx file using the same directory, so as long as
# ONNX_MODEL_PATH is correct, the external-data file resolves automatically.

# Must match detector.py's NO_DEFECT_LABELS["sem"] = {"Clean"} (verified in
# tests/test_detector.py). Kept as a plain constant here, rather than
# importing detector.py, because detector.py imports torch and this app is
# required to stay PyTorch-free.
NO_DEFECT_LABEL = "Clean"


@st.cache_resource
def load_session_and_classes():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASS_MAPPING_PATH}")

    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    with open(CLASS_MAPPING_PATH) as f:
        mapping = json.load(f)
    classes = mapping["classes"]

    if NO_DEFECT_LABEL not in classes:
        raise ValueError(
            f"Expected no-defect label '{NO_DEFECT_LABEL}' not found in classes {classes}. "
            f"Refusing to start -- defect/no-defect status would be wrong for every prediction."
        )
    return session, classes


def run_inference(session, classes, image: Image.Image):
    x = preprocess(image)
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    out = session.run(None, {input_name: x})[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    logits = out[0]
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    has_defect = pred_class != NO_DEFECT_LABEL

    return pred_class, confidence, has_defect, latency_ms


def main():
    st.set_page_config(page_title="SEM Defect Classification", page_icon=None, layout="centered")

    st.title("SEM Defect Classification")
    st.caption("ONNX Runtime inference -- model_sem_burst.onnx")

    try:
        session, classes = load_session_and_classes()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    with st.expander("Model info"):
        st.write(f"Classes ({len(classes)}): {classes}")
        st.write(f"No-defect label: {NO_DEFECT_LABEL}")
        st.write(f"Backend: ONNX Runtime, providers={session.get_providers()}")

    uploaded_file = st.file_uploader("Upload SEM Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a .jpg, .jpeg, or .png SEM image to classify.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()  # force decode now, so corrupt files fail here, not later
    except (UnidentifiedImageError, OSError) as e:
        st.error(f"Could not read this file as an image ({e}). Please upload a valid JPG or PNG.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running ONNX inference..."):
        try:
            pred_class, confidence, has_defect, latency_ms = run_inference(session, classes, image)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    st.subheader("Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", pred_class)
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with col2:
        status = "DEFECT DETECTED" if has_defect else "NO DEFECT"
        if has_defect:
            st.error(f"Status: {status}")
        else:
            st.success(f"Status: {status}")
        st.metric("Inference Latency", f"{latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
