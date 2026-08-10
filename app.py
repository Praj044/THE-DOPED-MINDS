"""
SEM Defect Classification - Streamlit App

Supports:
- Single or multiple SEM image uploads
- ONNX Runtime inference
- Batch progress tracking
- Results table
- CSV export
- Class distribution
- Per-image inspection

No PyTorch is required at runtime.
"""

import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, UnidentifiedImageError

from onnx_only_inference import preprocess


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

ONNX_MODEL_PATH = APP_DIR / "model_sem_burst.onnx"
CLASS_MAPPING_PATH = APP_DIR / "model_sem_burst.onnx.class_mapping.json"

NO_DEFECT_LABEL = "Clean"


# ---------------------------------------------------------
# Load ONNX model once
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {ONNX_MODEL_PATH}"
        )

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    with open(CLASS_MAPPING_PATH, encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    if NO_DEFECT_LABEL not in classes:
        raise ValueError(
            f"'{NO_DEFECT_LABEL}' not found in class list: {classes}"
        )

    return session, classes


# ---------------------------------------------------------
# Single image inference
# ---------------------------------------------------------

def run_inference(session, classes, image):
    x = preprocess(image)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()

    output = session.run(
        None,
        {input_name: x},
    )[0]

    latency_ms = (time.perf_counter() - start) * 1000

    logits = output[0]

    # Stable softmax
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

    prediction_index = int(np.argmax(probabilities))

    prediction = classes[prediction_index]
    confidence = float(probabilities[prediction_index])

    status = (
        "NO DEFECT"
        if prediction == NO_DEFECT_LABEL
        else "DEFECT DETECTED"
    )

    return prediction, confidence, status, latency_ms


# ---------------------------------------------------------
# Process one uploaded file
# ---------------------------------------------------------

def process_file(uploaded_file, session, classes):
    filename = uploaded_file.name

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")

        prediction, confidence, status, latency_ms = run_inference(
            session,
            classes,
            image,
        )

        return {
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence,
            "status": status,
            "latency_ms": latency_ms,
            "error": "",
            "image": image,
        }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:

        return {
            "filename": filename,
            "prediction": "",
            "confidence": None,
            "status": "ERROR",
            "latency_ms": None,
            "error": str(error),
            "image": None,
        }


# ---------------------------------------------------------
# CSV generation
# ---------------------------------------------------------

def create_csv(results):
    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(
        [
            "Filename",
            "Prediction",
            "Confidence (%)",
            "Status",
            "Latency (ms)",
            "Error",
        ]
    )

    for result in results:
        confidence = result["confidence"]

        writer.writerow(
            [
                result["filename"],
                result["prediction"],
                (
                    f"{confidence * 100:.2f}"
                    if confidence is not None
                    else ""
                ),
                result["status"],
                (
                    f"{result['latency_ms']:.2f}"
                    if result["latency_ms"] is not None
                    else ""
                ),
                result["error"],
            ]
        )

    return output.getvalue()


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def main():

    st.set_page_config(
        page_title="SEM Defect Classification",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 SEM Defect Classification")

    st.caption(
        "Edge AI semiconductor defect detection using "
        "ONNX Runtime"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    try:
        session, classes = load_model()

    except Exception as error:
        st.error(f"Failed to load model: {error}")
        st.stop()

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    with st.expander("Model Information"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Classes", len(classes))

        with col2:
            st.metric("Backend", "ONNX Runtime")

        with col3:
            st.metric("Input Size", "96 × 96")

        st.write("Classes:")
        st.write(", ".join(classes))

        st.write(
            f"No-defect class: **{NO_DEFECT_LABEL}**"
        )

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload SEM image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can select one or multiple SEM images.",
    )

    if not uploaded_files:

        st.info(
            "Upload one or multiple SEM images to start classification."
        )

        return

    # -----------------------------------------------------
    # Batch processing
    # -----------------------------------------------------

    st.subheader(
        f"Processing {len(uploaded_files)} image(s)"
    )

    progress_bar = st.progress(0)

    status_text = st.empty()

    results = []

    total = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):

        status_text.write(
            f"Processing {index + 1} / {total}: "
            f"{uploaded_file.name}"
        )

        result = process_file(
            uploaded_file,
            session,
            classes,
        )

        results.append(result)

        progress_bar.progress(
            (index + 1) / total
        )

    status_text.success(
        f"Finished processing {total} image(s)."
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    successful = [
        r for r in results
        if r["status"] != "ERROR"
    ]

    failed = [
        r for r in results
        if r["status"] == "ERROR"
    ]

    defects = [
        r for r in successful
        if r["status"] == "DEFECT DETECTED"
    ]

    clean = [
        r for r in successful
        if r["status"] == "NO DEFECT"
    ]

    latencies = [
        r["latency_ms"]
        for r in successful
        if r["latency_ms"] is not None
    ]

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    st.subheader("Batch Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total", total)

    with col2:
        st.metric("Successful", len(successful))

    with col3:
        st.metric("Failed", len(failed))

    with col4:
        st.metric("Defects", len(defects))

    with col5:
        st.metric("Clean", len(clean))

    st.metric(
        "Average Inference Latency",
        f"{average_latency:.2f} ms",
    )

    # -----------------------------------------------------
    # Results table
    # -----------------------------------------------------

    st.subheader("Classification Results")

    table_data = []

    for result in results:

        confidence = result["confidence"]

        table_data.append(
            {
                "Filename": result["filename"],
                "Prediction": result["prediction"],
                "Confidence": (
                    f"{confidence * 100:.1f}%"
                    if confidence is not None
                    else "-"
                ),
                "Status": result["status"],
                "Latency": (
                    f"{result['latency_ms']:.2f} ms"
                    if result["latency_ms"] is not None
                    else "-"
                ),
                "Error": result["error"],
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # CSV download
    # -----------------------------------------------------

    csv_data = create_csv(results)

    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="sem_classification_results.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    st.subheader("Class Distribution")

    class_counts = {
        class_name: 0
        for class_name in classes
    }

    for result in successful:

        prediction = result["prediction"]

        if prediction in class_counts:
            class_counts[prediction] += 1

    st.bar_chart(class_counts)

    # -----------------------------------------------------
    # Individual image inspection
    # -----------------------------------------------------

    st.subheader("Inspect Individual Image")

    selectable_results = [
        r for r in results
        if r["image"] is not None
    ]

    if selectable_results:

        selected_filename = st.selectbox(
            "Select an image",
            [
                r["filename"]
                for r in selectable_results
            ],
        )

        selected = next(
            r
            for r in selectable_results
            if r["filename"] == selected_filename
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                selected["image"],
                caption=selected["filename"],
                width="stretch",
            )

        with col2:

            st.write(
                f"### Prediction: {selected['prediction']}"
            )

            st.write(
                f"**Confidence:** "
                f"{selected['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Status:** {selected['status']}"
            )

            st.write(
                f"**Inference latency:** "
                f"{selected['latency_ms']:.2f} ms"
            )

    # -----------------------------------------------------
    # Errors
    # -----------------------------------------------------

    if failed:

        st.subheader("Files That Failed")

        for result in failed:

            st.error(
                f"{result['filename']}: "
                f"{result['error']}"
            )


if __name__ == "__main__":
    main()