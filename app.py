"""
SEM Defect Classification -- Streamlit App

Features:
- Single and multiple SEM image uploads
- ONNX Runtime inference
- Basic SEM/non-SEM input validation
- Per-image prediction, confidence, status, latency
- Batch progress tracking
- Batch summary
- Classification results table
- CSV export
- Class distribution
- Individual image inspection

Deployment path is PyTorch-free.
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


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ONNX_MODEL_PATH = BASE_DIR / "model_sem_burst.onnx"

CLASS_MAPPING_PATH = (
    BASE_DIR / "model_sem_burst.onnx.class_mapping.json"
)

NO_DEFECT_LABEL = "Clean"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SEM Defect Classification",
    page_icon="🔬",
    layout="wide",
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_session_and_classes():
    """
    Load ONNX model and class mapping once.
    """

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_MODEL_PATH}"
        )

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_MAPPING_PATH}"
        )

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    classes = mapping["classes"]

    if NO_DEFECT_LABEL not in classes:
        raise ValueError(
            f"Expected no-defect label '{NO_DEFECT_LABEL}' "
            f"not found in classes {classes}. "
            "Refusing to start because defect/no-defect "
            "status could be incorrect."
        )

    return session, classes


# =========================================================
# SEM IMAGE VALIDATION
# =========================================================

def is_likely_sem_image(image):
    """
    Basic safety validation to reject obviously non-SEM images.

    This is NOT a trained SEM-vs-non-SEM classifier.

    It is intended to reject common non-SEM inputs such as:
    - signatures
    - documents
    - certificates
    - screenshots
    - strongly colored photographs
    - nearly blank images
    - extremely low-detail images
    """

    image = image.convert("RGB")

    width, height = image.size

    # -----------------------------------------------------
    # 1. Resolution validation
    # -----------------------------------------------------

    if width < 64 or height < 64:
        return False, "Image resolution is too small."

    # -----------------------------------------------------
    # 2. Convert image to numpy
    # -----------------------------------------------------

    arr = np.asarray(image).astype(np.float32)

    gray = np.asarray(
        image.convert("L")
    ).astype(np.float32)

    # -----------------------------------------------------
    # 3. Color validation
    # -----------------------------------------------------

    channel_difference = np.mean(
        np.abs(arr[:, :, 0] - arr[:, :, 1])
        + np.abs(arr[:, :, 1] - arr[:, :, 2])
    )

    # Strongly colored photographs are unlikely to be SEM.
    if channel_difference > 35:
        return (
            False,
            "Image does not appear to be a grayscale SEM image.",
        )

    # -----------------------------------------------------
    # 4. Image statistics
    # -----------------------------------------------------

    mean_intensity = float(np.mean(gray))
    texture_strength = float(np.std(gray))

    bright_ratio = float(
        np.mean(gray > 245)
    )

    dark_ratio = float(
        np.mean(gray < 30)
    )

    # -----------------------------------------------------
    # 5. Gradient / edge estimation
    # -----------------------------------------------------

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    edge_strength = float(
        np.mean(gx) + np.mean(gy)
    )

    # -----------------------------------------------------
    # 6. Reject mostly blank images
    # -----------------------------------------------------

    if bright_ratio > 0.92:
        return (
            False,
            "Image appears to be mostly blank and "
            "does not look like an SEM image.",
        )

    # -----------------------------------------------------
    # 7. Reject mostly black images
    # -----------------------------------------------------

    if dark_ratio > 0.92:
        return (
            False,
            "Image contains insufficient visible structure "
            "for SEM classification.",
        )

    # -----------------------------------------------------
    # 8. Signature / document style detection
    # -----------------------------------------------------
    #
    # Many signatures and document scans have:
    # - mostly white background
    # - relatively small dark foreground
    # - low texture
    #
    # This is deliberately conservative.

    if (
        bright_ratio > 0.70
        and dark_ratio < 0.15
        and texture_strength < 70
    ):
        return (
            False,
            "Image appears to contain a mostly blank "
            "background and does not look like an SEM image.",
        )

    # -----------------------------------------------------
    # 9. Very low texture
    # -----------------------------------------------------

    if texture_strength < 15:
        return (
            False,
            "Image has insufficient texture for a typical SEM image.",
        )

    # -----------------------------------------------------
    # 10. Very low structural detail
    # -----------------------------------------------------

    if edge_strength < 8:
        return (
            False,
            "Image does not contain enough structural detail "
            "for a typical SEM image.",
        )

    return True, ""


# =========================================================
# ONNX INFERENCE
# =========================================================

def run_inference(session, classes, image):
    """
    Run ONNX inference on a single validated image.
    """

    x = preprocess(image)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()

    output = session.run(
        None,
        {
            input_name: x
        },
    )[0]

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    logits = output[0]

    # -----------------------------------------------------
    # Stable softmax
    # -----------------------------------------------------

    logits = logits - np.max(logits)

    probabilities = np.exp(logits)

    probabilities /= np.sum(probabilities)

    prediction_index = int(
        np.argmax(probabilities)
    )

    prediction = classes[prediction_index]

    confidence = float(
        probabilities[prediction_index]
    )

    status = (
        "NO DEFECT"
        if prediction == NO_DEFECT_LABEL
        else "DEFECT DETECTED"
    )

    return (
        prediction,
        confidence,
        status,
        latency_ms,
    )


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_uploaded_file(
    uploaded_file,
    session,
    classes,
):
    """
    Process one uploaded image.

    Validation happens BEFORE ONNX inference.

    A rejected/non-SEM image does not reach the
    defect classification model.
    """

    filename = uploaded_file.name

    try:

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        image = Image.open(uploaded_file)

        # Force actual image decoding
        image.load()

        image = image.convert("RGB")

        # -------------------------------------------------
        # SEM validation
        # -------------------------------------------------

        valid, validation_message = (
            is_likely_sem_image(image)
        )

        if not valid:

            return {
                "filename": filename,
                "prediction": "",
                "confidence": None,
                "status": "INVALID INPUT",
                "latency_ms": None,
                "error": validation_message,
                "image": image,
            }

        # -------------------------------------------------
        # ONNX inference
        # -------------------------------------------------

        (
            prediction,
            confidence,
            status,
            latency_ms,
        ) = run_inference(
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
    ) as e:

        return {
            "filename": filename,
            "prediction": "",
            "confidence": None,
            "status": "ERROR",
            "latency_ms": None,
            "error": str(e),
            "image": None,
        }


# =========================================================
# CSV GENERATION
# =========================================================

def create_csv(results):
    """
    Convert classification results to CSV.
    """

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(
        [
            "Filename",
            "Prediction",
            "Confidence",
            "Status",
            "Latency (ms)",
            "Error",
        ]
    )

    for result in results:

        confidence = result["confidence"]

        if confidence is None:
            confidence_text = ""
        else:
            confidence_text = (
                f"{confidence * 100:.2f}%"
            )

        latency = result["latency_ms"]

        if latency is None:
            latency_text = ""
        else:
            latency_text = f"{latency:.2f}"

        writer.writerow(
            [
                result["filename"],
                result["prediction"],
                confidence_text,
                result["status"],
                latency_text,
                result["error"],
            ]
        )

    return output.getvalue()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # =====================================================
    # HEADER
    # =====================================================

    st.title("🔬 SEM Defect Classification")

    st.caption(
        "Edge AI semiconductor defect detection using ONNX Runtime"
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    try:

        session, classes = (
            load_session_and_classes()
        )

    except Exception as e:

        st.error(
            f"Unable to load the classification model: {e}"
        )

        st.stop()

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander(
        "Model Information",
        expanded=False,
    ):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Inference Engine**")
            st.write("ONNX Runtime")

        with col2:
            st.write("**Execution Provider**")
            st.write("CPU")

        with col3:
            st.write("**Classes**")
            st.write(", ".join(classes))

        st.write(
            "**Input validation:** "
            "Basic SEM/non-SEM safety filter enabled"
        )

    # =====================================================
    # IMAGE UPLOAD
    # =====================================================

    st.subheader("Upload SEM Image(s)")

    uploaded_files = st.file_uploader(
        "Upload one or more SEM images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff",
        ],
        accept_multiple_files=True,
        help=(
            "Upload SEM images for defect classification. "
            "Non-SEM images such as signatures or documents "
            "may be rejected before inference."
        ),
    )

    # =====================================================
    # NO FILE
    # =====================================================

    if not uploaded_files:

        st.info(
            "Upload one or more SEM images to begin classification."
        )

        return

    # =====================================================
    # DISPLAY UPLOAD COUNT
    # =====================================================

    st.write(
        f"**{len(uploaded_files)} image(s) selected**"
    )

    # =====================================================
    # PREVIEW UPLOADS
    # =====================================================

    with st.expander(
        "View uploaded files",
        expanded=False,
    ):

        for file in uploaded_files:

            st.write(
                f"📄 {file.name} "
                f"({file.size / 1024:.1f} KB)"
            )

    # =====================================================
    # PROCESS BUTTON
    # =====================================================

    if st.button(
        "🚀 Run Classification",
        type="primary",
        width="stretch",
    ):

        results = []

        total_images = len(uploaded_files)

        progress_bar = st.progress(0)

        progress_text = st.empty()

        # -------------------------------------------------
        # Process images
        # -------------------------------------------------

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            progress_text.write(
                f"Processing image "
                f"{index + 1} of {total_images}: "
                f"`{uploaded_file.name}`"
            )

            result = process_uploaded_file(
                uploaded_file,
                session,
                classes,
            )

            results.append(result)

            progress_bar.progress(
                (index + 1) / total_images
            )

        progress_text.empty()

        st.success(
            f"Finished processing {total_images} image(s)."
        )

        # Save results in session state so the user
        # can interact with them after reruns.

        st.session_state["results"] = results

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    if "results" not in st.session_state:

        return

    results = st.session_state["results"]

    # =====================================================
    # BATCH SUMMARY
    # =====================================================

    st.subheader("Batch Summary")

    total = len(results)

    successful = sum(
        1
        for r in results
        if r["status"]
        in [
            "DEFECT DETECTED",
            "NO DEFECT",
        ]
    )

    failed = sum(
        1
        for r in results
        if r["status"] == "ERROR"
    )

    invalid = sum(
        1
        for r in results
        if r["status"] == "INVALID INPUT"
    )

    defects = sum(
        1
        for r in results
        if r["status"] == "DEFECT DETECTED"
    )

    clean = sum(
        1
        for r in results
        if r["status"] == "NO DEFECT"
    )

    latency_values = [
        r["latency_ms"]
        for r in results
        if r["latency_ms"] is not None
    ]

    if latency_values:

        average_latency = (
            sum(latency_values)
            / len(latency_values)
        )

    else:

        average_latency = 0.0

    # -----------------------------------------------------
    # Summary metrics
    # -----------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total",
            total,
        )

    with col2:
        st.metric(
            "Successful",
            successful,
        )

    with col3:
        st.metric(
            "Invalid / Error",
            invalid + failed,
        )

    with col4:
        st.metric(
            "Defects",
            defects,
        )

    with col5:
        st.metric(
            "Clean",
            clean,
        )

    st.metric(
        "Average Inference Latency",
        f"{average_latency:.2f} ms",
    )

    # =====================================================
    # CLASSIFICATION RESULTS
    # =====================================================

    st.subheader("Classification Results")

    table_data = []

    for result in results:

        confidence = result["confidence"]

        if confidence is None:
            confidence_text = "—"
        else:
            confidence_text = (
                f"{confidence * 100:.1f}%"
            )

        latency = result["latency_ms"]

        if latency is None:
            latency_text = "—"
        else:
            latency_text = (
                f"{latency:.2f} ms"
            )

        table_data.append(
            {
                "Filename": result["filename"],
                "Prediction": (
                    result["prediction"]
                    if result["prediction"]
                    else "—"
                ),
                "Confidence": confidence_text,
                "Status": result["status"],
                "Latency": latency_text,
                "Error": result["error"],
            }
        )

    st.dataframe(
        table_data,
        width="stretch",
        hide_index=True,
    )

    # =====================================================
    # CSV DOWNLOAD
    # =====================================================

    csv_data = create_csv(results)

    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_data,
        file_name="sem_classification_results.csv",
        mime="text/csv",
        width="stretch",
    )

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    st.subheader("Class Distribution")

    class_counts = {
        class_name: 0
        for class_name in classes
    }

    for result in results:

        prediction = result["prediction"]

        if prediction in class_counts:

            class_counts[prediction] += 1

    if any(
        count > 0
        for count in class_counts.values()
    ):

        st.bar_chart(class_counts)

    else:

        st.info(
            "No valid SEM predictions available "
            "for class distribution."
        )

    # =====================================================
    # INPUT VALIDATION DETAILS
    # =====================================================

    invalid_results = [
        r
        for r in results
        if r["status"] == "INVALID INPUT"
    ]

    if invalid_results:

        st.subheader("⚠️ Rejected Inputs")

        st.warning(
            "Some uploaded images were rejected before "
            "ONNX inference because they did not appear "
            "to be valid SEM images."
        )

        for result in invalid_results:

            st.write(
                f"**{result['filename']}** — "
                f"{result['error']}"
            )

    # =====================================================
    # ERROR DETAILS
    # =====================================================

    error_results = [
        r
        for r in results
        if r["status"] == "ERROR"
    ]

    if error_results:

        st.subheader("⚠️ Processing Errors")

        for result in error_results:

            st.error(
                f"{result['filename']}: "
                f"{result['error']}"
            )

    # =====================================================
    # INDIVIDUAL IMAGE INSPECTION
    # =====================================================

    valid_images = [
        r
        for r in results
        if r["image"] is not None
    ]

    if not valid_images:

        return

    st.subheader("Inspect Individual Image")

    filenames = [
        r["filename"]
        for r in valid_images
    ]

    selected_filename = st.selectbox(
        "Select an image",
        filenames,
    )

    selected_result = next(
        r
        for r in valid_images
        if r["filename"] == selected_filename
    )

    image_col, result_col = st.columns(
        [2, 1]
    )

    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    with image_col:

        st.image(
            selected_result["image"],
            caption=selected_result["filename"],
            width="stretch",
        )

    # -----------------------------------------------------
    # Result information
    # -----------------------------------------------------

    with result_col:

        status = selected_result["status"]

        if status == "DEFECT DETECTED":

            st.error(
                f"Status: {status}"
            )

        elif status == "NO DEFECT":

            st.success(
                f"Status: {status}"
            )

        elif status == "INVALID INPUT":

            st.warning(
                f"Status: {status}"
            )

        else:

            st.error(
                f"Status: {status}"
            )

        prediction = selected_result[
            "prediction"
        ]

        if prediction:

            st.metric(
                "Prediction",
                prediction,
            )

        confidence = selected_result[
            "confidence"
        ]

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%",
            )

        latency = selected_result[
            "latency_ms"
        ]

        if latency is not None:

            st.metric(
                "Inference Latency",
                f"{latency:.2f} ms",
            )

        error = selected_result[
            "error"
        ]

        if error:

            st.info(error)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()