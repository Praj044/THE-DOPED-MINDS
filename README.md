# 🔬 THE-DOPED-MINDS

### Edge AI Semiconductor Defect Classification from SEM Images

**THE-DOPED-MINDS** is an Edge AI system for automatically classifying semiconductor defects from **Scanning Electron Microscope (SEM) images**. The project combines a CNN-based computer vision model, burst-aware dataset splitting, ONNX model optimization, and a Streamlit deployment for fast CPU inference.

🌐 **Live Demo:** https://infersem.streamlit.app/

---

## 🚀 Overview

Manual inspection of semiconductor SEM images can be time-consuming and prone to inconsistencies. This project provides an automated inference pipeline that:

* Accepts one or multiple SEM images
* Validates uploaded inputs before inference
* Classifies semiconductor defects using a trained CNN
* Identifies **defect vs. clean** samples
* Reports prediction confidence
* Measures per-image inference latency
* Processes images using **ONNX Runtime on CPU**
* Provides batch-level statistics
* Allows classification results to be exported as CSV

The deployed application is intentionally **PyTorch-free at runtime**, using the exported ONNX model for lightweight inference.

---

## 🎯 Supported Defect Classes

The trained model classifies SEM images into **8 categories**:

| Class           | Description                      |
| --------------- | -------------------------------- |
| `Bridge`        | Bridge-type semiconductor defect |
| `CMP scratch`   | CMP-related scratch defect       |
| `Clean`         | No detected defect               |
| `LER`           | Line Edge Roughness              |
| `crack`         | Crack defect                     |
| `manforsed via` | Manforsed-via defect             |
| `open`          | Open-circuit type defect         |
| `short`         | Short-circuit type defect        |

`Clean` is treated as the **no-defect** class; all other classes are considered defects.

---

## 🧠 Model Architecture

The classification model is a lightweight custom CNN trained from scratch.

### Architecture

```text
Input Image
   │
   ▼
96 × 96 × 3
   │
   ▼
Conv2D (16) + BatchNorm + ReLU
   │
   ▼
MaxPool
   │
   ▼
Conv2D (32) + BatchNorm + ReLU
   │
   ▼
MaxPool
   │
   ▼
Conv2D (64) + BatchNorm + ReLU
   │
   ▼
MaxPool
   │
   ▼
Conv2D (64) + BatchNorm + ReLU
   │
   ▼
MaxPool
   │
   ▼
Fully Connected (128)
   │
   ▼
Dropout
   │
   ▼
8-Class Output
```

Images are resized to **96 × 96** and normalized using ImageNet-style mean and standard deviation values.

---

## ⚡ Edge AI / ONNX Optimization

The trained PyTorch model was exported to **ONNX** and executed using **ONNX Runtime**.

This allows the deployed application to avoid loading PyTorch and the original training checkpoint.

### Deployment Pipeline

```text
SEM Image
    │
    ▼
Input Validation
    │
    ▼
Image Preprocessing
    │
    ▼
ONNX Runtime
    │
    ▼
CNN Inference
    │
    ▼
Softmax Probabilities
    │
    ▼
Predicted Defect Class
    │
    ├── Confidence
    ├── Defect / Clean Status
    └── Inference Latency
```

---

## 🔍 Dataset Leakage Prevention

A major part of the project was addressing **near-duplicate / burst leakage**.

SEM images can be captured as bursts within seconds of one another. A naive random train/validation split can therefore place highly similar images from the same burst into both sets, producing artificially optimistic validation results.

To prevent this, the project implements a **burst-aware split**:

```text
SEM Dataset
     │
     ▼
Group temporally related images
     │
     ▼
Keep entire bursts together
     │
     ├──────────────┐
     ▼              ▼
 Train Split     Validation Split
```

This provides a more realistic evaluation of generalization.

---

## 📊 Verified Results

### Model Evaluation

The burst-aware validation set contains **101 images**.

* **Accuracy:** 98.02%
* **Macro Precision:** 0.9797
* **Macro Recall:** 0.9797
* **Macro F1:** 0.9790
* **Validation errors:** 2

### PyTorch vs ONNX

The exported ONNX model was verified against the original PyTorch model on the same validation images.

| Metric               |            Result |
| -------------------- | ----------------: |
| Prediction agreement |          **100%** |
| PyTorch latency      |     2.88 ms/image |
| ONNX Runtime latency | **0.63 ms/image** |
| Speed improvement    |        **~4.53×** |

The ONNX model therefore preserves prediction behavior while providing substantially faster CPU inference.

---

## 🌐 Streamlit Application

The deployed web application provides an interactive interface for SEM defect analysis.

### Features

* 📤 Single or multiple image upload
* 🔍 SEM/non-SEM input validation
* 🤖 ONNX Runtime inference
* 📈 Prediction confidence
* ⚠️ Defect / clean classification
* ⚡ Per-image latency measurement
* 📊 Batch summary
* 📋 Classification results table
* 📉 Class distribution
* 🖼️ Individual image inspection
* 📥 CSV result export
* ❌ Graceful handling of invalid/corrupt inputs

### Live Application

**https://infersem.streamlit.app/**

---

## 🛠️ Tech Stack

**Programming**

* Python

**Machine Learning / Computer Vision**

* PyTorch
* CNN
* Computer Vision
* Image Classification

**Model Optimization & Deployment**

* ONNX
* ONNX Runtime

**Web Application**

* Streamlit

**Image Processing**

* Pillow
* NumPy

**Testing & Evaluation**

* PyTest
* Model benchmarking
* PyTorch vs ONNX prediction verification

---

## 📁 Project Structure

```text
THE-DOPED-MINDS/
│
├── app.py
├── main.py
├── model.py
├── detector.py
├── data_loader.py
│
├── train.py
├── train_sem.py
├── train_sem_burst.py
├── evaluate.py
│
├── make_burst_split.py
├── make_val_split.py
├── download_dataset.py
│
├── export_onnx.py
├── export_onnx_sem.py
├── compare_pytorch_onnx.py
├── onnx_only_inference.py
├── inference_speed_test.py
├── analysis.py
│
├── model_sem_burst.onnx
├── model_sem_burst.onnx.data
├── model_sem_burst.onnx.class_mapping.json
│
├── best_sem_model_burst.pth
├── best_sem_model_burst.pth.class_mapping.json
│
├── tests/
│   └── test_detector.py
│
├── requirements.txt
└── requirements-app.txt
```

---

## ⚙️ Installation

### Full Development Environment

```bash
git clone <YOUR_REPOSITORY_URL>
cd THE-DOPED-MINDS

pip install -r requirements.txt
```

### Deployment / Inference Environment

The Streamlit application only requires the lightweight inference dependencies:

```bash
pip install -r requirements-app.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Upload one or more SEM images and click:

```text
🚀 Run Classification
```

---

## 🧪 Reproduce the Training Pipeline

### 1. Create Burst-Aware Dataset Split

```bash
python make_burst_split.py \
    --data_dir /path/to/DATASET \
    --output /path/to/burst_split \
    --val_split 0.2 \
    --gap_seconds 60 \
    --seed 42
```

### 2. Train the CNN

```bash
python train_sem_burst.py \
    --train_dir /path/to/burst_split/train \
    --val_dir /path/to/burst_split/val \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --seed 42 \
    --output best_sem_model_burst.pth
```

### 3. Evaluate

```bash
python evaluate.py \
    --data_dir /path/to/burst_split/val \
    --model_path best_sem_model_burst.pth \
    --dataset sem
```

### 4. Export to ONNX

```bash
python export_onnx_sem.py \
    --model_path best_sem_model_burst.pth \
    --class_mapping best_sem_model_burst.pth.class_mapping.json \
    --onnx_path model_sem_burst.onnx
```

### 5. Verify PyTorch vs ONNX

```bash
python compare_pytorch_onnx.py \
    --data_dir /path/to/burst_split/val \
    --pytorch_model best_sem_model_burst.pth \
    --onnx_model model_sem_burst.onnx \
    --class_mapping best_sem_model_burst.pth.class_mapping.json
```

---

## 🧪 Testing

Run the test suite with:

```bash
pytest tests/
```

The tests cover core defect detection behavior and input handling.

---

## 📈 Why ONNX?

The deployment model uses ONNX Runtime instead of PyTorch because it provides:

* Lightweight inference
* CPU-friendly execution
* Lower deployment overhead
* Faster inference in the measured benchmark
* Separation between training and production inference
* No need to ship the PyTorch training stack to the deployed application

The application therefore follows a simple **train → export → deploy → infer** workflow.

---

## ⚠️ Limitations

* The dataset contains only **324 images**, so the model requires further validation on larger datasets.
* Some classes have limited validation examples.
* The burst-aware validation split contains approximately **31%** of the dataset because complete bursts must remain together.
* Only one training seed has been evaluated.
* The current metrics are based on the burst-aware validation split rather than a completely independent external test set.
* The SEM input validator is a basic safety filter, **not a dedicated SEM-vs-non-SEM machine-learning classifier**.

---

## 🔮 Future Improvements

* Expand the dataset with more SEM samples
* Add an independent external test set
* Evaluate multiple random seeds
* Improve minority-class performance
* Add model explainability such as Grad-CAM
* Optimize the model further for edge hardware
* Add quantization for smaller/faster inference
* Introduce automated model monitoring
* Deploy on dedicated edge hardware for real-time semiconductor inspection

---

## 👨‍💻 Project

**THE-DOPED-MINDS**

An Edge AI approach to automated semiconductor SEM defect classification, combining **CNN-based computer vision, leakage-aware evaluation, ONNX optimization, and real-time web inference**.

🌐 **Live Demo:** https://infersem.streamlit.app/
