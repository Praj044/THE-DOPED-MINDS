# 🔬 THE-DOPED-MINDS

### Edge AI Semiconductor Defect Classification from SEM Images

**THE-DOPED-MINDS** is an Edge AI-based computer vision system designed to automatically classify semiconductor defects from **Scanning Electron Microscope (SEM) images**.

The project combines a lightweight CNN, leakage-aware dataset splitting, ONNX model optimization, and a Streamlit web application to provide fast CPU-based inference.

🌐 **Live Demo:** https://infersem.streamlit.app/

---

## 🚀 Overview

Manual inspection of semiconductor SEM images can be time-consuming and may introduce inconsistencies.

THE-DOPED-MINDS automates this process by allowing users to upload SEM images and obtain:

* Semiconductor defect classification
* Defect vs. clean identification
* Prediction confidence
* Per-image inference latency
* Batch-level classification statistics
* Class distribution analysis
* CSV export of prediction results

The production application uses **ONNX Runtime** instead of PyTorch for lightweight and fast CPU inference.

---

## ✨ Key Features

* 🧠 **CNN-based SEM defect classification**
* 🔬 **8-class semiconductor defect detection**
* ⚡ **ONNX Runtime CPU inference**
* 📊 **Confidence score for predictions**
* ⏱️ **Per-image inference latency measurement**
* 📦 **Batch image processing**
* 📈 **Class distribution visualization**
* 📥 **CSV result export**
* 🛡️ **Input validation and error handling**
* 🔒 **Burst-aware train/validation splitting**
* 🌐 **Streamlit web deployment**

---

## 🎯 Problem Statement

Semiconductor manufacturing requires accurate inspection of microscopic structures for identifying defects such as opens, shorts, cracks, scratches, and line-edge irregularities.

Traditional manual inspection can be:

* Time-consuming
* Difficult to scale
* Dependent on human expertise
* Inconsistent for large image volumes

This project explores an **Edge AI approach** for automated SEM image classification that can eventually be deployed on resource-constrained systems and semiconductor inspection pipelines.

---

# 📂 Dataset

The dataset contains SEM images representing **8 semiconductor classes**.

### Dataset Structure

```text
DATASET/
├── Bridge/
├── CMP scratch/
├── Clean/
├── LER/
├── crack/
├── manforsed via/
├── open/
└── short/
```

### Dataset Information

| Property     | Value                      |
| ------------ | -------------------------- |
| Total images | 333                        |
| Classes      | 8                          |
| Image type   | SEM images                 |
| Task         | Multi-class classification |
| Input        | SEM image                  |
| Output       | Defect class               |

### Classes

| Class           | Description                      |
| --------------- | -------------------------------- |
| `Bridge`        | Bridge-type semiconductor defect |
| `CMP scratch`   | CMP-related scratch defect       |
| `Clean`         | Defect-free sample               |
| `LER`           | Line Edge Roughness              |
| `crack`         | Crack defect                     |
| `manforsed via` | Manforsed-via defect             |
| `open`          | Open-circuit type defect         |
| `short`         | Short-circuit type defect        |

📥 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1uJ59JvSptqp9oR6rq_7sxr1ShFq_4B-i?usp=sharing)**

> The dataset is hosted separately from the source code and is not included directly in the GitHub repository.

---

# 🧠 Model Architecture

The project uses a lightweight custom **Convolutional Neural Network (CNN)** trained for SEM image classification.

### Architecture

```text
Input SEM Image
      │
      ▼
Resize → 96 × 96 × 3
      │
      ▼
Conv2D (16)
BatchNorm
ReLU
      │
      ▼
MaxPool
      │
      ▼
Conv2D (32)
BatchNorm
ReLU
      │
      ▼
MaxPool
      │
      ▼
Conv2D (64)
BatchNorm
ReLU
      │
      ▼
MaxPool
      │
      ▼
Conv2D (64)
BatchNorm
ReLU
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

### Image Preprocessing

Input images are:

1. Loaded using Pillow
2. Converted to RGB
3. Resized to **96 × 96**
4. Converted to tensors
5. Normalized using ImageNet-style normalization

---

# 🔬 Burst-Aware Dataset Splitting

A major focus of the project is preventing **data leakage caused by near-duplicate SEM images**.

SEM acquisition systems can capture multiple images within a short time period. Randomly splitting these images can result in images from the same acquisition burst appearing in both training and validation sets.

This can lead to artificially high validation performance.

### Approach

```text
SEM Dataset
     │
     ▼
Identify temporally related images
     │
     ▼
Group images into bursts
     │
     ▼
Keep complete bursts together
     │
     ├───────────────┐
     ▼               ▼
 Training Set    Validation Set
```

This provides a more realistic estimate of model generalization.

---

# 📊 Model Performance

The burst-aware validation set contains **101 images**.

### Validation Results

| Metric            |      Score |
| ----------------- | ---------: |
| Accuracy          | **98.02%** |
| Macro Precision   | **0.9797** |
| Macro Recall      | **0.9797** |
| Macro F1          | **0.9790** |
| Validation Errors |      **2** |

> These results are based on the burst-aware validation split and should not be interpreted as performance on a completely independent external dataset.

---

# ⚡ ONNX Optimization

After training, the PyTorch model is exported to **ONNX** for production inference.

### Deployment Pipeline

```text
PyTorch CNN
     │
     ▼
ONNX Export
     │
     ▼
ONNX Model
     │
     ▼
ONNX Runtime
     │
     ▼
CPU Inference
```

This removes the need to load the full PyTorch training stack during deployment.

### PyTorch vs ONNX

The ONNX model was verified against the original PyTorch model using the same validation images.

| Metric               |            Result |
| -------------------- | ----------------: |
| Prediction Agreement |          **100%** |
| PyTorch Latency      |     2.88 ms/image |
| ONNX Runtime Latency | **0.63 ms/image** |
| Approx. Speedup      |         **4.53×** |

The ONNX model maintained prediction consistency while providing significantly faster CPU inference in the benchmark.

---

# 🌐 Streamlit Application

The project is deployed as an interactive Streamlit application.

### 🔗 Live Demo

**https://infersem.streamlit.app/**

### Application Workflow

```text
Upload SEM Image(s)
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
CNN Prediction
        │
        ▼
Prediction + Confidence
        │
        ├── Defect / Clean
        ├── Inference Latency
        └── Batch Statistics
```

### Application Capabilities

Users can:

* Upload one or multiple images
* Run classification
* View predicted classes
* View confidence scores
* View inference latency
* Identify defect/clean status
* Analyze class distribution
* Inspect individual predictions
* Download results as CSV

---

# 🛠️ Tech Stack

### Programming

* Python

### Machine Learning

* PyTorch
* Convolutional Neural Networks
* Computer Vision
* Image Classification

### Model Deployment

* ONNX
* ONNX Runtime

### Web Application

* Streamlit

### Image Processing

* Pillow
* NumPy

### Testing & Evaluation

* PyTest
* Model benchmarking
* PyTorch vs ONNX verification

### Development Tools

* Git
* GitHub
* VS Code

---

# 📁 Project Structure

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

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd THE-DOPED-MINDS
```

Install the development dependencies:

```bash
pip install -r requirements.txt
```

For lightweight application deployment:

```bash
pip install -r requirements-app.txt
```

---

# ▶️ Run the Application Locally

Start Streamlit:

```bash
streamlit run app.py
```

The application will open at the local Streamlit URL displayed in the terminal.

Upload one or more SEM images and run the classifier.

---

# 🧪 Training Pipeline

### Step 1 — Download Dataset

Download the dataset from:

**[Google Drive Dataset](https://drive.google.com/drive/folders/1uJ59JvSptqp9oR6rq_7sxr1ShFq_4B-i?usp=sharing)**

Extract it into the required dataset directory.

---

### Step 2 — Create Burst-Aware Split

```bash
python make_burst_split.py \
    --data_dir /path/to/DATASET \
    --output /path/to/burst_split \
    --val_split 0.2 \
    --gap_seconds 60 \
    --seed 42
```

---

### Step 3 — Train the Model

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

---

### Step 4 — Evaluate

```bash
python evaluate.py \
    --data_dir /path/to/burst_split/val \
    --model_path best_sem_model_burst.pth \
    --dataset sem
```

---

### Step 5 — Export to ONNX

```bash
python export_onnx_sem.py \
    --model_path best_sem_model_burst.pth \
    --class_mapping best_sem_model_burst.pth.class_mapping.json \
    --onnx_path model_sem_burst.onnx
```

---

### Step 6 — Verify ONNX Predictions

```bash
python compare_pytorch_onnx.py \
    --data_dir /path/to/burst_split/val \
    --pytorch_model best_sem_model_burst.pth \
    --onnx_model model_sem_burst.onnx \
    --class_mapping best_sem_model_burst.pth.class_mapping.json
```

---

# 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

The tests cover core inference and input-handling behavior.

---

# 📈 Why ONNX?

ONNX Runtime was selected for production inference because it provides:

* Lightweight CPU inference
* Reduced deployment dependencies
* Faster measured inference
* Lower runtime overhead
* Easy separation between training and deployment
* Compatibility with future edge-device deployment

The resulting workflow is:

```text
Train → Evaluate → Export → Optimize → Deploy → Infer
```

---

# ⚠️ Limitations

* The current dataset contains **333 images**, which is relatively small for a production-grade computer vision system.
* Some defect classes have limited samples.
* The burst-aware validation strategy reduces the number of images available for validation because complete acquisition bursts must remain together.
* The reported metrics are based on a validation split rather than an independent external benchmark.
* Only a limited number of training configurations/seeds have been evaluated.
* Real-world semiconductor inspection environments may contain variations in imaging conditions, equipment, resolution, and sample preparation.

---

# 🔮 Future Improvements

* 📚 Expand the SEM dataset
* 🧪 Add an independent external test set
* 🔁 Evaluate multiple training seeds
* ⚖️ Improve minority-class performance
* 🔍 Add Grad-CAM/model explainability
* 🧮 Introduce INT8 quantization
* ⚡ Optimize the model for edge hardware
* 📦 Package the inference engine for embedded deployment
* 📊 Add model monitoring and drift detection
* 🏭 Integrate with automated semiconductor inspection workflows

---

# 📌 Key Takeaways

THE-DOPED-MINDS demonstrates an end-to-end **Edge AI computer vision pipeline**:

```text
SEM Dataset
     ↓
Leakage-Aware Data Splitting
     ↓
CNN Training
     ↓
Model Evaluation
     ↓
ONNX Export
     ↓
CPU Optimization
     ↓
Streamlit Deployment
     ↓
Real-Time SEM Defect Classification
```

The project achieves **98.02% validation accuracy**, maintains **100% PyTorch–ONNX prediction agreement**, and achieves approximately **4.53× faster measured inference** using ONNX Runtime.

---

# 🌐 Links

| Resource       | Link                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| 🚀 Live Demo   | https://infersem.streamlit.app/                                                                      |
| 📂 Dataset     | [Google Drive](https://drive.google.com/drive/folders/1uJ59JvSptqp9oR6rq_7sxr1ShFq_4B-i?usp=sharing) |
| 💻 Source Code | `<YOUR_GITHUB_REPOSITORY_URL>`                                                                       |

---

# 👨‍💻 Author

**Prajjwal Gupta**

B.Tech — ECE (VLSI Design & Technology)
Maharaja Agrasen Institute of Technology (MAIT)
GGSIPU

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and sharing feedback or suggestions for improving the semiconductor defect classification pipeline.
