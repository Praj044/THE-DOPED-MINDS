# THE-DOPED-MINDS

Edge AI defect classification for semiconductor SEM images. CNN trained from
scratch, converted to ONNX, served via a Streamlit web app.

## Current status: Phases 1-4 complete, verified. Phase 5 (cloud deployment) not started.

- **Phase 1** — found and fixed near-duplicate/burst leakage in the naive random
  train/val split (SEM images are burst screenshots seconds apart). Built a
  burst-aware split (`make_burst_split.py`) that keeps each burst on one side.
- **Phase 2** — trained a new checkpoint from scratch on the burst-aware split
  (`train_sem_burst.py` → `best_sem_model_burst.pth`), fixed-seed, reproducible.
- **Phase 3** — exported that checkpoint to ONNX (`export_onnx_sem.py` →
  `model_sem_burst.onnx` + `model_sem_burst.onnx.data`), verified 100%
  prediction agreement between PyTorch and ONNX Runtime, benchmarked latency.
- **Phase 4** — built a Streamlit web app (`app.py`) that runs inference
  through ONNX Runtime only, no PyTorch import, no training checkpoint.

## The model in use for deployment

```
best_sem_model_burst.pth   <- trained checkpoint (Phase 2), NOT best_sem_model.pth
model_sem_burst.onnx       <- exported graph (Phase 3)
model_sem_burst.onnx.data  <- external weights, REQUIRED alongside the .onnx file
```

Do not swap these for the older `best_sem_model.pth` / `model_sem.onnx` —
those were trained on the leaky random split from before Phase 1 and their
val-set accuracy is not trustworthy.

## Setup

Full project (training, evaluation, tests):
```bash
pip install -r requirements.txt
```

Web app only (what actually gets deployed):
```bash
pip install -r requirements-app.txt
```

## Run the web app

```bash
streamlit run app.py
```
Upload a `.jpg`/`.jpeg`/`.png` SEM image. The app shows predicted class,
confidence, defect/no-defect status, and inference latency in ms. Uses
ONNX Runtime exclusively — verified to import neither `torch` nor `model.py`.

## Reproduce the burst-aware split, training, and ONNX export

```bash
python make_burst_split.py --data_dir /path/to/DATASET --output /path/to/burst_split --val_split 0.2 --gap_seconds 60 --seed 42

python train_sem_burst.py --train_dir /path/to/burst_split/train --val_dir /path/to/burst_split/val \
    --epochs 30 --batch_size 16 --lr 0.001 --seed 42 --output best_sem_model_burst.pth

python evaluate.py --data_dir /path/to/burst_split/val --model_path best_sem_model_burst.pth --dataset sem

python export_onnx_sem.py --model_path best_sem_model_burst.pth \
    --class_mapping best_sem_model_burst.pth.class_mapping.json --onnx_path model_sem_burst.onnx

python compare_pytorch_onnx.py --data_dir /path/to/burst_split/val \
    --pytorch_model best_sem_model_burst.pth --onnx_model model_sem_burst.onnx \
    --class_mapping best_sem_model_burst.pth.class_mapping.json
```

The raw 324-image dataset (`DATASET.rar`) is not required to run the web app
and is not included in the backup ZIP — only the trained model, ONNX export,
and the scripts needed to regenerate the split/training if you have the
original data.

## Verified results (real runs, not estimated)

**Phase 2 — `best_sem_model_burst.pth` on the burst-aware val split (101 images):**
- Accuracy: 98.02%, Macro P/R/F1: 0.9797 / 0.9797 / 0.9790
- 2 errors: Bridge→open, manforsed via→Bridge

**Phase 3 — PyTorch vs ONNX, same 101 images:**
- Prediction agreement: 100% (0 mismatches)
- PyTorch latency: 2.88 ms/image, ONNX latency: 0.63 ms/image (4.53x speedup)
- ONNX total size (graph + `.onnx.data`) ≈ same as the PyTorch checkpoint (~1.4MB) — ONNX did not shrink the model here, only sped up inference

**Phase 4 — Streamlit app vs standalone ONNX script, same image:**
- Identical prediction and confidence (Bridge, 73.2%)
- Corrupt image upload handled gracefully (no crash)
- Confirmed zero PyTorch dependency at runtime

## Tests

```bash
pip install pytest
pytest tests/
```

## Known limitations

- 324 total images is a small dataset; several classes have single-digit
  validation support (`crack`=6, `LER`=9, `open`=9, `short`=9). Treat the
  per-class metrics as indicative, not tight estimates.
- The burst-aware val split is ~31% of the data (101/324), not the intended
  20%, because whole bursts had to be kept together in small classes.
- Only one training seed has been evaluated; sensitivity to seed is unknown.
- `evaluate.py`'s macro metrics and `compare_pytorch_onnx.py`'s agreement
  numbers are specific to this one val split — not yet tested against a
  truly independent, never-touched-until-the-end test set.
