# 3D Brain Tumor Segmentation
**Jadavpur University — Electronics & Tele-Communication Engineering**
*Niladri Sekhar Mondal & Rivuprovo De | Supervised by Prof. Ananda Shankar Chowdhury*

---

## What this project does

Implements the full hybrid pipeline from the final-year project report:

```
NIfTI MRI (4 modalities)
        │
        ▼
[1] Preprocessing       N4 bias correction, z-score normalisation, brain crop
        │
        ▼
[2] 3D U-Net            Sliding-window inference → voxel-wise probability map
        │
        ▼
[3] SLIC Supervoxels    Groups voxels into ~2000 perceptually meaningful regions
        │
        ▼
[4] Graph Cut           Min-cut on the Region Adjacency Graph → refined mask
        │
        ▼
Outputs: segmentation_mask.nii.gz | visualisation.png | Dice / HD95 metrics
```

---

## Project structure

```
brain_tumor_seg/
├── config.py               ← SET YOUR PATHS HERE FIRST
├── train.py                ← Step 1: train the 3D U-Net
├── predict.py              ← Step 2: run inference on new scans
├── plot_history.py         ← Optional: plot training curves
├── requirements.txt
├── models/
│   ├── unet3d.py           ← Attention 3D U-Net + loss functions + metrics
│   └── graph_cut.py        ← SLIC supervoxels + RAG + max-flow graph cut
└── utils/
    ├── preprocessing.py    ← NIfTI loading, bias correction, normalisation, patching
    ├── dataset.py          ← PyTorch Dataset (BraTS auto-discovery, augmentation)
    └── visualise.py        ← Matplotlib overlays, probability maps, history plots
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy matplotlib scikit-image scipy tqdm PyMaxflow
```

> **Windows note:** if `NUM_WORKERS > 0` causes errors, set `NUM_WORKERS = 0` in `config.py`.

### 2. Download the BraTS dataset

- Register and download from: https://www.synapse.org/#!Synapse:syn51156910/wiki/621282
- BraTS 2021 is recommended (the default naming convention is supported).
- Extract so the folder looks like:

```
BraTS2021/
    BraTS2021_00001/
        BraTS2021_00001_flair.nii.gz
        BraTS2021_00001_t1.nii.gz
        BraTS2021_00001_t1ce.nii.gz
        BraTS2021_00001_t2.nii.gz
        BraTS2021_00001_seg.nii.gz
    BraTS2021_00002/
        ...
```

### 3. Configure paths

Open `config.py` and set:

```python
BRATS_ROOT = r"C:\Users\YourName\data\BraTS2021"   # ← your actual path
DEVICE     = "cuda"   # or "cpu" if no GPU
```

---

## Training

```bash
python train.py
```

- Automatically splits the dataset into train / val / test.
- Saves the best checkpoint to `checkpoints/best_model.pth`.
- Saves the last checkpoint to `checkpoints/last_model.pth` (training resumes automatically if interrupted).
- Prints Dice score after every epoch.

To plot training curves after training:

```bash
python plot_history.py
```

---

## Inference (predict on new scans)

### Single subject

```bash
python predict.py \
    --flair  data/BraTS21_001/BraTS21_001_flair.nii.gz \
    --t1     data/BraTS21_001/BraTS21_001_t1.nii.gz    \
    --t1ce   data/BraTS21_001/BraTS21_001_t1ce.nii.gz  \
    --t2     data/BraTS21_001/BraTS21_001_t2.nii.gz    \
    --seg    data/BraTS21_001/BraTS21_001_seg.nii.gz   # optional ground-truth
```

### All test subjects (from the training split)

```bash
python predict.py --all_test
```

### Outputs (per subject, saved in `outputs/<subject_name>/`)

| File | Description |
|---|---|
| `segmentation_mask.nii.gz` | Binary tumour mask |
| `cnn_probability_map.nii.gz` | Raw CNN sigmoid probabilities |
| `visualisation.png` | Multi-panel figure with tumour highlighted |

The visualisation shows:
- FLAIR with **red tumour overlay**
- T1ce with **predicted contour**
- CNN **probability heatmap**
- Predicted mask vs ground-truth mask (when available)

### Terminal output example

```
──────────────────────────────────────────────────────────
  Subject: BraTS2021_00001
──────────────────────────────────────────────────────────
[1/4] Preprocessing …
      Volume shape: (4, 138, 172, 138)  (3.2s)
[2/4] Running 3D U-Net (sliding window) …
      Prob map range: [0.001, 0.997]  (8.4s)
[3/4] Supervoxel-based Graph Cut …
  [GraphCut] Generating supervoxels …
  [GraphCut] 1987 supervoxels generated
  [GraphCut] Aggregating CNN probabilities …
  [GraphCut] Building region adjacency graph …
  [GraphCut] Running min-cut on graph with 1987 nodes, 6241 edges …
  [GraphCut] Post-processing: keeping largest component …
      Tumour voxels: 12,405  (4.1s)
[4/4] Computing metrics and saving outputs …

  ┌─────────────────────────────────┐
  │         RESULTS SUMMARY          │
  ├─────────────────────────────────┤
  │  Dice (CNN only)  : 0.8412        │
  │  Dice (GraphCut)  : 0.8731        │
  │  HD95 (voxels)    : 6.24         │
  └─────────────────────────────────┘
```

---

## Quick data viewer (in PyCharm console)

```python
from utils.visualise import quick_view
quick_view("path/to/flair.nii.gz", "path/to/segmentation_mask.nii.gz")
```

---

## Key hyperparameters (all in `config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PATCH_SIZE` | (128,128,128) | Training patch size |
| `N_SUPERVOXELS` | 2000 | SLIC target supervoxels |
| `GC_LAMBDA` | 5.0 | Smoothness weight λ |
| `GC_SIGMA` | 0.3 | Pairwise Gaussian σ |
| `BASE_FILTERS` | 32 | U-Net base feature maps |
| `NUM_EPOCHS` | 150 | Max training epochs |
| `LR` | 1e-4 | Initial learning rate |
