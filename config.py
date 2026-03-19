# =============================================================================
#  config.py  —  All paths and hyperparameters in one place.
#  Edit this file before running anything else.
# =============================================================================

import os

# ── PATHS ────────────────────────────────────────────────────────────────────

# Root folder of the downloaded BraTS dataset.
# Expected structure inside:
#   BRATS_ROOT/
#     BraTS2021_00001/
#       BraTS2021_00001_flair.nii.gz
#       BraTS2021_00001_t1.nii.gz
#       BraTS2021_00001_t1ce.nii.gz
#       BraTS2021_00001_t2.nii.gz
#       BraTS2021_00001_seg.nii.gz   ← ground-truth label
#     BraTS2021_00002/
#       ...
BRATS_ROOT = r"C:\Users\YourName\data\BraTS2021"   # ← CHANGE THIS

# Where trained model checkpoints are saved
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# Where output masks / visualisations are written
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
PATCH_SIZE   = (128, 128, 128)   # (D, H, W) — training patch
PATCH_STRIDE = (64,  64,  64)    # stride for sliding-window inference

# ── DATA SPLIT ────────────────────────────────────────────────────────────────
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.10
RANDOM_SEED   = 42

# ── 3D U-NET ──────────────────────────────────────────────────────────────────
IN_CHANNELS  = 4          # flair, t1, t1ce, t2
OUT_CHANNELS = 1          # binary: tumour vs. background
BASE_FILTERS = 32         # doubles each encoder level → 32,64,128,256
ENCODER_DEPTHS = 4        # number of down-sampling levels

# ── TRAINING ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 1        # increase if you have more VRAM
NUM_EPOCHS     = 150
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
LR_PATIENCE    = 10       # ReduceLROnPlateau patience (epochs)
EARLY_STOP     = 20       # stop if val Dice doesn't improve for N epochs

# ── SUPERVOXEL (SLIC) ─────────────────────────────────────────────────────────
N_SUPERVOXELS  = 2000     # target number of supervoxels per volume
SLIC_COMPACTNESS = 0.1    # lower → tighter boundary adherence

# ── GRAPH CUT ─────────────────────────────────────────────────────────────────
GC_LAMBDA      = 5.0      # weight of the smoothness term  λ
GC_SIGMA       = 0.3      # σ in the pairwise Gaussian kernel

# ── MISC ──────────────────────────────────────────────────────────────────────
NUM_WORKERS    = 4        # DataLoader workers (set 0 on Windows if errors occur)
PIN_MEMORY     = True     # set False if RAM is tight
DEVICE         = "cuda"   # "cuda" or "cpu"
