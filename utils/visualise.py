# =============================================================================
#  utils/visualise.py
#  Matplotlib-based visualisation helpers.
#  Generates the highlighted tumour overlays, probability maps, and metrics.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Custom colourmap: transparent → vivid red (tumour highlight)
TUMOR_CMAP = LinearSegmentedColormap.from_list(
    "tumor", [(0, 0, 0, 0), (1, 0.1, 0.1, 0.85)], N=256)

PROB_CMAP  = plt.cm.inferno


# ── helpers ───────────────────────────────────────────────────────────────────

def _mid_slice(vol: np.ndarray, axis: int) -> np.ndarray:
    """Return the middle slice along the given axis."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def _normalise_for_display(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ── main visualisation function ───────────────────────────────────────────────

def visualise_results(volume4d:  np.ndarray,
                      prob_map:  np.ndarray,
                      pred_mask: np.ndarray,
                      gt_mask:   np.ndarray | None = None,
                      save_path: str = "visualisation.png",
                      title:     str = "Segmentation Result",
                      show:      bool = False):
    """
    Generate a comprehensive multi-panel visualisation and save it to disk.

    Layout (axial / coronal / sagittal views for each panel):
      Row 0 : FLAIR with tumour overlay (highlighted)
      Row 1 : T1ce with predicted mask border
      Row 2 : CNN probability heat-map
      Row 3 : Predicted mask  |  Ground-truth mask  (if available)

    Parameters
    ----------
    volume4d  : (4, D, H, W)  preprocessed 4-channel volume
    prob_map  : (D, H, W)     CNN sigmoid probabilities
    pred_mask : (D, H, W)     final binary mask (after graph cut)
    gt_mask   : (D, H, W)     ground-truth label  (optional)
    save_path : where to write the PNG
    title     : suptitle string
    show      : if True, call plt.show() (useful in interactive environments)
    """
    flair = _normalise_for_display(volume4d[0])   # channel 0 = FLAIR
    t1ce  = _normalise_for_display(volume4d[2])   # channel 2 = T1ce

    axes_labels = ['Axial', 'Coronal', 'Sagittal']
    # Each view returns a 2-D slice
    def slices(vol3d):
        return [
            np.rot90(_mid_slice(vol3d, axis=0)),   # axial
            np.rot90(_mid_slice(vol3d, axis=1)),   # coronal
            np.rot90(_mid_slice(vol3d, axis=2)),   # sagittal
        ]

    flair_s = slices(flair)
    t1ce_s  = slices(t1ce)
    prob_s  = slices(prob_map)
    pred_s  = slices(pred_mask)
    gt_s    = slices(gt_mask) if gt_mask is not None else None

    n_rows = 4 if gt_mask is None else 5
    n_cols = 3
    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4), facecolor='#0a0a0a')
    fig.suptitle(title, fontsize=16, color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.05, wspace=0.05)

    def ax(row, col):
        a = fig.add_subplot(gs[row, col])
        a.axis('off')
        return a

    # ── Row 0: FLAIR + tumour overlay ─────────────────────────────────────────
    for c in range(3):
        a = ax(0, c)
        a.imshow(flair_s[c], cmap='gray', vmin=0, vmax=1)
        a.imshow(pred_s[c].astype(float), cmap=TUMOR_CMAP, alpha=0.7, vmin=0, vmax=1)
        if c == 0:
            a.set_title("FLAIR + Tumour Overlay", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 1: T1ce + predicted contour ───────────────────────────────────────
    for c in range(3):
        a = ax(1, c)
        a.imshow(t1ce_s[c], cmap='gray', vmin=0, vmax=1)
        # Draw contour of predicted mask
        if pred_s[c].any():
            a.contour(pred_s[c], levels=[0.5], colors=['#ff4c6e'], linewidths=1.5)
        if c == 0:
            a.set_title("T1ce + Predicted Contour", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 2: CNN probability map ────────────────────────────────────────────
    for c in range(3):
        a = ax(2, c)
        im = a.imshow(prob_s[c], cmap=PROB_CMAP, vmin=0, vmax=1)
        if c == 0:
            a.set_title("CNN Probability Map", color='white',
                        fontsize=10, pad=4)
        a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
               color='#aaaaaa', fontsize=8, va='top')

    # ── Row 3: Prediction vs Ground Truth (side-by-side) ─────────────────────
    # If gt_mask is available show both; otherwise just prediction
    row3_data = [
        (pred_s,  "Predicted Mask (Graph Cut)", '#00e5ff'),
    ]
    if gt_s is not None:
        row3_data.append((gt_s, "Ground-Truth Mask", '#a259ff'))

    for col_offset, (data, lbl, colour) in enumerate(row3_data):
        for c in range(3):
            a = ax(3 + col_offset, c)
            # Background: FLAIR
            a.imshow(flair_s[c], cmap='gray', vmin=0, vmax=1, alpha=0.6)
            cmap_local = LinearSegmentedColormap.from_list(
                "c", [(0, 0, 0, 0), matplotlib.colors.to_rgba(colour, 0.85)], N=2)
            a.imshow(data[c].astype(float), cmap=cmap_local, vmin=0, vmax=1)
            if c == 0:
                a.set_title(lbl, color='white', fontsize=10, pad=4)
            a.text(0.02, 0.96, axes_labels[c], transform=a.transAxes,
                   color='#aaaaaa', fontsize=8, va='top')

    # ── Colourbar for probability map ─────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.45, 0.01, 0.15])
    sm = plt.cm.ScalarMappable(cmap=PROB_CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(tumour)", color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor='#ff4c6e', alpha=0.85, label='Tumour overlay'),
        Patch(facecolor='#ff4c6e', fill=False, edgecolor='#ff4c6e', label='Predicted contour'),
        Patch(facecolor='#00e5ff', alpha=0.85, label='Predicted mask'),
    ]
    if gt_mask is not None:
        legend_elements.append(
            Patch(facecolor='#a259ff', alpha=0.85, label='Ground truth'))
    fig.legend(handles=legend_elements, loc='lower left',
               bbox_to_anchor=(0.01, 0.01), ncol=2,
               framealpha=0.2, labelcolor='white', fontsize=8)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Visualisation saved → {save_path}")
    if show:
        plt.show()


# ── training history plot ─────────────────────────────────────────────────────

def plot_training_history(history_path: str, save_path: str):
    """
    Load the history .npy file saved by train.py and plot loss + dice curves.
    """
    history = np.load(history_path, allow_pickle=True).item()
    train_loss = history.get('train_loss', [])
    val_dice   = history.get('val_dice',   [])
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0a0a0a')

    for a in axes:
        a.set_facecolor('#111827')
        a.tick_params(colors='white')
        for spine in a.spines.values():
            spine.set_edgecolor('#334155')

    axes[0].plot(epochs, train_loss, color='#00e5ff', linewidth=2, label='Train Loss')
    axes[0].set_title('Training Loss (Dice+BCE)', color='white')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Loss', color='white')
    axes[0].legend(facecolor='#1e293b', labelcolor='white')

    axes[1].plot(epochs, val_dice, color='#a259ff', linewidth=2, label='Val Dice')
    axes[1].set_title('Validation Dice Score', color='white')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Dice', color='white')
    axes[1].set_ylim(0, 1)
    axes[1].legend(facecolor='#1e293b', labelcolor='white')

    fig.suptitle('Training History', color='white', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training history plot saved → {save_path}")


# ── quick single-image viewer (standalone use) ───────────────────────────────

def quick_view(nifti_path: str, mask_path: str | None = None):
    """
    Quickly display a NIfTI volume (+ optional mask overlay) in matplotlib.
    Useful for sanity-checking your data interactively in PyCharm.

    Usage:
        from utils.visualise import quick_view
        quick_view("subject_flair.nii.gz", "segmentation_mask.nii.gz")
    """
    import nibabel as nib
    vol  = np.asarray(nib.load(nifti_path).dataobj, dtype=np.float32)
    mask = None
    if mask_path:
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)

    vol_n = _normalise_for_display(vol)
    mid_d, mid_h, mid_w = [s // 2 for s in vol.shape]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='black')
    views = [
        (np.rot90(vol_n[mid_d, :, :]),  "Axial"),
        (np.rot90(vol_n[:, mid_h, :]),  "Coronal"),
        (np.rot90(vol_n[:, :, mid_w]),  "Sagittal"),
    ]
    mask_views = None
    if mask is not None:
        mask_views = [
            np.rot90(mask[mid_d, :, :]),
            np.rot90(mask[:, mid_h, :]),
            np.rot90(mask[:, :, mid_w]),
        ]

    for ax, (img, lbl), in zip(axes, views):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(lbl, color='white')
        ax.set_facecolor('black')

    if mask_views:
        for ax, mv in zip(axes, mask_views):
            ax.imshow(mv, cmap=TUMOR_CMAP, alpha=0.6, vmin=0, vmax=1)

    fig.suptitle(os.path.basename(nifti_path), color='white')
    plt.tight_layout()
    plt.show()
