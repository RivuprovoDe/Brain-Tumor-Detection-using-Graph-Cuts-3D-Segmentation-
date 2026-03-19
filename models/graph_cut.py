# =============================================================================
#  models/graph_cut.py
#  Stage 2 + 3 of the pipeline:
#    • SLIC 3D supervoxel generation
#    • Region Adjacency Graph construction
#    • Max-flow / min-cut energy minimisation (PyMaxflow)
# =============================================================================

import numpy as np
from skimage.segmentation import slic
from skimage.future import graph as sk_graph
from scipy.ndimage import label as ndimage_label

try:
    import maxflow
    MAXFLOW_AVAILABLE = True
except ImportError:
    MAXFLOW_AVAILABLE = False
    print("[GraphCut] WARNING: 'maxflow' not installed. "
          "Install with: pip install PyMaxflow\n"
          "Falling back to threshold-only segmentation.")


# ── 1. Supervoxel generation ──────────────────────────────────────────────────

def generate_supervoxels(volume: np.ndarray,
                         n_supervoxels: int,
                         compactness: float) -> np.ndarray:
    """
    Run 3D SLIC on a single-channel intensity volume.

    Parameters
    ----------
    volume        : (D, H, W) float32 — use e.g. the FLAIR channel
    n_supervoxels : target number of supervoxels
    compactness   : balance between colour and space (lower → tighter boundaries)

    Returns
    -------
    labels : (D, H, W) int32 — supervoxel label for each voxel (0-indexed)
    """
    # skimage SLIC expects (H, W, C) or 2-D; for 3-D use channel_axis=None
    # Normalise to [0,1] for SLIC
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.int32)
    norm = (volume - vmin) / (vmax - vmin)

    labels = slic(norm,
                  n_segments=n_supervoxels,
                  compactness=compactness,
                  multichannel=False,
                  enforce_connectivity=True,
                  start_label=0)
    return labels.astype(np.int32)


# ── 2. Aggregate CNN probabilities per supervoxel ─────────────────────────────

def aggregate_probabilities(prob_map: np.ndarray,
                             sv_labels: np.ndarray) -> dict[int, float]:
    """
    Compute the mean CNN probability for each supervoxel.

    Parameters
    ----------
    prob_map  : (D, H, W) float in [0,1]
    sv_labels : (D, H, W) int supervoxel label map

    Returns
    -------
    sv_probs : dict {sv_id -> mean_probability}
    """
    sv_ids  = np.unique(sv_labels)
    sv_probs = {}
    for sv_id in sv_ids:
        mask = (sv_labels == sv_id)
        sv_probs[sv_id] = float(prob_map[mask].mean())
    return sv_probs


# ── 3. Build Region Adjacency Graph ───────────────────────────────────────────

def build_rag(volume: np.ndarray,
              sv_labels: np.ndarray,
              sv_probs: dict[int, float]):
    """
    Build a Region Adjacency Graph (RAG) from supervoxels.
    Edge weight = mean intensity difference between adjacent supervoxels.

    Returns
    -------
    sv_ids    : sorted list of unique supervoxel IDs
    edges     : list of (id_i, id_j, weight) tuples
    sv_means  : dict {sv_id -> mean_intensity}
    """
    sv_ids = sorted(sv_probs.keys())
    id_to_idx = {sv: i for i, sv in enumerate(sv_ids)}

    # Compute mean intensity per supervoxel
    sv_means = {}
    for sv_id in sv_ids:
        mask = (sv_labels == sv_id)
        sv_means[sv_id] = float(volume[mask].mean())

    # Find adjacencies by checking 6-connectivity neighbours
    D, H, W = sv_labels.shape
    adjacencies = set()
    for axis in range(3):
        slices_a = [slice(None)] * 3
        slices_b = [slice(None)] * 3
        slices_a[axis] = slice(0, -1)
        slices_b[axis] = slice(1, None)
        a = sv_labels[tuple(slices_a)]
        b = sv_labels[tuple(slices_b)]
        pairs = np.stack([a.ravel(), b.ravel()], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        for p in pairs:
            key = (min(p[0], p[1]), max(p[0], p[1]))
            adjacencies.add(key)

    edges = []
    for (i, j) in adjacencies:
        diff = abs(sv_means[i] - sv_means[j])
        edges.append((i, j, diff))

    return sv_ids, edges, sv_means


# ── 4. Graph Cut energy minimisation ─────────────────────────────────────────

def run_graph_cut(sv_ids: list[int],
                  edges: list[tuple],
                  sv_probs: dict[int, float],
                  sv_means: dict[int, float],
                  lambda_: float,
                  sigma: float) -> dict[int, int]:
    """
    Solve the min-cut problem on the supervoxel RAG.

    Energy:  E(L) = Σ_i D_i(l_i)  +  λ Σ_{i,j} V_{ij}(l_i, l_j)

    Unary  D_i(1) = -ln(p_i),  D_i(0) = -ln(1 - p_i)
    Pairwise V_{ij} = exp(-||I_i - I_j||² / 2σ²)  if l_i ≠ l_j, else 0

    Returns
    -------
    labels : dict {sv_id -> 0 (background) or 1 (tumour)}
    """
    if not MAXFLOW_AVAILABLE:
        # Fallback: simple threshold at p = 0.5
        return {sv: int(p >= 0.5) for sv, p in sv_probs.items()}

    n = len(sv_ids)
    id_to_idx = {sv: i for i, sv in enumerate(sv_ids)}

    g = maxflow.Graph[float](n, len(edges))
    nodes = g.add_nodes(n)

    # ── Unary (terminal) edges ────────────────────────────────────────────────
    eps = 1e-8
    for sv in sv_ids:
        idx = id_to_idx[sv]
        p   = float(np.clip(sv_probs[sv], eps, 1 - eps))
        # source = tumour, sink = background
        cap_source = -np.log(p)        # cost of assigning to background
        cap_sink   = -np.log(1.0 - p) # cost of assigning to tumour
        g.add_tedge(nodes[idx], cap_source, cap_sink)

    # ── Pairwise (neighbour) edges ────────────────────────────────────────────
    for (i, j, _) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        diff  = sv_means[i] - sv_means[j]
        w     = lambda_ * np.exp(-(diff ** 2) / (2 * sigma ** 2))
        g.add_edge(nodes[idx_i], nodes[idx_j], w, w)

    # ── Solve ────────────────────────────────────────────────────────────────
    g.maxflow()

    labels = {}
    for sv in sv_ids:
        idx = id_to_idx[sv]
        # segment() returns 0 = source side (tumour), 1 = sink side (background)
        labels[sv] = 1 - g.get_segment(nodes[idx])

    return labels


# ── 5. Post-processing ────────────────────────────────────────────────────────

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    if mask.sum() == 0:
        return mask
    labeled, num_features = ndimage_label(mask)
    if num_features == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


# ── 6. Full pipeline (CNN prob → supervoxels → graph cut → final mask) ────────

def refine_with_graph_cut(prob_map: np.ndarray,
                           flair_vol: np.ndarray,
                           n_supervoxels: int,
                           compactness: float,
                           lambda_: float,
                           sigma: float,
                           post_process: bool = True) -> np.ndarray:
    """
    Full Stage 2 + 3 pipeline.

    Parameters
    ----------
    prob_map      : (D, H, W) CNN probability map in [0, 1]
    flair_vol     : (D, H, W) FLAIR intensity (used for supervoxels + pairwise term)
    n_supervoxels : target number of SLIC supervoxels
    compactness   : SLIC compactness
    lambda_       : smoothness weight
    sigma         : pairwise Gaussian width
    post_process  : whether to keep only the largest connected component

    Returns
    -------
    final_mask : (D, H, W) uint8 binary segmentation mask
    """
    print("  [GraphCut] Generating supervoxels …")
    sv_labels = generate_supervoxels(flair_vol, n_supervoxels, compactness)

    print(f"  [GraphCut] {len(np.unique(sv_labels))} supervoxels generated")

    print("  [GraphCut] Aggregating CNN probabilities …")
    sv_probs = aggregate_probabilities(prob_map, sv_labels)

    print("  [GraphCut] Building region adjacency graph …")
    sv_ids, edges, sv_means = build_rag(flair_vol, sv_labels, sv_probs)

    print(f"  [GraphCut] Running min-cut on graph with "
          f"{len(sv_ids)} nodes, {len(edges)} edges …")
    sv_labels_cut = run_graph_cut(sv_ids, edges, sv_probs, sv_means,
                                   lambda_=lambda_, sigma=sigma)

    # Map back to voxel space
    final_mask = np.zeros_like(sv_labels, dtype=np.uint8)
    for sv, lab in sv_labels_cut.items():
        final_mask[sv_labels == sv] = lab

    if post_process:
        print("  [GraphCut] Post-processing: keeping largest component …")
        final_mask = keep_largest_component(final_mask)

    return final_mask
