"""
utils/pipeline.py
-----------------
All mask-generation logic lives here.

Every operation that can be GPU-accelerated routes through utils/gpu.py.
When GPU is unavailable or disabled, the exact same CPU code path runs —
no changes to calling code needed.

Pipeline order (for all modes):
  1.  Percentile normalisation   (NumPy, always CPU — negligible time)
  2.  CLAHE                      (OpenCV CUDA or CPU)
  3.  Frangi vessel filter        (CuPy CUDA or scikit-image CPU)
  4.  Threshold                   (OpenCV, very fast on CPU)
  5.  Morphological opening       (OpenCV CUDA or CPU)
  6.  Distance-based dilation     (SciPy, CPU — no OpenCV CUDA equivalent)
  7.  Fill holes                  (SciPy, CPU)
  8.  Watershed separation        (scikit-image, CPU)
  9.  Area filtering              (OpenCV, CPU)

Performance profile (approximate, 2048×2048 image, Thin Structures mode):
  Step          CPU time    GPU time (RTX-class)   GPU speedup
  -----------------------------------------------------------
  Frangi        ~8-15 s     ~0.8-1.5 s             ~8-10x
  CLAHE         ~0.05 s     ~0.01 s                ~5x
  Morph open    ~0.02 s     ~0.005 s               ~4x
  Total         ~8-15 s     ~1-2 s                 ~6-8x total

For Blob/Cytoplasm/Diffuse modes (no Frangi), total GPU speedup is modest
(~1.5-2x) since the bottlenecks are SciPy distance transforms and watershed.
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
import streamlit as st

from utils.gpu import (
    apply_clahe_gpu,
    apply_frangi_gpu,
    morphological_open_gpu,
    gaussian_blur_gpu,
)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def percentile_normalize(img, lo=1, hi=99):
    """Percentile clip + stretch to uint8. CPU — fast, no GPU benefit."""
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, lo), np.percentile(img, hi)
    if p_hi > p_lo:
        img = np.clip(img, p_lo, p_hi)
        img = (img - p_lo) / (p_hi - p_lo)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def apply_clahe(img_8bit, clip_limit=2.0):
    """CLAHE — routes to GPU if available."""
    return apply_clahe_gpu(img_8bit, clip_limit=clip_limit)


def apply_frangi(img_8bit, sigma_lo=1.0, sigma_hi=5.0):
    """
    Frangi vessel filter. Routes to CuPy GPU if available;
    falls back to scikit-image CPU. GPU speedup ~8-10x here.
    """
    return apply_frangi_gpu(img_8bit, sigma_lo=sigma_lo, sigma_hi=sigma_hi)


# ── Thresholding ──────────────────────────────────────────────────────────────

def threshold_image(img, method="manual", value=30):
    """Binary threshold. CPU — transfer overhead not worth it."""
    if method == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = (img > value).astype(np.uint8) * 255
    return binary


# ── Morphology ────────────────────────────────────────────────────────────────

def morphological_open(binary, ksize):
    """Morphological opening — routes to GPU if available."""
    return morphological_open_gpu(binary, ksize)


def distance_dilation(binary, expand_px):
    """
    Isotropic distance-transform-based expansion.
    scipy.ndimage — no GPU equivalent worth the overhead.
    """
    if expand_px <= 0:
        return binary
    dist = ndi.distance_transform_edt(binary == 0)
    return (dist <= expand_px).astype(np.uint8) * 255


def fill_holes(binary, max_hole_size=0):
    """Fill enclosed holes. CPU-only (scipy)."""
    if max_hole_size == 0:
        filled = ndi.binary_fill_holes(binary > 0)
        return filled.astype(np.uint8) * 255

    inverted = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    result = binary.copy()
    h, w = binary.shape

    for i in range(1, num_labels):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_size:
            result[labels == i] = 255

    return result


def watershed_split(binary, min_distance=12):
    """Watershed separation. scikit-image CPU-only."""
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        return binary

    bool_mask = binary > 0
    if not bool_mask.any():
        return binary

    dist = ndi.distance_transform_edt(bool_mask)
    coords = peak_local_max(dist, min_distance=max(1, min_distance), labels=bool_mask)

    peak_mask = np.zeros(dist.shape, dtype=bool)
    if len(coords):
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    labels = watershed(-dist, markers, mask=bool_mask)
    return (labels > 0).astype(np.uint8) * 255


# ── Area filtering ────────────────────────────────────────────────────────────

def area_filter(binary, min_area=0, max_area=0):
    """Remove connected components outside [min_area, max_area]. CPU."""
    if min_area == 0 and max_area == 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area > 0 and a < min_area:
            continue
        if max_area > 0 and a > max_area:
            continue
        cleaned[labels == i] = 255
    return cleaned


# ── Main entry point (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=128)
def generate_mask(img_data, mode, params):
    """
    Generate a binary uint8 mask from a single-channel uint8 image.

    Parameters
    ----------
    img_data : 2-D uint8 numpy array (preview or full resolution)
    mode     : pipeline mode name string. "Model-based" routes to the
               optional addon system in utils/model_pipeline.py /
               plugins/ — see generate_mask_model_based() for details
               and the ModelAddonError contract.
    params   : flat dict of all pipeline parameters

    Returns
    -------
    Binary uint8 mask (0 or 255), same spatial size as img_data.
    GPU acceleration is transparent — no change at the call site.

    Raises
    ------
    plugins.base.ModelAddonError
        Only when mode == "Model-based" and no usable addon is
        selected/installed, or inference fails. Manual pipeline modes
        never raise this.
    """
    p = params
    img = img_data.copy()

    if mode == "Model-based":
        # Lazy import: utils.model_pipeline imports helpers from this
        # module, so importing it at module load time would create a
        # circular import. Deferred here, it's also only paid for when
        # this mode is actually used.
        from utils.model_pipeline import generate_mask_model_based
        return generate_mask_model_based(img, p)

    # 1. Percentile normalisation
    img = percentile_normalize(img, p.get("clip_lo", 1), p.get("clip_hi", 99))

    # 2. CLAHE (GPU)
    if p.get("use_clahe", False):
        img = apply_clahe(img, clip_limit=float(p.get("clahe_clip", 2.0)))

    # 3. Frangi (GPU — the big speedup)
    if p.get("use_frangi", False):
        img = apply_frangi(
            img,
            sigma_lo=float(p.get("frangi_lo", 1.0)),
            sigma_hi=float(p.get("frangi_hi", 5.0)),
        )

    # 4. Threshold
    binary = threshold_image(
        img,
        method=p.get("threshold_method", "manual"),
        value=int(p.get("threshold", 30)),
    )

    # 5. Morphological opening (GPU)
    binary = morphological_open(binary, int(p.get("open_ksize", 3)))

    # 6. Distance dilation
    binary = distance_dilation(binary, int(p.get("dilation_px", 0)))

    # 7. Fill holes
    if p.get("fill_holes", True):
        binary = fill_holes(binary, max_hole_size=int(p.get("max_hole_size", 0)))

    # 8. Watershed
    if p.get("watershed", False) and np.any(binary > 0):
        binary = watershed_split(binary, min_distance=int(p.get("watershed_min_dist", 12)))

    # 9. Area filter
    binary = area_filter(
        binary,
        min_area=int(p.get("min_area", 0)),
        max_area=int(p.get("max_area", 0)),
    )

    return binary
