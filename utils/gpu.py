"""
utils/gpu.py
------------
GPU acceleration layer for the mask pipeline.

Detection strategy (in order of preference):
  1. CuPy   — CUDA-based GPU arrays; used for Frangi (hessian eigenvalue computation)
              and other NumPy-style operations. Fastest for large float operations.
  2. OpenCV CUDA — cv2.cuda module, available when opencv-contrib-python is built
              with CUDA. Used for morphology, threshold, CLAHE, Gaussian blur.
  3. CPU fallback — standard NumPy / SciPy / OpenCV (always available).

Why split?
  • Frangi is dominated by Gaussian derivative convolutions and eigenvalue sorting
    on float32 arrays → CuPy wins here (~5–10× vs CPU scikit-image).
  • Morphology and CLAHE are dominated by memory bandwidth with small kernels
    → OpenCV CUDA wins because it keeps data on GPU without CuPy overhead.

Runtime queries
  All external code should use the three booleans:
      HAS_CUPY      — CuPy importable and a CUDA device is present
      HAS_CV_CUDA   — cv2.cuda module present and at least one CUDA device
      GPU_AVAILABLE — HAS_CUPY or HAS_CV_CUDA (anything is faster than pure CPU)

Switching at runtime
  Call `set_gpu_enabled(False)` to force CPU paths (useful for debugging or
  when the user explicitly disables GPU in the sidebar).
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable

import numpy as np
import cv2


# ── Detection (runs once at import time) ─────────────────────────────────────

def _detect_cupy() -> bool:
    try:
        import cupy as cp  # noqa: F401
        cp.cuda.Device(0).use()          # raises if no CUDA device
        _ = cp.zeros(1)                  # forces device init
        return True
    except Exception:
        return False


def _detect_cv_cuda() -> bool:
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception:
        return False


HAS_CUPY: bool = _detect_cupy()
HAS_CV_CUDA: bool = _detect_cv_cuda()
GPU_AVAILABLE: bool = HAS_CUPY or HAS_CV_CUDA

# Module-level flag — can be toggled by the UI at runtime
_GPU_ENABLED: bool = True


def set_gpu_enabled(enabled: bool) -> None:
    global _GPU_ENABLED
    _GPU_ENABLED = enabled


def gpu_is_active() -> bool:
    """True when GPU is both available and enabled by the user."""
    return GPU_AVAILABLE and _GPU_ENABLED


def gpu_status_string() -> str:
    """Human-readable status for display in the sidebar."""
    if not GPU_AVAILABLE:
        return "⚪ No GPU detected. Running on CPU"
    if not _GPU_ENABLED:
        return "⚫ GPU available but disabled"
    parts = []
    if HAS_CUPY:
        try:
            import cupy as cp
            props = cp.cuda.Device(0).attributes
            # CuPy ≥ 10 exposes device name differently
            name = getattr(cp.cuda.Device(0), "name", None) or "CUDA device"
            if isinstance(name, bytes):
                name = name.decode()
            parts.append(f"CuPy ({name})")
        except Exception:
            parts.append("CuPy (CUDA)")
    if HAS_CV_CUDA:
        parts.append("OpenCV CUDA")
    return "🟢 GPU active: " + " + ".join(parts)


# ── GPU-accelerated Frangi vessel filter ─────────────────────────────────────
#
# The scikit-image Frangi runs per-scale Gaussian derivatives on CPU.
# This reimplements the core (2-D case) using CuPy:
#   • Gaussian smoothing with cupy.ndimage.gaussian_filter
#   • Hessian matrix elements (Ixx, Ixy, Iyy) via second-order Gaussian derivatives
#   • Eigenvalue computation from the 2×2 Hessian determinant / trace
#   • Frangi vesselness: exp(−Rb²/2β²) · (1 − exp(−S²/2c²))
# For each scale the result is max-projected across scales.

def _frangi_cupy(
    img_f32: np.ndarray,
    sigmas: np.ndarray,
    beta: float = 0.5,
    c_ratio: float = 15.0,
    black_ridges: bool = False,
) -> np.ndarray:
    """
    GPU Frangi using CuPy. img_f32 is a float32 array in [0, 1].
    Returns float32 vesselness in [0, 1].
    """
    import cupy as cp
    import cupy.ndimage as cpndi

    gpu_img = cp.asarray(img_f32)
    result = cp.zeros_like(gpu_img)

    for sigma in sigmas:
        if sigma == 0:
            continue

        # Second-order Gaussian derivatives (Hessian elements)
        # We use gaussian_filter with order parameter
        Ixx = cpndi.gaussian_filter(gpu_img, sigma=sigma, order=(2, 0))
        Iyy = cpndi.gaussian_filter(gpu_img, sigma=sigma, order=(0, 2))
        Ixy = cpndi.gaussian_filter(gpu_img, sigma=sigma, order=(1, 1))

        # Scale normalisation (σ² enhances structures at this scale)
        scale = sigma ** 2
        Ixx *= scale
        Iyy *= scale
        Ixy *= scale

        if black_ridges:
            Ixx, Iyy, Ixy = -Ixx, -Iyy, -Ixy

        # Eigenvalues of the symmetric 2×2 Hessian
        # λ = (Ixx+Iyy)/2 ± sqrt(((Ixx-Iyy)/2)² + Ixy²)
        tmp1 = (Ixx + Iyy) * 0.5
        tmp2 = cp.sqrt(((Ixx - Iyy) * 0.5) ** 2 + Ixy ** 2)
        l1 = tmp1 - tmp2   # smaller eigenvalue
        l2 = tmp1 + tmp2   # larger eigenvalue (absolute)

        # Keep only pixels where l2 < 0 (bright tubular structures on dark bg)
        # Vesselness is 0 wherever the larger eigenvalue is positive
        valid = l2 < 0

        # Rb = |l1| / |l2|  (shape anisotropy — close to 0 for vessels)
        denom = cp.where(cp.abs(l2) > 1e-10, l2, cp.full_like(l2, 1e-10))
        Rb = l1 / denom

        # S² = l1² + l2²  (structure magnitude)
        S2 = l1 ** 2 + l2 ** 2

        # Heuristic c = half the max S in the image
        S_max = float(cp.sqrt(S2).max())
        c = S_max / c_ratio if S_max > 0 else 1.0

        beta2 = 2 * beta ** 2
        c2 = 2 * c ** 2

        vesselness = cp.exp(-(Rb ** 2) / beta2) * (1.0 - cp.exp(-S2 / c2))
        vesselness = cp.where(valid, vesselness, 0.0)

        result = cp.maximum(result, vesselness)

    # Normalise to [0, 1]
    vmax = float(result.max())
    if vmax > 0:
        result /= vmax

    return cp.asnumpy(result).astype(np.float32)


def apply_frangi_gpu(
    img_8bit: np.ndarray,
    sigma_lo: float = 1.0,
    sigma_hi: float = 5.0,
    beta: float = 0.5,
    c_ratio: float = 15.0,
) -> np.ndarray:
    """
    GPU Frangi if CuPy is available and GPU is enabled, else fall back to
    scikit-image CPU Frangi.

    Parameters
    ----------
    img_8bit : uint8 single-channel image
    sigma_lo : smallest vessel width (σ, pixels)
    sigma_hi : largest vessel width (σ, pixels)

    Returns
    -------
    uint8 vesselness image (0–255)
    """
    n_steps = max(2, int((sigma_hi - sigma_lo) / 0.5) + 1)
    sigmas = np.linspace(sigma_lo, sigma_hi, n_steps)
    norm = img_8bit.astype(np.float32) / 255.0

    if gpu_is_active() and HAS_CUPY:
        enhanced = _frangi_cupy(norm, sigmas, beta=beta, c_ratio=c_ratio)
    else:
        # CPU fallback via scikit-image
        try:
            from skimage.filters import frangi as ski_frangi
            enhanced = ski_frangi(norm, sigmas=sigmas, black_ridges=False)
            enhanced = enhanced.astype(np.float32)
            vmax = enhanced.max()
            if vmax > 0:
                enhanced /= vmax
        except ImportError:
            return img_8bit   # last resort: return unfiltered

    return (enhanced * 255).astype(np.uint8)


# ── GPU-accelerated CLAHE ─────────────────────────────────────────────────────

def apply_clahe_gpu(img_8bit: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE using OpenCV CUDA if available, else CPU.
    """
    if gpu_is_active() and HAS_CV_CUDA:
        try:
            clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(img_8bit)
            gpu_dst = clahe.apply(gpu_src)
            return gpu_dst.download()
        except Exception:
            pass   # fall through to CPU

    # CPU path
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img_8bit)


# ── GPU-accelerated morphological opening ─────────────────────────────────────

def morphological_open_gpu(binary: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphological opening using OpenCV CUDA if available, else CPU.
    """
    if ksize <= 1:
        return binary

    ksize = ksize if ksize % 2 == 1 else ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if gpu_is_active() and HAS_CV_CUDA:
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(binary)
            morph_filter = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, cv2.CV_8U, kernel
            )
            gpu_dst = morph_filter.apply(gpu_src)
            return gpu_dst.download()
        except Exception:
            pass   # fall through to CPU

    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ── GPU-accelerated Gaussian blur (used for pre-smoothing) ────────────────────

def gaussian_blur_gpu(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """
    Gaussian blur using OpenCV CUDA if available, else CPU.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if gpu_is_active() and HAS_CV_CUDA:
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(img)
            gauss_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8U, cv2.CV_8U, (ksize, ksize), sigma
            )
            gpu_dst = gauss_filter.apply(gpu_src)
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.GaussianBlur(img, (ksize, ksize), sigma)
