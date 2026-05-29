"""
utils/io_utils.py
-----------------
File I/O: loading multi-channel TIFFs, normalisation, preview downscaling,
mask saving, batch file listing.
"""

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import czifile
from PIL import Image

from config.defaults import PREVIEW_SIZES


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image_stack(path: str) -> np.ndarray | None:
    """
    Load a TIFF or CZI file and return a (C, H, W) uint array.
    Handles 2-D, 3-D, and 4-D TIFFs/CZIs with various axis orderings.
    Returns None on failure.
    """
    ext = str(path).lower().split('.')[-1]
    
    try:
        if ext == 'czi':
            with czifile.CziFile(path) as czi:
                arr = czi.asarray()
            arr = np.squeeze(arr)
        else:
            with tiff.TiffFile(str(path)) as tf:
                arr = tf.asarray()

        if arr.ndim == 2:
            # (H, W) → single-channel
            return arr[np.newaxis]

        elif arr.ndim == 3:
            s = arr.shape
            # Heuristic: smallest axis is most likely C
            if s[0] <= min(s[1], s[2]) and s[0] <= 16:
                return arr                          # already (C, H, W)
            elif s[2] <= min(s[0], s[1]) and s[2] <= 16:
                return np.transpose(arr, (2, 0, 1)) # (H, W, C) → (C, H, W)
            else:
                # Ambiguous depth / channel stack — treat first axis as C
                return arr

        elif arr.ndim >= 4:
            # Assume (Z, C, H, W) or (T, C, H, W) — max-project over first dimension
            return arr.max(axis=0)

        return arr

    except Exception:
        # PIL fallback for non-standard TIFFs
        if ext != 'czi':
            try:
                img = Image.open(path)
                arr = np.array(img)
                if arr.ndim == 3:
                    return np.transpose(arr, (2, 0, 1))
                return arr[np.newaxis]
            except Exception:
                return None
        return None


def normalize_to_8bit(arr: np.ndarray) -> np.ndarray:
    """
    Simple min–max normalise a single channel to uint8.
    Used at load time so downstream code always works with 0–255.
    """
    if arr.dtype == np.uint8:
        return arr
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        out = (arr.astype(np.float32) - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(arr, dtype=np.float32)
    return (out * 255).astype(np.uint8)


# ── Preview downscaling ───────────────────────────────────────────────────────

def make_preview(arr_8bit: np.ndarray, quality: str = "Low (fast)") -> np.ndarray:
    """
    Downscale a single-channel uint8 image for fast preview.
    Returns the original array if it already fits within the target size.
    """
    target = PREVIEW_SIZES.get(quality)
    if target is None:
        return arr_8bit  # High quality = no resize

    h, w = arr_8bit.shape
    if max(h, w) <= target:
        return arr_8bit

    scale = target / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    return cv2.resize(arr_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upscale_mask_to_full(mask_preview: np.ndarray, full_h: int, full_w: int) -> np.ndarray:
    """
    Nearest-neighbour upscale a preview-resolution mask to full resolution.
    Preserves binary values exactly.
    """
    if mask_preview.shape == (full_h, full_w):
        return mask_preview
    return cv2.resize(mask_preview, (full_w, full_h), interpolation=cv2.INTER_NEAREST)


# ── File listing ──────────────────────────────────────────────────────────────

def get_image_files(folder: str) -> list[str]:
    """Return sorted list of TIFF and CZI filenames in a directory."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff", ".czi"))
    )


# ── Saving ────────────────────────────────────────────────────────────────────

def build_filename(
    template: str,
    stem: str,
    mask_name: str,
    slot_idx: int,
    color: str = "",
) -> str:
    """
    Expand a filename template with available tokens.
    Tokens: {stem}, {mask_name}, {date}, {idx}, {color}
    """
    today = datetime.now().strftime("%Y%m%d")
    safe_mask_name = mask_name.strip().replace(" ", "_")
    # Strip # so the hex value is safe in filenames (e.g. FF4444)
    safe_color = color.lstrip("#").upper() if color else ""
    try:
        name = template.format(
            stem=stem,
            mask_name=safe_mask_name,
            date=today,
            idx=slot_idx,
            color=safe_color,
        )
    except KeyError:
        # Graceful fallback if user typed an unrecognised token
        name = f"{stem}_{safe_mask_name}"
    return name


def save_masks(
    resolved_masks: dict,
    slot_names: list[str],
    output_dir: str,
    filename_template: str,
    stem: str,
    fmt: str = "PNG",
    full_h: int | None = None,
    full_w: int | None = None,
    full_res_masks: dict | None = None,
    slot_colors: list[str] | None = None,
) -> list[str]:
    """
    Save resolved masks to disk.

    Parameters
    ----------
    resolved_masks   : {slot_idx: uint8 mask array}
    slot_names       : list of mask slot label strings
    output_dir       : directory to write files into
    filename_template: template string with {stem}, {mask_name}, {date}, {idx}, {color}
    stem             : filename stem of the source TIFF
    fmt              : "PNG" or "TIFF"
    full_h, full_w   : if provided, upscale masks before saving
    full_res_masks   : pre-computed full-resolution masks (skips upscaling)
    slot_colors      : hex color string per slot index (for {color} token)

    Returns
    -------
    List of paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = ".tif" if fmt == "TIFF" else ".png"
    saved = []

    for slot_idx, mask in resolved_masks.items():
        if slot_idx >= len(slot_names):
            continue

        # Upscale if needed
        if full_res_masks is not None and slot_idx in full_res_masks:
            out_mask = full_res_masks[slot_idx]
        elif full_h is not None and full_w is not None:
            out_mask = upscale_mask_to_full(mask, full_h, full_w)
        else:
            out_mask = mask

        color = slot_colors[slot_idx] if slot_colors and slot_idx < len(slot_colors) else ""
        fname = build_filename(filename_template, stem, slot_names[slot_idx], slot_idx, color=color) + ext
        fpath = os.path.join(output_dir, fname)

        if fmt == "TIFF":
            tiff.imwrite(fpath, out_mask)
        else:
            cv2.imwrite(fpath, out_mask)

        saved.append(fpath)

    return saved
