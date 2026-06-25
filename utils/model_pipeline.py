"""
utils/model_pipeline.py
------------------------
"Model-based" pipeline mode — routes to optional addons under plugins/
(Cellpose, StarDist, ...) for AI-based instance segmentation, then
applies the same lightweight post-processing (morphology / hole-fill /
area-filter) used by the manual pipelines so results plug straight into
the existing partition/logic/export machinery.

Import-safety
-------------
This module is safe to import even with zero ML addons installed —
plugins/registry.py only reports addons whose dependencies actually
import successfully. If the user selects "Model-based" mode without a
usable addon, `generate_mask_model_based` raises `ModelAddonError`,
which `components/channel_tuning.py` catches and shows as a
`st.warning` (the channel's preview mask becomes empty rather than the
app crashing).

Caching
-------
Model objects are cached with `st.cache_resource` (correct for
non-serialisable objects like loaded neural nets), keyed on
(addon_key, model_name, use_gpu). This is separate from the
`st.cache_data` cache on `generate_mask` / this function's result.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from plugins.base import ModelAddonError
from plugins.registry import get_addon
from utils.pipeline import (
    percentile_normalize,
    morphological_open,
    distance_dilation,
    fill_holes,
    area_filter,
)


@st.cache_resource(show_spinner=False)
def _load_model_cached(addon_key: str, model_name: str, use_gpu: bool):
    addon = get_addon(addon_key)
    if addon is None:
        raise ModelAddonError(f"Unknown model addon '{addon_key}'.")
    return addon.load_model(model_name, use_gpu=use_gpu)


def generate_mask_model_based(img_8bit: np.ndarray, params: dict) -> np.ndarray:
    """
    Run the selected model addon on `img_8bit`, then apply the standard
    morphology / hole-fill / area-filter post-processing.

    Parameters
    ----------
    img_8bit : 2-D uint8 numpy array (preview or full resolution)
    params   : flat dict of pipeline parameters, including
               "model_addon" (registry key, e.g. "cellpose"),
               "model_name" (addon-specific model identifier),
               "confidence", "diameter", "use_gpu", plus the usual
               post-processing keys (open_ksize, dilation_px,
               fill_holes, max_hole_size, min_area, max_area,
               clip_lo, clip_hi).

    Returns
    -------
    Binary uint8 mask (0 or 255), same spatial size as img_8bit.

    Raises
    ------
    ModelAddonError
        If no addon is selected, the selected addon isn't installed,
        or model loading/inference fails. Callers should catch this
        and degrade gracefully (e.g. st.warning + empty mask).
    """
    p = params
    addon_key = p.get("model_addon", "")
    model_name = p.get("model_name", "")

    if not addon_key:
        raise ModelAddonError(
            "No model addon selected. Choose one in the 'AI Model' section, "
            "or pick a different Pipeline mode."
        )

    addon = get_addon(addon_key)
    if addon is None:
        raise ModelAddonError(f"Unknown model addon '{addon_key}'.")

    if not addon.is_available():
        raise ModelAddonError(
            f"{addon.display_name} is selected but not installed. "
            f"Install with: {addon.install_hint}"
        )

    if not model_name:
        raise ModelAddonError(
            f"No model selected for {addon.display_name}. Pick one in the "
            f"'AI Model' section."
        )

    # Light preprocessing — percentile stretch only (the model handles its
    # own normalisation/contrast internally; CLAHE/Frangi are skipped).
    img = percentile_normalize(img_8bit, p.get("clip_lo", 1), p.get("clip_hi", 99))

    model = _load_model_cached(addon_key, model_name, bool(p.get("use_gpu", False)))
    binary = addon.segment(model, img, p)

    # Standard post-processing, shared with the manual pipelines.
    binary = morphological_open(binary, int(p.get("open_ksize", 1)))
    binary = distance_dilation(binary, int(p.get("dilation_px", 0)))

    if p.get("fill_holes", True):
        binary = fill_holes(binary, max_hole_size=int(p.get("max_hole_size", 0)))

    binary = area_filter(
        binary,
        min_area=int(p.get("min_area", 0)),
        max_area=int(p.get("max_area", 0)),
    )

    return binary
