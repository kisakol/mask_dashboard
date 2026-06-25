"""
plugins/stardist_addon.py
--------------------------
StarDist addon — star-convex object detection, strong for round nuclei.

Optional dependency:
    pip install stardist tensorflow

StarDist ships pretrained 2D models:
  - "2D_versatile_fluo" : fluorescence nuclei (DAPI-like) — best fit
    for GeoMx nuclear channels
  - "2D_versatile_he"   : H&E-stained nuclei
  - "2D_paper_dsb2018"  : the original DSB-2018 challenge model

API notes
---------
This targets `StarDist2D.from_pretrained()` and
`model.predict_instances(img, prob_thresh=...)`, the documented
stable StarDist 2D API. Inputs are normalised via csbdeep's
`normalize()` helper, matching the StarDist usage examples.
"""

from __future__ import annotations

import numpy as np

from plugins.base import ModelAddonError, SegmentationAddon


class StarDistAddon(SegmentationAddon):
    key = "stardist"
    display_name = "StarDist"
    description = (
        "Star-convex object detection. Particularly strong for round, "
        "well-separated nuclei (DAPI) — often segments touching nuclei "
        "more cleanly than threshold + watershed."
    )
    install_hint = "pip install stardist tensorflow"

    # ── Availability ──────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        try:
            import stardist  # noqa: F401
            import tensorflow  # noqa: F401
            return True
        except Exception:
            return False

    # ── Metadata ──────────────────────────────────────────────────────────

    @classmethod
    def model_options(cls) -> list[str]:
        return ["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"]

    @classmethod
    def default_params(cls) -> dict:
        return {
            "model_name": "2D_versatile_fluo",
            "confidence": 0.5,  # mapped to prob_thresh
            "use_gpu": False,   # StarDist/TF picks up GPU automatically
        }

    # ── Model loading ─────────────────────────────────────────────────────

    @classmethod
    def load_model(cls, model_name: str, use_gpu: bool = False):
        try:
            from stardist.models import StarDist2D
        except Exception as e:
            raise ModelAddonError(
                f"StarDist is not installed or failed to import ({e}). "
                f"Install with: {cls.install_hint}"
            ) from e

        try:
            return StarDist2D.from_pretrained(model_name)
        except Exception as e:
            raise ModelAddonError(
                f"Failed to load StarDist model '{model_name}': {e}"
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────

    @classmethod
    def segment(cls, model, img_8bit: np.ndarray, params: dict) -> np.ndarray:
        try:
            from csbdeep.utils import normalize
        except Exception as e:
            raise ModelAddonError(
                f"StarDist's csbdeep dependency failed to import: {e}"
            ) from e

        prob_thresh = float(params.get("confidence", 0.5))

        try:
            img_norm = normalize(img_8bit.astype(np.float32))
            labels, _details = model.predict_instances(img_norm, prob_thresh=prob_thresh)
        except Exception as e:
            raise ModelAddonError(f"StarDist inference failed: {e}") from e

        return (np.asarray(labels) > 0).astype(np.uint8) * 255
