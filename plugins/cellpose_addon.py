"""
plugins/cellpose_addon.py
--------------------------
Cellpose addon — generalist pretrained cell/nucleus segmentation.

Optional dependency:
    pip install cellpose

Cellpose ships pretrained models for round/oval objects:
  - "nuclei" : DAPI / Hoechst nuclei
  - "cyto2"  : whole-cell / cytoplasm (2nd-gen generalist model)
  - "cyto3"  : whole-cell / cytoplasm (3rd-gen generalist model)

These map naturally onto the "Blob" and "Cytoplasm" pipeline modes,
but as a learned model rather than threshold + morphology.

API notes
---------
This targets the `models.Cellpose` convenience wrapper, whose
`.eval()` signature (`masks, flows, styles, diams = model.eval(...)`)
has been stable across Cellpose 1.x–3.x. If a future Cellpose release
changes this, `segment()` raises `ModelAddonError` with the underlying
exception message rather than crashing the dashboard.
"""

from __future__ import annotations

import numpy as np

from plugins.base import ModelAddonError, SegmentationAddon


class CellposeAddon(SegmentationAddon):
    key = "cellpose"
    display_name = "Cellpose"
    description = (
        "Generalist deep-learning segmentation for round/oval cells and "
        "nuclei. Good starting point for Blob/Cytoplasm-style channels "
        "with variable shape or touching objects."
    )
    install_hint = "pip install cellpose"

    # ── Availability ──────────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cellpose  # noqa: F401
            return True
        except Exception:
            return False

    # ── Metadata ──────────────────────────────────────────────────────────

    @classmethod
    def model_options(cls) -> list[str]:
        return ["nuclei", "cyto2", "cyto3"]

    @classmethod
    def default_params(cls) -> dict:
        return {
            "model_name": "nuclei",
            "diameter": 30.0,   # 0 = let Cellpose auto-estimate
            "confidence": 0.4,  # mapped to flow_threshold
            "use_gpu": False,
        }

    # ── Model loading ─────────────────────────────────────────────────────

    @classmethod
    def load_model(cls, model_name: str, use_gpu: bool = False):
        try:
            from cellpose import models
        except Exception as e:
            raise ModelAddonError(
                f"Cellpose is not installed or failed to import ({e}). "
                f"Install with: {cls.install_hint}"
            ) from e

        try:
            return models.Cellpose(gpu=use_gpu, model_type=model_name)
        except Exception as e:
            raise ModelAddonError(
                f"Failed to load Cellpose model '{model_name}': {e}"
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────

    @classmethod
    def segment(cls, model, img_8bit: np.ndarray, params: dict) -> np.ndarray:
        diameter = float(params.get("diameter", 30.0))
        diameter = diameter if diameter > 0 else None

        # "confidence" (0-1, higher = stricter) is mapped onto Cellpose's
        # flow_threshold (typical range ~0.1-1.0, default 0.4). Higher
        # flow_threshold => fewer, more confident masks.
        flow_threshold = float(params.get("confidence", 0.4))

        try:
            masks, _flows, _styles, _diams = model.eval(
                img_8bit,
                diameter=diameter,
                channels=[0, 0],
                flow_threshold=flow_threshold,
            )
        except Exception as e:
            raise ModelAddonError(f"Cellpose inference failed: {e}") from e

        return (np.asarray(masks) > 0).astype(np.uint8) * 255
