"""
plugins/base.py
----------------
Base interface for optional "Model-based" segmentation addons
(Cellpose, StarDist, future custom U-Net / CellViT models, ...).

DESIGN GOAL
-----------
The core dashboard (pipeline.py, channel_tuning.py, etc.) must work
identically whether ZERO or MANY of these addons are installed.

To guarantee that:

  1. This module — and every addon module under plugins/ — must be
     SAFE TO IMPORT even if the underlying ML library (cellpose,
     stardist, torch, tensorflow, ...) is not installed. All heavy /
     optional imports happen lazily, inside methods, wrapped so that
     ImportError (or any other failure) is reported via
     `is_available() -> False` rather than raised at import time.

  2. Every addon declares its own dependency-availability check
     (`is_available`), human-readable metadata (`display_name`,
     `description`, `install_hint`), and a small, addon-agnostic
     parameter surface (`model_options`, `default_params`) so the UI
     can render generic widgets without special-casing each addon.

  3. `segment()` always returns a uint8 binary mask (0 / 255) with the
     same (H, W) as the input image — exactly like the output of the
     classic 9-stage pipeline — so it can be dropped into the same
     post-processing (morphology / hole-fill / area-filter) and the
     same partition/logic/export machinery downstream.

  4. Any failure during model loading or inference should raise
     `ModelAddonError` with a clear, user-facing message. Callers
     (utils/model_pipeline.py) catch this and surface it as a
     `st.warning` rather than crashing the whole app.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ModelAddonError(Exception):
    """Raised when a model addon can't be used (missing deps, bad params,
    inference failure, ...). Always carries a user-facing message."""


class SegmentationAddon(ABC):
    """Common interface every model-based segmentation addon implements."""

    # Short stable key used in session state / params (e.g. "cellpose")
    key: str = ""

    # Human-readable name shown in the UI (e.g. "Cellpose")
    display_name: str = ""

    # One-line description shown in the UI / tooltips
    description: str = ""

    # Shown to the user when is_available() is False, e.g.
    # "pip install cellpose"
    install_hint: str = ""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this addon's dependencies are importable.

        MUST NOT raise — wrap any import attempts in try/except and
        return False on failure.
        """
        ...

    @classmethod
    @abstractmethod
    def model_options(cls) -> list[str]:
        """Return the list of pretrained model identifiers this addon
        supports, for display in a selectbox."""
        ...

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict[str, Any]:
        """Return addon-specific default parameter values (merged into
        the generic 'Model-based' pipeline-mode params)."""
        ...

    @classmethod
    @abstractmethod
    def load_model(cls, model_name: str, use_gpu: bool = False):
        """Load (and ideally cache) the underlying model object.

        Raises
        ------
        ModelAddonError
            If the model can't be loaded (missing deps, bad model name,
            download failure, etc.)
        """
        ...

    @classmethod
    @abstractmethod
    def segment(cls, model, img_8bit: np.ndarray, params: dict) -> np.ndarray:
        """Run inference and return a uint8 binary mask (0/255), same
        spatial shape as `img_8bit`.

        Raises
        ------
        ModelAddonError
            If inference fails.
        """
        ...
