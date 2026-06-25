"""
plugins/registry.py
--------------------
Central registry of optional "Model-based" segmentation addons.

Adding a new addon
-------------------
1. Create plugins/<your_addon>.py implementing `SegmentationAddon`
   (see plugins/base.py). Keep all heavy imports lazy/inside methods.
2. Add it to ADDON_CLASSES below.

That's it — the UI (channel_tuning.py) and pipeline
(utils/model_pipeline.py) discover addons entirely through this
registry, so nothing else needs to change. If the addon's dependencies
aren't installed, it simply won't appear in `available_addons()`, and
the rest of the dashboard is unaffected.
"""

from __future__ import annotations

from plugins.base import SegmentationAddon
from plugins.cellpose_addon import CellposeAddon
from plugins.stardist_addon import StarDistAddon

# Every known addon, regardless of whether its dependencies are installed.
ADDON_CLASSES: dict[str, type[SegmentationAddon]] = {
    CellposeAddon.key: CellposeAddon,
    StarDistAddon.key: StarDistAddon,
}


def all_addons() -> dict[str, type[SegmentationAddon]]:
    """Every registered addon class, regardless of availability."""
    return dict(ADDON_CLASSES)


def available_addons() -> dict[str, type[SegmentationAddon]]:
    """Registered addon classes whose dependencies are importable."""
    result = {}
    for key, cls in ADDON_CLASSES.items():
        try:
            if cls.is_available():
                result[key] = cls
        except Exception:
            # Never let a broken addon's availability check break the
            # rest of the dashboard.
            continue
    return result


def get_addon(key: str) -> type[SegmentationAddon] | None:
    """Look up an addon class by key, or None if unknown."""
    return ADDON_CLASSES.get(key)
