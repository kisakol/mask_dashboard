"""
defaults.py
-----------
Application-wide constants, pipeline mode definitions, and default parameter sets.
All tunable defaults live here — change once, reflected everywhere.
"""

# ── Pipeline Modes ────────────────────────────────────────────────────────────
# Each mode provides a description and a sensible default parameter set.
# These are STARTING POINTS — every param is still adjustable by the user.

PIPELINE_MODES = {
    "Blob": {
        "description": "Round / oval cells — nuclei, immune cells, round tumour cells.",
        "params": {
            "clip_lo": 1, "clip_hi": 99,
            "use_clahe": True,  "clahe_clip": 2.0,
            "threshold_method": "manual", "threshold": 30,
            "open_ksize": 3,
            "dilation_px": 0,
            "fill_holes": True,  "max_hole_size": 0,
            "watershed": True,  "watershed_min_dist": 12,
            "use_frangi": False, "frangi_lo": 1.0, "frangi_hi": 5.0,
            "min_area": 150, "max_area": 0,
        },
    },
    "Cytoplasm": {
        "description": "Perinuclear halos, cytoplasm rings — expands from signal to fill gaps.",
        "params": {
            "clip_lo": 1, "clip_hi": 99,
            "use_clahe": True,  "clahe_clip": 2.0,
            "threshold_method": "manual", "threshold": 20,
            "open_ksize": 3,
            "dilation_px": 10,
            "fill_holes": True,  "max_hole_size": 800,
            "watershed": False, "watershed_min_dist": 10,
            "use_frangi": False, "frangi_lo": 1.0, "frangi_hi": 5.0,
            "min_area": 200, "max_area": 0,
        },
    },
    "Thin Structures": {
        "description": "Neuron axons, astrocyte processes — Frangi filter enhances fine filaments.",
        "params": {
            "clip_lo": 2, "clip_hi": 98,
            "use_clahe": True,  "clahe_clip": 3.0,
            "threshold_method": "manual", "threshold": 20,
            "open_ksize": 1,
            "dilation_px": 2,
            "fill_holes": False, "max_hole_size": 0,
            "watershed": False, "watershed_min_dist": 10,
            "use_frangi": True,  "frangi_lo": 1.0, "frangi_hi": 6.0,
            "min_area": 30,  "max_area": 0,
        },
    },
    "Diffuse": {
        "description": "Large glia soma, membrane staining, diffuse / low-contrast signal.",
        "params": {
            "clip_lo": 1, "clip_hi": 99,
            "use_clahe": False, "clahe_clip": 2.0,
            "threshold_method": "otsu", "threshold": 30,
            "open_ksize": 5,
            "dilation_px": 5,
            "fill_holes": True,  "max_hole_size": 2000,
            "watershed": False, "watershed_min_dist": 10,
            "use_frangi": False, "frangi_lo": 1.0, "frangi_hi": 5.0,
            "min_area": 500, "max_area": 0,
        },
    },
    "Custom": {
        "description": "Full manual control — no preset defaults applied.",
        "params": {
            "clip_lo": 1, "clip_hi": 99,
            "use_clahe": False, "clahe_clip": 2.0,
            "threshold_method": "manual", "threshold": 30,
            "open_ksize": 3,
            "dilation_px": 0,
            "fill_holes": True,  "max_hole_size": 0,
            "watershed": False, "watershed_min_dist": 10,
            "use_frangi": False, "frangi_lo": 1.0, "frangi_hi": 5.0,
            "min_area": 100, "max_area": 0,
        },
    },
}

PIPELINE_MODE_NAMES = list(PIPELINE_MODES.keys())

# ── Conflict resolution strategies ───────────────────────────────────────────
CONFLICT_STRATEGIES = [
    "Priority Order",   # Fall through to the ranked priority list
    "Give to First",    # Overlap pixel goes to the lower-index mask
    "Give to Second",   # Overlap pixel goes to the higher-index mask
    "Exclude Both",     # Overlap pixels removed from both masks
    "Keep Both",        # Overlap pixels kept in both (overlapping masks)
]

# ── Visual defaults ───────────────────────────────────────────────────────────
DEFAULT_CHANNEL_COLORS = [
    "#FF4444",  # Ch 0 — red
    "#44DDFF",  # Ch 1 — cyan
    "#AAFF44",  # Ch 2 — lime
    "#FF44DD",  # Ch 3 — magenta
    "#FFCC44",  # Ch 4 — amber
]

# Colors used in the composite (logic) preview — one per mask slot
DEFAULT_SLOT_COLORS = [
    "#FF4444",
    "#44DDFF",
    "#AAFF44",
    "#FF44DD",
    "#FFCC44",
]

# ── Export defaults ───────────────────────────────────────────────────────────
DEFAULT_FILENAME_TEMPLATE = "{stem}_{mask_name}"
EXPORT_FORMATS = ["PNG", "TIFF"]

FILENAME_TEMPLATE_HELP = (
    "Available tokens:\n"
    "- `{stem}` — source filename without extension\n"
    "- `{mask_name}` — mask slot label\n"
    "- `{date}` — today as YYYYMMDD\n"
    "- `{idx}` — slot index (0-based)\n"
    "- `{color}` — slot hex color without # (e.g. `FF4444`)\n\n"
    "Example: `{stem}_{mask_name}_{date}` → `ROI001_Nucleus_20250101`\n"
    "Example: `{stem}_{mask_name}_{color}` → `ROI001_Nucleus_FF4444`"
)

# ── Preview quality → max edge size (px) ─────────────────────────────────────
PREVIEW_SIZES = {"Low (fast)": 512, "Medium": 1024, "High (slow)": None}
