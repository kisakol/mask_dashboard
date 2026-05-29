# GeoMx Mask Dashboard v5

Interactive dashboard for generating, previewing, and exporting segmentation masks
from GeoMx multi-channel TIFF files.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

---

## Project Structure

```
geomx_mask_dashboard_v5/
│
├── app.py                        # Entry point — page config, session state, tab layout
│
├── config/
│   └── defaults.py               # All constants: pipeline modes, conflict strategies,
│                                 # default params, colors, export settings
│
├── utils/
│   ├── io_utils.py               # TIFF loading, normalisation, preview scaling, saving
│   ├── pipeline.py               # All mask generation logic (Blob, Cytoplasm, Thin, Diffuse)
│   ├── logic.py                  # Venn partitions, slot aggregation, conflict resolution
│   └── render.py                 # Overlays, histograms (Plotly), composite preview
│
├── components/
│   ├── sidebar.py                # File loading, channel selection, mask naming, presets
│   ├── channel_tuning.py         # Tab 1: per-channel histogram + params + live overlay
│   ├── logic_panel.py            # Tab 2: partition assignment, priority, composite preview
│   └── export_panel.py           # Tab 3: filename templates, single-file & batch save
│
└── requirements.txt
```

---

## Pipeline Modes

| Mode | Best for | Key operations |
|---|---|---|
| **Blob** | Nuclei, round immune cells | CLAHE → threshold → fill → watershed |
| **Cytoplasm** | Perinuclear halos, cytoplasm rings | Threshold → distance-expand → fill |
| **Thin Structures** | Neuron axons, astrocyte processes | Frangi filter → threshold → area filter |
| **Diffuse** | Large glia, membrane, low-contrast | Otsu → close → fill |
| **Custom** | Full manual control | All parameters exposed |

---

## Conflict Resolution Strategies (per pair)

- **Priority Order** — defers to the priority list (default)
- **Give to First / Second** — one mask always wins
- **Exclude Both** — overlap pixels removed from both (double-positives discarded)
- **Keep Both** — overlap allowed (masks may overlap in output)

---

## Filename Template Tokens

`{stem}` · `{mask_name}` · `{date}` (YYYYMMDD) · `{idx}` (slot index)

Example: `{stem}_{mask_name}_{date}` → `ROI001_Nucleus_20250101.png`

---

## Adding a New Pipeline Mode

1. Add an entry to `PIPELINE_MODES` in `config/defaults.py` with a `description` and `params` dict.
2. The mode will automatically appear in the Tab 1 mode selector.
3. If the mode needs a new processing step, add it to `utils/pipeline.py` and wire it up
   in the `generate_mask()` function — it reads everything from the `params` dict.

---

## Roadmap (Point 5 — Model-Based Auto-Segmentation)

Planned next phase: trained models for nuclei, cytoplasm, membrane, astrocyte,
microglia, neuron, cancer cell, and immune cell masks.

Architecture candidates:
- **Cellpose** (pre-trained, fine-tuneable) — nuclei/cytoplasm/membrane
- **StarDist** — convex nuclei segmentation
- **Custom U-Net** — trained on your GeoMx image library for domain-specific stains
