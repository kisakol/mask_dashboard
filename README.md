# GeoMx Mask Dashboard 

An interactive, browser-based dashboard for generating, previewing, and exporting segmentation masks from GeoMx multi-channel TIFF files. Built with Streamlit, with optional GPU acceleration for demanding stain types.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
   - [CPU-only (default)](#cpu-only-default)
   - [GPU acceleration (optional)](#gpu-acceleration-optional)
4. [Running the App](#running-the-app)
5. [Workflow Overview](#workflow-overview)
6. [Pipeline Modes](#pipeline-modes)
7. [Model Addons (Model-based mode)](#model-addons-model-based-mode)
8. [Overlap & Conflict Resolution](#overlap--conflict-resolution)
9. [Filename Templates](#filename-templates)
10. [Preset System](#preset-system)
11. [Batch Processing](#batch-processing)
12. [GPU Acceleration Details](#gpu-acceleration-details)
13. [Module Reference](#module-reference)
14. [Adding a New Pipeline Mode](#adding-a-new-pipeline-mode)
15. [Known Limitations](#known-limitations)
16. [Roadmap](#roadmap)

---

## Quick Start

```bash
# 1. Clone or download the project
cd geomx_mask_dashboard_v5

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

Paste your GeoMx TIFF folder path into the sidebar, select a file, and follow the three tabs.

---

## Project Structure

```
geomx_mask_dashboard_v5/
│
├── app.py                        # Entry point: page config, session state, tab layout
│
├── config/
│   └── defaults.py               # All constants — pipeline modes, conflict strategies,
│                                 # default params, colour palettes, export settings
│
├── utils/
│   ├── gpu.py                    # GPU detection + accelerated ops (Frangi/CLAHE/morph)
│   ├── pipeline.py               # All mask generation logic, @st.cache_data wrapped
│   ├── model_pipeline.py         # "Model-based" mode glue — dispatches to plugins/, then
│   │                             # applies the standard morphology/hole-fill/area-filter
│   ├── logic.py                  # Venn partitions, slot aggregation, conflict resolution
│   ├── render.py                 # Plotly histograms, channel overlays, composite preview
│   └── io_utils.py               # TIFF loading, normalisation, filename templating, saving
│
├── plugins/                      # Optional model-based segmentation addons
│   ├── base.py                   # SegmentationAddon interface + ModelAddonError
│   ├── registry.py                # Addon discovery — available_addons(), get_addon()
│   ├── cellpose_addon.py          # Cellpose addon (pip install cellpose)
│   └── stardist_addon.py          # StarDist addon (pip install stardist tensorflow)
│
├── components/
│   ├── sidebar.py                # File loader, channel selector, mask naming, preset UI
│   ├── channel_tuning.py         # Tab 1: per-channel histogram + parameters + live overlay
│   ├── logic_panel.py            # Tab 2: partition assignment, priority, composite preview
│   └── export_panel.py           # Tab 3: filename template, single-file & batch save
│
├── requirements.txt
└── README.md
```

**Dependency graph** (no circular imports):

```
app.py
  ├── components/sidebar.py       → utils/gpu, utils/io_utils, config/defaults, utils/logic
  ├── components/channel_tuning.py → utils/pipeline, utils/render, config/defaults
  ├── components/logic_panel.py  → utils/logic, utils/render, config/defaults
  └── components/export_panel.py → utils/io_utils, utils/logic, utils/pipeline, config/defaults

utils/pipeline.py → utils/gpu, scipy, cv2, skimage
utils/gpu.py      → cv2, (cupy optional), (skimage optional)
utils/logic.py    → numpy, cv2
utils/render.py   → numpy, cv2, PIL, plotly
utils/io_utils.py → numpy, cv2, tifffile, PIL
```

---

## Installation

### CPU-only (default)

Works on any machine without a GPU or with a GPU that has no CUDA drivers.

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `numpy` | Array operations |
| `opencv-python` | Image processing (CPU) |
| `Pillow` | Image enhancement, PIL overlays |
| `tifffile` | Multi-channel TIFF I/O |
| `scipy` | Distance transform, hole filling |
| `scikit-image` | Frangi filter (CPU), watershed, peak detection |
| `plotly` | Interactive histograms |

### GPU acceleration (optional)

Two independent GPU components. Install either, both, or neither. the app auto-detects.

#### Component 1: CuPy (Frangi on GPU. biggest speedup)

Requires CUDA toolkit (11.x or 12.x) and a CUDA-capable GPU.

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

Full installation guide: https://docs.cupy.dev/en/stable/install.html

#### Component 2: OpenCV with CUDA (CLAHE, morphology, Gaussian blur)

The standard `opencv-python` package does not include CUDA support. Options:

**Option A: conda/mamba (recommended if using Anaconda/Miniconda):**

```bash
conda install -c conda-forge opencv
# or
mamba install -c conda-forge opencv
```
Conda's OpenCV build links against the CUDA toolkit if it is present.

**Option B: pre-built CUDA wheels:**

```bash
pip uninstall opencv-python
# Download matching wheel from:
# https://github.com/cudawarping/opencv-python-cuda-wheels
pip install opencv_contrib_python_cuda-<version>.whl
```

**Option C: build from source:**

```bash
# Full guide: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
cmake -D WITH_CUDA=ON -D BUILD_opencv_python3=ON ...
```

The app detects GPU availability at startup and shows status in the sidebar. You can toggle GPU off at runtime for debugging without restarting.

---

## Running the App

```bash
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Disable browser auto-open (headless server)
streamlit run app.py --server.headless true
```

**Memory note:** Preview images are stored in session state. At "Low" quality (512px max edge), memory usage is negligible. At "High" quality (full resolution), expect ~50–200 MB per open file depending on image dimensions and channel count.

---

## Workflow Overview

The application follows a linear three-tab workflow:

```
Sidebar                  Tab 1                   Tab 2                  Tab 3
────────                 ──────────────           ──────────────         ──────────
Load TIFF          →     Tune each channel  →     Assign partitions →   Export masks
Select channels          (histogram + mask         to output slots       (filename template,
Name mask slots          overlay, live)           Set priority &         PNG/TIFF, batch)
Save/load presets                                 conflict rules
                                                  Live composite
                                                  preview
```

All previews run at the selected quality level (Low/Medium/High). Full resolution is computed only at export time.

---

## Pipeline Modes

Each channel is processed independently. Mode selection provides sensible defaults; every parameter is still individually adjustable.

### Blob
**Target:** Round/oval cells. DAPI-stained nuclei, immune cells, round tumour cells.

Processing chain:
1. Percentile clip (1st–99th) → stretch to uint8
2. CLAHE contrast enhancement
3. Manual or Otsu threshold
4. Morphological opening (noise removal)
5. Fill enclosed holes (all sizes)
6. Watershed separation (optional. splits touching nuclei)
7. Area filter

### Cytoplasm
**Target:** Perinuclear halos, cytoplasm rings, markers that stain around the nucleus.

Processing chain:
1. Percentile clip → stretch
2. CLAHE
3. Low threshold (captures dim cytoplasmic signal)
4. Morphological opening
5. Distance-transform expansion (`dilation_px`). isotropic outward growth
6. Selective hole filling (fills gaps up to `max_hole_size` px²)
7. Area filter

The distance expansion is the key step: it grows the thresholded signal outward uniformly in all directions, filling the cytoplasmic region even when the raw signal is patchy.

### Thin Structures
**Target:** Neuron axons, astrocyte processes, any filamentous/dendritic morphology.

Processing chain:
1. Percentile clip (2nd–98th, tighter to preserve dynamic range)
2. CLAHE (higher clip limit: `3.0` default)
3. **Frangi vessel filter**: enhances tubular/filamentous structures across multiple scales (σ = `frangi_lo` to `frangi_hi`). This is the computationally dominant step. GPU acceleration gives ~8–10× speedup here.
4. Threshold (on the Frangi-enhanced image)
5. Minimal morphological opening (kernel 1 recommended to preserve thin structures)
6. Small dilation (2px default. restores connectivity after thresholding)
7. Area filter (small min_area. processes often have small area despite being long)

**Important:** Do not use Watershed or large morphological kernels in this mode. both will destroy thin process morphology.

### Diffuse
**Target:** Large glia soma, membrane staining, low-contrast diffuse signal.

Processing chain:
1. Percentile clip → stretch (no CLAHE. avoids amplifying background noise in diffuse images)
2. Otsu threshold (adapts to the bimodal histogram typical of membrane stains)
3. Large morphological opening (smooths irregular boundaries)
4. Dilation (5px default. connects nearby diffuse blobs)
5. Large hole filling (up to 2000 px²)
6. Area filter (large min_area. ignores small noise blobs)

### Custom
Full manual control. No defaults are applied when switching to this mode; all sliders retain their current values.

### Model-based
**Target:** Any channel, when an installed AI segmentation addon outperforms threshold-based pipelines (e.g. touching/irregular nuclei, variable staining intensity).

Processing chain:
1. Percentile clip → stretch (light preprocessing only. the model handles its own normalisation; CLAHE/Frangi are skipped)
2. Selected addon's model runs inference on the preprocessed image, producing an instance mask
3. Morphological opening, distance dilation, hole filling, area filter. same post-processing as the manual modes, using the same controls

This mode is only meaningful if at least one addon under `plugins/` is installed (see [Model Addons](#model-addons-model-based-mode) below). With none installed, it still appears in the mode selector (for discoverability) but produces an empty mask and shows an install-hint warning. **it never breaks the rest of the dashboard.**

---

## Model Addons (Model-based mode)

"Model-based" mode is implemented as a small plugin system under `plugins/`, designed so that:

- The dashboard works identically with **zero** addons installed.
- Each addon is **independent**. installing/removing one never affects another, or the manual pipeline modes.
- A failed or missing addon degrades to an `st.warning` + empty mask for that channel, never a crash.

### Built-in addons

| Addon | Install | Best for |
|---|---|---|
| **Cellpose** | `pip install cellpose` | Generalist round/oval cells. nuclei`, `cyto2`, `cyto3` pretrained models. Good Blob/Cytoplasm alternative for variable shapes or touching objects. |
| **StarDist** | `pip install stardist tensorflow` | Star-convex object detection. 2D_versatile_fluo` is a strong DAPI-nuclei default. Often cleaner than threshold + watershed for densely packed nuclei. |

Both addons can use GPU acceleration via their own backends (PyTorch for Cellpose, TensorFlow for StarDist). independent of the CuPy/OpenCV-CUDA layer in `utils/gpu.py`.

### How it works

1. `components/channel_tuning.py` calls `plugins.registry.available_addons()`, which only returns addons whose dependencies actually import successfully (checked via each addon's `is_available()`, never raises).
2. If the list is non-empty, the "AI Model" section shows an **Addon** dropdown, a **Pretrained model** dropdown (populated from `addon.model_options()`), a **confidence/probability threshold** slider, an **object diameter** field, and a GPU toggle.
3. If the list is empty, the section instead shows install instructions for every registered addon (`plugins.registry.all_addons()`).
4. `utils/model_pipeline.generate_mask_model_based()` loads the model (cached via `st.cache_resource`, keyed on addon + model name + GPU flag), runs `addon.segment(...)`, and applies the standard morphology/hole-fill/area-filter steps.
5. Any failure, missing deps, bad model name, inference error, raises `plugins.base.ModelAddonError`, caught in `channel_tuning.py` and shown as `st.warning(...)` with an empty mask for that channel.

### Adding a new addon

1. Create `plugins/<your_addon>.py` implementing `plugins.base.SegmentationAddon`:
   - `is_available()` : try/except around the import, never raises
   - `model_options()` : list of pretrained model identifiers
   - `default_params()` : addon-specific defaults (merged into the generic Model-based params)
   - `load_model(model_name, use_gpu)` : return a loaded model object, or raise `ModelAddonError`
   - `segment(model, img_8bit, params)` : return a uint8 binary mask (0/255), or raise `ModelAddonError`
2. Register it in `plugins/registry.py`'s `ADDON_CLASSES` dict.
3. The UI and pipeline pick it up automatically through the registry. No other files need changing.

---

## Overlap & Conflict Resolution

### Venn Partitions

When multiple channels are selected, the app computes all exclusive pixel regions:

- **Ch0 only** : pixels positive in channel 0, negative in all others
- **Ch1 only** : pixels positive in channel 1, negative in all others
- **Ch0 ∩ Ch1** : pixels positive in both, negative in all others
- (and so on for 3–4 channels)

Each partition is assigned to an output mask slot (or left unassigned/discarded) via a dropdown in Tab 2.

### Priority Order

A ranked list of mask slots. After partition assignment, any pixel that still appears in more than one slot is resolved by this ordering: the highest-ranked slot claims the pixel; lower-ranked slots yield.

Reorder using the ↑/↓ buttons in Tab 2.

### Per-Pair Conflict Strategies

Applied *before* the priority waterfall. For every unique pair of mask slots (A vs B, A vs C, B vs C, etc.), you choose:

| Strategy | Effect |
|---|---|
| **Priority Order** | Defers to the priority list : default |
| **Give to A** | Overlap pixels always go to A; B yields |
| **Give to B** | Overlap pixels always go to B; A yields |
| **Exclude Both** | Overlap pixels removed from both masks (double-positives discarded) |
| **Keep Both** | Overlap pixels kept in both (masks allowed to overlap in output) |

**Typical biology mappings:**

| Scenario | Recommended strategy |
|---|---|
| DAPI (nucleus) vs cytoplasm marker | Give to DAPI : nucleus wins, cytoplasm ring wraps around |
| Two cell-type markers (double-positive is artefact) | Exclude Both |
| Astrocyte soma vs astrocyte processes | Give to soma (priority), processes are separate slot |
| Exploratory / unknown | Priority Order, adjust after seeing composite |

### Execution Order

```
1. Per-pair explicit strategies applied (in index order)
2. Priority waterfall: highest-priority slot takes contested pixels
3. Lower-priority slots lose those pixels
```

---

## Filename Templates

Output filenames are built from a user-defined template string. Available tokens:

| Token | Value | Example |
|---|---|---|
| `{stem}` | Source filename without extension | `ROI001` |
| `{mask_name}` | Mask slot label (spaces → underscores) | `Nucleus` |
| `{date}` | Today as YYYYMMDD | `20250601` |
| `{idx}` | Slot index, 0-based | `0` |
| `{color}` | Slot hex colour without `#` | `FF4444` |

**Examples:**

```
Template: {stem}_{mask_name}
Output:   ROI001_Nucleus.png

Template: {stem}_{mask_name}_{date}
Output:   ROI001_Nucleus_20250601.png

Template: {date}_{idx}_{stem}_{mask_name}
Output:   20250601_0_ROI001_Nucleus.png

Template: {stem}_{mask_name}_{color}
Output:   ROI001_Nucleus_FF4444.png
```

A live filename preview for each mask slot updates in real time as you type the template.

---

## Preset System

All tunable parameters can be saved as a named preset and reloaded in a later session.

**What is saved:**
- Per-channel pipeline parameters (mode + all sliders)
- Mask slot names and count
- Partition → slot assignment
- Priority order
- Per-pair conflict strategies
- Filename template

**Operations:**
- **Save**: name + click Save
- **Load**: select from dropdown + click Load
- **Delete**: select from dropdown + click Delete
- **Export all**: downloads `geomx_presets.json`
- **Import**: upload a `.json` preset file (merges with existing presets)

Presets survive browser refreshes (stored in Streamlit session state). They do **not** persist across server restarts unless exported to JSON. Export presets regularly if you want to share them between machines or colleagues.

---

## Batch Processing

Tab 3 → Batch processing section.

1. Select the files to process (defaults to all TIFFs in the folder)
2. Click **Run Batch Export**

The app applies the **same parameters** as the currently configured session to every selected file. Each file is loaded, masks are generated at full resolution, and outputs are saved to the output directory with filenames derived from the template.

Files that fail (missing channels, corrupt TIFF, etc.) are reported in a summary at the end without interrupting the rest of the batch.

**Note:** Batch runs do not update the preview in Tab 1 or Tab 2. Open a file manually in the sidebar to inspect results for a specific ROI.

---

## GPU Acceleration Details

The GPU status is shown at the top of the sidebar (⚪ no GPU / 🟢 active). A toggle disables GPU at runtime without restarting.

### What runs on GPU

| Operation | GPU backend | Typical speedup |
|---|---|---|
| **Frangi vessel filter** | CuPy (CUDA) | **~8–10×** |
| CLAHE | OpenCV CUDA | ~5× |
| Morphological opening | OpenCV CUDA | ~4× |
| Gaussian blur (internal) | OpenCV CUDA | ~3–5× |

### What always runs on CPU

| Operation | Reason |
|---|---|
| Percentile normalisation | Negligible time |
| Distance-transform dilation | No OpenCV CUDA equivalent; scipy only |
| Fill holes | scipy binary_fill_holes |
| Watershed | scikit-image CPU |
| Area filter | OpenCV CPU connectedComponents |

### Overall speedup by mode

| Mode | CPU (2048×2048) | GPU (RTX-class) | Speedup |
|---|---|---|---|
| Thin Structures (Frangi) | 8–15 s | 1–2 s | **~8–10×** |
| Blob / Cytoplasm / Diffuse | 0.2–0.5 s | 0.1–0.2 s | ~2× |

The dominant bottleneck across all modes on CPU is the Frangi filter. For Blob/Cytoplasm/Diffuse, which do not use Frangi, CPU performance is already fast and the GPU benefit is modest.

### CuPy Frangi implementation note

The GPU Frangi re-implements the 2-D Frangi vesselness formula using CuPy ndimage:
- Per-scale Hessian matrix computation (Ixx, Iyy, Ixy via second-order Gaussian derivatives)
- Scale-space normalisation (σ² weighting)
- Eigenvalue computation from the symmetric 2×2 Hessian (closed-form, no eigensolver needed)
- Vesselness: `exp(−Rb²/2β²) · (1 − exp(−S²/2c²))`, max-projected across scales

This matches the scikit-image Frangi output for the 2-D case (the CPU fallback). Results are numerically equivalent within float32 precision.

---

## Module Reference

### `config/defaults.py`

All constants in one place. To change a default, edit here. no other files need touching.

| Symbol | Type | Purpose |
|---|---|---|
| `PIPELINE_MODES` | dict | Mode names → description + default params |
| `CONFLICT_STRATEGIES` | list | Available strategy names |
| `DEFAULT_CHANNEL_COLORS` | list | Per-channel hex colours (up to 5 channels) |
| `DEFAULT_SLOT_COLORS` | list | Per-slot hex colours in composite preview |
| `DEFAULT_FILENAME_TEMPLATE` | str | `{stem}_{mask_name}` |
| `EXPORT_FORMATS` | list | `["PNG", "TIFF"]` |
| `FILENAME_TEMPLATE_HELP` | str | Help text shown in the export UI |
| `PREVIEW_SIZES` | dict | Quality label → max edge px (None = full res) |

### `utils/gpu.py`

GPU detection and accelerated operation wrappers.

| Symbol | Purpose |
|---|---|
| `HAS_CUPY` | bool. CuPy importable + CUDA device present |
| `HAS_CV_CUDA` | bool. OpenCV CUDA module present + device present |
| `GPU_AVAILABLE` | bool. either backend available |
| `gpu_is_active()` | Returns True when GPU is available and enabled |
| `set_gpu_enabled(bool)` | Runtime toggle (called by sidebar) |
| `gpu_status_string()` | Human-readable status for UI display |
| `apply_frangi_gpu(img, σ_lo, σ_hi)` | Frangi: CuPy GPU or scikit-image CPU |
| `apply_clahe_gpu(img, clip)` | CLAHE: OpenCV CUDA or CPU |
| `morphological_open_gpu(img, k)` | Opening: OpenCV CUDA or CPU |
| `gaussian_blur_gpu(img, k, σ)` | Gaussian blur: OpenCV CUDA or CPU |

### `utils/pipeline.py`

Stateless mask generation. All functions are pure (same in → same out).

| Symbol | Purpose |
|---|---|
| `generate_mask(img, mode, params)` | Main entry point, `@st.cache_data` decorated |
| `percentile_normalize(img, lo, hi)` | Robust normalisation |
| `apply_clahe(img, clip)` | Routes to gpu.py |
| `apply_frangi(img, σ_lo, σ_hi)` | Routes to gpu.py |
| `threshold_image(img, method, value)` | Manual or Otsu |
| `morphological_open(img, k)` | Routes to gpu.py |
| `distance_dilation(img, px)` | scipy distance transform expansion |
| `fill_holes(img, max_size)` | Selective or full hole filling |
| `watershed_split(img, min_dist)` | scikit-image watershed |
| `area_filter(img, min, max)` | Connected component area filter |

`generate_mask` takes a hashable `params` dict. Streamlit caches the result keyed on `(img_data bytes, mode, params)`. changing any parameter correctly invalidates only that channel's cached mask.

### `utils/logic.py`

Venn algebra and conflict resolution.

| Symbol | Purpose |
|---|---|
| `get_partitions(channels)` | All exclusive subsets of a channel list |
| `calculate_partition_masks(parts, ch_masks, h, w)` | Pixel mask per partition |
| `aggregate_to_slots(part_masks, assignment, n, h, w)` | Merge partitions into slot masks |
| `resolve_conflicts(slot_masks, priority, strategies)` | Apply per-pair rules then priority waterfall |
| `default_priority_order(n)` | Returns `[0, 1, 2, ...]` |

### `utils/render.py`

Pure visualisation functions (no Streamlit calls).

| Symbol | Purpose |
|---|---|
| `hex_to_rgb(hex)` | `'#FF4444'` → `(255, 68, 68)` |
| `histogram_figure(img, threshold, color)` | Plotly bar histogram with threshold vline |
| `channel_overlay(img, mask, color, brightness, opacity)` | Grayscale + coloured mask overlay |
| `composite_preview(resolved, colors, h, w, bg, bg_opacity)` | Multi-slot non-overlapping composite |

### `utils/io_utils.py`

File I/O with format-agnostic TIFF loading.

| Symbol | Purpose |
|---|---|
| `load_image_stack(path)` | Returns `(C, H, W)` uint array from any TIFF |
| `normalize_to_8bit(arr)` | Min–max stretch to uint8 |
| `make_preview(arr, quality)` | Downscale to preview size |
| `upscale_mask_to_full(mask, h, w)` | Nearest-neighbour upscale |
| `get_tiff_files(folder)` | Sorted list of `.tif`/`.tiff` files |
| `build_filename(template, stem, name, idx, color)` | Token expansion |
| `save_masks(resolved, names, dir, template, stem, ...)` | Write PNG or TIFF to disk |

---

## Adding a New Pipeline Mode

1. Open `config/defaults.py` and add an entry to `PIPELINE_MODES`:

```python
"My New Mode": {
    "description": "One-line description shown as tooltip in the UI.",
    "params": {
        "clip_lo": 1, "clip_hi": 99,
        "use_clahe": False, "clahe_clip": 2.0,
        "threshold_method": "manual", "threshold": 50,
        "open_ksize": 3,
        "dilation_px": 0,
        "fill_holes": True, "max_hole_size": 0,
        "watershed": False, "watershed_min_dist": 12,
        "use_frangi": False, "frangi_lo": 1.0, "frangi_hi": 5.0,
        "min_area": 100, "max_area": 0,
    },
},
```

2. The mode appears automatically in the Tab 1 mode selector. No other files need changing.

3. If the mode requires a new processing step not already in `utils/pipeline.py`, add the function there and wire it into `generate_mask()` by reading a new key from `params`.

---

## Known Limitations

- **Presets are session-scoped.** They live in Streamlit session state and are lost on server restart. Use the JSON export/import to persist them.
- **Batch mode uses preview-quality params.** The `@st.cache_data` cache is keyed on the full-resolution image content at export time, so full-res masks are computed correctly. However, if you change parameters after a batch run, you must re-run the batch.
- **TIFF axis inference is heuristic.** The loader assumes the smallest axis with ≤16 planes is the channel axis. Unusual TIFF axis orderings (e.g. T×C×H×W time series) may load incorrectly. Check the channel count shown in the sidebar.
- **Frangi is 2-D only.** For Z-stacks, the loader max-projects along Z before any processing. True 3-D Frangi is not implemented.
- **No undo.** Parameter changes are immediate. Use presets to checkpoint good parameter sets before experimenting.

---

## Roadmap

**Point 5: Model-based automatic segmentation**

✅ **Done:** The plugin architecture is implemented (`plugins/`, `utils/model_pipeline.py`, "Model-based" mode in Tab 1. see [Model Addons](#model-addons-model-based-mode)). Two pretrained-model addons ship today:

| Target | Addon | Pretrained model |
|---|---|---|
| Nuclei (DAPI) | StarDist | `2D_versatile_fluo` |
| Nuclei (DAPI) | Cellpose | `nuclei` |
| Cytoplasm / whole-cell | Cellpose | `cyto2` / `cyto3` |

**Still planned (pending custom training data):**

| Target | Architecture candidate |
|---|---|
| Membrane | Custom U-Net |
| Astrocyte | Custom U-Net, trained on GFAP-stained GeoMx data |
| Microglia | Custom U-Net, trained on Iba1 / CD68 data |
| Neuron | Custom U-Net or Cellpose with custom class |
| Cancer cells | CellViT or custom ViT-based segmentation |
| Immune cells | CellViT or custom |

These would each ship as additional addon modules under `plugins/` (see [Adding a new addon](#adding-a-new-addon)), requiring custom-trained weights rather than off-the-shelf pretrained models. Manual pipeline modes remain available for cases where models underperform on novel tissue types.
