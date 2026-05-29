"""
components/sidebar.py
---------------------
Sidebar: file loading, channel selection, mask slot naming, preset save/load.
All persistent state is written to st.session_state.
"""

import json
import os

import streamlit as st

from config.defaults import (
    DEFAULT_CHANNEL_COLORS,
    DEFAULT_FILENAME_TEMPLATE,
    PIPELINE_MODES,
    PREVIEW_SIZES,
)
from utils.gpu import GPU_AVAILABLE, gpu_status_string, set_gpu_enabled
from utils.io_utils import get_image_files, load_image_stack, make_preview, normalize_to_8bit
from utils.logic import default_priority_order


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔬 GeoMx Masks v5")
        st.divider()

        _section_gpu_status()
        st.divider()

        _section_data_loading()
        st.divider()

        if st.session_state.data_cache:
            _section_channel_selection()
            st.divider()
            _section_mask_slots()
            st.divider()
            _section_presets()



# ── Section: GPU status ───────────────────────────────────────────────────────

def _section_gpu_status() -> None:
    status = gpu_status_string()
    st.markdown(status)

    if GPU_AVAILABLE:
        enabled = st.toggle(
            "Use GPU acceleration",
            value=st.session_state.get("gpu_enabled", True),
            key="gpu_enabled",
            help=(
                "Toggle GPU off to force CPU paths for debugging.\n\n"
                "**What runs on GPU:**\n"
                "- Frangi vessel filter (CuPy) — biggest speedup (~8-10×)\n"
                "- CLAHE (OpenCV CUDA)\n"
                "- Morphological opening (OpenCV CUDA)\n\n"
                "**Always CPU:** distance dilation, hole filling, watershed, area filter."
            ),
        )
        set_gpu_enabled(enabled)


# ── Section: data loading ─────────────────────────────────────────────────────

def _section_data_loading() -> None:
    st.markdown("#### 📂 Data")

    folder = st.text_input(
        "Folder path",
        value=st.session_state.get("folder_path", os.getcwd()),
        key="folder_path",
        help="Paste the directory containing your GeoMx TIFF or CZI files.",
    )

    quality = st.select_slider(
        "Preview quality",
        options=list(PREVIEW_SIZES.keys()),
        value=st.session_state.get("preview_quality", "Low (fast)"),
        key="preview_quality",
        help="Lower = faster response. Full resolution is used at export.",
    )

    image_files = get_image_files(folder)

    if not image_files:
        st.caption("No image files found in folder.")
        return

    selected = st.selectbox("Select file", image_files, key="selected_file")

    full_path = os.path.join(folder, selected)

    # Reload when file or quality changes
    file_changed = st.session_state.get("current_file") != full_path
    quality_changed = st.session_state.get("_loaded_quality") != quality

    if file_changed or quality_changed:
        with st.spinner("Loading image stack…"):
            stack = load_image_stack(full_path)
            if stack is None:
                st.error("Failed to load image. Check the file.")
                return

            stack_8bit = {i: normalize_to_8bit(stack[i]) for i in range(stack.shape[0])}
            preview_cache = {
                i: make_preview(arr, quality) for i, arr in stack_8bit.items()
            }

        st.session_state.data_cache = stack_8bit
        st.session_state.preview_cache = preview_cache
        st.session_state.current_file = full_path
        st.session_state._loaded_quality = quality
        st.session_state.file_stem = os.path.splitext(selected)[0]

        # Reset derived state
        st.session_state.base_masks_preview = {}
        st.session_state.partitions_map = {}
        st.session_state.resolved_masks_preview = {}

        # Don't wipe channel_params — keeps user's tuning if they switch back
        st.rerun()

    n_ch = len(st.session_state.data_cache)
    first_ch = st.session_state.preview_cache[0]
    h, w = first_ch.shape
    st.caption(f"{n_ch} channel{'s' if n_ch > 1 else ''} · preview {w}×{h} px")


# ── Section: channel selection ────────────────────────────────────────────────

def _section_channel_selection() -> None:
    st.markdown("#### Channels")

    available = list(st.session_state.data_cache.keys())

    use_channels = st.multiselect(
        "Channels to mask",
        options=available,
        default=st.session_state.get("use_channels", available[:min(3, len(available))]),
        format_func=lambda c: f"Channel {c}",
        key="use_channels",
    )

    # Ensure channel_params has entries for all selected channels
    for ch in use_channels:
        if ch not in st.session_state.channel_params:
            st.session_state.channel_params[ch] = {
                "mode": "Blob",
                "color": DEFAULT_CHANNEL_COLORS[ch % len(DEFAULT_CHANNEL_COLORS)],
                **PIPELINE_MODES["Blob"]["params"],
            }


# ── Section: mask slots ───────────────────────────────────────────────────────

def _section_mask_slots() -> None:
    st.markdown("#### Output Masks")

    num_masks = st.number_input(
        "Number of output masks",
        min_value=1, max_value=6,
        value=st.session_state.get("num_masks", 3),
        key="num_masks",
    )

    # Resize slot list
    slots = st.session_state.mask_slots
    if len(slots) < num_masks:
        for i in range(len(slots), num_masks):
            slots.append(f"Mask {chr(65 + i)}")
    elif len(slots) > num_masks:
        slots = slots[:num_masks]
    st.session_state.mask_slots = slots

    # Sync priority order
    order = st.session_state.priority_order
    valid = [i for i in order if i < num_masks]
    missing = [i for i in range(num_masks) if i not in valid]
    st.session_state.priority_order = valid + missing

    st.caption("Rename slots:")
    for i in range(num_masks):
        new_name = st.text_input(
            f"Slot {i + 1}",
            value=slots[i],
            key=f"slot_name_{i}",
            label_visibility="collapsed",
        )
        st.session_state.mask_slots[i] = new_name


# ── Section: presets ──────────────────────────────────────────────────────────

def _section_presets() -> None:
    st.markdown("#### 💾 Presets")

    presets: dict = st.session_state.get("presets", {})

    col_name, col_save = st.columns([3, 1])
    with col_name:
        preset_name = st.text_input("Name", key="preset_name_input", placeholder="e.g. DAPI nuclei")
    with col_save:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save", key="btn_save_preset", use_container_width=True):
            if preset_name.strip():
                presets[preset_name.strip()] = {
                    "channel_params": dict(st.session_state.channel_params),
                    "mask_slots": list(st.session_state.mask_slots),
                    "num_masks": st.session_state.num_masks,
                    "partitions_map": {
                        str(k): v for k, v in st.session_state.partitions_map.items()
                    },
                    "priority_order": list(st.session_state.priority_order),
                    "conflict_strategies": {
                        str(k): v for k, v in st.session_state.conflict_strategies.items()
                    },
                    "filename_template": st.session_state.get("filename_template", DEFAULT_FILENAME_TEMPLATE),
                }
                st.session_state.presets = presets
                st.success(f"Saved '{preset_name.strip()}'")

    if presets:
        col_load, col_del = st.columns([3, 1])
        with col_load:
            selected_preset = st.selectbox("Load preset", options=["—"] + list(presets.keys()), key="preset_select")
        with col_del:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load", key="btn_load_preset", use_container_width=True):
                if selected_preset != "—":
                    _apply_preset(presets[selected_preset])

        if st.button("🗑 Delete selected preset", key="btn_del_preset"):
            if selected_preset != "—" and selected_preset in presets:
                del presets[selected_preset]
                st.session_state.presets = presets
                st.rerun()

        # JSON export/import
        with st.expander("Import / Export JSON"):
            preset_json = json.dumps(presets, indent=2)
            st.download_button(
                "⬇ Export all presets",
                data=preset_json,
                file_name="geomx_presets.json",
                mime="application/json",
            )
            uploaded = st.file_uploader("⬆ Import presets JSON", type="json", key="preset_upload")
            if uploaded is not None:
                try:
                    imported = json.loads(uploaded.read())
                    presets.update(imported)
                    st.session_state.presets = presets
                    st.success(f"Imported {len(imported)} preset(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")


def _apply_preset(preset: dict) -> None:
    """Write preset values back into session state and rerun."""
    if "channel_params" in preset:
        st.session_state.channel_params = preset["channel_params"]
    if "mask_slots" in preset:
        st.session_state.mask_slots = preset["mask_slots"]
    if "num_masks" in preset:
        st.session_state.num_masks = preset["num_masks"]
    if "priority_order" in preset:
        st.session_state.priority_order = preset["priority_order"]
    if "conflict_strategies" in preset:
        # Keys were serialised as strings; restore as tuples
        strategies = {}
        for k, v in preset["conflict_strategies"].items():
            try:
                key = tuple(int(x) for x in k.strip("()").split(",") if x.strip())
                strategies[key] = v
            except Exception:
                pass
        st.session_state.conflict_strategies = strategies
    if "partitions_map" in preset:
        pmap = {}
        for k, v in preset["partitions_map"].items():
            try:
                key = tuple(int(x) for x in k.strip("()").split(",") if x.strip())
                pmap[key] = v
            except Exception:
                pass
        st.session_state.partitions_map = pmap
    if "filename_template" in preset:
        st.session_state.filename_template = preset["filename_template"]
    st.rerun()
