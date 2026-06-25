"""
components/export_panel.py
--------------------------
Tab 3 — Export.

  • Output directory (relative or absolute)
  • Filename template with live preview
  • Format: PNG or TIFF
  • Single-file save (upscales preview masks to full resolution)
  • Batch mode: process every TIFF in the folder with the same params
"""

import os

import streamlit as st

from config.defaults import DEFAULT_FILENAME_TEMPLATE, EXPORT_FORMATS, FILENAME_TEMPLATE_HELP
from utils.io_utils import (
    build_filename,
    get_image_files,
    load_image_stack,
    make_preview,
    normalize_to_8bit,
    save_masks,
    upscale_mask_to_full,
)
from utils.logic import (
    aggregate_to_slots,
    calculate_partition_masks,
    get_partitions,
    resolve_conflicts,
)
from utils.pipeline import generate_mask


def render_export_panel_tab() -> None:
    use_channels: list[int] = st.session_state.get("use_channels", [])
    resolved: dict = st.session_state.get("resolved_masks_preview", {})
    mask_slots: list[str] = st.session_state.get("mask_slots", [])
    num_masks: int = st.session_state.get("num_masks", 3)
    file_stem: str = st.session_state.get("file_stem", "output")

    if not use_channels or not resolved:
        st.info("Define masks and logic in the first two tabs before exporting.")
        return

    st.caption(
        "Configure output options below. "
        "Masks are upscaled from preview to full resolution at save time."
    )

    # ── Output settings ───────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        folder_path: str = st.session_state.get("folder_path", os.getcwd())
        out_dir = st.text_input(
            "Output directory",
            value=os.path.join(folder_path, "Generated_Masks"),
            key="output_dir",
            help="Absolute or relative path. Created automatically if it doesn't exist.",
        )

        fmt = st.selectbox("File format", EXPORT_FORMATS, key="export_format")

    with col_b:
        template = st.text_input(
            "Filename template",
            value=st.session_state.get("filename_template", DEFAULT_FILENAME_TEMPLATE),
            key="filename_template",
            help=FILENAME_TEMPLATE_HELP,
        )

        # Resolve slot colors (from session state slot_colors, falling back to defaults)
        slot_colors = _get_slot_colors(num_masks)

        # Live preview of resolved filenames
        ext = ".tif" if fmt == "TIFF" else ".png"
        st.markdown("**Preview:**")
        for i in range(num_masks):
            slot_name = mask_slots[i] if i < len(mask_slots) else f"Slot{i}"
            color = slot_colors[i] if i < len(slot_colors) else ""
            fname = build_filename(template, file_stem, slot_name, i, color=color) + ext
            st.code(fname, language=None)

    st.divider()

    # ── Single-file save ──────────────────────────────────────────────────────
    st.markdown("#### Save current file")

    if st.button("💾 Generate & Save Masks (full resolution)", type="primary"):
        _save_current_file(use_channels, mask_slots, num_masks, out_dir, template, fmt, slot_colors)

    st.divider()

    # ── Batch mode ────────────────────────────────────────────────────────────
    st.markdown("#### Batch processing")
    st.caption(
        "Apply the **same parameters** to every TIFF in the source folder. "
        "Useful for processing a whole GeoMx ROI set with consistent settings."
    )

    folder_path = st.session_state.get("folder_path", os.getcwd())
    tiff_files = get_image_files(folder_path)
    current_file = os.path.basename(st.session_state.get("current_file", ""))

    st.info(
        f"Found **{len(tiff_files)}** TIFF file(s) in folder. "
        f"Current file: `{current_file}`."
    )

    selected_files = st.multiselect(
        "Files to process (default: all)",
        options=tiff_files,
        default=tiff_files,
        key="batch_files",
    )

    if st.button("Run Batch Export", type="secondary"):
        _run_batch(selected_files, folder_path, use_channels, mask_slots, num_masks, out_dir, template, fmt, slot_colors)


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_current_file(
    use_channels, mask_slots, num_masks, out_dir, template, fmt, slot_colors=None
) -> None:
    data_cache: dict = st.session_state.data_cache
    file_stem: str = st.session_state.get("file_stem", "output")
    channel_params: dict = st.session_state.channel_params
    pmap: dict = st.session_state.partitions_map
    priority: list = st.session_state.priority_order
    strategies: dict = st.session_state.get("conflict_strategies", {})
    preview_quality: str = st.session_state.get("preview_quality", "Low (fast)")

    with st.spinner("Computing full-resolution masks…"):
        try:
            full_h, full_w = data_cache[use_channels[0]].shape
            full_masks = {}

            for ch in use_channels:
                cp = channel_params.get(ch, {})
                gen_params = {k: v for k, v in cp.items() if k not in ("mode", "color")}
                mode = cp.get("mode", "Blob")
                img_full = data_cache[ch]
                full_masks[ch] = generate_mask(img_full, mode, gen_params)

            partitions = get_partitions(use_channels)
            part_masks = calculate_partition_masks(partitions, full_masks, full_h, full_w, channels=use_channels)
            slot_masks = aggregate_to_slots(part_masks, pmap, num_masks, full_h, full_w)
            resolved_full = resolve_conflicts(slot_masks, priority, strategies)

            saved = save_masks(
                resolved_full, mask_slots, out_dir, template, file_stem,
                fmt=fmt, slot_colors=slot_colors
            )

            st.success(f"✅ Saved {len(saved)} mask(s) to `{out_dir}`")
            for p in saved:
                st.code(os.path.basename(p), language=None)

        except Exception as e:
            st.error(f"Save failed: {e}")


def _run_batch(
    files: list[str],
    folder: str,
    use_channels: list[int],
    mask_slots: list[str],
    num_masks: int,
    out_dir: str,
    template: str,
    fmt: str,
    slot_colors: list[str] | None = None,
) -> None:
    if not files:
        st.warning("No files selected.")
        return

    channel_params: dict = st.session_state.channel_params
    pmap: dict = st.session_state.partitions_map
    priority: list = st.session_state.priority_order
    strategies: dict = st.session_state.get("conflict_strategies", {})
    preview_quality: str = st.session_state.get("preview_quality", "Low (fast)")

    progress = st.progress(0, text="Starting batch…")
    errors = []

    for file_idx, fname in enumerate(files):
        fpath = os.path.join(folder, fname)
        stem = os.path.splitext(fname)[0]
        progress.progress(file_idx / len(files), text=f"Processing {fname}…")

        try:
            stack = load_image_stack(fpath)
            if stack is None:
                errors.append(f"{fname}: could not load")
                continue

            stack_8bit = {i: normalize_to_8bit(stack[i]) for i in range(stack.shape[0])}
            available_ch = list(stack_8bit.keys())
            batch_channels = [ch for ch in use_channels if ch in available_ch]

            if not batch_channels:
                errors.append(f"{fname}: none of the selected channels found")
                continue

            full_h, full_w = stack_8bit[batch_channels[0]].shape
            full_masks = {}

            for ch in batch_channels:
                cp = channel_params.get(ch, {})
                gen_params = {k: v for k, v in cp.items() if k not in ("mode", "color")}
                mode = cp.get("mode", "Blob")
                full_masks[ch] = generate_mask(stack_8bit[ch], mode, gen_params)

            partitions = get_partitions(batch_channels)
            part_masks = calculate_partition_masks(partitions, full_masks, full_h, full_w, channels=batch_channels)
            slot_masks = aggregate_to_slots(part_masks, pmap, num_masks, full_h, full_w)
            resolved_full = resolve_conflicts(slot_masks, priority, strategies)

            save_masks(resolved_full, mask_slots, out_dir, template, stem, fmt=fmt, slot_colors=slot_colors)

        except Exception as e:
            errors.append(f"{fname}: {e}")

    progress.progress(1.0, text="Done.")

    if errors:
        st.warning(f"Completed with {len(errors)} error(s):")
        for err in errors:
            st.code(err, language=None)
    else:
        st.success(f"✅ Batch complete — {len(files)} file(s) saved to `{out_dir}`")

# ── Color helper ─────────────────────────────────────────────────────────────

def _get_slot_colors(num_masks: int) -> list[str]:
    """
    Return per-slot hex colors.
    Tries to pick colors from channel_params for the channel that 'owns' each
    slot (by convention slot i → channel i), falling back to DEFAULT_SLOT_COLORS.
    """
    from config.defaults import DEFAULT_SLOT_COLORS
    colors = []
    channel_params = st.session_state.get("channel_params", {})
    for i in range(num_masks):
        cp = channel_params.get(i, {})
        colors.append(cp.get("color", DEFAULT_SLOT_COLORS[i % len(DEFAULT_SLOT_COLORS)]))
    return colors
