"""
components/channel_tuning.py
----------------------------
Tab 1 — Per-channel parameter tuning.

For each selected channel:
  • Pipeline mode selector (Blob / Cytoplasm / Thin Structures / Diffuse /
    Custom / Model-based)
  • Preprocessing controls (percentile clip, CLAHE)
  • Thresholding (histogram with live threshold line + slider, or Otsu)
  • Morphology (open kernel, dilation, hole filling, watershed)
  • Thin-structure Frangi filter controls
  • AI Model controls (Model-based mode — optional Cellpose/StarDist addons)
  • Area filtering (min / max component size)
  • Live overlay: raw channel + coloured mask

Results stored in st.session_state.base_masks_preview for Tab 2.
"""

import numpy as np
import streamlit as st

from config.defaults import DEFAULT_CHANNEL_COLORS, PIPELINE_MODE_NAMES, PIPELINE_MODES
from plugins.base import ModelAddonError
from plugins.registry import all_addons, available_addons
from utils.pipeline import generate_mask
from utils.render import channel_overlay, histogram_figure


def render_channel_tuning_tab() -> None:
    use_channels: list[int] = st.session_state.get("use_channels", [])

    if not use_channels:
        st.info("Select channels in the sidebar → Channels to mask.")
        return

    if not st.session_state.preview_cache:
        st.info("Load a file in the sidebar first.")
        return

    st.caption(
        "Tune each channel independently. Masks update live at preview resolution. "
        "Full resolution is computed only at export."
    )

    # Rebuild fresh every run so this dict's keys exactly match use_channels —
    # no stale entries from a previously-deselected channel can linger and
    # corrupt downstream partition math (see Tab 2). Cheap thanks to
    # @st.cache_data on generate_mask.
    st.session_state.base_masks_preview = {}

    for ch in use_channels:
        _render_channel_panel(ch)
        st.divider()


# ── Per-channel panel ─────────────────────────────────────────────────────────

def _render_channel_panel(ch: int) -> None:
    img = st.session_state.preview_cache.get(ch)
    if img is None:
        return

    # Init params for this channel
    if ch not in st.session_state.channel_params:
        st.session_state.channel_params[ch] = {
            "mode": "Blob",
            "color": DEFAULT_CHANNEL_COLORS[ch % len(DEFAULT_CHANNEL_COLORS)],
            **PIPELINE_MODES["Blob"]["params"],
        }
    cp = st.session_state.channel_params[ch]

    with st.expander(f"**Channel {ch}**", expanded=True):
        col_left, col_right = st.columns([1, 2], gap="medium")

        # ── Left: all controls ────────────────────────────────────────────────
        with col_left:
            # Mode + color on same row
            mc1, mc2 = st.columns([3, 1])
            with mc1:
                mode = st.selectbox(
                    "Pipeline mode",
                    options=PIPELINE_MODE_NAMES,
                    index=PIPELINE_MODE_NAMES.index(cp.get("mode", "Blob")),
                    key=f"mode_{ch}",
                    help="\n\n".join(
                        f"**{k}** — {v['description']}" for k, v in PIPELINE_MODES.items()
                    ),
                )
            with mc2:
                color = st.color_picker(
                    "Color",
                    value=cp.get("color", DEFAULT_CHANNEL_COLORS[ch % len(DEFAULT_CHANNEL_COLORS)]),
                    key=f"color_{ch}",
                )

            # If mode changed, load defaults (preserve color)
            if mode != cp.get("mode"):
                new_p = PIPELINE_MODES[mode]["params"].copy()
                new_p["mode"] = mode
                new_p["color"] = color
                st.session_state.channel_params[ch] = new_p
                cp = st.session_state.channel_params[ch]
                st.rerun()

            cp["mode"] = mode
            cp["color"] = color

            # ── Preprocessing ─────────────────────────────────────────────────
            with st.expander("Preprocessing", expanded=False):
                pc1, pc2 = st.columns(2)
                with pc1:
                    cp["clip_lo"] = st.number_input(
                        "Clip low %", 0, 10,
                        int(cp.get("clip_lo", 1)), 1,
                        key=f"clip_lo_{ch}",
                    )
                with pc2:
                    cp["clip_hi"] = st.number_input(
                        "Clip high %", 90, 100,
                        int(cp.get("clip_hi", 99)), 1,
                        key=f"clip_hi_{ch}",
                    )
                if mode != "Model-based":
                    cp["use_clahe"] = st.toggle(
                        "CLAHE enhancement",
                        value=bool(cp.get("use_clahe", False)),
                        key=f"clahe_{ch}",
                        help="Contrast Limited Adaptive Histogram Equalization. Helps with uneven illumination.",
                    )
                    if cp["use_clahe"]:
                        cp["clahe_clip"] = st.slider(
                            "CLAHE clip limit", 1.0, 8.0,
                            float(cp.get("clahe_clip", 2.0)), 0.5,
                            key=f"clahe_clip_{ch}",
                        )

            # ── Thresholding ──────────────────────────────────────────────────
            if mode != "Model-based":
                with st.expander("Thresholding", expanded=True):
                    cp["threshold_method"] = st.radio(
                        "Method",
                        ["manual", "otsu"],
                        index=0 if cp.get("threshold_method", "manual") == "manual" else 1,
                        horizontal=True,
                        key=f"tmeth_{ch}",
                        help="Manual: you set the cutoff. Otsu: computed automatically from histogram.",
                    )
                    if cp["threshold_method"] == "manual":
                        cp["threshold"] = st.slider(
                            "Threshold", 0, 255,
                            int(cp.get("threshold", 30)),
                            key=f"thresh_{ch}",
                        )

            # ── Morphology ────────────────────────────────────────────────────
            with st.expander("Morphology & Shape", expanded=False):
                cp["open_ksize"] = st.slider(
                    "Smooth kernel (opening)", 1, 21,
                    int(cp.get("open_ksize", 3)), 2,
                    key=f"open_{ch}",
                    help="Removes small noise blobs and smooths mask edges. 1 = off.",
                )
                cp["dilation_px"] = st.slider(
                    "Expand mask (px)", 0, 40,
                    int(cp.get("dilation_px", 0)),
                    key=f"dil_{ch}",
                    help="Isotropic distance-based expansion. Use for cytoplasm gap-filling.",
                )
                cp["fill_holes"] = st.toggle(
                    "Fill enclosed holes",
                    value=bool(cp.get("fill_holes", True)),
                    key=f"fill_{ch}",
                )
                if cp["fill_holes"]:
                    cp["max_hole_size"] = st.slider(
                        "Max hole size (px², 0 = all)", 0, 10_000,
                        int(cp.get("max_hole_size", 0)), 50,
                        key=f"hole_{ch}",
                        help="Only fill holes smaller than this. 0 fills all enclosed holes.",
                    )

                if mode in ("Blob",):
                    cp["watershed"] = st.toggle(
                        "Watershed separation",
                        value=bool(cp.get("watershed", False)),
                        key=f"ws_{ch}",
                        help="Splits touching nuclei/blobs. Requires scikit-image.",
                    )
                    if cp.get("watershed"):
                        cp["watershed_min_dist"] = st.slider(
                            "Min cell distance (px)", 3, 60,
                            int(cp.get("watershed_min_dist", 12)),
                            key=f"wsd_{ch}",
                        )

            # ── Frangi (thin structures) ───────────────────────────────────────
            if mode == "Thin Structures":
                with st.expander("Frangi filter", expanded=True):
                    cp["use_frangi"] = st.toggle(
                        "Enable Frangi vessel filter",
                        value=bool(cp.get("use_frangi", True)),
                        key=f"frangi_{ch}",
                        help="Enhances thin filamentous structures before thresholding.",
                    )
                    if cp.get("use_frangi"):
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            cp["frangi_lo"] = st.number_input(
                                "Scale min (σ)", 0.5, 10.0,
                                float(cp.get("frangi_lo", 1.0)), 0.5,
                                key=f"flo_{ch}",
                            )
                        with fc2:
                            cp["frangi_hi"] = st.number_input(
                                "Scale max (σ)", 1.0, 20.0,
                                float(cp.get("frangi_hi", 5.0)), 0.5,
                                key=f"fhi_{ch}",
                            )

            # ── AI Model (Model-based mode) ─────────────────────────────────────
            if mode == "Model-based":
                with st.expander("AI Model", expanded=True):
                    _render_model_addon_controls(cp, ch)

            # ── Area filtering ────────────────────────────────────────────────
            with st.expander("Area filtering", expanded=False):
                ac1, ac2 = st.columns(2)
                with ac1:
                    cp["min_area"] = st.number_input(
                        "Min area (px²)", 0, 50_000,
                        int(cp.get("min_area", 100)), 50,
                        key=f"mina_{ch}",
                    )
                with ac2:
                    cp["max_area"] = st.number_input(
                        "Max area (px², 0 = ∞)", 0, 1_000_000,
                        int(cp.get("max_area", 0)), 500,
                        key=f"maxa_{ch}",
                        help="0 disables the upper bound.",
                    )

        # ── Right: histogram + overlay ────────────────────────────────────────
        with col_right:
            # Build params for mask generation (strip non-pipeline keys)
            gen_params = {k: v for k, v in cp.items() if k not in ("mode", "color")}
            try:
                mask = generate_mask(img, mode, gen_params)
            except ModelAddonError as e:
                st.warning(f"⚠️ {e}")
                mask = np.zeros_like(img, dtype=np.uint8)

            # Store for Tab 2
            st.session_state.base_masks_preview[ch] = mask

            # Histogram (above the preview image)
            thresh_line = int(cp["threshold"]) if cp.get("threshold_method") == "manual" else None
            fig = histogram_figure(img, threshold=thresh_line, color=cp.get("color", "#4488FF"))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

            # Brightness / opacity controls (compact, inline above image)
            vc1, vc2 = st.columns(2)
            with vc1:
                brightness = st.slider(
                    "Brightness", 0.1, 8.0, 1.0, 0.1,
                    key=f"bright_{ch}",
                )
            with vc2:
                opacity = st.slider(
                    "Mask opacity", 0.0, 1.0, 0.4, 0.05,
                    key=f"opac_{ch}",
                )

            # Overlay image
            overlay = channel_overlay(img, mask, cp.get("color", "#FF4444"), brightness, opacity)
            masked_px = int(np.sum(mask > 0))
            total_px = mask.size
            pct = 100 * masked_px / total_px if total_px > 0 else 0
            st.image(
                overlay,
                width='stretch',
                caption=f"Masked: {masked_px:,} px ({pct:.1f}%)",
            )

        # Write params back to session state
        st.session_state.channel_params[ch] = cp


# ── AI Model controls (Model-based mode) ───────────────────────────────────

def _render_model_addon_controls(cp: dict, ch: int) -> None:
    """
    Render the addon/model/confidence/diameter/GPU controls for
    "Model-based" mode, or an install-hint message if no addon's
    dependencies are currently importable.

    Mutates `cp` in place (mirrors the pattern used by the rest of this
    module — params are written back to session_state by the caller).
    """
    avail = available_addons()

    if not avail:
        st.info(
            "No model addons are installed yet. 'Model-based' mode will "
            "produce an empty mask until one is available.\n\n"
            "Install one of the following, then reload the app:"
        )
        for addon_cls in all_addons().values():
            st.markdown(
                f"- **{addon_cls.display_name}** — {addon_cls.description}\n"
                f"  \n  `{addon_cls.install_hint}`"
            )
        # Keep params consistent even with nothing selectable.
        cp["model_addon"] = ""
        cp["model_name"] = ""
        return

    addon_keys = list(avail.keys())
    current_addon = cp.get("model_addon") or addon_keys[0]
    if current_addon not in addon_keys:
        current_addon = addon_keys[0]

    chosen_addon_key = st.selectbox(
        "Addon",
        options=addon_keys,
        index=addon_keys.index(current_addon),
        format_func=lambda k: avail[k].display_name,
        key=f"model_addon_{ch}",
        help="\n\n".join(f"**{a.display_name}** — {a.description}" for a in avail.values()),
    )
    cp["model_addon"] = chosen_addon_key
    addon_cls = avail[chosen_addon_key]

    model_opts = addon_cls.model_options()
    current_model = cp.get("model_name") or model_opts[0]
    if current_model not in model_opts:
        current_model = model_opts[0]

    cp["model_name"] = st.selectbox(
        "Pretrained model",
        options=model_opts,
        index=model_opts.index(current_model),
        key=f"model_name_{ch}",
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        cp["confidence"] = st.slider(
            "Confidence / probability threshold", 0.0, 1.0,
            float(cp.get("confidence", 0.5)), 0.05,
            key=f"model_conf_{ch}",
            help=(
                "Higher = stricter, fewer detections. For Cellpose this maps "
                "to flow_threshold; for StarDist, to prob_thresh."
            ),
        )
    with mc2:
        cp["diameter"] = st.number_input(
            "Approx. object diameter (px, 0 = auto)", 0.0, 500.0,
            float(cp.get("diameter", 30.0)), 1.0,
            key=f"model_diam_{ch}",
            help="Used by Cellpose to set its internal scaling. Ignored by StarDist.",
        )

    cp["use_gpu"] = st.toggle(
        "Use GPU (if the addon's backend supports it)",
        value=bool(cp.get("use_gpu", False)),
        key=f"model_gpu_{ch}",
        help="Independent of the CPU/GPU toggle above — controlled by the addon's own backend (PyTorch/TensorFlow).",
    )

    st.caption(
        "Model output is post-processed with the Morphology and Area "
        "filtering settings below, same as the other modes."
    )
