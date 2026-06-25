"""
components/logic_panel.py
--------------------------
Tab 2 — Logic & Priority.

Three sections:
  A. Partition assignment — maps each Venn partition to a mask slot
  B. Priority & conflict rules — ordering + per-pair resolution strategy
  C. Live composite preview — non-overlapping result, always current
"""

from itertools import combinations

import numpy as np
import streamlit as st

from config.defaults import CONFLICT_STRATEGIES, DEFAULT_SLOT_COLORS
from utils.logic import (
    aggregate_to_slots,
    calculate_partition_masks,
    get_partitions,
    resolve_conflicts,
)
from utils.render import composite_preview


def render_logic_panel_tab() -> None:
    use_channels: list[int] = st.session_state.get("use_channels", [])
    base_masks: dict = st.session_state.get("base_masks_preview", {})
    mask_slots: list[str] = st.session_state.get("mask_slots", [])
    num_masks: int = st.session_state.get("num_masks", 3)

    if not use_channels or not base_masks:
        st.info("Tune channels in **Channel Tuning** first.")
        return

    # Make sure we have masks for all selected channels
    missing = [ch for ch in use_channels if ch not in base_masks]
    if missing:
        st.warning(f"Missing masks for channel(s): {missing}. Visit Channel Tuning tab first.")
        return

    # Compute spatial dimensions from preview masks
    first_mask = base_masks[use_channels[0]]
    h, w = first_mask.shape

    # ── Compute partitions ────────────────────────────────────────────────────
    partitions = get_partitions(use_channels)
    partition_masks = calculate_partition_masks(partitions, base_masks, h, w, channels=use_channels)

    st.caption(
        "Assign each Venn partition to an output mask slot, then set priority & conflict "
        "rules. The composite preview updates instantly."
    )

    top_left, top_right = st.columns([1, 1], gap="large")

    # ── A. Partition assignment ────────────────────────────────────────────────
    with top_left:
        st.markdown("#### A. Partition → Slot assignment")
        st.caption(
            "Each row is an exclusive region. Assign it to a mask slot or leave unassigned "
            "(those pixels are discarded)."
        )

        pmap: dict = st.session_state.get("partitions_map", {})
        slot_options = ["— Unassigned —"] + mask_slots[:num_masks]

        for part in partitions:
            in_ch = " ∩ ".join([f"Ch{c}" for c in part])
            out_ch = [c for c in use_channels if c not in part]
            excl = " ∉ " + ", ".join([f"Ch{c}" for c in out_ch]) if out_ch else " (all channels)"
            label = f"**{in_ch}**{excl}"

            # Pixel count for this partition
            pmask = partition_masks[part]
            n_px = int(np.sum(pmask > 0))

            current_slot = pmap.get(part, -1)
            current_idx = current_slot + 1 if current_slot >= 0 else 0  # offset for "Unassigned"

            pc1, pc2 = st.columns([3, 2])
            with pc1:
                st.markdown(f"{label}  \n<sub>{n_px:,} px</sub>", unsafe_allow_html=True)
            with pc2:
                chosen = st.selectbox(
                    "→",
                    options=slot_options,
                    index=min(current_idx, len(slot_options) - 1),
                    key=f"assign_{part}",
                    label_visibility="collapsed",
                )
                new_idx = slot_options.index(chosen) - 1  # -1 = unassigned
                pmap[part] = new_idx

        st.session_state.partitions_map = pmap

    # ── B. Priority & conflict rules ──────────────────────────────────────────
    with top_right:
        st.markdown("#### B. Priority & conflict rules")

        # Priority order (reorderable list)
        st.markdown("**Priority order** (drag slots up/down — top = wins overlap):")
        priority: list[int] = st.session_state.get("priority_order", list(range(num_masks)))

        # Ensure priority_order is current size
        valid = [i for i in priority if i < num_masks]
        missing_idx = [i for i in range(num_masks) if i not in valid]
        priority = valid + missing_idx
        st.session_state.priority_order = priority

        for pos, slot_idx in enumerate(priority):
            slot_label = mask_slots[slot_idx] if slot_idx < len(mask_slots) else f"Slot {slot_idx}"
            row = st.columns([3, 1, 1])
            with row[0]:
                badge_color = DEFAULT_SLOT_COLORS[slot_idx % len(DEFAULT_SLOT_COLORS)]
                st.markdown(
                    f'<span style="background:{badge_color};color:#111;'
                    f'padding:2px 8px;border-radius:4px;font-weight:600">'
                    f"#{pos + 1} {slot_label}</span>",
                    unsafe_allow_html=True,
                )
            with row[1]:
                if pos > 0 and st.button("↑", key=f"up_{pos}", use_container_width=True):
                    priority[pos], priority[pos - 1] = priority[pos - 1], priority[pos]
                    st.session_state.priority_order = priority
                    st.rerun()
            with row[2]:
                if pos < len(priority) - 1 and st.button("↓", key=f"dn_{pos}", use_container_width=True):
                    priority[pos], priority[pos + 1] = priority[pos + 1], priority[pos]
                    st.session_state.priority_order = priority
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-pair conflict strategies
        if num_masks >= 2:
            st.markdown("**Per-pair overlap rules:**")
            st.caption(
                "Applied before the priority waterfall. "
                "'Priority Order' defers to the list above."
            )

            strategies: dict = st.session_state.get("conflict_strategies", {})
            slot_labels = mask_slots[:num_masks]

            for i, j in combinations(range(num_masks), 2):
                name_i = slot_labels[i] if i < len(slot_labels) else f"Slot {i}"
                name_j = slot_labels[j] if j < len(slot_labels) else f"Slot {j}"

                current = strategies.get((i, j), "Priority Order")
                if current not in CONFLICT_STRATEGIES:
                    current = "Priority Order"

                # Remap "Give to First/Second" labels to actual slot names
                strat_labels = [
                    "Priority Order",
                    f"Give to {name_i}",
                    f"Give to {name_j}",
                    "Exclude Both",
                    "Keep Both (allow overlap)",
                ]
                # Map back to canonical keys
                strat_map = {
                    "Priority Order": "Priority Order",
                    f"Give to {name_i}": "Give to First",
                    f"Give to {name_j}": "Give to Second",
                    "Exclude Both": "Exclude Both",
                    "Keep Both (allow overlap)": "Keep Both",
                }
                current_label = {v: k for k, v in strat_map.items()}.get(current, "Priority Order")

                sc1, sc2 = st.columns([2, 3])
                with sc1:
                    st.markdown(f"<sub>{name_i} vs {name_j}</sub>", unsafe_allow_html=True)
                with sc2:
                    chosen_label = st.selectbox(
                        "rule",
                        options=strat_labels,
                        index=strat_labels.index(current_label),
                        key=f"strat_{i}_{j}",
                        label_visibility="collapsed",
                    )
                    strategies[(i, j)] = strat_map[chosen_label]

            st.session_state.conflict_strategies = strategies

    st.divider()

    # ── C. Live composite preview ─────────────────────────────────────────────
    st.markdown("#### C. Live Composite Preview (non-overlapping)")

    slot_masks = aggregate_to_slots(partition_masks, pmap, num_masks, h, w)
    resolved = resolve_conflicts(
        slot_masks,
        st.session_state.priority_order,
        st.session_state.get("conflict_strategies", {}),
    )

    # Store for export tab
    st.session_state.resolved_masks_preview = resolved

    # Build slot colors from channel_params or defaults
    slot_colors = _get_slot_colors(mask_slots, num_masks)

    # Background channel selector — any currently-selected channel, or none
    bg_options = ["None"] + [f"Channel {ch}" for ch in use_channels]
    if st.session_state.get("bg_channel_choice") not in bg_options:
        # Reset BEFORE widget instantiation to avoid a stale value that is no
        # longer a valid option (e.g. after deselecting a channel).
        st.session_state.bg_channel_choice = bg_options[1] if len(bg_options) > 1 else "None"

    bgc1, bgc2 = st.columns([2, 1])
    with bgc1:
        bg_choice = st.selectbox(
            "Background channel",
            options=bg_options,
            key="bg_channel_choice",
        )
    with bgc2:
        bg_opacity = st.slider(
            "Background opacity", 0.0, 1.0,
            st.session_state.get("bg_opacity", 0.25), 0.05,
            key="bg_opacity",
        )

    bg_img = None
    if bg_choice != "None" and st.session_state.preview_cache:
        bg_ch = int(bg_choice.replace("Channel ", ""))
        bg_img = st.session_state.preview_cache.get(bg_ch)

    comp = composite_preview(
        resolved,
        slot_colors,
        h, w,
        bg_img=bg_img,
        bg_opacity=bg_opacity,
    )

    # Legend
    legend_cols = st.columns(min(num_masks, 5))
    for i, col in enumerate(legend_cols):
        if i < num_masks:
            slot_label = mask_slots[i] if i < len(mask_slots) else f"Slot {i}"
            n_px = int(np.sum(resolved.get(i, np.zeros((h, w))) > 0))
            pct = 100 * n_px / (h * w) if (h * w) > 0 else 0
            with col:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<div style="width:14px;height:14px;border-radius:3px;'
                    f'background:{slot_colors[i]};flex-shrink:0"></div>'
                    f'<span style="font-size:0.85rem"><b>{slot_label}</b><br>'
                    f'<sub>{n_px:,} px ({pct:.1f}%)</sub></span></div>',
                    unsafe_allow_html=True,
                )

    st.image(comp, width='stretch', caption="Final resolved masks — what will be saved.")


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_slot_colors(mask_slots: list[str], num_masks: int) -> list[str]:
    """Build a per-slot color list from channel_params or fall back to defaults."""
    colors = []
    for i in range(num_masks):
        # Try to find a channel that is assigned to this slot and use its color
        for ch, cp in st.session_state.channel_params.items():
            # Heuristic: slot i → channel i color (no formal mapping here)
            pass
        colors.append(DEFAULT_SLOT_COLORS[i % len(DEFAULT_SLOT_COLORS)])
    return colors
