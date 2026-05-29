"""
utils/logic.py
--------------
Partition algebra for non-overlapping mask generation.

Key concepts:
  - Partitions: every exclusive boolean combination of N channels
    e.g. for channels [0, 1]:  (0,) = in 0 only,  (1,) = in 1 only,  (0,1) = in both
  - Slot: a named output mask (e.g. "Nucleus", "Cytoplasm")
  - Assignment: user maps each partition → a slot (or Unassigned)
  - Conflict: when two slots overlap after assignment, resolved per user rule
  - Priority: final tie-breaker ordering
"""

from itertools import combinations

import cv2
import numpy as np


# ── Partition computation ─────────────────────────────────────────────────────

def get_partitions(channels: list[int]) -> list[tuple]:
    """
    Return all non-empty exclusive subsets of `channels`.
    E.g. [0, 1, 2] →  (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)
    Order: singletons first, then pairs, then triples.
    """
    parts = []
    for r in range(1, len(channels) + 1):
        for combo in combinations(channels, r):
            parts.append(tuple(sorted(combo)))
    return parts


def calculate_partition_masks(
    partitions: list[tuple],
    channel_masks: dict,
    height: int,
    width: int,
) -> dict:
    """
    For each partition, compute the pixel mask where exactly those channels
    are positive and all others are negative.

    Parameters
    ----------
    partitions    : output of get_partitions()
    channel_masks : {ch_idx: uint8 mask (0/255)}
    height, width : spatial dimensions

    Returns
    -------
    {partition_tuple: uint8 mask (0/255)}
    """
    all_ch = set(channel_masks.keys())
    result = {}
    for part in partitions:
        acc = np.ones((height, width), dtype=bool)
        for ch in part:
            acc &= channel_masks[ch] > 0
        for ch in all_ch - set(part):
            acc &= channel_masks[ch] == 0
        result[part] = acc.astype(np.uint8) * 255
    return result


# ── Slot aggregation ──────────────────────────────────────────────────────────

def aggregate_to_slots(
    partition_masks: dict,
    assignment: dict,
    num_slots: int,
    height: int,
    width: int,
) -> dict:
    """
    Merge partition masks into slot masks according to the user's assignment.

    Parameters
    ----------
    assignment : {partition_tuple: slot_idx}  (-1 = unassigned)

    Returns
    -------
    {slot_idx: uint8 mask}
    """
    slots = {i: np.zeros((height, width), dtype=np.uint8) for i in range(num_slots)}
    for part, mask in partition_masks.items():
        idx = assignment.get(part, -1)
        if 0 <= idx < num_slots:
            slots[idx] = cv2.bitwise_or(slots[idx], mask)
    return slots


# ── Conflict resolution ───────────────────────────────────────────────────────

def resolve_conflicts(
    slot_masks: dict,
    priority_order: list[int],
    conflict_strategies: dict,
) -> dict:
    """
    Resolve pixel-level overlaps between mask slots.

    Processing order
    ----------------
    1. Apply per-pair explicit strategies (Give to First/Second, Exclude Both, Keep Both).
       These operate on the current state of each mask — applied before priority.
    2. Apply priority ordering for any remaining overlaps:
       higher-priority slots claim contested pixels; lower-priority masks yield.

    Parameters
    ----------
    slot_masks         : {slot_idx: uint8 mask}
    priority_order     : [slot_idx, ...] — index 0 = highest priority
    conflict_strategies: {(i, j): strategy_str}  i < j always

    Returns
    -------
    {slot_idx: resolved uint8 mask}
    """
    resolved = {idx: m.copy() for idx, m in slot_masks.items()}
    slot_indices = sorted(resolved.keys())

    # ── Step 1: per-pair explicit strategies ─────────────────────────────────
    for i, j in combinations(slot_indices, 2):
        strategy = (
            conflict_strategies.get((i, j))
            or conflict_strategies.get((j, i))
            or "Priority Order"
        )
        if strategy == "Priority Order":
            continue  # Handled in step 2

        overlap = cv2.bitwise_and(resolved[i], resolved[j])
        if not np.any(overlap):
            continue

        if strategy == "Give to First":
            # j yields to i
            resolved[j] = cv2.bitwise_and(resolved[j], cv2.bitwise_not(overlap))
        elif strategy == "Give to Second":
            # i yields to j
            resolved[i] = cv2.bitwise_and(resolved[i], cv2.bitwise_not(overlap))
        elif strategy == "Exclude Both":
            resolved[i] = cv2.bitwise_and(resolved[i], cv2.bitwise_not(overlap))
            resolved[j] = cv2.bitwise_and(resolved[j], cv2.bitwise_not(overlap))
        # "Keep Both" → do nothing; pixels stay in both

    # ── Step 2: priority waterfall for any remaining overlaps ─────────────────
    occupied = np.zeros_like(list(resolved.values())[0], dtype=np.uint8)
    for idx in priority_order:
        if idx not in resolved:
            continue
        mask = resolved[idx]
        clean = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))
        resolved[idx] = clean
        occupied = cv2.bitwise_or(occupied, clean)

    return resolved


def default_priority_order(num_slots: int) -> list[int]:
    """Return [0, 1, 2, ...] as the default priority ordering."""
    return list(range(num_slots))
