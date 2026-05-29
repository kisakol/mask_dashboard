"""
utils/render.py
---------------
Pure visualization helpers — no Streamlit calls, no side effects.
All functions return numpy arrays or Plotly figures.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import plotly.graph_objects as go


# ── Colour helpers ────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#FF4444' → (255, 68, 68)"""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Histogram ─────────────────────────────────────────────────────────────────

def histogram_figure(
    img: np.ndarray,
    threshold: int | None = None,
    color: str = "#4488FF",
) -> go.Figure:
    """
    Plotly bar histogram of pixel intensities with an optional threshold vline.
    Designed to be compact (height=110px) — sits above the threshold slider.
    """
    hist, bins = np.histogram(img.flatten(), bins=128, range=(0, 255))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bins[:-1],
        y=hist,
        marker_color=color,
        marker_line_width=0,
        name="intensity",
    ))

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_width=2,
            line_dash="dot",
            line_color="#FF4444",
            annotation=dict(
                text=f"T={threshold}",
                font_color="#FF4444",
                showarrow=False,
                yref="paper", y=1.0,
                xanchor="left",
            ),
        )

    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.0,
        xaxis=dict(
            showgrid=False,
            range=[0, 255],
            tickfont=dict(size=9, color="#888"),
            title=None,
        ),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


# ── Channel overlay ───────────────────────────────────────────────────────────

def channel_overlay(
    img_8bit: np.ndarray,
    mask: np.ndarray | None,
    color_hex: str,
    brightness: float = 1.0,
    opacity: float = 0.4,
) -> np.ndarray:
    """
    Blend a grayscale channel image with a coloured mask overlay.

    Returns
    -------
    uint8 RGB numpy array
    """
    # Grayscale → RGB
    rgb = np.stack([img_8bit] * 3, axis=-1).astype(np.uint8)

    # Brightness via PIL for smooth scaling beyond 1.0
    if brightness != 1.0:
        pil = Image.fromarray(rgb)
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
        rgb = np.array(pil, dtype=np.uint8)

    if mask is None or not np.any(mask):
        return rgb

    rgb_color = hex_to_rgb(color_hex)
    overlay = rgb.copy()

    # Semi-transparent fill
    colored = np.zeros_like(overlay)
    colored[mask > 0] = rgb_color
    overlay = cv2.addWeighted(overlay, 1.0, colored, opacity, 0)

    # Crisp contour outline (1-px)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, rgb_color, 1)

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── Composite (logic) preview ─────────────────────────────────────────────────

def composite_preview(
    resolved_masks: dict,
    slot_colors: list[str],
    height: int,
    width: int,
    bg_img: np.ndarray | None = None,
    bg_opacity: float = 0.3,
) -> np.ndarray:
    """
    Build an RGB composite image showing all resolved (non-overlapping) masks.

    Parameters
    ----------
    resolved_masks : {slot_idx: uint8 mask}
    slot_colors    : hex color per slot index
    bg_img         : optional grayscale uint8 background channel
    bg_opacity     : blend weight for background channel

    Returns
    -------
    uint8 RGB array (H, W, 3)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Optional background
    if bg_img is not None:
        bg_resized = cv2.resize(bg_img, (width, height), interpolation=cv2.INTER_AREA)
        bg_rgb = np.stack([bg_resized] * 3, axis=-1).astype(np.float32)
        canvas = (bg_rgb * bg_opacity).astype(np.uint8)

    for slot_idx, mask in sorted(resolved_masks.items()):
        if slot_idx >= len(slot_colors):
            continue
        color = hex_to_rgb(slot_colors[slot_idx])
        # Draw fill
        canvas[mask > 0] = color
        # Draw contours on top
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1)

    return canvas
