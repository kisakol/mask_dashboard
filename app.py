"""
app.py — GeoMx Mask Dashboard v5
---------------------------------
Entry point. Run with:
    streamlit run app.py

Architecture:
  Sidebar   → file loading, channel selection, mask naming, presets
  Tab 1     → per-channel tuning (histogram + params + overlay)
  Tab 2     → logic & priority (partition assignment, conflict rules, composite preview)
  Tab 3     → export (filename template, format, single-file & batch save)
"""

import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="GeoMx Mask Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Push the entire main content area below the Streamlit toolbar ── */
/* The toolbar (Deploy / ⋮) is ~3rem tall and fixed; without this the  */
/* tab bar sits behind it and is nearly invisible.                      */
section[data-testid="stMain"] > div:first-child {
    padding-top: 3.5rem !important;
}

/* Tighter bottom padding */
.block-container { padding-bottom: 1rem; }

/* Expander headers slightly bolder */
.streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; }

/* Compact number inputs */
input[type=number] { padding: 2px 6px !important; }

/* Reduce gap between stacked elements */
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.1rem; }

/* Tab bar — make labels a touch larger and bolder */
button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "data_cache": {},
    "preview_cache": {},
    "current_file": None,
    "file_stem": "output",
    "preview_quality": "Low (fast)",
    "folder_path": "",
    "use_channels": [],
    "channel_params": {},
    "mask_slots": ["Nucleus", "Cytoplasm", "Other"],
    "num_masks": 3,
    "partitions_map": {},
    "priority_order": [0, 1, 2],
    "conflict_strategies": {},
    "base_masks_preview": {},
    "resolved_masks_preview": {},
    "filename_template": "{stem}_{mask_name}",
    "export_format": "PNG",
    "presets": {},
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Import components (after set_page_config) ─────────────────────────────────
from components.sidebar import render_sidebar
from components.channel_tuning import render_channel_tuning_tab
from components.logic_panel import render_logic_panel_tab
from components.export_panel import render_export_panel_tab

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar()

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.data_cache:
    st.markdown("# 🔬 GeoMx Mask Dashboard")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **① Load**
        - Paste folder path in sidebar
        - Select your GeoMx TIFF or CZI
        - Choose preview quality
        """)
    with c2:
        st.markdown("""
        **② Tune**
        - Pick pipeline mode per channel
        - Adjust threshold with live histogram
        - Preview masks in real time
        """)
    with c3:
        st.markdown("""
        **③ Export**
        - Set priority & conflict rules
        - Preview the non-overlapping composite
        - Save PNG/TIFF back to GeoMx
        """)
    st.stop()

if not st.session_state.use_channels:
    st.warning("⬅ Select at least one channel in the sidebar.")
    st.stop()

# ── Main tab layout ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Channel Tuning",
    "⚖️  Logic & Priority",
    "💾  Export",
])

with tab1:
    render_channel_tuning_tab()

with tab2:
    render_logic_panel_tab()

with tab3:
    render_export_panel_tab()
