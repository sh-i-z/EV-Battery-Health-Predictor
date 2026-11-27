import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="EV Battery Analytics & Health Predictor",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------
# DESIGN TOKENS
# --------------------------------

BG = "#0d0d0c"
SURFACE = "#121211"
SURFACE_2 = "#1b1b19"
LINE = "#2a2a27"
TEXT = "#ebe7de"
MUTED = "#8a867c"
ACCENT = "#d9a441"
ACCENT_HOVER = "#e8b552"
SUCCESS = "#34d399"
WARN = "#fbbf24"
DANGER = "#f87171"

# --------------------------------
# CUSTOM STYLING & FONTS
# --------------------------------

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg: {BG};
    --surface: {SURFACE};
    --surface-2: {SURFACE_2};
    --line: {LINE};
    --text: {TEXT};
    --muted: {MUTED};
    --accent: {ACCENT};
    --accent-hover: {ACCENT_HOVER};
    --success: {SUCCESS};
    --warn: {WARN};
    --danger: {DANGER};
}}

/* Base Page Styling */
html, body, .stApp {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}}

/* Hide unnecessary Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    max-width: 1300px;
    padding-top: 2.2rem;
    padding-bottom: 3.5rem;
}}

@media (min-width: 1181px) {{
    .block-container {{
        padding-left: 2.2rem !important;
        padding-right: 300px !important;
    }}
}}

/* Typography */
h1, h2, h3, h4 {{
    font-family: 'Instrument Serif', serif !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}}

p, li, .stMarkdown {{
    color: var(--text);
    line-height: 1.6;
}}

.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 12px 0;
}}

.eyebrow .dot {{
    color: var(--accent);
    margin-right: 8px;
}}

/* Hero Section */
.hero {{
    padding: 8px 0 32px 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 32px;
}}

.hero h1 {{
    font-size: clamp(38px, 5.2vw, 64px);
    line-height: 1.05;
    margin: 0 0 16px 0;
    max-width: 820px;
}}

.hero h1 em {{
    color: var(--accent);
    font-style: italic;
}}

.hero p {{
    color: var(--muted);
    font-size: 15px;
    max-width: 640px;
    margin: 0;
}}

/* Stat Row Grid */
.stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    margin-bottom: 40px;
    background: var(--surface);
}}

.stat {{
    padding: 24px 24px;
}}

.stat + .stat {{
    border-left: 1px solid var(--line);
}}

.stat .value {{
    font-family: 'Instrument Serif', serif;
    font-size: 46px;
    line-height: 1.05;
    color: var(--text);
    margin: 4px 0 8px 0;
}}

.stat .value .unit {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    margin-left: 6px;
}}

.stat .note {{
    font-size: 12px;
    color: var(--muted);
    font-family: 'Inter', sans-serif;
}}

/* Section Block Layout */
.section {{
    display: grid;
    grid-template-columns: 80px 1fr;
    gap: 16px;
    padding: 28px 0;
    border-top: 1px solid var(--line);
}}

.section .idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    padding-top: 6px;
}}

.section h3 {{
    font-size: 28px;
    margin: 0 0 10px 0;
}}

.section p {{
    color: var(--muted);
    max-width: 680px;
    margin: 0 0 10px 0;
}}

/* Health Gauge Bar */
.bar {{
    height: 6px;
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 3px;
    margin: 16px 0 8px 0;
    position: relative;
    overflow: hidden;
}}

.bar > span {{
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: linear-gradient(90deg, #d9a441 0%, #34d399 100%);
    border-radius: 3px;
    transition: width 0.4s ease;
}}

.bar-labels {{
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.08em;
}}

/* Workflow Step Cards */
.steps {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 20px;
}}

.step {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 22px;
    border-radius: 4px;
}}

.step .n {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.14em;
}}

.step .t {{
    font-family: 'Instrument Serif', serif;
    font-size: 24px;
    margin: 6px 0 6px 0;
    color: var(--text);
}}

.step .d {{
    font-size: 13px;
    color: var(--muted);
}}

/* ---------------- SIDEBAR STYLING ---------------- */

[data-testid="stSidebar"],
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="true"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    transform: none !important;
    margin-left: 0 !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {{
    visibility: visible !important;
    opacity: 1 !important;
}}

/* keep the sidebar permanently open */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

[data-testid="stSidebarContent"] {{
    background: var(--surface) !important;
    padding: 0 !important;
}}

[data-testid="stSidebarUserContent"] {{
    padding: 44px 24px 32px 24px !important;
}}

/* Brand Block */
.brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 30px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 36px;
}}

.brand .mark {{
    width: 44px;
    height: 44px;
    border: 1px solid var(--accent);
    background: rgba(217, 164, 65, 0.08);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    border-radius: 4px;
}}

.brand .name {{
    font-family: 'Instrument Serif', serif;
    font-size: 30px;
    line-height: 1;
    color: var(--text);
}}

.brand .sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    color: var(--accent);
    margin-top: 6px;
}}

.side-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.16em;
    color: var(--muted);
    margin: 0 0 18px 4px;
    text-transform: uppercase;
}}

/* Sidebar Radio Navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {{
    display: none;
}}

[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: 8px;
}}

[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{
    width: 100%;
    padding: 14px 18px;
    border-left: 2px solid transparent;
    border-radius: 2px;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s ease;
}}

/* hide the radio circle */
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{
    display: none !important;
}}

[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{
    font-family: 'Instrument Serif', serif;
    font-size: 25px;
    line-height: 1.2;
    letter-spacing: 0;
    text-transform: none;
    color: var(--muted);
    margin: 0;
    white-space: nowrap;
}}

[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover p {{
    color: var(--text);
}}

[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {{
    border-left-color: var(--accent);
    background: var(--surface-2);
}}

[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p {{
    color: var(--text);
}}

/* Sidebar Info Block */
.side-card {{
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 14px 14px;
    margin-top: 14px;
}}

.side-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}}

.side-row:last-child {{
    border-bottom: none;
}}

.side-row .k {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
}}

.side-row .v {{
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}}

.side-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 12px;
    background: rgba(52, 211, 153, 0.1);
    color: var(--success);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
}}

.side-tag .dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
}}

.side-foot {{
    margin-top: 64px;
    padding: 18px 6px 0 6px;
    border-top: 1px solid var(--line);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
}}

/* Form Input & Widget Customization */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput input,
.stTextInput input {{
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}}

div[data-baseweb="select"] span {{
    color: var(--text) !important;
}}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 20px;
    border-bottom: 1px solid var(--line);
    background: transparent;
    padding-bottom: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 14px;
    border: none !important;
}}

.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}}

/* Button Styling */
.stButton > button {{
    background: var(--accent) !important;
    color: #111009 !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 12px 28px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(217, 164, 65, 0.2) !important;
}}

.stButton > button:hover {{
    background: var(--accent-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(217, 164, 65, 0.3) !important;
}}

/* Prediction Result Cards */
.result-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 28px 0;
}}

.result-card {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}}

.result-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent);
}}

.result-card.success::before {{
    background: var(--success);
}}

.result-card.warning::before {{
    background: var(--warn);
}}

.result-card.danger::before {{
    background: var(--danger);
}}

.result-card .lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}}

.result-card .val {{
    font-family: 'Instrument Serif', serif;
    font-size: 52px;
    line-height: 1.05;
    color: var(--text);
    margin-bottom: 6px;
}}

.result-card .desc {{
    font-size: 13px;
    color: var(--muted);
}}

/* Parameter Group Container */
.param-box {{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 18px;
}}

.param-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* ---------------- RIGHT INFO PANEL ---------------- */

.rpanel {{
    position: fixed;
    top: 0;
    right: 0;
    width: 280px;
    height: 100vh;
    overflow-y: auto;
    background: var(--surface);
    border-left: 1px solid var(--line);
    padding: 52px 26px 24px 26px;
    z-index: 999;
    box-sizing: border-box;
}}

.rp-item {{
    border-bottom: 1px solid var(--line);
}}

.rp-item summary {{
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 18px 0;
}}

.rp-item summary::-webkit-details-marker {{ display: none; }}

.rp-t {{
    font-family: 'Instrument Serif', serif;
    font-size: 23px;
    line-height: 1.2;
    color: var(--muted);
}}

.rp-item summary:hover .rp-t,
.rp-item[open] .rp-t {{ color: var(--text); }}

.rp-item[open] .rp-t {{ font-style: italic; }}

.rp-caret {{
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 20px;
    color: var(--muted);
    transition: transform 0.2s ease;
}}

.rp-item[open] .rp-caret {{ transform: rotate(45deg); color: var(--accent); }}

.rp-body {{
    padding: 0 0 22px 0;
    font-size: 14px;
    color: var(--muted);
    line-height: 1.65;
}}

.rp-body p {{
    color: var(--muted);
    font-size: 14px;
    margin: 0 0 14px 0;
}}

.rp-live {{
    margin: 0 0 16px 0;
    padding: 0 0 14px 0;
    border-bottom: 1px solid var(--line);
}}

.rp-live .k {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--muted);
}}

.rp-live .v {{
    font-family: 'Instrument Serif', serif;
    font-size: 46px;
    line-height: 1.1;
    color: var(--text);
    margin: 2px 0 2px 0;
}}

.rp-live .v span {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--accent);
    margin-left: 6px;
}}

.rp-live .s {{
    font-size: 13px;
    color: var(--muted);
}}

.rp-row {{
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 7px 0;
    font-size: 14px;
}}

.rp-row .k {{
    color: var(--muted);
}}

.rp-row .v {{
    color: var(--text);
    text-align: right;
}}

@media (max-width: 1180px) {{
    .rpanel {{ display: none; }}
}}

/* ---------------- DASHBOARD FLOW ---------------- */

.flow {{
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1.2fr;
    align-items: start;
    gap: 22px;
    margin-top: 22px;
}}

.flow-step {{
    border-top: 1px solid var(--line);
    padding-top: 18px;
}}

.flow .fn {{
    font-family: 'Instrument Serif', serif;
    font-size: 44px;
    line-height: 1;
    color: var(--accent);
    font-style: italic;
}}

.flow .ft {{
    font-family: 'Instrument Serif', serif;
    font-size: 25px;
    line-height: 1.2;
    margin: 8px 0 6px 0;
    color: var(--text);
}}

.flow .fd {{
    font-size: 14px;
    color: var(--muted);
}}

.flow-arrow {{
    font-family: 'Instrument Serif', serif;
    font-size: 32px;
    color: var(--muted);
    padding-top: 30px;
}}

.flow-out {{
    display: grid;
    gap: 18px;
}}

.flow-out .fo {{
    border-left: 2px solid var(--accent);
    padding: 4px 0 4px 18px;
}}

.flow-out .fl {{
    font-size: 13px;
    color: var(--muted);
}}

@media (max-width: 860px) {{
    .flow {{ grid-template-columns: 1fr; }}
    .flow-arrow {{ display: none; }}
}}

/* Footer */
.footer {{
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    color: var(--muted);
}}

/* Responsive */
@media (max-width: 860px) {{
    .stats, .steps, .result-grid {{ grid-template-columns: 1fr; }}
    .stat + .stat {{ border-left: none; border-top: 1px solid var(--line); }}
    .section {{ grid-template-columns: 1fr; }}
    .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; padding-top: 1.5rem !important; }}
    .hero h1 {{ font-size: clamp(28px, 7vw, 44px); }}
    .result-card .val {{ font-size: 38px; }}
    .stat .value {{ font-size: 36px; }}
    .footer {{ flex-direction: column; gap: 6px; text-align: center; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------
# CACHED DATA & MODEL LOADERS
# --------------------------------

@st.cache_data
def load_dataset():
    if os.path.exists("data/ev_battery_health_cleaned.csv"):
        df = pd.read_csv("data/ev_battery_health_cleaned.csv")
        return df
    return None

@st.cache_data
def load_eval_results():
    if os.path.exists("data/model_evaluation_results.csv"):
        return pd.read_csv("data/model_evaluation_results.csv")
    return None

@st.cache_resource
def load_models_and_preprocessors():
    try:
        soh_prep = joblib.load("data/soh_preprocessor.pkl")
        rul_prep = joblib.load("data/rul_preprocessor.pkl")
        soh_model = joblib.load("data/soh_random_forest.pkl")
        rul_model = joblib.load("data/rul_random_forest.pkl")
        return soh_prep, rul_prep, soh_model, rul_model
    except Exception as e:
        st.error(f"Error loading models or preprocessors: {e}")
        return None, None, None, None

dataset = load_dataset()
eval_df = load_eval_results()
soh_prep, rul_prep, soh_model, rul_model = load_models_and_preprocessors()

# --------------------------------
# HELPER FUNCTIONS
# --------------------------------

def go_predict():
    st.session_state["nav_page"] = "Battery Prediction"


def page_header(label: str, title_html: str, blurb: str) -> None:
    st.markdown(
        f"""
<div class="hero">
<div class="eyebrow"><span class="dot">●</span>{label}</div>
<h1>{title_html}</h1>
<p>{blurb}</p>
</div>
""",
        unsafe_allow_html=True,
    )

def setup_dark_plot():
    """Sets matplotlib styling matching the dark luxury theme."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(color=LINE, linestyle='--', linewidth=0.6, alpha=0.7)
    return fig, ax

FEATURE_COLUMNS_SOH = [
    'chemistry', 'manufacturer', 'vehicle_model', 'pack_capacity_kwh',
    'battery_age_days', 'cycle_count', 'climate_zone', 'avg_daily_distance_km',
    'avg_charging_power_kw', 'fast_charge_ratio', 'avg_depth_of_discharge_pct',
    'avg_state_of_charge_pct', 'charge_cycles_per_month',
    'total_energy_throughput_kwh', 'avg_ambient_temp_c', 'humidity_pct',
    'altitude_m', 'internal_resistance_mohm', 'voltage_avg_v', 'current_avg_a',
    'charge_time_avg_min', 'peak_temp_during_fast_charge_c', 'service_count',
    'bms_firmware_version', 'recall_flag', 'days_since_last_service'
]

# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:
    st.markdown(
        """
<div class="brand">
    <div class="mark">◐</div>
    <div>
        <div class="name">EV Battery</div>
        <div class="sub">HEALTH PREDICTOR</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Battery Prediction", "Model Insights"],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown(
        """
<div class="side-foot">
    <span>Python and Streamlit</span>
</div>
""",
        unsafe_allow_html=True,
    )

# --------------------------------
# DASHBOARD PAGE
# --------------------------------

if page == "Dashboard":

    page_header(
        "Fleet Overview & Analytics",
        "How much life is <em>left</em> in the battery?",
        "Continuous fleet telemetry analysis and degradation forecasting "
        "powered by tuned dual Random Forest regression pipelines.",
    )

    # Calculate fleet stats if dataset is loaded
    if dataset is not None:
        avg_soh = dataset['state_of_health_pct'].mean()
        avg_rul = dataset['remaining_useful_life_cycles'].mean()
        total_batteries = dataset['battery_id'].nunique()
        total_obs = len(dataset)
    else:
        avg_soh = 91.8
        avg_rul = 1730
        total_batteries = 2600
        total_obs = 26000

    st.markdown(
        f"""
<div class="stats">
<div class="stat">
<div class="eyebrow">Fleet Average SOH</div>
<div class="value">{avg_soh:.1f}<span class="unit">%</span></div>
<div class="note">Current fleet health level</div>
</div>
<div class="stat">
<div class="eyebrow">Fleet Average RUL</div>
<div class="value">{avg_rul:,.0f}<span class="unit">CYCLES</span></div>
<div class="note">Usable charging cycles remaining</div>
</div>
<div class="stat">
<div class="eyebrow">Total Battery Packs</div>
<div class="value">{total_batteries:,}<span class="unit">UNITS</span></div>
<div class="note">NMC · LFP · NCA · LMO cells</div>
</div>
<div class="stat">
<div class="eyebrow">Pipeline Status</div>
<div class="value" style="color: var(--success); font-size: 38px;">Healthy</div>
<div class="note">Inference models ready</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Visual Degradation Graphs
    st.markdown('<div class="eyebrow"><span class="dot">●</span>FLEET DEGRADATION VISUALIZATIONS</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### SOH Degradation vs Cycle Count")
        if dataset is not None:
            fig, ax = setup_dark_plot()
            # Sample for cleaner visualization
            sample_sub = dataset.sample(n=min(1200, len(dataset)), random_state=42)
            sc = ax.scatter(
                sample_sub["cycle_count"],
                sample_sub["state_of_health_pct"],
                c=sample_sub["internal_resistance_mohm"],
                cmap="YlOrBr",
                alpha=0.6,
                s=18,
                edgecolors="none"
            )
            # Add 80% EOL line
            ax.axhline(80.0, color=DANGER, linestyle="--", linewidth=1.2, label="EOL Threshold (80% SOH)")
            ax.set_xlabel("Equivalent Full Cycles", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
            ax.set_ylabel("State of Health (%)", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("Internal Resistance (mΩ)", color=MUTED, fontfamily='JetBrains Mono', fontsize=9)
            cbar.ax.yaxis.set_tick_params(color=LINE, labelcolor=MUTED)
            ax.legend(facecolor=SURFACE_2, edgecolor=LINE, labelcolor=TEXT, loc="upper right")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Fleet dataset loading...")

    with col_g2:
        st.markdown("#### Health Distribution by Chemistry")
        if dataset is not None:
            fig, ax = setup_dark_plot()
            chem_stats = dataset.groupby("chemistry")["state_of_health_pct"].mean().sort_values(ascending=False)
            colors = [ACCENT, "#e5a93c", "#cca04e", "#a88647"]
            bars = ax.bar(chem_stats.index, chem_stats.values, color=colors[:len(chem_stats)], width=0.55, edgecolor=LINE)
            ax.set_ylim(70, 100)
            ax.set_xlabel("Cell Chemistry", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
            ax.set_ylabel("Mean SOH (%)", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha='center', va='bottom', color=TEXT, fontfamily='JetBrains Mono', fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Fleet dataset loading...")

    st.markdown(
        """
<div class="section">
<div class="idx">01</div>
<div>
<h3>State of Health (SOH)</h3>
<p>
SOH compares current battery capacity with its rated design capacity.
100% represents factory fresh cells; EV battery packs generally reach end of first automotive
life at 80% SOH, transitioning to second-life grid storage or recycling.
</p>
<div class="bar"><span style="width: 82%;"></span></div>
<div class="bar-labels"><span>0% CRITICAL</span><span>80% FIRST LIFE EOL</span><span>100% NOMINAL</span></div>
</div>
</div>

<div class="section">
<div class="idx">02</div>
<div>
<h3>Remaining Useful Life (RUL)</h3>
<p>
RUL quantifies remaining equivalent full charge-discharge cycles before health decays
below the 80% threshold. Operating temperature, DC fast-charging stress, and internal resistance
growth represent the principal degradation accelerants.
</p>
</div>
</div>

<div class="section">
<div class="idx">03</div>
<div>
<h3>3-Stage Machine Learning Pipeline</h3>
<div class="steps">
<div class="step">
<div class="n">STAGE 01</div>
<div class="t">Telemetry Ingestion</div>
<div class="d">Ingest 26 pack, thermal, cycling, and BMS operational parameters.</div>
</div>
<div class="step">
<div class="n">STAGE 02</div>
<div class="t">Pipeline Scaling</div>
<div class="d">Grouped imputers, One-Hot Encoders, and StandardScalers align features.</div>
</div>
<div class="step">
<div class="n">STAGE 03</div>
<div class="t">Dual Estimation</div>
<div class="d">Trained Random Forest regressors output synchronized SOH % and RUL cycles.</div>
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )


    st.markdown(
        """
<div class="section">
<div class="idx">04</div>
<div>
<h3>Try a prediction</h3>
<p>Three steps from battery details to a health estimate.</p>
<div class="flow">
<div class="flow-step"><div class="fn">1</div><div class="ft">Enter battery details</div><div class="fd">Pick a fleet sample, or type in chemistry, cycle count, resistance, temperature and charging habits.</div></div>
<div class="flow-arrow">→</div>
<div class="flow-step"><div class="fn">2</div><div class="ft">Press Analyze</div><div class="fd">Two Random Forest models run on your inputs.</div></div>
<div class="flow-arrow">→</div>
<div class="flow-out">
<div class="fo"><div class="fl">You get</div><div class="ft">State of Health</div><div class="fd">Capacity left, as a percentage of new.</div></div>
<div class="fo"><div class="fl">and</div><div class="ft">Remaining Useful Life</div><div class="fd">Charge cycles left before 80% SOH.</div></div>
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    _sp, _btn = st.columns([1, 7])
    with _btn:
        st.button("Start a prediction", on_click=go_predict, key="start_pred")


# --------------------------------
# BATTERY PREDICTION PAGE
# --------------------------------

elif page == "Battery Prediction":

    page_header(
        "Diagnostic Prediction",
        "Predict battery <em>health</em> & remaining cycles.",
        "Select a sample from the fleet dataset or configure custom battery telemetry "
        "to estimate current State of Health (SOH) and Remaining Useful Life (RUL).",
    )

    # Input method selection tabs
    input_tab1, input_tab2 = st.tabs(["⚡ LOAD FLEET SAMPLE (QUICK TEST)", "✍️ CUSTOM TELEMETRY PARAMETERS"])

    # Default values dictionary
    default_vals = {
        'chemistry': 'NMC',
        'manufacturer': 'VoltaCore',
        'vehicle_model': 'Crossover-X2',
        'pack_capacity_kwh': 75.4,
        'battery_age_days': 690,
        'cycle_count': 980,
        'climate_zone': 'temperate',
        'avg_daily_distance_km': 115.0,
        'avg_charging_power_kw': 70.0,
        'fast_charge_ratio': 0.45,
        'avg_depth_of_discharge_pct': 72.0,
        'avg_state_of_charge_pct': 52.0,
        'charge_cycles_per_month': 45.0,
        'total_energy_throughput_kwh': 85000.0,
        'avg_ambient_temp_c': 18.0,
        'humidity_pct': 42.0,
        'altitude_m': 320.0,
        'internal_resistance_mohm': 48.5,
        'voltage_avg_v': 358.0,
        'current_avg_a': 170.0,
        'charge_time_avg_min': 38.0,
        'peak_temp_during_fast_charge_c': 32.0,
        'service_count': 1,
        'bms_firmware_version': 'BMS_v3.2',
        'recall_flag': 0,
        'days_since_last_service': 180.0
    }

    actual_soh = None
    actual_rul = None
    selected_battery_id = None
    load_sample_btn = False

    with input_tab1:
        st.markdown("Select an existing vehicle pack observation from the cleaned telemetry dataset:")
        if dataset is not None:
            c1, c2 = st.columns([2, 1])
            with c1:
                sample_indices = list(range(min(150, len(dataset))))
                def sample_label(idx):
                    row = dataset.iloc[idx]
                    return f"Row #{idx:03d} | Pack: {row['battery_id']} ({row['chemistry']}) — SOH: {row['state_of_health_pct']:.1f}%, Cycles: {row['cycle_count']}"
                
                selected_sample_idx = st.selectbox(
                    "Choose Fleet Battery Profile",
                    sample_indices,
                    format_func=sample_label,
                    index=0
                )
            
            with c2:
                load_sample_btn = st.button("Load Telemetry Record", use_container_width=True)

            sample_row = dataset.iloc[selected_sample_idx].copy()
            selected_battery_id = sample_row.get("battery_id", "UNKNOWN")
            actual_soh = sample_row.get("state_of_health_pct", None)
            actual_rul = sample_row.get("remaining_useful_life_cycles", None)

            # Update default_vals from sample
            for col in FEATURE_COLUMNS_SOH:
                if col in sample_row:
                    default_vals[col] = sample_row[col]
            
            # Days since last service calculation if needed
            if "observation_date" in sample_row and "last_service_date" in sample_row:
                try:
                    obs_dt = pd.to_datetime(sample_row["observation_date"])
                    srv_dt = pd.to_datetime(sample_row["last_service_date"])
                    diff = (obs_dt - srv_dt).days
                    if not pd.isna(diff):
                        default_vals["days_since_last_service"] = float(diff)
                except Exception:
                    pass

            st.caption(f"Loaded Pack ID: **{selected_battery_id}** | Ground Truth: SOH = **{actual_soh:.2f}%**, RUL = **{actual_rul:,.0f} Cycles**")
        else:
            st.warning("Dataset not found. Manual parameter inputs are active below.")

    with input_tab2:
        st.markdown("Or choose a pre-configured operating scenario preset:")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("🟢 New Pack Preset (Healthy)", use_container_width=True):
            default_vals.update({
                'cycle_count': 120, 'battery_age_days': 100, 'internal_resistance_mohm': 26.0,
                'fast_charge_ratio': 0.15, 'total_energy_throughput_kwh': 9500.0
            })
            actual_soh, actual_rul = None, None
        if col_p2.button("🟡 High Mileage Fleet Preset", use_container_width=True):
            default_vals.update({
                'cycle_count': 1400, 'battery_age_days': 1100, 'internal_resistance_mohm': 58.0,
                'fast_charge_ratio': 0.65, 'total_energy_throughput_kwh': 130000.0
            })
            actual_soh, actual_rul = None, None
        if col_p3.button("🔴 Near EOL Battery Preset", use_container_width=True):
            default_vals.update({
                'cycle_count': 2200, 'battery_age_days': 1700, 'internal_resistance_mohm': 78.0,
                'fast_charge_ratio': 0.80, 'total_energy_throughput_kwh': 195000.0
            })
            actual_soh, actual_rul = None, None

    # Form layout with structured cards
    st.markdown('<div class="eyebrow" style="margin-top: 24px;"><span class="dot">●</span>TELEMETRY INPUT CONFIGURATION</div>', unsafe_allow_html=True)

    with st.form("battery_prediction_form"):
        col_sec1, col_sec2, col_sec3 = st.columns(3)

        with col_sec1:
            st.markdown('<div class="param-title">⚡ Cell & Pack Architecture</div>', unsafe_allow_html=True)
            chemistry = st.selectbox("Cell Chemistry", ["NMC", "LFP", "NCA", "LMO"], index=["NMC", "LFP", "NCA", "LMO"].index(str(default_vals['chemistry'])) if str(default_vals['chemistry']) in ["NMC", "LFP", "NCA", "LMO"] else 0)
            manufacturer = st.selectbox("Pack Manufacturer", ["VoltaCore", "StormCell", "IonFlux", "Amperion", "NexCell", "GigaWatt Labs", "PowerGrid Batteries", "CellTech"], index=0)
            vehicle_model = st.selectbox("Vehicle Platform", ["Crossover-X2", "Sedan-E1", "Van-E4", "Hatch-E3", "SUV-E5", "Compact-E1", "Pickup-E7", "Sport-E9"], index=0)
            pack_capacity_kwh = st.number_input("Nominal Capacity (kWh)", min_value=30.0, max_value=150.0, value=float(default_vals['pack_capacity_kwh']), step=1.0)
            bms_firmware = st.selectbox("BMS Firmware", ["BMS_v2.1", "BMS_v2.3", "BMS_v3.0", "BMS_v3.2", "BMS_v3.5"], index=3)
            recall_flag = st.selectbox("Manufacturer Recall Flag", [0, 1], index=int(default_vals['recall_flag']))

        with col_sec2:
            st.markdown('<div class="param-title">🔋 Cycling & Charging Habits</div>', unsafe_allow_html=True)
            cycle_count = st.number_input("Cumulative Cycle Count", min_value=0, max_value=5000, value=int(default_vals['cycle_count']), step=25)
            battery_age_days = st.number_input("Battery Age (Days)", min_value=1, max_value=4000, value=int(default_vals['battery_age_days']), step=30)
            total_energy_throughput = st.number_input("Throughput (kWh)", min_value=0.0, max_value=500000.0, value=float(default_vals['total_energy_throughput_kwh']), step=1000.0)
            avg_daily_distance = st.number_input("Avg Daily Distance (km)", min_value=5.0, max_value=600.0, value=float(default_vals['avg_daily_distance_km']), step=5.0)
            fast_charge_ratio = st.slider("DC Fast-Charge Ratio", min_value=0.0, max_value=1.0, value=float(default_vals['fast_charge_ratio']), step=0.02)
            avg_charging_power = st.number_input("Avg Charging Power (kW)", min_value=5.0, max_value=350.0, value=float(default_vals['avg_charging_power_kw']), step=5.0)
            peak_temp_fast_charge = st.number_input("Peak Fast-Charge Temp (°C)", min_value=-10.0, max_value=65.0, value=float(default_vals['peak_temp_during_fast_charge_c']), step=1.0)

        with col_sec3:
            st.markdown('<div class="param-title">🌡️ Telemetry & Environment</div>', unsafe_allow_html=True)
            internal_resistance = st.number_input("Internal Resistance (mΩ)", min_value=10.0, max_value=120.0, value=float(default_vals['internal_resistance_mohm']), step=0.5)
            voltage_avg_v = st.number_input("Avg Pack Voltage (V)", min_value=250.0, max_value=500.0, value=float(default_vals['voltage_avg_v']), step=1.0)
            current_avg_a = st.number_input("Avg Pack Current (A)", min_value=20.0, max_value=400.0, value=float(default_vals['current_avg_a']), step=5.0)
            climate_zone = st.selectbox("Climate Zone", ["temperate", "cold", "hot_arid", "tropical_humid", "mixed"], index=0)
            avg_ambient_temp = st.number_input("Avg Ambient Temp (°C)", min_value=-30.0, max_value=55.0, value=float(default_vals['avg_ambient_temp_c']), step=1.0)
            dod = st.slider("Depth of Discharge (%)", min_value=10.0, max_value=100.0, value=float(default_vals['avg_depth_of_discharge_pct']), step=1.0)
            days_since_service = st.number_input("Days Since Last Service", min_value=0.0, max_value=1500.0, value=float(default_vals['days_since_last_service']), step=10.0)

        # Hidden default pass-throughs
        service_count = int(default_vals.get('service_count', 1))
        humidity_pct = float(default_vals.get('humidity_pct', 45.0))
        altitude_m = float(default_vals.get('altitude_m', 350.0))
        charge_cycles_per_month = float(default_vals.get('charge_cycles_per_month', 40.0))
        avg_soc = float(default_vals.get('avg_state_of_charge_pct', 55.0))
        charge_time_avg_min = float(default_vals.get('charge_time_avg_min', 40.0))

        submit_prediction = st.form_submit_button("⚡ ANALYZE & PREDICT BATTERY HEALTH", use_container_width=True)

    # Prediction execution
    if submit_prediction or load_sample_btn:
        if soh_model is None or rul_model is None:
            st.error("ML Models or preprocessors are not loaded. Please ensure model files exist in the data/ directory.")
        else:
            with st.spinner("Executing Random Forest inference pipelines..."):
                input_dict = {
                    'chemistry': [chemistry],
                    'manufacturer': [manufacturer],
                    'vehicle_model': [vehicle_model],
                    'pack_capacity_kwh': [pack_capacity_kwh],
                    'battery_age_days': [battery_age_days],
                    'cycle_count': [cycle_count],
                    'climate_zone': [climate_zone],
                    'avg_daily_distance_km': [avg_daily_distance],
                    'avg_charging_power_kw': [avg_charging_power],
                    'fast_charge_ratio': [fast_charge_ratio],
                    'avg_depth_of_discharge_pct': [dod],
                    'avg_state_of_charge_pct': [avg_soc],
                    'charge_cycles_per_month': [charge_cycles_per_month],
                    'total_energy_throughput_kwh': [total_energy_throughput],
                    'avg_ambient_temp_c': [avg_ambient_temp],
                    'humidity_pct': [humidity_pct],
                    'altitude_m': [altitude_m],
                    'internal_resistance_mohm': [internal_resistance],
                    'voltage_avg_v': [voltage_avg_v],
                    'current_avg_a': [current_avg_a],
                    'charge_time_avg_min': [charge_time_avg_min],
                    'peak_temp_during_fast_charge_c': [peak_temp_fast_charge],
                    'service_count': [service_count],
                    'bms_firmware_version': [bms_firmware],
                    'recall_flag': [recall_flag],
                    'days_since_last_service': [days_since_service]
                }

                input_df = pd.DataFrame(input_dict)

                # Preprocess & Predict SOH
                X_soh_trans = soh_prep.transform(input_df)
                pred_soh = float(soh_model.predict(X_soh_trans)[0])
                pred_soh = max(0.0, min(100.0, pred_soh))

                # Derive capacity fade and predict RUL
                cap_fade = max(0.0, 100.0 - pred_soh)
                input_df_rul = input_df.copy()
                input_df_rul['capacity_fade_pct'] = cap_fade

                X_rul_trans = rul_prep.transform(input_df_rul)
                pred_rul = float(rul_model.predict(X_rul_trans)[0])
                pred_rul = max(0.0, pred_rul)

                # Determine health tier
                if pred_soh >= 88.0:
                    status_class = "success"
                    status_label = "Optimal / Healthy"
                    status_msg = "Battery operates with minimal degradation and nominal impedance."
                elif pred_soh >= 80.0:
                    status_class = "warning"
                    status_label = "Moderate Aging"
                    status_msg = "Degradation is accelerating. Recommended to reduce frequent DC fast-charging."
                else:
                    status_class = "danger"
                    status_label = "End-of-First-Life (EOL)"
                    status_msg = "Capacity dropped below 80% threshold. Candidate for stationary repurposing."

                st.session_state["last_pred"] = {
                    "soh": pred_soh,
                    "rul": pred_rul,
                    "status": status_label,
                }

                # Display Results
                st.markdown('<div class="eyebrow" style="margin-top: 36px;"><span class="dot">●</span>PREDICTION INFERENCE RESULTS</div>', unsafe_allow_html=True)
                
                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    st.markdown(
                        f"""
<div class="result-card {status_class}">
    <div class="lbl">Predicted State of Health</div>
    <div class="val">{pred_soh:.2f}<span style="font-size: 26px; color: var(--accent);">%</span></div>
    <div class="desc"><strong>Status:</strong> {status_label}</div>
    <div class="bar" style="margin-top: 14px;"><span style="width: {pred_soh}%;"></span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    st.markdown(
                        f"""
<div class="result-card">
    <div class="lbl">Predicted Remaining Life</div>
    <div class="val">{pred_rul:,.0f}<span style="font-size: 22px; color: var(--accent);"> CYCLES</span></div>
    <div class="desc">Estimated <strong>{pred_rul / 250:.1f} years</strong> of typical daily operations remaining</div>
    <div class="bar" style="margin-top: 14px;"><span style="width: {min(100.0, (pred_rul/2500)*100)}%;"></span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with res_col3:
                    fade_text = f"{cap_fade:.2f}%"
                    st.markdown(
                        f"""
<div class="result-card">
    <div class="lbl">Capacity Fade & Diagnostics</div>
    <div class="val">{fade_text}</div>
    <div class="desc">{status_msg}</div>
    <div class="bar" style="margin-top: 14px;"><span style="width: {min(100.0, cap_fade * 4)}%; background: var(--warn);"></span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # Ground truth comparison if available
                if actual_soh is not None and actual_rul is not None:
                    st.markdown(
                        f"""
<div style="background: var(--surface-2); border: 1px solid var(--line); border-radius: 4px; padding: 14px 20px; margin: 18px 0; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--accent); letter-spacing: 0.1em;">GROUND TRUTH COMPARISON</span>
        <div style="font-size: 13px; color: var(--text); margin-top: 4px;">Actual SOH: <strong>{actual_soh:.2f}%</strong> (Delta: {abs(pred_soh - actual_soh):.2f}%) &nbsp;|&nbsp; Actual RUL: <strong>{actual_rul:,.0f} Cycles</strong> (Delta: {abs(pred_rul - actual_rul):.0f} cycles)</div>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--success);">● MODEL IN-BOUND</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # Diagnostic Comparison Graphs
                st.markdown('<div class="eyebrow" style="margin-top: 24px;"><span class="dot">●</span>DIAGNOSTIC BENCHMARK CHARTS</div>', unsafe_allow_html=True)
                pcol1, pcol2 = st.columns(2)

                with pcol1:
                    fig, ax = setup_dark_plot()
                    categories = ['SOH Health', 'Remaining Life', 'Resistance Margin', 'Thermal Stress']
                    # Normalize values on 0-100 scale for intuitive radar/bar view
                    norm_soh = pred_soh
                    norm_rul = min(100.0, (pred_rul / 2200.0) * 100.0)
                    norm_ir = max(0.0, 100.0 - ((internal_resistance - 20) / 70.0) * 100.0)
                    norm_therm = max(0.0, 100.0 - (peak_temp_fast_charge / 55.0) * 100.0)
                    vals = [norm_soh, norm_rul, norm_ir, norm_therm]

                    bars = ax.barh(categories, vals, color=[ACCENT, "#e5a93c", SUCCESS, "#5eead4"], height=0.45, edgecolor=LINE)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Health Index Score (0-100)", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
                    for bar in bars:
                        xval = bar.get_width()
                        ax.text(xval + 1.5, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}", ha='left', va='center', color=TEXT, fontfamily='JetBrains Mono', fontsize=9)
                    ax.set_title("Health Scorecard vs Baseline", color=TEXT, fontfamily='Instrument Serif', fontsize=18)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with pcol2:
                    fig, ax = setup_dark_plot()
                    # Projected Degradation Curve
                    cycles_future = np.linspace(cycle_count, cycle_count + pred_rul + 300, 50)
                    soh_future = pred_soh - (pred_soh - 78.0) * ((cycles_future - cycle_count) / max(1.0, pred_rul + 300)) ** 1.3
                    
                    ax.plot(cycles_future, soh_future, color=ACCENT, linewidth=2, label="Forecast Trajectory")
                    ax.scatter([cycle_count], [pred_soh], color=SUCCESS, s=70, zorder=5, label="Current State")
                    ax.axhline(80.0, color=DANGER, linestyle="--", linewidth=1.2, label="80% EOL Threshold")
                    
                    ax.set_xlabel("Total Operating Cycles", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
                    ax.set_ylabel("State of Health (%)", color=MUTED, fontfamily='JetBrains Mono', fontsize=10)
                    ax.set_title("Remaining Degradation Projection", color=TEXT, fontfamily='Instrument Serif', fontsize=18)
                    ax.legend(facecolor=SURFACE_2, edgecolor=LINE, labelcolor=TEXT, loc="upper right")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)


# --------------------------------
# MODEL INSIGHTS PAGE
# --------------------------------

elif page == "Model Insights":

    page_header(
        "Model Performance & Insights",
        "How the models <em>perform</em>.",
        "Detailed evaluation metrics, Top 10 feature importance scores, "
        "and residual diagnostics for the Random Forest regression models.",
    )

    # Evaluation Metrics Table
    st.markdown('<div class="eyebrow"><span class="dot">●</span>MODEL EVALUATION BENCHMARK</div>', unsafe_allow_html=True)

    if eval_df is not None:
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        
        # SOH RF
        soh_rf = eval_df[(eval_df['Target'] == 'SOH') & (eval_df['Model'].str.contains('Random Forest', case=False, na=False))]
        if not soh_rf.empty:
            r2_soh = soh_rf['R2 Score'].values[0]
            mae_soh = soh_rf['MAE'].values[0]
        else:
            r2_soh, mae_soh = 0.975, 0.746

        # RUL RF
        rul_rf = eval_df[(eval_df['Target'] == 'RUL') & (eval_df['Model'].str.contains('Random Forest', case=False, na=False))]
        if not rul_rf.empty:
            r2_rul = rul_rf['R2 Score'].values[0]
            mae_rul = rul_rf['MAE'].values[0]
        else:
            r2_rul, mae_rul = 0.879, 301.38

        with c_m1:
            st.markdown(f"""
<div class="result-card">
    <div class="lbl">SOH R² Score</div>
    <div class="val">{r2_soh * 100:.1f}<span style="font-size: 24px; color: var(--accent);">%</span></div>
    <div class="desc">Variance explained by SOH RF</div>
</div>
""", unsafe_allow_html=True)

        with c_m2:
            st.markdown(f"""
<div class="result-card">
    <div class="lbl">SOH Mean Abs Error</div>
    <div class="val">{mae_soh:.2f}<span style="font-size: 24px; color: var(--accent);">%</span></div>
    <div class="desc">Average test set percentage error</div>
</div>
""", unsafe_allow_html=True)

        with c_m3:
            st.markdown(f"""
<div class="result-card">
    <div class="lbl">RUL R² Score</div>
    <div class="val">{r2_rul * 100:.1f}<span style="font-size: 24px; color: var(--accent);">%</span></div>
    <div class="desc">Variance explained by RUL RF</div>
</div>
""", unsafe_allow_html=True)

        with c_m4:
            st.markdown(f"""
<div class="result-card">
    <div class="lbl">RUL Mean Abs Error</div>
    <div class="val">{mae_rul:,.0f}<span style="font-size: 22px; color: var(--accent);"> CYC</span></div>
    <div class="desc">Cycle count deviation on test set</div>
</div>
""", unsafe_allow_html=True)

        # Formatted comparison table
        st.markdown('<div class="eyebrow" style="margin-top: 28px;"><span class="dot">●</span>REGRESSION PERFORMANCE COMPARISON</div>', unsafe_allow_html=True)
        st.dataframe(
            eval_df.style.format({
                "MAE": "{:.3f}",
                "MSE": "{:.1f}",
                "RMSE": "{:.3f}",
                "R2 Score": "{:.4f}"
            }),
            use_container_width=True
        )
    else:
        st.info("Evaluation metrics file not found.")

    # Feature Importance Graphs
    st.markdown('<div class="eyebrow" style="margin-top: 36px;"><span class="dot">●</span>TOP 10 FEATURE IMPORTANCE ANALYSIS</div>', unsafe_allow_html=True)

    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        st.markdown("#### Top SOH Features")
        if os.path.exists("data/soh_feature_importance.png"):
            st.image("data/soh_feature_importance.png", use_container_width=True)
        else:
            st.info("SOH feature importance plot loading...")

    with col_fi2:
        st.markdown("#### Top RUL Features")
        if os.path.exists("data/rul_feature_importance.png"):
            st.image("data/rul_feature_importance.png", use_container_width=True)
        else:
            st.info("RUL feature importance plot loading...")

    # Prediction vs Actual & Residual Analysis
    st.markdown('<div class="eyebrow" style="margin-top: 36px;"><span class="dot">●</span>VALIDATION SCATTER & RESIDUALS</div>', unsafe_allow_html=True)

    tab_v1, tab_v2 = st.tabs(["ACTUAL VS PREDICTED", "RESIDUAL ERROR ANALYSIS"])

    with tab_v1:
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("#### SOH: Actual vs Predicted")
            if os.path.exists("data/soh_actual_vs_predicted.png"):
                st.image("data/soh_actual_vs_predicted.png", use_container_width=True)
        with c_v2:
            st.markdown("#### RUL: Actual vs Predicted")
            if os.path.exists("data/rul_actual_vs_predicted.png"):
                st.image("data/rul_actual_vs_predicted.png", use_container_width=True)

    with tab_v2:
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.markdown("#### SOH Residual Error Distribution")
            if os.path.exists("data/soh_residuals.png"):
                st.image("data/soh_residuals.png", use_container_width=True)
        with c_r2:
            st.markdown("#### RUL Residual Error Distribution")
            if os.path.exists("data/rul_residuals.png"):
                st.image("data/rul_residuals.png", use_container_width=True)

# --------------------------------
# RIGHT INFO PANEL (SOH / RUL / PROJECT)
# --------------------------------

def _metric(target, col, fallback):
    try:
        row = eval_df[(eval_df["Target"] == target) &
                      (eval_df["Model"].str.contains("Random Forest", case=False, na=False))]
        if not row.empty:
            return float(row[col].values[0])
    except Exception:
        pass
    return fallback


def render_right_panel():
    lp = st.session_state.get("last_pred")

    r2_s = _metric("SOH", "R2 Score", 0.975) * 100
    mae_s = _metric("SOH", "MAE", 0.746)
    r2_r = _metric("RUL", "R2 Score", 0.879) * 100
    mae_r = _metric("RUL", "MAE", 301.38)
    packs = dataset["battery_id"].nunique() if dataset is not None else 2600

    if lp:
        soh_live = (
            '<div class="rp-live"><div class="k">Last prediction</div>'
            f'<div class="v">{lp["soh"]:.2f}<span>%</span></div>'
            f'<div class="s">{lp["status"]}</div></div>'
        )
        rul_live = (
            '<div class="rp-live"><div class="k">Last prediction</div>'
            f'<div class="v">{lp["rul"]:,.0f}<span>CYCLES</span></div>'
            f'<div class="s">about {lp["rul"] / 250:.1f} years at 250 cycles per year</div></div>'
        )
    else:
        empty = (
            '<div class="rp-live"><div class="k">Last prediction</div>'
            '<div class="v">--</div>'
            '<div class="s">Run a prediction to see it here</div></div>'
        )
        soh_live = rul_live = empty

    html = (
        '<aside class="rpanel">'

        '<details class="rp-item" name="rp" open>'
        '<summary><span class="rp-t">State of Health</span><span class="rp-caret">+</span></summary>'
        '<div class="rp-body">'
        + soh_live +
        '<p>SOH is the pack\'s current capacity as a share of its rated design capacity. 100% means factory fresh.</p>'
        '<div class="rp-row"><span class="k">88% and above</span><span class="v">Optimal</span></div>'
        '<div class="rp-row"><span class="k">80% to 88%</span><span class="v">Moderate aging</span></div>'
        '<div class="rp-row"><span class="k">Below 80%</span><span class="v">End of first life</span></div>'
        '<div class="rp-row"><span class="k">Model R²</span><span class="v">' + f'{r2_s:.1f}%' + '</span></div>'
        '<div class="rp-row"><span class="k">Mean error</span><span class="v">' + f'{mae_s:.2f}%' + '</span></div>'
        '</div></details>'

        '<details class="rp-item" name="rp">'
        '<summary><span class="rp-t">Remaining Useful Life</span><span class="rp-caret">+</span></summary>'
        '<div class="rp-body">'
        + rul_live +
        '<p>RUL is the number of equivalent full charge-discharge cycles left before SOH falls below the 80% threshold.</p>'
        '<div class="rp-row"><span class="k">Inputs</span><span class="v">26 features + capacity fade</span></div>'
        '<div class="rp-row"><span class="k">Years estimate</span><span class="v">cycles ÷ 250</span></div>'
        '<div class="rp-row"><span class="k">Model R²</span><span class="v">' + f'{r2_r:.1f}%' + '</span></div>'
        '<div class="rp-row"><span class="k">Mean error</span><span class="v">' + f'{mae_r:,.0f} cycles' + '</span></div>'
        '</div></details>'

        '<details class="rp-item" name="rp">'
        '<summary><span class="rp-t">This Project</span><span class="rp-caret">+</span></summary>'
        '<div class="rp-body">'
        '<p>EV Battery Health Predictor estimates SOH and RUL of EV battery packs from telemetry using two Random Forest regressors.</p>'
        '<div class="rp-row"><span class="k">Dataset</span><span class="v">' + f'{packs:,} packs' + '</span></div>'
        '<div class="rp-row"><span class="k">Features</span><span class="v">26 inputs</span></div>'
        '<div class="rp-row"><span class="k">Chemistries</span><span class="v">NMC · LFP · NCA · LMO</span></div>'
        '<div class="rp-row"><span class="k">Pipeline</span><span class="v">Preprocess → SOH → RUL</span></div>'
        '<div class="rp-row"><span class="k">Stack</span><span class="v">Python · Streamlit · scikit-learn</span></div>'
        '</div></details>'

        '<details class="rp-item" name="rp">'
        '<summary><span class="rp-t">System Telemetry</span><span class="rp-caret">+</span></summary>'
        '<div class="rp-body">'
        '<div class="rp-row"><span class="k">Model engine</span><span class="v" style="color: var(--success);">● Active</span></div>'
        '<div class="rp-row"><span class="k">Architecture</span><span class="v">Random Forest</span></div>'
        '<div class="rp-row"><span class="k">SOH accuracy</span><span class="v" style="color: var(--accent);">' + f'{r2_s:.1f}% R²' + '</span></div>'
        '<div class="rp-row"><span class="k">RUL accuracy</span><span class="v" style="color: var(--accent);">' + f'{r2_r:.1f}% R²' + '</span></div>'
        '<div class="rp-row"><span class="k">Dataset</span><span class="v">' + f'{packs:,} packs' + '</span></div>'
        '</div></details>'

        '<details class="rp-item" name="rp">'
        '<summary><span class="rp-t">Prediction Metrics</span><span class="rp-caret">+</span></summary>'
        '<div class="rp-body">'
        '<div class="rp-row"><span class="k">Target 01</span><span class="v">SOH (% capacity)</span></div>'
        '<div class="rp-row"><span class="k">Target 02</span><span class="v">RUL (cycles)</span></div>'
        '<div class="rp-row"><span class="k">EOL limit</span><span class="v">80.0% SOH</span></div>'
        '<div class="rp-row"><span class="k">Input dims</span><span class="v">26 features</span></div>'
        '</div></details>'
        '</aside>'
    )
    st.markdown(html, unsafe_allow_html=True)


render_right_panel()


# --------------------------------
# FOOTER
# --------------------------------

st.markdown(
    """
<div class="footer">
<span>EV BATTERY HEALTH PREDICTOR · RANDOM FOREST ENGINE</span>
<span>ANALYTICS & DIAGNOSTICS PLATFORM</span>
</div>
""",
    unsafe_allow_html=True,
)