from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import base64
from datetime import date
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predictor import load_teams, predict_match
import simulator

simulator = importlib.reload(simulator)

build_round_of_32 = simulator.build_round_of_32
monte_carlo_tournament = simulator.monte_carlo_tournament
qualified_teams = simulator.qualified_teams
simulate_group_stage = simulator.simulate_group_stage


GENERATED_DATA_PATH = ROOT / "data" / "team_ratings.csv"
ALL_RATINGS_PATH = ROOT / "data" / "all_team_ratings.csv"
SAMPLE_DATA_PATH = ROOT / "data" / "sample_team_ratings.csv"
MODEL_REPORT_PATH = ROOT / "models" / "model_report.txt"
MODEL_METRICS_PATH = ROOT / "models" / "model_metrics.json"
CALIBRATION_PATH = ROOT / "models" / "calibration_curve.csv"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
HERO_IMAGE_PATH = ROOT / "app" / "assets" / "football.jpg"
MATCHES_PATH = ROOT / "data" / "international_matches.csv"


st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
)

def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


hero_background = image_data_uri(HERO_IMAGE_PATH) if HERO_IMAGE_PATH.exists() else ""

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Libre+Baskerville:wght@400;700&display=swap');

        :root {
            --ink: #17201d;
            --muted: #5d6b65;
            --line: #dfe7e2;
            --surface: #ffffff;
            --surface-soft: #f6f9f7;
            --green: #111111;
            --teal: #111111;
            --red: #111111;
            --gold: #111111;
        }

        html, body, .stApp,
        button, input, textarea, select,
        p, label, table {
            font-family: "Libre Baskerville", Georgia, "Times New Roman", serif;
        }

        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
        }

        .stApp {
            background: #ffffff;
            color: var(--ink);
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        div[data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            padding-top: 0;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        div[data-testid="stTabs"] button {
            border-radius: 6px 6px 0 0;
            padding: 0.7rem 1rem;
            color: #111111 !important;
            font-weight: 400;
            opacity: 1;
        }

        div[data-testid="stTabs"] button p {
            color: #111111 !important;
            opacity: 1;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #000000 !important;
            font-weight: 700;
            border-bottom-color: #000000;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #000000;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 6px 18px rgba(21, 52, 39, 0.06);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 200;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(21, 52, 39, 0.05);
        }

        .stButton > button {
            border-radius: 2px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--ink);
            font-weight: 700;
            white-space: nowrap;
            box-shadow: 0 6px 16px rgba(21, 52, 39, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--green);
            box-shadow: 0 10px 24px rgba(21, 52, 39, 0.12);
        }

        .app-header {
            width: min(96vw, 1800px);
            min-height: 440px;
            background-image:
                linear-gradient(180deg, rgba(8, 18, 14, 0.1), rgba(8, 18, 14, 0.58)),
                url("__HERO_BACKGROUND__");
            background-size: cover;
            background-position: center top;
            border-radius: 6px;
            padding: clamp(1.3rem, 4vw, 3rem);
            box-shadow: 0 18px 42px rgba(21, 52, 39, 0.18);
            margin: 0 0 1.1rem 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .hero-kicker {
            color: #ffffff;
            font-family: "Cormorant Garamond", Georgia, "Times New Roman", serif;
            font-size: clamp(1.2rem, 2vw, 1.2rem);
            line-height: 1;
            font-weight: 700;
            margin-bottom: 1.2rem;
            text-align: center;
        }

        .app-header .app-title {
            font-family: "Cormorant Garamond", Georgia, "Times New Roman", serif;
            font-size: clamp(4.8rem, 10.5vw, 9rem);
            line-height: 0.82;
            font-weight: 600;
            letter-spacing: -0.045em;
            margin: 0;
            color: #ffffff !important;
            white-space: nowrap;
            text-shadow: 0 3px 22px rgba(0, 0, 0, 0.78);
            text-align: center;
        }

        .app-subtitle {
            color: #ffffff;
            font-family: "Cormorant Garamond", Georgia, "Times New Roman", serif;
            font-size: clamp(1.15rem, 1.8vw, 1.55rem);
            line-height: 1.35;
            font-weight: 600;
            margin: 2.8rem auto 0;
            max-width: 900px;
            text-align: center;
        }

        .hero-copy {
            max-width: 1120px;
            background: transparent;
            border: 0;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.95rem 0 1.1rem 0;
        }

        .summary-card {
            position: relative;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem;
            box-shadow: none;
            text-align: center;
        }

        .summary-info {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 0.95rem;
            height: 0.95rem;
            margin-left: 0.35rem;
            border: 1px solid var(--muted);
            border-radius: 50%;
            color: var(--muted);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 0.62rem;
            line-height: 1;
            vertical-align: middle;
            cursor: help;
        }

        .summary-info::after {
            content: attr(data-tooltip);
            position: absolute;
            z-index: 20;
            left: 50%;
            bottom: calc(100% + 0.55rem);
            width: max-content;
            max-width: 230px;
            padding: 0.5rem 0.65rem;
            border-radius: 6px;
            background: var(--ink);
            color: #ffffff;
            font-family: "Libre Baskerville", Georgia, "Times New Roman", serif;
            font-size: 0.72rem;
            font-weight: 400;
            line-height: 1.35;
            text-align: center;
            white-space: normal;
            opacity: 0;
            pointer-events: none;
            transform: translate(-50%, 0.25rem);
            transition: opacity 120ms ease, transform 120ms ease;
        }

        .summary-info:hover::after,
        .summary-info:focus::after {
            opacity: 1;
            transform: translate(-50%, 0);
        }

        .match-card:hover,
        .team-card:hover,
        .prob-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(21, 52, 39, 0.12);
        }

        .app-header .summary-grid {
            max-width: 760px;
            margin-bottom: 0;
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .summary-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 600;
        }

        .prob-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 6px 18px rgba(21, 52, 39, 0.05);
        }

        .prob-row {
            display: grid;
            grid-template-columns: minmax(120px, 1.2fr) 4fr 70px;
            align-items: center;
            gap: 0.8rem;
        }

        .prob-label {
            font-weight: 600;
            color: var(--ink);
            overflow-wrap: anywhere;
        }

        .prob-track {
            height: 12px;
            background: #e6eee9;
            border-radius: 999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
        }

        .prob-value {
            font-weight: 600;
            text-align: right;
            color: var(--ink);
        }

        .today-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0 1.1rem 0;
            box-shadow: 0 10px 26px rgba(21, 52, 39, 0.07);
        }

        .today-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-bottom: 0.8rem;
        }

        .today-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--ink);
        }

        .today-date {
            color: var(--muted);
            font-weight: 500;
        }

        .match-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .match-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem;
            background: var(--surface-soft);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .match-meta {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 500;
            margin-bottom: 0.45rem;
        }

        .match-teams {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .match-pick {
            color: var(--green);
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .mini-probs {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 500;
        }

        .team-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1rem 0;
        }

        .team-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 3px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(21, 52, 39, 0.05);
            text-align: center;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        div[data-testid="stSelectbox"] label {
            justify-content: center;
            width: 100%;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #111111;
            border-color: #111111;
            color: #ffffff;
            text-align: center;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child {
            justify-content: center;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] span,
        div[data-testid="stSelectbox"] [data-baseweb="select"] input {
            color: #ffffff !important;
        }

        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
            fill: #ffffff;
        }

        .team-name {
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }

        .flag {
            display: inline-block;
            margin-right: 0.35rem;
            filter: saturate(1.05);
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
        }

        .stat-box {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 3px;
            padding: 0.55rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.6rem;
            font-weight: 400;
            text-transform: uppercase;
        }

        .stat-value {
            color: var(--ink);
            font-size: 0.85rem;
            font-weight: 400;
        }

        .confidence-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            margin: 0.2rem 0 0.75rem 0;
            color: #ffffff;
            font-weight: 600;
            background: var(--teal);
        }

        .prediction-title {
            margin: 1.25rem 0 0.45rem;
            color: var(--ink);
            font-family: "Libre Baskerville", Georgia, "Times New Roman", serif;
            font-size: 1.2rem;
            font-weight: 400;
            line-height: 1.4;
            text-align: center;
        }

        .explanation-title {
            margin: 1.25rem 0 0.65rem;
            color: var(--ink);
            font-family: "Libre Baskerville", Georgia, "Times New Roman", serif;
            font-size: 1.2rem;
            font-weight: 400;
            line-height: 1.4;
            text-align: center;
        }

        .explanation-table {
            width: 100%;
            border: 1px solid var(--line);
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(21, 52, 39, 0.05);
            overflow: hidden;
        }

        .explanation-table th,
        .explanation-table td {
            padding: 0.65rem 0.8rem;
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            font-size: 0.85rem;
            font-weight: 400;
            text-align: center;
            vertical-align: middle;
        }

        .explanation-table th {
            background: var(--surface-soft);
            color: var(--muted);
        }

        .explanation-table tr:last-child td {
            border-bottom: 0;
        }

        .prediction-summary {
            margin: 0 0 1rem;
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 400;
            line-height: 1.5;
            text-align: center;
        }

        .tournament-summary {
            max-width: 760px;
            margin: 0 auto 1.1rem;
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 400;
            line-height: 1.65;
            text-align: center;
        }

        .control-guide {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .control-guide-card {
            padding: 0.85rem 0.9rem;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 0;
            text-align: center;
        }

        .control-guide-title {
            margin-bottom: 0.3rem;
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 400;
        }

        .control-guide-copy {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 400;
            line-height: 1.5;
        }

        .performance-kpis {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .performance-kpi {
            padding: 0.9rem;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 0;
            text-align: center;
        }

        .performance-kpi-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 400;
            text-transform: uppercase;
        }

        .performance-kpi-value {
            margin: 0.3rem 0;
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 400;
        }

        .performance-kpi-help {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 400;
            line-height: 1.45;
        }

        .table-scroll {
            width: 100%;
            overflow-x: auto;
            margin-bottom: 1rem;
        }

        .dataset-note {
            max-width: 860px;
            margin: 0 auto 1rem;
            padding: 1rem 1.1rem;
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 0;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.65;
            text-align: center;
        }

        .dataset-note a {
            color: var(--ink);
            text-decoration: underline;
        }

        .prediction-kpis {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .prediction-kpi {
            padding: 0.85rem 0.9rem;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 0;
            box-shadow: 0 6px 18px rgba(21, 52, 39, 0.06);
            text-align: center;
        }

        .prediction-kpi--winner {
            background: #e4f7ec;
            border: 2px solid #0aa96e;
        }

        .prediction-kpi-label {
            min-height: 2.5rem;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 400;
            line-height: 1.4;
        }

        .prediction-kpi-value {
            margin-top: 0.25rem;
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 400;
            line-height: 1.25;
        }

        @media (max-width: 760px) {
            .summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .app-header {
                width: calc(100vw - 1rem);
                min-height: 420px;
                background-position: center top;
            }

            .app-header .app-title {
                font-size: clamp(3rem, 16vw, 4.6rem);
                letter-spacing: -0.055em;
            }

            .hero-kicker {
                font-size: 1.15rem;
                margin-bottom: 1.1rem;
            }

            .app-subtitle {
                font-size: 1rem;
                line-height: 1.45;
                margin-top: 1.8rem;
            }

            .prob-row {
                grid-template-columns: 1fr;
                gap: 0.45rem;
            }

            .prob-value {
                text-align: left;
            }

            .prediction-kpis {
                grid-template-columns: 1fr;
            }

            .control-guide {
                grid-template-columns: 1fr;
            }

            .performance-kpis {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .prediction-kpi-label {
                min-height: auto;
            }

            .today-header {
                display: block;
            }

            .match-grid {
                grid-template-columns: 1fr;
            }

            .team-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """.replace("__HERO_BACKGROUND__", hero_background),
    unsafe_allow_html=True,
)


@st.cache_data
def get_teams() -> pd.DataFrame:
    data_path = GENERATED_DATA_PATH if GENERATED_DATA_PATH.exists() else SAMPLE_DATA_PATH
    return load_teams(data_path)


@st.cache_data
def get_prediction_teams() -> pd.DataFrame:
    data_path = ALL_RATINGS_PATH if ALL_RATINGS_PATH.exists() else GENERATED_DATA_PATH
    if not data_path.exists():
        data_path = SAMPLE_DATA_PATH
    return load_teams(data_path)


@st.cache_data
def run_simulations(simulations: int, seed: int) -> pd.DataFrame:
    return monte_carlo_tournament(get_teams(), simulations=simulations, seed=seed)


@st.cache_data
def get_model_metrics() -> dict:
    if not MODEL_METRICS_PATH.exists():
        return {}
    return json.loads(MODEL_METRICS_PATH.read_text())


@st.cache_data
def get_calibration_curve() -> pd.DataFrame:
    if not CALIBRATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CALIBRATION_PATH)


@st.cache_data
def get_model_report() -> str:
    if not MODEL_REPORT_PATH.exists():
        return ""
    return MODEL_REPORT_PATH.read_text()


@st.cache_data
def get_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def get_international_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        return pd.DataFrame()
    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"]).dt.date
    return matches


def metric_grid(items: list[tuple[str, str]]) -> None:
    st.markdown(metric_grid_html(items), unsafe_allow_html=True)


def performance_metric_grid(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="performance-kpi">'
        f'<div class="performance-kpi-label">{escape(label)}</div>'
        f'<div class="performance-kpi-value">{escape(value)}</div>'
        f'<div class="performance-kpi-help">{escape(help_text)}</div>'
        f'</div>'
        for label, value, help_text in items
    )
    st.markdown(f'<div class="performance-kpis">{cards}</div>', unsafe_allow_html=True)


def centered_table(frame: pd.DataFrame) -> None:
    table = frame.to_html(index=False, classes="explanation-table", border=0, justify="center")
    st.markdown(f'<div class="table-scroll">{table}</div>', unsafe_allow_html=True)


KPI_DESCRIPTIONS = {
    "Teams": "Number of national teams included in the tournament simulation.",
    "Groups": "Number of groups used in the tournament group stage.",
    "Model": "Machine-learning model currently used to calculate match probabilities.",
    "Accuracy": "Share of test matches for which the model predicted the correct result.",
    "Top Champion": "Team with the highest simulated probability of winning the tournament.",
    "Title Chance": "Estimated probability that the leading team wins the tournament.",
    "Simulations": "Number of complete tournament runs used for the estimate.",
    "Seed": "The simulation's replay code. Reusing it recreates the same set of results.",
    "Log Loss": "Prediction error that penalizes confident incorrect probabilities; lower is better.",
    "Brier": "Mean squared error of predicted probabilities; lower is better.",
    "RPS": "Ranked Probability Score across win, draw, and loss outcomes; lower is better.",
    "ECE": "Expected Calibration Error between predicted confidence and observed results; lower is better.",
    "Training Rows": "Historical matches used to train the selected model.",
    "Test Rows": "Held-out historical matches used to evaluate the model.",
}


def metric_grid_html(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="summary-card">'
        f'<div class="summary-label">{label}'
        f'<span class="summary-info" tabindex="0" role="img" aria-label="{escape(KPI_DESCRIPTIONS.get(label, label))}" '
        f'data-tooltip="{escape(KPI_DESCRIPTIONS.get(label, label))}">i</span>'
        f'</div>'
        f'<div class="summary-value">{value}</div>'
        f'</div>'
        for label, value in items
    )
    return f'<div class="summary-grid">{cards}</div>'


def probability_panel(rows: list[tuple[str, float, str]]) -> None:
    html_rows = []
    for label, probability, color in rows:
        percent = probability * 100
        html_rows.append(
            f'<div class="prob-card">'
            f'<div class="prob-row">'
            f'<div class="prob-label">{label}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width: {percent:.1f}%; background: {color};"></div>'
            f'</div>'
            f'<div class="prob-value">{percent:.1f}%</div>'
            f'</div>'
            f'</div>'
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


TEAM_COUNTRY_CODES = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bolivia": "BO",
    "Brazil": "BR",
    "Cameroon": "CM",
    "Canada": "CA",
    "Cape Verde": "CV",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Denmark": "DK",
    "DR Congo": "CD",
    "Ecuador": "EC",
    "Egypt": "EG",
    "England": "GB",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Honduras": "HN",
    "Iran": "IR",
    "Iraq": "IQ",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Mexico": "MX",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nigeria": "NG",
    "Panama": "PA",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Switzerland": "CH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
}


def flag_emoji(team: str) -> str:
    country_code = TEAM_COUNTRY_CODES.get(team)
    if not country_code:
        return ""
    return "".join(chr(127397 + ord(letter)) for letter in country_code.upper())


def team_label(team: str) -> str:
    flag = flag_emoji(team)
    return f"{flag} {team}" if flag else team


def team_html(team: str) -> str:
    flag = flag_emoji(team)
    if not flag:
        return escape(str(team))
    return f'<span class="flag">{flag}</span>{escape(str(team))}'


def confidence_text(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "Strong lean", "var(--green)"
    if probability >= 0.5:
        return "Moderate lean", "var(--teal)"
    if probability >= 0.4:
        return "Open match", "var(--gold)"
    return "Very tight", "var(--red)"


def team_card(row: pd.Series) -> str:
    stats = [
        ("Elo", f"{row['elo']:.0f}"),
        ("Rank", f"{row['fifa_rank']:.0f}"),
        ("Form", f"{row['form_points']:.0f}"),
        ("Attack", f"{row['attack_rating']:.1f}"),
        ("Defense", f"{row['defense_rating']:.1f}"),
        ("GD", f"{row.get('goal_difference_recent', 0):+.2f}"),
    ]
    stat_html = "".join(
        f'<div class="stat-box"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'
        for label, value in stats
    )
    return (
        f'<div class="team-card">'
        f'<div class="team-name">{team_html(str(row["team"]))}</div>'
        f'<div class="stat-grid">{stat_html}</div>'
        f'</div>'
    )


def team_snapshot(teams: pd.DataFrame, team_a: str, team_b: str) -> None:
    a = teams.loc[teams["team"] == team_a].iloc[0]
    b = teams.loc[teams["team"] == team_b].iloc[0]
    st.markdown(f'<div class="team-grid">{team_card(a)}{team_card(b)}</div>', unsafe_allow_html=True)


def todays_match_rows(prediction_teams: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, date | None]:
    if matches.empty:
        return pd.DataFrame(), None

    world_cup = matches[
        (matches["tournament"] == "FIFA World Cup")
        & (matches["home_score"].isna())
        & (matches["away_score"].isna())
    ].copy()
    if world_cup.empty:
        return pd.DataFrame(), None

    today = date.today()
    todays_matches = world_cup[world_cup["date"] == today]
    match_date = today

    if todays_matches.empty:
        future_matches = world_cup[world_cup["date"] > today].sort_values("date")
        if future_matches.empty:
            return pd.DataFrame(), None
        match_date = future_matches.iloc[0]["date"]
        todays_matches = future_matches[future_matches["date"] == match_date]

    rows = []
    available_teams = set(prediction_teams["team"])
    for match in todays_matches.itertuples(index=False):
        home_team = match.home_team
        away_team = match.away_team
        kickoff = getattr(match, "time", "TBD") if hasattr(match, "time") else "TBD"
        city = getattr(match, "city", "")
        country = getattr(match, "country", "")

        if home_team in available_teams and away_team in available_teams:
            prediction = predict_match(prediction_teams, home_team, away_team)
            probabilities = {
                home_team: prediction.team_a_win,
                "Draw": prediction.draw,
                away_team: prediction.team_b_win,
            }
            pick, pick_probability = max(probabilities.items(), key=lambda item: item[1])
            prediction_text = f"{team_label(pick) if pick != 'Draw' else pick} {pick_probability:.1%}"
            probability_text = (
                f"{team_label(home_team)} {prediction.team_a_win:.0%} | "
                f"Draw {prediction.draw:.0%} | "
                f"{team_label(away_team)} {prediction.team_b_win:.0%}"
            )
        else:
            missing = sorted({home_team, away_team} - available_teams)
            prediction_text = "Prediction unavailable"
            probability_text = f"Missing ratings: {', '.join(missing)}"

        rows.append(
            {
                "kickoff": kickoff,
                "home_team": home_team,
                "away_team": away_team,
                "location": ", ".join(part for part in [city, country] if part),
                "prediction": prediction_text,
                "probabilities": probability_text,
            }
        )

    return pd.DataFrame(rows), match_date


def todays_matches_panel(prediction_teams: pd.DataFrame) -> None:
    rows, match_date = todays_match_rows(prediction_teams, get_international_matches())
    if rows.empty or match_date is None:
        return

    cards = []
    for row in rows.itertuples(index=False):
        cards.append(
            f'<div class="match-card">'
            f'<div class="match-meta">{escape(str(row.kickoff))} · {escape(str(row.location))}</div>'
            f'<div class="match-teams">{team_html(str(row.home_team))} vs {team_html(str(row.away_team))}</div>'
            f'<div class="match-pick">{escape(str(row.prediction))}</div>'
            f'<div class="mini-probs">{escape(str(row.probabilities))}</div>'
            f'</div>'
        )

    panel_title = "Today's Matches" if match_date == date.today() else "Next Matchday"
    st.markdown(
        f'<section class="today-panel">'
        f'<div class="today-header">'
        f'<div class="today-title">{panel_title}</div>'
        f'<div class="today-date">{match_date.strftime("%B %d, %Y")}</div>'
        f'</div>'
        f'<div class="match-grid">{"".join(cards)}</div>'
        f'</section>',
        unsafe_allow_html=True,
    )

    options = [f"{team_label(row.home_team)} vs {team_label(row.away_team)}" for row in rows.itertuples(index=False)]
    selected = st.selectbox("Inspect fixture", options, key="today_fixture")
    selected_row = rows.iloc[options.index(selected)]
    home_team = selected_row["home_team"]
    away_team = selected_row["away_team"]
    if home_team in set(prediction_teams["team"]) and away_team in set(prediction_teams["team"]):
        prediction = predict_match(prediction_teams, home_team, away_team)
        probability_panel(
            [
                (f"{team_label(home_team)} win", prediction.team_a_win, "var(--green)"),
                ("Draw", prediction.draw, "var(--gold)"),
                (f"{team_label(away_team)} win", prediction.team_b_win, "var(--red)"),
            ]
        )


teams = get_teams()
prediction_teams = get_prediction_teams()
team_names = teams["team"].tolist()

metrics = get_model_metrics()
selected_model = metrics.get("selected_model", "Formula fallback")
accuracy_label = f"{metrics['accuracy']:.1%}" if metrics else "N/A"

st.markdown(
    '<section class="app-header">'
    '<div class="hero-copy">'
    '<div>'
    '<div class="hero-kicker">🏆</div>'
    '<div class="hero-kicker">World Cup 2026</div>'
    '<div class="app-title">PREDICTOR</div>'
    '<div class="app-subtitle">a streamlit app to predict individual match outcomes and runs Monte Carlo tournament simulations</div>'
    '</div>'
    '</div>'
    '</section>',
    unsafe_allow_html=True,
)
metric_grid(
    [
        ("Teams", f"{len(teams):,}"),
        ("Groups", f"{teams['group'].nunique():,}"),
        ("Model", selected_model),
        ("Accuracy", accuracy_label),
    ]
)

todays_matches_panel(prediction_teams)

tab_match, tab_tournament, tab_performance = st.tabs(
    ["Match Predictor", "Monte Carlo Tournament Simulator", "Model Performance"]
)

with tab_match:
    if "team_a_select" not in st.session_state:
        st.session_state["team_a_select"] = team_names[0]
    if "team_b_select" not in st.session_state:
        st.session_state["team_b_select"] = team_names[1 if len(team_names) > 1 else 0]

    left, right = st.columns(2)
    with left:
        team_a = st.selectbox("Team A", team_names, key="team_a_select", format_func=team_label)
    with right:
        team_b = st.selectbox("Team B", team_names, key="team_b_select", format_func=team_label)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        team_snapshot(teams, team_a, team_b)
        prediction = predict_match(teams, team_a, team_b)
        outcomes = {
            f"{team_label(team_a)} win": prediction.team_a_win,
            "Draw": prediction.draw,
            f"{team_label(team_b)} win": prediction.team_b_win,
        }
        favorite, favorite_probability = max(outcomes.items(), key=lambda item: item[1])
        confidence, _ = confidence_text(favorite_probability)
        team_a_highlight = " prediction-kpi--winner" if prediction.team_a_win >= prediction.team_b_win else ""
        team_b_highlight = " prediction-kpi--winner" if prediction.team_b_win > prediction.team_a_win else ""

        st.markdown(
            f'<div class="prediction-title">Prediction</div>'
            f'<div class="prediction-summary">{confidence}: {favorite} ({favorite_probability:.1%})</div>'
            f'<div class="prediction-kpis">'
            f'<div class="prediction-kpi{team_a_highlight}"><div class="prediction-kpi-label">{team_label(team_a)} win</div>'
            f'<div class="prediction-kpi-value">{prediction.team_a_win:.1%}</div></div>'
            f'<div class="prediction-kpi"><div class="prediction-kpi-label">Draw</div>'
            f'<div class="prediction-kpi-value">{prediction.draw:.1%}</div></div>'
            f'<div class="prediction-kpi{team_b_highlight}"><div class="prediction-kpi-label">{team_label(team_b)} win</div>'
            f'<div class="prediction-kpi-value">{prediction.team_b_win:.1%}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="explanation-title">Why the model thinks this</div>', unsafe_allow_html=True)
        explanation = prediction.explanation.copy()
        explanation["impact"] = explanation["impact"].map(lambda value: f"{value:+.2f}")
        st.markdown(
            explanation.to_html(index=False, classes="explanation-table", border=0, justify="center"),
            unsafe_allow_html=True,
        )

with tab_tournament:
    st.markdown(
        '<div class="prediction-title">Monte Carlo Tournament Simulator</div>'
        '<div class="tournament-summary">'
        'This simulator plays the entire World Cup many times using each team’s match probabilities. '
        'The results show how often every team reaches each stage. More simulations give steadier estimates.'
        '</div>'
        '<div class="control-guide">'
        '<div class="control-guide-card"><div class="control-guide-title">1. Choose simulations</div>'
        '<div class="control-guide-copy">Enter how many complete World Cups to test. For example, 1,000 means the simulator plays the tournament 1,000 times. Start with 100 for speed.</div></div>'
        '<div class="control-guide-card"><div class="control-guide-title">2. Choose a random seed</div>'
        '<div class="control-guide-copy">A random seed is a replay code for the simulation. The same seed gives the same results; changing it creates a new set of possible results. It does not make a team stronger.</div></div>'
        '<div class="control-guide-card"><div class="control-guide-title">3. Choose a stage</div>'
        '<div class="control-guide-copy">Choose the milestone to compare, such as reaching the final or winning the World Cup. A 25% result means the team achieved it in about 25 of every 100 runs.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    simulations = col_a.number_input(
        "Number of simulations",
        min_value=100,
        max_value=2000,
        value=100,
        step=100,
        help="The number of complete tournament runs. Higher values are steadier but take longer.",
    )
    seed = col_b.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
        help="Think of this as a replay code. The same number recreates the same simulation; another number creates a fresh one.",
    )
    stage_to_chart = col_c.selectbox(
        "Result to explore",
        ["champion", "final", "semifinal", "quarterfinal", "round_of_16", "round_of_32"],
        format_func=lambda value: "Win the World Cup" if value == "champion" else f"Reach the {value.replace('_', ' ').title()}",
        help="Choose the tournament milestone shown in the comparison chart.",
    )

    results = run_simulations(simulations, seed)

    leader = results.iloc[0]
    metric_grid(
        [
            ("Top Champion", team_label(str(leader["team"]))),
            ("Title Chance", f"{leader['champion']:.1%}"),
            ("Simulations", f"{simulations:,}"),
            ("Seed", f"{seed:,}"),
        ]
    )

    if stage_to_chart == "champion":
        chart_title = "Chance of Winning the World Cup"
        chart_description = (
            "This graph compares how often each team won the World Cup across all the simulated tournaments. "
            "A taller bar means the team won more often."
        )
    else:
        stage_name = stage_to_chart.replace("_", " ").title()
        chart_title = f"Chance of Reaching the {stage_name}"
        chart_description = (
            f"This graph compares how often each team reached the {stage_name} across all the simulated tournaments. "
            "A taller bar means the team reached this stage more often."
        )
    st.markdown(
        f'<div class="prediction-title">{chart_title}</div>'
        f'<div class="tournament-summary">{chart_description}</div>',
        unsafe_allow_html=True,
    )
    _, chart_control, _ = st.columns([1.4, 1, 1.4])
    top_n = chart_control.number_input(
        "Teams to show",
        min_value=6,
        max_value=24,
        value=12,
        step=2,
        help="This changes only how many teams appear in the chart, not the simulation results.",
    )
    top_chances = results.sort_values(stage_to_chart, ascending=False).head(top_n)[["team", stage_to_chart]].copy()
    top_chances["team"] = top_chances["team"].map(team_label)
    st.bar_chart(top_chances, x="team", y=stage_to_chart, height=360)

    tournament_table = results.assign(
        team=lambda df: df["team"].map(team_label),
        round_of_32=lambda df: df["round_of_32"].map("{:.1%}".format),
        round_of_16=lambda df: df["round_of_16"].map("{:.1%}".format),
        quarterfinal=lambda df: df["quarterfinal"].map("{:.1%}".format),
        semifinal=lambda df: df["semifinal"].map("{:.1%}".format),
        final=lambda df: df["final"].map("{:.1%}".format),
        champion=lambda df: df["champion"].map("{:.1%}".format),
    ).rename(
        columns={
            "team": "Team",
            "round_of_32": "Reach Round of 32",
            "round_of_16": "Reach Round of 16",
            "quarterfinal": "Reach Quarter-final",
            "semifinal": "Reach Semi-final",
            "final": "Reach Final",
            "champion": "Win World Cup",
        }
    )
    centered_table(tournament_table)

    with st.expander("One example: group-stage standings"):
        st.markdown(
            '<div class="tournament-summary">This is one possible version of the group stage from the selected random seed. '
            'It shows how the teams could finish in their groups. It is an example, not the average prediction.</div>',
            unsafe_allow_html=True,
        )
        example_standings = simulate_group_stage(teams, np.random.default_rng(seed))
        st.markdown(
            example_standings.to_html(index=False, classes="explanation-table", border=0, justify="center"),
            unsafe_allow_html=True,
        )

    with st.expander("One example: Round of 32 matches"):
        st.markdown(
            '<div class="tournament-summary">These are the first knockout matches created from the example group standings above. '
            'The teams change when the random seed changes.</div>',
            unsafe_allow_html=True,
        )
        example_qualifiers = qualified_teams(example_standings)
        bracket = build_round_of_32(example_qualifiers)
        pairings = pd.DataFrame(
            [
                {"match": index // 2 + 1, "team_a": team_label(bracket[index]), "team_b": team_label(bracket[index + 1])}
                for index in range(0, len(bracket), 2)
            ]
        )
        pairings = pairings.rename(columns={"match": "Match", "team_a": "Team A", "team_b": "Team B"})
        st.markdown(
            pairings.to_html(index=False, classes="explanation-table", border=0, justify="center"),
            unsafe_allow_html=True,
        )

with tab_performance:
    metrics = get_model_metrics()
    calibration = get_calibration_curve()
    feature_importance = get_feature_importance()
    report = get_model_report()

    if not metrics:
        st.warning("Train the model with `python3 src/train_model.py` to generate performance metrics.")
    else:
        st.markdown(
            '<div class="prediction-title">Model Performance</div>'
            '<div class="tournament-summary">These results show how well the prediction model performed on matches it had not seen before. '
            'Higher accuracy is better; lower error scores are better.</div>',
            unsafe_allow_html=True,
        )
        performance_metric_grid(
            [
                ("Model", str(metrics["selected_model"]), "The prediction method that performed best during testing."),
                ("Accuracy", f"{metrics['accuracy']:.1%}", "The share of test matches where the correct result was picked. Higher is better."),
                ("Log Loss", f"{metrics['log_loss']:.3f}", "An error score that strongly punishes confident mistakes. Lower is better."),
                ("Brier Score", f"{metrics['brier_score']:.3f}", "How close the predicted chances were to what happened. Lower is better."),
            ]
        )
        performance_metric_grid(
            [
                ("RPS", f"{metrics['ranked_probability_score']:.3f}", "An overall error score for win, draw, and loss chances. Lower is better."),
                ("Confidence Gap", f"{metrics['expected_calibration_error']:.2%}", "The gap between the model’s confidence and real results. Smaller is better."),
                ("Training Matches", f"{metrics['training_rows']:,}", "Past matches the model used to learn patterns."),
                ("Test Matches", f"{metrics['test_rows']:,}", "Separate matches used to check the model fairly after training."),
            ]
        )

        st.markdown(
            '<div class="prediction-title">Dataset Used</div>'
            '<div class="dataset-note">'
            'The project uses the <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017" target="_blank">'
            'International Football Results dataset by Mart Jürisoo on Kaggle</a>. It contains men’s international match results, '
            'including the teams, score, date, competition, location, and whether the match was played at a neutral venue. '
            'The model learns from completed matches dated 1990 onward. Matches from 2022 onward are kept separate to test it on newer games it did not train on. '
            'Elo, form, attack, and defence ratings are calculated from these match results rather than copied from a separate ratings dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        comparison = pd.DataFrame(metrics["model_comparison"])
        st.markdown(
            '<div class="prediction-title">Model Comparison</div>'
            '<div class="tournament-summary">This table compares the models that were tested. '
            'Look for higher accuracy and lower Log Loss, Brier Score, RPS, and Confidence Gap.</div>',
            unsafe_allow_html=True,
        )
        comparison_table = comparison.assign(
            accuracy=lambda df: df["accuracy"].map("{:.1%}".format),
            log_loss=lambda df: df["log_loss"].map("{:.3f}".format),
            brier_score=lambda df: df["brier_score"].map("{:.3f}".format),
            rps=lambda df: df["rps"].map("{:.3f}".format),
            ece=lambda df: df["ece"].map("{:.2%}".format),
            draw_precision=lambda df: df["draw_precision"].map("{:.1%}".format),
            draw_recall=lambda df: df["draw_recall"].map("{:.1%}".format),
        ).rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "log_loss": "Log Loss",
                "brier_score": "Brier Score",
                "rps": "RPS",
                "ece": "Confidence Gap",
                "draw_precision": "Draw Precision",
                "draw_recall": "Draw Recall",
            }
        )
        centered_table(comparison_table)

        if not calibration.empty:
            st.markdown(
                '<div class="prediction-title">Predicted Chances Compared with Real Results</div>'
                '<div class="tournament-summary">This graph checks whether the model’s confidence matches reality. '
                'When the predicted and actual lines stay close together, the probabilities are trustworthy.</div>',
                unsafe_allow_html=True,
            )
            chart_data = calibration[
                ["mean_predicted_probability", "actual_rate"]
            ].rename(
                columns={
                    "mean_predicted_probability": "Predicted",
                    "actual_rate": "Actual",
                }
            )
            st.line_chart(chart_data, height=320)
            calibration_table = calibration.assign(
                mean_predicted_probability=lambda df: df["mean_predicted_probability"].map("{:.2%}".format),
                actual_rate=lambda df: df["actual_rate"].map("{:.2%}".format),
            ).rename(
                columns={
                    "mean_predicted_probability": "Predicted Chance",
                    "actual_rate": "What Actually Happened",
                    "count": "Matches in This Group",
                }
            )
            centered_table(calibration_table)

        if not feature_importance.empty:
            st.markdown(
                '<div class="prediction-title">What Influences the Predictions</div>'
                '<div class="tournament-summary">This graph shows which pieces of team information affect the model most. '
                'A longer bar means the model relies more on that factor; it does not mean the factor helps a team win.</div>',
                unsafe_allow_html=True,
            )
            top_features = feature_importance.head(15).copy()
            top_features["feature"] = top_features["feature"].map(lambda value: value.replace("_", " ").title())
            st.bar_chart(top_features, x="feature", y="importance_mean", height=360)
            feature_table = top_features.assign(
                importance_mean=lambda df: df["importance_mean"].map("{:.5f}".format),
                importance_std=lambda df: df["importance_std"].map("{:.5f}".format),
            ).rename(
                columns={
                    "feature": "Team Information",
                    "importance_mean": "Average Influence",
                    "importance_std": "How Much It Varies",
                }
            )
            centered_table(feature_table)

        with st.expander("Technical details for advanced users"):
            st.markdown(
                '<div class="tournament-summary">This is the full technical training report. '
                'Most users can rely on the simpler summaries above.</div>',
                unsafe_allow_html=True,
            )
            st.code(report)
