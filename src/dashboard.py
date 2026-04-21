"""
dashboard.py

Streamlit F1 Predictor dashboard.

Deployment modes:
  LOCAL  — full pipeline control (generate predictions, evaluate, retrain)
  CLOUD  — read-only view of pre-built predictions and eval history.
           Update by running the pipeline locally and pushing to GitHub.

Run locally:
    streamlit run src/dashboard.py

Deploy:
    Push to GitHub → Streamlit Community Cloud → set main file = src/dashboard.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Deployment detection ───────────────────────────────────────────────────────
# Running on Streamlit Cloud when STREAMLIT_SHARING_MODE or the Streamlit
# Cloud environment variable is set.  We also detect it by checking whether
# the pipeline scripts can actually be executed (they can't on the cloud tier
# because the working directory and environment are read-only).

_IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") == "1"
    or os.environ.get("HOME", "").startswith("/home/appuser")
    or not Path("src/predict.py").exists()
)

# ── Path setup ────────────────────────────────────────────────────────────────
# Resolve paths relative to this file so they work whether the app is run
# from the project root (locally) or from any directory (Streamlit Cloud).

_SRC_DIR     = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SRC_DIR.parent.resolve()

CACHE_DIR       = _PROJECT_ROOT / "cache"
PREDICTIONS_DIR = _PROJECT_ROOT / "predictions"
MODELS_DIR      = _PROJECT_ROOT / "models"
PROCESSED_DIR   = _PROJECT_ROOT / "data" / "processed"
EVAL_HISTORY    = MODELS_DIR / "eval_history.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Predictor",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<div class='scanline-overlay'></div>", unsafe_allow_html=True)
st.markdown("<div class='hud-ribbon'></div>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Outfit:wght@200;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-deep: #0A0A0B;
    --kinetic-cyan: #00F2FF;
    --redline: #FF2D55;
    --carbon-gray: #1C1C1E;
    --text-primary: #F0F0F0;
    --text-muted: #808080;
    --glass-bg: rgba(28, 28, 30, 0.75);
    --border-dim: rgba(255, 255, 255, 0.05);
}

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background-color: var(--bg-deep); color: var(--text-primary); }
section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid var(--border-dim); }

/* Typography */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important; font-size: 2.8rem !important;
    letter-spacing: 0.15em !important; color: var(--kinetic-cyan) !important;
    text-transform: uppercase; margin-bottom: 0 !important;
}
h2 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.4rem !important; letter-spacing: 0.1em !important;
    color: #fff !important; text-transform: uppercase;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--carbon-gray);
}
h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; color: var(--text-muted) !important;
    letter-spacing: 0.15em; text-transform: uppercase;
}

/* Metrics & Radar */
[data-testid="metric-container"] {
    background: var(--glass-bg); backdrop-filter: blur(8px);
    border: 1px solid var(--border-dim); border-left: 3px solid var(--kinetic-cyan);
    border-radius: 2px; padding: 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.65rem !important;
    color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important; font-weight: 700 !important;
    font-size: 1.8rem !important; color: #fff !important;
}

/* Indicators */
.tactical-border {
    border: 1px solid var(--carbon-gray);
    position: relative; overflow: hidden;
}
.tactical-border::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 10px; height: 10px; border-top: 2px solid var(--kinetic-cyan); border-left: 2px solid var(--kinetic-cyan);
}
.tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    color: var(--kinetic-cyan); letter-spacing: 0.1em; border: 1px solid rgba(0,242,255,0.2);
    padding: 2px 6px; border-radius: 3px; display: inline-block;
}
.redline-tag { color: var(--redline); border-color: rgba(255,45,85,0.2); }

/* Components */
.stButton > button {
    background: transparent !important; color: var(--kinetic-cyan) !important;
    border: 1px solid var(--kinetic-cyan) !important;
    font-family: 'Orbitron', sans-serif !important; font-size: 0.8rem !important;
    border-radius: 0 !important; width: 100% !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    background: var(--kinetic-cyan) !important; color: #000 !important;
    box-shadow: 0 0 15px rgba(0, 242, 255, 0.4) !important;
}

.stDataFrame { font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important; }
.cloud-notice {
    background: rgba(255, 45, 85, 0.05); border: 1px solid rgba(255, 45, 85, 0.2);
    padding: 0.8rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--redline); line-height: 1.4;
}

/* HUD Overlay Elements */
.hud-ribbon {
    height: 4px; background: linear-gradient(90deg, var(--kinetic-cyan), var(--redline), transparent);
    margin-bottom: 2rem;
}

@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}
.scanline-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
                linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
    z-index: 10; pointer-events: none; opacity: 0.15;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_eval_history() -> list:
    if EVAL_HISTORY.exists():
        with open(EVAL_HISTORY) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=300)
def load_prediction(year: int, round_number: int) -> dict | None:
    path = PREDICTIONS_DIR / f"{year}_R{round_number:02d}_prediction.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def get_season_schedule(year: int) -> pd.DataFrame | None:
    try:
        return fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return None


def run_script(script: str, args: list[str]) -> tuple[bool, str, str]:
    """Run a pipeline script. Always use project-root as working directory."""
    r = subprocess.run(
        [sys.executable, str(_SRC_DIR / script)] + args,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    return r.returncode == 0, r.stdout, r.stderr


def team_colour(team: str) -> str:
    colours = {
        "Red Bull Racing":   "#3671C6",
        "Ferrari":           "#E8002D",
        "McLaren":           "#FF8000",
        "Mercedes":          "#27F4D2",
        "Aston Martin":      "#229971",
        "Alpine":            "#FF87BC",
        "Williams":          "#64C4FF",
        "RB":                "#6692FF",
        "Audi":              "#52E252",
        "Haas F1 Team":      "#B6BABD",
        "Cadillac":          "#C8102E",
        # Historical
        "AlphaTauri":        "#6692FF",
        "Toro Rosso":        "#6692FF",
        "Racing Bulls":      "#6692FF",
        "Racing Point":      "#F596C8",
        "Force India":       "#F596C8",
        "Renault":           "#FFF500",
        "Alfa Romeo":        "#C92D4B",
        "Sauber":            "#52E252",
        "Kick Sauber":       "#52E252",
    }
    return colours.get(team, "#555555")


def _plotly_layout(height: int = 400) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        xaxis=dict(
            gridcolor="#1C1C1E",
            zerolinecolor="#333",
            tickfont=dict(color="#808080", family="JetBrains Mono", size=9),
        ),
        yaxis=dict(
            gridcolor="#1C1C1E",
            zerolinecolor="#333",
            tickfont=dict(color="#808080", family="JetBrains Mono", size=9),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            font=dict(color="#808080", family="JetBrains Mono", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def _hex_to_rgba(c: str, alpha: float = 0.12) -> str:
    """Convert any color string to rgba() — Plotly rejects CSS hex+alpha (#rrggbbaa)."""
    if c.startswith("rgba"):
        return c
    if c.startswith("rgb("):
        return c.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    if c.startswith("#") and len(c) in (7, 9):
        r2, g2, b2 = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f"rgba({r2},{g2},{b2},{alpha})"
    return c


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1>GP-01</h1>", unsafe_allow_html=True)
    st.markdown("<h3>TACTICAL PREDICTOR</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height:2px; background:var(--kinetic-cyan); margin:1rem 0 2rem 0; opacity:0.3'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏁  Pre-Race", "📊  Post-Race", "📈  Season", "🔍  Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div class='tag'>Models</div>", unsafe_allow_html=True)
    for mname, label in [("quali_model", "Quali"), ("race_model", "Race")]:
        path = MODELS_DIR / f"{mname}_metrics.json"
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            if label == "Quali":
                st.markdown(
                    f"<div class='tag'>{label} — MAE {m.get('MAE_positions', '—')} "
                    f"| ρ {m.get('Spearman_rho', '—')}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='tag'>{label} — Brier {m.get('Brier_score', '—')} "
                    f"| Top-3 {m.get('Top3_avg_overlap', '—')}/3</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    if _IS_CLOUD:
        st.markdown(
            "<div class='cloud-notice'>☁️ Cloud mode — pipeline runs locally.<br>"
            "Push updated files to GitHub to refresh predictions.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='tag'>Retrain models</div>", unsafe_allow_html=True)
        if st.button("Run features + train"):
            with st.spinner("Running features.py..."):
                ok1, _, err1 = run_script("features.py", [])
            with st.spinner("Running train.py..."):
                ok2, _, err2 = run_script("train.py", [])
            if ok1 and ok2:
                st.success("Models retrained.")
                st.cache_data.clear()
            else:
                st.error("Retrain failed.")
                if not ok1:
                    st.code(err1[-2000:], language="text")
                if not ok2:
                    st.code(err2[-2000:], language="text")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _round_selector(year: int, key_suffix: str = "") -> int:
    schedule = get_season_schedule(year)
    if schedule is not None:
        options = {
            f"R{int(r['RoundNumber'])} — {r['EventName']}": int(r["RoundNumber"])
            for _, r in schedule.iterrows()
        }
        label = st.selectbox("Round", list(options.keys()),
                             key=f"round_{year}_{key_suffix}")
        return options[label]
    return st.number_input("Round number", min_value=1, max_value=24, value=1,
                           key=f"round_num_{year}_{key_suffix}")


def _prediction_table_html(df: pd.DataFrame) -> str:
    html = (
        "<table style='width:100%;border-collapse:collapse;"
        "font-family:JetBrains Mono,monospace;font-size:0.75rem'>"
        "<tr style='color:var(--text-muted);border-bottom:1px solid var(--carbon-gray)'>"
        "<th style='text-align:left;padding:8px 4px;font-size:0.6rem;letter-spacing:0.1em'>RANK</th>"
        "<th style='text-align:left;padding:8px 4px;font-size:0.6rem;letter-spacing:0.1em'>GRID</th>"
        "<th style='text-align:left;padding:8px 4px;font-size:0.6rem;letter-spacing:0.1em'>DRIVER</th>"
        "<th style='text-align:left;padding:8px 4px;font-size:0.6rem;letter-spacing:0.1em'>TEAM</th>"
        "<th style='text-align:right;padding:8px 4px;font-size:0.6rem;letter-spacing:0.1em'>PODIUM%</th></tr>"
    )
    for _, row in df.iterrows():
        rank    = int(row["PredictedRaceRank"])
        grid    = int(row["PredictedQualiPos"])
        prob    = row["PodiumProbability"] * 100
        color   = team_colour(row["TeamName"])
        is_top3 = rank <= 3
        rank_style = (
            "color:var(--kinetic-cyan);font-family:Orbitron,sans-serif;font-weight:700;font-size:1.0rem"
            if is_top3
            else "color:var(--text-muted);font-family:Orbitron,sans-serif;font-size:1.0rem"
        )
        html += (
            f"<tr style='border-bottom:1px solid var(--border-dim);"
            f"{'background:rgba(0,242,255,0.03)' if is_top3 else ''}'>"
            f"<td style='padding:10px 4px;{rank_style}'>P{rank}</td>"
            f"<td style='padding:10px 4px;color:var(--text-muted)'>P{grid}</td>"
            f"<td style='padding:10px 4px;color:#fff;font-family:Outfit,sans-serif;font-weight:600'>{row['Driver']}</td>"
            f"<td style='padding:10px 4px'>"
            f"<span style='color:{color};font-size:0.65rem;text-transform:uppercase'>{row['TeamName']}</span></td>"
            f"<td style='padding:10px 4px;text-align:right;"
            f"color:{'var(--kinetic-cyan)' if is_top3 else 'var(--text-muted)'}'>{prob:.1f}%</td></tr>"
        )
    return html + "</table>"


# ── Page 1: Pre-Race ──────────────────────────────────────────────────────────

if "Pre-Race" in page:
    st.markdown("<h1>P-RACE.26</h1>", unsafe_allow_html=True)
    st.markdown("<h3>PRE-RACE INTELLIGENCE & PROBABILITY</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        year         = st.selectbox("Season", [2026, 2025, 2024], index=0)
        round_number = _round_selector(year, "prerace")

        if _IS_CLOUD:
            st.markdown(
                "<div class='cloud-notice'>☁️ Predictions updated on saturday after FP2.</div>",
                unsafe_allow_html=True,
    )
        else:
            run_btn = st.button("Generate Prediction")

    with col2:
        pred = load_prediction(year, round_number)

        if not _IS_CLOUD:
            if run_btn:
                with st.spinner(f"Predicting {year} R{round_number}..."):
                    ok, _, err = run_script(
                        "predict.py",
                        ["--year", str(year), "--round", str(round_number)],
                    )
                if ok:
                    st.cache_data.clear()
                    pred = load_prediction(year, round_number)
                    st.success("Prediction generated.")
                else:
                    st.error("Prediction failed.")
                    st.code((err or "")[-2000:], language="text")

        if pred is None:
            if _IS_CLOUD:
                st.info("No prediction for this round yet. Run the pipeline locally and push to GitHub.")
            else:
                st.info("No prediction yet. Click **Generate Prediction**.")
        else:
            st.markdown(f"## {pred['event']} {pred['year']} · Round {pred['round']}")
            st.markdown(f"<div class='tag'>{pred['circuit']}</div><br>",
                        unsafe_allow_html=True)

            df = pd.DataFrame(pred["predictions"])

            # Podium probability bar chart
            fig = go.Figure()
            for _, row in df.head(10).iterrows():
                fig.add_trace(go.Bar(
                    x=[row["PodiumProbability"] * 100],
                    y=[row["Driver"]],
                    orientation="h",
                    marker_color=team_colour(row["TeamName"]),
                    showlegend=False,
                    text=f"{row['PodiumProbability']*100:.1f}%",
                    textposition="outside",
                    textfont=dict(family="Space Mono", size=11, color="#888"),
                ))
            layout = _plotly_layout(340)
            layout["xaxis"] = dict(
                showgrid=False,
                showticklabels=False,
                range=[0, min(df["PodiumProbability"].max() * 120, 115)],
            )
            layout["yaxis"] = dict(
                autorange="reversed",
                tickfont=dict(family="Space Mono", size=11, color="#888"),
                gridcolor="#111",
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Full Grid Prediction")
            st.markdown(_prediction_table_html(df), unsafe_allow_html=True)


# ── Page 2: Post-Race ─────────────────────────────────────────────────────────

elif "Post-Race" in page:
    st.markdown("<h1>POST-RACE</h1>", unsafe_allow_html=True)
    st.markdown("<h3>RETROSPECTIVE ANALYSIS & MODEL ACCURACY</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        year         = st.selectbox("Season", [2026, 2025, 2024], index=0)
        round_number = _round_selector(year, "postrace")

        if _IS_CLOUD:
            st.markdown(
                "<div class='cloud-notice'>Run <code>python src/evaluate.py "
                "--year {year} --round {round_number}</code> locally "
                "then push to GitHub.</div>".format(
                    year=year, round_number=round_number
                ),
                unsafe_allow_html=True,
            )
        else:
            eval_btn = st.button("Evaluate Round")

    with col2:
        if not _IS_CLOUD and eval_btn:
            if load_prediction(year, round_number) is None:
                with st.spinner("Generating prediction first..."):
                    ok, _, err = run_script(
                        "predict.py",
                        ["--year", str(year), "--round", str(round_number)],
                    )
                if not ok:
                    st.error("Could not generate prediction.")
                    st.code(err[-2000:], language="text")
                    st.stop()

            with st.spinner("Evaluating..."):
                ok, _, err = run_script(
                    "evaluate.py",
                    ["--year", str(year), "--round", str(round_number)],
                )
            if ok:
                st.cache_data.clear()
                st.success("Evaluation complete.")
            else:
                st.error("Evaluation failed.")
                st.code(err[-2000:], language="text")

        history = load_eval_history()
        key     = f"{year}_R{round_number:02d}"
        entry   = next((h for h in history if h.get("key") == key), None)

        if entry is None:
            if _IS_CLOUD:
                st.info("No evaluation for this round yet. Run locally and push to GitHub.")
            else:
                st.info("No evaluation yet. Click **Evaluate Round**.")
        else:
            q = entry.get("quali", {})
            r = entry.get("race",  {})
            st.markdown(f"## {entry['event']} {entry['year']} · Round {entry['round']}")
            st.markdown(f"<div class='tag'>{entry.get('circuit', '')}</div><br>",
                        unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE (positions)",
                      f"{q['MAE_positions']:.2f}" if q.get("MAE_positions") else "—")
            m2.metric("Spearman ρ",
                      f"{q['Spearman_rho']:.3f}"  if q.get("Spearman_rho")  else "—")
            m3.metric("Brier score",
                      f"{r['Brier_score']:.4f}"   if r.get("Brier_score")   else "—")
            m4.metric("Top-3 overlap",
                      f"{r['Top3_overlap']}/3"     if r.get("Top3_overlap") is not None else "—")

            st.markdown("---")
            pred = load_prediction(year, round_number)
            if pred:
                st.markdown("### Driver breakdown")
                df = pd.DataFrame(pred["predictions"])
                try:
                    actual_s = fastf1.get_session(year, round_number, "R")
                    actual_s.load(telemetry=False, weather=False, messages=False)
                    res = actual_s.results.copy()
                    if "Abbreviation" in res.columns:
                        res = res.rename(columns={"Abbreviation": "Driver"})
                    res["ActualPos"] = pd.to_numeric(res["Position"], errors="coerce")
                    res["Podium"]    = (res["ActualPos"] <= 3).astype(int)
                    df = df.merge(res[["Driver", "ActualPos", "Podium"]],
                                  on="Driver", how="left")
                except Exception:
                    df["ActualPos"] = None
                    df["Podium"]    = None

                tbl = (
                    "<table style='width:100%;border-collapse:collapse;"
                    "font-family:Space Mono,monospace;font-size:0.78rem'>"
                    "<tr style='color:#444;border-bottom:1px solid #1a1a1a'>"
                    + "".join(
                        f"<th style='text-align:left;padding:6px 4px'>{c}</th>"
                        for c in ["Driver", "P.Grid", "A.Pos", "Delta", "Podium%", "Result"]
                    )
                    + "</tr>"
                )
                for _, row in df.sort_values("PredictedQualiPos").iterrows():
                    pgrid  = int(row["PredictedQualiPos"])
                    apos   = int(row["ActualPos"]) if pd.notna(row.get("ActualPos")) else "—"
                    delta  = f"{pgrid - int(apos):+d}" if apos != "—" else "—"
                    prob   = f"{row['PodiumProbability']*100:.1f}%"
                    is_pod = row.get("Podium") == 1
                    dc = (
                        "#00d4ff" if (delta != "—" and int(delta) <= 0)
                        else "#ff4444" if delta != "—"
                        else "#555"
                    )
                    tbl += (
                        f"<tr style='border-bottom:1px solid #111;"
                        f"{'background:#0a1a1a' if is_pod else ''}'>"
                        f"<td style='padding:7px 4px;color:#ddd'>{row['Driver']}</td>"
                        f"<td style='padding:7px 4px;color:#555'>P{pgrid}</td>"
                        f"<td style='padding:7px 4px;color:#555'>"
                        f"{'P'+str(apos) if apos != '—' else '—'}</td>"
                        f"<td style='padding:7px 4px;color:{dc}'>{delta}</td>"
                        f"<td style='padding:7px 4px;"
                        f"color:{'#00d4ff' if is_pod else '#555'}'>{prob}</td>"
                        f"<td style='padding:7px 4px'>"
                        f"{'<span class=\"podium-badge\">PODIUM</span>' if is_pod else ''}"
                        f"</td></tr>"
                    )
                st.markdown(tbl + "</table>", unsafe_allow_html=True)


# ── Page 3: Season ────────────────────────────────────────────────────────────

elif "Season" in page:
    st.markdown("<h1>SEASON</h1>", unsafe_allow_html=True)
    st.markdown("<h3>CHAMPIONSHIP STANDINGS & PROGRESSION</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem'></div>", unsafe_allow_html=True)

    history = load_eval_history()
    if not history:
        st.info("No evaluation history yet.")
    else:
        years_available = sorted(set(h["year"] for h in history), reverse=True)
        selected_year   = st.selectbox("Season", years_available)
        season_history  = [h for h in history if h["year"] == selected_year]

        if not season_history:
            st.warning(f"No evaluations for {selected_year} yet.")
        else:
            maes     = [h["quali"]["MAE_positions"] for h in season_history
                        if h["quali"].get("MAE_positions") is not None]
            rhos     = [h["quali"]["Spearman_rho"]  for h in season_history
                        if h["quali"].get("Spearman_rho")  is not None]
            briers   = [h["race"]["Brier_score"]    for h in season_history
                        if h["race"].get("Brier_score")    is not None]
            overlaps = [h["race"]["Top3_overlap"]   for h in season_history
                        if h["race"].get("Top3_overlap")   is not None]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg MAE (pos)",
                      f"{sum(maes)/len(maes):.2f}"           if maes     else "—")
            c2.metric("Avg Spearman ρ",
                      f"{sum(rhos)/len(rhos):.3f}"           if rhos     else "—")
            c3.metric("Avg Brier",
                      f"{sum(briers)/len(briers):.4f}"       if briers   else "—")
            c4.metric("Avg Top-3",
                      f"{sum(overlaps)/len(overlaps):.2f}/3" if overlaps else "—")

            st.markdown("---")
            events  = [h["event"][:18]                for h in season_history]
            rho_v   = [h["quali"].get("Spearman_rho") for h in season_history]
            brier_v = [h["race"].get("Brier_score")   for h in season_history]
            top3_v  = [h["race"].get("Top3_overlap")  for h in season_history]
            chaos_v = [h.get("is_chaos", False)        for h in season_history]

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### Quali accuracy (Spearman ρ)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=events, y=rho_v, mode="lines+markers",
                    line=dict(color="#00d4ff", width=2),
                    marker=dict(size=7, color="#00d4ff"),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                ))
                fig.add_hline(y=0.72, line_dash="dot", line_color="#333",
                              annotation_text="Baseline 0.72",
                              annotation_font_color="#444")
                layout = _plotly_layout(260)
                layout["yaxis"] = dict(
                    range=[0, 1.05], gridcolor="#111",
                    tickfont=dict(family="Space Mono", size=10, color="#555"),
                )
                layout["xaxis"] = dict(
                    gridcolor="#111",
                    tickfont=dict(family="Space Mono", size=9, color="#555"),
                    tickangle=-30,
                )
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("### Race accuracy (Brier score)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=events, y=brier_v, mode="lines+markers",
                    line=dict(color="#ff6b35", width=2),
                    marker=dict(size=7, color="#ff6b35"),
                    fill="tozeroy", fillcolor="rgba(255,107,53,0.05)",
                ))
                fig.add_hline(y=0.0873, line_dash="dot", line_color="#333",
                              annotation_text="Baseline 0.087",
                              annotation_font_color="#444")
                layout = _plotly_layout(260)
                layout["yaxis"] = dict(
                    gridcolor="#111",
                    tickfont=dict(family="Space Mono", size=10, color="#555"),
                )
                layout["xaxis"] = dict(
                    gridcolor="#111",
                    tickfont=dict(family="Space Mono", size=9, color="#555"),
                    tickangle=-30,
                )
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top-3 podium overlap per race")
            st.markdown(
                "<div class='tag'>Grey bars = high SC rate circuits "
                "(chaos races — excluded from trend line)</div><br>",
                unsafe_allow_html=True,
            )
            bar_colors = []
            for v, is_chaos in zip(top3_v or [], chaos_v):
                if is_chaos:
                    bar_colors.append("rgba(100,100,100,0.4)")
                elif v == 3:
                    bar_colors.append("rgba(0,212,255,1.0)")
                elif v == 2:
                    bar_colors.append("rgba(0,212,255,0.55)")
                else:
                    bar_colors.append("rgba(0,212,255,0.2)")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=events, y=top3_v,
                marker_color=bar_colors,
                text=[f"{v}/3" for v in (top3_v or [])],
                textposition="outside",
                textfont=dict(family="Space Mono", size=10, color="#555"),
            ))
            fig.add_hline(y=1.83, line_dash="dot", line_color="#333",
                          annotation_text="Baseline 1.83",
                          annotation_font_color="#444")
            for i, (ev, is_c) in enumerate(zip(events, chaos_v)):
                if is_c:
                    fig.add_annotation(
                        x=ev, y=0.15, text="SC", showarrow=False,
                        font=dict(size=8, color="#666", family="Space Mono"),
                    )
            layout = _plotly_layout(220)
            layout["yaxis"] = dict(
                range=[0, 3.5], gridcolor="#111",
                tickfont=dict(family="Space Mono", size=10, color="#555"),
            )
            layout["xaxis"] = dict(
                gridcolor="#111",
                tickfont=dict(family="Space Mono", size=9, color="#555"),
                tickangle=-30,
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Round-by-round breakdown")
            rows = []
            for h in season_history:
                q  = h.get("quali", {})
                rc = h.get("race",  {})
                rows.append({
                    "Round":      f"R{h['round']}",
                    "Event":      h["event"][:24],
                    "Circuit":    h.get("circuit", ""),
                    "SC Race":    "SC" if h.get("is_chaos") else "",
                    "MAE pos":    q.get("MAE_positions"),
                    "Spearman ρ": q.get("Spearman_rho"),
                    "Brier":      rc.get("Brier_score"),
                    "Top-3":      f"{rc['Top3_overlap']}/3"
                                  if rc.get("Top3_overlap") is not None else "—",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAE pos":    st.column_config.NumberColumn(format="%.2f"),
                    "Spearman ρ": st.column_config.NumberColumn(format="%.3f"),
                    "Brier":      st.column_config.NumberColumn(format="%.4f"),
                },
            )

            if not _IS_CLOUD:
                st.markdown("---")
                st.markdown("<div class='tag'>Batch evaluate all completed rounds</div>",
                            unsafe_allow_html=True)
                if st.button(f"Batch evaluate {selected_year}"):
                    with st.spinner(f"Running batch_evaluate.py for {selected_year}..."):
                        ok, _, err = run_script("batch_evaluate.py",
                                                ["--year", str(selected_year)])
                    if ok:
                        st.cache_data.clear()
                        st.success("Batch evaluation complete.")
                    else:
                        st.error("Batch evaluation failed.")
                        st.code(err[-3000:], language="text")


# ── Page 4: Driver Analysis ────────────────────────────────────────────────────

elif "Analysis" in page:
    with st.sidebar:
        st.markdown("<h1>GP-01</h1>", unsafe_allow_html=True)
        st.markdown("<h3>TACTICAL PREDICTOR</h3>", unsafe_allow_html=True)
        st.markdown("<div style='height:2px; background:var(--kinetic-cyan); margin:1rem 0 2rem 0; opacity:0.3'></div>", unsafe_allow_html=True)

    year_sel  = st.selectbox("Season", [2026, 2025, 2024], index=0,
                             key="analysis_year")
    round_sel = _round_selector(year_sel, "analysis")
    pred      = load_prediction(year_sel, round_sel)

    if pred is None:
        st.info("No prediction found for this round. "
                "Generate one on the Pre-Race page first.")
        st.stop()

    df_pred = pd.DataFrame(pred["predictions"])
    drivers  = sorted(df_pred["Driver"].tolist())

    st.markdown(f"## {pred['event']} {pred['year']} · Round {pred['round']}")
    st.markdown(f"<div class='tag'>{pred['circuit']}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📡 Radar", "⚔️ Head-to-Head", "🔬 SHAP Explainer"])

    quali_path = PROCESSED_DIR / "quali_features.parquet"
    race_path  = PROCESSED_DIR / "race_features.parquet"

    # ── Tab 1: Radar chart ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Driver capability radar")
        st.markdown(
            "<div class='tag'>Five axes: quali pace · race pace · circuit affinity "
            "· recent form · reliability</div><br>",
            unsafe_allow_html=True,
        )
        selected_drivers = st.multiselect(
            "Select drivers to compare (max 5)",
            options=drivers,
            default=drivers[:3],
            max_selections=5,
            key="radar_drivers",
        )

        if not selected_drivers:
            st.info("Select at least one driver.")
        elif not quali_path.exists() or not race_path.exists():
            st.warning("Feature tables not found. Run `python src/features.py` first.")
        else:
            @st.cache_data(ttl=600)
            def _load_features():
                q = pd.read_parquet(str(quali_path))
                r = pd.read_parquet(str(race_path))
                return q, r

            qf, rf = _load_features()

            def _norm(s: pd.Series) -> pd.Series:
                mn, mx = s.min(), s.max()
                return (s - mn) / (mx - mn + 1e-9)

            qf_recent = qf[qf["Year"] >= year_sel - 1]
            rf_recent = rf[rf["Year"] >= year_sel - 1]

            q_pace_norm = 1 - _norm(
                qf_recent.groupby("Driver")["RollingQualiGap"].mean().dropna()
            )
            r_pace_norm = 1 - _norm(
                rf_recent.groupby("Driver")["RollingAvgFinish"].mean().dropna()
            )
            circ_norm = 1 - _norm(
                qf_recent[qf_recent["CircuitShortName"] == pred["circuit"]]
                .groupby("Driver")["CircuitAvgQualiGap"].mean().dropna()
            )
            form_norm = _norm(
                rf_recent.groupby("Driver")["RollingPoints"].mean().dropna()
            )
            rel_norm = 1 - _norm(
                rf_recent.groupby("Driver")["RollingDNFRate"].mean().dropna()
            )

            categories  = ["Quali pace", "Race pace", "Circuit affinity",
                           "Recent form", "Reliability"]
            fig_radar   = go.Figure()
            colours_r   = ["#00F2FF", "#FF2D55", "#FFB800", "#A855F7", "#22C55E"]

            for i, driver in enumerate(selected_drivers):
                vals = [
                    float(q_pace_norm.get(driver, 0.5)),
                    float(r_pace_norm.get(driver, 0.5)),
                    float(circ_norm.get(driver, 0.5)),
                    float(form_norm.get(driver, 0.5)),
                    float(rel_norm.get(driver, 0.5)),
                ]
                vals_closed = vals + [vals[0]]
                cats_closed = categories + [categories[0]]
                col = colours_r[i % len(colours_r)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed,
                    theta=cats_closed,
                    fill="toself",
                    fillcolor=_hex_to_rgba(col, 0.12),
                    line=dict(color=col, width=2),
                    name=driver,
                ))

            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        tickfont=dict(size=9, color="#555", family="JetBrains Mono"),
                        gridcolor="#1C1C1E",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11, family="JetBrains Mono", color="#808080"),
                        gridcolor="#1C1C1E",
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                showlegend=True,
                legend=dict(
                    font=dict(family="Space Mono", size=11, color="#888"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown(
                "<div class='tag'>Axes normalised 0–1 within the field "
                "using last 2 seasons of data.</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 2: Head-to-Head ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Head-to-head comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            driver_a = st.selectbox("Driver A", drivers, index=0, key="h2h_a")
        with col_b:
            driver_b = st.selectbox(
                "Driver B",
                [d for d in drivers if d != driver_a],
                index=0,
                key="h2h_b",
            )

        if not quali_path.exists():
            st.warning("Feature tables not found.")
        else:
            @st.cache_data(ttl=600)
            def _load_h2h_features():
                q = pd.read_parquet(str(quali_path))
                r = pd.read_parquet(str(race_path))
                return q, r

            qf, rf = _load_h2h_features()

            def _driver_stats(driver: str) -> dict:
                qd       = qf[qf["Driver"] == driver]
                rd       = rf[rf["Driver"] == driver]
                recent_q = qd[qd["Year"] >= year_sel - 1]
                recent_r = rd[rd["Year"] >= year_sel - 1]
                circ_q   = qd[qd["CircuitShortName"] == pred["circuit"]]

                pred_row  = df_pred[df_pred["Driver"] == driver]
                podium_pct = (
                    float(pred_row["PodiumProbability"].iloc[0]) * 100
                    if not pred_row.empty else 0.0
                )
                pred_grid = (
                    int(pred_row["PredictedQualiPos"].iloc[0])
                    if not pred_row.empty else "—"
                )

                def _safe_mean(s):
                    v = s.mean()
                    return float(v) if pd.notna(v) else None

                return {
                    "Predicted grid":         pred_grid,
                    "Podium probability":     f"{podium_pct:.1f}%",
                    "Avg quali gap (recent)":
                        f"{_safe_mean(recent_q['RollingQualiGap']):.3f}s"
                        if _safe_mean(recent_q.get("RollingQualiGap", pd.Series(dtype=float))) is not None
                        else "—",
                    "Avg finish pos (recent)":
                        f"{_safe_mean(recent_r['RollingAvgFinish']):.1f}"
                        if not recent_r.empty else "—",
                    "Podium rate (recent)":
                        f"{(_safe_mean(recent_r['RollingPodiumRate']) or 0)*100:.0f}%"
                        if not recent_r.empty else "—",
                    "DNF rate (recent)":
                        f"{(_safe_mean(recent_r['RollingDNFRate']) or 0)*100:.0f}%"
                        if not recent_r.empty else "—",
                    "Circuit visits":
                        f"{int(circ_q['CircuitVisits'].max())}"
                        if not circ_q.empty else "0",
                    "Circuit avg quali gap":
                        f"{circ_q['CircuitAvgQualiGap'].iloc[-1]:.3f}s"
                        if not circ_q.empty else "—",
                    "H2H quali win rate":
                        f"{(_safe_mean(recent_q['H2H_QualiWinRate']) or 0.5)*100:.0f}%"
                        if not recent_q.empty else "—",
                }

            stats_a = _driver_stats(driver_a)
            stats_b = _driver_stats(driver_b)

            table_html = (
                "<table style='width:100%;border-collapse:collapse;"
                "font-family:JetBrains Mono,monospace;font-size:0.72rem'>"
                f"<tr style='border-bottom:2px solid var(--carbon-gray)'>"
                f"<th style='text-align:left;padding:8px 6px;color:var(--text-muted)'>DATASET</th>"
                f"<th style='text-align:center;padding:8px 6px;color:var(--kinetic-cyan)'>{driver_a}</th>"
                f"<th style='text-align:center;padding:8px 6px;color:var(--redline)'>{driver_b}</th>"
                f"</tr>"
            )
            for key in stats_a:
                va = stats_a[key]
                vb = stats_b[key]
                table_html += (
                    f"<tr style='border-bottom:1px solid var(--border-dim)'>"
                    f"<td style='padding:7px 6px;color:var(--text-muted)'>{key}</td>"
                    f"<td style='padding:7px 6px;text-align:center;color:#ddd'>{va}</td>"
                    f"<td style='padding:7px 6px;text-align:center;color:#ddd'>{vb}</td>"
                    f"</tr>"
                )
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            common_rounds = (
                set(qf[qf["Driver"] == driver_a][["Year", "RoundNumber"]].apply(tuple, axis=1))
                & set(qf[qf["Driver"] == driver_b][["Year", "RoundNumber"]].apply(tuple, axis=1))
            )
            if common_rounds:
                qa = qf[qf["Driver"] == driver_a].set_index(["Year", "RoundNumber"])
                qb = qf[qf["Driver"] == driver_b].set_index(["Year", "RoundNumber"])
                shared = pd.DataFrame(index=pd.MultiIndex.from_tuples(common_rounds))
                shared["posA"] = qa.loc[shared.index, "QualiPos"]
                shared["posB"] = qb.loc[shared.index, "QualiPos"]
                shared = shared.dropna()
                wins_a = int((shared["posA"] < shared["posB"]).sum())
                wins_b = int((shared["posA"] > shared["posB"]).sum())
                total  = len(shared)

                st.markdown("<br>", unsafe_allow_html=True)
                ca, cb, cc = st.columns(3)
                ca.metric(f"{driver_a} quali wins", f"{wins_a}/{total}")
                cb.metric("Races together", str(total))
                cc.metric(f"{driver_b} quali wins", f"{wins_b}/{total}")

    # ── Tab 3: SHAP explainer ──────────────────────────────────────────────────
    with tab3:
        st.markdown("### SHAP feature contributions")
        st.markdown(
            "<div class='tag'>Why the model gave each driver their podium "
            "probability — feature-by-feature breakdown</div><br>",
            unsafe_allow_html=True,
        )

        if not SHAP_AVAILABLE:
            st.error(
                "SHAP is not installed. "
                "Run: `pip install shap` in your environment then restart."
            )
        elif not race_path.exists():
            st.warning("Race feature table not found. Run `python src/features.py` first.")
        else:
            shap_driver = st.selectbox(
                "Select driver to explain", drivers, key="shap_driver"
            )

            @st.cache_data(ttl=600)
            def _compute_shap(driver: str, year: int, rnd: int):
                """Compute SHAP values for one driver's race prediction."""
                from sklearn.preprocessing import LabelEncoder
                import sys as _sys
                _sys.path.insert(0, str(_SRC_DIR))
                from config import RACE_FEATURES

                base = xgb.XGBClassifier()
                base.load_model(str(MODELS_DIR / "race_model.json"))

                enc_path = MODELS_DIR / "race_model_encoders.json"
                with open(enc_path) as f:
                    raw_enc = json.load(f)
                encoders = {}
                for col, classes in raw_enc.items():
                    le = LabelEncoder()
                    le.classes_ = np.array(classes)
                    encoders[col] = le

                metrics_path = MODELS_DIR / "race_model_metrics.json"
                impute_stats = {}
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        impute_stats = json.load(f).get("impute_stats", {})

                rf_df = pd.read_parquet(str(race_path))
                mask  = (
                    (rf_df["Driver"] == driver)
                    & (rf_df["Year"] == year)
                    & (rf_df["RoundNumber"] == rnd)
                )
                row = rf_df[mask]
                if row.empty:
                    row = (
                        rf_df[rf_df["Driver"] == driver]
                        .sort_values(["Year", "RoundNumber"])
                        .tail(1)
                    )
                if row.empty:
                    return None, None

                rf_enc = rf_df.copy()
                for col in ["Driver", "TeamName", "CircuitShortName"]:
                    enc_col = f"{col}_enc"
                    if col in encoders and col in rf_enc.columns:
                        le = encoders[col]
                        rf_enc[enc_col] = rf_enc[col].apply(
                            lambda v, le=le: int(le.transform([v])[0])
                            if v in le.classes_ else 0
                        )

                row_enc = rf_enc[mask]
                if row_enc.empty:
                    row_enc = (
                        rf_enc[rf_enc["Driver"] == driver]
                        .sort_values(["Year", "RoundNumber"])
                        .tail(1)
                    )

                available = [f for f in RACE_FEATURES if f in row_enc.columns]
                X = row_enc[available].copy()
                for col in available:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(impute_stats.get(col, 0.0))

                explainer   = shap.TreeExplainer(base)
                shap_values = explainer.shap_values(X)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                result = (
                    pd.DataFrame({
                        "feature": available,
                        "shap":    sv,
                        "value":   X.iloc[0].values,
                    })
                    .sort_values("shap", key=abs, ascending=False)
                    .head(15)
                )

                base_val = (
                    float(explainer.expected_value[1])
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else float(explainer.expected_value)
                )
                return result, base_val

            with st.spinner(f"Computing SHAP values for {shap_driver}..."):
                shap_df, base_val = _compute_shap(shap_driver, year_sel, round_sel)

            if shap_df is None:
                st.warning(f"No feature data found for {shap_driver}.")
            else:
                pred_row = df_pred[df_pred["Driver"] == shap_driver]
                if not pred_row.empty:
                    prob_val = float(pred_row["PodiumProbability"].iloc[0]) * 100
                    st.markdown(
                        f"**{shap_driver}** predicted podium probability: **{prob_val:.1f}%**"
                    )

                colours_shap = [
                    "rgba(0,242,255,0.85)" if v > 0 else "rgba(255,45,85,0.85)"
                    for v in shap_df["shap"]
                ]
                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    x=shap_df["shap"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker_color=colours_shap,
                    text=[
                        f"+{v:.3f}" if v > 0 else f"{v:.3f}"
                        for v in shap_df["shap"]
                    ],
                    textposition="outside",
                    textfont=dict(family="Space Mono", size=10, color="#666"),
                ))
                fig_shap.add_vline(x=0, line_color="#333", line_width=1)
                layout_s = _plotly_layout(440)
                layout_s["xaxis"] = dict(
                    title=dict(
                        text="SHAP value (impact on podium probability)",
                        font=dict(size=10, color="#555", family="Space Mono"),
                    ),
                    gridcolor="#111",
                    tickfont=dict(family="Space Mono", size=9, color="#555"),
                )
                layout_s["yaxis"] = dict(
                    autorange="reversed",
                    tickfont=dict(family="Space Mono", size=10, color="#888"),
                )
                layout_s["margin"] = dict(l=0, r=80, t=10, b=40)
                fig_shap.update_layout(**layout_s)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("### Feature values used")
                display = shap_df[["feature", "value", "shap"]].copy()
                display.columns = ["Feature", "Value used", "SHAP contribution"]
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Value used":
                            st.column_config.NumberColumn(format="%.4f"),
                        "SHAP contribution":
                            st.column_config.NumberColumn(format="%.4f"),
                    },
                )
                st.markdown(
                    "<div class='tag'>Cyan bars push probability UP · "
                    "Orange bars pull probability DOWN · "
                    "Bar length = magnitude of impact</div>",
                    unsafe_allow_html=True,
                )