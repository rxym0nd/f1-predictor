"""
evaluate.py  (item #21 — chaos race detection added)

Post-race evaluation with chaos flagging.

A race is flagged as chaotic when:
  - Safety car or red flag deployed for ≥ 10% of total race laps, OR
  - Spearman ρ < 0.55 (model was essentially random — indicative of chaos)

Chaotic races are:
  - Still recorded in eval_history.json with their real metrics
  - Flagged with "chaos": true so dashboards and trend reports can
    optionally exclude them from rolling averages
  - Trend lines in print_trend_report() show both raw and
    chaos-excluded averages when ≥ 2 chaos races are present

Usage:
    python src/evaluate.py --year 2026 --round 3
"""

import argparse
import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR       = Path("cache")
PREDICTIONS_DIR = Path("predictions")
MODELS_DIR      = Path("models")
EVAL_HISTORY    = MODELS_DIR / "eval_history.json"

# ── Chaos detection thresholds ────────────────────────────────────────────────
# A race that crosses either threshold is flagged as chaotic.
# Neither threshold excludes the race from history — they only add a flag.
CHAOS_RHO_THRESHOLD     = 0.55   # Spearman ρ below this = chaotic race
CHAOS_SC_LAP_FRACTION   = 0.10   # SC/VSC/red-flag laps > 10% of total = chaotic


def _count_sc_laps(year: int, round_number: int) -> tuple[int, int]:
    """
    Returns (sc_laps, total_laps) by loading race track status data.
    SC/VSC/red-flag periods count as safety car laps.
    Returns (0, 0) if data cannot be loaded.
    """
    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)

        # FastF1 track_status: status codes 4=SC, 5=Red flag, 6=VSC, 7=VSC ending
        sc_codes = {"4", "5", "6", "7"}

        if session.track_status is None or session.track_status.empty:
            return 0, 0

        ts = session.track_status.copy()

        # Compute total session duration in seconds
        if "Time" not in ts.columns:
            return 0, 0

        ts["Time_s"] = pd.to_timedelta(ts["Time"]).dt.total_seconds()
        ts_sorted    = ts.sort_values("Time_s").reset_index(drop=True)

        # Compute duration of each status period
        ts_sorted["Duration_s"] = ts_sorted["Time_s"].shift(-1) - ts_sorted["Time_s"]
        ts_sorted = ts_sorted.dropna(subset=["Duration_s"])

        total_s = ts_sorted["Duration_s"].sum()
        sc_s    = ts_sorted[ts_sorted["Status"].isin(sc_codes)]["Duration_s"].sum()

        if total_s <= 0:
            return 0, 0

        return int(sc_s), int(total_s)

    except Exception:
        return 0, 0


def is_chaos_race(
    sc_seconds: int,
    total_seconds: int,
    spearman_rho: float | None,
) -> tuple[bool, str]:
    """
    Returns (is_chaos, reason_string).
    """
    reasons = []

    if total_seconds > 0:
        sc_fraction = sc_seconds / total_seconds
        if sc_fraction >= CHAOS_SC_LAP_FRACTION:
            reasons.append(
                f"SC/VSC/red flag for {sc_fraction*100:.0f}% of session time"
            )

    if spearman_rho is not None and spearman_rho < CHAOS_RHO_THRESHOLD:
        reasons.append(f"Spearman ρ={spearman_rho:.3f} below chaos threshold")

    return len(reasons) > 0, "; ".join(reasons)


def fetch_actual_quali(year: int, round_number: int) -> pd.DataFrame:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    for label in ("Q", "SQ"):
        try:
            session = fastf1.get_session(year, round_number, label)
            session.load(telemetry=False, weather=False, messages=False)
            laps = session.laps.copy()
            if laps.empty:
                continue
            break
        except Exception as exc:
            log.warning("Could not load %s for %d R%d: %s", label, year, round_number, exc)
            continue
    else:
        return pd.DataFrame(columns=["Driver", "ActualQualiPos"])

    if "Abbreviation" in laps.columns and "Driver" not in laps.columns:
        laps = laps.rename(columns={"Abbreviation": "Driver"})
    laps["LapTime_s"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
    laps = laps.dropna(subset=["LapTime_s"])
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "ActualQualiPos"])
    median = laps["LapTime_s"].median()
    laps = laps[laps["LapTime_s"] <= median * 1.20]
    best = (
        laps.groupby("Driver")["LapTime_s"].min().reset_index()
        .rename(columns={"LapTime_s": "BestTime_s"})
    )
    best["ActualQualiPos"] = best["BestTime_s"].rank(method="min").astype(int)
    log.info("Fetched actual quali results: %d drivers", len(best))
    return best[["Driver", "ActualQualiPos"]]


def fetch_actual_race(year: int, round_number: int) -> pd.DataFrame:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results.copy()
    if "Abbreviation" in results.columns and "Driver" not in results.columns:
        results = results.rename(columns={"Abbreviation": "Driver"})
    results["ActualFinishPos"] = pd.to_numeric(results["Position"], errors="coerce")
    results["ActualPodium"]    = (results["ActualFinishPos"] <= 3).astype(int)
    log.info("Fetched actual race results: %d drivers", len(results))
    return results[["Driver", "ActualFinishPos", "ActualPodium"]]


def load_prediction(year: int, round_number: int) -> tuple[pd.DataFrame, str, str]:
    path = PREDICTIONS_DIR / f"{year}_R{round_number:02d}_prediction.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction at {path}. "
            f"Run: python src/predict.py --year {year} --round {round_number}"
        )
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["predictions"])
    log.info("Loaded prediction for %s %d R%d (%d drivers)",
             data["event"], year, round_number, len(df))
    return df, data["event"], data["circuit"]


def compute_quali_metrics(merged: pd.DataFrame) -> dict:
    valid = merged.dropna(subset=["ActualQualiPos", "PredictedQualiPos"])
    if len(valid) < 3:
        return {"MAE_positions": None, "Spearman_rho": None}
    y_true = valid["ActualQualiPos"].values
    y_pred = valid["PredictedQualiPos"].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "MAE_positions": round(mae, 4),
        "Spearman_rho":  round(float(rho), 4) if not np.isnan(rho) else None,
    }


def compute_race_metrics(merged: pd.DataFrame) -> dict:
    valid = merged.dropna(subset=["ActualPodium", "PodiumProbability"])
    if len(valid) < 3:
        return {"Brier_score": None, "Top3_overlap": None}
    y_true = valid["ActualPodium"].values
    y_prob = np.clip(valid["PodiumProbability"].values, 0.05, 0.95)
    brier = float(brier_score_loss(y_true, y_prob))
    predicted_top3 = set(valid.nlargest(3, "PodiumProbability")["Driver"].tolist())
    actual_podium  = set(valid[valid["ActualPodium"] == 1]["Driver"].tolist())
    return {
        "Brier_score":  round(brier, 4),
        "Top3_overlap": len(predicted_top3 & actual_podium),
    }


def load_eval_history() -> list:
    if EVAL_HISTORY.exists():
        with open(EVAL_HISTORY) as f:
            return json.load(f)
    return []


def save_eval_history(history: list):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_HISTORY, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Saved eval history → %s (%d entries)", EVAL_HISTORY, len(history))


def update_history(
    history, year, round_number, event, circuit,
    q_metrics, r_metrics,
    chaos: bool = False, chaos_reason: str = "",
):
    key   = f"{year}_R{round_number:02d}"
    entry = {
        "key":          key,
        "year":         year,
        "round":        round_number,
        "event":        event,
        "circuit":      circuit,
        "quali":        q_metrics,
        "race":         r_metrics,
        "chaos":        chaos,
        "chaos_reason": chaos_reason,
    }
    history = [h for h in history if h.get("key") != key]
    history.append(entry)
    history.sort(key=lambda h: (h["year"], h["round"]))
    return history


def print_trend_report(history: list):
    recent = history[-5:]
    if not recent:
        return

    log.info("")
    log.info("── Rolling performance (last %d rounds) ──────────────────", len(recent))
    log.info("%-20s %-8s %-10s %-8s %-8s %-6s",
             "Event", "MAE pos", "Spearman ρ", "Brier", "Top-3", "Flag")
    log.info("-" * 65)

    for h in recent:
        q    = h.get("quali", {})
        r    = h.get("race",  {})
        flag = "🌀 CHAOS" if h.get("chaos") else ""
        log.info(
            "%-20s %-8s %-10s %-8s %-8s %-6s",
            h["event"][:20],
            f"{q['MAE_positions']:.2f}" if q.get("MAE_positions") is not None else "—",
            f"{q['Spearman_rho']:.3f}"  if q.get("Spearman_rho")  is not None else "—",
            f"{r['Brier_score']:.4f}"   if r.get("Brier_score")   is not None else "—",
            f"{r['Top3_overlap']}/3"    if r.get("Top3_overlap")  is not None else "—",
            flag,
        )

    # Show trend both with and without chaos races
    all_maes   = [h["quali"].get("MAE_positions") for h in recent if h["quali"].get("MAE_positions")]
    all_briers = [h["race"].get("Brier_score")    for h in recent if h["race"].get("Brier_score")]
    clean      = [h for h in recent if not h.get("chaos")]
    clean_maes   = [h["quali"].get("MAE_positions") for h in clean if h["quali"].get("MAE_positions")]
    clean_briers = [h["race"].get("Brier_score")    for h in clean if h["race"].get("Brier_score")]

    n_chaos = sum(1 for h in recent if h.get("chaos"))
    if n_chaos > 0:
        log.info("")
        log.info("Including chaos races:   MAE %.2f | Brier %.4f",
                 sum(all_maes)/len(all_maes)     if all_maes   else 0,
                 sum(all_briers)/len(all_briers) if all_briers else 0)
        if clean_maes:
            log.info("Excluding chaos races:   MAE %.2f | Brier %.4f",
                     sum(clean_maes)/len(clean_maes),
                     sum(clean_briers)/len(clean_briers) if clean_briers else 0)
    else:
        if len(all_maes) >= 2:
            d = "↓" if all_maes[-1] < all_maes[0] else "↑" if all_maes[-1] > all_maes[0] else "→"
            log.info("Quali MAE trend: %s (%.2f → %.2f)", d, all_maes[0], all_maes[-1])
        if len(all_briers) >= 2:
            d = "↓" if all_briers[-1] < all_briers[0] else "↑" if all_briers[-1] > all_briers[0] else "→"
            log.info("Brier trend:    %s (%.4f → %.4f)", d, all_briers[0], all_briers[-1])


def evaluate(year: int, round_number: int):
    pred_df, event, circuit = load_prediction(year, round_number)

    try:
        actual_quali = fetch_actual_quali(year, round_number)
    except Exception as exc:
        log.warning("Could not fetch quali: %s", exc)
        actual_quali = pd.DataFrame(columns=["Driver", "ActualQualiPos"])

    try:
        actual_race = fetch_actual_race(year, round_number)
    except Exception as exc:
        log.warning("Could not fetch race: %s", exc)
        actual_race = pd.DataFrame(columns=["Driver", "ActualFinishPos", "ActualPodium"])

    merged = pred_df.copy()
    merged = merged.merge(actual_quali, on="Driver", how="left") if len(actual_quali) > 0 \
        else merged.assign(ActualQualiPos=np.nan)
    merged = merged.merge(actual_race, on="Driver", how="left") if len(actual_race) > 0 \
        else merged.assign(ActualFinishPos=np.nan, ActualPodium=np.nan)

    q_metrics = compute_quali_metrics(merged)
    r_metrics = compute_race_metrics(merged)

    # ── Chaos detection ───────────────────────────────────────────────────────
    sc_s, total_s = _count_sc_laps(year, round_number)
    chaos, chaos_reason = is_chaos_race(
        sc_s, total_s, q_metrics.get("Spearman_rho")
    )

    # ── Print breakdown ───────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 66)
    log.info("EVALUATION — %s %d Round %d%s",
             event, year, round_number,
             "  [CHAOS RACE]" if chaos else "")
    log.info("═" * 66)
    if chaos:
        log.info("⚠  Chaos flag: %s", chaos_reason)
        log.info("   This race is excluded from clean trend averages.")
        log.info("")

    log.info("%-6s %-6s %-6s %-6s %-6s %-10s",
             "Driver", "PGrid", "AGrid", "ΔGrid", "APod", "PodProb%")
    log.info("-" * 66)
    for _, row in merged.sort_values("PredictedQualiPos").iterrows():
        pgrid = row.get("PredictedQualiPos")
        agrid = row.get("ActualQualiPos")
        delta = f"{int(pgrid)-int(agrid):+d}" if pd.notna(pgrid) and pd.notna(agrid) else "—"
        apod  = "✓" if row.get("ActualPodium") == 1 else "✗" if pd.notna(row.get("ActualPodium")) else "—"
        log.info(
            "%-6s %-6s %-6s %-6s %-6s %-10s %-10s",
            str(row["Driver"]),
            str(int(pgrid)) if pd.notna(pgrid) else "—",
            str(int(agrid)) if pd.notna(agrid) else "—",
            delta, apod,
            f"{row.get('PodiumProbability', 0)*100:.1f}%",
            "PODIUM" if row.get("ActualPodium") == 1 else "",
        )
    log.info("═" * 66)
    log.info("QUALI  MAE: %s | ρ: %s",
             f"{q_metrics['MAE_positions']:.4f}" if q_metrics["MAE_positions"] else "n/a",
             f"{q_metrics['Spearman_rho']:.4f}"  if q_metrics["Spearman_rho"]  else "n/a")
    log.info("RACE   Brier: %s | Top-3: %s/3",
             f"{r_metrics['Brier_score']:.4f}" if r_metrics["Brier_score"] else "n/a",
             r_metrics["Top3_overlap"] if r_metrics["Top3_overlap"] is not None else "n/a")

    history = load_eval_history()
    history = update_history(
        history, year, round_number, event, circuit,
        q_metrics, r_metrics, chaos, chaos_reason,
    )
    save_eval_history(history)
    print_trend_report(history)
    log.info("\nNext: python src/features.py && python src/train.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    args = parser.parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    evaluate(args.year, args.round_number)


if __name__ == "__main__":
    main()