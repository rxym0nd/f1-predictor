"""
batch_evaluate.py

Runs the full predict → evaluate loop across a range of rounds.

Usage:
    python src/batch_evaluate.py --year 2024
    python src/batch_evaluate.py --year 2025 --from-round 1 --to-round 12
    python src/batch_evaluate.py --year 2026
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR       = Path("cache")
PREDICTIONS_DIR = Path("predictions")
MODELS_DIR      = Path("models")
_AVAILABILITY_HOURS = 3


def _race_date_for_round(schedule: pd.DataFrame, round_number: int) -> datetime | None:
    row = schedule[schedule["RoundNumber"] == round_number]
    if row.empty:
        return None
    for col in ("Session5Date", "EventDate"):
        if col in row.columns:
            val = row[col].iloc[0]
            if pd.notna(val):
                dt = pd.Timestamp(val)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                return dt.to_pydatetime()
    return None


def get_completed_rounds(year: int) -> list[int]:
    """Date-based completion check — no session.load() probing."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        log.error("Could not fetch schedule for %d: %s", year, exc)
        return []

    now       = datetime.now(timezone.utc)
    completed = []
    for _, row in schedule.iterrows():
        rnd     = int(row["RoundNumber"])
        race_dt = _race_date_for_round(schedule, rnd)
        if race_dt is None:
            continue
        if now > race_dt + pd.Timedelta(hours=_AVAILABILITY_HOURS):
            completed.append(rnd)

    log.info("Found %d completed rounds for %d", len(completed), year)
    return completed


def run_predict(year: int, rnd: int) -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "src/predict.py", "--year", str(year), "--round", str(rnd)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.warning("predict.py failed for %d R%d:\n%s", year, rnd, r.stderr[-500:])
    return r.returncode == 0, r.stderr


def run_evaluate(year: int, rnd: int) -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "src/evaluate.py", "--year", str(year), "--round", str(rnd)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.warning("evaluate.py failed for %d R%d:\n%s", year, rnd, r.stderr[-500:])
    return r.returncode == 0, r.stderr


def load_eval_history() -> list:
    path = MODELS_DIR / "eval_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def print_summary(year: int, results: list[dict]):
    if not results:
        return
    log.info("")
    log.info("═" * 72)
    log.info("BATCH EVALUATION SUMMARY — %d", year)
    log.info("═" * 72)
    log.info("%-5s %-22s %-10s %-12s %-8s %-6s",
             "Round", "Event", "MAE pos", "Spearman ρ", "Brier", "Top-3")
    log.info("-" * 72)

    maes = []
    rhos = []
    briers = []
    overlaps = []

    for r in results:
        q  = r.get("quali", {})
        rc = r.get("race",  {})
        mae = q.get("MAE_positions")
        rho = q.get("Spearman_rho")
        brier = rc.get("Brier_score")
        top3 = rc.get("Top3_overlap")
        flag = " ✗" if r.get("status", "ok") != "ok" else ""
        log.info(
            "%-5s %-22s %-10s %-12s %-8s %-6s%s",
            f"R{r['round']}", r.get("event","—")[:22],
            f"{mae:.2f}" if mae is not None else "—",
            f"{rho:.3f}" if rho is not None else "—",
            f"{brier:.4f}" if brier is not None else "—",
            f"{top3}/3" if top3 is not None else "—",
            flag,
        )
        if mae is not None:
            maes.append(mae)
        if rho is not None:
            rhos.append(rho)
        if brier is not None:
            briers.append(brier)
        if top3 is not None:
            overlaps.append(top3)

    log.info("-" * 72)
    if maes:
        log.info(
            "%-5s %-22s %-10s %-12s %-8s %-6s",
            "AVG", "",
            f"{sum(maes)/len(maes):.2f}",
            f"{sum(rhos)/len(rhos):.3f}"           if rhos     else "—",
            f"{sum(briers)/len(briers):.4f}"       if briers   else "—",
            f"{sum(overlaps)/len(overlaps):.2f}/3" if overlaps else "—",
        )
    log.info("═" * 72)
    log.info("Rounds failed: %d | History entries: %d",
             sum(1 for r in results if r.get("status") != "ok"),
             len(load_eval_history()))


def batch_evaluate(year: int, from_round: int, to_round: int | None):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed_rounds(year)
    if not completed:
        log.error("No completed rounds for %d.", year)
        return

    target = [r for r in completed
               if r >= from_round and (to_round is None or r <= to_round)]
    if not target:
        log.error("No rounds in range for %d.", year)
        return

    log.info("Batch evaluating %d rounds for %d: %s",
             len(target), year, ", ".join(f"R{r}" for r in target))

    results = []
    for i, rnd in enumerate(target, 1):
        log.info("\n── [%d/%d] %d R%d ──────────────────────", i, len(target), year, rnd)
        pred_path = PREDICTIONS_DIR / f"{year}_R{rnd:02d}_prediction.json"
        if not pred_path.exists():
            ok, _ = run_predict(year, rnd)
            if not ok:
                results.append({"round": rnd, "event": f"R{rnd}", "status": "predict_failed"})
                continue

        ok, _ = run_evaluate(year, rnd)
        if not ok:
            results.append({"round": rnd, "event": f"R{rnd}", "status": "evaluate_failed"})
            continue

        key   = f"{year}_R{rnd:02d}"
        entry = next((h for h in load_eval_history() if h.get("key") == key), None)
        if entry:
            entry["status"] = "ok"
            results.append(entry)
        else:
            results.append({"round": rnd, "event": f"R{rnd}", "status": "no_history"})

    print_summary(year, results)
    log.info("\nBatch complete. Retrain: python src/features.py && python src/train.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",       type=int, required=True)
    parser.add_argument("--from-round", type=int, default=1, dest="from_round")
    parser.add_argument("--to-round",   type=int, default=None, dest="to_round")
    args = parser.parse_args()
    batch_evaluate(args.year, args.from_round, args.to_round)


if __name__ == "__main__":
    main()