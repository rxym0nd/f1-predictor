"""
predict.py

Pre-race prediction runner.

Bug fixes over previous version:
  1. CIRCUIT BUG — fetch_entry_list fell back to a prior round's session for
     the driver list, and then used THAT round's circuit name for every
     circuit-specific feature (affinity, SC rate, flags, overtaking difficulty).
     Fix: circuit and event_name are now always resolved from the TARGET round's
     FastF1 schedule entry, independent of whichever round was used for the
     driver list fallback.

  2. FP2/FP3 NOT APPLIED — ingest.py saves raw FP2/FP3 parquets for the
     current weekend, but predict.py only reads from processed parquets which
     are only updated when features.py is re-run. With fresh FP data sitting
     unused, FP3_BestLap_s, FP3_PaceRank, FP2_LongRunPace_s etc. were all
     imputed from train-set medians — every driver got the same flat value.
     Fix: predict.py now reads the raw FP2/FP3 parquets for the current round
     directly (using the same extraction functions as features.py) and injects
     the real values before imputation runs.

  3. MISSING FEATURES — ConChampDelta, ConChampPos, CareerRaceCount, and
     CircuitSCRate were never computed in build_prediction_features; they fell
     through to train-set median imputation.  This made all 2026 rookies look
     like 150-race veterans (CareerRaceCount imputed to median ~150) and gave
     every team the same championship context.
     Fix: all four features are now computed from prior_race / KNOWN_SC_RATES
     inside build_prediction_features.

Usage:
    python src/predict.py --year 2026 --round 3
"""

import argparse
import json
import logging
from pathlib import Path
import time

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from config import (
    CHAOS_CIRCUITS,
    DEFAULT_SC_RATE,
    KNOWN_SC_RATES,
    QUALI_FEATURES,
    RACE_FEATURES,
    ROLLING_WINDOW,
    circuit_type_flags,
    grid_difficulty_score,
    normalise_team,
    years_since_reg_change,
)

# Import FP extraction functions from features.py so logic stays in sync
from features import extract_fp2_longruns, extract_fp3_pace

try:
    from weather import get_forecast_for_round as _get_weather_forecast
    _WEATHER_AVAILABLE = True
except ImportError:
    _get_weather_forecast = None  # type: ignore
    _WEATHER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR       = Path("cache")
RAW_DIR         = Path("data/raw")
PROCESSED_DIR   = Path("data/processed")
MODELS_DIR      = Path("models")
PREDICTIONS_DIR = Path("predictions")

# Maximum age of raw data files before a staleness warning is raised.
_STALE_DATA_DAYS = 8

# Long-run threshold must match features.py
_LONG_RUN_THRESHOLD = 1.07


def _check_stale_data():
    """Warn if raw data files haven't been updated recently."""
    parquets = list(RAW_DIR.glob("*.parquet"))
    if not parquets:
        return
    latest_mtime = max(p.stat().st_mtime for p in parquets)
    age_days = (time.time() - latest_mtime) / 86400
    if age_days > _STALE_DATA_DAYS:
        log.warning(
            "Raw data is %.1f days old (threshold: %d days). "
            "Run ingest.py to fetch FP2/FP3/Q data for this weekend "
            "before generating predictions — sector gap features will "
            "otherwise be imputed from a different circuit.",
            age_days, _STALE_DATA_DAYS,
        )


def _check_processed_stale(year: int, round_number: int):
    """
    Warn if raw FP parquets for this round are newer than the processed
    parquets, meaning features.py has not been re-run since ingest.py ran.
    predict.py now injects FP features directly so this is non-fatal, but
    the processed parquets still carry historical rolling features that
    should also be kept current.
    """
    raw_fp_files = list(RAW_DIR.glob(f"{year}_R{round_number:02d}_FP*_laps.parquet"))
    processed_file = PROCESSED_DIR / "quali_features.parquet"

    if not raw_fp_files or not processed_file.exists():
        return

    newest_raw = max(p.stat().st_mtime for p in raw_fp_files)
    proc_mtime = processed_file.stat().st_mtime

    if newest_raw > proc_mtime:
        log.warning(
            "Raw FP data for %d R%d is newer than processed parquets. "
            "Run 'python src/features.py && python src/train.py' to incorporate "
            "the latest data into rolling features for future rounds. "
            "This prediction uses FP features injected directly from raw files.",
            year, round_number,
        )


def _silence_fastf1():
    ff1 = logging.getLogger("fastf1")
    orig = ff1.level
    ff1.setLevel(logging.CRITICAL)
    return ff1, orig


def _restore_fastf1(ff1_logger, orig_level: int):
    ff1_logger.setLevel(orig_level)


# ── Model + encoder loading ───────────────────────────────────────────────────

def load_quali_model() -> xgb.XGBRanker:
    model = xgb.XGBRanker()
    model.load_model(str(MODELS_DIR / "quali_model.json"))
    log.info("Loaded quali model")
    return model


def load_race_model() -> tuple:
    """
    Load the base XGBoost classifier and the IsotonicRegression calibrator.
    Returns (base_model, calibrator).  calibrator may be None if not found.
    """
    base = xgb.XGBClassifier()
    base.load_model(str(MODELS_DIR / "race_model.json"))
    log.info("Loaded race model")

    cal_path = MODELS_DIR / "race_model_calibrated.pkl"
    if cal_path.exists():
        calibrator = joblib.load(str(cal_path))
        log.info("Loaded isotonic calibrator")
    else:
        calibrator = None
        log.warning("No calibrator found — raw probabilities will be used")
    return base, calibrator


def load_encoders(name: str) -> dict:
    path = MODELS_DIR / f"{name}_encoders.json"
    with open(path) as f:
        raw = json.load(f)
    encoders = {}
    for col, classes in raw.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        encoders[col] = le
    log.info("Loaded encoders for %s (%d columns)", name, len(encoders))
    return encoders


def load_impute_stats(name: str) -> dict[str, float]:
    """Load train-set imputation medians saved by train.py."""
    path = MODELS_DIR / f"{name}_metrics.json"
    if path.exists():
        with open(path) as f:
            metrics = json.load(f)
        return metrics.get("impute_stats", {})
    return {}


# ── Historical data loaders ───────────────────────────────────────────────────

def load_historical_quali_best() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "quali_features.parquet")


def load_historical_race_results() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "race_features.parquet")
    if "FinishPos" not in df.columns and "Position" in df.columns:
        df["FinishPos"] = pd.to_numeric(df["Position"], errors="coerce")
    if "DNF" not in df.columns:
        df["DNF"] = df.get("Status", pd.Series("Finished", index=df.index)).apply(
            lambda s: 0 if str(s).startswith("Finished") or str(s).startswith("+") else 1
        )
    return df


# ── FP raw data loader for current round ─────────────────────────────────────

def _load_raw_fp_laps(year: int, round_number: int, session_type: str) -> pd.DataFrame:
    """
    Load raw FP laps parquet for the specific round saved by ingest.py.
    Returns an empty DataFrame if the file does not exist yet.
    Normalises LapTime to seconds and driver abbreviation column.
    """
    pattern = f"{year}_R{round_number:02d}_{session_type}_laps.parquet"
    path = RAW_DIR / pattern
    if not path.exists():
        log.debug("No raw %s laps for %d R%d — will be imputed", session_type, year, round_number)
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Normalise driver column
    if "Abbreviation" in df.columns and "Driver" not in df.columns:
        df = df.rename(columns={"Abbreviation": "Driver"})

    # Convert LapTime to seconds
    try:
        df["LapTime_s"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    except Exception:
        df["LapTime_s"] = pd.to_numeric(df.get("LapTime_s", np.nan), errors="coerce")

    df = df.dropna(subset=["LapTime_s"])
    log.info("Loaded %d raw %s laps for %d R%d", len(df), session_type, year, round_number)
    return df


def _load_current_round_fp_features(year: int, round_number: int) -> dict[str, pd.Series]:
    """
    Load FP2 and FP3 raw laps for this specific round from data/raw/,
    extract pace features using the same functions as features.py,
    and return a dict mapping driver abbreviation -> feature dict.

    Returns an empty dict if no raw FP data exists for this round.
    The caller merges these values into the prediction DataFrame before
    imputation runs, so real FP data always takes precedence over medians.
    """
    fp3_laps = _load_raw_fp_laps(year, round_number, "FP3")
    fp2_laps = _load_raw_fp_laps(year, round_number, "FP2")

    fp3_pace     = extract_fp3_pace(fp3_laps)
    fp2_longruns = extract_fp2_longruns(fp2_laps)

    # Filter to just this round (should already be, but be safe)
    if not fp3_pace.empty:
        fp3_pace = fp3_pace[
            (fp3_pace["Year"] == year) & (fp3_pace["RoundNumber"] == round_number)
        ]
    if not fp2_longruns.empty:
        fp2_longruns = fp2_longruns[
            (fp2_longruns["Year"] == year) & (fp2_longruns["RoundNumber"] == round_number)
        ]

    # Build a driver-keyed lookup
    result: dict[str, dict] = {}

    for _, row in fp3_pace.iterrows():
        d = str(row["Driver"])
        result.setdefault(d, {})
        result[d]["FP3_BestLap_s"]      = row["FP3_BestLap_s"]
        result[d]["FP3_GapToFastest_s"] = row["FP3_GapToFastest_s"]
        result[d]["FP3_PaceRank"]        = row["FP3_PaceRank"]

    for _, row in fp2_longruns.iterrows():
        d = str(row["Driver"])
        result.setdefault(d, {})
        result[d]["FP2_LongRunPace_s"]  = row["FP2_LongRunPace_s"]
        result[d]["FP2_LongRunRank"]    = row["FP2_LongRunRank"]
        result[d]["FP2_LongRunDelta_s"] = row["FP2_LongRunDelta_s"]

    if result:
        fp3_count = sum(1 for v in result.values() if "FP3_BestLap_s" in v)
        fp2_count = sum(1 for v in result.values() if "FP2_LongRunPace_s" in v)
        log.info(
            "Injecting live FP features for %d R%d: "
            "FP3 pace for %d drivers, FP2 long-run for %d drivers",
            year, round_number, fp3_count, fp2_count,
        )
    else:
        log.warning(
            "No raw FP data found for %d R%d — FP features will be imputed from medians. "
            "Run ingest.py to fetch FP2/FP3 data for this weekend.",
            year, round_number,
        )

    return result


# ── Entry list fetcher ────────────────────────────────────────────────────────

def fetch_entry_list(year: int, round_number: int) -> pd.DataFrame:
    """
    Fetch the driver entry list for the target round via FastF1.

    BUG FIX: The circuit name and event name are always resolved from the
    TARGET round's schedule entry (via fastf1.get_event), regardless of
    which fallback round was used to obtain the driver list.  Previously,
    when R3 returned 0 drivers and we fell back to R2, the R2 circuit
    (Shanghai) was used for all circuit-specific features instead of the
    R3 circuit (Suzuka) — poisoning affinity, SC rate, flags, overtaking
    difficulty, and weather historical fallbacks.

    Falls back to up to 2 previous rounds if the target has no data yet.
    Treats 0-driver loads as failures (FastF1 quirk for future sessions).
    """
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # ── Step 1: Resolve the TARGET round's circuit from the schedule ──────────
    # This is done independently and always refers to the round we are
    # predicting — never to a fallback round.
    target_circuit    = ""
    target_event_name = ""
    try:
        ff1_log, orig_lvl = _silence_fastf1()
        try:
            target_event      = fastf1.get_event(year, round_number)
            target_circuit    = target_event["Location"]
            target_event_name = target_event["EventName"]
        finally:
            _restore_fastf1(ff1_log, orig_lvl)
        log.info(
            "Resolved target round: %s %d Round %d (circuit: %s)",
            target_event_name, year, round_number, target_circuit,
        )
    except Exception as exc:
        log.warning(
            "Could not resolve circuit for %d R%d from schedule: %s — "
            "will use fallback session's circuit as last resort.",
            year, round_number, exc,
        )

    # ── Step 2: Get the driver list, falling back to prior rounds if needed ───
    attempts = [round_number]
    if round_number > 1:
        attempts.append(round_number - 1)
    if round_number > 2:
        attempts.append(round_number - 2)

    last_exc: Exception | None = None
    results = pd.DataFrame()
    fallback_circuit    = target_circuit      # will be overwritten only as last resort
    fallback_event_name = target_event_name

    for attempt_round in attempts:
        try:
            ff1_log, orig_lvl = _silence_fastf1()
            try:
                session = fastf1.get_session(year, attempt_round, "R")
                session.load(telemetry=False, weather=False, messages=False)
            finally:
                _restore_fastf1(ff1_log, orig_lvl)

            results = session.results.copy()

            if results.empty or len(results) == 0:
                log.warning(
                    "R%d (%s) loaded but returned 0 drivers — "
                    "session data not yet available, trying fallback",
                    attempt_round, session.event["EventName"],
                )
                continue

            if attempt_round != round_number:
                log.info(
                    "R%d not yet available — using driver list from R%d (%s) as proxy",
                    round_number, attempt_round, session.event["EventName"],
                )
                # Only use the fallback circuit if we couldn't resolve the target
                if not target_circuit:
                    fallback_circuit    = session.event["Location"]
                    fallback_event_name = session.event["EventName"]
                    log.warning(
                        "Could not resolve target circuit — falling back to R%d circuit: %s",
                        attempt_round, fallback_circuit,
                    )
            else:
                log.info(
                    "Fetched entry list for %s (%d drivers)",
                    session.event["EventName"], len(results),
                )
            break

        except Exception as exc:
            last_exc = exc
            log.warning("Could not load R%d: %s", attempt_round, exc)
            continue
    else:
        raise RuntimeError(
            f"Could not fetch a valid entry list for {year} R{round_number}. "
            "All fallback rounds also returned no data. "
            f"Last error: {last_exc}"
        )

    # ── Step 3: Normalise columns ─────────────────────────────────────────────
    if "Abbreviation" in results.columns and "Driver" not in results.columns:
        results = results.rename(columns={"Abbreviation": "Driver"})

    # Always stamp the TARGET circuit/event, not the fallback session's
    circuit    = target_circuit    or fallback_circuit
    event_name = target_event_name or fallback_event_name

    entry = results[["Driver", "TeamName"]].copy()
    entry["TeamName"]         = entry["TeamName"].map(normalise_team)
    entry["CircuitShortName"] = circuit
    entry["EventName"]        = event_name
    entry["Year"]             = year
    entry["RoundNumber"]      = round_number
    return entry.reset_index(drop=True)


# ── Feature builder ───────────────────────────────────────────────────────────

def build_prediction_features(
    entry: pd.DataFrame,
    year: int,
    round_number: int,
    hist_quali: pd.DataFrame,
    hist_race: pd.DataFrame,
    fp_features: dict,
) -> pd.DataFrame:
    """
    Build all feature rows for a future race using only pre-race data.

    fp_features — dict of {driver: {FP3_BestLap_s, FP3_PaceRank, ...}}
                  returned by _load_current_round_fp_features().  These
                  are applied before imputation so real FP data takes
                  precedence over train-set medians.

    All columns pre-initialised to NaN before any loop runs.
    """
    df = entry.copy()
    circuit = df["CircuitShortName"].iloc[0]

    for col in [
        "QualiphaseReached", "YearsSinceLastRegChange", "GridPenaltyPlaces",
        "StartCompound_enc", "StartTyreLife", "FreshStartTyre",
        "RollingAvgStints", "RollingAvgDegRate",
        "CircuitAvgStints", "CircuitDegRate",
        "IsChaosCircuit",
        "RollingQualiGap", "RollingQualiPos", "RollingQualiStdGap",
        "ConRollingQualiGap",
        "CircuitAvgQualiGap", "CircuitAvgQualiPos", "CircuitVisits",
        "H2H_QualiWinRate",
        "RollingAvgFinish", "RollingAvgGrid", "RollingPoints",
        "RollingPodiumRate", "RollingDNFRate", "DNFStreak",
        "ConRollingAvgFinish", "ConRollingPoints",
        "CumPointsBefore", "ChampionshipPos_norm",
        "AirTemp_mean", "TrackTemp_mean", "Humidity_mean", "Rainfall_any",
        "IsStreetCircuit", "IsHighDownforce", "IsLowDownforce",
        # Features previously missing from this builder
        "ConChampDelta", "ConChampPos",
        "CareerRaceCount",
        "CircuitSCRate",
        # FP features — will be overwritten from fp_features dict if available
        "FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank",
        "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    ]:
        df[col] = np.nan

    # ── Filter history ────────────────────────────────────────────────────────
    # prior_quali / prior_race: everything before this round.
    # Used for global medians, weather fallback, circuit affinity, champ context.
    prior_quali = hist_quali[
        (hist_quali["Year"] < year) |
        ((hist_quali["Year"] == year) & (hist_quali["RoundNumber"] < round_number))
    ].copy()

    prior_race = hist_race[
        (hist_race["Year"] < year) |
        ((hist_race["Year"] == year) & (hist_race["RoundNumber"] < round_number))
    ].copy()

    # ── Regulation cold-start ─────────────────────────────────────────────────
    # rolling_quali / rolling_race: used for ALL rolling driver/constructor form.
    # When years_since_reg_change == 0 (new regs), historical rolling data from
    # the previous era actively misleads the model. VER won 2025 on merit but
    # Red Bull's 2026 car is mid-pack — the model must not know 2025 VER stats.
    # Restricting to current-season data forces imputation to global medians for
    # most drivers, which levels the field and lets FP2/FP3 pace dominate.
    # H2H is also era-scoped: 2025 teammate pairs are completely different in 2026.
    # Circuit affinity still uses full history (circuit layout is unchanged).
    _reg_cold_start = years_since_reg_change(year) == 0
    if _reg_cold_start:
        rolling_quali = prior_quali[prior_quali["Year"] == year].copy()
        rolling_race  = prior_race[prior_race["Year"] == year].copy()
        log.info(
            "Regulation cold-start mode active (YearsSinceLastRegChange=0): "
            "rolling features restricted to %d season data "
            "(%d quali rows, %d race rows). FP2/FP3 features will dominate.",
            year, len(rolling_quali), len(rolling_race),
        )
    else:
        rolling_quali = prior_quali
        rolling_race  = prior_race

    # Global medians always from full history for stable fallback values
    global_q_gap  = prior_quali["GapToPole_s"].median()
    global_q_pos  = prior_quali["QualiPos"].median()
    global_finish = (
        pd.to_numeric(prior_race.get("FinishPos", pd.Series(dtype=float)),
                      errors="coerce").median()
        if len(prior_race) > 0 else 10.0
    )

    weights = np.arange(1, ROLLING_WINDOW + 1, dtype=float)

    def _wavg(series: pd.Series) -> float:
        arr = series.values
        w = weights[-len(arr):]
        return float(np.dot(arr, w) / w.sum()) if len(arr) > 0 else float("nan")

    # ── 1. Driver rolling quali form (era-scoped) ─────────────────────────────
    for driver in df["Driver"]:
        mask = df["Driver"] == driver
        dq = (rolling_quali[rolling_quali["Driver"] == driver]
              .sort_values(["Year", "RoundNumber"]).tail(ROLLING_WINDOW))
        if len(dq) == 0:
            # Cold-start: no same-era data — leave NaN so imputation uses
            # train medians. This intentionally levels the field for all
            # drivers rather than locking in stale cross-era values.
            if not _reg_cold_start:
                df.loc[mask, "RollingQualiGap"]    = global_q_gap
                df.loc[mask, "RollingQualiPos"]    = global_q_pos
                df.loc[mask, "RollingQualiStdGap"] = 0.0
        else:
            df.loc[mask, "RollingQualiGap"]    = _wavg(dq["GapToPole_s"])
            df.loc[mask, "RollingQualiPos"]    = _wavg(dq["QualiPos"])
            std = dq["GapToPole_s"].std(ddof=0)
            df.loc[mask, "RollingQualiStdGap"] = 0.0 if np.isnan(std) else float(std)

    # ── 2. Constructor rolling quali form (era-scoped) ────────────────────────
    for team in df["TeamName"].unique():
        mask = df["TeamName"] == team
        tq = (rolling_quali[rolling_quali["TeamName"] == team]
              .sort_values(["Year", "RoundNumber"]).tail(ROLLING_WINDOW * 2))
        if len(tq) > 0:
            team_avg = tq.groupby(["Year", "RoundNumber"])["GapToPole_s"].mean()
            df.loc[mask, "ConRollingQualiGap"] = _wavg(team_avg.tail(ROLLING_WINDOW))
        elif not _reg_cold_start:
            df.loc[mask, "ConRollingQualiGap"] = global_q_gap
        # cold-start + no data: leave NaN → imputed to median

    # ── 3. Circuit affinity (full history — circuit layout unchanged by regs) ──
    for driver in df["Driver"]:
        mask   = df["Driver"] == driver
        c_hist = prior_quali[
            (prior_quali["Driver"] == driver) &
            (prior_quali["CircuitShortName"] == circuit)
        ]
        d_all = prior_quali[prior_quali["Driver"] == driver]
        if len(c_hist) == 0:
            df.loc[mask, "CircuitAvgQualiGap"] = (
                float(d_all["GapToPole_s"].median()) if len(d_all) > 0 else global_q_gap
            )
            df.loc[mask, "CircuitAvgQualiPos"] = (
                float(d_all["QualiPos"].median()) if len(d_all) > 0 else global_q_pos
            )
            df.loc[mask, "CircuitVisits"] = 0.0
        else:
            df.loc[mask, "CircuitAvgQualiGap"] = float(c_hist["GapToPole_s"].mean())
            df.loc[mask, "CircuitAvgQualiPos"] = float(c_hist["QualiPos"].mean())
            df.loc[mask, "CircuitVisits"]      = float(len(c_hist))

    # ── 4. Teammate H2H (era-scoped) ──────────────────────────────────────────
    # Cold-start: many teams have new pairings in 2026 (HAM/LEC at Ferrari,
    # ANT/RUS at Mercedes). 2025 H2H where they were on different teams is
    # meaningless — correctly falls back to 0.5 for new pairings.
    for team in df["TeamName"].unique():
        team_drivers = df[df["TeamName"] == team]["Driver"].tolist()
        if len(team_drivers) < 2:
            df.loc[df["TeamName"] == team, "H2H_QualiWinRate"] = 0.5
            continue
        for driver in team_drivers:
            mask = df["Driver"] == driver
            teammate = [d for d in team_drivers if d != driver]
            driver_q   = rolling_quali[rolling_quali["Driver"] == driver]
            teammate_q = rolling_quali[rolling_quali["Driver"].isin(teammate)]
            common_sessions = set(
                zip(driver_q["Year"], driver_q["RoundNumber"])
            ) & set(zip(teammate_q["Year"], teammate_q["RoundNumber"]))
            if not common_sessions:
                df.loc[mask, "H2H_QualiWinRate"] = 0.5
                continue
            wins = 0
            for (y, r) in sorted(common_sessions)[-ROLLING_WINDOW:]:
                d_pos = driver_q[
                    (driver_q["Year"] == y) & (driver_q["RoundNumber"] == r)
                ]["QualiPos"].min()
                t_pos = teammate_q[
                    (teammate_q["Year"] == y) & (teammate_q["RoundNumber"] == r)
                ]["QualiPos"].min()
                if pd.notna(d_pos) and pd.notna(t_pos):
                    wins += int(d_pos < t_pos)
            total = min(len(common_sessions), ROLLING_WINDOW)
            df.loc[mask, "H2H_QualiWinRate"] = wins / total if total > 0 else 0.5

    # ── 5. Driver rolling race form (era-scoped) ──────────────────────────────
    for driver in df["Driver"]:
        mask = df["Driver"] == driver
        dr = (rolling_race[rolling_race["Driver"] == driver]
              .sort_values(["Year", "RoundNumber"]).tail(ROLLING_WINDOW))
        if len(dr) == 0:
            if not _reg_cold_start:
                df.loc[mask, "RollingAvgFinish"]  = global_finish
                df.loc[mask, "RollingAvgGrid"]    = global_finish
                df.loc[mask, "RollingPoints"]     = 0.0
                df.loc[mask, "RollingPodiumRate"] = 0.0
                df.loc[mask, "RollingDNFRate"]    = 0.0
                df.loc[mask, "DNFStreak"]         = 0.0
            # cold-start + no data: leave NaN → imputed to median (levels field)
        else:
            finish = pd.to_numeric(dr["FinishPos"], errors="coerce")
            points = pd.to_numeric(dr["Points"],    errors="coerce").fillna(0)
            dnf    = pd.to_numeric(dr.get("DNF", pd.Series(0, index=dr.index)),
                                   errors="coerce").fillna(0)
            df.loc[mask, "RollingAvgFinish"]  = _wavg(finish.dropna())
            df.loc[mask, "RollingAvgGrid"]    = _wavg(finish.dropna())
            df.loc[mask, "RollingPoints"]     = _wavg(points)
            df.loc[mask, "RollingPodiumRate"] = _wavg((finish <= 3).astype(float))
            df.loc[mask, "RollingDNFRate"]    = _wavg(dnf)
            df.loc[mask, "DNFStreak"]         = float(dnf.tail(3).sum())

    # ── 6. Constructor rolling race form (era-scoped) ─────────────────────────
    for team in df["TeamName"].unique():
        mask = df["TeamName"] == team
        tr = (rolling_race[rolling_race["TeamName"] == team]
              .sort_values(["Year", "RoundNumber"]).tail(ROLLING_WINDOW * 2))
        if len(tr) > 0:
            team_avg = (
                tr.groupby(["Year", "RoundNumber"])
                .agg(
                    AvgFinish=(
                        "FinishPos",
                        lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    ),
                    TotalPoints=(
                        "Points",
                        lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum(),
                    ),
                )
                .reset_index()
            )
            df.loc[mask, "ConRollingAvgFinish"] = _wavg(
                team_avg["AvgFinish"].tail(ROLLING_WINDOW)
            )
            df.loc[mask, "ConRollingPoints"] = _wavg(
                team_avg["TotalPoints"].tail(ROLLING_WINDOW)
            )
        elif not _reg_cold_start:
            df.loc[mask, "ConRollingAvgFinish"] = global_finish
            df.loc[mask, "ConRollingPoints"]    = 0.0
        # cold-start + no data: leave NaN → imputed to median

    # ── 7. Championship context (current year only) ───────────────────────────
    cy_race = prior_race[prior_race["Year"] == year]
    for driver in df["Driver"]:
        mask = df["Driver"] == driver
        pts = pd.to_numeric(
            cy_race[cy_race["Driver"] == driver].get(
                "Points", pd.Series(dtype=float)
            ),
            errors="coerce",
        ).fillna(0).sum()
        df.loc[mask, "CumPointsBefore"] = float(pts)

    max_pts = df["CumPointsBefore"].max()
    df["ChampionshipPos_norm"] = df["CumPointsBefore"] / max(float(max_pts), 1.0)

    # ── 8. Constructor championship context (FIX: was never computed) ─────────
    # ConChampDelta = points gap to the constructor championship leader.
    # ConChampPos   = constructor standing position (1 = P1).
    cy_team_pts: dict[str, float] = {}
    for team in df["TeamName"].unique():
        pts = pd.to_numeric(
            cy_race[cy_race["TeamName"] == team].get("Points", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()
        cy_team_pts[team] = float(pts)

    max_con_pts = max(cy_team_pts.values()) if cy_team_pts else 0.0
    sorted_teams = sorted(cy_team_pts.items(), key=lambda x: x[1], reverse=True)
    team_pos_map = {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}

    for team in df["TeamName"].unique():
        mask = df["TeamName"] == team
        team_pts_val = cy_team_pts.get(team, 0.0)
        df.loc[mask, "ConChampDelta"] = max(0.0, max_con_pts - team_pts_val)
        df.loc[mask, "ConChampPos"]   = float(team_pos_map.get(team, len(df["TeamName"].unique())))

    # ── 9. Career race count (FIX: was never computed, rookies looked like vets) ──
    # Count races completed by each driver prior to this round across all history.
    for driver in df["Driver"]:
        mask = df["Driver"] == driver
        count = len(prior_race[prior_race["Driver"] == driver])
        df.loc[mask, "CareerRaceCount"] = float(count)

    # ── 10. Circuit SC rate (FIX: was never computed, always imputed to median) ──
    # Look up from KNOWN_SC_RATES — same source used in features.py.
    df["CircuitSCRate"] = float(KNOWN_SC_RATES.get(circuit, DEFAULT_SC_RATE))

    # ── 11. Weather ───────────────────────────────────────────────────────────
    forecast = None
    _weather_source = "defaults"

    if _WEATHER_AVAILABLE:
        try:
            forecast = _get_weather_forecast(year, round_number)
        except Exception as exc:
            log.warning("Weather forecast failed: %s", exc)

    if forecast is not None:
        df["AirTemp_mean"]   = float(forecast["AirTemp_mean"]   or 25.0)
        df["TrackTemp_mean"] = float(forecast["TrackTemp_mean"] or 35.0)
        df["Humidity_mean"]  = float(forecast["Humidity_mean"]  or 50.0)
        df["Rainfall_any"]   = bool(forecast["Rainfall_any"])
        _weather_source = "OpenMeteo forecast"
    else:
        c_weather = prior_quali[prior_quali["CircuitShortName"] == circuit]
        if len(c_weather) > 0 and "AirTemp_mean" in c_weather.columns:
            df["AirTemp_mean"]   = float(c_weather["AirTemp_mean"].mean())
            df["TrackTemp_mean"] = float(c_weather["TrackTemp_mean"].mean())
            df["Humidity_mean"]  = float(c_weather["Humidity_mean"].mean())
            df["Rainfall_any"]   = False
            _weather_source = "historical average"
        else:
            df["AirTemp_mean"]   = 25.0
            df["TrackTemp_mean"] = 35.0
            df["Humidity_mean"]  = 50.0
            df["Rainfall_any"]   = False

    log.info("Weather source: %s (Air=%.1f°C, Rain=%s)",
             _weather_source, float(df["AirTemp_mean"].iloc[0]),
             "YES" if bool(df["Rainfall_any"].iloc[0]) else "no")

    # ── 12. Circuit flags ─────────────────────────────────────────────────────
    for col, val in circuit_type_flags(circuit).items():
        df[col] = val

    # ── 13. Regulation cycle context ──────────────────────────────────────────
    df["YearsSinceLastRegChange"] = years_since_reg_change(year)

    # ── 14. Grid penalty — default 0 (no penalty known pre-race) ──────────────
    if "GridPenaltyPlaces" not in df.columns or df["GridPenaltyPlaces"].isna().all():
        df["GridPenaltyPlaces"] = 0

    # ── 15. QualiphaseReached — imputed at median=2 pre-quali ─────────────────
    if "QualiphaseReached" not in df.columns or df["QualiphaseReached"].isna().all():
        df["QualiphaseReached"] = 2

    # ── 16. Tyre defaults ─────────────────────────────────────────────────────
    for _tyre_col, _tyre_default in [
        ("StartCompound_enc", 2),
        ("StartTyreLife",     0),
        ("FreshStartTyre",    1),
        ("RollingAvgStints",  1.5),
        ("RollingAvgDegRate", 0.05),
        ("CircuitAvgStints",  1.5),
        ("CircuitDegRate",    0.05),
    ]:
        if _tyre_col not in df.columns or df[_tyre_col].isna().all():
            df[_tyre_col] = _tyre_default

    # ── 17. Chaos flag ────────────────────────────────────────────────────────
    df["IsChaosCircuit"] = int(circuit in CHAOS_CIRCUITS)

    # ── 18. Inject live FP2/FP3 features (FIX: raw data now used directly) ────
    # Overwrite NaN placeholders with real values extracted from this weekend's
    # raw FP parquets.  Drivers with no FP data keep the NaN and will be
    # imputed via impute_prediction_features.
    fp_cols = [
        "FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank",
        "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    ]
    for driver in df["Driver"]:
        if driver not in fp_features:
            continue
        driver_fp = fp_features[driver]
        mask = df["Driver"] == driver
        for col in fp_cols:
            if col in driver_fp:
                df.loc[mask, col] = driver_fp[col]

    return df


# ── Encoding + imputation ─────────────────────────────────────────────────────

def encode_for_prediction(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        known = set(le.classes_)

        def safe_encode(val, le=le, known=known, col=col):
            val = str(val)
            if val in known:
                return int(le.transform([val])[0])
            val_lower = val.lower()
            for cls in le.classes_:
                if val_lower[:6] in cls.lower() or cls.lower()[:6] in val_lower:
                    return int(le.transform([cls])[0])
            log.warning("Unseen label '%s' for '%s' — encoding as 0", val, col)
            return 0

        df[f"{col}_enc"] = df[col].apply(safe_encode)
    return df


def impute_prediction_features(
    df: pd.DataFrame,
    feature_cols: list,
    fallback_stats: dict[str, float],
    hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill NaNs using train-set medians (from metrics JSON) with a fallback
    to current-data medians. Creates missing columns rather than crashing.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            median = fallback_stats.get(
                col,
                pd.to_numeric(hist_df[col], errors="coerce").median()
                if col in hist_df.columns else 0.0,
            )
            df[col] = median
        elif df[col].isnull().any():
            median = fallback_stats.get(
                col,
                pd.to_numeric(hist_df[col], errors="coerce").median()
                if col in hist_df.columns else 0.0,
            )
            df[col] = df[col].fillna(median)
    return df


# ── Main prediction pipeline ──────────────────────────────────────────────────

def predict(year: int, round_number: int) -> pd.DataFrame:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _check_stale_data()
    _check_processed_stale(year, round_number)

    q_model              = load_quali_model()
    r_model, r_calibrator = load_race_model()
    q_encoders           = load_encoders("quali_model")
    r_encoders           = load_encoders("race_model")
    q_stats              = load_impute_stats("quali_model")
    r_stats              = load_impute_stats("race_model")

    hist_quali = load_historical_quali_best()
    hist_race  = load_historical_race_results()

    # Load live FP features for this round from raw parquets
    fp_features = _load_current_round_fp_features(year, round_number)

    entry = fetch_entry_list(year, round_number)

    if entry.empty:
        raise RuntimeError(
            f"Entry list for {year} R{round_number} is empty. "
            "The session data is not yet available in FastF1."
        )

    log.info(
        "Predicting: %s %d Round %d (%d drivers)",
        entry["EventName"].iloc[0], year, round_number, len(entry),
    )

    df = build_prediction_features(
        entry, year, round_number, hist_quali, hist_race, fp_features
    )
    circuit = df["CircuitShortName"].iloc[0]

    # ── Stage 1: Quali grid ───────────────────────────────────────────────────
    df_q = encode_for_prediction(df, q_encoders)
    df_q = impute_prediction_features(df_q, QUALI_FEATURES, q_stats, hist_quali)

    X_q = df_q[QUALI_FEATURES].copy().reset_index(drop=True)
    q_scores = q_model.predict(X_q, validate_features=False)
    df_q["_score"] = q_scores
    pred_quali_pos = (
        df_q["_score"].rank(ascending=False, method="first").astype(int).values
    )
    df_q = df_q.drop(columns=["_score"])
    df["PredictedQualiPos"] = pred_quali_pos
    df["PredictedGapToPole_s"] = (df["PredictedQualiPos"] - 1) * 0.10

    # Update QualiphaseReached from predicted grid position
    df["QualiphaseReached"] = df["PredictedQualiPos"].apply(
        lambda p: 3 if p <= 10 else (2 if p <= 15 else 1)
    )

    log.info("Predicted qualifying grid:")
    for _, row in df.sort_values("PredictedQualiPos").iterrows():
        log.info("  P%-2d  %-4s  %s",
                 row["PredictedQualiPos"], row["Driver"], row["TeamName"])

    # ── Stage 2: Race podium ──────────────────────────────────────────────────
    df["QualiPos"]    = df["PredictedQualiPos"]
    df["GapToPole_s"] = df["PredictedGapToPole_s"]

    circuit_hist = hist_quali[
        (hist_quali["CircuitShortName"] == circuit) &
        (hist_quali["Year"] >= year - 3)
    ]
    pole_base = (
        float(circuit_hist["BestQualiTime_s"].min())
        if len(circuit_hist) > 0 and "BestQualiTime_s" in circuit_hist.columns
        else 80.0
    )
    df["BestQualiTime_s"] = pole_base + df["PredictedQualiPos"] * 0.1

    df["GridDifficultyScore"] = df.apply(
        lambda row: grid_difficulty_score(row["PredictedQualiPos"], circuit), axis=1
    )

    df_r = encode_for_prediction(df, r_encoders)
    df_r = impute_prediction_features(df_r, RACE_FEATURES, r_stats, hist_race)

    X_r = df_r[RACE_FEATURES].copy().reset_index(drop=True)
    raw_probs = r_model.predict_proba(X_r, validate_features=False)[:, 1]
    if r_calibrator is not None:
        podium_probs = r_calibrator.predict(raw_probs)
    else:
        podium_probs = raw_probs
    df["PodiumProbability"] = podium_probs

    prob_sum = df["PodiumProbability"].sum()
    df["PodiumProbability_norm"] = (
        (df["PodiumProbability"] / prob_sum) * 3 if prob_sum > 0
        else df["PodiumProbability"]
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output = (
        df[["Driver", "TeamName", "PredictedQualiPos",
            "PodiumProbability", "PodiumProbability_norm"]]
        .copy()
        .sort_values("PodiumProbability", ascending=False)
        .reset_index(drop=True)
    )
    output["PredictedRaceRank"] = output.index + 1

    log.info("=" * 62)
    log.info("RACE PREDICTION — %s %d Round %d",
             entry["EventName"].iloc[0], year, round_number)
    log.info("=" * 62)
    log.info("%-5s %-5s %-6s %-22s %s", "Race", "Grid", "Driver", "Team", "Podium%")
    log.info("-" * 62)
    for _, row in output.iterrows():
        log.info(
            "P%-4d P%-4d %-6s %-22s %.1f%%",
            row["PredictedRaceRank"], row["PredictedQualiPos"],
            row["Driver"], row["TeamName"][:22],
            row["PodiumProbability"] * 100,
        )
    log.info("=" * 62)

    out_path = PREDICTIONS_DIR / f"{year}_R{round_number:02d}_prediction.json"
    with open(out_path, "w") as f:
        json.dump({
            "year":    year,
            "round":   round_number,
            "event":   entry["EventName"].iloc[0],
            "circuit": entry["CircuitShortName"].iloc[0],
            "predictions": output[[
                "Driver", "TeamName",
                "PredictedQualiPos", "PredictedRaceRank",
                "PodiumProbability", "PodiumProbability_norm",
            ]].to_dict(orient="records"),
        }, f, indent=2)
    log.info("Saved → %s", out_path)

    return output


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    args = parser.parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    predict(args.year, args.round_number)


if __name__ == "__main__":
    main()