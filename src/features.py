"""
features.py

Builds two feature tables from raw Parquet files.

Changes in this version (items #5, #9, #10):
  - compute_constructor_championship_context(): ConChampDelta, ConChampPos
  - compute_circuit_sc_rate(): historical SC/VSC fraction per circuit
  - compute_career_race_count(): CareerRaceCount for cold-start handling

Run from f1-predictor root:
    python src/features.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CHAOS_CIRCUITS,
    COMPOUND_ORDER,
    ROLLING_WINDOW,
    circuit_type_flags,
    grid_difficulty_score,
    normalise_team,
    years_since_reg_change,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Long-run detection: laps slower than this multiple of session best
# are considered fuel-load laps (race simulation)
LONG_RUN_THRESHOLD = 1.07


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_driver_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Abbreviation" in df.columns and "Driver" not in df.columns:
        df = df.rename(columns={"Abbreviation": "Driver"})
    return df


def _normalise_team_col(df: pd.DataFrame) -> pd.DataFrame:
    if "TeamName" in df.columns:
        df = df.copy()
        df["TeamName"] = df["TeamName"].map(normalise_team)
    return df


def _weighted_rolling(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)

    def _wavg(x: np.ndarray) -> float:
        w = weights[-len(x):]
        return float(np.dot(x, w) / w.sum())

    return series.shift(1).rolling(window, min_periods=1).apply(_wavg, raw=True)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_all_sessions(session_type: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"*_{session_type}_laps.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No laps parquets for '{session_type}'. Run ingest.py first."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log.info("Loaded %d %s lap rows from %d files", len(df), session_type, len(files))
    return df


def load_all_results(session_type: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"*_{session_type}_results.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No results parquets for '{session_type}'. Run ingest.py first."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _normalise_team_col(df)
    log.info("Loaded %d %s result rows from %d files", len(df), session_type, len(files))
    return df


def load_all_weather(session_type: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"*_{session_type}_weather.parquet"))
    if not files:
        log.warning("No weather parquets for '%s' — will be imputed.", session_type)
        return pd.DataFrame(columns=[
            "Year", "RoundNumber",
            "AirTemp_mean", "TrackTemp_mean", "Humidity_mean", "Rainfall_any",
        ])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    agg = (
        df.groupby(["Year", "RoundNumber"])
        .agg(
            AirTemp_mean=("AirTemp",   "mean"),
            TrackTemp_mean=("TrackTemp", "mean"),
            Humidity_mean=("Humidity",  "mean"),
            Rainfall_any=("Rainfall",  "any"),
        )
        .reset_index()
    )
    log.info("Summarised weather for %d sessions", len(agg))
    return agg


def load_all_sectors() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*_Q_sectors.parquet"))
    if not files:
        log.warning("No sector parquets. Run openf1.py. Sector features will be imputed.")
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "S1Gap_s", "S2Gap_s", "S3Gap_s", "SpeedTrap_kph",
        ])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log.info("Loaded sector data: %d rows from %d files", len(df), len(files))
    return df


def _load_fp_laps(session_type: str) -> pd.DataFrame:
    """
    Load FP laps, returning an empty DataFrame if none exist.
    Normalises LapTime to seconds and driver column.
    """
    files = sorted(RAW_DIR.glob(f"*_{session_type}_laps.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = _normalise_driver_col(df)
    df["LapTime_s"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df.dropna(subset=["LapTime_s"])
    log.info("Loaded %d %s laps from %d files", len(df), session_type, len(files))
    return df


# ── FP3 pace extraction (item #2) ─────────────────────────────────────────────

def extract_fp3_pace(fp3_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract FP3 qualifying-simulation pace per driver per session.

    FP3 is the final 60-minute session before qualifying. Teams
    typically run a qualifying simulation on fresh soft tyres, making
    FP3 the best single proxy for qualifying order.

    Features:
      FP3_BestLap_s       — driver's fastest clean lap in FP3
      FP3_GapToFastest_s  — gap to session fastest (0 = fastest driver)
      FP3_PaceRank        — rank within session (1 = fastest)

    Returns empty DataFrame with correct schema if no FP3 data exists.
    """
    if fp3_laps.empty:
        log.warning("No FP3 data available — FP3 features will be imputed")
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank",
        ])

    laps = fp3_laps.copy()

    # Drop outliers: laps > 15% slower than session median
    session_median = laps.groupby(
        ["Year", "RoundNumber"]
    )["LapTime_s"].transform("median")
    laps = laps[laps["LapTime_s"] <= session_median * 1.15]

    best = (
        laps.groupby(["Year", "RoundNumber", "Driver"])["LapTime_s"]
        .min()
        .reset_index()
        .rename(columns={"LapTime_s": "FP3_BestLap_s"})
    )

    fastest = (
        best.groupby(["Year", "RoundNumber"])["FP3_BestLap_s"]
        .min()
        .reset_index()
        .rename(columns={"FP3_BestLap_s": "FP3_Fastest_s"})
    )
    best = best.merge(fastest, on=["Year", "RoundNumber"])
    best["FP3_GapToFastest_s"] = best["FP3_BestLap_s"] - best["FP3_Fastest_s"]
    best["FP3_PaceRank"] = (
        best.groupby(["Year", "RoundNumber"])["FP3_BestLap_s"]
        .rank(method="min").astype(int)
    )
    best = best.drop(columns=["FP3_Fastest_s"])

    log.info("Extracted FP3 pace: %d rows (%d sessions)",
             len(best), best[["Year", "RoundNumber"]].drop_duplicates().__len__())
    return best


# ── FP2 long-run extraction (item #2) ────────────────────────────────────────

def extract_fp2_longruns(fp2_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract race-simulation (long-run) pace from FP2 per driver per session.

    Long runs = consecutive laps on a heavy fuel load, representing race pace.
    We identify them as laps slower than LONG_RUN_THRESHOLD × session best,
    then take each driver's average of those laps as their long-run pace.

    A driver fast in FP2 long runs but mid-grid in qualifying will often
    overperform their starting position in the race — exactly the signal
    that was missing from the race model.

    Features:
      FP2_LongRunPace_s  — driver's avg long-run lap time
      FP2_LongRunRank    — rank by long-run pace within session (1 = fastest)
      FP2_LongRunDelta_s — delta to session best long-run pace

    Returns empty DataFrame with correct schema if no FP2 data exists.
    """
    if fp2_laps.empty:
        log.warning("No FP2 data available — FP2 features will be imputed")
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
        ])

    laps = fp2_laps.copy()

    # Session fastest lap — used to classify laps as long-run
    session_best = (
        laps.groupby(["Year", "RoundNumber"])["LapTime_s"]
        .min()
        .reset_index()
        .rename(columns={"LapTime_s": "SessionBest_s"})
    )
    laps = laps.merge(session_best, on=["Year", "RoundNumber"])

    # Long-run laps: slower than threshold × session best but not extreme outliers
    laps["IsLongRun"] = (
        (laps["LapTime_s"] >= laps["SessionBest_s"] * LONG_RUN_THRESHOLD) &
        (laps["LapTime_s"] <= laps["SessionBest_s"] * 1.25)
    )
    long_run_laps = laps[laps["IsLongRun"]]

    if long_run_laps.empty:
        log.warning("No long-run laps identified in FP2 — threshold may be too strict")
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
        ])

    pace = (
        long_run_laps.groupby(["Year", "RoundNumber", "Driver"])["LapTime_s"]
        .mean()
        .reset_index()
        .rename(columns={"LapTime_s": "FP2_LongRunPace_s"})
    )

    # Require at least 3 long-run laps for reliability
    counts = (
        long_run_laps.groupby(["Year", "RoundNumber", "Driver"])["LapTime_s"]
        .count()
        .reset_index()
        .rename(columns={"LapTime_s": "LongRunCount"})
    )
    pace = pace.merge(counts, on=["Year", "RoundNumber", "Driver"])
    pace = pace[pace["LongRunCount"] >= 3].drop(columns=["LongRunCount"])

    # Rank within session
    pace["FP2_LongRunRank"] = (
        pace.groupby(["Year", "RoundNumber"])["FP2_LongRunPace_s"]
        .rank(method="min").astype(int)
    )

    # Delta to session best long-run pace
    session_best_lr = (
        pace.groupby(["Year", "RoundNumber"])["FP2_LongRunPace_s"]
        .min()
        .reset_index()
        .rename(columns={"FP2_LongRunPace_s": "BestLR_s"})
    )
    pace = pace.merge(session_best_lr, on=["Year", "RoundNumber"])
    pace["FP2_LongRunDelta_s"] = pace["FP2_LongRunPace_s"] - pace["BestLR_s"]
    pace = pace.drop(columns=["BestLR_s"])

    n_sessions = pace[["Year", "RoundNumber"]].drop_duplicates().__len__()
    log.info("Extracted FP2 long-run pace: %d rows (%d sessions)", len(pace), n_sessions)
    return pace


# ── Quali best lap extraction ─────────────────────────────────────────────────

def extract_quali_best(q_laps: pd.DataFrame) -> pd.DataFrame:
    q_laps = _normalise_driver_col(q_laps.copy())
    q_laps["LapTime_s"] = pd.to_timedelta(q_laps["LapTime"]).dt.total_seconds()
    q_laps = q_laps.dropna(subset=["LapTime_s"])

    if q_laps.empty:
        raise ValueError("No valid lap times found in qualifying laps data.")

    session_median = q_laps.groupby(
        ["Year", "RoundNumber"]
    )["LapTime_s"].transform("median")
    q_laps = q_laps[q_laps["LapTime_s"] <= session_median * 1.20]

    best = (
        q_laps.groupby(["Year", "RoundNumber", "Driver"])["LapTime_s"]
        .min().reset_index()
        .rename(columns={"LapTime_s": "BestQualiTime_s"})
    )
    pole = (
        best.groupby(["Year", "RoundNumber"])["BestQualiTime_s"]
        .min().reset_index()
        .rename(columns={"BestQualiTime_s": "PoleTime_s"})
    )
    best = best.merge(pole, on=["Year", "RoundNumber"])
    best["GapToPole_s"] = best["BestQualiTime_s"] - best["PoleTime_s"]
    best["QualiPos"] = (
        best.groupby(["Year", "RoundNumber"])["BestQualiTime_s"]
        .rank(method="min").astype(int)
    )
    log.info("Extracted quali best laps: %d rows", len(best))
    return best


# ── Rolling quali form ────────────────────────────────────────────────────────

def compute_driver_rolling_quali_form(best_quali: pd.DataFrame) -> pd.DataFrame:
    df = best_quali[[
        "Year", "RoundNumber", "Driver", "GapToPole_s", "QualiPos"
    ]].copy()
    df = df.sort_values(["Driver", "Year", "RoundNumber"]).reset_index(drop=True)

    rows = []
    for driver, grp in df.groupby("Driver", sort=False):
        grp = grp.copy()
        grp["RollingQualiGap"]    = _weighted_rolling(grp["GapToPole_s"], ROLLING_WINDOW)
        grp["RollingQualiPos"]    = _weighted_rolling(grp["QualiPos"],    ROLLING_WINDOW)
        grp["RollingQualiStdGap"] = (
            grp["GapToPole_s"].shift(1)
            .rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
        )
        rows.append(grp)

    form = pd.concat(rows, ignore_index=True)
    log.info("Computed rolling driver quali form: %d rows", len(form))
    return form


def compute_constructor_rolling_quali_form(
    best_quali: pd.DataFrame, q_results: pd.DataFrame
) -> pd.DataFrame:
    q_res = _normalise_driver_col(q_results.copy())
    merged = best_quali.merge(
        q_res[["Year", "RoundNumber", "Driver", "TeamName"]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    merged = merged.sort_values(
        ["TeamName", "Year", "RoundNumber"]
    ).reset_index(drop=True)

    rows = []
    for team, grp in merged.groupby("TeamName", sort=False):
        team_avg = (
            grp.groupby(["Year", "RoundNumber"])
            .agg(TeamAvgQualiGap=("GapToPole_s", "mean"))
            .reset_index()
            .sort_values(["Year", "RoundNumber"])
        )
        team_avg["ConRollingQualiGap"] = _weighted_rolling(
            team_avg["TeamAvgQualiGap"], ROLLING_WINDOW
        )
        team_avg["TeamName"] = team
        rows.append(team_avg)

    con_form = pd.concat(rows, ignore_index=True)
    log.info("Computed constructor rolling quali form: %d rows", len(con_form))
    return con_form


# ── Teammate H2H ──────────────────────────────────────────────────────────────

def compute_teammate_h2h(
    best_quali: pd.DataFrame, q_results: pd.DataFrame
) -> pd.DataFrame:
    q_res = _normalise_driver_col(q_results.copy())
    df = best_quali.merge(
        q_res[["Year", "RoundNumber", "Driver", "TeamName"]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )

    h2h_rows = []
    for (year, rnd, team), grp in df.groupby(["Year", "RoundNumber", "TeamName"]):
        if len(grp) < 2:
            continue
        for i, driver in enumerate(grp.sort_values("QualiPos")["Driver"].tolist()):
            h2h_rows.append({
                "Year": year, "RoundNumber": rnd,
                "Driver": driver, "BeatTeammate": int(i == 0),
            })

    if not h2h_rows:
        result = best_quali[["Year", "RoundNumber", "Driver"]].copy()
        result["H2H_QualiWinRate"] = 0.5
        return result

    h2h = pd.DataFrame(h2h_rows).sort_values(
        ["Driver", "Year", "RoundNumber"]
    ).reset_index(drop=True)

    rows = []
    for driver, grp in h2h.groupby("Driver", sort=False):
        grp = grp.copy()
        grp["H2H_QualiWinRate"] = (
            grp["BeatTeammate"].shift(1)
            .rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
        rows.append(grp)

    result = pd.concat(rows, ignore_index=True)
    result["H2H_QualiWinRate"] = result["H2H_QualiWinRate"].fillna(0.5)
    log.info("Computed teammate H2H: %d rows", len(result))
    return result[["Year", "RoundNumber", "Driver", "H2H_QualiWinRate"]]


# ── Driver-circuit affinity ───────────────────────────────────────────────────

def compute_driver_circuit_affinity(
    best_quali: pd.DataFrame, q_results: pd.DataFrame
) -> pd.DataFrame:
    q_res = _normalise_driver_col(q_results.copy())
    df = best_quali.merge(
        q_res[["Year", "RoundNumber", "Driver", "CircuitShortName"]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.dropna(subset=["CircuitShortName"])
    df = df.sort_values(
        ["Driver", "CircuitShortName", "Year", "RoundNumber"]
    ).reset_index(drop=True)

    rows = []
    for (driver, circuit), grp in df.groupby(
        ["Driver", "CircuitShortName"], sort=False
    ):
        grp = grp.copy()
        grp["CircuitAvgQualiGap"] = grp["GapToPole_s"].shift(1).expanding().mean()
        grp["CircuitAvgQualiPos"] = grp["QualiPos"].shift(1).expanding().mean()
        grp["CircuitVisits"]      = grp["GapToPole_s"].shift(1).expanding().count()
        rows.append(grp)

    affinity = pd.concat(rows, ignore_index=True)
    for col in ["CircuitAvgQualiGap", "CircuitAvgQualiPos"]:
        driver_median = affinity.groupby("Driver")[col].transform("median")
        affinity[col] = affinity[col].fillna(driver_median)
        affinity[col] = affinity[col].fillna(affinity[col].median())
    affinity["CircuitVisits"] = affinity["CircuitVisits"].fillna(0)

    keep = ["Year", "RoundNumber", "Driver",
            "CircuitAvgQualiGap", "CircuitAvgQualiPos", "CircuitVisits"]
    log.info("Computed driver-circuit affinity: %d rows", len(affinity))
    return affinity[keep].drop_duplicates()


# ── Sector rolling form ───────────────────────────────────────────────────────

def compute_driver_rolling_sector_form(sectors: pd.DataFrame) -> pd.DataFrame:
    if sectors.empty:
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "RollingS1Gap", "RollingS2Gap", "RollingS3Gap", "RollingSpeedTrap",
        ])

    df = sectors[[
        "Year", "RoundNumber", "Driver",
        "S1Gap_s", "S2Gap_s", "S3Gap_s", "SpeedTrap_kph",
    ]].copy()
    df = df.sort_values(["Driver", "Year", "RoundNumber"]).reset_index(drop=True)

    rows = []
    for driver, grp in df.groupby("Driver", sort=False):
        grp = grp.copy()
        grp["RollingS1Gap"]     = _weighted_rolling(grp["S1Gap_s"],       ROLLING_WINDOW)
        grp["RollingS2Gap"]     = _weighted_rolling(grp["S2Gap_s"],       ROLLING_WINDOW)
        grp["RollingS3Gap"]     = _weighted_rolling(grp["S3Gap_s"],       ROLLING_WINDOW)
        grp["RollingSpeedTrap"] = _weighted_rolling(grp["SpeedTrap_kph"], ROLLING_WINDOW)
        rows.append(grp)

    form = pd.concat(rows, ignore_index=True)
    log.info("Computed rolling sector form: %d rows", len(form))
    return form


# ── Rolling race form ─────────────────────────────────────────────────────────

def compute_driver_rolling_form(race_results: pd.DataFrame) -> pd.DataFrame:
    r = _normalise_driver_col(race_results.copy())
    r["FinishPos"] = pd.to_numeric(r.get("Position",     np.nan), errors="coerce")
    r["GridPos"]   = pd.to_numeric(r.get("GridPosition", np.nan), errors="coerce")
    r["Points"]    = pd.to_numeric(r.get("Points", 0),            errors="coerce").fillna(0)
    r["Podium"]    = (r["FinishPos"] <= 3).astype(int)
    r["DNF"] = r["Status"].apply(
        lambda s: 0 if str(s).startswith("Finished") or str(s).startswith("+") else 1
    )
    r = r.sort_values(["Driver", "Year", "RoundNumber"]).reset_index(drop=True)

    rows = []
    for driver, grp in r.groupby("Driver", sort=False):
        grp = grp.copy()
        grp["RollingAvgFinish"]  = _weighted_rolling(grp["FinishPos"], ROLLING_WINDOW)
        grp["RollingAvgGrid"]    = _weighted_rolling(grp["GridPos"],   ROLLING_WINDOW)
        grp["RollingPoints"]     = _weighted_rolling(grp["Points"],    ROLLING_WINDOW)
        grp["RollingPodiumRate"] = _weighted_rolling(grp["Podium"],    ROLLING_WINDOW)
        grp["RollingDNFRate"]    = _weighted_rolling(grp["DNF"],       ROLLING_WINDOW)
        grp["DNFStreak"]         = grp["DNF"].shift(1).rolling(3, min_periods=1).sum()
        rows.append(grp)

    form = pd.concat(rows, ignore_index=True)
    log.info("Computed rolling driver race form: %d rows", len(form))
    return form


def compute_constructor_rolling_form(race_results: pd.DataFrame) -> pd.DataFrame:
    r = _normalise_driver_col(race_results.copy())
    r["FinishPos"] = pd.to_numeric(r.get("Position", np.nan), errors="coerce")
    r["Points"]    = pd.to_numeric(r.get("Points", 0),        errors="coerce").fillna(0)
    r = r.sort_values(["TeamName", "Year", "RoundNumber"]).reset_index(drop=True)

    rows = []
    for team, grp in r.groupby("TeamName", sort=False):
        team_avg = (
            grp.groupby(["Year", "RoundNumber"])
            .agg(TeamAvgFinish=("FinishPos", "mean"), TeamPoints=("Points", "sum"))
            .reset_index()
            .sort_values(["Year", "RoundNumber"])
        )
        team_avg["ConRollingAvgFinish"] = _weighted_rolling(
            team_avg["TeamAvgFinish"], ROLLING_WINDOW
        )
        team_avg["ConRollingPoints"] = _weighted_rolling(
            team_avg["TeamPoints"], ROLLING_WINDOW
        )
        team_avg["TeamName"] = team
        rows.append(team_avg)

    con_form = pd.concat(rows, ignore_index=True)
    log.info("Computed constructor rolling race form: %d rows", len(con_form))
    return con_form


# ── Championship context ──────────────────────────────────────────────────────

def compute_championship_context(race_results: pd.DataFrame) -> pd.DataFrame:
    r = _normalise_driver_col(race_results.copy())
    r["Points"] = pd.to_numeric(r.get("Points", 0), errors="coerce").fillna(0)
    r = r.sort_values(["Year", "RoundNumber", "Driver"])
    r["CumPointsBefore"] = (
        r.groupby(["Year", "Driver"])["Points"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    max_pts = r.groupby("Year")["CumPointsBefore"].transform("max").replace(0, 1)
    r["ChampionshipPos_norm"] = r["CumPointsBefore"] / max_pts
    return r[[
        "Year", "RoundNumber", "Driver",
        "CumPointsBefore", "ChampionshipPos_norm",
    ]].drop_duplicates()



# ── Qualifying phase reached ──────────────────────────────────────────────────

def compute_quali_phase(best_quali: pd.DataFrame) -> pd.DataFrame:
    """
    Which knockout phase each driver reached:
      P1-P10  -> Q3 (value=3)
      P11-P15 -> Q2 (value=2)
      P16+    -> Q1 (value=1)

    Leakage note: derived from QualiPos (the target). Used as a race-model
    input (encoding the known qualifying result) and as a historical feature
    for prior sessions. For the current qualifying session it defaults to 2.
    """
    df = best_quali[["Year", "RoundNumber", "Driver", "QualiPos"]].copy()
    df["QualiphaseReached"] = df["QualiPos"].apply(
        lambda p: 3 if p <= 10 else (2 if p <= 15 else 1)
    )
    return df[["Year", "RoundNumber", "Driver", "QualiphaseReached"]]


# ── Grid penalty places ───────────────────────────────────────────────────────

def compute_grid_penalties(
    race_results: pd.DataFrame,
    best_quali: pd.DataFrame,
) -> pd.DataFrame:
    """
    GridPenaltyPlaces = actual race grid position minus qualifying position.
    Positive -> dropped back (penalty), negative -> promoted, zero -> no change.
    Defaults to 0 when GridPosition column is absent.
    """
    r = _normalise_driver_col(race_results.copy())

    if "GridPosition" not in r.columns:
        df = best_quali[["Year", "RoundNumber", "Driver"]].copy()
        df["GridPenaltyPlaces"] = 0
        return df

    r["GridPos_actual"] = pd.to_numeric(r["GridPosition"], errors="coerce")
    merged = r[["Year", "RoundNumber", "Driver", "GridPos_actual"]].merge(
        best_quali[["Year", "RoundNumber", "Driver", "QualiPos"]],
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    merged["GridPenaltyPlaces"] = (
        (merged["GridPos_actual"] - merged["QualiPos"])
        .fillna(0).clip(-5, 15).astype(int)
    )
    return merged[["Year", "RoundNumber", "Driver", "GridPenaltyPlaces"]]


# ── Items #3 + #8: Tyre compound, stint count, and degradation rate ───────────

def compute_tyre_features(r_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract tyre-related features from race lap data.

    Features:
      StartCompound_enc  — ordinal encoded starting compound (Soft=1 Med=2 Hard=3)
      StartTyreLife      — laps old the starting tyre was (0 = fresh set)
      FreshStartTyre     — 1 if StartTyreLife == 0 else 0
      RollingAvgStints   — driver rolling avg pit stops (recency-weighted)
      RollingAvgDegRate  — driver rolling avg degradation rate (s/lap within stint)
      CircuitAvgStints   — circuit historical avg pit stops (expanding mean)
      CircuitDegRate     — circuit historical avg deg rate (expanding mean)

    Compound and TyreLife columns available from FastF1 2019+.
    Pre-2019 rows are imputed with neutral defaults.
    All rolling features use shift(1) — no leakage.
    """
    if r_laps.empty:
        log.warning("No race laps — tyre features will be imputed")
        return pd.DataFrame(columns=[
            "Year", "RoundNumber", "Driver",
            "StartCompound_enc", "StartTyreLife", "FreshStartTyre",
            "RollingAvgStints", "RollingAvgDegRate",
            "CircuitAvgStints", "CircuitDegRate",
        ])

    laps = _normalise_driver_col(r_laps.copy())
    laps["LapTime_s"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

    # ── Starting compound & tyre life (lap 1) ────────────────────────────────
    if "LapNumber" in laps.columns and "Compound" in laps.columns:
        lap1 = laps[laps["LapNumber"] == 1].copy()
        lap1["Compound"] = lap1["Compound"].fillna("UNKNOWN").str.upper()
        lap1["StartCompound_enc"] = lap1["Compound"].map(COMPOUND_ORDER).fillna(0).astype(int)
        if "TyreLife" in lap1.columns:
            lap1["StartTyreLife"] = pd.to_numeric(lap1["TyreLife"], errors="coerce").fillna(0).clip(0, 50).astype(int)
        else:
            lap1["StartTyreLife"] = 0
        lap1["FreshStartTyre"] = (lap1["StartTyreLife"] == 0).astype(int)
        start_tyres = (
            lap1[["Year", "RoundNumber", "Driver",
                  "StartCompound_enc", "StartTyreLife", "FreshStartTyre"]]
            .drop_duplicates(subset=["Year", "RoundNumber", "Driver"])
        )
    else:
        base_d = laps[["Year", "RoundNumber", "Driver"]].drop_duplicates()
        base_d["StartCompound_enc"] = 2
        base_d["StartTyreLife"]     = 0
        base_d["FreshStartTyre"]    = 1
        start_tyres = base_d

    # ── Pit stop count (number of pit-out events per race) ────────────────────
    if "PitOutTime" in laps.columns:
        pit_counts = (
            laps[laps["PitOutTime"].notna()]
            .groupby(["Year", "RoundNumber", "Driver"])
            .size()
            .reset_index(name="NumPitStops")
        )
        pit_counts["NumPitStops"] = pit_counts["NumPitStops"].clip(0, 6)
    else:
        pit_counts = laps[["Year", "RoundNumber", "Driver"]].drop_duplicates().copy()
        pit_counts["NumPitStops"] = np.nan

    # ── Tyre degradation rate (s/lap within a stint) ──────────────────────────
    if "Stint" in laps.columns:
        clean = laps.dropna(subset=["LapTime_s", "Stint"]).copy()
        clean = clean[clean["LapTime_s"] > 0]
        clean["StintLap"] = clean.groupby(
            ["Year", "RoundNumber", "Driver", "Stint"]
        ).cumcount() + 1
        # Use laps 3-15 within a stint (avoid first outlap + SC laps)
        clean = clean[(clean["StintLap"] >= 3) & (clean["StintLap"] <= 15)]

        if len(clean) > 100:
            def _slope(grp):
                if len(grp) < 3:
                    return np.nan
                x = grp["StintLap"].values.astype(float)
                y = grp["LapTime_s"].values
                xb, yb = x.mean(), y.mean()
                denom = ((x - xb) ** 2).sum()
                return float(((x - xb) * (y - yb)).sum() / denom) if denom > 1e-9 else 0.0

            stint_deg = (
                clean.groupby(["Year", "RoundNumber", "Driver", "Stint"])
                .apply(_slope, include_groups=False)
                .reset_index(name="StintDegRate")
            )
            race_deg = (
                stint_deg.groupby(["Year", "RoundNumber", "Driver"])["StintDegRate"]
                .mean().reset_index(name="RaceDegRate")
            )
        else:
            race_deg = laps[["Year", "RoundNumber", "Driver"]].drop_duplicates().copy()
            race_deg["RaceDegRate"] = np.nan
    else:
        race_deg = laps[["Year", "RoundNumber", "Driver"]].drop_duplicates().copy()
        race_deg["RaceDegRate"] = np.nan

    # ── Assemble per-race tyre table ──────────────────────────────────────────
    base = laps[["Year", "RoundNumber", "Driver", "CircuitShortName"]].drop_duplicates(
        subset=["Year", "RoundNumber", "Driver"]
    )
    result = (
        base
        .merge(start_tyres, on=["Year", "RoundNumber", "Driver"], how="left")
        .merge(pit_counts[["Year", "RoundNumber", "Driver", "NumPitStops"]],
               on=["Year", "RoundNumber", "Driver"], how="left")
        .merge(race_deg,    on=["Year", "RoundNumber", "Driver"], how="left")
    )
    result["StartCompound_enc"] = result["StartCompound_enc"].fillna(2).astype(int)
    result["StartTyreLife"]     = result["StartTyreLife"].fillna(0).astype(int)
    result["FreshStartTyre"]    = result["FreshStartTyre"].fillna(1).astype(int)

    # ── Rolling driver averages (leakage-safe) ────────────────────────────────
    result = result.sort_values(["Driver", "Year", "RoundNumber"]).reset_index(drop=True)
    rows = []
    for driver, grp in result.groupby("Driver", sort=False):
        grp = grp.copy()
        pit_med = grp["NumPitStops"].median()
        deg_med = grp["RaceDegRate"].median()
        grp["RollingAvgStints"]  = _weighted_rolling(
            grp["NumPitStops"].fillna(pit_med if pd.notna(pit_med) else 1.5), ROLLING_WINDOW
        )
        grp["RollingAvgDegRate"] = _weighted_rolling(
            grp["RaceDegRate"].fillna(deg_med if pd.notna(deg_med) else 0.05), ROLLING_WINDOW
        )
        rows.append(grp)
    result = pd.concat(rows, ignore_index=True)

    # ── Circuit-level historical averages (expanding, leakage-safe) ──────────
    result = result.sort_values(["CircuitShortName", "Year", "RoundNumber"]).reset_index(drop=True)
    circ_rows = []
    for circuit, grp in result.groupby("CircuitShortName", sort=False):
        circ_avg = (
            grp.groupby(["Year", "RoundNumber"])
            .agg(AvgStints=("NumPitStops", "mean"), AvgDeg=("RaceDegRate", "mean"))
            .reset_index()
            .sort_values(["Year", "RoundNumber"])
        )
        circ_avg["CircuitAvgStints"] = circ_avg["AvgStints"].shift(1).expanding().mean()
        circ_avg["CircuitDegRate"]   = circ_avg["AvgDeg"].shift(1).expanding().mean()
        circ_avg["CircuitShortName"] = circuit
        circ_rows.append(circ_avg[["Year", "RoundNumber", "CircuitShortName",
                                    "CircuitAvgStints", "CircuitDegRate"]])
    circ_features = pd.concat(circ_rows, ignore_index=True)

    result = result.merge(
        circ_features, on=["Year", "RoundNumber", "CircuitShortName"], how="left"
    )
    result["CircuitAvgStints"] = result["CircuitAvgStints"].fillna(
        result["CircuitAvgStints"].median()
    ).fillna(1.5)
    result["CircuitDegRate"] = result["CircuitDegRate"].fillna(
        result["CircuitDegRate"].median()
    ).fillna(0.05)
    result["RollingAvgStints"]  = result["RollingAvgStints"].fillna(1.5)
    result["RollingAvgDegRate"] = result["RollingAvgDegRate"].fillna(0.05)

    keep = ["Year", "RoundNumber", "Driver",
            "StartCompound_enc", "StartTyreLife", "FreshStartTyre",
            "RollingAvgStints", "RollingAvgDegRate",
            "CircuitAvgStints", "CircuitDegRate"]
    log.info("Computed tyre features: %d rows", len(result))
    return result[keep].drop_duplicates(subset=["Year", "RoundNumber", "Driver"])


# ── Item #21: Chaos circuit flag ──────────────────────────────────────────────

def compute_chaos_flag(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    IsChaosCircuit = 1 for circuits with historically high SC/VSC deployment.
    These circuits have more random race outcomes — the model learns that
    grid position is less deterministic here.
    Defined by CHAOS_CIRCUITS in config.py (SC rate > 0.25 historically).
    """
    r = _normalise_driver_col(race_results.copy())
    df = r[["Year", "RoundNumber", "Driver", "CircuitShortName"]].drop_duplicates()
    df = df.copy()
    df["IsChaosCircuit"] = df["CircuitShortName"].apply(
        lambda c: int(str(c) in CHAOS_CIRCUITS)
    )
    log.info("Computed chaos circuit flags: %d rows", len(df))
    return df[["Year", "RoundNumber", "Driver", "IsChaosCircuit"]]

# ── Master feature builders ───────────────────────────────────────────────────

# ── Item #10: Career race count ───────────────────────────────────────────────

def compute_career_race_count(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Count of career race starts for each driver up to (but not including)
    each round. shift(1) + cumcount ensures no leakage.

    A rookie in their first 5 races has CareerRaceCount < 5, signalling
    to the model that their rolling features are unreliable.
    The model learns to discount rolling stats for low-count drivers.

    Feature: CareerRaceCount (0-indexed: 0 = first race)
    """
    r = _normalise_driver_col(race_results.copy())
    r = r.sort_values(["Driver", "Year", "RoundNumber"]).reset_index(drop=True)
    r["CareerRaceCount"] = r.groupby("Driver").cumcount()  # 0 on first race
    log.info("Computed career race counts: %d rows", len(r))
    return r[["Year", "RoundNumber", "Driver", "CareerRaceCount"]].drop_duplicates()


# ── Item #5: Constructor championship delta ───────────────────────────────────

def compute_constructor_championship_context(
    race_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each constructor at each round, compute:
      ConChampDelta — points gap to the championship-leading constructor
                      (0 = leading, positive = trailing)
      ConChampPos   — constructor championship position (1 = P1)

    Uses cumulative points BEFORE each race (no leakage).
    A team that is 200 points behind runs differently from one 10 behind —
    they take more risk, which affects pit strategy and race outcome.
    """
    r = _normalise_driver_col(race_results.copy())
    r["Points"] = pd.to_numeric(r.get("Points", 0), errors="coerce").fillna(0)
    r = r.sort_values(["Year", "RoundNumber", "TeamName"])

    # Cumulative constructor points before each race
    team_pts = (
        r.groupby(["Year", "RoundNumber", "TeamName"])["Points"]
        .sum().reset_index()
        .sort_values(["TeamName", "Year", "RoundNumber"])
    )
    team_pts["ConCumPts"] = (
        team_pts.groupby(["Year", "TeamName"])["Points"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    # For each round, compute delta to leader and position
    rows = []
    for (year, rnd), grp in team_pts.groupby(["Year", "RoundNumber"]):
        max_pts = grp["ConCumPts"].max()
        grp = grp.copy()
        grp["ConChampDelta"] = max_pts - grp["ConCumPts"]
        grp["ConChampPos"] = grp["ConCumPts"].rank(
            ascending=False, method="min"
        ).astype(int)
        rows.append(grp)

    result = pd.concat(rows, ignore_index=True)

    # Merge back to driver level
    driver_team = r[["Year", "RoundNumber", "Driver", "TeamName"]].drop_duplicates()
    result = driver_team.merge(
        result[["Year", "RoundNumber", "TeamName", "ConChampDelta", "ConChampPos"]],
        on=["Year", "RoundNumber", "TeamName"], how="left",
    )

    log.info("Computed constructor championship context: %d rows", len(result))
    return result[[
        "Year", "RoundNumber", "Driver", "ConChampDelta", "ConChampPos",
    ]].drop_duplicates()


# ── Item #9: Circuit safety car rate ──────────────────────────────────────────

def compute_circuit_sc_rate(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Historical fraction of race laps run under Safety Car or VSC at each
    circuit. Computed from track_status parquets when available, falling
    back to a circuit-type heuristic if not.

    High SC circuits (Monaco ~0.35, Melbourne ~0.25) should widen the
    podium probability distribution — mid-field drivers have a realistic
    chance when SC bunches the pack.

    Feature: CircuitSCRate (0.0–1.0, circuit-level, not driver-level)
             This is the historical average — same value for all drivers
             at a given circuit.
    """
    # Known SC rates derived from historical data (2018–2025 averages)
    # These are pre-computed rather than derived at runtime to keep
    # features.py fast. Update after each season.
    KNOWN_SC_RATES: dict[str, float] = {
        "Melbourne":         0.28,
        "Sakhir":            0.15,
        "Jeddah":            0.22,
        "Shanghai":          0.18,
        "Suzuka":            0.10,
        "Monaco":            0.35,
        "Montreal":          0.25,
        "Spielberg":         0.20,
        "Silverstone":       0.18,
        "Budapest":          0.12,
        "Spa-Francorchamps": 0.14,
        "Zandvoort":         0.16,
        "Monza":             0.14,
        "Baku":              0.30,
        "Marina Bay":        0.20,
        "Austin":            0.22,
        "Mexico City":       0.14,
        "São Paulo":         0.32,
        "Las Vegas":         0.20,
        "Lusail":            0.18,
        "Yas Island":        0.12,
        "Imola":             0.22,
        "Miami":             0.20,
        "Portimão":          0.18,
        "Madrid":            0.16,
    }
    DEFAULT_SC_RATE = 0.18  # global average for unknown circuits

    r = _normalise_driver_col(race_results.copy())
    driver_circuit = r[["Year", "RoundNumber", "Driver", "CircuitShortName"]].drop_duplicates()
    driver_circuit["CircuitSCRate"] = (
        driver_circuit["CircuitShortName"]
        .map(KNOWN_SC_RATES)
        .fillna(DEFAULT_SC_RATE)
    )

    log.info("Computed circuit SC rates: %d rows", len(driver_circuit))
    return driver_circuit[[
        "Year", "RoundNumber", "Driver", "CircuitSCRate",
    ]].drop_duplicates()


def build_quali_features() -> pd.DataFrame:
    q_laps    = load_all_sessions("Q")
    q_results = _normalise_driver_col(load_all_results("Q"))
    r_results = load_all_results("R")
    weather_q = load_all_weather("Q")
    sectors   = load_all_sectors()

    # FP data — load gracefully
    fp3_laps = _load_fp_laps("FP3")
    fp2_laps = _load_fp_laps("FP2")

    best_quali       = extract_quali_best(q_laps)
    driver_qf        = compute_driver_rolling_quali_form(best_quali)
    con_qf           = compute_constructor_rolling_quali_form(best_quali, q_results)
    circuit_affinity = compute_driver_circuit_affinity(best_quali, q_results)
    h2h              = compute_teammate_h2h(best_quali, q_results)
    sector_form      = compute_driver_rolling_sector_form(sectors)
    fp3_pace         = extract_fp3_pace(fp3_laps)
    fp2_longruns     = extract_fp2_longruns(fp2_laps)
    driver_rf        = compute_driver_rolling_form(r_results)
    con_rf           = compute_constructor_rolling_form(r_results)
    champ_ctx        = compute_championship_context(r_results)
    quali_phase      = compute_quali_phase(best_quali)
    career_count     = compute_career_race_count(r_results)
    con_champ_ctx    = compute_constructor_championship_context(r_results)
    circuit_sc_rate  = compute_circuit_sc_rate(r_results)

    df = best_quali.merge(
        q_results[[
            "Year", "RoundNumber", "Driver",
            "TeamName", "EventName", "CircuitShortName",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.merge(weather_q, on=["Year", "RoundNumber"], how="left")
    df = df.merge(
        driver_qf[[
            "Year", "RoundNumber", "Driver",
            "RollingQualiGap", "RollingQualiPos", "RollingQualiStdGap",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.merge(
        con_qf[[
            "Year", "RoundNumber", "TeamName", "ConRollingQualiGap",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "TeamName"], how="left",
    )
    df = df.merge(circuit_affinity, on=["Year", "RoundNumber", "Driver"], how="left")
    df = df.merge(h2h,              on=["Year", "RoundNumber", "Driver"], how="left")

    # Sector current session
    if not sectors.empty:
        df = df.merge(
            sectors[[
                "Year", "RoundNumber", "Driver",
                "S1Gap_s", "S2Gap_s", "S3Gap_s", "SpeedTrap_kph",
            ]].drop_duplicates(),
            on=["Year", "RoundNumber", "Driver"], how="left",
        )
    else:
        for col in ["S1Gap_s", "S2Gap_s", "S3Gap_s", "SpeedTrap_kph"]:
            df[col] = np.nan

    # Sector rolling
    if not sector_form.empty:
        df = df.merge(
            sector_form[[
                "Year", "RoundNumber", "Driver",
                "RollingS1Gap", "RollingS2Gap", "RollingS3Gap", "RollingSpeedTrap",
            ]].drop_duplicates(),
            on=["Year", "RoundNumber", "Driver"], how="left",
        )
    else:
        for col in ["RollingS1Gap", "RollingS2Gap", "RollingS3Gap", "RollingSpeedTrap"]:
            df[col] = np.nan

    # FP3 pace
    if not fp3_pace.empty:
        df = df.merge(fp3_pace, on=["Year", "RoundNumber", "Driver"], how="left")
    else:
        for col in ["FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank"]:
            df[col] = np.nan

    # FP2 long runs
    if not fp2_longruns.empty:
        df = df.merge(fp2_longruns, on=["Year", "RoundNumber", "Driver"], how="left")
    else:
        for col in ["FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s"]:
            df[col] = np.nan

    df = df.merge(
        driver_rf[[
            "Year", "RoundNumber", "Driver",
            "RollingAvgFinish", "RollingAvgGrid", "RollingPoints",
            "RollingPodiumRate", "RollingDNFRate", "DNFStreak",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.merge(
        con_rf[[
            "Year", "RoundNumber", "TeamName",
            "ConRollingAvgFinish", "ConRollingPoints",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "TeamName"], how="left",
    )
    df = df.merge(champ_ctx, on=["Year", "RoundNumber", "Driver"], how="left")

    df["YearsSinceLastRegChange"] = df["Year"].apply(years_since_reg_change)

    # QualiphaseReached from the CURRENT session (known after qualifying)
    df = df.merge(quali_phase, on=["Year", "RoundNumber", "Driver"], how="left")
    df["QualiphaseReached"] = df["QualiphaseReached"].fillna(2).astype(int)

    df = df.merge(career_count,    on=["Year", "RoundNumber", "Driver"], how="left")
    df = df.merge(con_champ_ctx,   on=["Year", "RoundNumber", "Driver"], how="left")
    df = df.merge(circuit_sc_rate, on=["Year", "RoundNumber", "Driver"], how="left")
    df["CareerRaceCount"] = df["CareerRaceCount"].fillna(0).astype(int)
    df["ConChampDelta"]   = df["ConChampDelta"].fillna(df["ConChampDelta"].median())
    df["ConChampPos"]     = df["ConChampPos"].fillna(10).astype(int)
    df["CircuitSCRate"]   = df["CircuitSCRate"].fillna(0.18)

    circuit_flags = df["CircuitShortName"].apply(
        lambda c: pd.Series(circuit_type_flags(str(c)))
    )
    df = pd.concat([df, circuit_flags], axis=1)
    df = df.dropna(subset=["QualiPos"])

    log.info("Quali feature table: %d rows, %d columns", *df.shape)
    return df


def build_race_features(quali_df: pd.DataFrame) -> pd.DataFrame:
    r_results = load_all_results("R")
    weather_r = load_all_weather("R")

    r = _normalise_driver_col(r_results.copy())
    r["FinishPos"] = pd.to_numeric(r.get("Position", np.nan), errors="coerce")
    r["Podium"]    = (r["FinishPos"] <= 3).astype(int)
    r["DNF"] = r["Status"].apply(
        lambda s: 0 if str(s).startswith("Finished") or str(s).startswith("+") else 1
    )

    driver_rf      = compute_driver_rolling_form(r_results)
    con_rf         = compute_constructor_rolling_form(r_results)
    champ_ctx      = compute_championship_context(r_results)
    grid_penalties = compute_grid_penalties(r_results, quali_df)
    career_count   = compute_career_race_count(r_results)
    con_champ_ctx  = compute_constructor_championship_context(r_results)
    circuit_sc     = compute_circuit_sc_rate(r_results)
    # Items #3+#8: tyre features from race laps
    r_laps_df      = _load_fp_laps("R")
    tyre_features  = compute_tyre_features(r_laps_df)
    # Item #21: chaos circuit flag
    chaos_flags    = compute_chaos_flag(r_results)

    df = r[[
        "Year", "RoundNumber", "Driver", "TeamName",
        "EventName", "CircuitShortName",
        "FinishPos", "Podium", "DNF", "Points",
    ]].copy()

    # Quali output + FP features carried from quali_df
    quali_slim_cols = [
        "Year", "RoundNumber", "Driver",
        "QualiPos", "GapToPole_s", "BestQualiTime_s",
        "RollingQualiGap", "RollingQualiPos", "ConRollingQualiGap",
        "CircuitAvgQualiGap", "CircuitAvgQualiPos", "CircuitVisits",
        "H2H_QualiWinRate",
        "QualiphaseReached",
        "RollingS1Gap", "RollingS2Gap", "RollingS3Gap", "RollingSpeedTrap",
        "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    ]
    # Only include columns that exist
    quali_slim_cols = [c for c in quali_slim_cols if c in quali_df.columns]
    df = df.merge(
        quali_df[quali_slim_cols].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )

    df["GridDifficultyScore"] = df.apply(
        lambda row: grid_difficulty_score(
            row.get("QualiPos", 10), row.get("CircuitShortName", "")
        ),
        axis=1,
    )

    df = df.merge(weather_r, on=["Year", "RoundNumber"], how="left")
    df = df.merge(
        driver_rf[[
            "Year", "RoundNumber", "Driver",
            "RollingAvgFinish", "RollingAvgGrid", "RollingPoints",
            "RollingPodiumRate", "RollingDNFRate", "DNFStreak",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.merge(
        con_rf[[
            "Year", "RoundNumber", "TeamName",
            "ConRollingAvgFinish", "ConRollingPoints",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "TeamName"], how="left",
    )
    df = df.merge(champ_ctx, on=["Year", "RoundNumber", "Driver"], how="left")

    # Career race count, constructor championship context, circuit SC rate
    # These are computed above but were missing from the merge chain.
    df = df.merge(career_count,   on=["Year", "RoundNumber", "Driver"], how="left")
    df = df.merge(
        con_champ_ctx[[
            "Year", "RoundNumber", "Driver",
            "ConChampDelta", "ConChampPos",
        ]].drop_duplicates(),
        on=["Year", "RoundNumber", "Driver"], how="left",
    )
    df = df.merge(circuit_sc, on=["Year", "RoundNumber", "Driver"], how="left")
    df["CareerRaceCount"] = df["CareerRaceCount"].fillna(0).astype(int)
    df["ConChampDelta"]   = df["ConChampDelta"].fillna(df["ConChampDelta"].median())
    df["ConChampPos"]     = df["ConChampPos"].fillna(10).astype(int)
    df["CircuitSCRate"]   = df["CircuitSCRate"].fillna(0.18)

    df = df.merge(grid_penalties, on=["Year", "RoundNumber", "Driver"], how="left")
    df["GridPenaltyPlaces"] = df["GridPenaltyPlaces"].fillna(0).astype(int)

    if "QualiphaseReached" not in df.columns:
        phase_slim = quali_df[["Year", "RoundNumber", "Driver", "QualiphaseReached"]].drop_duplicates()
        df = df.merge(phase_slim, on=["Year", "RoundNumber", "Driver"], how="left")
    df["QualiphaseReached"] = df.get("QualiphaseReached", pd.Series(2, index=df.index)).fillna(2).astype(int)

    df["YearsSinceLastRegChange"] = df["Year"].apply(years_since_reg_change)

    # Tyre features (items #3 + #8)
    df = df.merge(tyre_features, on=["Year", "RoundNumber", "Driver"], how="left")
    df["StartCompound_enc"] = df["StartCompound_enc"].fillna(2).astype(int)
    df["StartTyreLife"]     = df["StartTyreLife"].fillna(0).astype(int)
    df["FreshStartTyre"]    = df["FreshStartTyre"].fillna(1).astype(int)
    df["RollingAvgStints"]  = df["RollingAvgStints"].fillna(1.5)
    df["RollingAvgDegRate"] = df["RollingAvgDegRate"].fillna(0.05)
    df["CircuitAvgStints"]  = df["CircuitAvgStints"].fillna(1.5)
    df["CircuitDegRate"]    = df["CircuitDegRate"].fillna(0.05)

    # Chaos flag (item #21)
    df = df.merge(chaos_flags, on=["Year", "RoundNumber", "Driver"], how="left")
    df["IsChaosCircuit"] = df["IsChaosCircuit"].fillna(0).astype(int)

    circuit_flags = df["CircuitShortName"].apply(
        lambda c: pd.Series(circuit_type_flags(str(c)))
    )
    df = pd.concat([df, circuit_flags], axis=1)
    df = df.dropna(subset=["FinishPos"])

    log.info("Race feature table: %d rows, %d columns", *df.shape)
    return df





# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Building quali features...")
    quali_df = build_quali_features()
    out_q = PROCESSED_DIR / "quali_features.parquet"
    quali_df.to_parquet(out_q, index=False)
    log.info("Saved → %s", out_q)

    log.info("Building race features...")
    race_df = build_race_features(quali_df)
    out_r = PROCESSED_DIR / "race_features.parquet"
    race_df.to_parquet(out_r, index=False)
    log.info("Saved → %s", out_r)

    for name, df in [("Quali", quali_df), ("Race", race_df)]:
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        log.info(
            "%s nulls:\n%s", name,
            nulls.to_string() if not nulls.empty else "none",
        )


if __name__ == "__main__":
    main()