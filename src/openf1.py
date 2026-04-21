"""
openf1.py

Fetches qualifying sector times and speed trap data from the OpenF1
REST API (https://api.openf1.org/v1) and saves them as Parquet files.

OpenF1 coverage: reliable sector times from 2023 onwards.
For 2018–2022, we fall back to FastF1 sector times already stored
in the laps parquets.

Files written:
  data/raw/{year}_R{round:02d}_Q_sectors.parquet
    One row per driver per qualifying session.
    Columns: Year, RoundNumber, Driver, TeamName, CircuitShortName,
             BestS1_s, BestS2_s, BestS3_s,
             S1Gap_s, S2Gap_s, S3Gap_s,   ← delta to session best
             SpeedTrap_kph                 ← fastest speed trap reading

Run from f1-predictor root:
    python src/openf1.py                  # fetch all missing seasons
    python src/openf1.py --year 2026      # fetch one season only

Safe to re-run — already-saved sessions are skipped.
"""

import argparse
import logging
import time
from pathlib import Path

import fastf1  # type: ignore
import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR       = Path("cache")
RAW_DIR         = Path("data/raw")
OPENF1_BASE     = "https://api.openf1.org/v1"

# OpenF1 has reliable sector data from 2023; earlier we use FastF1 fallback
OPENF1_FROM_YEAR = 2023

# Seasons to process (updated alongside ingest.py)
HISTORICAL_SEASONS = list(range(2018, 2026))
CURRENT_SEASON     = 2026

REQUEST_TIMEOUT    = 30   # seconds per API call
RETRY_SLEEP        = 5    # seconds between retries
MAX_RETRIES        = 3


# ── OpenF1 API helpers ────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> list | None:
    """GET request with retries. Returns list of records or None on failure."""
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                sleep = RETRY_SLEEP * (attempt + 1)
                log.warning("Rate limited by OpenF1 — sleeping %ds", sleep)
                time.sleep(sleep)
            else:
                log.warning("HTTP error from OpenF1: %s", e)
                return None
        except Exception as e:
            log.warning("OpenF1 request failed (attempt %d): %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)
    return None


def get_session_key(year: int, round_number: int) -> int | None:
    """Return the OpenF1 session_key for a qualifying session."""
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        row   = sched[sched["RoundNumber"] == round_number]
        if row.empty:
            return None
        circuit_name = row["Location"].iloc[0]
    except Exception as e:
        log.warning("Could not get circuit name for %d R%d: %s", year, round_number, e)
        return None

    records = _get("sessions", {
        "year":         year,
        "session_name": "Qualifying",
        "circuit_short_name": circuit_name.lower().replace(" ", "_"),
    })

    if not records:
        # Try without circuit filter — some names don't match exactly
        records = _get("sessions", {
            "year":         year,
            "session_name": "Qualifying",
        })
        if records:
            # Filter by approximate circuit name match
            records = [
                r for r in records
                if circuit_name[:4].lower() in
                   r.get("circuit_short_name", "").lower()
            ]

    if not records:
        log.warning("No OpenF1 session found for %d R%d", year, round_number)
        return None

    # If multiple matches, take the one closest to the round number
    return records[0].get("session_key")


def fetch_openf1_sectors(session_key: int) -> pd.DataFrame:
    """
    Fetch per-lap sector times from OpenF1 for a session.
    Returns best sector times per driver.
    """
    records = _get("laps", {"session_key": session_key})
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # OpenF1 lap columns: driver_number, lap_number,
    # duration_sector_1, duration_sector_2, duration_sector_3,
    # st_speed (speed trap)
    needed = ["driver_number", "duration_sector_1",
              "duration_sector_2", "duration_sector_3"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df = df.rename(columns={
        "driver_number":    "DriverNum",
        "duration_sector_1": "S1_s",
        "duration_sector_2": "S2_s",
        "duration_sector_3": "S3_s",
        "st_speed":          "SpeedTrap_kph",
    })

    # Convert to numeric
    for col in ["S1_s", "S2_s", "S3_s", "SpeedTrap_kph"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop outlier laps (>20% slower than session median per sector)
    for col in ["S1_s", "S2_s", "S3_s"]:
        if col in df.columns:
            med = df[col].median()
            if pd.notna(med) and med > 0:
                df = df[df[col] <= med * 1.20]

    # Best sector time per driver
    best = df.groupby("DriverNum").agg(
        BestS1_s=("S1_s", "min"),
        BestS2_s=("S2_s", "min"),
        BestS3_s=("S3_s", "min"),
        SpeedTrap_kph=("SpeedTrap_kph", "max"),
    ).reset_index()

    return best


def fetch_openf1_drivers(session_key: int) -> pd.DataFrame:
    """Fetch driver number → abbreviation mapping from OpenF1."""
    records = _get("drivers", {"session_key": session_key})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "driver_number" not in df.columns or "name_acronym" not in df.columns:
        return pd.DataFrame()
    return df[["driver_number", "name_acronym", "team_name"]].rename(columns={
        "driver_number": "DriverNum",
        "name_acronym":  "Driver",
        "team_name":     "TeamName_of1",
    })


# ── FastF1 fallback for 2018–2022 ─────────────────────────────────────────────

def fetch_fastf1_sectors(year: int, round_number: int) -> pd.DataFrame:
    """
    Extract per-driver best sector times from saved FastF1 laps parquet.
    Used for 2018–2022 where OpenF1 coverage is incomplete.
    """
    laps_path = RAW_DIR / f"{year}_R{round_number:02d}_Q_laps.parquet"
    if not laps_path.exists():
        log.debug("No FastF1 laps parquet for %d R%d Q", year, round_number)
        return pd.DataFrame()

    laps = pd.read_parquet(laps_path)

    # Normalise driver column
    if "Abbreviation" in laps.columns and "Driver" not in laps.columns:
        laps = laps.rename(columns={"Abbreviation": "Driver"})

    # Sector times are stored as timedeltas — convert to seconds
    sector_cols = {
        "Sector1Time": "S1_s",
        "Sector2Time": "S2_s",
        "Sector3Time": "S3_s",
        "SpeedST":     "SpeedTrap_kph",
    }
    for src, dst in sector_cols.items():
        if src in laps.columns:
            if laps[src].dtype == "timedelta64[ns]" or str(laps[src].dtype).startswith("timedelta"):
                laps[dst] = pd.to_timedelta(laps[src]).dt.total_seconds()
            else:
                laps[dst] = pd.to_numeric(laps[src], errors="coerce")

    if "S1_s" not in laps.columns:
        return pd.DataFrame()

    # Filter valid laps
    laps = laps.dropna(subset=["S1_s"])
    lap_time_col = None
    for c in ["LapTime", "LapTime_s"]:
        if c in laps.columns:
            lap_time_col = c
            break

    if lap_time_col:
        if str(laps[lap_time_col].dtype).startswith("timedelta"):
            laps["LapTime_s_tmp"] = pd.to_timedelta(laps[lap_time_col]).dt.total_seconds()
        else:
            laps["LapTime_s_tmp"] = pd.to_numeric(laps[lap_time_col], errors="coerce")
        median = laps["LapTime_s_tmp"].median()
        if pd.notna(median):
            laps = laps[laps["LapTime_s_tmp"] <= median * 1.20]

    best = laps.groupby("Driver").agg(
        BestS1_s=("S1_s", "min"),
        BestS2_s=("S2_s", "min"),
        BestS3_s=("S3_s", "min"),
        SpeedTrap_kph=("SpeedTrap_kph", "max"),
    ).reset_index()

    # Attach TeamName
    if "TeamName" in laps.columns:
        team_map = laps.groupby("Driver")["TeamName"].first()
        best["TeamName"] = best["Driver"].map(team_map)

    return best


# ── Sector delta computation ───────────────────────────────────────────────────

def compute_sector_deltas(best: pd.DataFrame) -> pd.DataFrame:
    """
    Add S1Gap_s, S2Gap_s, S3Gap_s — delta from each driver's best sector
    to the fastest sector time in the session (not necessarily the same lap).
    A gap of 0.0 means the driver set the best sector in the field.
    """
    best = best.copy()
    for col, gap_col in [("BestS1_s", "S1Gap_s"),
                          ("BestS2_s", "S2Gap_s"),
                          ("BestS3_s", "S3Gap_s")]:
        if col in best.columns:
            session_best = best[col].min()
            best[gap_col] = best[col] - session_best
        else:
            best[gap_col] = np.nan
    return best


# ── Main session processor ────────────────────────────────────────────────────

def process_session(year: int, round_number: int) -> pd.DataFrame | None:
    """
    Fetch sector data for one qualifying session.
    Uses OpenF1 for 2023+, FastF1 fallback for earlier years.
    Returns a DataFrame ready to save, or None on failure.
    """
    if year >= OPENF1_FROM_YEAR:
        log.info("Fetching OpenF1 sectors for %d R%d", year, round_number)
        session_key = get_session_key(year, round_number)
        if session_key is None:
            log.warning("No session key found — falling back to FastF1")
            best = fetch_fastf1_sectors(year, round_number)
        else:
            best = fetch_openf1_sectors(session_key)
            if best.empty:
                log.warning("OpenF1 returned empty data — falling back to FastF1")
                best = fetch_fastf1_sectors(year, round_number)
            else:
                # Map driver numbers to abbreviations
                drivers = fetch_openf1_drivers(session_key)
                if not drivers.empty:
                    best = best.merge(
                        drivers[["DriverNum", "Driver", "TeamName_of1"]],
                        on="DriverNum",
                        how="left",
                    )
                    if "TeamName" not in best.columns:
                        best = best.rename(columns={"TeamName_of1": "TeamName"})
    else:
        log.info("Using FastF1 sectors for %d R%d (pre-OpenF1)", year, round_number)
        best = fetch_fastf1_sectors(year, round_number)

    if best.empty:
        log.warning("No sector data for %d R%d", year, round_number)
        return None

    # Compute deltas
    best = compute_sector_deltas(best)

    # Attach circuit info from existing results parquet
    results_path = RAW_DIR / f"{year}_R{round_number:02d}_Q_results.parquet"
    if results_path.exists():
        results = pd.read_parquet(results_path)
        if "Abbreviation" in results.columns:
            results = results.rename(columns={"Abbreviation": "Driver"})
        if "Driver" in results.columns and "CircuitShortName" in results.columns:
            circuit_map = results.set_index("Driver")["CircuitShortName"].to_dict()
            best["CircuitShortName"] = best["Driver"].map(circuit_map)
            if "TeamName" not in best.columns and "TeamName" in results.columns:
                team_map = results.set_index("Driver")["TeamName"].to_dict()
                best["TeamName"] = best["Driver"].map(team_map)

    best["Year"]        = year
    best["RoundNumber"] = round_number

    # Keep only essential columns
    keep = ["Year", "RoundNumber", "Driver", "TeamName", "CircuitShortName",
            "BestS1_s", "BestS2_s", "BestS3_s",
            "S1Gap_s", "S2Gap_s", "S3Gap_s",
            "SpeedTrap_kph"]
    keep = [c for c in keep if c in best.columns]
    return best[keep]


def already_saved(year: int, round_number: int) -> bool:
    path = RAW_DIR / f"{year}_R{round_number:02d}_Q_sectors.parquet"
    return path.exists()


def save_sectors(df: pd.DataFrame, year: int, round_number: int):
    path = RAW_DIR / f"{year}_R{round_number:02d}_Q_sectors.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved sectors → %s (%d drivers)", path.name, len(df))


# ── Main loop ─────────────────────────────────────────────────────────────────

def fetch_all_sectors(year_filter: int | None = None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    seasons = HISTORICAL_SEASONS + [CURRENT_SEASON]
    if year_filter is not None:
        seasons = [year_filter]

    total_saved = total_skipped = total_failed = 0

    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            log.error("Could not fetch schedule for %d: %s", year, e)
            continue

        rounds = schedule["RoundNumber"].tolist()
        log.info("── %d  (%d rounds) ──────────────────────────", year, len(rounds))

        for rnd in rounds:
            # Skip future rounds for current season
            if year == CURRENT_SEASON:
                laps_exist = (RAW_DIR / f"{year}_R{rnd:02d}_Q_laps.parquet").exists()
                results_exist = (RAW_DIR / f"{year}_R{rnd:02d}_Q_results.parquet").exists()
                if not laps_exist and not results_exist:
                    log.debug("Skip %d R%d — no raw data yet", year, rnd)
                    total_skipped += 1
                    continue

            if already_saved(year, rnd):
                log.debug("Skip %d R%d — sectors already saved", year, rnd)
                total_skipped += 1
                continue

            df = process_session(year, rnd)

            if df is None or df.empty:
                total_failed += 1
                continue

            save_sectors(df, year, rnd)
            total_saved += 1
            time.sleep(1)   # be polite to OpenF1

    log.info(
        "Done. Saved: %d  |  Skipped: %d  |  Failed: %d",
        total_saved, total_skipped, total_failed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch qualifying sector times from OpenF1"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Fetch only this season (default: all seasons)"
    )
    args = parser.parse_args()
    fetch_all_sectors(year_filter=args.year)


if __name__ == "__main__":
    main()