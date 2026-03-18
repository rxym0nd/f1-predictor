"""
ingest.py

Fetches qualifying, race, FP2 and FP3 sessions from FastF1 and
saves raw data as Parquet files in data/raw/.

Session types fetched:
  Q   — Qualifying         (required)
  R   — Race               (required)
  FP2 — Free Practice 2   (optional — race long-run pace signal)
  FP3 — Free Practice 3   (optional — final setup run, predicts quali order)

FP1 is omitted — teams run mixed programmes and the data is noisy.

FP data is marked OPTIONAL. Sessions from 2018–2020 often have timing
data missing from the FastF1 API; these are silently skipped and do not
count as failures. Q and R failures always count.

Seasons covered:
  2018–2025 — full seasons
  2026      — current season, completed rounds only

Run from f1-predictor root:
    python src/ingest.py

Safe to re-run — already-saved sessions are skipped.
"""

import logging
import logging as _logging
import time
from datetime import datetime, timezone
from pathlib import Path

import fastf1
from fastf1.exceptions import RateLimitExceededError
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR          = Path("cache")
RAW_DIR            = Path("data/raw")
HISTORICAL_SEASONS = list(range(2018, 2026))   # 2018–2025 inclusive
CURRENT_SEASON     = 2026

# Required sessions — failures always counted and reported
REQUIRED_SESSIONS = ["Q", "R"]

# Optional sessions — data may not exist for older rounds; missing data
# is silently skipped and does NOT inflate the failure count
OPTIONAL_SESSIONS = ["FP2", "FP3"]

SESSION_TYPES = OPTIONAL_SESSIONS + REQUIRED_SESSIONS  # FP2, FP3, Q, R

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


# ── Core fetch ────────────────────────────────────────────────────────────────

def _silence_fastf1(level: int = _logging.CRITICAL):
    """Temporarily raise the FastF1 log level to suppress internal noise."""
    ff1 = _logging.getLogger("fastf1")
    original = ff1.level
    ff1.setLevel(level)
    return ff1, original


def _restore_fastf1(ff1_logger, original_level: int):
    ff1_logger.setLevel(original_level)


def fetch_session(
    year: int,
    round_number: int,
    session_type: str,
    optional: bool = False,
) -> dict | None:
    """
    Load a single session and return a dict of DataFrames.

    For optional sessions (FP2/FP3):
      - FastF1 internal debug noise is suppressed
      - If timing data is unavailable the function returns None quietly
      - The caller treats None as a soft skip, not a failure

    For required sessions (Q/R):
      - Normal logging applies
      - None means a real failure

    Retries once after 65 minutes on rate limit.
    """
    for attempt in range(2):
        # Suppress FastF1's verbose internal logging for optional sessions
        # that are likely to have no data (2018–2020 FP sessions especially)
        if optional:
            ff1_logger, orig_level = _silence_fastf1()

        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(telemetry=False, weather=True, messages=False)

            # ── Safely access laps ────────────────────────────────────────────
            # session.load() does not raise even when laps fail internally.
            # Accessing session.laps afterwards can raise DataNotLoadedError.
            # We handle this explicitly so the caller gets a clean None.
            try:
                laps = session.laps.copy()
            except Exception:
                if optional:
                    # FP data not available for this session — silent skip
                    if optional:
                        _restore_fastf1(ff1_logger, orig_level)
                    log.debug(
                        "No laps data for %d R%d %s — skipping (optional session)",
                        year, round_number, session_type,
                    )
                    return None
                raise   # For Q/R, re-raise so the outer handler logs it

            results = session.results.copy()
            weather = (
                session.weather_data.copy()
                if session.weather_data is not None
                else pd.DataFrame()
            )

            for df in (laps, results, weather):
                df["Year"]             = year
                df["RoundNumber"]      = round_number
                df["SessionType"]      = session_type
                df["EventName"]        = session.event["EventName"]
                df["CircuitShortName"] = session.event["Location"]

            if optional:
                _restore_fastf1(ff1_logger, orig_level)

            return {"laps": laps, "results": results, "weather": weather}

        except RateLimitExceededError:
            if optional:
                _restore_fastf1(ff1_logger, orig_level)
            if attempt == 0:
                log.warning("Rate limit hit — sleeping 65 minutes before retrying...")
                time.sleep(65 * 60)
            else:
                log.warning(
                    "Rate limit hit again after sleep — skipping %d R%d %s",
                    year, round_number, session_type,
                )
                return None

        except Exception as exc:
            if optional:
                _restore_fastf1(ff1_logger, orig_level)
            # For optional sessions, downgrade to debug so we don't fill
            # the terminal with unavoidable warnings about old FP data
            if optional:
                log.debug(
                    "Could not load %d R%d %s — %s (optional, skipping silently)",
                    year, round_number, session_type, exc,
                )
            else:
                log.warning(
                    "Could not load %d R%d %s — %s",
                    year, round_number, session_type, exc,
                )
            return None

    return None


def save_session(data: dict, year: int, round_number: int, session_type: str):
    """Save each non-empty DataFrame as a Parquet file."""
    prefix = RAW_DIR / f"{year}_R{round_number:02d}_{session_type}"
    files_written = 0
    for name, df in data.items():
        if df.empty:
            continue
        path = Path(f"{prefix}_{name}.parquet")
        df.to_parquet(path, index=False)
        files_written += 1
    if files_written > 0:
        log.info("Saved %d R%d %s", year, round_number, session_type)
    else:
        log.debug("Nothing to save for %d R%d %s (all DataFrames empty)",
                  year, round_number, session_type)


def already_saved(year: int, round_number: int, session_type: str) -> bool:
    prefix = f"{year}_R{round_number:02d}_{session_type}"
    return any(RAW_DIR.glob(f"{prefix}_*.parquet"))


def get_schedule(year: int) -> pd.DataFrame | None:
    for attempt in range(2):
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except RateLimitExceededError:
            if attempt == 0:
                log.warning(
                    "Rate limit fetching schedule for %d — sleeping 65 min...", year
                )
                time.sleep(65 * 60)
            else:
                log.error("Could not fetch schedule for %d after retry.", year)
                return None
        except Exception as exc:
            log.error("Could not fetch schedule for %d — %s", year, exc)
            return None
    return None


def race_date_from_schedule(
    schedule: pd.DataFrame, round_number: int
) -> datetime | None:
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


def is_round_completed(schedule: pd.DataFrame, round_number: int) -> bool:
    race_dt = race_date_from_schedule(schedule, round_number)
    if race_dt is None:
        return False
    return datetime.now(timezone.utc) > race_dt + pd.Timedelta(hours=3)


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest_season(
    year: int, current_season: bool = False
) -> tuple[int, int, int, int]:
    """
    Returns (saved, skipped, failed, fp_unavailable).
    fp_unavailable counts FP sessions where data simply doesn't exist
    in the API — these are expected for 2018–2020 and are NOT failures.
    """
    schedule = get_schedule(year)
    if schedule is None:
        return 0, 0, 0, 0

    rounds = schedule["RoundNumber"].tolist()
    log.info("── %d  (%d rounds) ──────────────────────────", year, len(rounds))

    saved = skipped = failed = fp_unavailable = 0

    for round_number in rounds:
        # For current season, skip rounds that haven't happened yet
        if current_season and not all(
            already_saved(year, round_number, st) for st in REQUIRED_SESSIONS
        ):
            if not is_round_completed(schedule, round_number):
                log.info("Skip %d R%d — not yet completed", year, round_number)
                skipped += len(SESSION_TYPES)
                continue

        for session_type in SESSION_TYPES:
            is_optional = session_type in OPTIONAL_SESSIONS

            if already_saved(year, round_number, session_type):
                log.debug("Skip %d R%d %s — already saved",
                          year, round_number, session_type)
                skipped += 1
                continue

            data = fetch_session(
                year, round_number, session_type, optional=is_optional
            )

            if data is None:
                if is_optional:
                    fp_unavailable += 1   # expected, not a real failure
                else:
                    failed += 1
                continue

            save_session(data, year, round_number, session_type)
            saved += 1
            time.sleep(2)

    return saved, skipped, failed, fp_unavailable


def ingest_all():
    setup_dirs()
    total_saved = total_skipped = total_failed = total_fp_unavail = 0

    for year in HISTORICAL_SEASONS:
        s, sk, f, fp = ingest_season(year, current_season=False)
        total_saved += s
        total_skipped += sk
        total_failed += f
        total_fp_unavail += fp

    log.info("── %d  (current season — completed rounds only) ──", CURRENT_SEASON)
    s, sk, f, fp = ingest_season(CURRENT_SEASON, current_season=True)
    total_saved += s
    total_skipped += sk
    total_failed += f
    total_fp_unavail += fp

    log.info(
        "Done. Saved: %d  |  Skipped: %d  |  Failed (Q/R): %d  |  "
        "FP unavailable (expected for old seasons): %d",
        total_saved, total_skipped, total_failed, total_fp_unavail,
    )
    if total_fp_unavail > 0:
        log.info(
            "Note: %d FP sessions had no timing data in the FastF1 API. "
            "This is normal for 2018–2020. FP features will be imputed "
            "from global medians for those rounds during training.",
            total_fp_unavail,
        )


if __name__ == "__main__":
    ingest_all()