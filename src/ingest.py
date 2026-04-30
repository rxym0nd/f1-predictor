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

For the current season, each session is checked individually against its
scheduled date — FP2/FP3/Q are fetched as soon as they complete, even
if the race hasn't happened yet. This ensures sector and FP features
are available for mid-weekend predictions.

Seasons covered:
  2018–2025 — full seasons
  2026      — current season, per-session date checking

Run from f1-predictor root:
    python src/ingest.py

Safe to re-run — already-saved sessions are skipped.
"""

import logging
import logging as _logging
import time
from datetime import datetime, timezone
from pathlib import Path

import fastf1  # type: ignore
from fastf1.exceptions import RateLimitExceededError  # type: ignore
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR          = Path("cache")
RAW_DIR            = Path("data/raw")
HISTORICAL_SEASONS = [2024, 2025]              # Faster run: only last 2 years + current
CURRENT_SEASON     = 2026

REQUIRED_SESSIONS = ["Q", "R"]
OPTIONAL_SESSIONS = ["FP2", "FP3"]
SESSION_TYPES     = OPTIONAL_SESSIONS + REQUIRED_SESSIONS  # FP2, FP3, Q, R

# Maps each session type to its scheduled date column in the FastF1 schedule
_SESSION_DATE_COL: dict[str, str] = {
    "FP1": "Session1Date",
    "FP2": "Session2Date",
    "FP3": "Session3Date",
    "Q":   "Session4Date",
    "R":   "Session5Date",
}

# Buffer after session ends before we try to fetch (avoids hitting API too early)
_SESSION_BUFFER_HOURS = 2

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


# ── FastF1 log silencing ──────────────────────────────────────────────────────

def _silence_fastf1(level: int = _logging.CRITICAL):
    ff1 = _logging.getLogger("fastf1")
    original = ff1.level
    ff1.setLevel(level)
    return ff1, original


def _restore_fastf1(ff1_logger, original_level: int):
    ff1_logger.setLevel(original_level)


# ── Session date helpers ──────────────────────────────────────────────────────

def _session_date(
    schedule: pd.DataFrame, round_number: int, session_type: str
) -> datetime | None:
    """Return the UTC datetime for a specific session, or None if unknown."""
    row = schedule[schedule["RoundNumber"] == round_number]
    if row.empty:
        return None
    col = _SESSION_DATE_COL.get(session_type)
    if col is None or col not in row.columns:
        return None
    val = row[col].iloc[0]
    if pd.isna(val):
        return None
    dt = pd.Timestamp(val)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime()


def _session_has_completed(
    schedule: pd.DataFrame, round_number: int, session_type: str
) -> bool:
    """True if the session's scheduled end time (+ buffer) has passed."""
    session_dt = _session_date(schedule, round_number, session_type)
    if session_dt is None:
        return False
    cutoff = session_dt + pd.Timedelta(hours=_SESSION_BUFFER_HOURS)
    return datetime.now(timezone.utc) > cutoff


def _completed_sessions(
    schedule: pd.DataFrame, round_number: int
) -> list[str]:
    """Return list of session types whose scheduled date has passed."""
    return [
        st for st in SESSION_TYPES
        if _session_has_completed(schedule, round_number, st)
    ]


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_session(
    year: int,
    round_number: int,
    session_type: str,
    optional: bool = False,
    max_retries: int = 3,
) -> dict | None:
    """
    Load a single session and return a dict of DataFrames.
    Returns None on failure.
    Optional sessions suppress FastF1 noise and treat missing data as a
    soft skip rather than a failure.
    Uses exponential backoff with jitter for transient errors.
    """
    import random

    for attempt in range(max_retries):
        if optional:
            ff1_logger, orig_level = _silence_fastf1()

        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(telemetry=False, weather=True, messages=False)

            try:
                laps = session.laps.copy()
            except Exception:
                if optional:
                    _restore_fastf1(ff1_logger, orig_level)
                    log.debug(
                        "No laps for %d R%d %s — optional, skipping",
                        year, round_number, session_type,
                    )
                    return None
                raise

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
            if attempt < max_retries - 1:
                wait = min(65 * 60, (2 ** attempt) * 60 + random.uniform(0, 30))
                log.warning(
                    "Rate limit hit (attempt %d/%d) — sleeping %.0fs...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                log.warning(
                    "Rate limit persists — skipping %d R%d %s",
                    year, round_number, session_type,
                )
                return None

        except (ConnectionError, TimeoutError, OSError) as exc:
            if optional:
                _restore_fastf1(ff1_logger, orig_level)
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                log.warning(
                    "Network error %d R%d %s (attempt %d/%d): %s — retrying in %.0fs",
                    year, round_number, session_type, attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error(
                    "Network error persists for %d R%d %s after %d attempts: %s",
                    year, round_number, session_type, max_retries, exc,
                )
                return None

        except Exception as exc:
            if optional:
                _restore_fastf1(ff1_logger, orig_level)
                log.debug(
                    "Could not load %d R%d %s — %s (optional)",
                    year, round_number, session_type, exc,
                )
            else:
                log.warning(
                    "Could not load %d R%d %s — %s",
                    year, round_number, session_type, exc,
                )
            return None

    return None


def save_session(
    data: dict, year: int, round_number: int, session_type: str
):
    prefix = RAW_DIR / f"{year}_R{round_number:02d}_{session_type}"
    written = 0
    for name, df in data.items():
        if df.empty:
            continue
        df.to_parquet(Path(f"{prefix}_{name}.parquet"), index=False)
        written += 1
    if written:
        log.info("Saved %d R%d %s", year, round_number, session_type)


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
                log.error("Could not fetch schedule for %d.", year)
                return None
        except Exception as exc:
            log.error("Could not fetch schedule for %d — %s", year, exc)
            return None
    return None


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest_season(
    year: int, current_season: bool = False
) -> tuple[int, int, int, int]:
    """
    Returns (saved, skipped, failed, fp_unavailable).

    For current_season=True, each session is checked individually:
    - FP2/FP3/Q are fetched as soon as their scheduled time has passed
    - R is only fetched after race day
    - Future sessions are skipped cleanly without flooding the log
    """
    schedule = get_schedule(year)
    if schedule is None:
        return 0, 0, 0, 0

    rounds = schedule["RoundNumber"].tolist()
    log.info("── %d  (%d rounds) ──────────────────────────", year, len(rounds))

    saved = skipped = failed = fp_unavailable = 0

    for round_number in rounds:

        if current_season:
            completed = _completed_sessions(schedule, round_number)
            if not completed:
                log.info(
                    "Skip %d R%d — weekend not started yet", year, round_number
                )
                skipped += len(SESSION_TYPES)
                continue
        else:
            completed = SESSION_TYPES  # all sessions available for historical

        for session_type in SESSION_TYPES:
            is_optional = session_type in OPTIONAL_SESSIONS

            # Skip sessions that haven't happened yet (current season only)
            if current_season and session_type not in completed:
                log.debug(
                    "Skip %d R%d %s — not yet completed",
                    year, round_number, session_type,
                )
                skipped += 1
                continue

            if already_saved(year, round_number, session_type):
                log.debug(
                    "Skip %d R%d %s — already saved",
                    year, round_number, session_type,
                )
                skipped += 1
                continue

            data = fetch_session(
                year, round_number, session_type, optional=is_optional
            )

            if data is None:
                if is_optional:
                    fp_unavailable += 1
                else:
                    failed += 1
                continue

            save_session(data, year, round_number, session_type)
            saved += 1
            time.sleep(2)

    return saved, skipped, failed, fp_unavailable


def ingest_all():
    setup_dirs()
    ts = ts_sk = tf = tfp = 0

    for year in HISTORICAL_SEASONS:
        s, sk, f, fp = ingest_season(year, current_season=False)
        ts += s
        ts_sk += sk
        tf += f
        tfp += fp

    log.info(
        "── %d  (current season — per-session date checking) ──",
        CURRENT_SEASON,
    )
    s, sk, f, fp = ingest_season(CURRENT_SEASON, current_season=True)
    ts += s
    ts_sk += sk
    tf += f
    tfp += fp

    log.info(
        "Done. Saved: %d  |  Skipped: %d  |  Failed (Q/R): %d  |  "
        "FP unavailable (expected for old seasons): %d",
        ts, ts_sk, tf, tfp,
    )
    if tfp > 0:
        log.info(
            "Note: %d FP sessions had no timing data — normal for 2018–2020. "
            "FP features will be imputed from medians for those rounds.",
            tfp,
        )


if __name__ == "__main__":
    ingest_all()