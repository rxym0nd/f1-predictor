"""
weather.py  (item #4 — weather forecast API)

Fetches race-weekend weather forecasts from OpenMeteo
(https://api.open-meteo.com) for use in predictions.

OpenMeteo is free, requires no API key, and covers all F1 circuits.
For historical sessions the model already uses FastF1 weather data.
This module is used exclusively by predict.py to get forward-looking
weather for the upcoming race weekend.

Usage (from predict.py):
    from weather import fetch_race_weekend_forecast

    forecast = fetch_race_weekend_forecast(
        lat=35.3717, lon=136.9236,          # Suzuka
        race_date="2026-03-29",
        session_hour_utc=6,                 # race start UTC
    )
    # forecast is a dict with keys matching our feature columns:
    # AirTemp_mean, TrackTemp_mean, Humidity_mean, Rainfall_any

Circuit coordinates are stored in CIRCUIT_COORDS below.
FastF1's Location string is used as the lookup key.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR       = Path("cache")
OPENMETEO_BASE  = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 20

# ── Circuit coordinates ───────────────────────────────────────────────────────
# Key = FastF1 session.event["Location"] (must match exactly)
# Value = (latitude, longitude)

CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "Melbourne":     (-37.8497,  144.9680),
    "Shanghai":      ( 31.3389,  121.2198),
    "Suzuka":        ( 34.8431,  136.5419),
    "Sakhir":        ( 26.0325,   50.5106),
    "Jeddah":        ( 21.6319,   39.1044),
    "Miami":         ( 25.9581,  -80.2389),
    "Imola":         ( 44.3439,   11.7167),
    "Monaco":        ( 43.7347,    7.4206),
    "Barcelona":     ( 41.5700,    2.2611),
    "Montréal":      ( 45.5000,  -73.5228),
    "Spielberg":     ( 47.2197,   14.7647),
    "Silverstone":   ( 52.0786,   -1.0169),
    "Spa-Francorchamps": (50.4372, 5.9714),
    "Budapest":      ( 47.5789,   19.2486),
    "Zandvoort":     ( 52.3888,    4.5409),
    "Monza":         ( 45.6156,    9.2811),
    "Baku":          ( 40.3725,   49.8533),
    "Marina Bay":    (  1.2914,  103.8639),
    "Austin":        ( 30.1328,  -97.6411),
    "Mexico City":   ( 19.4042,  -99.0907),
    "São Paulo":     (-23.7036,  -46.6997),
    "Las Vegas":     ( 36.1699, -115.1398),
    "Lusail":        ( 25.4900,   51.4542),
    "Yas Island":    ( 24.4672,   54.6031),
    # 2026 additions
    "Portimão":      ( 37.2272,   -8.6267),
    "Madrid":        ( 40.5722,   -3.6442),
}


def _get_race_schedule_info(
    year: int, round_number: int
) -> tuple[str, str, int] | None:
    """
    Returns (circuit_location, race_date_str, race_hour_utc) or None.
    race_date_str is YYYY-MM-DD, race_hour_utc is the UTC hour of race start.
    """
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        event    = fastf1.get_event(year, round_number)
        location = event["Location"]

        # Session5 = race. Try to get the session date.
        for col in ("Session5DateUtc", "Session5Date"):
            if col in event.index and pd.notna(event[col]):
                dt = pd.Timestamp(event[col])
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                return (
                    location,
                    dt.strftime("%Y-%m-%d"),
                    dt.hour,
                )

        # Fallback: use EventDate (usually Saturday or Sunday)
        if pd.notna(event.get("EventDate")):
            dt = pd.Timestamp(event["EventDate"])
            return location, dt.strftime("%Y-%m-%d"), 14  # assume 14:00 UTC

        return None

    except Exception as exc:
        log.warning("Could not get schedule info for %d R%d: %s", year, round_number, exc)
        return None


def fetch_race_weekend_forecast(
    lat: float,
    lon: float,
    race_date: str,
    session_hour_utc: int = 14,
) -> dict | None:
    """
    Fetch hourly weather forecast for a race session from OpenMeteo.

    Returns a dict with keys:
        AirTemp_mean    — forecast air temperature (°C)
        TrackTemp_mean  — estimated track temperature (air × 1.4, capped at 60°C)
        Humidity_mean   — relative humidity (%)
        Rainfall_any    — True/False whether rain is forecast

    Returns None if the forecast cannot be fetched (e.g. >16 days ahead).
    OpenMeteo provides free forecasts up to 16 days ahead.
    """
    # Build a ±3-hour window around session start
    params = {
        "latitude":          lat,
        "longitude":         lon,
        "hourly":            "temperature_2m,relativehumidity_2m,precipitation_probability,precipitation",
        "start_date":        race_date,
        "end_date":          race_date,
        "timezone":          "UTC",
        "forecast_days":     1,
    }

    try:
        resp = requests.get(OPENMETEO_BASE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("OpenMeteo request failed: %s", exc)
        return None

    hourly = data.get("hourly", {})
    hours  = hourly.get("time", [])
    temps  = hourly.get("temperature_2m", [])
    humids = hourly.get("relativehumidity_2m", [])
    precip = hourly.get("precipitation", [])
    precip_prob = hourly.get("precipitation_probability", [])

    if not hours:
        log.warning("OpenMeteo returned no hourly data for %s", race_date)
        return None

    # Take the 3-hour window around the session
    start_h = max(0, session_hour_utc - 1)
    end_h   = min(23, session_hour_utc + 2)
    window  = [i for i, t in enumerate(hours) if start_h <= int(t[11:13]) <= end_h]

    if not window:
        window = list(range(len(hours)))  # fallback: full day

    def _avg(lst, idx):
        vals = [lst[i] for i in idx if i < len(lst) and lst[i] is not None]
        return sum(vals) / len(vals) if vals else None

    air_temp    = _avg(temps,       window)
    humidity    = _avg(humids,      window)
    rain_total  = sum(precip[i] for i in window if i < len(precip) and precip[i])
    rain_prob   = _avg(precip_prob, window) or 0

    track_temp = None
    if air_temp is not None:
        track_temp = min(air_temp * 1.4, 60.0)

    rainfall_any = rain_total > 0.5 or rain_prob > 50

    result = {
        "AirTemp_mean":   round(air_temp,   1) if air_temp   is not None else None,
        "TrackTemp_mean": round(track_temp, 1) if track_temp is not None else None,
        "Humidity_mean":  round(humidity,   1) if humidity   is not None else None,
        "Rainfall_any":   rainfall_any,
    }

    log.info(
        "Weather forecast for %s: Air=%.1f°C, Track=%.1f°C, "
        "Humidity=%.0f%%, Rain=%s",
        race_date,
        result["AirTemp_mean"]   or 0,
        result["TrackTemp_mean"] or 0,
        result["Humidity_mean"]  or 0,
        "YES" if result["Rainfall_any"] else "no",
    )
    return result


def get_forecast_for_round(year: int, round_number: int) -> dict | None:
    """
    High-level entry point used by predict.py.
    Looks up circuit coordinates and fetches the race-day forecast.
    Returns a weather dict or None if unavailable.
    """
    info = _get_race_schedule_info(year, round_number)
    if info is None:
        return None

    location, race_date, race_hour = info
    coords = CIRCUIT_COORDS.get(location)

    if coords is None:
        log.warning(
            "No coordinates for circuit '%s' — add to CIRCUIT_COORDS in weather.py",
            location,
        )
        return None

    # Check if the race is within OpenMeteo's 16-day window
    try:
        race_dt = datetime.strptime(race_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_ahead = (race_dt - datetime.now(timezone.utc)).days
        if days_ahead > 15:
            log.info(
                "Race is %d days away — outside OpenMeteo's 16-day window. "
                "Using historical weather averages instead.",
                days_ahead,
            )
            return None
        if days_ahead < 0:
            log.info("Race already completed — forecast not needed.")
            return None
    except Exception:
        pass

    lat, lon = coords
    return fetch_race_weekend_forecast(lat, lon, race_date, race_hour)