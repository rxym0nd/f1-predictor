"""
news.py

Lightweight F1 news & penalty scraper for pre-race intelligence.

Fetches:
  1. Grid penalties from FIA documents / F1 API
  2. Latest F1 headlines from the official F1 website RSS
  3. Driver status updates (injuries, replacements, power-unit changes)

Results are stored in predictions/{year}_R{round}_news.json and
consumed by predict.py to adjust grid positions and flag uncertainty.

No external API keys required — uses public RSS/HTML endpoints only.

Run:
    python src/news.py --year 2026 --round 4
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("predictions")
CACHE_DIR       = Path("cache")
REQUEST_TIMEOUT = 15
MAX_RETRIES     = 3


# ── Known grid penalty patterns ──────────────────────────────────────────────

# These are manually maintained for the current season and injected into
# predict.py before grid position assignment.  In a production system this
# would be replaced by an FIA document parser.

# Format: {(year, round): {"Driver_abbrev": penalty_places}}
# Positive number = grid drop; negative = promotion (rare)
MANUAL_PENALTIES: dict[tuple[int, int], dict[str, int]] = {
    # Example: (2026, 5): {"VER": 10},  # 10-place engine penalty
}


# ── RSS/API Endpoints ─────────────────────────────────────────────────────────

F1_RSS_URL       = "https://www.formula1.com/content/fom-website/en/latest/all.xml"
F1_NEWS_API      = "https://api.formula1.com/v1/editorial-newslist/articles"
OPENF1_BASE      = "https://api.openf1.org/v1"

# Penalty-related keywords for filtering headlines
PENALTY_KEYWORDS = [
    "grid penalty", "grid drop", "pit lane start", "engine penalty",
    "gearbox penalty", "reprimand", "disqualified", "stewards",
    "penalty", "back of grid", "power unit", "pu change",
    "component change",
]

# Driver status keywords
STATUS_KEYWORDS = [
    "ruled out", "injury", "injured", "withdraw", "replacement",
    "reserve driver", "stand-in", "miss", "will not race",
    "sidelined", "fracture",
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, headers: dict | None = None) -> str | None:
    """GET with retries. Returns response text or None."""
    if headers is None:
        headers = {
            "User-Agent": "F1Predictor/1.0 (research project)",
            "Accept": "application/xml, application/json, text/html",
        }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                log.warning("HTTP %s from %s: %s", resp.status_code, url, e)
                return None
        except Exception as e:
            log.warning("Request to %s failed (attempt %d): %s", url, attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    return None


def _get_json(url: str, params: dict | None = None) -> dict | list | None:
    """GET JSON with retries."""
    text = _get(url, params)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── F1 News Headlines Fetcher ─────────────────────────────────────────────────

def fetch_f1_headlines(max_articles: int = 30) -> list[dict]:
    """
    Fetch latest F1 headlines from the official RSS feed.
    Returns list of {title, link, published, summary}.
    """
    headlines = []

    # Try XML RSS feed first
    xml = _get(F1_RSS_URL)
    if xml:
        # Simple XML parsing without lxml dependency
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml)
        links  = re.findall(r"<link>(https://www\.formula1\.com/.*?)</link>", xml)
        dates  = re.findall(r"<pubDate>(.*?)</pubDate>", xml)

        for i, title in enumerate(titles[:max_articles]):
            headlines.append({
                "title":     title.strip(),
                "link":      links[i] if i < len(links) else "",
                "published": dates[i] if i < len(dates) else "",
                "source":    "f1.com",
            })
        log.info("Fetched %d headlines from F1 RSS", len(headlines))

    # Fallback: try the F1 API
    if not headlines:
        data = _get_json(F1_NEWS_API, {"limit": str(max_articles)})
        if isinstance(data, dict) and "items" in data:
            for item in data["items"][:max_articles]:
                headlines.append({
                    "title":     item.get("title", ""),
                    "link":      item.get("slug", ""),
                    "published": item.get("updatedAt", ""),
                    "source":    "f1-api",
                })
            log.info("Fetched %d headlines from F1 API", len(headlines))

    return headlines


# ── Penalty & Status Classifier ───────────────────────────────────────────────

def classify_headlines(headlines: list[dict]) -> dict:
    """
    Classify headlines into categories:
      - penalties: grid/engine/gearbox penalties
      - status_changes: injuries, withdrawals, replacements
      - weather: weather-related headlines
      - other: general F1 news
    """
    result = {
        "penalties":      [],
        "status_changes": [],
        "weather":        [],
        "other":          [],
    }

    for h in headlines:
        title_lower = h["title"].lower()

        if any(kw in title_lower for kw in PENALTY_KEYWORDS):
            result["penalties"].append(h)
        elif any(kw in title_lower for kw in STATUS_KEYWORDS):
            result["status_changes"].append(h)
        elif any(w in title_lower for w in ["rain", "wet", "weather", "storm"]):
            result["weather"].append(h)
        else:
            result["other"].append(h)

    return result


# ── Grid Penalty Resolver ────────────────────────────────────────────────────

def extract_penalty_from_title(title: str) -> tuple[str | None, int | None]:
    """
    Try to extract driver abbreviation and penalty places from a headline.
    Returns (driver_abbrev, penalty_places) or (None, None).
    """
    title_lower = title.lower()

    # Common patterns:
    # "Verstappen hit with 10-place grid penalty"
    # "Hamilton given five-place grid drop"
    # "Leclerc to start from pit lane"

    # Number extraction
    number_map = {
        "three": 3, "five": 5, "ten": 10, "fifteen": 15,
        "twenty": 20, "three-place": 3, "five-place": 5,
        "10-place": 10, "5-place": 5, "3-place": 3,
    }

    penalty = None
    for word, num in number_map.items():
        if word in title_lower and ("grid" in title_lower or "penalty" in title_lower):
            penalty = num
            break

    if penalty is None:
        match = re.search(r"(\d+)[- ]place", title_lower)
        if match:
            penalty = int(match.group(1))

    if "pit lane" in title_lower or "back of grid" in title_lower:
        penalty = 20  # Effective pit lane / back of grid

    # Driver extraction — look for known 3-letter abbreviations in context
    driver_abbrevs = [
        "VER", "NOR", "LEC", "PIA", "HAM", "RUS", "SAI", "ALO",
        "GAS", "OCO", "STR", "TSU", "ALB", "SAR", "BOT", "ZHO",
        "MAG", "HUL", "RIC", "LAW", "PER", "BEA", "HAD", "ANT",
        "DRU", "BOR", "COL", "DOO", "LIN", "BER",
    ]

    found_driver = None
    for abbr in driver_abbrevs:
        # Check for abbreviation or full-ish name
        if abbr.lower() in title_lower or abbr in title:
            found_driver = abbr
            break

    return found_driver, penalty


def get_grid_penalties(year: int, round_number: int) -> dict[str, int]:
    """
    Get grid penalties for a specific round.
    Combines manual entries + auto-detected from headlines.
    """
    # Start with manual entries
    penalties = dict(MANUAL_PENALTIES.get((year, round_number), {}))

    # Try to auto-detect from headlines
    headlines = fetch_f1_headlines()
    classified = classify_headlines(headlines)

    for h in classified["penalties"]:
        driver, places = extract_penalty_from_title(h["title"])
        if driver and places and driver not in penalties:
            penalties[driver] = places
            log.info(
                "Auto-detected penalty: %s → %d places (from: %s)",
                driver, places, h["title"],
            )

    if penalties:
        log.info("Grid penalties for %d R%d: %s", year, round_number, penalties)
    else:
        log.info("No grid penalties detected for %d R%d", year, round_number)

    return penalties


# ── Main: Build news intelligence file ────────────────────────────────────────

def build_news_report(year: int, round_number: int) -> dict:
    """
    Build a comprehensive pre-race news intelligence report.
    Saved as predictions/{year}_R{round}_news.json.
    """
    log.info("Building news report for %d R%d", year, round_number)

    headlines  = fetch_f1_headlines()
    classified = classify_headlines(headlines)
    penalties  = get_grid_penalties(year, round_number)

    report = {
        "year":          year,
        "round":         round_number,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "grid_penalties": penalties,
        "headline_counts": {
            "total":          len(headlines),
            "penalties":      len(classified["penalties"]),
            "status_changes": len(classified["status_changes"]),
            "weather":        len(classified["weather"]),
        },
        "penalty_headlines": [
            {"title": h["title"], "link": h.get("link", ""), "source": h.get("source", "")}
            for h in classified["penalties"]
        ],
        "status_headlines": [
            {"title": h["title"], "link": h.get("link", ""), "source": h.get("source", "")}
            for h in classified["status_changes"]
        ],
        "weather_headlines": [
            {"title": h["title"], "link": h.get("link", ""), "source": h.get("source", "")}
            for h in classified["weather"]
        ],
        "recent_headlines": [
            {"title": h["title"], "source": h.get("source", "")}
            for h in headlines[:15]
        ],
    }

    # Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{year}_R{round_number:02d}_news.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved news report → %s", out_path)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 pre-race news & penalty scraper")
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    report = build_news_report(args.year, args.round)

    # Print summary
    print(f"\n{'='*60}")
    print(f"NEWS INTELLIGENCE — {args.year} Round {args.round}")
    print(f"{'='*60}")
    print(f"Headlines scraped:  {report['headline_counts']['total']}")
    print(f"Penalty mentions:   {report['headline_counts']['penalties']}")
    print(f"Status changes:     {report['headline_counts']['status_changes']}")
    print(f"Weather mentions:   {report['headline_counts']['weather']}")

    if report["grid_penalties"]:
        print(f"\nGRID PENALTIES:")
        for driver, places in report["grid_penalties"].items():
            print(f"  {driver}: {places}-place drop")

    if report["penalty_headlines"]:
        print(f"\nPENALTY HEADLINES:")
        for h in report["penalty_headlines"][:5]:
            print(f"  • {h['title']}")

    if report["status_headlines"]:
        print(f"\nSTATUS CHANGES:")
        for h in report["status_headlines"][:5]:
            print(f"  • {h['title']}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

