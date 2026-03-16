"""
config.py

Single source of truth for constants shared across the pipeline.
"""

ROLLING_WINDOW: int = 5

REGULATION_CHANGE_YEARS: list[int] = [2017, 2022, 2026]


def years_since_reg_change(year: int) -> int:
    past = [y for y in REGULATION_CHANGE_YEARS if y <= year]
    return year - max(past) if past else year - REGULATION_CHANGE_YEARS[0]


TEAM_NAME_MAP: dict[str, str] = {
    "Toro Rosso":   "RB",
    "AlphaTauri":   "RB",
    "Racing Bulls": "RB",
    "Force India":  "Aston Martin",
    "Racing Point": "Aston Martin",
    "Renault":      "Alpine",
    "Alfa Romeo":   "Audi",
    "Sauber":       "Audi",
    "Kick Sauber":  "Audi",
    "Haas F1 Team": "Haas F1 Team",
    "Cadillac":     "Cadillac",
}


def normalise_team(name: str) -> str:
    return TEAM_NAME_MAP.get(str(name), str(name))


STREET_CIRCUITS: set[str] = {
    "Monaco", "Baku", "Marina Bay", "Jeddah",
    "Melbourne", "Miami", "Las Vegas", "Montreal",
}
HIGH_DOWNFORCE_CIRCUITS: set[str] = {
    "Monaco", "Marina Bay", "Budapest", "Zandvoort",
}
LOW_DOWNFORCE_CIRCUITS: set[str] = {
    "Monza", "Baku", "Spa-Francorchamps",
}
CIRCUIT_OVERTAKING_DIFFICULTY: dict[str, float] = {
    "Monaco": 2.0, "Marina Bay": 1.8, "Budapest": 1.5,
    "Zandvoort": 1.3, "Baku": 0.7, "Monza": 0.6,
    "Spa-Francorchamps": 0.8,
}
_DEFAULT_OVERTAKING_DIFFICULTY = 1.0

COMPOUND_ORDER: dict[str, int] = {
    "SOFT": 1, "MEDIUM": 2, "HARD": 3,
    "INTERMEDIATE": 4, "WET": 5, "UNKNOWN": 0,
}

# Circuits with historically high SC/VSC rate (>25% of laps).
# IsChaosCircuit=1 signals the model that grid position is less
# deterministic and mid-field podiums are more likely.
CHAOS_CIRCUITS: set[str] = {
    "Melbourne",
    "Monaco",
    "Baku",
    "São Paulo",
    "Montreal",
}


def circuit_type_flags(circuit_name: str) -> dict[str, int]:
    c = str(circuit_name)
    return {
        "IsStreetCircuit": int(c in STREET_CIRCUITS),
        "IsHighDownforce": int(c in HIGH_DOWNFORCE_CIRCUITS),
        "IsLowDownforce":  int(c in LOW_DOWNFORCE_CIRCUITS),
    }


def grid_difficulty_score(quali_pos: float, circuit_name: str) -> float:
    difficulty = CIRCUIT_OVERTAKING_DIFFICULTY.get(
        str(circuit_name), _DEFAULT_OVERTAKING_DIFFICULTY
    )
    return float(quali_pos) * difficulty


# ── Feature lists ──────────────────────────────────────────────────────────────

QUALI_FEATURES: list[str] = [
    "Driver_enc", "TeamName_enc", "CircuitShortName_enc",
    "Year", "RoundNumber",
    "YearsSinceLastRegChange",
    # ── Item #10: driver career race count ────────────────────────────────
    # Encodes experience level — rookies have no rolling history so their
    # imputed medians are misleading. The model can learn to discount
    # rolling features for low-experience drivers automatically.
    "CareerRaceCount",
    # ── End item #10 ──────────────────────────────────────────────────────
    "RollingQualiGap", "RollingQualiPos", "RollingQualiStdGap",
    "ConRollingQualiGap",
    "CircuitAvgQualiGap", "CircuitAvgQualiPos", "CircuitVisits",
    "H2H_QualiWinRate",
    "S1Gap_s", "S2Gap_s", "S3Gap_s",
    "RollingS1Gap", "RollingS2Gap", "RollingS3Gap",
    "SpeedTrap_kph", "RollingSpeedTrap",
    "FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank",
    "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    "RollingAvgFinish", "RollingAvgGrid", "RollingPoints",
    "RollingPodiumRate", "RollingDNFRate", "DNFStreak",
    "ConRollingAvgFinish", "ConRollingPoints",
    # ── Item #5: constructor championship delta ───────────────────────────
    "ConChampDelta",        # points gap between this constructor and P1 constructor
    "ConChampPos",          # constructor championship position (1 = leading)
    # ── End item #5 ───────────────────────────────────────────────────────
    "CumPointsBefore", "ChampionshipPos_norm",
    # ── Item #9: circuit safety car frequency ─────────────────────────────
    "CircuitSCRate",        # historical fraction of laps run under SC/VSC at this circuit
    # ── End item #9 ───────────────────────────────────────────────────────
    "AirTemp_mean", "TrackTemp_mean", "Humidity_mean", "Rainfall_any",
    "IsStreetCircuit", "IsHighDownforce", "IsLowDownforce",
]

RACE_FEATURES: list[str] = [
    "Driver_enc", "TeamName_enc", "CircuitShortName_enc",
    "Year", "RoundNumber",
    "YearsSinceLastRegChange",
    # Item #10
    "CareerRaceCount",
    "QualiPos", "GapToPole_s", "BestQualiTime_s",
    "QualiphaseReached",
    "GridPenaltyPlaces",
    "GridDifficultyScore",
    "RollingQualiGap", "RollingQualiPos", "ConRollingQualiGap",
    "CircuitAvgQualiGap", "CircuitAvgQualiPos", "CircuitVisits",
    "H2H_QualiWinRate",
    "RollingS1Gap", "RollingS2Gap", "RollingS3Gap", "RollingSpeedTrap",
    "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    # Tyre compound + degradation (items #3 + #8)
    "StartCompound_enc",
    "StartTyreLife",
    "FreshStartTyre",
    "RollingAvgStints",
    "RollingAvgDegRate",
    "CircuitAvgStints",
    "CircuitDegRate",
    "RollingAvgFinish", "RollingAvgGrid", "RollingPoints",
    "RollingPodiumRate", "RollingDNFRate", "DNFStreak",
    "ConRollingAvgFinish", "ConRollingPoints",
    # Item #5
    "ConChampDelta",
    "ConChampPos",
    # Item #9
    "CircuitSCRate",
    "CumPointsBefore", "ChampionshipPos_norm",
    "AirTemp_mean", "TrackTemp_mean", "Humidity_mean", "Rainfall_any",
    "IsStreetCircuit", "IsHighDownforce", "IsLowDownforce",
    # Item #21 — chaos detection
    "IsChaosCircuit",
]