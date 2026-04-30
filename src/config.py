"""
config.py

Single source of truth for constants shared across the pipeline.
"""

ROLLING_WINDOW: int = 5

REGULATION_CHANGE_YEARS: list[int] = [2017, 2022, 2026]
REGULATION_ERA_DECAY = 0.70


def years_since_reg_change(year: int) -> int:
    past = [y for y in REGULATION_CHANGE_YEARS if y <= year]
    return year - max(past) if past else year - REGULATION_CHANGE_YEARS[0]


def get_era_weight(train_year: int, max_year: int, base_decay: float) -> float:
    weight = base_decay ** (max_year - train_year)
    for reg_year in REGULATION_CHANGE_YEARS:
        if train_year < reg_year <= max_year:
            weight *= REGULATION_ERA_DECAY
    return float(weight)


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

# ── Circuit safety car rates ───────────────────────────────────────────────────
# Historical fraction of race laps run under SC or VSC (2018–2025 averages).
# Shared by features.py (training) and predict.py (inference) to ensure
# identical values at both stages.  Update after each season.
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
DEFAULT_SC_RATE: float = 0.18  # global average for unknown circuits


def circuit_type_flags(circuit_name: str) -> dict[str, int]:
    c = str(circuit_name)
    return {
        "IsStreetCircuit": int(c in STREET_CIRCUITS),
        "IsHighDownforce": int(c in HIGH_DOWNFORCE_CIRCUITS),
        "IsLowDownforce":  int(c in LOW_DOWNFORCE_CIRCUITS),
    }


def grid_difficulty_score(quali_pos: float, circuit_name: str) -> float:
    difficulty = CIRCUIT_OVERTAKING_DIFFICULTY.get(
        str(circuit_name), 1.0
    )
    return float(quali_pos) * difficulty


# ── Feature lists ──────────────────────────────────────────────────────────────

FP_MISSING_INDICATOR_FEATURES: list[str] = ["FP3_missing", "FP2_missing"]

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
    "DriverElo", "TeamElo", "EloGap",
    *FP_MISSING_INDICATOR_FEATURES,
    "FP3_BestLap_s", "FP3_GapToFastest_s", "FP3_PaceRank",
    "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    "FP2_Degradation_s_per_lap",
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
    "DriverElo", "TeamElo", "EloGap",
    *FP_MISSING_INDICATOR_FEATURES,
    "FP2_LongRunPace_s", "FP2_LongRunRank", "FP2_LongRunDelta_s",
    "FP2_Degradation_s_per_lap",
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