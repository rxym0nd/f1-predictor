# Architecture — F1 Predictor

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastF1    │────▶│   ingest.py  │────▶│  data/raw/   │
│   OpenF1    │────▶│   openf1.py  │     │  (parquets)  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌───────▼───────┐
                    │   config.py  │────▶│  features.py  │
                    │   elo.py     │     │  (50+ feats)  │
                    └──────────────┘     └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │ data/processed│
                                         │ quali_features│
                                         │ race_features │
                                         └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌───────▼───────┐
                    │   tune.py    │────▶│   train.py    │
                    │   (Optuna)   │     │ (XGB models)  │
                    └──────────────┘     └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │   models/    │
                                         │ *.json *.pkl │
                                         └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌───────▼───────┐
                    │  weather.py  │────▶│  predict.py   │
                    └──────────────┘     └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │ predictions/ │
                                         │  *.json      │
                                         └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌───────▼───────┐
                    │  evaluate.py │◀────│ dashboard.py  │
                    │  batch_eval  │     │  (Streamlit)  │
                    └──────────────┘     └───────────────┘
```

---

## Data Pipeline

### 1. Ingestion (`ingest.py`)

Fetches qualifying, race, FP2, and FP3 sessions from the FastF1 API.

- **Historical seasons** (2024–2025): All sessions fetched in bulk
- **Current season** (2026): Per-session date checking — FP2/FP3/Q fetched as soon as they complete
- **Error handling**: Exponential backoff with jitter for rate limits and network errors (3 retries)
- **Idempotent**: Already-saved sessions are skipped automatically

Output: `data/raw/{year}_R{round}_{session}_{laps|results|weather}.parquet`

### 2. Sector Times (`openf1.py`)

Fetches qualifying sector times and speed trap data from the OpenF1 REST API.

- Coverage: 2023+ (reliable sector data); earlier seasons use FastF1 fallback
- Retry logic: 3 attempts with exponential backoff for HTTP errors
- Connection timeout: 30 seconds per request

Output: `data/raw/{year}_R{round}_Q_sectors.parquet`

### 3. Feature Engineering (`features.py`)

Transforms raw session data into 50+ ML-ready features across two tables:

#### Qualifying Features (1057 rows, 30+ columns)
| Category | Features |
|----------|----------|
| Quali pace | `GapToPole_s`, `BestQualiTime_s`, `QualiphaseReached` |
| Rolling form | `RollingQualiGap`, `RollingQualiPos`, `RollingQualiStdGap` |
| Constructor | `ConRollingQualiGap` |
| Circuit | `CircuitAvgQualiGap`, `CircuitAvgQualiPos`, `CircuitVisits` |
| FP pace | `FP3_BestLap_s`, `FP3_GapToFastest_s`, `FP2_LongRunPace_s` |
| Teammate | `H2H_QualiWinRate` |
| Elo | `DriverElo`, `TeamElo`, `EloGap` |
| Missing flags | `FP3_missing`, `FP2_missing` |

#### Race Features (1064 rows, 65 columns)
All qualifying features plus:
| Category | Features |
|----------|----------|
| Race form | `RollingAvgFinish`, `RollingAvgGrid`, `RollingPoints` |
| Reliability | `RollingPodiumRate`, `RollingDNFRate`, `DNFStreak` |
| Championship | `CumPointsBefore`, `ChampionshipPos_norm`, `ConChampDelta`, `ConChampPos` |
| Context | `CareerRaceCount`, `CircuitSCRate`, `GridDifficultyScore` |
| Weather | `AirTemp_mean`, `Rainfall`, `TrackTemp_mean`, `Humidity_mean`, `WindSpeed_mean` |
| Circuit type | `IsStreet`, `IsTight`, `IsPowerTrack` |
| Regulation | `YearsSinceLastRegChange` |

---

## Models

### Qualifying Model — XGBRanker

- **Task**: Learn-to-rank driver grid positions
- **Target**: `QualiPos` (position 1–20)
- **Group**: Each qualifying session (year + round) is one ranking group
- **Evaluation**: Spearman ρ correlation between predicted and actual order, MAE in positions

### Race Model — XGBClassifier

- **Task**: Binary classification (podium = top 3 finish)
- **Target**: `Podium` (1 if finished P1–P3, else 0)
- **Calibration**: Platt scaling (isotonic fallback) for well-calibrated probabilities
- **Evaluation**: Brier score, Top-3 overlap (how many of predicted top 3 actually finished there)

### Hyperparameter Tuning (`tune.py`)

Optuna TPE sampler with 7-parameter search space:

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 200–1000 |
| `learning_rate` | 0.005–0.1 (log) |
| `max_depth` | 3–10 |
| `subsample` | 0.5–0.9 |
| `colsample_bytree` | 0.5–0.95 |
| `reg_alpha` | 0.01–10 (log) |
| `reg_lambda` | 0.01–10 (log) |

Best parameters saved to `models/best_params_*.json` and auto-loaded by `train.py`.

### Training Strategy (`train.py`)

- **Sample weighting**: Recent seasons weighted higher (exponential decay); regulation-era transitions get 70% weight to prevent cross-era pollution
- **Hierarchical imputation**: Missing values filled via Circuit median → Team median → Global median → 0.0 chain
- **Missing indicators**: Binary flags (`FP3_missing`, `FP2_missing`) let the model learn when practice data is unavailable
- **Impute stats**: Saved in `models/*_metrics.json` for consistent train/predict imputation

---

## Bayesian Elo System (`elo.py`)

Each driver starts at 1500 Elo. After every race:

```
K = 32 × (20 / field_size)      # Normalize for field size
ΔElo = K × (actual_score - expected_score)
```

- **Expected score**: Based on Elo difference vs each opponent (`1 / (1 + 10^(Δ/400))`)
- **DNF handling**: Mechanical DNFs don't penalize Elo (driver gets 50th percentile score)
- **Team Elo**: Average of both drivers' Elo ratings
- **EloGap**: `DriverElo - TeamElo` — reveals intra-team strength differential

---

## Prediction Pipeline (`predict.py`)

For a given year + round, the prediction pipeline:

1. **Loads grid** — Entry list from FastF1 schedule
2. **Computes rolling features** — From historical data, era-scoped for regulation changes
3. **Injects FP data** — FP2 long-run pace and FP3 qualifying simulation
4. **Calculates Elo** — Live pre-race ratings (no leakage from future rounds)
5. **Encodes + imputes** — Using saved encoders and hierarchical impute stats
6. **Predicts** — Qualifying positions via XGBRanker, then race probabilities via calibrated XGBClassifier
7. **Normalizes** — Podium probabilities normalized to sum to 3 (exactly 3 podium spots)
8. **Outputs** — JSON with driver rankings, probabilities, and Elo ratings

### Regulation Cold-Start (Year 0)

When `YearsSinceLastRegChange == 0`:
- Rolling features restricted to current-era data only
- Pre-2026 rolling statistics are NOT carried forward
- Model relies more heavily on FP2/FP3 pace and Elo ratings
- Drivers with no same-era history get NaN → imputed to field median

---

## Dashboard (`dashboard.py`)

Streamlit app with 4 pages:

### Pre-Race Intelligence
- Podium probability horizontal bar chart (top 10)
- Full grid prediction table with team colors
- **Elo Power Rankings** — Driver and constructor Elo bar charts
- **Model Confidence Banner** — Color-coded alert based on regulation era status

### Post-Race Analysis
- Predicted vs actual comparison table
- Per-driver delta visualization
- Spearman ρ, Brier score, Top-3 overlap metrics

### Season Overview
- Accuracy trend charts (ρ, Brier) across completed rounds
- Podium overlap bar chart with chaos race (SC) flags
- Round-by-round breakdown table
- Batch evaluation trigger

### Driver Analysis
- **Radar chart** — 5 axes: quali pace, race pace, circuit affinity, form, reliability
- **Head-to-head** — Side-by-side stats for any two drivers
- **SHAP Explainer** — Feature-by-feature breakdown of why a driver received their probability

---

## Weekly Automation (`weekly_pipeline.ps1`)

Windows Task Scheduler–friendly PowerShell script:

```
Ingest → Sectors → Features → Train → Batch Eval → Predict Next Round
```

- **Critical steps** (ingest, features, train): Abort on failure
- **Non-critical steps** (sectors, batch eval): Log warning and continue
- **Logging**: Full output to `logs/pipeline_YYYY-MM-DD.log`
- **Next-round detection**: Automatically finds the next upcoming race

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastf1` | F1 session data API |
| `xgboost` | Gradient boosting models |
| `scikit-learn` | Preprocessing, calibration, metrics |
| `optuna` | Hyperparameter optimization |
| `pandas` / `numpy` | Data manipulation |
| `streamlit` | Interactive dashboard |
| `plotly` | Charts and visualizations |
| `shap` | Model explainability |
| `requests` | OpenF1 API calls |
