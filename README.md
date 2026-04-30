# 🏎️ F1 Predictor

Machine learning pipeline for Formula 1 qualifying and race predictions. Uses XGBoost models with Bayesian Elo ratings, hierarchical imputation, and regulation-era awareness to predict grid positions and podium probabilities.

## Features

- **Qualifying model** — XGBRanker predicting grid order (Spearman ρ: 0.69)
- **Race model** — Calibrated XGBClassifier predicting podium probability (Brier: 0.047)
- **Bayesian Elo** — Driver and constructor strength ratings updated race-by-race
- **Regulation cold-start** — Era-scoped rolling features prevent stale data pollution on rule changes
- **Hierarchical imputation** — Circuit → Team → Global fallback chain for missing data
- **Automated tuning** — Optuna hyperparameter optimization for both models
- **Live dashboard** — Streamlit app with radar charts, SHAP explainability, and Elo rankings
- **Weekly automation** — PowerShell pipeline for hands-free operation during the season

## Quick Start

```powershell
# Clone and install
git clone https://github.com/rxym0nd/f1-predictor.git
cd f1-predictor
python -m venv f1env
f1env\Scripts\activate
pip install -r requirements.txt

# Ingest data (fetches from FastF1 — takes 10-30 min first run)
python src/ingest.py

# Build features, tune, train
python src/features.py
python src/tune.py --trials 30
python src/train.py

# Predict next race
python src/predict.py --year 2026 --round 4

# Launch dashboard
streamlit run src/dashboard.py
```

## Pipeline Commands

| Command | Purpose |
|---------|---------|
| `python src/ingest.py` | Fetch Q, R, FP2, FP3 sessions from FastF1 |
| `python src/openf1.py` | Fetch sector times from OpenF1 API |
| `python src/features.py` | Build quali + race feature tables |
| `python src/tune.py --trials N` | Optuna hyperparameter search |
| `python src/train.py` | Train models (auto-loads tuned params) |
| `python src/predict.py --year Y --round R` | Generate race prediction |
| `python src/evaluate.py --year Y --round R` | Evaluate against actuals |
| `python src/batch_evaluate.py --year Y` | Evaluate all completed rounds |
| `streamlit run src/dashboard.py` | Launch interactive dashboard |

## Weekly Automation

```powershell
# One-time setup: create Windows Task Scheduler job
# Or run manually each Monday:
powershell -ExecutionPolicy Bypass -File src\weekly_pipeline.ps1
```

The pipeline automatically: ingests new data → rebuilds features → retrains → evaluates → predicts next round.

## Project Structure

```
f1-predictor/
├── src/
│   ├── config.py          # Feature lists, constants, regulation eras
│   ├── ingest.py          # FastF1 data fetching with exponential backoff
│   ├── openf1.py          # OpenF1 sector time fetching
│   ├── features.py        # Feature engineering (50+ features)
│   ├── elo.py             # Bayesian Elo rating system
│   ├── tune.py            # Optuna hyperparameter optimization
│   ├── train.py           # Model training with hierarchical imputation
│   ├── predict.py         # Live prediction with pre-race state awareness
│   ├── evaluate.py        # Single-round evaluation
│   ├── batch_evaluate.py  # Season-wide batch evaluation
│   ├── dashboard.py       # Streamlit dashboard
│   ├── weather.py         # Weather forecast integration
│   └── weekly_pipeline.ps1 # Automated weekly pipeline
├── data/
│   ├── raw/               # Parquet files from FastF1/OpenF1
│   └── processed/         # Feature tables (quali_features, race_features)
├── models/                # Trained models, encoders, metrics, tuned params
├── predictions/           # JSON prediction files
├── cache/                 # FastF1 cache
└── requirements.txt
```

## Model Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep-dive.

**Qualifying:** XGBRanker with 26 features → predicts grid position ordering.

**Race:** XGBClassifier (binary: podium vs non-podium) → Platt-calibrated probabilities → ranked by probability.

**Key features:** Rolling quali/race form, Elo ratings, circuit affinity, FP2/FP3 pace, championship context, weather, constructor form.

## Dashboard

The Streamlit dashboard runs locally or on Streamlit Community Cloud:

- **Pre-Race** — Podium probability bars, full grid prediction, Elo power rankings, model confidence banner
- **Post-Race** — Prediction vs actuals comparison
- **Season** — Accuracy trends (Spearman ρ, Brier, Top-3 overlap) across rounds
- **Analysis** — Driver radar charts, head-to-head comparison, SHAP feature explanations

## License

MIT
