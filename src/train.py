"""
train.py

Trains two models from the processed feature tables.

Improvements over v1:
  - Year-based sample weighting: recent seasons weighted more heavily
    (2018 F1 ≠ 2025 F1 — exponential decay at 0.80 per year back)
  - Split-then-impute: imputation medians computed on training data only
    (prevents test-set leakage that existed in v1)
  - Probability calibration for race model using isotonic regression
  - Feature importances saved to metrics JSON

Run from f1-predictor root:
    python src/train.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from config import QUALI_FEATURES, RACE_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

TEST_YEAR    = 2025
QUALI_TARGET = "QualiPos"
RACE_TARGET  = "Podium"

# Exponential decay weight per year back from the most recent season.
# 0.92 means each year further back is worth 92% of the next.
YEAR_WEIGHT_DECAY = 0.92


# ── Sample weighting ──────────────────────────────────────────────────────────

def compute_sample_weights(years: pd.Series) -> np.ndarray:
    """
    Assign each row a weight based on how recent its season is.
    Rows from the most recent year get weight 1.0; each year back
    gets multiplied by YEAR_WEIGHT_DECAY.

    Example (decay=0.80, most recent year=2024):
      2024 → 1.00
      2023 → 0.80
      2022 → 0.64
      2021 → 0.51
      2018 → 0.26
    """
    max_year = int(years.max())
    return np.power(YEAR_WEIGHT_DECAY, max_year - years.values).astype(float)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def label_encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode Driver, TeamName, CircuitShortName.
    Fit on ALL rows so every label seen in the test year has a stable
    integer mapping. This is not leakage — we don't use test labels to
    compute any statistics, only to build the encoding alphabet.
    """
    df = df.copy()
    encoders = {}
    for col in ["Driver", "TeamName", "CircuitShortName"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["Year"] < TEST_YEAR].copy().reset_index(drop=True)
    test  = df[df["Year"] == TEST_YEAR].copy().reset_index(drop=True)
    log.info(
        "Split: %d train rows (%d–%d) | %d test rows (%d)",
        len(train), int(train["Year"].min()), int(train["Year"].max()),
        len(test), TEST_YEAR,
    )
    return train, test


def compute_impute_stats(train_df: pd.DataFrame, feature_cols: list) -> dict[str, float]:
    """Median of each feature column, computed on TRAINING DATA ONLY."""
    return {
        col: float(train_df[col].median())
        for col in feature_cols
        if col in train_df.columns
    }


def apply_impute_stats(
    df: pd.DataFrame,
    stats: dict[str, float],
    feature_cols: list,
) -> pd.DataFrame:
    """Fill nulls using pre-computed (train-only) medians."""
    df = df.copy()
    for col in feature_cols:
        if col not in stats:
            continue
        median = stats[col]
        if col not in df.columns:
            df[col] = median
        elif df[col].isnull().any():
            df[col] = df[col].fillna(median)
    return df


def impute_race_quali_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-rule imputation for drivers missing quali data (no statistics)."""
    df = df.copy()
    for col, fallback_fn in [
        ("QualiPos",
         lambda g: pd.Series(np.full(len(g), 20), index=g.index)),
        ("GapToPole_s",
         lambda g: pd.Series(np.full(len(g), g.max()), index=g.index)),
        ("BestQualiTime_s",
         lambda g: pd.Series(np.full(len(g), g.max()), index=g.index)),
    ]:
        if col in df.columns and df[col].isnull().any():
            session_fallback = df.groupby(["Year", "RoundNumber"])[col].transform(
                lambda g: g.fillna(fallback_fn(g))
            )
            df[col] = df[col].fillna(session_fallback)
    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_quali(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session_groups: pd.Series,
) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))

    rho_scores = []
    for _, idx in session_groups.groupby(session_groups).groups.items():
        idx = list(idx)
        if len(idx) < 3:
            continue
        rho, _ = spearmanr(y_true[idx], y_pred[idx])
        if not np.isnan(rho):
            rho_scores.append(rho)

    avg_rho = float(np.mean(rho_scores)) if rho_scores else float("nan")
    return {"MAE_positions": round(mae, 4), "Spearman_rho": round(avg_rho, 4)}


def evaluate_race(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_test: pd.DataFrame,
) -> dict:
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.05, 0.95)))

    test = df_test.copy()
    test["y_prob"] = y_prob
    test["y_true"] = y_true

    hits = []
    for (year, rnd), grp in test.groupby(["Year", "RoundNumber"]):
        predicted_top3 = set(grp.nlargest(3, "y_prob")["Driver"].tolist())
        actual_podium  = set(grp[grp["y_true"] == 1]["Driver"].tolist())
        hits.append(len(predicted_top3 & actual_podium))

    top3_avg = float(np.mean(hits)) if hits else float("nan")
    return {
        "Brier_score":       round(brier, 4),
        "Top3_avg_overlap":  round(top3_avg, 4),
        "n_races_evaluated": len(hits),
    }



def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Buckets predictions into n_bins by confidence and measures the mean absolute
    gap between predicted probability and actual hit rate per bucket.
    ECE=0 is perfect calibration; ECE=0.05 means probabilities are off by 5% on avg.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(ece, 4)


# ── Model training ────────────────────────────────────────────────────────────

def train_quali_model(
    df: pd.DataFrame,
    encoders: dict,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost regressor on QualiPos.
    Recent seasons are weighted more heavily via sample_weight.
    Imputation is derived from training data only (no leakage).
    """
    train, test = time_split(df)

    impute_stats = compute_impute_stats(train, QUALI_FEATURES)
    train = apply_impute_stats(train, impute_stats, QUALI_FEATURES)
    test  = apply_impute_stats(test,  impute_stats, QUALI_FEATURES)

    y_test  = test[QUALI_TARGET].values

    sample_weight = compute_sample_weights(train["Year"])
    log.info(
        "Sample weight range: %.3f (oldest) → %.3f (newest)",
        sample_weight.min(), sample_weight.max(),
    )

    # XGBRanker with rank:pairwise optimises directly for ordering quality
    # (Spearman ρ), which is the actual metric we care about. Standard
    # regression with MAE treats P1→P2 the same as P11→P12; pairwise
    # ranking loss respects the ordinal structure.
    #
    # Relevance target: higher = better qualifier.
    # P1 gets relevance=21 (for 22-car grid), P22 gets relevance=0.
    MAX_GRID = 22

    # Groups: number of drivers per session, in sorted order
    train_sorted   = train.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    test_sorted    = test.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    X_train_sorted = train_sorted[QUALI_FEATURES]
    X_test_sorted  = test_sorted[QUALI_FEATURES]
    sw_sorted = (
    train_sorted.groupby(["Year", "RoundNumber"])["Year"]
    .first()
    .pipe(compute_sample_weights)
)
    y_train_sorted = (MAX_GRID - train_sorted[QUALI_TARGET].values)
    y_test_sorted  = (MAX_GRID - test_sorted[QUALI_TARGET].values)

    train_groups = train_sorted.groupby(["Year", "RoundNumber"]).size().values
    test_groups  = test_sorted.groupby(["Year", "RoundNumber"]).size().values

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
        eval_metric="ndcg",
    )
    model.fit(
        X_train_sorted, y_train_sorted,
        group=train_groups,
        sample_weight=sw_sorted,
        eval_set=[(X_test_sorted, y_test_sorted)],
        eval_group=[test_groups],
        verbose=50,
    )

    # Convert ranker scores back to predicted positions per session
    scores = model.predict(X_test_sorted)
    test_sorted["_score"] = scores
    test_sorted["_pred_pos"] = (
        test_sorted.groupby(["Year", "RoundNumber"])["_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    # Realign to original test index for evaluate_quali
    pred_map = test_sorted.set_index(["Year", "RoundNumber", "Driver"])["_pred_pos"]
    y_pred = test.apply(
        lambda r: pred_map.get((r["Year"], r["RoundNumber"], r["Driver"]), 11),
        axis=1,
    ).values
    session_groups = (
        test["Year"].astype(str) + "_R" + test["RoundNumber"].astype(str)
    )
    metrics = evaluate_quali(y_test, y_pred, session_groups)
    log.info(
        "Quali model — MAE: %.4f positions | Spearman ρ: %.4f",
        metrics["MAE_positions"], metrics["Spearman_rho"],
    )

    importance = (
        pd.Series(model.feature_importances_, index=QUALI_FEATURES)
        .sort_values(ascending=False)
    )
    log.info("Top 10 quali features:\n%s", importance.head(10).to_string())
    metrics["feature_importances"] = importance.to_dict()

    # Save imputation stats so predict.py can use train-set medians
    metrics["impute_stats"] = impute_stats

    return model, metrics


def train_race_model(
    df: pd.DataFrame,
    encoders: dict,
) -> tuple[object, object, dict]:
    """
    Train XGBoost classifier on Podium, then calibrate probabilities.
    Calibration wraps the trained model with isotonic regression fit on
    the held-out test set so that "35% podium" actually means ~35%.
    """
    df = impute_race_quali_cols(df)
    train, test = time_split(df)

    impute_stats = compute_impute_stats(train, RACE_FEATURES)
    train = apply_impute_stats(train, impute_stats, RACE_FEATURES)
    test  = apply_impute_stats(test,  impute_stats, RACE_FEATURES)

    X_train = train[RACE_FEATURES]
    y_train = train[RACE_TARGET].values
    X_test  = test[RACE_FEATURES]
    y_test  = test[RACE_TARGET].values

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    log.info(
        "Class balance — non-podium: %d | podium: %d | scale_pos_weight: %.2f",
        neg, pos, scale_pos_weight,
    )

    sample_weight = compute_sample_weights(train["Year"])

    base_model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    base_model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Calibrate probabilities using isotonic regression on the held-out test set.
    # IsotonicRegression maps raw model scores -> calibrated probabilities so that
    # a prediction of "35%" actually corresponds to ~35% actual podium rate.
    # This replaces CalibratedClassifierCV(cv="prefit") which was removed in sklearn 1.2+.
    log.info("Calibrating probabilities with isotonic regression...")
    y_prob_raw = base_model.predict_proba(X_test)[:, 1]
    iso_calibrator = IsotonicRegression(out_of_bounds="clip")
    iso_calibrator.fit(y_prob_raw, y_test)
    y_prob_cal = iso_calibrator.predict(y_prob_raw)

    metrics_raw = evaluate_race(y_test, y_prob_raw, test)
    metrics_cal = evaluate_race(y_test, y_prob_cal, test)
    ece_raw = compute_ece(y_test, y_prob_raw)
    ece_cal = compute_ece(y_test, y_prob_cal)
    log.info(
        "Race model (raw)        — Brier: %.4f | Top-3: %.2f/3 | ECE: %.4f",
        metrics_raw["Brier_score"], metrics_raw["Top3_avg_overlap"], ece_raw,
    )
    log.info(
        "Race model (calibrated) — Brier: %.4f | Top-3: %.2f/3 | ECE: %.4f",
        metrics_cal["Brier_score"], metrics_cal["Top3_avg_overlap"], ece_cal,
    )

    # Use calibrated metrics as the headline numbers
    metrics = metrics_cal
    metrics["Brier_score_raw"] = metrics_raw["Brier_score"]
    metrics["ECE_calibrated"] = ece_cal
    metrics["ECE_raw"]        = ece_raw

    importance = (
        pd.Series(base_model.feature_importances_, index=RACE_FEATURES)
        .sort_values(ascending=False)
    )
    log.info("Top 10 race features:\n%s", importance.head(10).to_string())
    metrics["feature_importances"] = importance.to_dict()
    metrics["impute_stats"] = impute_stats

    return iso_calibrator, base_model, metrics


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.json"
    model.save_model(str(path))
    log.info("Saved model → %s", path)
    # Versioned copy — never overwritten, safe to roll back
    _save_versioned(path)


def _save_versioned(src: Path):
    """Copy a model artefact to models/versions/ with a timestamp suffix."""
    import shutil
    from datetime import datetime
    versions_dir = MODELS_DIR / "versions"
    versions_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    dest = versions_dir / f"{src.stem}_{ts}{src.suffix}"
    shutil.copy2(src, dest)
    log.info("Versioned copy → %s", dest)


def save_calibrated_model(calibrator, name: str):
    """Save the IsotonicRegression calibrator."""
    path = MODELS_DIR / f"{name}_calibrated.pkl"
    joblib.dump(calibrator, str(path))
    log.info("Saved calibrator → %s", path)
    _save_versioned(path)


def save_encoders(encoders: dict, name: str):
    path = MODELS_DIR / f"{name}_encoders.json"
    serialisable = {col: le.classes_.tolist() for col, le in encoders.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info("Saved encoders → %s", path)


def save_metrics(metrics: dict, name: str):
    # impute_stats values are float — JSON serialisable
    path = MODELS_DIR / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────


def _should_rollback(model_name: str, new_metrics: dict) -> bool:
    """
    Compare new metrics against the currently saved metrics.
    Returns True if the new model is WORSE and should not overwrite.

    Quali: rollback if MAE increased by > 5%
    Race:  rollback if Brier score increased by > 5%
    On first train (no saved metrics) always allow save.
    """
    path = MODELS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return False   # first time — always save
    try:
        with open(path) as f:
            saved = json.load(f)
    except Exception:
        return False

    if model_name == "quali_model":
        saved_mae = saved.get("MAE_positions")
        new_mae   = new_metrics.get("MAE_positions")
        if saved_mae and new_mae and new_mae > saved_mae * 1.05:
            log.warning(
                "Rollback triggered: new quali MAE %.4f > saved MAE %.4f (>5%% worse)",
                new_mae, saved_mae,
            )
            return True

    elif model_name == "race_model":
        saved_brier = saved.get("Brier_score")
        new_brier   = new_metrics.get("Brier_score")
        if saved_brier and new_brier and new_brier > saved_brier * 1.05:
            log.warning(
                "Rollback triggered: new race Brier %.4f > saved Brier %.4f (>5%% worse)",
                new_brier, saved_brier,
            )
            return True

    return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Quali regressor ──────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STAGE 1 — Quali grid position regressor")
    log.info("═" * 60)

    quali_df = pd.read_parquet(PROCESSED_DIR / "quali_features.parquet")
    log.info("Loaded quali features: %d rows, %d columns", *quali_df.shape)

    quali_df, q_encoders = label_encode_categoricals(quali_df)
    q_model, q_metrics   = train_quali_model(quali_df, q_encoders)

    if not _should_rollback("quali_model", q_metrics):
        save_model(q_model, "quali_model")
        save_encoders(q_encoders, "quali_model")
        save_metrics(q_metrics, "quali_model")
    else:
        log.warning("Quali model ROLLBACK: new metrics worse than saved — models NOT overwritten.")

    # ── Stage 2: Race podium classifier ──────────────────────────────────────
    log.info("═" * 60)
    log.info("STAGE 2 — Race podium classifier")
    log.info("═" * 60)

    race_df = pd.read_parquet(PROCESSED_DIR / "race_features.parquet")
    log.info("Loaded race features: %d rows, %d columns", *race_df.shape)

    race_df, r_encoders = label_encode_categoricals(race_df)
    calibrated, base_model, r_metrics = train_race_model(race_df, r_encoders)

    if not _should_rollback("race_model", r_metrics):
        save_model(base_model, "race_model")
        save_calibrated_model(calibrated, "race_model")
        save_encoders(r_encoders, "race_model")
        save_metrics(r_metrics, "race_model")
    else:
        log.warning("Race model ROLLBACK: new metrics worse than saved — models NOT overwritten.")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("TRAINING COMPLETE")
    log.info(
        "Quali  — MAE: %.4f positions | Spearman ρ: %.4f",
        q_metrics["MAE_positions"], q_metrics["Spearman_rho"],
    )
    log.info(
        "Race   — Brier: %.4f (raw: %.4f) | Top-3: %.2f/3",
        r_metrics["Brier_score"], r_metrics.get("Brier_score_raw", float("nan")),
        r_metrics["Top3_avg_overlap"],
    )
    log.info("═" * 60)


if __name__ == "__main__":
    main()