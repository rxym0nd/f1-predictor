"""
tune.py

Hyperparameter optimization using Optuna for F1 Predictor.
Optimizes XGBRanker for Quali and XGBClassifier for Race.
Saves best parameters to models/quali_params.json and models/race_params.json.
"""

import argparse
import json
import logging
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb

from train import (
    compute_sample_weights,
    label_encode_categoricals,
    time_split,
    evaluate_quali,
    evaluate_race,
    QUALI_TARGET,
    RACE_TARGET
)
from config import QUALI_FEATURES, RACE_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")


def objective_quali(trial, train_df, test_df):
    params = {
        "objective": "rank:pairwise",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }

    MAX_GRID = 22

    train_sorted = train_df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    test_sorted  = test_df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    
    X_train = train_sorted[QUALI_FEATURES]
    X_test  = test_sorted[QUALI_FEATURES]
    
    sw_train = train_sorted.groupby(["Year", "RoundNumber"])["Year"].first().pipe(compute_sample_weights)
    
    y_train = (MAX_GRID - train_sorted[QUALI_TARGET].values)
    y_test  = (MAX_GRID - test_sorted[QUALI_TARGET].values)

    train_groups = train_sorted.groupby(["Year", "RoundNumber"]).size().values

    model = xgb.XGBRanker(**params)
    model.fit(
        X_train, y_train,
        group=train_groups,
        sample_weight=sw_train,
        verbose=False
    )

    scores = model.predict(X_test)
    test_sorted["_score"] = scores
    test_sorted["_pred_pos"] = (
        test_sorted.groupby(["Year", "RoundNumber"])["_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    
    pred_map = test_sorted.set_index(["Year", "RoundNumber", "Driver"])["_pred_pos"]
    y_pred = test_df.apply(
        lambda r: pred_map.get((r["Year"], r["RoundNumber"], r["Driver"]), 11),
        axis=1,
    ).values
    session_groups = test_df["Year"].astype(str) + "_R" + test_df["RoundNumber"].astype(str)
    
    metrics = evaluate_quali(test_df[QUALI_TARGET].values, y_pred, session_groups)
    
    # Optuna minimizes by default, so we return MAE
    return metrics["MAE_positions"]


def objective_race(trial, train_df, test_df):
    params = {
        "objective": "binary:logistic",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }

    X_train = train_df[RACE_FEATURES]
    y_train = train_df[RACE_TARGET].values
    X_test  = test_df[RACE_FEATURES]
    y_test  = test_df[RACE_TARGET].values

    sw_train = compute_sample_weights(train_df["Year"])
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        verbose=False
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # We use raw probabilities for optimization evaluation to simplify
    metrics = evaluate_race(y_test, y_pred_proba, test_df)
    
    # Optuna minimizes by default, Brier score is lower=better
    return metrics["Brier_score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Quali Optimization ───────────────────────────────────────────────────
    log.info("Starting Quali optimization...")
    quali_df = pd.read_parquet(PROCESSED_DIR / "quali_features.parquet")
    quali_df, _ = label_encode_categoricals(quali_df)
    
    # Using 2025 as validation set for Optuna
    import train
    train.TEST_YEAR = 2025
    train_q, test_q = time_split(quali_df)
    
    # We should impute missing features here so XGB doesn't get messed up if some are fully missing
    from train import compute_impute_stats, apply_impute_stats
    q_stats = compute_impute_stats(train_q, QUALI_FEATURES)
    train_q = apply_impute_stats(train_q, q_stats, QUALI_FEATURES)
    test_q  = apply_impute_stats(test_q,  q_stats, QUALI_FEATURES)

    study_q = optuna.create_study(direction="minimize")
    study_q.optimize(lambda trial: objective_quali(trial, train_q, test_q), n_trials=args.trials)
    
    best_q = study_q.best_params
    log.info("Best Quali params: %s", best_q)
    with open(MODELS_DIR / "quali_params.json", "w") as f:
        json.dump(best_q, f, indent=2)

    # ── Race Optimization ────────────────────────────────────────────────────
    log.info("Starting Race optimization...")
    race_df = pd.read_parquet(PROCESSED_DIR / "race_features.parquet")
    race_df, _ = label_encode_categoricals(race_df)
    
    train_r, test_r = time_split(race_df)
    
    r_stats = compute_impute_stats(train_r, RACE_FEATURES)
    train_r = apply_impute_stats(train_r, r_stats, RACE_FEATURES)
    test_r  = apply_impute_stats(test_r,  r_stats, RACE_FEATURES)

    study_r = optuna.create_study(direction="minimize")
    study_r.optimize(lambda trial: objective_race(trial, train_r, test_r), n_trials=args.trials)
    
    best_r = study_r.best_params
    log.info("Best Race params: %s", best_r)
    with open(MODELS_DIR / "race_params.json", "w") as f:
        json.dump(best_r, f, indent=2)

if __name__ == "__main__":
    main()
