"""
simulate.py

Monte Carlo Race Simulator for f1-predictor.
Simulates a race 10,000 times using the output of predict.py to generate 
a probability distribution of finishing positions for each driver.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config import CHAOS_CIRCUITS, KNOWN_SC_RATES, DEFAULT_SC_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("predictions")
NUM_SIMULATIONS = 10000

def load_prediction(year: int, round_number: int) -> dict:
    path = PREDICTIONS_DIR / f"{year}_R{round_number:02d}_prediction.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def run_simulation(pred_data: dict, num_sims: int = NUM_SIMULATIONS) -> pd.DataFrame:
    circuit = pred_data.get("circuit", "")
    is_chaos = circuit in CHAOS_CIRCUITS
    sc_rate = KNOWN_SC_RATES.get(circuit, DEFAULT_SC_RATE)

    drivers = []
    base_scores = []
    dnf_probs = []

    # Parse drivers
    for row in pred_data["predictions"]:
        drivers.append(row["Driver"])
        q_pos = row.get("PredictedQualiPos", 20)
        podium_prob = row.get("PodiumProbability", 0.0)
        
        # Base score: lower is better. 
        # Quali pos is a strong anchor. Podium probability acts as a strong boost.
        # The expected race pace is roughly QualiPos - a bonus for podium prob.
        score = float(q_pos) - (float(podium_prob) * 5.0)
        base_scores.append(score)

        # Base DNF chance
        # E.g. 10% base + slightly higher if starting further back (mid-pack chaos)
        dnf_p = 0.08 + (q_pos / 20.0) * 0.06
        if is_chaos:
            dnf_p *= 1.4  # Increase DNF rate by 40% on chaos circuits
        dnf_probs.append(dnf_p)

    num_drivers = len(drivers)
    base_scores = np.array(base_scores)
    dnf_probs = np.array(dnf_probs)

    # Variance for the simulation
    # Base standard deviation for position variance
    base_std = 2.5
    std_dev = base_std * (1.0 + sc_rate * 2.0)
    if is_chaos:
        std_dev += 1.5

    log.info(f"Simulation parameters — Circuit: {circuit} | Chaos: {is_chaos} | SC Rate: {sc_rate:.2f} | Variance: {std_dev:.2f}")

    # Generate random scores: [num_sims, num_drivers]
    # Normal distribution around base_score
    random_scores = np.random.normal(loc=base_scores, scale=std_dev, size=(num_sims, num_drivers))

    # Determine DNFs
    # Uniform random [0,1]
    dnf_rolls = np.random.random(size=(num_sims, num_drivers))
    dnf_mask = dnf_rolls < dnf_probs

    # For DNFs, set score to infinity so they sort to the back
    random_scores[dnf_mask] = np.inf

    # In case of multiple DNFs in one simulation, we want them ranked randomly at the back.
    # Add a small random tiebreaker to all scores.
    random_scores += np.random.random(size=(num_sims, num_drivers)) * 0.01

    # Get rankings (1-indexed) for each simulation. Lower score = better position.
    # argsort gives the index of the drivers in sorted order. We need the rank of each driver.
    # The rank of a driver is the position of their index in the argsort output.
    sorted_indices = np.argsort(random_scores, axis=1)
    
    # Create an array of ranks [num_sims, num_drivers]
    ranks = np.empty_like(sorted_indices)
    for i in range(num_sims):
        # assign rank 1..20 to the drivers in order of sorted_indices
        ranks[i, sorted_indices[i]] = np.arange(1, num_drivers + 1)

    # Now, anyone who DNF'd gets rank = 99 (or similar representation) or we can just keep them ranked at the back
    # but we should record them as DNF. Let's create a separate array for actual finish pos.
    finish_pos = ranks.copy()
    finish_pos[dnf_mask] = 0 # 0 will represent DNF

    # Aggregate results
    results = []
    for i, driver in enumerate(drivers):
        driver_pos = finish_pos[:, i]
        dnf_count = np.sum(driver_pos == 0)
        
        # Positions ignoring DNFs
        valid_pos = driver_pos[driver_pos > 0]
        exp_pos = float(np.mean(valid_pos)) if len(valid_pos) > 0 else 20.0
        
        win_prob = np.sum(driver_pos == 1) / num_sims
        podium_prob_sim = np.sum((driver_pos >= 1) & (driver_pos <= 3)) / num_sims
        points_prob = np.sum((driver_pos >= 1) & (driver_pos <= 10)) / num_sims
        dnf_prob_sim = dnf_count / num_sims

        # Position distribution (1 to 20, plus DNF)
        pos_dist = {}
        for p in range(1, 21):
            pos_dist[str(p)] = round(float(np.sum(driver_pos == p) / num_sims), 4)
        pos_dist["DNF"] = round(float(dnf_prob_sim), 4)

        orig_data = pred_data["predictions"][i]
        
        results.append({
            "Driver": driver,
            "TeamName": orig_data["TeamName"],
            "PredictedQualiPos": orig_data["PredictedQualiPos"],
            "ModelPodiumProb": orig_data["PodiumProbability"],
            "SimExpectedPos": round(exp_pos, 2),
            "SimWinProb": round(win_prob, 4),
            "SimPodiumProb": round(podium_prob_sim, 4),
            "SimPointsProb": round(points_prob, 4),
            "SimDNFProb": round(dnf_prob_sim, 4),
            "PositionDistribution": pos_dist
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("SimExpectedPos").reset_index(drop=True)
    df_results["Rank"] = df_results.index + 1

    return df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--sims", type=int, default=NUM_SIMULATIONS, help="Number of simulations to run")
    args = parser.parse_args()

    log.info(f"Loading predictions for {args.year} Round {args.round}...")
    try:
        pred_data = load_prediction(args.year, args.round)
    except FileNotFoundError as e:
        log.error(e)
        return

    log.info(f"Running Monte Carlo simulation ({args.sims} iterations)...")
    df_results = run_simulation(pred_data, num_sims=args.sims)

    # Logging top 10
    log.info("=" * 72)
    log.info(f"MONTE CARLO RESULTS — {pred_data.get('event', 'Unknown')} {args.year} (N={args.sims})")
    log.info("=" * 72)
    log.info(f"{'Rank':<5} {'Driver':<5} {'Exp Pos':<8} {'Win%':<8} {'Podium%':<8} {'Pts%':<8} {'DNF%':<8}")
    log.info("-" * 72)
    for _, row in df_results.head(10).iterrows():
        log.info(f"{row['Rank']:<5} {row['Driver']:<5} {row['SimExpectedPos']:<8.2f} "
                 f"{row['SimWinProb']*100:<7.1f}% {row['SimPodiumProb']*100:<7.1f}% "
                 f"{row['SimPointsProb']*100:<7.1f}% {row['SimDNFProb']*100:<7.1f}%")
    log.info("=" * 72)

    # Save output
    out_path = PREDICTIONS_DIR / f"{args.year}_R{args.round:02d}_simulations.json"
    with open(out_path, "w") as f:
        json.dump({
            "year": args.year,
            "round": args.round,
            "event": pred_data.get("event"),
            "circuit": pred_data.get("circuit"),
            "num_simulations": args.sims,
            "results": df_results.to_dict(orient="records"),
        }, f, indent=2)
    log.info(f"Saved Monte Carlo distributions → {out_path}")

if __name__ == "__main__":
    main()
