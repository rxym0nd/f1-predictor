"""
elo.py

Bayesian Elo rating system for F1 drivers and constructors.
Updates ratings after every race and provides pre-race ratings for predictions.
"""
import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 16.0

def update_elo(current_rating: float, actual_score: float, expected_score: float, k: float = K_FACTOR) -> float:
    return current_rating + k * (actual_score - expected_score)

def get_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def compute_race_elo_updates(race_df: pd.DataFrame, current_elo: dict[str, float]) -> dict[str, float]:
    """
    Computes new Elo ratings after a single race.
    race_df must contain 'Driver', 'FinishPos', 'DNF'.
    """
    drivers = race_df["Driver"].tolist()
    positions = race_df["FinishPos"].tolist()
    dnfs = race_df.get("DNF", pd.Series([0]*len(drivers))).tolist()
    
    n = len(drivers)
    if n <= 1:
        return current_elo.copy()
        
    new_elo = current_elo.copy()
    k_scaled = K_FACTOR / (n - 1)
    
    ratings = {d: current_elo.get(d, INITIAL_ELO) for d in drivers}
    
    for i in range(n):
        d_a = drivers[i]
        pos_a = positions[i]
        dnf_a = dnfs[i]
        
        actual_total = 0.0
        expected_total = 0.0
        
        for j in range(n):
            if i == j:
                continue
            d_b = drivers[j]
            pos_b = positions[j]
            dnf_b = dnfs[j]
            
            expected_total += get_expected_score(ratings[d_a], ratings[d_b])
            
            if dnf_a and dnf_b:
                actual_total += 0.5  # draw
            elif dnf_a:
                actual_total += 0.0  # loss
            elif dnf_b:
                actual_total += 1.0  # win
            elif pos_a < pos_b:
                actual_total += 1.0
            elif pos_a > pos_b:
                actual_total += 0.0
            else:
                actual_total += 0.5
                
        new_elo[d_a] = update_elo(ratings[d_a], actual_total, expected_total, k_scaled)
        
    return new_elo

def append_elo_features(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through historical races sequentially and computes pre-race Elo
    for drivers and teams. Appends DriverElo, TeamElo, EloGap columns.
    """
    df = race_results.copy()
    
    df = df.sort_values(["Year", "RoundNumber", "FinishPos"])
    
    driver_elo = {}
    team_elo = {}
    
    df["DriverElo"] = INITIAL_ELO
    df["TeamElo"] = INITIAL_ELO
    
    for (year, round_number), group in df.groupby(["Year", "RoundNumber"], sort=False):
        for idx, row in group.iterrows():
            df.loc[idx, "DriverElo"] = driver_elo.get(row["Driver"], INITIAL_ELO)
            df.loc[idx, "TeamElo"] = team_elo.get(row["TeamName"], INITIAL_ELO)
            
        driver_elo = compute_race_elo_updates(group, driver_elo)
        
        team_group = group.groupby("TeamName").agg(
            FinishPos=("FinishPos", "min"),
            DNF=("DNF", "min")
        ).reset_index().rename(columns={"TeamName": "Driver"})
        
        team_updates = compute_race_elo_updates(team_group, team_elo)
        team_elo.update(team_updates)

    df["EloGap"] = df["TeamElo"] - df["DriverElo"]
    return df

def get_current_elo(race_results: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns the final (Driver Elo, Team Elo) dictionaries after processing all races.
    """
    df = race_results.copy()
    df = df.sort_values(["Year", "RoundNumber", "FinishPos"])
    
    driver_elo = {}
    team_elo = {}
    
    for (year, round_number), group in df.groupby(["Year", "RoundNumber"], sort=False):
        driver_elo = compute_race_elo_updates(group, driver_elo)
        
        team_group = group.groupby("TeamName").agg(
            FinishPos=("FinishPos", "min"),
            DNF=("DNF", "min")
        ).reset_index().rename(columns={"TeamName": "Driver"})
        
        team_updates = compute_race_elo_updates(team_group, team_elo)
        team_elo.update(team_updates)
        
    return driver_elo, team_elo
