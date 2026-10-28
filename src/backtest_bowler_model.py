
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"

TEST_SEASONS = {"2024", "2025", "2026"}
MIN_HISTORY_OVERS = 20


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("CRICREPLAY - BOWLER MODEL BACKTEST")
print("=" * 65)

balls = pd.read_csv(
    BALL_FILE,
    low_memory=False,
    dtype={"match_id": str, "season": str},
)

overs = pd.read_csv(
    OVER_FILE,
    low_memory=False,
    dtype={"match_id": str},
)

balls["season"] = balls["season"].astype(str).str.strip()
overs["match_id"] = overs["match_id"].astype(str).str.strip()
balls["match_id"] = balls["match_id"].astype(str).str.strip()

# Each match should belong to exactly one season.
season_counts = balls.groupby("match_id")["season"].nunique()

if (season_counts > 1).any():
    raise ValueError(
        "Some match IDs are associated with multiple seasons. "
        "Inspect the source data before continuing."
    )

match_seasons = (
    balls[["match_id", "season"]]
    .drop_duplicates(subset=["match_id"])
)

overs = overs.merge(
    match_seasons,
    on="match_id",
    how="left",
    validate="many_to_one",
)

if overs["season"].isna().any():
    raise ValueError(
        "Some over-summary rows could not be mapped to a season."
    )

print(f"\nBall-by-ball rows: {len(balls):,}")
print(f"Over-summary rows: {len(overs):,}")
print(f"Unique matches: {balls['match_id'].nunique():,}")


# --------------------------------------------------
# 3. PREPARE COMPLETE OVERS AND PHASES
# --------------------------------------------------

# Keep complete six-legal-ball overs only.
data = overs.loc[overs["legal_balls"] == 6].copy()

data["over"] = pd.to_numeric(data["over"], errors="coerce")
data["runs_conceded"] = pd.to_numeric(
    data["runs_conceded"], errors="coerce"
)
data["wickets"] = pd.to_numeric(data["wickets"], errors="coerce")

# Standard T20 over phases using zero-based over numbers.
def classify_phase(over_number):
    if 0 <= over_number <= 5:
        return "Powerplay"
    if 6 <= over_number <= 14:
        return "Middle"
    if 15 <= over_number <= 19:
        return "Death"
    return None


data["phase"] = data["over"].apply(classify_phase)

data = data.dropna(
    subset=[
        "season",
        "bowler",
        "phase",
        "runs_conceded",
        "wickets",
    ]
).copy()

data["season"] = data["season"].astype(str).str.strip()

data = data.loc[data["phase"].notna()].copy()

print(f"Complete valid overs: {len(data):,}")
print(f"Training seasons: before {', '.join(sorted(TEST_SEASONS))}")
print(f"Testing seasons: {', '.join(sorted(TEST_SEASONS))}")


# --------------------------------------------------
# 4. SPLIT CHRONOLOGICALLY
# --------------------------------------------------

train = data.loc[~data["season"].isin(TEST_SEASONS)].copy()
test = data.loc[data["season"].isin(TEST_SEASONS)].copy()

if train.empty or test.empty:
    raise ValueError(
        "Training or test data is empty. Check the season labels."
    )

print(f"\nTraining complete overs: {len(train):,}")
print(f"Test complete overs: {len(test):,}")
print(f"Training matches: {train['match_id'].nunique():,}")
print(f"Test matches: {test['match_id'].nunique():,}")


# --------------------------------------------------
# 5. LEARN BOWLER-PHASE HISTORICAL RATES
# --------------------------------------------------

bowler_stats = (
    train.groupby(["bowler", "phase"])
    .agg(
        historical_overs=("runs_conceded", "size"),
        historical_mean_runs=("runs_conceded", "mean"),
        historical_mean_wickets=("wickets", "mean"),
    )
    .reset_index()
)

# Fair baseline: average runs conceded per complete over
# in each phase, calculated using training data only.
phase_baseline = (
    train.groupby("phase")["runs_conceded"]
    .mean()
    .to_dict()
)

print("\nTraining-set phase averages:")
for phase in ["Powerplay", "Middle", "Death"]:
    if phase in phase_baseline:
        print(f"{phase}: {phase_baseline[phase]:.2f} runs per over")



# --------------------------------------------------
# 6. MAKE PREDICTIONS ON UNSEEN OVERS
# --------------------------------------------------

# Merge using BOTH bowler and phase.
# This must be a many-to-one merge: each test over
# should match at most one historical bowler-phase row.
test = test.merge(
    bowler_stats,
    on=["bowler", "phase"],
    how="left",
    validate="many_to_one",
)

# Keep only bowlers with enough historical experience.
eligible = test.loc[
    test["historical_overs"] >= MIN_HISTORY_OVERS
].copy()

eligible["bowler_prediction"] = eligible["historical_mean_runs"]

eligible["phase_baseline_prediction"] = eligible["phase"].map(
    phase_baseline
)

if eligible.empty:
    raise ValueError(
        "No test overs have sufficient historical bowler data. "
        "Check the data or minimum-history threshold."
    )

# Safety checks: catch accidental row multiplication.
assert len(test) == len(
    data.loc[data["season"].isin(TEST_SEASONS)]
), "Test rows unexpectedly changed during the merge!"

assert eligible["match_id"].nunique() <= test["match_id"].nunique()

print(f"\nTest overs before history filter: {len(test):,}")
print(f"Eligible test overs after history filter: {len(eligible):,}")

# --------------------------------------------------
# 7. EVALUATE PREDICTION ERRORS
# --------------------------------------------------

def calculate_mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


actual_runs = eligible["runs_conceded"].to_numpy()
bowler_predictions = eligible["bowler_prediction"].to_numpy()
baseline_predictions = eligible[
    "phase_baseline_prediction"
].to_numpy()

bowler_mae = calculate_mae(actual_runs, bowler_predictions)
baseline_mae = calculate_mae(actual_runs, baseline_predictions)

bowler_rmse = calculate_rmse(actual_runs, bowler_predictions)
baseline_rmse = calculate_rmse(actual_runs, baseline_predictions)

improvement = (
    (baseline_mae - bowler_mae) / baseline_mae * 100
    if baseline_mae > 0
    else 0.0
)

print("\n" + "=" * 65)
print("OVERALL TEST RESULTS")
print("=" * 65)

print(f"Eligible unseen overs: {len(eligible):,}")
print(f"Eligible unseen matches: {eligible['match_id'].nunique():,}")
print(f"Minimum historical overs: {MIN_HISTORY_OVERS}")

print("\nRuns conceded per over:")
print(f"Bowler model MAE:       {bowler_mae:.3f}")
print(f"Phase baseline MAE:     {baseline_mae:.3f}")
print(f"Bowler model RMSE:      {bowler_rmse:.3f}")
print(f"Phase baseline RMSE:    {baseline_rmse:.3f}")

print(f"\nMAE improvement vs baseline: {improvement:.2f}%")


# --------------------------------------------------
# 8. BREAKDOWN BY PHASE
# --------------------------------------------------

print("\n" + "=" * 65)
print("RESULTS BY PHASE")
print("=" * 65)

phase_results = []

for phase in ["Powerplay", "Middle", "Death"]:
    subset = eligible.loc[eligible["phase"] == phase]

    if subset.empty:
        continue

    actual = subset["runs_conceded"].to_numpy()
    predicted = subset["bowler_prediction"].to_numpy()
    baseline = subset["phase_baseline_prediction"].to_numpy()

    model_mae = calculate_mae(actual, predicted)
    base_mae = calculate_mae(actual, baseline)

    phase_improvement = (
        (base_mae - model_mae) / base_mae * 100
        if base_mae > 0
        else 0.0
    )

    phase_results.append(
        {
            "Phase": phase,
            "Test overs": len(subset),
            "Model MAE": round(model_mae, 3),
            "Baseline MAE": round(base_mae, 3),
            "MAE improvement %": round(phase_improvement, 2),
        }
    )

if phase_results:
    print(pd.DataFrame(phase_results).to_string(index=False))


# --------------------------------------------------
# 9. SAVE PREDICTIONS FOR INSPECTION
# --------------------------------------------------

output_file = DATA_DIR / "backtest_predictions.csv"

columns_to_save = [
    "match_id",
    "season",
    "phase",
    "bowler",
    "bowling_team",
    "over",
    "runs_conceded",
    "wickets",
    "historical_overs",
    "historical_mean_runs",
    "bowler_prediction",
    "phase_baseline_prediction",
]

eligible[columns_to_save].to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("\nBacktest complete.")
