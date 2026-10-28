
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"
OUTPUT_FILE = DATA_DIR / "baseline_comparison.csv"

TEST_SEASONS = {"2024", "2025", "2026"}
MIN_HISTORY_OVERS = 20


def classify_phase(over_number):
    if 0 <= over_number <= 5:
        return "Powerplay"
    if 6 <= over_number <= 14:
        return "Middle"
    if 15 <= over_number <= 19:
        return "Death"
    return None


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# 1. Load data and map seasons to over-summary records.
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

balls["match_id"] = balls["match_id"].str.strip()
overs["match_id"] = overs["match_id"].str.strip()
balls["season"] = balls["season"].str.strip()

season_map = balls[["match_id", "season"]].drop_duplicates()

if season_map["match_id"].duplicated().any():
    raise ValueError("A match ID maps to multiple seasons.")

overs = overs.merge(
    season_map,
    on="match_id",
    how="left",
    validate="many_to_one",
)

if overs["season"].isna().any():
    raise ValueError("Some over records have no season.")

# 2. Keep complete overs and assign phases.
data = overs.loc[overs["legal_balls"] == 6].copy()

data["over"] = pd.to_numeric(data["over"], errors="coerce")
data["runs_conceded"] = pd.to_numeric(
    data["runs_conceded"], errors="coerce"
)

data["phase"] = data["over"].apply(classify_phase)

data = data.dropna(
    subset=[
        "season",
        "phase",
        "bowler",
        "bowling_team",
        "runs_conceded",
    ]
).copy()

data["season"] = data["season"].astype(str).str.strip()

train = data.loc[~data["season"].isin(TEST_SEASONS)].copy()
test = data.loc[data["season"].isin(TEST_SEASONS)].copy()

print(f"Training overs: {len(train):,}")
print(f"Test overs: {len(test):,}")

# 3. Calculate all historical averages from training data only.
phase_stats = (
    train.groupby("phase")["runs_conceded"]
    .mean()
    .rename("phase_prediction")
    .reset_index()
)

team_phase_stats = (
    train.groupby(["bowling_team", "phase"])["runs_conceded"]
    .mean()
    .rename("team_phase_prediction")
    .reset_index()
)

bowler_phase_stats = (
    train.groupby(["bowler", "phase"])
    .agg(
        historical_overs=("runs_conceded", "size"),
        bowler_phase_prediction=("runs_conceded", "mean"),
    )
    .reset_index()
)

# 4. Merge without multiplying test rows.
test = test.merge(
    phase_stats,
    on="phase",
    how="left",
    validate="many_to_one",
)

test = test.merge(
    team_phase_stats,
    on=["bowling_team", "phase"],
    how="left",
    validate="many_to_one",
)

test = test.merge(
    bowler_phase_stats,
    on=["bowler", "phase"],
    how="left",
    validate="many_to_one",
)

assert len(test) == len(
    data.loc[data["season"].isin(TEST_SEASONS)]
), "Unexpected test-row multiplication!"

# Keep the same eligible overs for a fair comparison.
eligible = test.loc[
    test["historical_overs"] >= MIN_HISTORY_OVERS
].copy()

if eligible.empty:
    raise ValueError("No test overs meet the history threshold.")

# If a team-phase combination is unavailable, fall back to
# the overall phase average, calculated from training data.
eligible["team_phase_prediction"] = (
    eligible["team_phase_prediction"].fillna(
        eligible["phase_prediction"]
    )
)

methods = {
    "Phase average": "phase_prediction",
    "Team + phase": "team_phase_prediction",
    "Bowler + phase": "bowler_phase_prediction",
}

# 5. Overall comparison.
actual = eligible["runs_conceded"].to_numpy()

overall_results = []

for name, column in methods.items():
    predictions = eligible[column].to_numpy()

    overall_results.append(
        {
            "Method": name,
            "Test overs": len(eligible),
            "MAE": round(mae(actual, predictions), 4),
            "RMSE": round(rmse(actual, predictions), 4),
        }
    )

print("\n" + "=" * 65)
print("OVERALL BASELINE COMPARISON")
print("=" * 65)
print(pd.DataFrame(overall_results).to_string(index=False))

# 6. Results by test season.
print("\n" + "=" * 65)
print("RESULTS BY SEASON")
print("=" * 65)

season_results = []

for season in sorted(TEST_SEASONS):
    subset = eligible.loc[eligible["season"] == season]

    if subset.empty:
        continue

    actual_season = subset["runs_conceded"].to_numpy()

    for name, column in methods.items():
        predictions = subset[column].to_numpy()

        season_results.append(
            {
                "Season": season,
                "Method": name,
                "Overs": len(subset),
                "MAE": round(mae(actual_season, predictions), 4),
                "RMSE": round(rmse(actual_season, predictions), 4),
            }
        )

season_df = pd.DataFrame(season_results)
print(season_df.to_string(index=False))

# 7. Results by phase.
print("\n" + "=" * 65)
print("RESULTS BY PHASE")
print("=" * 65)

phase_results = []

for phase in ["Powerplay", "Middle", "Death"]:
    subset = eligible.loc[eligible["phase"] == phase]

    if subset.empty:
        continue

    actual_phase = subset["runs_conceded"].to_numpy()

    for name, column in methods.items():
        predictions = subset[column].to_numpy()

        phase_results.append(
            {
                "Phase": phase,
                "Method": name,
                "Overs": len(subset),
                "MAE": round(mae(actual_phase, predictions), 4),
                "RMSE": round(rmse(actual_phase, predictions), 4),
            }
        )

print(pd.DataFrame(phase_results).to_string(index=False))

# 8. Save predictions and metrics.
eligible.to_csv(OUTPUT_FILE, index=False)

print(f"\nEligible test overs: {len(eligible):,}")
print(f"Eligible test matches: {eligible['match_id'].nunique():,}")
print(f"Saved comparison data to: {OUTPUT_FILE}")
print("\nEvaluation complete.")
