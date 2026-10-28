
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"

OUTPUT_FILE = DATA_DIR / "smoothed_model_predictions.csv"
RESULTS_FILE = DATA_DIR / "smoothed_model_results.csv"

# Final test seasons must not be used to tune the model.
TEST_SEASONS = {"2024", "2025", "2026"}

# Tune smoothing strength using this earlier validation period.
VALIDATION_SEASONS = {"2021", "2022", "2023"}

# Minimum historical complete overs for a bowler-phase record.
MIN_HISTORY_OVERS = 20

# Larger k means more weight is given to the team average.
SMOOTHING_VALUES = [5, 10, 20, 40, 80]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def classify_phase(over_number):
    if 0 <= over_number <= 5:
        return "Powerplay"
    if 6 <= over_number <= 14:
        return "Middle"
    if 15 <= over_number <= 19:
        return "Death"
    return None


def calculate_mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def get_stats(training_data):
    """Calculate historical statistics from the supplied data only."""

    phase_stats = (
        training_data.groupby("phase")["runs_conceded"]
        .mean()
        .to_dict()
    )

    team_phase_stats = (
        training_data.groupby(["bowling_team", "phase"])
        .agg(
            team_overs=("runs_conceded", "size"),
            team_mean_runs=("runs_conceded", "mean"),
        )
        .reset_index()
    )

    bowler_phase_stats = (
        training_data.groupby(["bowler", "phase"])
        .agg(
            historical_overs=("runs_conceded", "size"),
            bowler_mean_runs=("runs_conceded", "mean"),
        )
        .reset_index()
    )

    return phase_stats, team_phase_stats, bowler_phase_stats


def make_predictions(evaluation_data, training_data, smoothing_k):
    """Predict outcomes using training data only."""

    phase_stats, team_stats, bowler_stats = get_stats(training_data)

    result = evaluation_data.copy()

    result = result.merge(
        team_stats,
        on=["bowling_team", "phase"],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        bowler_stats,
        on=["bowler", "phase"],
        how="left",
        validate="many_to_one",
    )

    # Baseline 1: phase average.
    result["phase_prediction"] = result["phase"].map(phase_stats)

    # Baseline 2: team + phase average.
    result["team_phase_prediction"] = result[
        "team_mean_runs"
    ].fillna(result["phase_prediction"])

    # Baseline 3: bowler + phase average.
    result["bowler_phase_prediction"] = result[
        "bowler_mean_runs"
    ]

    # Smoothed model:
    # bowler weight = n / (n + k)
    # team weight = k / (n + k)
    #
    # When historical bowler data is limited, the team average
    # receives more weight. With more history, the bowler average
    # receives more weight.

    n = result["historical_overs"].astype(float)

    bowler_weight = n / (n + smoothing_k)
    team_weight = smoothing_k / (n + smoothing_k)

    result["smoothed_prediction"] = (
        bowler_weight * result["bowler_mean_runs"]
        + team_weight * result["team_phase_prediction"]
    )

    # Keep only bowlers meeting the historical experience threshold.
    result = result.loc[
        result["historical_overs"] >= MIN_HISTORY_OVERS
    ].copy()

    return result


def evaluate_methods(predictions, dataset_name):
    methods = {
        "Phase average": "phase_prediction",
        "Team + phase": "team_phase_prediction",
        "Bowler + phase": "bowler_phase_prediction",
        "Smoothed model": "smoothed_prediction",
    }

    actual = predictions["runs_conceded"].to_numpy()
    rows = []

    for method_name, column in methods.items():
        predicted = predictions[column].to_numpy()

        rows.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "Overs": len(predictions),
            "MAE": calculate_mae(actual, predicted),
            "RMSE": calculate_rmse(actual, predicted),
        })

    return rows


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

print("=" * 68)
print("CRICREPLAY - SMOOTHED BOWLER MODEL")
print("=" * 68)

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
    raise ValueError("Some over-summary records have no season.")

# Use complete six-legal-ball overs only.
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

if not TEST_SEASONS.issubset(set(data["season"])):
    raise ValueError("One or more final test seasons are missing.")

if not VALIDATION_SEASONS.issubset(set(data["season"])):
    raise ValueError("One or more validation seasons are missing.")

# Earlier seasons for fitting the tuning model.
tuning_train = data.loc[
    ~data["season"].isin(TEST_SEASONS | VALIDATION_SEASONS)
].copy()

validation = data.loc[
    data["season"].isin(VALIDATION_SEASONS)
].copy()

# Final model training uses all seasons through 2023.
final_train = data.loc[
    ~data["season"].isin(TEST_SEASONS)
].copy()

final_test = data.loc[
    data["season"].isin(TEST_SEASONS)
].copy()

print(f"\nAll complete overs: {len(data):,}")
print(f"Tuning training overs: {len(tuning_train):,}")
print(f"Validation overs: {len(validation):,}")
print(f"Final training overs: {len(final_train):,}")
print(f"Final test overs: {len(final_test):,}")


# ============================================================
# 4. TUNE SMOOTHING STRENGTH ON VALIDATION DATA
# ============================================================

print("\n" + "=" * 68)
print("STEP A - TUNING ON 2021-2023")
print("=" * 68)

tuning_rows = []
tuning_predictions = {}

for k in SMOOTHING_VALUES:
    predicted_validation = make_predictions(
        validation,
        tuning_train,
        smoothing_k=k,
    )

    tuning_predictions[k] = predicted_validation

    actual = predicted_validation["runs_conceded"].to_numpy()
    predicted = predicted_validation["smoothed_prediction"].to_numpy()

    tuning_rows.append({
        "Smoothing k": k,
        "Validation overs": len(predicted_validation),
        "MAE": calculate_mae(actual, predicted),
        "RMSE": calculate_rmse(actual, predicted),
    })

tuning_results = pd.DataFrame(tuning_rows)

print(tuning_results.round(4).to_string(index=False))

if tuning_results.empty or tuning_results["Validation overs"].max() == 0:
    raise ValueError("No eligible validation overs were found.")

# Select k using validation MAE only.
best_row = tuning_results.loc[tuning_results["MAE"].idxmin()]
best_k = int(best_row["Smoothing k"])

print(f"\nSelected smoothing k: {best_k}")
print(
    f"Validation MAE for selected k: "
    f"{best_row['MAE']:.4f}"
)


# ============================================================
# 5. FINAL EVALUATION ON UNSEEN 2024-2026 DATA
# ============================================================

print("\n" + "=" * 68)
print("STEP B - FINAL TEST ON 2024-2026")
print("=" * 68)

# Use the chosen smoothing strength without retuning on test data.
final_predictions = make_predictions(
    final_test,
    final_train,
    smoothing_k=best_k,
)

if final_predictions.empty:
    raise ValueError("No eligible final test overs were found.")

# All four methods are evaluated on exactly the same overs.
final_methods = {
    "Phase average": "phase_prediction",
    "Team + phase": "team_phase_prediction",
    "Bowler + phase": "bowler_phase_prediction",
    "Smoothed model": "smoothed_prediction",
}

test_actual = final_predictions["runs_conceded"].to_numpy()

test_rows = []

for name, column in final_methods.items():
    predicted = final_predictions[column].to_numpy()

    test_rows.append({
        "Dataset": "Test 2024-2026",
        "Method": name,
        "Overs": len(final_predictions),
        "MAE": calculate_mae(test_actual, predicted),
        "RMSE": calculate_rmse(test_actual, predicted),
    })

test_results = pd.DataFrame(test_rows)

print("\nOverall test results:")
print(test_results.round(4).to_string(index=False))

print("\nResults by test season:")

season_rows = []

for season in sorted(TEST_SEASONS):
    subset = final_predictions.loc[
        final_predictions["season"] == season
    ]

    if subset.empty:
        continue

    actual = subset["runs_conceded"].to_numpy()

    for name, column in final_methods.items():
        predicted = subset[column].to_numpy()

        season_rows.append({
            "Season": season,
            "Method": name,
            "Overs": len(subset),
            "MAE": calculate_mae(actual, predicted),
            "RMSE": calculate_rmse(actual, predicted),
        })

season_results = pd.DataFrame(season_rows)
print(season_results.round(4).to_string(index=False))


# ============================================================
# 6. SAVE RESULTS
# ============================================================

all_results = pd.concat(
    [
        tuning_results.assign(
            Dataset="Validation 2021-2023",
            Method="Smoothed model",
        ).rename(columns={"Smoothing k": "Selected/tuned k"}),
        test_results.assign(**{"Selected/tuned k": best_k}),
    ],
    ignore_index=True,
    sort=False,
)

all_results.to_csv(RESULTS_FILE, index=False)

final_predictions.to_csv(OUTPUT_FILE, index=False)

print(f"\nEligible final test overs: {len(final_predictions):,}")
print(
    f"Eligible final test matches: "
    f"{final_predictions['match_id'].nunique():,}"
)
print(f"Selected smoothing k: {best_k}")
print(f"Saved predictions: {OUTPUT_FILE}")
print(f"Saved summary results: {RESULTS_FILE}")
print("\nSmoothed model evaluation complete.")
