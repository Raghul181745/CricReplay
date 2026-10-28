
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CRICREPLAY - WHAT-IF SIMULATOR V2
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"

RECENT_SEASONS = {"2022", "2023", "2024", "2025", "2026"}

SIMULATIONS = 5000
RANDOM_SEED = 42
SMOOTHING_K = 80


def get_positive_integer(prompt, default, maximum=None):
    """Read a positive whole number."""

    while True:
        value = input(prompt).strip()

        try:
            number = int(value) if value else default
        except ValueError:
            print("Please enter a whole number.")
            continue

        if number < 1:
            print("Enter a positive whole number.")
            continue

        if maximum is not None and number > maximum:
            print(f"Enter a number no greater than {maximum}.")
            continue

        return number


def classify_phase(over_number):
    """Classify zero-based T20 over numbers."""

    if 0 <= over_number <= 5:
        return "Powerplay"
    if 6 <= over_number <= 14:
        return "Middle"
    if 15 <= over_number <= 19:
        return "Death"
    return None


def select_bowler(eligible, label):
    """Choose a bowler-team record using its displayed number."""

    while True:
        value = input(
            f"\nSelect Bowler {label} by number: "
        ).strip()

        try:
            number = int(value)
        except ValueError:
            print("Enter a valid whole number.")
            continue

        if not 1 <= number <= len(eligible):
            print(f"Choose a number from 1 to {len(eligible)}.")
            continue

        return eligible.iloc[number - 1]


def simulate_spells(over_records, number_of_overs, rng):
    """Sample historical complete overs with replacement."""

    if over_records.empty:
        raise ValueError("No historical overs available.")

    run_values = over_records["runs_conceded"].to_numpy(
        dtype=float
    )
    wicket_values = over_records["wickets"].to_numpy(
        dtype=float
    )

    choices = rng.integers(
        0,
        len(over_records),
        size=(SIMULATIONS, number_of_overs),
    )

    simulated_runs = run_values[choices].sum(axis=1)
    simulated_wickets = wicket_values[choices].sum(axis=1)

    return simulated_runs, simulated_wickets


def summarize_results(
    label, bowler, runs, wickets, number_of_overs
):
    """Summarize simulated outcomes for the selected spell."""

    return {
        "Option": label,
        "Bowler": bowler["bowler"],
        "Team": bowler["bowling_team"],
        "Historical overs": int(bowler["historical_overs"]),
        "Historical matches": int(bowler["historical_matches"]),
        "Mean simulated runs": runs.mean(),
        "Median simulated runs": np.median(runs),
        "Runs 10th percentile": np.percentile(runs, 10),
        "Runs 90th percentile": np.percentile(runs, 90),
        "Smoothed expected runs for spell": (
            bowler["smoothed_expected_runs"] * number_of_overs
        ),
        "Mean recorded wickets": wickets.mean(),
        "Median recorded wickets": np.median(wickets),
    }


def main():
    # --------------------------------------------------------
    # 1. Check and load data
    # --------------------------------------------------------

    if not BALL_FILE.exists() or not OVER_FILE.exists():
        print("Required data files were not found.")
        print("Run: python src/process_data.py")
        return

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
    balls["season"] = balls["season"].str.strip()
    overs["match_id"] = overs["match_id"].str.strip()

    season_map = balls[["match_id", "season"]].drop_duplicates()

    if season_map["match_id"].duplicated().any():
        raise ValueError(
            "A match ID maps to multiple seasons. "
            "Inspect the source data before proceeding."
        )

    data = overs.merge(
        season_map,
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    if data["season"].isna().any():
        raise ValueError(
            "Some over records could not be mapped to a season."
        )

    # --------------------------------------------------------
    # 2. Restrict data to the recent five seasons
    # --------------------------------------------------------

    available_seasons = set(data["season"].unique())
    missing_seasons = RECENT_SEASONS - available_seasons

    if missing_seasons:
        raise ValueError(
            f"Expected recent seasons are missing: "
            f"{sorted(missing_seasons)}"
        )

    data = data[
        data["season"].isin(RECENT_SEASONS)
    ].copy()

    # Convert columns to numeric values.
    data["over"] = pd.to_numeric(
        data["over"], errors="coerce"
    )
    data["legal_balls"] = pd.to_numeric(
        data["legal_balls"], errors="coerce"
    )
    data["runs_conceded"] = pd.to_numeric(
        data["runs_conceded"], errors="coerce"
    )
    data["wickets"] = pd.to_numeric(
        data["wickets"], errors="coerce"
    )

    # Keep complete six-legal-ball overs.
    data = data[data["legal_balls"] == 6].copy()
    data["phase"] = data["over"].apply(classify_phase)

    data = data.dropna(
        subset=[
            "match_id",
            "season",
            "phase",
            "bowler",
            "bowling_team",
            "runs_conceded",
            "wickets",
        ]
    ).copy()

    if data.empty:
        print("No complete overs available for these seasons.")
        return

    print("\n=== CRICREPLAY: WHAT-IF LAB ===")
    print("Recent-history bowling comparison and simulation.")
    print(f"Seasons included: {', '.join(sorted(RECENT_SEASONS))}")
    print(f"Complete overs available: {len(data):,}")
    print(f"Matches available: {data['match_id'].nunique():,}")

    # --------------------------------------------------------
    # 3. Choose phase and scenario settings
    # --------------------------------------------------------

    phases = {
        "1": "Powerplay",
        "2": "Middle",
        "3": "Death",
    }

    print("\nChoose a bowling phase:")
    for number, phase_name in phases.items():
        print(f"{number}. {phase_name}")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice not in phases:
        print("Invalid phase selection.")
        return

    phase = phases[choice]

    number_of_overs = get_positive_integer(
        "\nHow many overs should we simulate? (default 2): ",
        default=2,
        maximum=5,
    )

    minimum_overs = get_positive_integer(
        "Minimum historical complete overs per bowler "
        "(default 20): ",
        default=20,
    )

    phase_data = data[data["phase"] == phase].copy()

    if phase_data.empty:
        print("No historical overs are available for this phase.")
        return

    # --------------------------------------------------------
    # 4. Calculate historical bowler and team averages
    # --------------------------------------------------------

    team_stats = (
        phase_data.groupby("bowling_team")
        .agg(
            team_overs=("runs_conceded", "size"),
            team_mean_runs=("runs_conceded", "mean"),
        )
        .to_dict("index")
    )

    eligible = (
        phase_data.groupby(
            ["bowler", "bowling_team"],
            dropna=False,
        )
        .agg(
            historical_overs=("runs_conceded", "size"),
            historical_matches=("match_id", "nunique"),
            historical_mean_runs=("runs_conceded", "mean"),
            historical_total_runs=("runs_conceded", "sum"),
            recorded_wickets=("wickets", "sum"),
        )
        .reset_index()
    )

    eligible = eligible[
        eligible["historical_overs"] >= minimum_overs
    ].copy()

    if len(eligible) < 2:
        print("Not enough eligible bowlers for comparison.")
        return

    # Smooth each bowler's historical average toward their
    # team's average for the selected phase.
    eligible["team_mean_runs"] = eligible[
        "bowling_team"
    ].map(
        lambda team: team_stats[team]["team_mean_runs"]
    )

    n = eligible["historical_overs"].astype(float)

    eligible["smoothed_expected_runs"] = (
        n / (n + SMOOTHING_K)
        * eligible["historical_mean_runs"]
        + SMOOTHING_K / (n + SMOOTHING_K)
        * eligible["team_mean_runs"]
    )

    eligible = eligible.sort_values(
        ["bowler", "bowling_team"]
    ).reset_index(drop=True)

    record_groups = {
        (bowler, team): group.reset_index(drop=True)
        for (bowler, team), group in phase_data.groupby(
            ["bowler", "bowling_team"],
            dropna=False,
        )
    }

    eligible["_records"] = [
        record_groups[(row["bowler"], row["bowling_team"])]
        for _, row in eligible.iterrows()
    ]

    print(f"\nPhase: {phase}")
    print(f"Scenario length: {number_of_overs} overs")
    print(f"Simulations per bowler: {SIMULATIONS:,}")
    print(f"Minimum historical overs: {minimum_overs}")
    print(f"Smoothing constant: {SMOOTHING_K}")

    print("\nEligible bowler-team records:")

    for index, row in eligible.iterrows():
        print(
            f"{index + 1}. {row['bowler']} "
            f"({row['bowling_team']})"
            f" | Overs: {int(row['historical_overs'])}"
            f" | Matches: {int(row['historical_matches'])}"
            f" | Historical runs/over: "
            f"{row['historical_mean_runs']:.2f}"
            f" | Smoothed runs/over: "
            f"{row['smoothed_expected_runs']:.2f}"
        )

    # --------------------------------------------------------
    # 5. Choose two different bowlers
    # --------------------------------------------------------

    bowler_a = select_bowler(eligible, "A")
    bowler_b = select_bowler(eligible, "B")

    if (
        bowler_a["bowler"].casefold()
        == bowler_b["bowler"].casefold()
    ):
        print(
            "Choose two different bowlers. "
            "Select another option and run the script again."
        )
        return

    # --------------------------------------------------------
    # 6. Simulate historical outcomes
    # --------------------------------------------------------

    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    seed_a, seed_b = seed_sequence.spawn(2)

    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)

    runs_a, wickets_a = simulate_spells(
        bowler_a["_records"],
        number_of_overs,
        rng_a,
    )

    runs_b, wickets_b = simulate_spells(
        bowler_b["_records"],
        number_of_overs,
        rng_b,
    )

    # --------------------------------------------------------
    # 7. Summarize the distributions
    # --------------------------------------------------------

    results = pd.DataFrame([
        summarize_results(
            "A", bowler_a, runs_a, wickets_a, number_of_overs
        ),
        summarize_results(
            "B", bowler_b, runs_b, wickets_b, number_of_overs
        ),
    ])

    print("\n=== SIMULATED OUTCOME DISTRIBUTIONS ===")
    print(results.round(2).to_string(index=False))

    # --------------------------------------------------------
    # 8. Compare outcomes
    # --------------------------------------------------------

    print("\n=== SIMULATION COMPARISON ===")

    print(
        f"Option A has fewer simulated runs: "
        f"{np.mean(runs_a < runs_b) * 100:.1f}%"
    )
    print(
        f"Option B has fewer simulated runs: "
        f"{np.mean(runs_b < runs_a) * 100:.1f}%"
    )
    print(
        f"Equal simulated runs: "
        f"{np.mean(runs_a == runs_b) * 100:.1f}%"
    )

    print(
        f"\nOption A has more recorded wickets: "
        f"{np.mean(wickets_a > wickets_b) * 100:.1f}%"
    )
    print(
        f"Option B has more recorded wickets: "
        f"{np.mean(wickets_b > wickets_a) * 100:.1f}%"
    )
    print(
        f"Equal recorded wickets: "
        f"{np.mean(wickets_a == wickets_b) * 100:.1f}%"
    )

    # --------------------------------------------------------
    # 9. Save the current comparison
    # --------------------------------------------------------

    output_file = DATA_DIR / "latest_what_if_comparison.csv"
    results.to_csv(output_file, index=False)

    print(f"\nComparison saved to: {output_file}")

    print("\n=== IMPORTANT LIMITATIONS ===")
    print(
        "The simulation resamples historical complete overs "
        "with replacement; it does not simulate individual balls."
    )
    print(
        "Smoothed expected runs are calculated for the selected "
        "spell length. Simulated means are separate estimates "
        "derived from sampled historical overs."
    )
    print(
        "The simulation does not adjust for opponent, venue, "
        "match situation, or changes in a bowler's form."
    )
    print(
        "Recorded wickets may include dismissals not credited "
        "to the bowler. They are labelled recorded wickets here."
    )
    print(
        "Comparison percentages describe this simulation setup, "
        "not real-world probabilities of future performance."
    )

    print("\nWhat-If simulation complete.")


if __name__ == "__main__":
    main()
