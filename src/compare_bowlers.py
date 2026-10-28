
from pathlib import Path
import pandas as pd


# Find the project folder and processed data
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"
INPUT_FILE = DATA_DIR / "bowler_phase_stats.csv"


def main():
    # Step 1: Check whether the statistics file exists
    if not INPUT_FILE.exists():
        print("Bowler statistics not found.")
        print("Please run: python src/bowling_analysis.py")
        return

    # Step 2: Load the processed bowling statistics
    stats = pd.read_csv(INPUT_FILE)

    required_columns = [
        "bowling_team",
        "bowler",
        "phase",
        "matches",
        "legal_balls",
        "runs_conceded",
        "wickets",
        "economy",
        "wickets_per_6_balls",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in stats.columns
    ]

    if missing_columns:
        print("Required columns are missing:", missing_columns)
        print("Please rerun: python src/bowling_analysis.py")
        return

    print("\n=== CRICREPLAY: BOWLING COMPARISON ===")

    # Step 3: Let the user select a bowling phase
    print("\nAvailable phases:")
    print("1. Powerplay")
    print("2. Middle")
    print("3. Death")

    phase_choices = {
        "1": "Powerplay",
        "2": "Middle",
        "3": "Death",
    }

    choice = input("\nChoose a phase (1/2/3): ").strip()

    if choice not in phase_choices:
        print("Invalid choice. Please run the program again.")
        return

    selected_phase = phase_choices[choice]

    # Step 4: Ask for the minimum sample size
    minimum_balls_input = input(
        "Minimum legal balls per bowler (default 60): "
    ).strip()

    try:
        minimum_balls = (
            int(minimum_balls_input)
            if minimum_balls_input
            else 60
        )

        if minimum_balls < 1:
            raise ValueError

    except ValueError:
        print("Enter a positive whole number.")
        return

    # Step 5: Filter bowlers by phase and sample size
    candidates = stats[
        (stats["phase"] == selected_phase)
        & (stats["legal_balls"] >= minimum_balls)
    ].copy()

    if candidates.empty:
        print(
            "\nNo bowlers meet that sample-size requirement "
            "for this phase."
        )
        print("Try a lower minimum number of legal balls.")
        return

    # Step 6: Recalculate metrics to verify the saved values
    candidates["economy"] = (
        candidates["runs_conceded"] * 6
        / candidates["legal_balls"]
    )

    candidates["wickets_per_6_balls"] = (
        candidates["wickets"] * 6
        / candidates["legal_balls"]
    )

    # Step 7: Rank by lower economy, then higher wicket rate
    candidates = candidates.sort_values(
        by=["economy", "wickets_per_6_balls"],
        ascending=[True, False],
    )

    # Step 8: Select and format the output columns
    columns = [
        "bowling_team",
        "bowler",
        "phase",
        "matches",
        "legal_balls",
        "runs_conceded",
        "wickets",
        "economy",
        "wickets_per_6_balls",
    ]

    display_columns = candidates[columns].copy()

    display_columns["economy"] = (
        display_columns["economy"].round(2)
    )

    display_columns["wickets_per_6_balls"] = (
        display_columns["wickets_per_6_balls"].round(2)
    )

    # Step 9: Display the comparison
    print(f"\nSelected phase: {selected_phase}")
    print(f"Minimum legal balls: {minimum_balls}")
    print(f"Eligible bowlers: {len(candidates)}")
    print("\nHistorical comparison:\n")

    print(
        display_columns.head(20).to_string(index=False)
    )

    print(
        "\nInterpretation: lower economy means fewer runs "
        "conceded per six legal balls. Wickets per six balls "
        "measures historical wicket frequency."
    )

    print(
        "\nThese results are descriptive historical comparisons, "
        "not guaranteed outcomes or match-specific recommendations."
    )


if __name__ == "__main__":
    main()
