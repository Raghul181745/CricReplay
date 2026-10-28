
from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"
INPUT_FILE = DATA_DIR / "bowler_phase_stats.csv"


def select_bowler(eligible, label):
    """Select a bowler using their number in the displayed list."""

    while True:
        selection = input(
            f"\nSelect Bowler {label} by number: "
        ).strip()

        try:
            number = int(selection)
        except ValueError:
            print("Please enter a valid whole number.")
            continue

        if number < 1 or number > len(eligible):
            print(f"Choose a number between 1 and {len(eligible)}.")
            continue

        return eligible.iloc[number - 1]


def main():
    # 1. Load the historical statistics
    if not INPUT_FILE.exists():
        print("Bowler statistics not found.")
        print("Run: python src/bowling_analysis.py")
        return

    stats = pd.read_csv(INPUT_FILE)

    required_columns = [
        "bowling_team",
        "bowler",
        "phase",
        "matches",
        "legal_balls",
        "runs_conceded",
        "wickets",
    ]

    missing = [
        column for column in required_columns
        if column not in stats.columns
    ]

    if missing:
        print("Missing required columns:", missing)
        return

    # 2. Recalculate metrics from historical totals
    stats = stats[stats["legal_balls"] > 0].copy()

    stats["economy"] = (
        stats["runs_conceded"] * 6 / stats["legal_balls"]
    )

    stats["wickets_per_6_balls"] = (
        stats["wickets"] * 6 / stats["legal_balls"]
    )

    print("\n=== CRICREPLAY: WHAT-IF ENGINE ===")
    print("Compare two bowlers using historical IPL phase statistics.")

    # 3. Choose the phase
    phases = {
        "1": "Powerplay",
        "2": "Middle",
        "3": "Death",
    }

    print("\nChoose a phase:")
    for number, phase_name in phases.items():
        print(f"{number}. {phase_name}")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice not in phases:
        print("Invalid phase selection.")
        return

    phase = phases[choice]
    phase_stats = stats[stats["phase"] == phase].copy()

    # 4. Filter out very small samples
    try:
        minimum_balls_text = input(
            "Minimum legal balls for each bowler (default 60): "
        ).strip()

        minimum_balls = (
            int(minimum_balls_text) if minimum_balls_text else 60
        )

        if minimum_balls < 1:
            raise ValueError

    except ValueError:
        print("Enter a positive whole number.")
        return

    eligible = phase_stats[
        phase_stats["legal_balls"] >= minimum_balls
    ].copy()

    if len(eligible) < 2:
        print("Not enough eligible bowlers for comparison.")
        return

    # 5. Sort and number the eligible bowlers
    eligible = eligible.sort_values(
        ["bowler", "bowling_team"]
    ).reset_index(drop=True)

    print(f"\nPhase: {phase}")
    print(f"Minimum legal balls: {minimum_balls}")
    print(f"Eligible bowler records: {len(eligible)}")
    print("\nAvailable bowlers:")

    for index, row in eligible.iterrows():
        print(
            f"{index + 1}. {row['bowler']} ({row['bowling_team']})"
            f" | Balls: {int(row['legal_balls'])}"
            f" | Economy: {row['economy']:.2f}"
            f" | Wickets: {int(row['wickets'])}"
        )

    # 6. Select bowlers by number
    bowler_a = select_bowler(eligible, "A")
    bowler_b = select_bowler(eligible, "B")

    # Prevent selecting the same player twice
    if bowler_a["bowler"].casefold() == bowler_b["bowler"].casefold():
        print("You selected the same player twice.")
        print("Please rerun the program and choose two different bowlers.")
        return

    # 7. Display their historical records side by side
    comparison = pd.DataFrame([
        {
            "Option": "A",
            "Bowler": bowler_a["bowler"],
            "Team": bowler_a["bowling_team"],
            "Matches": int(bowler_a["matches"]),
            "Legal balls": int(bowler_a["legal_balls"]),
            "Runs conceded": int(bowler_a["runs_conceded"]),
            "Wickets": int(bowler_a["wickets"]),
            "Economy": bowler_a["economy"],
            "Wickets per 6 balls": bowler_a["wickets_per_6_balls"],
        },
        {
            "Option": "B",
            "Bowler": bowler_b["bowler"],
            "Team": bowler_b["bowling_team"],
            "Matches": int(bowler_b["matches"]),
            "Legal balls": int(bowler_b["legal_balls"]),
            "Runs conceded": int(bowler_b["runs_conceded"]),
            "Wickets": int(bowler_b["wickets"]),
            "Economy": bowler_b["economy"],
            "Wickets per 6 balls": bowler_b["wickets_per_6_balls"],
        },
    ])

    print("\n=== HISTORICAL WHAT-IF COMPARISON ===")
    print(f"Phase: {phase}\n")
    print(comparison.round(2).to_string(index=False))

    # 8. Calculate differences: Option A minus Option B
    economy_difference = (
        bowler_a["economy"] - bowler_b["economy"]
    )

    wicket_rate_difference = (
        bowler_a["wickets_per_6_balls"]
        - bowler_b["wickets_per_6_balls"]
    )

    print("\n=== DIFFERENCE: A MINUS B ===")
    print(f"Economy difference: {economy_difference:+.2f}")
    print(
        "Wickets-per-6-balls difference: "
        f"{wicket_rate_difference:+.2f}"
    )

    if economy_difference < 0:
        print("Historical economy advantage: Option A.")
    elif economy_difference > 0:
        print("Historical economy advantage: Option B.")
    else:
        print("Historical economy: tied.")

    if wicket_rate_difference > 0:
        print("Historical wicket-rate advantage: Option A.")
    elif wicket_rate_difference < 0:
        print("Historical wicket-rate advantage: Option B.")
    else:
        print("Historical wicket rate: tied.")

    print(
        "\nIMPORTANT: These are aggregate historical records, "
        "not predictions of the same match or situation."
    )
    print(
        "Different seasons, teams, opponents, venues and match "
        "conditions may affect the comparison."
    )


if __name__ == "__main__":
    main()
