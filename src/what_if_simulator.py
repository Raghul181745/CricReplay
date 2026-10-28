
from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"
INPUT_FILE = DATA_DIR / "bowler_phase_stats.csv"


def choose_number(prompt, minimum, maximum=None):
    """Read a valid whole-number choice."""

    while True:
        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if number < minimum:
            print(f"Enter a number of at least {minimum}.")
            continue

        if maximum is not None and number > maximum:
            print(f"Enter a number between {minimum} and {maximum}.")
            continue

        return number


def main():
    # 1. Load historical bowler-phase statistics
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

    stats = stats[stats["legal_balls"] > 0].copy()

    stats["economy"] = (
        stats["runs_conceded"] * 6 / stats["legal_balls"]
    )

    stats["wickets_per_6_balls"] = (
        stats["wickets"] * 6 / stats["legal_balls"]
    )

    print("\n=== CRICREPLAY: BOWLING WHAT-IF SIMULATOR ===")
    print("Compare estimated outcomes of two bowling choices.")

    # 2. Select the phase
    phases = {
        "1": "Powerplay",
        "2": "Middle",
        "3": "Death",
    }

    print("\nChoose a phase:")
    for number, phase_name in phases.items():
        print(f"{number}. {phase_name}")

    phase_choice = input("Enter 1, 2, or 3: ").strip()

    if phase_choice not in phases:
        print("Invalid phase selection.")
        return

    phase = phases[phase_choice]

    # 3. Select the number of overs to compare
    overs = choose_number(
        "\nHow many overs should we compare? Enter 1-4: ",
        1,
        4,
    )

    legal_balls = overs * 6

    # 4. Filter by phase and minimum sample size
    try:
        minimum_text = input(
            "Minimum historical legal balls per bowler (default 60): "
        ).strip()

        minimum_balls = int(minimum_text) if minimum_text else 60

        if minimum_balls < 1:
            raise ValueError

    except ValueError:
        print("Enter a positive whole number.")
        return

    eligible = stats[
        (stats["phase"] == phase)
        & (stats["legal_balls"] >= minimum_balls)
    ].copy()

    if len(eligible) < 2:
        print("Not enough eligible bowler records for comparison.")
        return

    eligible = eligible.sort_values(
        ["bowler", "bowling_team"]
    ).reset_index(drop=True)

    # 5. Display the choices
    print(f"\nPhase: {phase}")
    print(f"Scenario length: {overs} overs ({legal_balls} legal balls)")
    print(f"Minimum historical sample: {minimum_balls} legal balls")
    print("\nAvailable bowlers:")

    for index, row in eligible.iterrows():
        print(
            f"{index + 1}. {row['bowler']} ({row['bowling_team']})"
            f" | Historical balls: {int(row['legal_balls'])}"
            f" | Economy: {row['economy']:.2f}"
            f" | Wickets: {int(row['wickets'])}"
        )

    # 6. Select two different bowlers
    a_number = choose_number(
        "\nSelect Bowler A by number: ",
        1,
        len(eligible),
    )

    b_number = choose_number(
        "Select Bowler B by number: ",
        1,
        len(eligible),
    )

    bowler_a = eligible.iloc[a_number - 1]
    bowler_b = eligible.iloc[b_number - 1]

    if bowler_a["bowler"].casefold() == bowler_b["bowler"].casefold():
        print("Choose two different players.")
        return

    # 7. Estimate runs and wickets using historical rates
    results = []

    for option, bowler in [("A", bowler_a), ("B", bowler_b)]:
        expected_runs = bowler["economy"] * overs

        expected_wickets = (
            bowler["wickets_per_6_balls"] * legal_balls / 6
        )

        results.append({
            "Option": option,
            "Bowler": bowler["bowler"],
            "Team": bowler["bowling_team"],
            "Historical balls": int(bowler["legal_balls"]),
            "Historical economy": bowler["economy"],
            "Estimated runs": expected_runs,
            "Estimated wickets": expected_wickets,
        })

    comparison = pd.DataFrame(results)

    # 8. Show the comparison
    print("\n=== WHAT-IF SCENARIO RESULTS ===")
    print(f"Phase: {phase}")
    print(f"Scenario duration: {overs} overs\n")
    print(comparison.round(2).to_string(index=False))

    runs_difference = (
        results[0]["Estimated runs"] - results[1]["Estimated runs"]
    )

    wickets_difference = (
        results[0]["Estimated wickets"]
        - results[1]["Estimated wickets"]
    )

    print("\n=== OPTION A MINUS OPTION B ===")
    print(f"Estimated runs difference: {runs_difference:+.2f}")
    print(f"Estimated wickets difference: {wickets_difference:+.2f}")

    if runs_difference < 0:
        print("Lower estimated runs conceded: Option A.")
    elif runs_difference > 0:
        print("Lower estimated runs conceded: Option B.")
    else:
        print("Estimated runs conceded: tied.")

    if wickets_difference > 0:
        print("Higher estimated wicket rate: Option A.")
    elif wickets_difference < 0:
        print("Higher estimated wicket rate: Option B.")
    else:
        print("Estimated wicket rate: tied.")

    print(
        "\nLIMITATION: These are historical-rate estimates, "
        "not predictions of an actual match."
    )
    print(
        "The estimates assume historical rates remain constant "
        "over the selected overs. They do not account for the "
        "opposition, venue, match situation or individual deliveries."
    )


if __name__ == "__main__":
    main()

