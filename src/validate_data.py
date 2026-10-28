
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

def main():
    if not BALL_FILE.exists():
        print(f"ERROR: File not found: {BALL_FILE}")
        return

    if not OVER_FILE.exists():
        print(f"ERROR: File not found: {OVER_FILE}")
        return

    deliveries = pd.read_csv(BALL_FILE, low_memory=False)
    overs = pd.read_csv(OVER_FILE, low_memory=False)

    print("=== CRICREPLAY DATA VALIDATION ===")

    # Dataset sizes
    print("\n1. DATASET SIZES")
    print(f"Delivery rows: {len(deliveries):,}")
    print(f"Over-summary rows: {len(overs):,}")

    # Required columns
    delivery_columns = [
        "match_id",
        "innings",
        "over",
        "delivery_number",
        "batting_team",
        "bowling_team",
        "bowler",
        "total_runs",
        "legal_ball",
        "wickets",
        "bowler_runs_conceded",
        "bowler_wickets",
    ]

    summary_columns = [
        "match_id",
        "innings",
        "over",
        "bowler",
        "runs_conceded",
        "legal_balls",
        "wickets",
    ]

    print("\n2. REQUIRED COLUMNS")

    missing_delivery_columns = [
        column
        for column in delivery_columns
        if column not in deliveries.columns
    ]

    missing_summary_columns = [
        column
        for column in summary_columns
        if column not in overs.columns
    ]

    print(f"Missing delivery columns: {missing_delivery_columns}")
    print(f"Missing summary columns: {missing_summary_columns}")

    if missing_delivery_columns or missing_summary_columns:
        print("\nValidation stopped: required columns are missing.")
        print("Run process_data.py again using the updated code.")
        return

    # Missing values
    print("\n3. MISSING VALUES")

    for column in delivery_columns:
        print(
            f"{column}: "
            f"{deliveries[column].isna().sum():,}"
        )

    # Match and team coverage
    print("\n4. MATCH AND TEAM COVERAGE")
    print(f"Unique matches: {deliveries['match_id'].nunique():,}")
    print(
        "Batting teams: "
        f"{deliveries['batting_team'].nunique()}"
    )
    print(
        "Bowling teams: "
        f"{deliveries['bowling_team'].nunique()}"
    )

    # Delivery uniqueness
    print("\n5. DELIVERY UNIQUENESS")

    delivery_key = [
        "match_id",
        "innings",
        "over",
        "delivery_number",
    ]

    duplicate_count = deliveries.duplicated(
        subset=delivery_key
    ).sum()

    print(f"Duplicate delivery keys: {duplicate_count:,}")

    # Run and delivery checks
    print("\n6. RUN AND DELIVERY CHECKS")
    print(f"Total runs: {deliveries['total_runs'].sum():,}")
    print(
        "Legal deliveries: "
        f"{deliveries['legal_ball'].sum():,}"
    )
    print(
        "Illegal deliveries: "
        f"{(~deliveries['legal_ball']).sum():,}"
    )
    print(
        "Negative total runs: "
        f"{(deliveries['total_runs'] < 0).sum():,}"
    )
    print(
        "Negative bowler-charged runs: "
        f"{(deliveries['bowler_runs_conceded'] < 0).sum():,}"
    )

    # Wicket checks
    print("\n7. WICKET CHECKS")
    print(
        "All dismissal events: "
        f"{deliveries['wickets'].sum():,}"
    )
    print(
        "Bowler-credited wickets: "
        f"{deliveries['bowler_wickets'].sum():,}"
    )

    # Check summary totals against delivery-level totals.
    print("\n8. SUMMARY CONSISTENCY")

    expected_runs = deliveries["bowler_runs_conceded"].sum()
    summary_runs = overs["runs_conceded"].sum()

    expected_legal_balls = deliveries["legal_ball"].sum()
    summary_legal_balls = overs["legal_balls"].sum()

    expected_wickets = deliveries["bowler_wickets"].sum()
    summary_wickets = overs["wickets"].sum()

    print(
        f"Bowler runs match: "
        f"{expected_runs == summary_runs}"
    )
    print(
        f"Legal-ball totals match: "
        f"{expected_legal_balls == summary_legal_balls}"
    )
    print(
        f"Bowler-wicket totals match: "
        f"{expected_wickets == summary_wickets}"
    )

    print("\n=== VALIDATION COMPLETE ===")
    print(
        "Note: Matching totals confirm consistency between the "
        "two files, not full cricket-rule correctness."
    )


if __name__ == "__main__":
    main()
