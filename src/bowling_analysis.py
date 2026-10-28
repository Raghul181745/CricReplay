
from pathlib import Path

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "Data" / "processed"

OVER_FILE = DATA_DIR / "over_summary.csv"

OUTPUT_FILE = DATA_DIR / "bowler_phase_stats.csv"


# ==================================================
# 2. DEFINE T20 INNINGS PHASES
# ==================================================

def get_phase(over_number):
    """
    Classify an over using Cricsheet's zero-based numbering.

    Cricsheet index 0 = broadcast over 1.
    Standard 20-over T20 phases:
    Powerplay: overs 1-6
    Middle:    overs 7-15
    Death:     overs 16-20
    """

    if 0 <= over_number <= 5:
        return "Powerplay"

    elif 6 <= over_number <= 14:
        return "Middle"

    else:
        return "Death"


# ==================================================
# 3. LOAD AND CHECK THE DATA
# ==================================================

def main():

    print("Starting CricReplay bowling analysis...")

    if not OVER_FILE.exists():
        print(f"ERROR: Input file not found: {OVER_FILE}")
        print("Please run: python src/process_data.py")
        return

    overs = pd.read_csv(OVER_FILE, low_memory=False)

    required_columns = [
        "match_id",
        "innings",
        "over",
        "bowling_team",
        "bowler",
        "runs_conceded",
        "legal_balls",
        "wickets",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in overs.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        print("Regenerate over_summary.csv using process_data.py.")
        return

    print(f"Over-summary records loaded: {len(overs):,}")


    # ==================================================
    # 4. ASSIGN EACH OVER TO A PHASE
    # ==================================================

    overs["phase"] = overs["over"].apply(get_phase)

    print("\nOver-phase classification completed.")

    print("\nOver records by phase:")
    print(overs["phase"].value_counts().to_string())


    # ==================================================
    # 5. AGGREGATE BOWLER PERFORMANCE
    # ==================================================

    stats = overs.groupby(
        ["bowling_team", "bowler", "phase"],
        as_index=False,
    ).agg(
        runs_conceded=("runs_conceded", "sum"),
        legal_balls=("legal_balls", "sum"),
        wickets=("wickets", "sum"),
        overs_bowled=("over", "size"),
        matches=("match_id", "nunique"),
    )


    # ==================================================
    # 6. CALCULATE BOWLING METRICS
    # ==================================================

    # Economy = runs conceded per six legal balls.
    stats["economy"] = (
        stats["runs_conceded"] * 6
        / stats["legal_balls"].replace(0, float("nan"))
    )

    # Wickets per six legal balls.
    stats["wickets_per_6_balls"] = (
        stats["wickets"] * 6
        / stats["legal_balls"].replace(0, float("nan"))
    )

    # Average runs conceded per wicket.
    # Undefined when no wickets have been taken.
    stats["bowling_average"] = (
        stats["runs_conceded"]
        / stats["wickets"].replace(0, float("nan"))
    )


    # ==================================================
    # 7. ADD SAMPLE-SIZE INDICATORS
    # ==================================================

    # These help us avoid trusting very small samples.
    stats["sample_size_warning"] = (
        stats["legal_balls"] < 60
    )

    stats["sample_note"] = stats["legal_balls"].apply(
        lambda balls: (
            "Small sample"
            if balls < 60
            else "Larger sample"
        )
    )


    # ==================================================
    # 8. SORT AND SAVE RESULTS
    # ==================================================

    stats = stats.sort_values(
        by=["phase", "economy"],
        ascending=[True, True],
        na_position="last",
    )

    stats.to_csv(OUTPUT_FILE, index=False)


    # ==================================================
    # 9. DISPLAY THE REPORT
    # ==================================================

    print("\n" + "=" * 55)
    print("CRICREPLAY BOWLER PHASE ANALYSIS")
    print("=" * 55)

    print(f"Bowler-phase records: {len(stats):,}")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    print("\nSample results:")

    display_columns = [
        "bowling_team",
        "bowler",
        "phase",
        "overs_bowled",
        "matches",
        "runs_conceded",
        "legal_balls",
        "wickets",
        "economy",
        "wickets_per_6_balls",
        "sample_note",
    ]

    print(
        stats[display_columns]
        .head(20)
        .round(2)
        .to_string(index=False)
    )

    print("\nPhase summary:")

    phase_summary = stats.groupby("phase").agg(
        bowlers=("bowler", "nunique"),
        total_runs=("runs_conceded", "sum"),
        legal_balls=("legal_balls", "sum"),
        wickets=("wickets", "sum"),
    )

    phase_summary["economy"] = (
        phase_summary["total_runs"] * 6
        / phase_summary["legal_balls"].replace(
            0, float("nan")
        )
    )

    print(phase_summary.round(2).to_string())

    print("\nAnalysis completed successfully.")
    print(
        "Note: Historical performance is descriptive, "
        "not a guarantee of future results."
    )


# ==================================================
# 10. RUN THE PROGRAM
# ==================================================

if __name__ == "__main__":
    main()
