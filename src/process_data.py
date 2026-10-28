
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CRICREPLAY - DATA PROCESSING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "Data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "Data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOWLER_CREDITED_DISMISSALS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
    "hit the ball twice",
    "obstructing the field",
}


def get_bowling_team(teams, batting_team):
    """Return the opponent for an ordinary two-team IPL match."""
    opponents = [team for team in teams if team != batting_team]

    if len(opponents) != 1:
        raise ValueError(
            f"Expected exactly one opponent for {batting_team}; "
            f"found {opponents}"
        )

    return opponents[0]


def process_match(file_path):
    """Extract ball-by-ball records from one Cricsheet JSON file."""

    with open(file_path, "r", encoding="utf-8") as file:
        match = json.load(file)

    info = match["info"]
    rows = []

    date_values = info.get("dates", [])
    match_date = date_values[0] if date_values else None

    for innings_number, innings in enumerate(
        match["innings"], start=1
    ):
        batting_team = innings["team"]
        bowling_team = get_bowling_team(
            info["teams"], batting_team
        )

        for over_data in innings["overs"]:
            over_number = over_data["over"]

            for delivery_number, delivery in enumerate(
                over_data["deliveries"], start=1
            ):
                runs = delivery["runs"]
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                wicket_types = [
                    wicket["kind"].lower().replace("_", " ")
                    for wicket in wickets
                ]

                is_wide = "wides" in extras
                is_no_ball = "noballs" in extras

                # A wide or no-ball is not a legal delivery.
                legal_ball = not (is_wide or is_no_ball)

                # Total team runs exclude byes/leg-byes from
                # the bowler's figures, but a no-ball penalty
                # is charged to the bowler.
                excluded_extras = (
                    extras.get("byes", 0)
                    + extras.get("legbyes", 0)
                    + extras.get("penalty", 0)
                )

                bowler_runs_conceded = (
                    runs["total"] - excluded_extras
                )

                # Total dismissal events, regardless of whether
                # the bowler receives credit.
                dismissal_events = len(wickets)

                bowler_wickets = sum(
                    kind in BOWLER_CREDITED_DISMISSALS
                    for kind in wicket_types
                )

                rows.append({
                    "match_id": file_path.stem,
                    "date": match_date,
                    "season": info.get("season"),
                    "venue": info.get("venue"),
                    "innings": innings_number,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_number,
                    "delivery_number": delivery_number,
                    "batter": delivery["batter"],
                    "bowler": delivery["bowler"],
                    "batter_runs": runs["batter"],
                    "total_runs": runs["total"],
                    "extras": runs["extras"],
                    "legal_ball": legal_ball,
                    "wickets": dismissal_events,
                    "wicket_types": ",".join(wicket_types),
                    "bowler_runs_conceded": bowler_runs_conceded,
                    "bowler_wickets": bowler_wickets,
                })

    return rows


def main():
    json_files = sorted(RAW_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {RAW_DIR}")
        return

    all_rows = []
    failed_files = []

    print(f"Found {len(json_files):,} JSON files.")
    print("Processing matches...")

    for file_path in json_files:
        try:
            all_rows.extend(process_match(file_path))
        except (
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            failed_files.append((file_path.name, str(error)))

    if not all_rows:
        print("No delivery records were processed.")
        return

    deliveries = pd.DataFrame(all_rows)

    ball_file = OUTPUT_DIR / "ball_by_ball.csv"
    deliveries.to_csv(ball_file, index=False)

    # Keep the existing over_summary.csv structure and add
    # total dismissal events for downstream validation.
    group_columns = [
        "match_id",
        "innings",
        "over",
        "batting_team",
        "bowling_team",
        "bowler",
    ]

    over_summary = (
        deliveries.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            total_runs=("total_runs", "sum"),
            runs_conceded=("bowler_runs_conceded", "sum"),
            legal_balls=("legal_ball", "sum"),
            wickets=("bowler_wickets", "sum"),
            dismissal_events=("wickets", "sum"),
            delivery_records=("delivery_number", "count"),
        )
    )

    over_file = OUTPUT_DIR / "over_summary.csv"
    over_summary.to_csv(over_file, index=False)

    print("\n=== PROCESSING REPORT ===")
    print(f"JSON files found: {len(json_files):,}")
    print(
        f"Matches processed: "
        f"{deliveries['match_id'].nunique():,}"
    )
    print(f"Delivery records: {len(deliveries):,}")
    print(f"Bowler-over records: {len(over_summary):,}")
    print(f"Total team runs: {deliveries['total_runs'].sum():,}")
    print(
        "Bowler-charged runs: "
        f"{deliveries['bowler_runs_conceded'].sum():,}"
    )
    print(
        "Total dismissal events: "
        f"{deliveries['wickets'].sum():,}"
    )
    print(
        "Bowler-credited wickets: "
        f"{deliveries['bowler_wickets'].sum():,}"
    )
    print(f"Ball-by-ball file: {ball_file}")
    print(f"Over-summary file: {over_file}")

    if failed_files:
        print(f"\nFiles skipped: {len(failed_files):,}")
        for filename, error in failed_files:
            print(f"{filename}: {error}")
    else:
        print("\nNo files were skipped.")

    print("\nData processing complete.")


if __name__ == "__main__":
    main()
