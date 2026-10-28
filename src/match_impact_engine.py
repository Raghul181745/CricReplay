"""Experimental IPL full-match what-if simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIMULATIONS = 5000


def classify_phase(over_number: int) -> str | None:
    over_number = int(over_number)

    if 0 <= over_number <= 5:
        return "Powerplay"
    if 6 <= over_number <= 14:
        return "Middle"
    if 15 <= over_number <= 19:
        return "Death"
    return None


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def prepare_deliveries(balls: pd.DataFrame) -> pd.DataFrame:
    df = balls.copy()

    required = [
        "match_id",
        "innings",
        "over",
        "batting_team",
        "bowling_team",
        "bowler",
        "total_runs",
        "legal_ball",
        "wickets",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    for column in ["innings", "over", "total_runs", "wickets"]:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    df["legal_ball"] = _as_bool(df["legal_ball"])

    for column in ["bowler", "bowling_team", "batting_team"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["phase"] = df["over"].map(classify_phase)

    return df[df["phase"].notna()].copy()


def _pool_arrays(pool: pd.DataFrame):
    return (
        pool["total_runs"].to_numpy(dtype=np.int16),
        pool["legal_ball"].to_numpy(dtype=bool),
        pool["wickets"].to_numpy(dtype=np.int8),
    )


def _draw_over(
    pool_arrays,
    rng,
    runs,
    wickets,
    legal_balls,
    target=None,
):
    run_values, legal_values, wicket_values = pool_arrays

    if len(run_values) == 0:
        return runs, wickets, legal_balls

    legal_this_over = 0
    attempts = 0

    while (
        legal_this_over < 6
        and wickets < 10
        and attempts < 30
    ):
        index = int(rng.integers(0, len(run_values)))

        runs += int(run_values[index])
        wickets += int(wicket_values[index])
        legal_balls += int(legal_values[index])
        legal_this_over += int(legal_values[index])
        attempts += 1

        if target is not None and runs >= target:
            break

    return runs, wickets, legal_balls


def _build_pool_cache(df):
    team_phase = {}
    team_all = {}

    for team, team_rows in df.groupby("bowling_team", sort=False):
        team_all[str(team)] = _pool_arrays(team_rows)

        for phase, phase_rows in team_rows.groupby("phase", sort=False):
            team_phase[(str(team), str(phase))] = _pool_arrays(
                phase_rows
            )

    all_pool = _pool_arrays(df)

    return team_phase, team_all, all_pool


def _simulate_innings(
    batting_team,
    bowling_team,
    start_over,
    start_runs,
    start_wickets,
    start_legal_balls,
    rng,
    team_phase_pools,
    team_all_pools,
    all_pool,
    target=None,
    selected_over=None,
    selected_bowler_pool=None,
):
    runs = int(start_runs)
    wickets = int(start_wickets)
    legal_balls = int(start_legal_balls)

    for over_no in range(int(start_over), 20):
        if wickets >= 10:
            break

        if target is not None and runs >= target:
            break

        if (
            selected_over is not None
            and over_no == selected_over
            and selected_bowler_pool is not None
        ):
            pool_arrays = selected_bowler_pool
        else:
            phase = classify_phase(over_no)

            pool_arrays = team_phase_pools.get(
                (str(bowling_team), phase)
            )

            if pool_arrays is None or len(pool_arrays[0]) == 0:
                pool_arrays = team_all_pools.get(
                    str(bowling_team), all_pool
                )

        runs, wickets, legal_balls = _draw_over(
            pool_arrays,
            rng,
            runs,
            wickets,
            legal_balls,
            target=target,
        )

    return {
        "runs": runs,
        "wickets": wickets,
        "legal_balls": legal_balls,
    }


def _simulate_chase(
    batting_team,
    bowling_team,
    target,
    rng,
    team_phase_pools,
    team_all_pools,
    all_pool,
):
    innings = _simulate_innings(
        batting_team,
        bowling_team,
        start_over=0,
        start_runs=0,
        start_wickets=0,
        start_legal_balls=0,
        rng=rng,
        team_phase_pools=team_phase_pools,
        team_all_pools=team_all_pools,
        all_pool=all_pool,
        target=target,
    )

    if innings["runs"] >= target:
        outcome = "batting_win"
    elif innings["runs"] == target - 1:
        outcome = "tie"
    else:
        outcome = "bowling_win"

    return innings, outcome


def simulate_match_impact(
    balls: pd.DataFrame,
    match_id,
    innings_no: int,
    selected_over: int,
    replacement_bowler: str,
    simulations: int = SIMULATIONS,
    seed: int = 42,
) -> dict:
    """
    Compare the original bowler with a replacement bowler.

    Over numbers are zero-based, matching the processed dataset.
    Simulations hold the score before the selected over fixed.
    """

    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    df = prepare_deliveries(balls)

    match = df[
        df["match_id"].astype(str) == str(match_id)
    ].copy()

    if match.empty:
        raise ValueError("The selected match was not found.")

    innings_no = int(innings_no)
    selected_over = int(selected_over)

    chosen = match[
        (match["innings"] == innings_no)
        & (match["over"] == selected_over)
    ]

    if chosen.empty:
        raise ValueError(
            "No deliveries were found for the selected innings and over."
        )

    batting_team = str(chosen["batting_team"].iloc[0])
    bowling_team = str(chosen["bowling_team"].iloc[0])

    bowler_mode = chosen["bowler"].mode()
    original_bowler = (
        str(bowler_mode.iloc[0]) if not bowler_mode.empty else "Unknown"
    )

    selected_phase = classify_phase(selected_over)

    original_pool = df[
        (df["bowler"] == original_bowler)
        & (df["bowling_team"] == bowling_team)
        & (df["phase"] == selected_phase)
    ]

    if original_pool.empty:
        original_pool = df[
            (df["bowling_team"] == bowling_team)
            & (df["phase"] == selected_phase)
        ]

    replacement_pool = df[
        (df["bowler"] == str(replacement_bowler))
        & (df["bowling_team"] == bowling_team)
        & (df["phase"] == selected_phase)
    ]

    if original_pool.empty:
        raise ValueError(
            "No historical deliveries were found for the original bowler "
            "or bowling team in this phase."
        )

    if replacement_pool.empty:
        raise ValueError(
            "No historical deliveries were found for the replacement "
            "bowler in this phase and team."
        )

    before = match[
        (match["innings"] == innings_no)
        & (match["over"] < selected_over)
    ]

    start_runs = int(before["total_runs"].sum())
    start_wickets = int(before["wickets"].sum())
    start_legal = int(before["legal_ball"].sum())

    first_innings = match[match["innings"] == 1]

    chase_target = None

    if innings_no >= 2 and not first_innings.empty:
        chase_target = int(first_innings["total_runs"].sum()) + 1

    second_rows = match[match["innings"] == 2]

    second_batting_team = (
        str(second_rows["batting_team"].iloc[0])
        if not second_rows.empty
        else ""
    )

    second_bowling_team = (
        str(second_rows["bowling_team"].iloc[0])
        if not second_rows.empty
        else ""
    )

    team_phase_pools, team_all_pools, all_pool = _build_pool_cache(df)

    scenarios = {
        "original": _pool_arrays(original_pool),
        "replacement": _pool_arrays(replacement_pool),
    }

    results = {}

    for scenario_index, (scenario, selected_pool) in enumerate(
        scenarios.items()
    ):
        rng = np.random.default_rng(seed + scenario_index)

        batting_wins = 0
        bowling_wins = 0
        ties = 0

        first_scores = []
        second_scores = []

        for _ in range(simulations):
            if innings_no == 1:
                first = _simulate_innings(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    start_over=selected_over,
                    start_runs=start_runs,
                    start_wickets=start_wickets,
                    start_legal_balls=start_legal,
                    rng=rng,
                    team_phase_pools=team_phase_pools,
                    team_all_pools=team_all_pools,
                    all_pool=all_pool,
                    selected_over=selected_over,
                    selected_bowler_pool=selected_pool,
                )

                first_scores.append(first["runs"])

                if second_batting_team and second_bowling_team:
                    second, outcome = _simulate_chase(
                        batting_team=second_batting_team,
                        bowling_team=second_bowling_team,
                        target=first["runs"] + 1,
                        rng=rng,
                        team_phase_pools=team_phase_pools,
                        team_all_pools=team_all_pools,
                        all_pool=all_pool,
                    )

                    second_scores.append(second["runs"])

                    if outcome == "batting_win":
                        # The second-innings batting team wins.
                        batting_wins += 1
                    elif outcome == "bowling_win":
                        bowling_wins += 1
                    else:
                        ties += 1
                else:
                    ties += 1

            else:
                second = _simulate_innings(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    start_over=selected_over,
                    start_runs=start_runs,
                    start_wickets=start_wickets,
                    start_legal_balls=start_legal,
                    rng=rng,
                    team_phase_pools=team_phase_pools,
                    team_all_pools=team_all_pools,
                    all_pool=all_pool,
                    target=chase_target,
                    selected_over=selected_over,
                    selected_bowler_pool=selected_pool,
                )

                first_score = (
                    int(first_innings["total_runs"].sum())
                    if not first_innings.empty
                    else 0
                )

                first_scores.append(first_score)
                second_scores.append(second["runs"])

                if (
                    chase_target is not None
                    and second["runs"] >= chase_target
                ):
                    batting_wins += 1
                elif (
                    chase_target is not None
                    and second["runs"] == chase_target - 1
                ):
                    ties += 1
                else:
                    bowling_wins += 1

        results[scenario] = {
            "scenario": scenario,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "selected_bowler": (
                original_bowler
                if scenario == "original"
                else str(replacement_bowler)
            ),
            "simulations": int(simulations),
            "batting_win_pct": batting_wins / simulations * 100,
            "bowling_win_pct": bowling_wins / simulations * 100,
            "tie_pct": ties / simulations * 100,
            "mean_first_innings_runs": (
                float(np.mean(first_scores)) if first_scores else 0.0
            ),
            "mean_second_innings_runs": (
                float(np.mean(second_scores)) if second_scores else 0.0
            ),
        }

    results["metadata"] = {
        "match_id": str(match_id),
        "innings": innings_no,
        "over": selected_over + 1,
        "phase": selected_phase,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "original_bowler": original_bowler,
        "replacement_bowler": str(replacement_bowler),
        "start_runs": start_runs,
        "start_wickets": start_wickets,
        "target": chase_target,
    }

    return results