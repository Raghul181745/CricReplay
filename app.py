from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.match_impact_engine import (
    simulate_match_impact,
    classify_phase as impact_classify_phase,
)

# ============================================================
# CRICREPLAY - CRICKET WHAT-IF LAB
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data" / "processed"

BALL_FILE = DATA_DIR / "ball_by_ball.csv"
OVER_FILE = DATA_DIR / "over_summary.csv"

RECENT_SEASONS = {"2022", "2023", "2024", "2025", "2026"}
SIMULATIONS = 5000
RANDOM_SEED = 42
SMOOTHING_K = 80

PHASES = ["Powerplay", "Middle", "Death"]

st.set_page_config(
    page_title="CricReplay | Cricket What-If Lab",
    page_icon="🏏",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_phase(over_number):
    if 0 <= int(over_number) <= 5:
        return "Powerplay"
    if 6 <= int(over_number) <= 14:
        return "Middle"
    if 15 <= int(over_number) <= 19:
        return "Death"
    return None


@st.cache_data(show_spinner="Loading IPL data...")
def load_balls():
    if not BALL_FILE.exists():
        raise FileNotFoundError(f"Cannot find {BALL_FILE}")

    df = pd.read_csv(
        BALL_FILE,
        low_memory=False,
        dtype={"match_id": str, "season": str},
    )

    required = [
        "match_id", "date", "season", "venue", "innings",
        "batting_team", "bowling_team", "over",
        "delivery_number", "batter", "bowler",
        "batter_runs", "total_runs", "extras", "legal_ball",
        "wickets", "wicket_types", "bowler_runs_conceded",
        "bowler_wickets",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns in ball_by_ball.csv: "
            + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "innings", "over", "delivery_number",
        "batter_runs", "total_runs", "extras",
        "wickets", "bowler_runs_conceded",
        "bowler_wickets",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).fillna(0)

    df["match_id"] = df["match_id"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip()

    df["legal_ball"] = (
        df["legal_ball"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
        .astype(bool)
    )

    df["over"] = df["over"].astype(int)
    df["innings"] = df["innings"].astype(int)
    df["date"] = df["date"].astype(str)
    df["phase"] = df["over"].apply(classify_phase)

    return df


@st.cache_data(show_spinner="Preparing historical over statistics...")
def load_bowler_over_data():
    if not OVER_FILE.exists():
        raise FileNotFoundError(f"Cannot find {OVER_FILE}")

    balls = load_balls()

    overs = pd.read_csv(
        OVER_FILE,
        low_memory=False,
        dtype={"match_id": str},
    )

    overs["match_id"] = overs["match_id"].astype(str).str.strip()

    season_map = balls[["match_id", "season"]].drop_duplicates()

    if season_map["match_id"].duplicated().any():
        raise ValueError(
            "A match ID maps to multiple seasons. "
            "Please inspect the source data."
        )

    data = overs.merge(
        season_map,
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    data = data[data["season"].isin(RECENT_SEASONS)].copy()

    for column in [
        "over", "legal_balls", "runs_conceded", "wickets"
    ]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data[data["legal_balls"] == 6].copy()
    data["phase"] = data["over"].apply(classify_phase)

    data = data.dropna(
        subset=[
            "match_id", "season", "phase", "bowler",
            "bowling_team", "runs_conceded", "wickets",
        ]
    )

    data["runs_conceded"] = data["runs_conceded"].astype(float)
    data["wickets"] = data["wickets"].astype(float)

    return data


def prepare_bowler_stats(data, phase, minimum_overs):
    phase_data = data[data["phase"] == phase].copy()

    if phase_data.empty:
        return phase_data, pd.DataFrame()

    team_stats = (
        phase_data.groupby("bowling_team")
        .agg(team_mean_runs=("runs_conceded", "mean"))
    )

    eligible = (
        phase_data.groupby(["bowler", "bowling_team"])
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

    if eligible.empty:
        return phase_data, eligible

    eligible["team_mean_runs"] = eligible["bowling_team"].map(
        team_stats["team_mean_runs"]
    )

    n = eligible["historical_overs"].astype(float)

    eligible["smoothed_expected_runs"] = (
        n / (n + SMOOTHING_K)
        * eligible["historical_mean_runs"]
        + SMOOTHING_K / (n + SMOOTHING_K)
        * eligible["team_mean_runs"]
    )

    return (
        phase_data,
        eligible.sort_values(
            ["bowler", "bowling_team"]
        ).reset_index(drop=True),
    )


def simulate_spells(over_records, number_of_overs, rng):
    """Existing complete-over comparison simulator."""

    if over_records.empty:
        raise ValueError("No historical overs are available.")

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

    return (
        run_values[choices].sum(axis=1),
        wicket_values[choices].sum(axis=1),
    )


def summarize_results(label, bowler, runs, wickets, number_of_overs):
    return {
        "Option": label,
        "Bowler": bowler["bowler"],
        "Team": bowler["bowling_team"],
        "Historical overs": int(bowler["historical_overs"]),
        "Historical matches": int(bowler["historical_matches"]),
        "Mean simulated runs": float(runs.mean()),
        "Median simulated runs": float(np.median(runs)),
        "Runs 10th percentile": float(np.percentile(runs, 10)),
        "Runs 90th percentile": float(np.percentile(runs, 90)),
        "Smoothed expected runs for spell": (
            float(bowler["smoothed_expected_runs"])
            * number_of_overs
        ),
        "Mean recorded wickets": float(wickets.mean()),
        "Median recorded wickets": float(np.median(wickets)),
    }


def bowler_label(row):
    return f"{row['bowler']} ({row['bowling_team']})"


def summarize_actual_match(match_balls):
    """Summarize observed innings and infer a result only when supported by the data.

    This is derived from ball-by-ball data, not official result metadata. It
    deliberately returns an unknown result for an apparently incomplete chase.
    Super-over and DLS results may require official match metadata.
    """
    if match_balls.empty:
        return [], "Result unavailable from the supplied ball-by-ball data."

    innings_rows = []
    for innings_no, innings_df in match_balls.groupby("innings", sort=True):
        if innings_df.empty:
            continue
        legal_balls = int(innings_df["legal_ball"].sum())
        runs = int(innings_df["total_runs"].sum())
        wickets = int(innings_df["wickets"].sum())
        innings_rows.append({
            "innings": int(innings_no),
            "team": str(innings_df["batting_team"].iloc[0]),
            "runs": runs,
            "wickets": wickets,
            "legal_balls": legal_balls,
            "overs_display": f"{legal_balls // 6}.{legal_balls % 6}",
        })

    if len(innings_rows) < 2:
        return innings_rows, "Winner unavailable — the data contains fewer than two innings."

    first, second = innings_rows[0], innings_rows[1]
    # A chase exceeding the first-innings total indicates that the chasing team won.
    if second["runs"] > first["runs"]:
        result = f"{second['team']} won (derived from the recorded innings scores)."
    elif second["runs"] == first["runs"]:
        completed = second["wickets"] >= 10 or second["legal_balls"] >= 120
        if completed:
            result = "The recorded innings scores are tied. A super-over result, if any, is not represented here."
        else:
            result = "Result uncertain — the innings scores are tied but the chase may be incomplete."
    else:
        second_innings_complete = second["wickets"] >= 10 or second["legal_balls"] >= 120
        if second_innings_complete:
            result = f"{first['team']} won (derived from the recorded innings scores)."
        else:
            result = ("Winner unavailable from ball-by-ball data alone: the second innings ended below "
                      "the first-innings score before 20 overs and 10 wickets. The match may be incomplete "
                      "or affected by DLS/other result rules.")

    return innings_rows, result


def show_actual_match_result(match_balls, key_prefix="actual_match"):
    """Render actual score summary without presenting simulated results as facts."""
    innings_rows, result_text = summarize_actual_match(match_balls)
    st.markdown("### 🏆 Actual match result (from recorded deliveries)")
    if len(innings_rows) >= 1:
        cols = st.columns(min(len(innings_rows), 4))
        for index, innings in enumerate(innings_rows):
            with cols[index % len(cols)]:
                st.metric(
                    f"Innings {innings['innings']} · {innings['team']}",
                    f"{innings['runs']}/{innings['wickets']}",
                    f"{innings['overs_display']} overs",
                )
    if " won " in result_text:
        st.success(result_text)
    else:
        st.info(result_text)
    st.caption(
        "Result is inferred from the available ball-by-ball innings. "
        "Official result metadata is needed to reliably handle DLS, abandoned matches, "
        "super overs and other exceptional outcomes."
    )


# ============================================================
# NEW: DELIVERY-LEVEL WHAT-IF SIMULATOR
# ============================================================

def simulate_delivery_level_over(
    history,
    n_simulations=SIMULATIONS,
    seed=RANDOM_SEED,
):
    """
    Simulate one over using historical deliveries from a replacement
    bowler. Extras do not count toward the six legal balls.
    """

    if history.empty:
        raise ValueError(
            "No historical deliveries are available for this bowler."
        )

    rng = np.random.default_rng(seed)

    total_runs = history["total_runs"].to_numpy(dtype=int)
    legal_balls = history["legal_ball"].to_numpy(dtype=bool)
    dismissals = history["wickets"].to_numpy(dtype=int)
    bowler_wickets = history["bowler_wickets"].to_numpy(dtype=int)

    simulated_runs = np.zeros(n_simulations, dtype=int)
    simulated_dismissals = np.zeros(n_simulations, dtype=int)
    simulated_bowler_wickets = np.zeros(n_simulations, dtype=int)
    simulated_legal_balls = np.zeros(n_simulations, dtype=int)

    for i in range(n_simulations):
        legal_count = 0
        deliveries = 0

        while (
            legal_count < 6
            and simulated_dismissals[i] < 10
            and deliveries < 30
        ):
            index = int(rng.integers(0, len(history)))

            simulated_runs[i] += total_runs[index]
            simulated_dismissals[i] += dismissals[index]
            simulated_bowler_wickets[i] += bowler_wickets[index]

            if legal_balls[index]:
                legal_count += 1

            deliveries += 1

        simulated_legal_balls[i] = legal_count

    return (
        simulated_runs,
        simulated_dismissals,
        simulated_bowler_wickets,
        simulated_legal_balls,
    )


# ============================================================
# HEADER AND NAVIGATION
# ============================================================

# ============================================================
# CRICREPLAY VISUAL DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --cric-bg: #07111b;
        --cric-panel: #101d2b;
        --cric-panel-soft: #14283a;
        --cric-lime: #c7f36b;
        --cric-orange: #ff8a4c;
        --cric-text: #f4f7fb;
        --cric-muted: #a8b8c8;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 82% 7%, rgba(38, 119, 96, .25), transparent 31%),
          radial-gradient(ellipse at 12% 22%, rgba(34, 94, 145, .20), transparent 35%),
          linear-gradient(145deg, #06101a 0%, #0a1825 54%, #0c201f 100%);
        color: var(--cric-text);
    }
    [data-testid="stHeader"] { background: rgba(5, 15, 24, .78); }
    [data-testid="stSidebar"] { background: #0b1824; border-right: 1px solid rgba(199,243,107,.13); }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #dce6ef; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px; }
    .cric-hero {
        position: relative; overflow: hidden;
        padding: 1.4rem 1.65rem 1.35rem; margin: .15rem 0 1.35rem;
        border: 1px solid rgba(199,243,107,.22); border-radius: 22px;
        background: linear-gradient(105deg, rgba(11,27,40,.98) 0%, rgba(12,47,52,.92) 58%, rgba(18,69,63,.75) 100%);
        box-shadow: 0 18px 48px rgba(0,0,0,.22);
    }
    .cric-hero:after {
        content: ""; position:absolute; right:-35px; top:-105px; width:380px; height:380px;
        border:1px solid rgba(199,243,107,.16); border-radius:50%;
        box-shadow: 0 0 0 28px rgba(199,243,107,.025), 0 0 0 58px rgba(199,243,107,.025), 0 0 0 88px rgba(199,243,107,.02);
        pointer-events:none;
    }
    .cric-eyebrow { color:var(--cric-lime); text-transform:uppercase; letter-spacing:.19em; font-size:.7rem; font-weight:850; margin-bottom:.55rem; }
    .cric-title { color:#f7fbff; font-size:clamp(2rem,4vw,3rem); line-height:1.04; font-weight:900; letter-spacing:-.055em; margin:0; }
    .cric-title span { color:var(--cric-lime); }
    .cric-subtitle { color:#bdcbd7; font-size:1rem; max-width:800px; line-height:1.55; margin:.65rem 0 0; }
    h1,h2,h3 { letter-spacing:-.03em; }
    h2 { margin-top:1.15rem !important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(18,36,52,.94),rgba(10,25,38,.94)); border:1px solid rgba(184,211,225,.13); padding:1rem 1.05rem; border-radius:16px; box-shadow:0 10px 28px rgba(0,0,0,.12); }
    [data-testid="stMetricLabel"] { color:#b9c8d6 !important; }
    [data-testid="stMetricValue"] { color:#f7fbff !important; font-weight:850; }
    [data-testid="stMetricDelta"] { font-weight:700; }
    [data-testid="stBaseButton-primary"] { background:linear-gradient(100deg,#c7f36b,#a8df4c)!important; color:#122017!important; border:0!important; font-weight:850!important; border-radius:11px!important; min-height:2.75rem; box-shadow:0 7px 20px rgba(168,223,76,.13); }
    [data-testid="stBaseButton-primary"]:hover { filter:brightness(1.06); transform:translateY(-1px); }
    [data-testid="stBaseButton-secondary"] { border-color:rgba(199,243,107,.25)!important; border-radius:11px!important; }
    [data-testid="stSelectbox"] > div > div,[data-testid="stMultiSelect"] > div > div,[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input { border-radius:10px!important; }
    [data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid rgba(184,211,225,.14); border-radius:12px; overflow:hidden; }
    [data-testid="stExpander"] { border:1px solid rgba(184,211,225,.15); border-radius:12px; background:rgba(16,31,47,.48); }
    [data-testid="stAlert"] { border-radius:12px; }
    hr { border-color:rgba(184,211,225,.14)!important; }
    .cric-section-kicker { color:var(--cric-lime); text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:850; }
    .cric-dashboard-hero { position:relative; overflow:hidden; padding:2rem 2rem 1.8rem; border-radius:24px; border:1px solid rgba(199,243,107,.19); background:linear-gradient(112deg,rgba(12,28,41,.97),rgba(12,49,50,.91) 63%,rgba(20,76,59,.78)); box-shadow:0 18px 50px rgba(0,0,0,.19); margin:.5rem 0 1.5rem; }
    .cric-dashboard-hero:before { content:""; position:absolute; width:350px;height:350px;right:-75px;bottom:-205px;border:1px solid rgba(199,243,107,.25);border-radius:50%;box-shadow:0 0 0 25px rgba(199,243,107,.035),0 0 0 55px rgba(199,243,107,.03),0 0 0 85px rgba(199,243,107,.025); }
    .cric-hero-tag { display:inline-block; padding:.36rem .7rem; border:1px solid rgba(199,243,107,.24); border-radius:999px; color:var(--cric-lime); background:rgba(199,243,107,.07); font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
    .cric-dashboard-title { font-size:clamp(2.25rem,5vw,3.8rem); line-height:1.03; font-weight:900; letter-spacing:-.06em; margin:.95rem 0 .65rem; color:#f7fbff; }
    .cric-dashboard-title span { color:var(--cric-lime); }
    .cric-dashboard-copy { max-width:720px; font-size:1.03rem; line-height:1.65; color:#c1d0dc; }
    .cric-dashboard-foot { color:#a9c0c9; font-size:.78rem; letter-spacing:.04em; margin-top:1rem; }
    .cric-card { min-height:218px; height:100%; padding:1.3rem; border-radius:19px; border:1px solid rgba(184,211,225,.15); background:linear-gradient(150deg,rgba(20,39,55,.97),rgba(9,24,36,.98)); box-shadow:0 14px 34px rgba(0,0,0,.17); transition:transform .18s ease,border-color .18s ease; }
    .cric-card:hover { transform:translateY(-3px); border-color:rgba(199,243,107,.4); }
    .cric-card-icon { width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-size:1.55rem;margin-bottom:1rem;border-radius:14px;background:rgba(199,243,107,.09);border:1px solid rgba(199,243,107,.15); }
    .cric-card-title { color:#f7fbff; font-size:1.13rem; font-weight:850; margin-bottom:.5rem; }
    .cric-card-copy { color:#afc0ce; font-size:.9rem; line-height:1.58; min-height:72px; }
    .cric-card-meta { color:var(--cric-lime); text-transform:uppercase; font-size:.68rem; font-weight:850; letter-spacing:.12em; margin-top:.9rem; }
    .cric-section-head { display:flex; align-items:center; gap:.7rem; margin:.5rem 0 .35rem; }
    .cric-step { background:rgba(18,37,52,.75); border:1px solid rgba(184,211,225,.13); border-radius:15px; padding:1rem; min-height:110px; }
    .cric-step-num { color:var(--cric-lime); font-size:.72rem; letter-spacing:.12em; font-weight:900; }
    .cric-step-title { color:#f4f7fb; font-weight:800; margin:.4rem 0 .25rem; }
    .cric-step-copy { color:#aebfcd; font-size:.85rem; line-height:1.45; }
    @media(max-width:700px){.cric-hero{padding:1.05rem;border-radius:16px}.cric-dashboard-hero{padding:1.3rem;border-radius:18px}[data-testid="stMetric"]{padding:.75rem}.cric-card{min-height:unset}}
    </style>
    <div class="cric-hero">
      <div class="cric-eyebrow">IPL analytics · Decision intelligence</div>
      <div class="cric-title">🏏 Cric<span>Replay</span></div>
      <div class="cric-subtitle">A cricket analyst's lab for historical performance, over-level experiments and match-impact scenarios.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dashboard-first navigation: tools are selected from the main landing page.
st.session_state.setdefault("cricreplay_page", "Dashboard")
page = st.session_state["cricreplay_page"]

try:
    balls = load_balls()
except Exception as exc:
    st.error(f"Could not load ball-by-ball data: {exc}")
    st.info(
        "Check that app.py is in your Cric_Replay folder and "
        "Data/processed/ball_by_ball.csv exists."
    )
    st.stop()


# ============================================================
# DASHBOARD + PAGE ROUTING
# ============================================================

if page != "Dashboard":
    nav_left, nav_title, nav_right = st.columns([1.2, 3.8, 1.2])
    with nav_left:
        if st.button("← Dashboard", use_container_width=True, key="back_to_dashboard"):
            st.session_state["cricreplay_page"] = "Dashboard"
            st.rerun()
    with nav_title:
        st.caption("CRICREPLAY WORKSPACE  /  " + page.upper())
    with nav_right:
        if st.button("Home ⌂", use_container_width=True, key="home_shortcut"):
            st.session_state["cricreplay_page"] = "Dashboard"
            st.rerun()

if page == "Dashboard":
    st.markdown(
        """<div class="cric-dashboard-hero">
          <span class="cric-hero-tag">IPL DATA · ANALYST WORKSPACE</span>
          <div class="cric-dashboard-title">Read the game.<br>Test the <span>decision.</span></div>
          <div class="cric-dashboard-copy">Explore historical IPL bowling performance, replay a key over, and model how a different bowling choice could influence the rest of a match.</div>
          <div class="cric-dashboard-foot">HISTORICAL DATA &nbsp; • &nbsp; 5,000-RUN SCENARIOS &nbsp; • &nbsp; TRANSPARENT MODEL LIMITATIONS</div>
        </div>""",
        unsafe_allow_html=True,
    )
    try:
        dashboard_data = load_bowler_over_data()
        match_count = int(dashboard_data["match_id"].nunique())
        over_count = int(len(dashboard_data))
        team_count = int(dashboard_data["bowling_team"].nunique())
        season_count = int(dashboard_data["season"].nunique())
    except Exception:
        match_count = int(balls["match_id"].nunique())
        over_count = 0
        team_count = int(balls["bowling_team"].nunique())
        season_count = int(balls["season"].nunique())

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("IPL matches in dataset", f"{match_count:,}")
    k2.metric("Complete overs analysed", f"{over_count:,}" if over_count else "—")
    k3.metric("Bowling teams", f"{team_count:,}")
    k4.metric("Seasons available", f"{season_count:,}")

    st.markdown("<div class='cric-section-kicker'>Choose your workspace</div>", unsafe_allow_html=True)
    st.markdown("## Three ways to investigate the game")
    st.caption("Choose a workspace below. Your dashboard stays one click away.")
    card1, card2, card3 = st.columns(3, gap="large")
    with card1:
        st.markdown('<div class="cric-card"><div class="cric-card-icon">📊</div><div class="cric-card-title">Bowler Comparison</div><div class="cric-card-copy">Benchmark bowlers by phase, compare economy and wicket performance, and explore historical sample sizes.</div><div class="cric-card-meta">01 / Performance analytics</div></div>', unsafe_allow_html=True)
        if st.button("Open bowler analytics  ↗", use_container_width=True, key="dash_bowler"):
            st.session_state["cricreplay_page"] = "Bowler Comparison"
            st.rerun()
    with card2:
        st.markdown('<div class="cric-card"><div class="cric-card-icon">🎬</div><div class="cric-card-title">Match Replay & What-If</div><div class="cric-card-copy">Inspect a real over, review its deliveries and compare it with a simulated replacement-bowler over.</div><div class="cric-card-meta">02 / Over-level lab</div></div>', unsafe_allow_html=True)
        if st.button("Open match replay  ↗", use_container_width=True, key="dash_replay"):
            st.session_state["cricreplay_page"] = "Match Replay & What-If"
            st.rerun()
    with card3:
        st.markdown('<div class="cric-card"><div class="cric-card-icon">🏆</div><div class="cric-card-title">Match Impact</div><div class="cric-card-copy">Compare original and replacement bowling scenarios across 5,000 experimental full-match simulations.</div><div class="cric-card-meta">03 / Decision impact</div></div>', unsafe_allow_html=True)
        if st.button("Open match impact  ↗", use_container_width=True, key="dash_impact"):
            st.session_state["cricreplay_page"] = "Match Impact"
            st.rerun()

    st.markdown("<div class='cric-section-kicker'>Analyst workflow</div>", unsafe_allow_html=True)
    st.markdown("## From match data to a what-if decision")
    step1, step2, step3 = st.columns(3, gap="medium")
    with step1:
        st.markdown('<div class="cric-step"><div class="cric-step-num">STEP 01</div><div class="cric-step-title">Select the context</div><div class="cric-step-copy">Choose a match, innings, phase or bowler from historical IPL data.</div></div>', unsafe_allow_html=True)
    with step2:
        st.markdown('<div class="cric-step"><div class="cric-step-num">STEP 02</div><div class="cric-step-title">Change one decision</div><div class="cric-step-copy">Replace the bowler and run a reproducible historical resampling scenario.</div></div>', unsafe_allow_html=True)
    with step3:
        st.markdown('<div class="cric-step"><div class="cric-step-num">STEP 03</div><div class="cric-step-title">Compare the outcomes</div><div class="cric-step-copy">Review score and win-rate estimates with the model&#39;s limitations in view.</div></div>', unsafe_allow_html=True)
    st.info("Analyst note: simulations are exploratory, not calibrated predictions. Batter matchups, venue, pressure, strategy, DLS and every cricket rule are not fully modelled.")

# ============================================================
# PAGE 1: BOWLER COMPARISON
# ============================================================

elif page == "Bowler Comparison":

    try:
        data = load_bowler_over_data()
    except Exception as exc:
        st.error(f"Could not load bowling statistics: {exc}")
        st.stop()

    st.markdown("## 📊 Historical Data Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Complete overs", f"{len(data):,}")
    c2.metric("Matches", f"{data['match_id'].nunique():,}")
    c3.metric("Bowling teams", f"{data['bowling_team'].nunique():,}")
    c4.metric("Seasons", f"{data['season'].nunique():,}")

    with st.expander("Explore dataset by season"):
        summary = (
            data.groupby("season")
            .agg(
                Matches=("match_id", "nunique"),
                Complete_overs=("runs_conceded", "size"),
                Runs_conceded=("runs_conceded", "sum"),
                Recorded_wickets=("wickets", "sum"),
            )
            .reset_index()
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("⚙️ Bowler comparison settings", expanded=True):
        settings_a, settings_b, settings_c = st.columns(3)
        phase = settings_a.selectbox("Bowling phase", PHASES, index=2)
        n_overs = settings_b.slider("Spell length (overs)", 1, 5, 2)
        minimum_overs = settings_c.slider("Minimum historical overs per bowler", 10, 100, 20, step=5)
        st.caption(f"{SIMULATIONS:,} simulations per bowler | Smoothing k={SMOOTHING_K}")

    phase_data, eligible = prepare_bowler_stats(
        data, phase, minimum_overs
    )

    st.markdown(f"## 🎯 {phase} Bowling Analysis")

    if eligible.empty:
        st.warning(
            "No bowlers meet this threshold. Reduce minimum "
            "historical overs in the sidebar."
        )
        st.stop()

    display = eligible[
        [
            "bowler", "bowling_team", "historical_overs",
            "historical_matches", "historical_mean_runs",
            "smoothed_expected_runs",
        ]
    ].rename(
        columns={
            "bowler": "Bowler",
            "bowling_team": "Team",
            "historical_overs": "Historical overs",
            "historical_matches": "Matches",
            "historical_mean_runs": "Historical runs/over",
            "smoothed_expected_runs": "Smoothed runs/over",
        }
    )

    st.dataframe(
        display.style.format({
            "Historical runs/over": "{:.2f}",
            "Smoothed runs/over": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("## 🆚 Compare Two Bowlers")

    labels = [
        bowler_label(row)
        for _, row in eligible.iterrows()
    ]

    label_index = {
        label: i for i, label in enumerate(labels)
    }

    ca, cb = st.columns(2)

    with ca:
        selected_a = st.selectbox(
            "Choose Bowler A", labels, key="bowler_a"
        )

    row_a = eligible.iloc[label_index[selected_a]]

    eligible_b = [
        label for label in labels
        if eligible.iloc[label_index[label]]["bowler"].casefold()
        != row_a["bowler"].casefold()
    ]

    if not eligible_b:
        st.warning("At least two different bowlers are needed.")
        st.stop()

    with cb:
        selected_b = st.selectbox(
            "Choose Bowler B", eligible_b, key="bowler_b"
        )

    bowler_a = eligible.iloc[label_index[selected_a]]
    bowler_b = eligible.iloc[label_index[selected_b]]

    ca, cb = st.columns(2)

    for col, row in [(ca, bowler_a), (cb, bowler_b)]:
        with col:
            st.markdown(f"### {row['bowler']}")
            st.caption(row["bowling_team"])

            st.metric(
                "Historical runs/over",
                f"{row['historical_mean_runs']:.2f}",
            )

            st.metric(
                "Smoothed runs/over",
                f"{row['smoothed_expected_runs']:.2f}",
            )

            st.write(
                f"Historical overs: {int(row['historical_overs'])}"
            )
            st.write(
                f"Historical matches: {int(row['historical_matches'])}"
            )

    if st.button(
        "▶ Run Comparison Simulation",
        type="primary",
        use_container_width=True,
    ):
        records_a = phase_data[
            (phase_data["bowler"] == bowler_a["bowler"])
            & (phase_data["bowling_team"] == bowler_a["bowling_team"])
        ].reset_index(drop=True)

        records_b = phase_data[
            (phase_data["bowler"] == bowler_b["bowler"])
            & (phase_data["bowling_team"] == bowler_b["bowling_team"])
        ].reset_index(drop=True)

        seed_a, seed_b = np.random.SeedSequence(
            RANDOM_SEED
        ).spawn(2)

        runs_a, wickets_a = simulate_spells(
            records_a, n_overs, np.random.default_rng(seed_a)
        )

        runs_b, wickets_b = simulate_spells(
            records_b, n_overs, np.random.default_rng(seed_b)
        )

        results = pd.DataFrame([
            summarize_results("A", bowler_a, runs_a, wickets_a, n_overs),
            summarize_results("B", bowler_b, runs_b, wickets_b, n_overs),
        ])

        st.session_state["comparison_results"] = {
            "results": results,
            "runs_a": runs_a,
            "runs_b": runs_b,
            "wickets_a": wickets_a,
            "wickets_b": wickets_b,
            "a": selected_a,
            "b": selected_b,
            "phase": phase,
            "overs": n_overs,
        }

    saved = st.session_state.get("comparison_results")

    if (
        saved
        and saved["a"] == selected_a
        and saved["b"] == selected_b
        and saved["phase"] == phase
        and saved["overs"] == n_overs
    ):
        results = saved["results"]
        runs_a = saved["runs_a"]
        runs_b = saved["runs_b"]
        wickets_a = saved["wickets_a"]
        wickets_b = saved["wickets_b"]

        st.markdown("### 📈 Simulation Results")

        st.dataframe(
            results.round(2),
            use_container_width=True,
            hide_index=True,
        )

        hist = pd.DataFrame({
            "Runs conceded": np.concatenate([runs_a, runs_b]),
            "Option": (
                [f"A: {bowler_a['bowler']}"] * len(runs_a)
                + [f"B: {bowler_b['bowler']}"] * len(runs_b)
            ),
        })

        st.bar_chart(
            hist.groupby(["Runs conceded", "Option"])
            .size()
            .unstack(fill_value=0),
            use_container_width=True,
        )

        x, y, z = st.columns(3)
        x.metric(
            "A conceded fewer runs",
            f"{np.mean(runs_a < runs_b) * 100:.1f}%",
        )
        y.metric(
            "B conceded fewer runs",
            f"{np.mean(runs_b < runs_a) * 100:.1f}%",
        )
        z.metric(
            "Equal runs",
            f"{np.mean(runs_a == runs_b) * 100:.1f}%",
        )

        x, y, z = st.columns(3)
        x.metric(
            "A recorded more wickets",
            f"{np.mean(wickets_a > wickets_b) * 100:.1f}%",
        )
        y.metric(
            "B recorded more wickets",
            f"{np.mean(wickets_b > wickets_a) * 100:.1f}%",
        )
        z.metric(
            "Equal wickets",
            f"{np.mean(wickets_a == wickets_b) * 100:.1f}%",
        )

        st.download_button(
            "⬇ Download Results as CSV",
            results.to_csv(index=False).encode("utf-8"),
            "cricreplay_what_if_comparison.csv",
            "text/csv",
            use_container_width=True,
        )

        st.info(
            "These percentages describe the simulation setup, "
            "not real-world probabilities."
        )

    with st.expander("📘 Methodology and limitations"):
        st.markdown("""
        - Uses historical complete overs from IPL seasons 2022–2026.
        - Powerplay: overs 0–5; Middle: 6–14; Death: 15–19.
        - Historical averages are smoothed using k=80.
        - Complete overs are resampled with replacement.
        - Batter, venue, pitch, opponent and match pressure are not modelled.
        - Results are not calibrated predictions of future performance.
        """)


# ============================================================
# PAGE 2: MATCH REPLAY AND WHAT-IF
# ============================================================

elif page == "Match Replay & What-If":
    st.markdown("## 🎬 Match Replay & What-If Lab")

    st.write(
        "Select a real IPL match, inspect an over, and test how "
        "a replacement bowler might perform using historical deliveries."
    )

    match_info = (
        balls.groupby("match_id", as_index=False)
        .agg(
            Date=("date", "first"),
            Season=("season", "first"),
            Venue=("venue", "first"),
        )
    )

    team_rows = (
        balls[["match_id", "innings", "batting_team"]]
        .drop_duplicates()
        .sort_values(["match_id", "innings"])
    )

    teams_map = (
        team_rows.groupby("match_id")["batting_team"]
        .apply(lambda s: list(dict.fromkeys(s.tolist())))
        .to_dict()
    )

    match_info["Teams"] = match_info["match_id"].map(
        lambda mid: " vs ".join(teams_map.get(mid, []))
    )

    match_info = match_info.sort_values(
        ["Date", "Season", "match_id"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    match_labels = {
        r["match_id"]: (
            f"{r['Date']} | {r['Teams']} | "
            f"{r['Venue']} ({r['Season']})"
        )
        for _, r in match_info.iterrows()
    }

    selected_match_id = st.selectbox(
        "1. Choose an IPL match",
        match_info["match_id"].tolist(),
        format_func=lambda mid: match_labels.get(mid, mid),
        key="replay_match",
    )

    match = balls[
        balls["match_id"] == selected_match_id
    ].copy().sort_values(["innings", "over", "delivery_number"])

    show_actual_match_result(match, key_prefix="replay_actual")

    innings_options = sorted(match["innings"].unique().tolist())

    innings_teams = (
        match.groupby("innings")["batting_team"].first().to_dict()
    )

    selected_innings = st.selectbox(
        "2. Choose innings",
        innings_options,
        format_func=lambda i: (
            f"Innings {i} — {innings_teams.get(i, 'Unknown team')}"
        ),
        key="replay_innings",
    )

    innings_balls = match[
        match["innings"] == selected_innings
    ].copy().sort_values(["over", "delivery_number"])

    total_runs = int(innings_balls["total_runs"].sum())
    total_dismissals = int(innings_balls["wickets"].sum())
    legal_count = int(innings_balls["legal_ball"].sum())

    c1, c2, c3 = st.columns(3)

    c1.metric("Actual innings score", f"{total_runs}/{total_dismissals}")
    c2.metric(
        "Legal balls",
        f"{legal_count} ({legal_count // 6}.{legal_count % 6} overs)",
    )
    c3.metric("Venue", str(innings_balls["venue"].iloc[0]))

    # Over-by-over scorecard.
    over_summary = (
        innings_balls.groupby("over", as_index=False)
        .agg(
            Bowling_Team=("bowling_team", "first"),
            Runs=("total_runs", "sum"),
            Dismissals=("wickets", "sum"),
            Legal_Balls=("legal_ball", "sum"),
            Original_Bowler=(
                "bowler",
                lambda s: ", ".join(
                    dict.fromkeys(s.dropna().astype(str).tolist())
                ),
            ),
        )
        .sort_values("over")
    )

    over_summary["Over"] = over_summary["over"] + 1
    over_summary["Score after over"] = over_summary["Runs"].cumsum()
    over_summary["Wickets after over"] = over_summary["Dismissals"].cumsum()

    st.markdown("### Actual Over-by-Over Scorecard")

    st.dataframe(
        over_summary[
            [
                "Over", "Original_Bowler", "Runs", "Dismissals",
                "Score after over", "Wickets after over", "Legal_Balls",
            ]
        ].rename(columns={
            "Original_Bowler": "Original bowler(s)",
            "Runs": "Runs this over",
            "Dismissals": "Wickets this over",
            "Legal_Balls": "Legal balls this over",
        }),
        use_container_width=True,
        hide_index=True,
    )

    available_overs = sorted(innings_balls["over"].unique().tolist())

    selected_over = st.selectbox(
        "3. Choose an over to investigate",
        available_overs,
        format_func=lambda o: f"Over {o + 1}",
        key="replay_selected_over",
    )

    over_balls = innings_balls[
        innings_balls["over"] == selected_over
    ].copy().sort_values("delivery_number")

    original_bowler_names = list(
        dict.fromkeys(
            over_balls["bowler"].dropna().astype(str).tolist()
        )
    )

    original_runs = int(over_balls["total_runs"].sum())
    original_dismissals = int(over_balls["wickets"].sum())
    original_bowler_wickets = int(over_balls["bowler_wickets"].sum())
    original_legal = int(over_balls["legal_ball"].sum())

    phase = classify_phase(selected_over)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Actual over runs", original_runs)
    c2.metric("Dismissal events", original_dismissals)
    c3.metric("Bowler-credited wickets", original_bowler_wickets)
    c4.metric("Legal balls", original_legal)

    st.write(
        "**Original bowler(s):** "
        + (", ".join(original_bowler_names) or "Not recorded")
    )
    st.write(f"**Phase:** {phase or 'Outside standard T20 overs'}")

    delivery_view = over_balls[
        [
            "delivery_number", "batter", "bowler", "batter_runs",
            "extras", "total_runs", "legal_ball", "wickets",
            "wicket_types",
        ]
    ].rename(columns={
        "delivery_number": "Source delivery number",
        "batter": "Batter",
        "bowler": "Bowler",
        "batter_runs": "Batter runs",
        "extras": "Extras",
        "total_runs": "Total runs",
        "legal_ball": "Legal delivery",
        "wickets": "Dismissal events",
        "wicket_types": "Dismissal type(s)",
    })

    with st.expander("View deliveries in the selected over"):
        st.dataframe(
            delivery_view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download selected over CSV",
            delivery_view.to_csv(index=False).encode("utf-8"),
            f"cricreplay_match_{selected_match_id}_"
            f"innings_{selected_innings}_over_{selected_over + 1}.csv",
            "text/csv",
        )

    # --------------------------------------------------------
    # ALTERNATIVE BOWLER SELECTOR
    # --------------------------------------------------------

    st.divider()

    st.markdown("## 🔄 What if a different bowler had bowled this over?")

    st.write(
        "Choose a replacement bowler. The app will sample that "
        "bowler's historical deliveries from the same phase and "
        "simulate 5,000 alternative overs."
    )

    bowling_team = str(over_balls["bowling_team"].iloc[0])
    season = str(over_balls["season"].iloc[0])

    phase_history = balls[
        (balls["bowling_team"] == bowling_team)
        & (balls["phase"] == phase)
    ].copy()

    # Prefer the same team and season when enough bowlers are available.
    season_history = phase_history[
        phase_history["season"] == season
    ]

    if season_history["bowler"].nunique() >= 2:
        candidate_history = season_history
        history_scope = f"{season} season"
    else:
        candidate_history = phase_history
        history_scope = "all available seasons for this bowling team"

    delivery_counts = (
        candidate_history.groupby("bowler")
        .size()
        .sort_values(ascending=False)
    )

    candidate_names = [
        str(name)
        for name, count in delivery_counts.items()
        if count >= 12 and str(name) not in original_bowler_names
    ]

    if not phase or phase_history.empty:
        st.warning(
            "No historical deliveries are available for this team's "
            "selected bowling phase."
        )

    elif not candidate_names:
        st.warning(
            "No replacement bowler has enough historical deliveries "
            "for this team and phase. Try another match or over."
        )

    else:
        selected_replacement = st.selectbox(
            "4. Choose replacement bowler",
            candidate_names,
            key="replacement_bowler",
        )

        replacement_history = candidate_history[
            candidate_history["bowler"].astype(str)
            == selected_replacement
        ].copy()

        st.caption(
            f"Historical sample: {len(replacement_history):,} deliveries "
            f"from {history_scope}. Minimum sample: 12 deliveries."
        )

        if st.button(
            "▶ Run 5,000 What-If Simulations",
            type="primary",
            use_container_width=True,
            key="run_over_what_if",
        ):
            try:
                (
                    sim_runs,
                    sim_dismissals,
                    sim_bowler_wickets,
                    sim_legal_balls,
                ) = simulate_delivery_level_over(replacement_history)

                st.session_state["over_what_if"] = {
                    "match_id": selected_match_id,
                    "innings": int(selected_innings),
                    "over": int(selected_over),
                    "replacement": selected_replacement,
                    "runs": sim_runs,
                    "dismissals": sim_dismissals,
                    "bowler_wickets": sim_bowler_wickets,
                    "legal_balls": sim_legal_balls,
                    "actual_runs": original_runs,
                    "actual_dismissals": original_dismissals,
                    "actual_bowler_wickets": original_bowler_wickets,
                    "original_bowlers": original_bowler_names,
                    "phase": phase,
                }

            except Exception as exc:
                st.error(f"Simulation failed: {exc}")

        result = st.session_state.get("over_what_if")

        if (
            result
            and result["match_id"] == selected_match_id
            and result["innings"] == int(selected_innings)
            and result["over"] == int(selected_over)
            and result["replacement"] == selected_replacement
        ):
            sim_runs = result["runs"]
            sim_dismissals = result["dismissals"]
            sim_bowler_wickets = result["bowler_wickets"]

            st.markdown("### 📊 Actual Over vs Simulated Replacement")

            st.write(
                f"**Actual:** {result['actual_runs']} runs, "
                f"{result['actual_dismissals']} dismissal event(s), "
                f"bowled by "
                f"{', '.join(result['original_bowlers']) or 'unknown'}."
            )

            a, b, c, d = st.columns(4)

            a.metric(
                "Replacement mean runs",
                f"{sim_runs.mean():.2f}",
                delta=(
                    f"{sim_runs.mean() - result['actual_runs']:+.2f} "
                    "vs actual"
                ),
                delta_color="inverse",
            )

            b.metric(
                "Median simulated runs",
                f"{np.median(sim_runs):.0f}",
            )

            c.metric(
                "Mean simulated dismissals",
                f"{sim_dismissals.mean():.2f}",
            )

            d.metric(
                "Mean bowler-credited wickets",
                f"{sim_bowler_wickets.mean():.2f}",
            )

            a, b, c = st.columns(3)

            a.metric(
                "Simulations below actual runs",
                f"{np.mean(sim_runs < result['actual_runs']) * 100:.1f}%",
            )

            b.metric(
                "Simulations equal actual runs",
                f"{np.mean(sim_runs == result['actual_runs']) * 100:.1f}%",
            )

            c.metric(
                "Simulations above actual runs",
                f"{np.mean(sim_runs > result['actual_runs']) * 100:.1f}%",
            )

            st.markdown("#### Simulated Run Distribution")

            distribution = pd.Series(
                sim_runs,
                name="Simulated runs",
            )

            st.bar_chart(
                distribution.value_counts().sort_index(),
                use_container_width=True,
            )

            st.write(
                f"**10th–90th percentile:** "
                f"{np.percentile(sim_runs, 10):.0f}–"
                f"{np.percentile(sim_runs, 90):.0f} runs"
            )

            export = pd.DataFrame({
                "simulated_runs": sim_runs,
                "dismissal_events": sim_dismissals,
                "bowler_credited_wickets": sim_bowler_wickets,
                "legal_balls": result["legal_balls"],
            })

            st.download_button(
                "⬇ Download What-If Simulation Results",
                export.to_csv(index=False).encode("utf-8"),
                f"cricreplay_what_if_{selected_match_id}_"
                f"over_{selected_over + 1}.csv",
                "text/csv",
                use_container_width=True,
            )

            st.warning(
                "These are experimental historical simulations, not "
                "predictions of what definitely would have happened. "
                "The model does not yet account for the exact batter "
                "matchup, venue, pressure or remaining match situation. "
                "It does not calculate whether the match winner changes."
            )

    st.divider()

    st.markdown("### Next development step")

    st.write(
        "After verifying this over-level simulator, we can extend it "
        "to the remaining innings and estimate how often alternative "
        "bowling decisions change the simulated match result."
    )



# ============================================================
# PAGE 3: FULL MATCH IMPACT
# ============================================================

else:
    st.markdown("## 🏏 Match Impact")
    st.write(
        "Compare the original bowler with a replacement and estimate "
        "how the decision may affect the rest of the innings and match."
    )

    st.caption(
        "Each scenario uses 5,000 historical-delivery simulations. "
        "These are exploratory estimates, not calibrated win probabilities."
    )

    match_info = (
        balls.groupby("match_id", as_index=False)
        .agg(
            Date=("date", "first"),
            Season=("season", "first"),
            Venue=("venue", "first"),
        )
        .sort_values(["Date", "Season", "match_id"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    team_map = (
        balls[["match_id", "batting_team"]]
        .drop_duplicates()
        .groupby("match_id")["batting_team"]
        .apply(lambda values: list(dict.fromkeys(values.tolist())))
        .to_dict()
    )

    match_info["Teams"] = match_info["match_id"].map(
        lambda match_id: " vs ".join(team_map.get(match_id, []))
    )

    match_labels = {
        row["match_id"]: (
            f"{row['Date']} | {row['Teams']} | {row['Venue']} ({row['Season']})"
        )
        for _, row in match_info.iterrows()
    }

    selected_impact_match = st.selectbox(
        "1. Choose an IPL match",
        options=match_info["match_id"].astype(str).tolist(),
        format_func=lambda match_id: match_labels.get(match_id, match_id),
        key="impact_match_id",
    )

    selected_impact_match_balls = balls[
        balls["match_id"].astype(str) == str(selected_impact_match)
    ].copy()

    show_actual_match_result(selected_impact_match_balls, key_prefix="impact_actual")

    innings_options = sorted(
        selected_impact_match_balls["innings"].dropna().astype(int).unique().tolist()
    )

    selected_impact_innings = st.selectbox(
        "2. Choose innings",
        options=innings_options,
        format_func=lambda innings_no: (
            f"Innings {innings_no} — "
            + str(
                selected_impact_match_balls.loc[
                    selected_impact_match_balls["innings"] == innings_no,
                    "batting_team",
                ].iloc[0]
            )
        ),
        key="impact_innings",
    )

    impact_innings_balls = selected_impact_match_balls[
        selected_impact_match_balls["innings"] == selected_impact_innings
    ].copy()

    impact_over_options = sorted(
        impact_innings_balls["over"].astype(int).unique().tolist()
    )

    selected_impact_over = st.selectbox(
        "3. Choose the over where the bowling decision changes",
        options=impact_over_options,
        format_func=lambda over_no: f"Over {over_no + 1}",
        key="impact_over",
    )

    selected_impact_over_balls = impact_innings_balls[
        impact_innings_balls["over"] == selected_impact_over
    ].copy()

    impact_original_mode = selected_impact_over_balls["bowler"].dropna().astype(str).mode()
    if impact_original_mode.empty:
        st.error("The selected over has no recorded bowler.")
        st.stop()

    impact_original_bowler = str(impact_original_mode.iloc[0])
    impact_bowling_team = str(selected_impact_over_balls["bowling_team"].iloc[0])
    impact_phase = impact_classify_phase(selected_impact_over)

    if impact_phase is None:
        st.error("The selected over is outside the standard 20-over innings.")
        st.stop()

    st.write(f"**Original bowler:** {impact_original_bowler}")
    st.write(f"**Bowling team:** {impact_bowling_team}")
    st.write(f"**Phase:** {impact_phase}")

    impact_candidates = balls[
        (balls["bowling_team"].astype(str) == impact_bowling_team)
        & (balls["phase"] == impact_phase)
        & (balls["bowler"].notna())
    ].copy()

    candidate_counts = impact_candidates.groupby("bowler").size().sort_values(ascending=False)
    impact_candidate_names = [
        str(name)
        for name, count in candidate_counts.items()
        if int(count) >= 12 and str(name) != impact_original_bowler
    ]

    if not impact_candidate_names:
        st.warning(
            "No alternative bowler has at least 12 historical deliveries "
            "for this team and phase. Choose another match or over."
        )
        st.stop()

    selected_impact_replacement = st.selectbox(
        "4. Choose replacement bowler",
        options=impact_candidate_names,
        key="impact_replacement_bowler",
    )

    if st.button(
        "🚀 Run 5,000 Full-Match Simulations",
        type="primary",
        use_container_width=True,
        key="run_full_match_impact",
    ):
        try:
            with st.spinner(
                "Running 5,000 simulations for each bowling scenario..."
            ):
                impact_results = simulate_match_impact(
                    balls=balls,
                    match_id=str(selected_impact_match),
                    innings_no=int(selected_impact_innings),
                    selected_over=int(selected_impact_over),
                    replacement_bowler=selected_impact_replacement,
                    simulations=SIMULATIONS,
                    seed=RANDOM_SEED,
                )

            st.session_state["full_match_impact_results"] = {
                "match_id": str(selected_impact_match),
                "innings": int(selected_impact_innings),
                "over": int(selected_impact_over),
                "replacement": selected_impact_replacement,
                "results": impact_results,
            }

        except Exception as exc:
            st.error(f"Full-match simulation failed: {exc}")

    saved_impact = st.session_state.get("full_match_impact_results")

    if (
        saved_impact
        and saved_impact["match_id"] == str(selected_impact_match)
        and saved_impact["innings"] == int(selected_impact_innings)
        and saved_impact["over"] == int(selected_impact_over)
        and saved_impact["replacement"] == selected_impact_replacement
    ):
        impact_results = saved_impact["results"]
        impact_metadata = impact_results["metadata"]
        original_result = impact_results["original"]
        replacement_result = impact_results["replacement"]

        st.success("Simulation completed.")

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Selected over", impact_metadata["over"])
        metric_b.metric("Score before over", impact_metadata["start_runs"])
        metric_c.metric("Wickets before over", impact_metadata["start_wickets"])

        if int(selected_impact_innings) == 1:
            first_innings_batting_team = impact_metadata["batting_team"]
            second_innings_rows = selected_impact_match_balls[
                selected_impact_match_balls["innings"] == 2
            ]

            if second_innings_rows.empty:
                st.warning(
                    "The source data has no second innings for this match. "
                    "The engine can estimate the first-innings score, but a "
                    "full-match winner comparison is not available."
                )
            else:
                second_innings_batting_team = str(
                    second_innings_rows["batting_team"].iloc[0]
                )

                impact_table = pd.DataFrame([
                    {
                        "Scenario": "Original bowler",
                        "Bowler": original_result["selected_bowler"],
                        f"{second_innings_batting_team} win %": original_result["batting_win_pct"],
                        f"{first_innings_batting_team} win %": original_result["bowling_win_pct"],
                        "Tie %": original_result["tie_pct"],
                        "Mean first-innings score": original_result["mean_first_innings_runs"],
                        "Mean second-innings score": original_result["mean_second_innings_runs"],
                    },
                    {
                        "Scenario": "Replacement bowler",
                        "Bowler": replacement_result["selected_bowler"],
                        f"{second_innings_batting_team} win %": replacement_result["batting_win_pct"],
                        f"{first_innings_batting_team} win %": replacement_result["bowling_win_pct"],
                        "Tie %": replacement_result["tie_pct"],
                        "Mean first-innings score": replacement_result["mean_first_innings_runs"],
                        "Mean second-innings score": replacement_result["mean_second_innings_runs"],
                    },
                ])

                st.markdown("### Estimated full-match outcomes")
                st.dataframe(
                    impact_table.round(2),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            impact_batting_team = impact_metadata["batting_team"]
            impact_bowling_team = impact_metadata["bowling_team"]

            impact_table = pd.DataFrame([
                {
                    "Scenario": "Original bowler",
                    "Bowler": original_result["selected_bowler"],
                    f"{impact_batting_team} win %": original_result["batting_win_pct"],
                    f"{impact_bowling_team} win %": original_result["bowling_win_pct"],
                    "Tie %": original_result["tie_pct"],
                    "Mean first-innings score": original_result["mean_first_innings_runs"],
                    "Mean second-innings score": original_result["mean_second_innings_runs"],
                },
                {
                    "Scenario": "Replacement bowler",
                    "Bowler": replacement_result["selected_bowler"],
                    f"{impact_batting_team} win %": replacement_result["batting_win_pct"],
                    f"{impact_bowling_team} win %": replacement_result["bowling_win_pct"],
                    "Tie %": replacement_result["tie_pct"],
                    "Mean first-innings score": replacement_result["mean_first_innings_runs"],
                    "Mean second-innings score": replacement_result["mean_second_innings_runs"],
                },
            ])

            st.markdown("### Estimated full-match outcomes")
            st.dataframe(
                impact_table.round(2),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 📉 Estimated change from replacing the bowler")
        win_columns = [column for column in impact_table.columns if column.endswith(" win %")]
        if win_columns:
            for team_column in win_columns:
                original_pct = float(impact_table.loc[0, team_column])
                replacement_pct = float(impact_table.loc[1, team_column])
                delta = replacement_pct - original_pct
                st.metric(
                    f"{team_column.removesuffix(' win %')} estimated win chance",
                    f"{replacement_pct:.2f}%",
                    delta=f"{delta:+.2f} percentage points vs original",
                    delta_color="normal",
                )
        st.caption(
            "These are changes between two model scenarios, not proof that the real match winner would change."
        )

        st.markdown("### Innings score comparison")
        score_chart = pd.DataFrame({
            "Scenario": ["Original bowler", "Replacement bowler"],
            "Mean first-innings runs": [
                original_result["mean_first_innings_runs"],
                replacement_result["mean_first_innings_runs"],
            ],
            "Mean second-innings runs": [
                original_result["mean_second_innings_runs"],
                replacement_result["mean_second_innings_runs"],
            ],
        }).set_index("Scenario")

        try:
            import altair as alt
            score_chart_long = score_chart.reset_index().melt(
                id_vars="Scenario", var_name="Innings", value_name="Mean runs"
            )
            grouped_chart = alt.Chart(score_chart_long).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("Scenario:N", title="Bowling scenario", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("Innings:N"),
                y=alt.Y("Mean runs:Q", title="Mean simulated runs", scale=alt.Scale(zero=True)),
                color=alt.Color("Innings:N", title="Innings"),
                tooltip=["Scenario", "Innings", alt.Tooltip("Mean runs:Q", format=".2f")],
            ).properties(height=360)
            st.altair_chart(grouped_chart, use_container_width=True)
        except Exception:
            st.bar_chart(score_chart, use_container_width=True)

        if "impact_table" in locals():
            st.download_button(
                "⬇ Download Match Impact CSV",
                data=impact_table.to_csv(index=False).encode("utf-8"),
                file_name="cricreplay_match_impact.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_full_match_impact",
            )

        st.info(
            "This experimental model resamples historical deliveries by "
            "bowling team and phase. It does not fully model batter skill, "
            "partnerships, venue, pressure, strategy, or every cricket rule. "
            "Win percentages are exploratory estimates, not calibrated "
            "real-world probabilities."
        )

st.caption(
    "CricReplay — Cricket What-If Lab | Historical analysis "
    "and experimental simulation"
)