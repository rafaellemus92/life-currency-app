import streamlit as st
import math
import numpy as np
import pandas as pd
import altair as alt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------
# Life Currency — Streamlit MVP
# A personalized Monte Carlo decision engine
# ---------------------------------------------
# How to run locally:
# 1) pip install streamlit numpy pandas altair
# 2) streamlit run life_currency_streamlit_app.py
# ---------------------------------------------

st.set_page_config(page_title="Life Currency — Monte Carlo", page_icon="💱", layout="wide")

CURRENCIES = [
    "time", "energy", "focus", "money", "health", "relationships", "joy", "learning", "faith"
]

DEFAULT_WEIGHTS = {
    # Negative weight for time means time spent has opportunity cost
    "time": -0.05,
    "energy": 0.20,
    "focus": 0.20,
    "money": 0.10,
    "health": 0.20,
    "relationships": 0.15,
    "joy": 0.15,
    "learning": 0.10,
    "faith": 0.10,
}

BUCKET_BINS = [-1e9, -1.0, -0.2, 0.2, 1.0, 1e9]
BUCKET_LABELS = [
    "High Negative", "Moderate Negative", "Neutral", "Moderate Positive", "High Net Positive"
]

@dataclass
class Action:
    name: str
    costs: Dict[str, float]
    expected_returns: Dict[str, float]
    volatility: float = 0.20  # multiplicative stdev on returns (lognormal factor)

    def sample_returns(self, rng: np.random.Generator) -> Dict[str, float]:
        if self.volatility <= 0:
            factor = 1.0
        else:
            sigma = self.volatility
            mu = -0.5 * sigma**2
            factor = float(rng.lognormal(mean=mu, sigma=sigma))
        return {k: v * factor for k, v in self.expected_returns.items()}

# -----------------------------
# Utility functions
# -----------------------------

def score_from_deltas(deltas: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * v for k, v in deltas.items()))


def bucketize(scores: np.ndarray) -> pd.Series:
    cats = pd.cut(scores, bins=BUCKET_BINS, labels=BUCKET_LABELS, right=True)
    return cats.value_counts(normalize=True).sort_index() * 100


# Sleep response curve: benefits peak near 8 hours, fall off below that
# Returns (health, energy, focus, joy) in relative points per day
# Simple smooth curve using a quadratic penalty around 8h

def sleep_returns(hours: float) -> Dict[str, float]:
    # Peak at 8h -> small decline above, sharper decline below
    diff = hours - 8.0
    health = 14 - 2.2 * (diff ** 2)
    energy = 10 - 1.5 * (diff ** 2)
    focus = 6 - 0.9 * (diff ** 2)
    joy = 3 - 0.3 * (diff ** 2)
    # Clip negatives moderately (sleep loss can be very costly)
    return {
        "health": max(-15.0, health),
        "energy": max(-12.0, energy),
        "focus": max(-10.0, focus),
        "joy": max(-5.0, joy),
    }


# Workout returns: higher if morning (focus boost), scale roughly by minutes

def workout_returns(minutes: int, when: str) -> Dict[str, float]:
    hours = minutes / 60.0
    base_health = 18 * hours
    base_energy = 8 * hours
    base_focus = 6 * hours
    base_joy = 6 * hours
    if when == "Morning":
        base_focus *= 1.25
    elif when == "Evening":
        base_joy *= 1.15
    return {
        "health": base_health,
        "energy": base_energy,
        "focus": base_focus,
        "joy": base_joy,
    }


def family_returns(minutes: int) -> Dict[str, float]:
    hours = minutes / 60.0
    return {
        "relationships": 18 * hours,
        "joy": 10 * hours,
        "health": 2 * hours,
        "faith": 0.5 * hours,
    }


def side_project_returns(minutes: int) -> Dict[str, float]:
    hours = minutes / 60.0
    return {
        "learning": 14 * hours,
        "focus": 6 * hours,
        "joy": 3 * hours,
        "money": 4 * hours,  # small immediate; upside modeled stochastically elsewhere if desired
    }


def meditate_returns(minutes: int) -> Dict[str, float]:
    hours = minutes / 60.0
    return {
        "focus": 8 * hours,
        "joy": 5 * hours,
        "health": 3 * hours,
    }


def faith_returns(minutes: int) -> Dict[str, float]:
    hours = minutes / 60.0
    return {
        "faith": 10 * hours,
        "relationships": 4 * hours,
        "joy": 4 * hours,
        "focus": 2 * hours,
    }


def doomscroll_returns(minutes: int) -> Dict[str, float]:
    hours = minutes / 60.0
    return {
        "joy": 1 * hours,
        "focus": -6 * hours,
        "energy": -4 * hours,
    }


# Build actions from scenario inputs

def build_actions_from_scenario(
    sleep_hours: float,
    workout_min: int,
    workout_when: str,
    family_min: int,
    side_min: int,
    meditate_min: int,
    doom_min: int,
    faith_min: int,
    volatility_map: Dict[str, float],
) -> List[Action]:
    actions: List[Action] = []

    # Sleep action (costs time, returns via sleep_returns)
    sr = sleep_returns(sleep_hours)
    actions.append(Action(
        name=f"Sleep {sleep_hours:.1f}h",
        costs={"time": sleep_hours},
        expected_returns=sr,
        volatility=volatility_map.get("sleep", 0.15)
    ))

    if workout_min > 0:
        wr = workout_returns(workout_min, workout_when)
        actions.append(Action(
            name=f"Workout {workout_min}m ({workout_when})",
            costs={"time": workout_min/60.0, "energy": 6 * (workout_min/60.0)},
            expected_returns=wr,
            volatility=volatility_map.get("workout", 0.20)
        ))

    if family_min > 0:
        fr = family_returns(family_min)
        actions.append(Action(
            name=f"Family {family_min}m",
            costs={"time": family_min/60.0},
            expected_returns=fr,
            volatility=volatility_map.get("family", 0.12)
        ))

    if side_min > 0:
        srp = side_project_returns(side_min)
        actions.append(Action(
            name=f"LifeCurrency Project {side_min}m",
            costs={"time": side_min/60.0, "focus": 4 * (side_min/60.0), "energy": 3 * (side_min/60.0)},
            expected_returns=srp,
            volatility=volatility_map.get("side", 0.30)
        ))

    if meditate_min > 0:
        mr = meditate_returns(meditate_min)
        actions.append(Action(
            name=f"Meditate {meditate_min}m",
            costs={"time": meditate_min/60.0},
            expected_returns=mr,
            volatility=volatility_map.get("meditate", 0.20)
        ))

    if faith_min > 0:
        frt = faith_returns(faith_min)
        actions.append(Action(
            name=f"Faith/Chapel {faith_min}m",
            costs={"time": faith_min/60.0},
            expected_returns=frt,
            volatility=volatility_map.get("faith", 0.18)
        ))

    if doom_min > 0:
        dr = doomscroll_returns(doom_min)
        actions.append(Action(
            name=f"Doomscroll {doom_min}m",
            costs={"time": doom_min/60.0},
            expected_returns=dr,
            volatility=volatility_map.get("doom", 0.40)
        ))

    return actions


# Monte Carlo core

def simulate_plan(actions: List[Action], weights: Dict[str, float], n_iter: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_iter):
        totals = {c: 0.0 for c in CURRENCIES}
        for a in actions:
            # apply costs
            for k, v in a.costs.items():
                totals[k] = totals.get(k, 0.0) - float(v)
            # apply stochastic returns
            sampled = a.sample_returns(rng)
            for k, v in sampled.items():
                totals[k] = totals.get(k, 0.0) + float(v)
        totals["score"] = score_from_deltas(totals, weights)
        rows.append(totals)
    df = pd.DataFrame(rows)
    buckets = bucketize(df["score"].to_numpy())
    return df, buckets


def pretty_bucket_chart(bucket_series: pd.Series, title: str):
    dfb = bucket_series.reset_index()
    dfb.columns = ["Outcome", "Percent"]
    chart = (
        alt.Chart(dfb)
        .mark_bar()
        .encode(
            x=alt.X("Outcome", sort=BUCKET_LABELS),
            y=alt.Y("Percent", title="% of trials"),
            tooltip=["Outcome", alt.Tooltip("Percent", format=".1f")],
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c in CURRENCIES] + ["score"]
    desc = df[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    return desc


# -----------------------------
# Sidebar — User Profile & Settings
# -----------------------------
st.sidebar.header("👤 Your Profile (Defaults Pre-Filled)")
age = st.sidebar.number_input("Age", value=33, min_value=18, max_value=100)
sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"], index=0)
background = st.sidebar.text_input("Background / Identity", value="Mexican American")
marital = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Partnered"], index=1)
children = st.sidebar.text_input("Children", value="Daughter 18mo, Son 3mo")
salary = st.sidebar.number_input("Annual Salary ($)", value=240_000, min_value=0, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱ Schedule Defaults")
work_start = st.sidebar.time_input("Work start", value=pd.to_datetime("07:30").time())
work_end = st.sidebar.time_input("Work end", value=pd.to_datetime("17:00").time())
commute_each = st.sidebar.number_input("Commute each way (min)", value=30, min_value=0, max_value=180)
sleep_goal = st.sidebar.number_input("Sleep goal (hours)", value=8.0, min_value=4.0, max_value=10.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Simulation Settings")
n_iter = st.sidebar.slider("Monte Carlo iterations", min_value=1000, max_value=20000, value=5000, step=1000)
seed = st.sidebar.number_input("Random seed", value=42)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚 Currency Weights (to score)")
weights = {}
for c in CURRENCIES:
    default = DEFAULT_WEIGHTS.get(c, 0.0)
    weights[c] = st.sidebar.slider(f"Weight: {c}", min_value=-0.5, max_value=0.5, value=float(default), step=0.01)

st.sidebar.markdown("---")
exchange_rate = st.sidebar.slider(
    "Dollars per 1.0 score point (QOL ↔ $)", min_value=0, max_value=200, value=50, step=5
)

# -----------------------------
# Main — Scenario Builder
# -----------------------------
st.title("💱 Life Currency — Monte Carlo Decision Engine")
st.caption("Personalized simulation to compare daily choices using multi-currency trade-offs.")

st.markdown("""
Use **Scenario A** and **Scenario B** to compare choices like **sleep 8h vs 7h + 60m workout**, morning vs evening exercise, or more family time vs side project. 
Results show probabilistic outcomes and a QOL-adjusted dollar view.
""")

with st.expander("Scenario A — Inputs", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        a_sleep = st.slider("Sleep (hours)", 4.0, 10.0, value=float(sleep_goal), step=0.5, key="A_sleep")
        a_workout_min = st.slider("Workout (min)", 0, 120, value=60, step=15, key="A_workout")
        a_workout_when = st.selectbox("Workout time", ["Morning", "Midday", "Evening"], index=0, key="A_when")
    with col2:
        a_family_min = st.slider("Family time (min)", 0, 180, value=60, step=15, key="A_family")
        a_side_min = st.slider("LifeCurrency project (min)", 0, 180, value=45, step=15, key="A_side")
        a_meditate_min = st.slider("Meditate (min)", 0, 60, value=10, step=5, key="A_meditate")
    with col3:
        a_faith_min = st.slider("Faith/Chapel (min)", 0, 120, value=20, step=10, key="A_faith")
        a_doom_min = st.slider("Doomscroll (min)", 0, 120, value=15, step=5, key="A_doom")
        
    volA = {
        "sleep": 0.15, "workout": 0.22, "family": 0.12, "side": 0.30,
        "meditate": 0.18, "faith": 0.18, "doom": 0.40
    }
    actions_A = build_actions_from_scenario(
        a_sleep, a_workout_min, a_workout_when,
        a_family_min, a_side_min, a_meditate_min, a_doom_min, a_faith_min,
        volA,
    )

with st.expander("Scenario B — Inputs", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        b_sleep = st.slider("Sleep (hours)", 4.0, 10.0, value=float(max(6.0, sleep_goal-1.0)), step=0.5, key="B_sleep")
        b_workout_min = st.slider("Workout (min)", 0, 120, value=60, step=15, key="B_workout")
        b_workout_when = st.selectbox("Workout time", ["Morning", "Midday", "Evening"], index=0, key="B_when")
    with col2:
        b_family_min = st.slider("Family time (min)", 0, 180, value=60, step=15, key="B_family")
        b_side_min = st.slider("LifeCurrency project (min)", 0, 180, value=60, step=15, key="B_side")
        b_meditate_min = st.slider("Meditate (min)", 0, 60, value=5, step=5, key="B_meditate")
    with col3:
        b_faith_min = st.slider("Faith/Chapel (min)", 0, 120, value=15, step=10, key="B_faith")
        b_doom_min = st.slider("Doomscroll (min)", 0, 120, value=10, step=5, key="B_doom")

    volB = {
        "sleep": 0.15, "workout": 0.22, "family": 0.12, "side": 0.30,
        "meditate": 0.18, "faith": 0.18, "doom": 0.40
    }
    actions_B = build_actions_from_scenario(
        b_sleep, b_workout_min, b_workout_when,
        b_family_min, b_side_min, b_meditate_min, b_doom_min, b_faith_min,
        volB,
    )

run_btn = st.button("Run Monte Carlo for Both Scenarios", type="primary")

if run_btn:
    colA, colB = st.columns(2)

    with st.spinner("Simulating Scenario A…"):
        dfA, bucketsA = simulate_plan(actions_A, weights, n_iter=n_iter, seed=int(seed))
    with st.spinner("Simulating Scenario B…"):
        dfB, bucketsB = simulate_plan(actions_B, weights, n_iter=n_iter, seed=int(seed)+1)

    with colA:
        st.subheader("Scenario A Results")
        pretty_bucket_chart(bucketsA, "Outcome Distribution — Scenario A")
        st.dataframe(summary_table(dfA))

    with colB:
        st.subheader("Scenario B Results")
        pretty_bucket_chart(bucketsB, "Outcome Distribution — Scenario B")
        st.dataframe(summary_table(dfB))

    # Comparison block
    st.markdown("---")
    st.subheader("📊 Scenario Comparison")

    # Compare expected score
    meanA = float(dfA["score"].mean())
    meanB = float(dfB["score"].mean())
    delta_score = meanB - meanA

    # QOL-adjusted dollar view
    qol_dollars_A = meanA * exchange_rate
    qol_dollars_B = meanB * exchange_rate
    delta_dollars = delta_score * exchange_rate

    comp_df = pd.DataFrame({
        "Scenario": ["A", "B"],
        "Mean Score": [meanA, meanB],
        f"QOL-$ (rate ${exchange_rate}/pt)": [qol_dollars_A, qol_dollars_B],
    })
    st.dataframe(comp_df.set_index("Scenario"))

    # Friendly text
    verdict = "Scenario B" if delta_score > 0 else ("Scenario A" if delta_score < 0 else "Tie")
    st.success(
        f"Verdict: **{verdict}** looks better on average. Δscore = {delta_score:.3f}, which is ≈ ${delta_dollars:,.0f} in QOL-adjusted value at ${exchange_rate}/pt."
    )

    # Show top currency differences
    avgA = dfA[CURRENCIES].mean()
    avgB = dfB[CURRENCIES].mean()
    delta = (avgB - avgA).sort_values(ascending=False)
    st.markdown("**Average currency deltas (B − A):**")
    st.dataframe(delta.to_frame("Δ per day").T)

else:
    st.info("Set your two scenarios above, then click **Run Monte Carlo for Both Scenarios**.")
