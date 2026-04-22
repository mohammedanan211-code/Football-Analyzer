"""
player_analysis.py
==================
Individual player metrics:
- Radar chart comparisons
- Role classification
- Performance scoring
- Influence index
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")


# ─── Player feature extraction ───────────────────────────────────────────────

def extract_player_features(events_df: pd.DataFrame, player_name: str) -> dict:
    """Extract per-player metrics from events."""
    player = events_df[events_df["player_name"] == player_name]
    passes = player[player["type"] == "Pass"]
    shots  = player[player["type"] == "Shot"]
    received = events_df[events_df["recipient_name"] == player_name]

    total_passes   = len(passes)
    completed      = len(passes[passes["outcome"] == "Complete"])
    completion_pct = completed / max(total_passes, 1) * 100

    goals  = int(shots["outcome"].eq("Goal").sum())
    assists_proxy = len(
        passes[(passes["outcome"] == "Complete") & (passes["end_x"] > 85)]
    )

    avg_x = float(passes["x"].mean()) if total_passes > 0 else 0
    progressive_pct = float(
        ((passes["end_x"] - passes["x"]) > 10).mean()
    ) * 100 if total_passes > 0 else 0

    long_ball_pct = float(
        (passes.get("pass_length", pd.Series(dtype=float)) > 32).mean()
    ) * 100 if total_passes > 0 else 0

    right_pct = float((passes["y"] > 50).mean()) * 100 if total_passes > 0 else 0
    left_pct  = float((passes["y"] < 30).mean()) * 100 if total_passes > 0 else 0
    central_pct = 100 - right_pct - left_pct

    return {
        "player": player_name,
        "total_passes": total_passes,
        "completion_pct": round(completion_pct, 1),
        "progressive_pct": round(progressive_pct, 1),
        "long_ball_pct": round(long_ball_pct, 1),
        "avg_position_x": round(avg_x, 1),
        "shots": len(shots),
        "goals": goals,
        "assists_proxy": assists_proxy,
        "passes_received": len(received),
        "right_pct": round(right_pct, 1),
        "left_pct": round(left_pct, 1),
        "central_pct": round(central_pct, 1),
    }


def classify_player_role(features: dict) -> str:
    """Classify player role based on stats."""
    x = features.get("avg_position_x", 60)
    goals = features.get("goals", 0)
    prog  = features.get("progressive_pct", 0)

    if x < 25:    return "Goalkeeper"
    if x < 40:    return "Central Defender"
    if x < 50 and prog > 30: return "Ball-Playing Defender"
    if x < 55:    return "Defensive Midfielder"
    if x < 65 and prog > 35: return "Box-to-Box Midfielder"
    if x < 65:    return "Central Midfielder"
    if x < 75:    return "Attacking Midfielder"
    if goals > 2: return "Striker"
    return "Wide Midfielder / Winger"


# ─── Radar chart ─────────────────────────────────────────────────────────────

RADAR_METRICS = [
    ("Passing Volume",   "total_passes",       200,  False),
    ("Completion %",     "completion_pct",      100,  False),
    ("Progression %",    "progressive_pct",     60,   False),
    ("Goals",            "goals",               8,    False),
    ("Passes Received",  "passes_received",     180,  False),
    ("Key Ball %",       "assists_proxy",       20,   False),
]

def _scale(val, max_val):
    return min(float(val) / max(max_val, 1), 1.0)


def plot_player_radar(features_list: list[dict],
                      labels: list[str],
                      figsize=(9, 9),
                      player_colors=None) -> plt.Figure:
    """
    Radar (spider) chart comparing up to 3 players.
    """
    n_metrics = len(RADAR_METRICS)
    angles = [2 * np.pi * i / n_metrics for i in range(n_metrics)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=figsize,
                            subplot_kw=dict(polar=True),
                            facecolor="#0d1a2d")
    ax.set_facecolor("#0d1a2d")
    ax.spines["polar"].set_color("#2a3a5c")

    # Grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in RADAR_METRICS],
                        color="white", fontsize=9, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%","40%","60%","80%","100%"],
                        color="#555577", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(color="#2a3a5c", linewidth=0.7, alpha=0.6)
    ax.set_ylim(0, 1)

    colors = player_colors if player_colors else ["#FFD700", "#4fc3f7", "#ff6b6b", "#90ee90"]
    for i, (feats, label) in enumerate(zip(features_list, labels)):
        values = [_scale(feats.get(m[1], 0), m[2]) for m in RADAR_METRICS]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, color=color, lw=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15),
              facecolor="#0d1a2d", edgecolor="#2a3a5c",
              labelcolor="white", fontsize=9)
    ax.set_title("Player Comparison — Radar Chart",
                  color="white", fontsize=13, fontweight="bold",
                  y=1.08)
    return fig


# ─── Team Radar Chart ─────────────────────────────────────────────────────────

TEAM_RADAR_METRICS = [
    ("Ball Retention",   "completion_rate",      0.95,  False),
    ("Territory",        "territory_pct",        0.75,  False),
    ("Short Passing",    "short_pass_pct",       0.70,  False),
    ("Directness",       "long_ball_pct",        0.35,  False),
    ("Forward Drive",    "forward_pass_pct",     0.60,  False),
    ("Verticality",      "vertical_progression", 12.0,  False),
    ("Tempo",            "tempo",                8.0,   False),
    ("Shot Volume",      "shot_rate",            0.04,  False),
]

def plot_team_radar(features_list: list[dict],
                    labels: list[str],
                    figsize=(9, 9),
                    team_colors=None) -> plt.Figure:
    """
    Radar chart comparing two or more teams on tactical metrics.
    """
    n_metrics = len(TEAM_RADAR_METRICS)
    angles = [2 * np.pi * i / n_metrics for i in range(n_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize,
                            subplot_kw=dict(polar=True),
                            facecolor="#0d1a2d")
    ax.set_facecolor("#0d1a2d")
    ax.spines["polar"].set_color("#2a3a5c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in TEAM_RADAR_METRICS],
                        color="white", fontsize=9, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%","40%","60%","80%","100%"],
                        color="#555577", fontsize=7)
    ax.grid(color="#2a3a5c", linewidth=0.7, alpha=0.6)
    ax.set_ylim(0, 1)

    colors = team_colors if team_colors else ["#4fc3f7", "#ff8c00", "#FFD700", "#ff6b6b"]
    for i, (feats, label) in enumerate(zip(features_list, labels)):
        values = [_scale(feats.get(m[1], 0), m[2]) for m in TEAM_RADAR_METRICS]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, color=color, lw=3, label=label, marker='o', markersize=4)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15),
              facecolor="#0d1a2d", edgecolor="#2a3a5c",
              labelcolor="white", fontsize=9)
    ax.set_title("Tactical DNA Comparison",
                  color="white", fontsize=14, fontweight="bold",
                  y=1.1)
    return fig


# ─── Team player summary table ───────────────────────────────────────────────

def get_team_player_stats(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Return per-player stats dataframe for a team."""
    team_passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["team"] == team_name)
    ]
    players = team_passes["player_name"].unique()

    rows = []
    for p in players:
        feats = extract_player_features(events_df, p)
        feats["role"] = classify_player_role(feats)
        rows.append(feats)

    df = pd.DataFrame(rows).sort_values("total_passes", ascending=False)
    df = df.rename(columns={
        "player": "Player",
        "total_passes": "Passes",
        "completion_pct": "Completion %",
        "progressive_pct": "Progressive %",
        "goals": "Goals",
        "assists_proxy": "Key Balls",
        "role": "Role",
        "avg_position_x": "Avg X",
    })
    return df[[
        "Player", "Role", "Passes", "Completion %",
        "Progressive %", "Goals", "Key Balls", "Avg X"
    ]]


# ─── Influence index ─────────────────────────────────────────────────────────

def compute_influence_index(events_df: pd.DataFrame,
                             player_name: str,
                             G=None,
                             metrics_df=None) -> float:
    """
    Composite influence index (0–100) combining:
    - Pass volume (20%)
    - Completion rate (20%)
    - Betweenness centrality (30%)
    - Progressive passing (15%)
    - Goal contributions (15%)
    """
    feats = extract_player_features(events_df, player_name)

    vol_score  = min(feats["total_passes"] / 200, 1.0) * 20
    comp_score = feats["completion_pct"] / 100 * 20
    prog_score = feats["progressive_pct"] / 60 * 15
    gc_score   = min((feats["goals"] + feats["assists_proxy"]) / 10, 1.0) * 15

    between_score = 0
    if metrics_df is not None and not metrics_df.empty:
        row = metrics_df[metrics_df["player"] == player_name]
        if not row.empty:
            between_score = float(row["betweenness_centrality"].iloc[0]) * 30 / 0.5

    total = vol_score + comp_score + prog_score + gc_score + between_score
    return round(min(total, 100), 1)
