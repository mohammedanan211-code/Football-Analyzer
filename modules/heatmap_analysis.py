"""
heatmap_analysis.py
===================
Spatial analysis and heatmaps:
- Player position heatmaps (KDE)
- Team attacking / defensive zone maps
- Zone control comparison
- Shot maps
- Progressive pass maps
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")


# ─── Pitch helper (shared) ───────────────────────────────────────────────────

def _draw_pitch_heatmap(ax, line_color="white", alpha=0.35, lw=1.0):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")
    lc, a = line_color, alpha

    ax.plot([0,120,120,0,0], [0,0,80,80,0], color=lc, lw=lw, alpha=a)
    ax.plot([60,60], [0,80], color=lc, lw=lw, alpha=a)
    circle = plt.Circle((60,40), 9.15, color=lc, fill=False, lw=lw, alpha=a)
    ax.add_patch(circle)
    for x_s, x_e in [(0,16.5),(103.5,120)]:
        ax.plot([x_s,x_e,x_e,x_s,x_s],[13.84,13.84,66.16,66.16,13.84],
                color=lc, lw=lw, alpha=a)
    for x_s, x_e in [(0,5.5),(114.5,120)]:
        ax.plot([x_s,x_e,x_e,x_s,x_s],[30.34,30.34,49.66,49.66,30.34],
                color=lc, lw=lw, alpha=a)
    ax.plot(11,40,"o", color=lc, markersize=2, alpha=a)
    ax.plot(109,40,"o", color=lc, markersize=2, alpha=a)


# ─── KDE heatmap ─────────────────────────────────────────────────────────────

def _kde_heatmap(x_vals, y_vals, sigma=4, bins=(120, 80)):
    h, xedges, yedges = np.histogram2d(x_vals, y_vals,
                                        bins=bins, range=[[0,120],[0,80]])
    h = gaussian_filter(h.T, sigma=sigma)
    return h, xedges, yedges


# ─── Custom colourmaps ───────────────────────────────────────────────────────

CMAP_HEAT = LinearSegmentedColormap.from_list(
    "football_heat",
    ["#0d1a2d", "#1a3a5c", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8", "#FFFFFF"]
)
CMAP_FIRE = LinearSegmentedColormap.from_list(
    "football_fire",
    ["#0d1a2d", "#3a0000", "#8b0000", "#cc2200", "#ff4500", "#ff8c00", "#FFD700"]
)
CMAP_GREEN = LinearSegmentedColormap.from_list(
    "football_green",
    ["#0d1a2d", "#003300", "#006400", "#228B22", "#32CD32", "#90EE90", "#CCFFCC"]
)


# ─── Team heatmap ─────────────────────────────────────────────────────────────

def plot_team_heatmap(events_df: pd.DataFrame, team_name: str,
                      event_type: str = "Pass",
                      figsize=(13, 8.5)) -> plt.Figure:
    """
    Full-pitch team heatmap (pass origins or shot origins).
    """
    filtered = events_df[
        (events_df["type"] == event_type) &
        (events_df["team"] == team_name)
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1a2d")
    ax.set_facecolor("#0d1a2d")

    if filtered.empty:
        ax.text(60, 40, "No data", ha="center", va="center",
                color="white", fontsize=14)
        return fig

    h, xe, ye = _kde_heatmap(filtered["x"].values, filtered["y"].values)
    ax.imshow(h, origin="lower", extent=[0, 120, 0, 80],
              cmap=CMAP_HEAT, aspect="equal", interpolation="bilinear",
              alpha=0.92, zorder=1)
    _draw_pitch_heatmap(ax)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=CMAP_HEAT,
                                norm=plt.Normalize(vmin=h.min(), vmax=h.max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical",
                      fraction=0.025, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("Activity Density", color="white", fontsize=9)

    fig.suptitle(f"{team_name} — {event_type} Heatmap",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_title("Brighter = Higher concentration of activity",
                 color="#8899aa", fontsize=9)
    plt.tight_layout()
    return fig


# ─── Player heatmap ──────────────────────────────────────────────────────────

def plot_player_heatmap(events_df: pd.DataFrame, player_name: str,
                        figsize=(13, 8.5)) -> plt.Figure:
    """Individual player position heatmap."""
    player_events = events_df[events_df["player_name"] == player_name]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1a2d")
    ax.set_facecolor("#0d1a2d")

    if player_events.empty:
        ax.text(60, 40, f"No data for {player_name}",
                ha="center", va="center", color="white", fontsize=12)
        return fig

    h, _, _ = _kde_heatmap(player_events["x"].values, player_events["y"].values, sigma=5)
    ax.imshow(h, origin="lower", extent=[0,120,0,80],
              cmap=CMAP_FIRE, aspect="equal", interpolation="bilinear",
              alpha=0.90, zorder=1)
    _draw_pitch_heatmap(ax)

    # Average position marker
    avg_x, avg_y = player_events["x"].mean(), player_events["y"].mean()
    ax.scatter(avg_x, avg_y, s=200, c="#FFD700", edgecolors="white",
               linewidths=2, zorder=5, marker="*")
    ax.text(avg_x, avg_y + 4, f"Avg pos\n({avg_x:.0f}, {avg_y:.0f})",
            ha="center", color="white", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#0d1a2d", alpha=0.7))

    fig.suptitle(f"{player_name} — Position Heatmap",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ─── Shot map ────────────────────────────────────────────────────────────────

def plot_shot_map(events_df: pd.DataFrame, team_name: str,
                  figsize=(13, 8.5)) -> plt.Figure:
    """Visualise shot locations, outcomes, and a goal mouth diagram."""
    shots = events_df[
        (events_df["type"] == "Shot") &
        (events_df["team"] == team_name)
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"width_ratios": [3, 1]},
                              facecolor="#0d1a2d")

    # ── Left: pitch shot map ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d1a2d")
    _draw_pitch_heatmap(ax, line_color="#3a3a5c", alpha=0.9, lw=1.2)

    # Shade attacking half
    ax.add_patch(mpatches.FancyArrowPatch(
        (60, 0), (60, 80), color="none"))
    ax.fill_betweenx([0, 80], 60, 120, alpha=0.07, color="#00ff88")

    if not shots.empty:
        outcome_colors = {
            "Goal": "#FFD700",
            "Saved": "#4fc3f7",
            "Off T": "#ff6b6b",
            "Blocked": "#bb86fc",
        }
        for outcome, color in outcome_colors.items():
            mask = shots["outcome"].str.contains(outcome, case=False, na=False)
            if mask.any():
                size = 300 if outcome == "Goal" else 100
                marker = "*" if outcome == "Goal" else "o"
                ax.scatter(shots[mask]["x"], shots[mask]["y"],
                           s=size, c=color, marker=marker,
                           edgecolors="white", linewidths=0.5,
                           label=outcome, zorder=5, alpha=0.9)

        ax.legend(loc="lower left", facecolor="#0d1a2d",
                  edgecolor="#2a3a5c", labelcolor="white", fontsize=8)

    ax.set_title(f"{team_name} — Shot Map", color="white",
                 fontsize=12, fontweight="bold")

    # ── Right: goal mouth diagram ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.set_xlim(32, 48)
    ax2.set_ylim(0, 3)
    ax2.set_aspect("equal")
    ax2.axis("off")
    # Goal posts
    goal_rect = mpatches.FancyBboxPatch((36, 0), 8, 2.44,
                                         linewidth=2, edgecolor="#f0c040",
                                         facecolor="none",
                                         boxstyle="square,pad=0")
    ax2.add_patch(goal_rect)
    # Crossbar
    ax2.plot([36, 44], [2.44, 2.44], color="#f0c040", lw=2)

    if not shots.empty:
        goals = shots[shots["outcome"] == "Goal"]
        if not goals.empty:
            gx = goals["end_y"].clip(32, 48)
            gy = np.random.uniform(0.1, 2.2, len(goals))
            ax2.scatter(gx, gy, c="#FFD700", s=120, zorder=5,
                        edgecolors="white", lw=0.5, marker="*")

    ax2.set_title("Goal Mouth", color="white", fontsize=9, fontweight="bold")
    ax2.text(40, -0.3, f"{len(shots)} shots  |  "
             f"{shots['outcome'].eq('Goal').sum()} goals",
             ha="center", color="#8899aa", fontsize=8)

    plt.tight_layout()
    return fig


# ─── Progressive pass map ────────────────────────────────────────────────────

def plot_progressive_passes(events_df: pd.DataFrame, team_name: str,
                             figsize=(13, 8.5)) -> plt.Figure:
    """
    Map passes that advance the ball significantly towards the goal.
    Progressive = end_x - start_x > 10 AND crosses a third boundary.
    """
    passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["outcome"] == "Complete") &
        (events_df["team"] == team_name)
    ].copy()

    passes["progressive"] = (
        (passes["end_x"] - passes["x"] > 10) &
        ((passes["x"] < 80) | (passes["end_x"] > passes["x"]))
    )
    prog = passes[passes["progressive"]]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1a2d")
    ax.set_facecolor("#0d1a2d")
    _draw_pitch_heatmap(ax)

    if prog.empty:
        ax.text(60, 40, "No progressive passes found",
                ha="center", va="center", color="white", fontsize=12)
        fig.suptitle(f"{team_name} — Progressive Passes",
                     color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    prog_gain = prog["end_x"] - prog["x"]
    norm = plt.Normalize(vmin=prog_gain.min(), vmax=prog_gain.max())
    cmap = CMAP_GREEN

    for _, row in prog.iterrows():
        gain = row["end_x"] - row["x"]
        color = cmap(norm(gain))
        ax.annotate("",
                    xy=(row["end_x"], row["end_y"]),
                    xytext=(row["x"], row["y"]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=0.8 + 1.5 * norm(gain),
                        alpha=0.55,
                        connectionstyle="arc3,rad=0.0"
                    ))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("Metres Gained", color="white", fontsize=9)

    fig.suptitle(f"{team_name} — Progressive Pass Map ({len(prog)} passes)",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_title("Arrows show direction & gain; greener = more progression",
                 color="#8899aa", fontsize=9)
    plt.tight_layout()
    return fig


# ─── Zone control comparison ─────────────────────────────────────────────────

def plot_zone_control(team_a_events: pd.DataFrame, team_b_events: pd.DataFrame,
                      team_a_name: str, team_b_name: str,
                      figsize=(14, 5)) -> plt.Figure:
    """Compare zone occupation between two teams."""
    zones = [(0, 40, "Defensive\nThird"), (40, 80, "Middle\nThird"), (80, 120, "Attacking\nThird")]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0d1a2d")
    fig.suptitle(f"Zone Control: {team_a_name} vs {team_b_name}",
                 color="white", fontsize=13, fontweight="bold")

    colors_a = "#4fc3f7"
    colors_b = "#ff8c00"

    for ax, (x_start, x_end, label) in zip(axes, zones):
        ax.set_facecolor("#0d1a2d")
        a_count = len(team_a_events[
            (team_a_events["type"] == "Pass") &
            (team_a_events["x"].between(x_start, x_end))
        ])
        b_count = len(team_b_events[
            (team_b_events["type"] == "Pass") &
            (team_b_events["x"].between(x_start, x_end))
        ])

        total = a_count + b_count or 1
        a_pct = a_count / total * 100
        b_pct = b_count / total * 100

        bars = ax.barh([team_a_name[:12], team_b_name[:12]],
                        [a_pct, b_pct],
                        color=[colors_a, colors_b],
                        height=0.5, edgecolor="none")

        for bar, pct in zip(bars, [a_pct, b_pct]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{pct:.0f}%", va="center", color="white", fontsize=9)

        ax.set_xlim(0, 105)
        ax.set_title(label, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a3a5c")
        ax.spines["left"].set_color("#2a3a5c")
        ax.set_facecolor("#0d1a2d")

    plt.tight_layout()
    return fig
