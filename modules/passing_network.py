"""
passing_network.py
==================
Graph-based passing network analysis.
- Builds directed weighted graph from event data
- Computes centrality, betweenness, playmaker scores
- Identifies key passing corridors
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")


# ─── Pitch drawing helper ────────────────────────────────────────────────────

def draw_pitch(ax, color="#1a1a2e", line_color="#3a3a5c", alpha=0.9):
    ax.set_facecolor(color)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    lc = line_color
    lw = 1.2

    # Outline & centre
    ax.plot([0,120,120,0,0], [0,0,80,80,0], color=lc, lw=lw)
    ax.plot([60,60], [0,80], color=lc, lw=lw)
    circle = plt.Circle((60,40), 9.15, color=lc, fill=False, lw=lw)
    ax.add_patch(circle)
    ax.plot(60, 40, "o", color=lc, markersize=2)

    # Penalty areas
    for x_start, x_end in [(0, 16.5), (103.5, 120)]:
        ax.plot([x_start, x_end, x_end, x_start, x_start],
                [13.84, 13.84, 66.16, 66.16, 13.84], color=lc, lw=lw)
    for x_start, x_end in [(0, 5.5), (114.5, 120)]:
        ax.plot([x_start, x_end, x_end, x_start, x_start],
                [30.34, 30.34, 49.66, 49.66, 30.34], color=lc, lw=lw)

    # Goals
    for x, dx in [(0, -2), (120, 2)]:
        ax.plot([x, x+dx, x+dx, x], [34.76, 34.76, 45.24, 45.24], color="#f0c040", lw=2)

    # Penalty spots
    ax.plot(11, 40, "o", color=lc, markersize=2)
    ax.plot(109, 40, "o", color=lc, markersize=2)

    return ax


# ─── Core network builder ────────────────────────────────────────────────────

def build_passing_network(events_df: pd.DataFrame, min_passes: int = 2):
    """
    Build a directed weighted NetworkX graph from pass events.

    Returns
    -------
    G : nx.DiGraph
    node_positions : dict  player_name -> (avg_x, avg_y)
    node_metrics   : pd.DataFrame
    """
    passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["outcome"] == "Complete") &
        (events_df["recipient_name"].notna())
    ].copy()

    if passes.empty:
        return nx.DiGraph(), {}, pd.DataFrame()

    # ── Edge weights & lengths ───────────────────────────────────────────────
    edge_data = (
        passes.groupby(["player_name", "recipient_name"])
        .agg(
            weight=("type", "size"),
            avg_length=("pass_length", "mean")
        )
        .reset_index()
    )
    edge_data = edge_data[edge_data["weight"] >= min_passes]

    G = nx.DiGraph()
    for _, row in edge_data.iterrows():
        G.add_edge(row["player_name"], row["recipient_name"], 
                   weight=int(row["weight"]),
                   avg_length=float(row["avg_length"]))

    # ── Node average positions ────────────────────────────────────────────────
    passer_pos = passes.groupby("player_name")[["x", "y"]].mean()
    receiver_pos = passes.groupby("recipient_name")[["end_x", "end_y"]].mean()
    receiver_pos.columns = ["x", "y"]
    combined = pd.concat([passer_pos, receiver_pos]).groupby(level=0).mean()

    node_positions = {name: (row["x"], row["y"]) for name, row in combined.iterrows()}

    # ── Network metrics ───────────────────────────────────────────────────────
    metrics = []
    undirected = G.to_undirected()
    degree_cent  = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank     = nx.pagerank(G, weight="weight", alpha=0.85)
    in_degree    = dict(G.in_degree(weight="weight"))
    out_degree   = dict(G.out_degree(weight="weight"))

    for node in G.nodes():
        pos = node_positions.get(node, (60, 40))
        metrics.append({
            "player": node,
            "x": pos[0],
            "y": pos[1],
            "degree_centrality": round(degree_cent.get(node, 0), 4),
            "betweenness_centrality": round(between_cent.get(node, 0), 4),
            "pagerank": round(pagerank.get(node, 0), 4),
            "passes_sent": int(out_degree.get(node, 0)),
            "passes_received": int(in_degree.get(node, 0)),
            "total_involvement": int(in_degree.get(node, 0) + out_degree.get(node, 0)),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("betweenness_centrality", ascending=False)

    # Mark playmakers (top 3 by combined score)
    metrics_df["playmaker_score"] = (
        0.4 * metrics_df["betweenness_centrality"] +
        0.4 * metrics_df["pagerank"] +
        0.2 * metrics_df["degree_centrality"]
    )
    metrics_df["is_playmaker"] = False
    top3 = metrics_df.nlargest(3, "playmaker_score").index
    metrics_df.loc[top3, "is_playmaker"] = True

    return G, node_positions, metrics_df


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_passing_network(G, node_positions, metrics_df, team_name: str,
                         figsize=(14, 9), player_color="#4fc3f7", playmaker_color="#FFD700",
                         short_pass_color="#4fc3f7", long_pass_color="#ff8c00") -> plt.Figure:
    """
    Render a beautiful passing network on a dark pitch.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d1a")
    draw_pitch(ax, color="#0d1a2d", line_color="#2a3a5c")

    if G.number_of_nodes() == 0:
        ax.text(60, 40, "No pass data", ha="center", va="center",
                color="white", fontsize=14)
        return fig

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_lengths = [G[u][v].get("avg_length", 15) for u, v in G.edges()]
    
    max_w = max(edge_weights) if edge_weights else 1
    
    # Color map for length: Short Color -> Long Color
    from matplotlib.colors import LinearSegmentedColormap
    length_cmap = LinearSegmentedColormap.from_list("pass_length_cmap", [short_pass_color, long_pass_color])
    norm_len = Normalize(vmin=10, vmax=35) # standard range for short to long

    for (u, v), w, l in zip(G.edges(), edge_weights, edge_lengths):
        if u not in node_positions or v not in node_positions:
            continue
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        alpha = 0.4 + 0.5 * (w / max_w)
        lw = 1.0 + 4.0 * (w / max_w)
        color = length_cmap(norm_len(l))
        
        ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=lw,
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.1"
                    ))

    # ── Nodes ─────────────────────────────────────────────────────────────────
    player_metrics = metrics_df.set_index("player")

    for node in G.nodes():
        if node not in node_positions:
            continue
        x, y = node_positions[node]
        m = player_metrics.loc[node] if node in player_metrics.index else None
        is_playmaker = bool(m["is_playmaker"]) if m is not None else False
        bc = float(m["betweenness_centrality"]) if m is not None else 0
        size = 150 + 800 * bc
        color = playmaker_color if is_playmaker else player_color
        edge_c = "#FF6B35" if is_playmaker else "#1565C0"

        ax.scatter(x, y, s=size, c=color, edgecolors=edge_c,
                   linewidths=2.5, zorder=5)

        short_name = node.split()[-1]
        ax.text(x, y + 3.5, short_name,
                ha="center", va="bottom", fontsize=7,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1a2d",
                          edgecolor="none", alpha=0.7))

    # ── Legend & title ────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=playmaker_color, label="Playmaker (Top 3)"),
        mpatches.Patch(color=player_color, label="Player"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#0d1a2d", edgecolor="#2a3a5c",
              labelcolor="white", fontsize=9)

    fig.suptitle(f"{team_name} — Passing Network",
                 color="white", fontsize=15, fontweight="bold", y=0.97)
    ax.set_title("Node size = centrality | Arrow thickness = frequency | Color = pass length (Short → Long)",
                 color="#8899aa", fontsize=9, pad=6)

    plt.tight_layout()
    return fig


# ─── Key corridor detection ──────────────────────────────────────────────────

def get_key_corridors(G, node_positions, top_n: int = 5) -> pd.DataFrame:
    """Return the top passing corridors by weight."""
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({"from": u, "to": v, "passes": data["weight"]})
    df = pd.DataFrame(rows).sort_values("passes", ascending=False)

    if node_positions:
        def corridor_zone(u, v):
            x1 = node_positions.get(u, (60, 40))[0]
            x2 = node_positions.get(v, (60, 40))[0]
            avg_x = (x1 + x2) / 2
            if avg_x < 40:
                return "Defensive Third"
            elif avg_x < 80:
                return "Middle Third"
            else:
                return "Attacking Third"
        df["zone"] = df.apply(lambda r: corridor_zone(r["from"], r["to"]), axis=1)

    return df.head(top_n)
