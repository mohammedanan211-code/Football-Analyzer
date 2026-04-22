"""
tactical_analysis.py
====================
Advanced tactical analysis:
- Tactical style detection (K-Means + PCA)
- Possession chain extraction & classification
- Formation detection from positional data
- Mid-match tactical shift detection
- Pressing intensity scoring
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# ─── Feature engineering ─────────────────────────────────────────────────────

def extract_tactical_features(events_df: pd.DataFrame, team_name: str) -> dict:
    """
    Extract 15+ tactical features from event data.
    Returns a feature dict for one team.
    """
    passes = events_df[(events_df["type"] == "Pass") & (events_df["team"] == team_name)].copy()
    shots  = events_df[(events_df["type"] == "Shot")  & (events_df["team"] == team_name)].copy()

    total_passes = len(passes)
    if total_passes == 0:
        return {}

    completed = passes[passes["outcome"] == "Complete"]
    completion_rate = len(completed) / total_passes

    # Directional tendencies
    passes["direction"] = np.degrees(np.arctan2(
        passes["end_y"] - passes["y"],
        passes["end_x"] - passes["x"]
    ))
    forward_passes = passes[passes["end_x"] > passes["x"]]
    backward_passes = passes[passes["end_x"] < passes["x"]]
    sideways_passes = passes[abs(passes["end_x"] - passes["x"]) < 5]

    # Vertical progression
    avg_x_start = passes["x"].mean()
    avg_x_end = passes["end_x"].mean()
    vertical_progression = avg_x_end - avg_x_start

    # Territory
    territory_pct = (passes["x"] > 60).mean()  # % of passes in opp half

    # Width
    avg_width = passes["y"].std()

    # Pass length distribution
    avg_pass_length = passes["pass_length"].mean() if "pass_length" in passes else 0
    long_ball_pct = (passes["pass_length"] > 32).mean() if "pass_length" in passes else 0
    short_pass_pct = (passes["pass_length"] < 15).mean() if "pass_length" in passes else 0

    # Final third entries
    final_third_entries = (
        (passes["end_x"] > 80) & (passes["x"] <= 80)
    ).sum()
    final_third_rate = final_third_entries / max(total_passes, 1)

    # Shot metrics
    shots_total = len(shots)
    shot_rate = shots_total / max(total_passes, 1)

    # Flank bias (left = low y, right = high y)
    right_flank = (passes["y"] > 53).mean()
    left_flank  = (passes["y"] < 27).mean()
    central     = ((passes["y"] >= 27) & (passes["y"] <= 53)).mean()

    # Build-up phase (passes in own half)
    buildup_passes = passes[passes["x"] < 60]
    buildup_ratio  = len(buildup_passes) / max(total_passes, 1)

    # Tempo (passes per minute)
    if "minute" in passes.columns:
        duration = passes["minute"].max() - passes["minute"].min()
        tempo = total_passes / max(duration, 1)
    else:
        tempo = 0

    return {
        "team": team_name,
        "total_passes": total_passes,
        "completion_rate": round(completion_rate, 3),
        "forward_pass_pct": round(len(forward_passes) / total_passes, 3),
        "backward_pass_pct": round(len(backward_passes) / total_passes, 3),
        "sideways_pass_pct": round(len(sideways_passes) / total_passes, 3),
        "long_ball_pct": round(long_ball_pct, 3),
        "short_pass_pct": round(short_pass_pct, 3),
        "avg_pass_length": round(avg_pass_length, 2),
        "vertical_progression": round(vertical_progression, 2),
        "territory_pct": round(territory_pct, 3),
        "avg_width": round(avg_width, 2),
        "right_flank_pct": round(right_flank, 3),
        "left_flank_pct": round(left_flank, 3),
        "central_pct": round(central, 3),
        "buildup_ratio": round(buildup_ratio, 3),
        "final_third_rate": round(final_third_rate, 3),
        "shot_rate": round(shot_rate, 4),
        "tempo": round(tempo, 2),
    }


# ─── Style classifier (rule-based + K-Means) ─────────────────────────────────

STYLE_RULES = {
    "Tiki-Taka (Possession)": {
        "short_pass_pct": (">", 0.50),
        "completion_rate": (">", 0.82),
        "territory_pct": (">", 0.55),
    },
    "Counter-Attack": {
        "long_ball_pct": (">", 0.25),
        "vertical_progression": (">", 3.5),
        "forward_pass_pct": (">", 0.55),
    },
    "Direct / Long-Ball": {
        "long_ball_pct": (">", 0.35),
        "avg_pass_length": (">", 25),
        "short_pass_pct": ("<", 0.30),
    },
    "High-Press": {
        "territory_pct": (">", 0.60),
        "tempo": (">", 5),
    },
    "Balanced": {},  # fallback
}

STYLE_DESCRIPTIONS = {
    "Tiki-Taka (Possession)": "Patient, short-passing build-up through midfield with high ball retention.",
    "Counter-Attack": "Quick vertical transitions, exploiting space behind the defence.",
    "Direct / Long-Ball": "Bypasses midfield with direct long balls to attackers.",
    "High-Press": "Aggressive high territorial pressing to win back possession quickly.",
    "Balanced": "Mixed approach combining elements of possession and direct play.",
}

def _rule_match(features: dict, rules: dict) -> int:
    score = 0
    for key, (op, threshold) in rules.items():
        val = features.get(key, 0)
        if op == ">" and val > threshold:
            score += 1
        elif op == "<" and val < threshold:
            score += 1
    return score


def classify_tactical_style(features: dict) -> tuple[str, str, dict]:
    """
    Returns (style_name, description, match_scores).
    """
    scores = {style: _rule_match(features, rules) for style, rules in STYLE_RULES.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "Balanced"
    return best, STYLE_DESCRIPTIONS[best], scores


# ─── Multi-team clustering (K-Means + PCA) ───────────────────────────────────

FEATURE_COLS = [
    "completion_rate", "forward_pass_pct", "long_ball_pct",
    "short_pass_pct", "avg_pass_length", "vertical_progression",
    "territory_pct", "avg_width", "right_flank_pct", "left_flank_pct",
    "central_pct", "buildup_ratio", "final_third_rate", "tempo"
]

def cluster_team_styles(all_features: list[dict], n_clusters: int = 3) -> pd.DataFrame:
    """
    K-Means clustering of team tactical styles.
    all_features: list of feature dicts from extract_tactical_features()
    Returns dataframe with cluster labels + PCA coords.
    """
    df = pd.DataFrame(all_features)
    if len(df) < n_clusters:
        n_clusters = max(2, len(df))

    # Use only numeric feature cols present
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # PCA for 2D viz
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]
    df["pca_var_explained"] = sum(pca.explained_variance_ratio_)

    cluster_labels = {
        0: "Possession-Based",
        1: "Direct / Counter",
        2: "High-Intensity Press",
        3: "Mixed / Adaptive",
    }
    df["cluster_label"] = df["cluster"].map(cluster_labels).fillna("Cluster " + df["cluster"].astype(str))

    return df


# ─── Possession chain analysis ───────────────────────────────────────────────

def extract_possession_chains(events_df: pd.DataFrame, team_name: str,
                               min_length: int = 3) -> list[dict]:
    """
    Extract possession chains — sequences of consecutive passes by same team.
    """
    team_events = events_df[events_df["team"] == team_name].sort_values(
        ["period", "minute", "second"]
    ).copy()

    chains = []
    current_chain = []

    for _, row in team_events.iterrows():
        if row["type"] == "Pass":
            current_chain.append(row)
        else:
            if len(current_chain) >= min_length:
                chain_df = pd.DataFrame(current_chain)
                start_x = chain_df["x"].iloc[0]
                end_x   = chain_df["end_x"].iloc[-1]
                chains.append({
                    "length": len(chain_df),
                    "start_x": round(start_x, 1),
                    "end_x": round(end_x, 1),
                    "x_gain": round(end_x - start_x, 1),
                    "start_zone": _zone(start_x),
                    "end_zone": _zone(end_x),
                    "sequence": " → ".join(chain_df["player_name"].str.split().str[-1].tolist()),
                    "period": int(chain_df["period"].iloc[0]),
                    "minute": int(chain_df["minute"].iloc[0]),
                    "reaches_final_third": end_x > 80,
                    "avg_y": round(chain_df["y"].mean(), 1),
                    "flank": "Left" if chain_df["y"].mean() < 30 else (
                             "Right" if chain_df["y"].mean() > 50 else "Central"),
                })
            current_chain = []

    chains.sort(key=lambda x: x["x_gain"], reverse=True)
    return chains


def _zone(x: float) -> str:
    if x < 40:   return "Defensive Third"
    elif x < 80: return "Middle Third"
    else:         return "Attacking Third"


def summarise_chains(chains: list[dict]) -> dict:
    if not chains:
        return {}
    df = pd.DataFrame(chains)
    return {
        "total_chains": len(df),
        "avg_chain_length": round(df["length"].mean(), 1),
        "max_chain_length": int(df["length"].max()),
        "chains_reaching_final_third": int(df["reaches_final_third"].sum()),
        "avg_x_gain": round(df["x_gain"].mean(), 1),
        "dominant_flank": df["flank"].mode()[0] if len(df) > 0 else "N/A",
        "top_chain": df.sort_values("length", ascending=False).iloc[0]["sequence"],
    }


# ─── Formation detection ─────────────────────────────────────────────────────

FORMATION_TEMPLATES = {
    "4-3-3": [(1,40),(20,65),(20,48),(20,32),(20,15),(40,55),(40,40),(40,25),(80,65),(65,40),(80,15)],
    "4-4-2": [(1,40),(20,65),(20,48),(20,32),(20,15),(50,65),(50,48),(50,32),(50,15),(90,55),(90,25)],
    "3-5-2": [(1,40),(20,58),(20,40),(20,22),(40,68),(40,52),(40,40),(40,28),(40,12),(90,55),(90,25)],
    "4-2-3-1":[(1,40),(20,65),(20,48),(20,32),(20,15),(38,52),(38,28),(60,65),(60,40),(60,15),(90,40)],
    "5-3-2": [(1,40),(20,72),(20,56),(20,40),(20,24),(20,8),(45,55),(45,40),(45,25),(85,55),(85,25)],
}

def detect_formation(events_df: pd.DataFrame, team_name: str) -> tuple[str, float]:
    """
    Estimate team formation by matching average player positions to templates.
    Returns (formation_string, confidence_score).
    """
    passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["team"] == team_name)
    ].copy()

    if passes.empty:
        return "Unknown", 0.0

    avg_pos = passes.groupby("player_name")[["x", "y"]].mean()
    if len(avg_pos) < 5:
        return "Unknown", 0.0

    actual_positions = avg_pos.values
    # Sort by x position (GK first)
    actual_sorted = actual_positions[np.argsort(actual_positions[:, 0])]

    best_formation, best_score = "4-3-3", 0.0
    for formation, template in FORMATION_TEMPLATES.items():
        template_arr = np.array(template)
        template_sorted = template_arr[np.argsort(template_arr[:, 0])]

        n = min(len(actual_sorted), len(template_sorted))
        dists = np.linalg.norm(
            actual_sorted[:n] - template_sorted[:n], axis=1
        )
        score = 1 / (1 + dists.mean() / 20)  # normalised similarity

        if score > best_score:
            best_score = score
            best_formation = formation

    return best_formation, round(best_score, 3)


# ─── Tactical shift detection ────────────────────────────────────────────────

def detect_tactical_shifts(events_df: pd.DataFrame, team_name: str,
                            window: int = 15) -> pd.DataFrame:
    """
    Detect mid-match tactical changes by comparing rolling feature windows.
    Uses Isolation Forest anomaly detection on rolling metrics.
    """
    passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["team"] == team_name)
    ].sort_values(["minute"]).copy()

    if len(passes) < window * 2:
        return pd.DataFrame()

    passes["minute_bin"] = (passes["minute"] // 5) * 5
    grouped = passes.groupby("minute_bin").agg(
        pass_count=("type", "count"),
        avg_x=("x", "mean"),
        avg_y=("y", "mean"),
        avg_length=("pass_length", "mean"),
        fwd_pct=("end_x", lambda s: (s > passes.loc[s.index, "x"]).mean()),
        territory=("x", lambda s: (s > 60).mean()),
    ).reset_index()

    if len(grouped) < 4:
        return grouped

    # Isolation Forest for anomaly detection
    feature_cols = ["pass_count", "avg_x", "avg_length", "fwd_pct", "territory"]
    X = grouped[feature_cols].fillna(0)

    iso = IsolationForest(contamination=0.15, random_state=42)
    grouped["is_shift"] = iso.fit_predict(X) == -1

    # Rolling change magnitude
    for col in ["avg_x", "avg_length", "territory"]:
        grouped[f"{col}_change"] = grouped[col].diff().abs()

    grouped["shift_score"] = (
        grouped["avg_x_change"].fillna(0) * 0.4 +
        grouped["territory_change"].fillna(0) * 50 +
        grouped["avg_length_change"].fillna(0) * 0.3
    )
    grouped["shift_label"] = grouped.apply(
        lambda r: _label_shift(r), axis=1
    )
    return grouped


def _label_shift(row) -> str:
    if not row.get("is_shift", False):
        return "Normal"
    if row.get("avg_x_change", 0) > 8:
        return "⬆️ Pushed Higher"
    if row.get("avg_x_change", 0) > 5:
        return "⬇️ Dropped Deeper"
    if row.get("territory_change", 0) > 0.15:
        return "🔴 More Territorial"
    if row.get("avg_length_change", 0) > 5:
        return "⚡ More Direct"
    return "🔄 Tactical Shift"


# ─── Pressing intensity ───────────────────────────────────────────────────────

def calculate_pressing_intensity(events_df: pd.DataFrame, team_name: str) -> dict:
    """
    Estimate pressing intensity using PPDA-style metric
    (passes allowed per defensive action proxy).
    """
    team_passes = events_df[
        (events_df["type"] == "Pass") &
        (events_df["team"] == team_name)
    ]
    high_press = team_passes[team_passes["x"] > 60]
    pressing_ratio = len(high_press) / max(len(team_passes), 1)

    return {
        "pressing_ratio": round(pressing_ratio, 3),
        "high_press_passes": len(high_press),
        "pressing_intensity": (
            "Elite Press" if pressing_ratio > 0.6 else
            "High Press" if pressing_ratio > 0.45 else
            "Mid-Block" if pressing_ratio > 0.30 else
            "Low Block"
        )
    }
