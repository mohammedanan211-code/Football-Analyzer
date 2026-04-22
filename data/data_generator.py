"""
data_generator.py
=================
Generates realistic StatsBomb-style football event data for demo/testing.
Replace with real StatsBomb data using:
    from statsbombpy import sb
    events = sb.events(match_id=<match_id>)
"""

import numpy as np
import pandas as pd
import random

# ─── Pitch dimensions (StatsBomb uses 120x80) ───────────────────────────────
PITCH_LENGTH = 120
PITCH_WIDTH  = 80

# ─── Team rosters ────────────────────────────────────────────────────────────
TEAMS = {
    "FC Barcelona": {
        "style": "possession",
        "players": [
            {"id": 1,  "name": "GK joan garcia",  "position": "GK",  "x_base": 5,   "y_base": 40},
            {"id": 2,  "name": "RB Jules kounde",      "position": "RB",  "x_base": 25,  "y_base": 65},
            {"id": 3,  "name": "CB Torres",      "position": "CB",  "x_base": 20,  "y_base": 50},
            {"id": 4,  "name": "CB cubarsi",       "position": "CB",  "x_base": 20,  "y_base": 30},
            {"id": 5,  "name": "LB joao cancelo",     "position": "LB",  "x_base": 25,  "y_base": 15},
            {"id": 6,  "name": "CDM De jong",      "position": "CDM", "x_base": 40,  "y_base": 40},
            {"id": 7,  "name": "CM Pedri",       "position": "CM",  "x_base": 55,  "y_base": 55},
            {"id": 8,  "name": "CM Fermin lopez",        "position": "CM",  "x_base": 55,  "y_base": 25},
            {"id": 9,  "name": "RW Lamine yamal",      "position": "RW",  "x_base": 75,  "y_base": 65},
            {"id": 10, "name": "LW Raphina",     "position": "CAM", "x_base": 70,  "y_base": 40},
            {"id": 11, "name": "ST Lewandowski",       "position": "ST",  "x_base": 95,  "y_base": 40},
        ]
    },
    "Athletico Madrid": {
        "style": "counter",
        "players": [
            {"id": 21, "name": "GK oblak",      "position": "GK",  "x_base": 5,   "y_base": 40},
            {"id": 22, "name": "RB Molina",     "position": "RB",  "x_base": 20,  "y_base": 68},
            {"id": 23, "name": "CB Gimnez",      "position": "CB",  "x_base": 18,  "y_base": 52},
            {"id": 24, "name": "CB Le normand",       "position": "CB",  "x_base": 18,  "y_base": 28},
            {"id": 25, "name": "LB simione",   "position": "LB",  "x_base": 20,  "y_base": 12},
            {"id": 26, "name": "CDM Koke",    "position": "CDM", "x_base": 35,  "y_base": 40},
            {"id": 27, "name": "CM Antoine Griezman",       "position": "CM",  "x_base": 50,  "y_base": 55},
            {"id": 28, "name": "CM Julian Alvarez ",        "position": "CM",  "x_base": 50,  "y_base": 25},
            {"id": 29, "name": "RW Almanda",       "position": "RW",  "x_base": 80,  "y_base": 68},
            {"id": 30, "name": "LW Lookman",     "position": "LW",  "x_base": 80,  "y_base": 12},
            {"id": 31, "name": "ST sorloth",      "position": "ST",  "x_base": 98,  "y_base": 40},
        ]
    },
    "Real Madrid Cf": {
        "style": "direct",
        "players": [
            {"id": 41, "name": "GK courtois",     "position": "GK",  "x_base": 5,   "y_base": 40},
            {"id": 42, "name": "RB Trent alexander arnold",       "position": "RB",  "x_base": 22,  "y_base": 66},
            {"id": 43, "name": "CB Eder Militao",       "position": "CB",  "x_base": 18,  "y_base": 50},
            {"id": 44, "name": "CB Rudiger",       "position": "CB",  "x_base": 18,  "y_base": 30},
            {"id": 45, "name": "LB alvaro careras",       "position": "LB",  "x_base": 22,  "y_base": 14},
            {"id": 46, "name": "CDM  camavinga",      "position": "CM",  "x_base": 45,  "y_base": 55},
            {"id": 47, "name": "CM valverde",      "position": "CM",  "x_base": 45,  "y_base": 25},
            {"id": 48, "name": "RW Rodrygo",      "position": "RW",  "x_base": 72,  "y_base": 66},
            {"id": 49, "name": "LW vinicius junior",       "position": "LW",  "x_base": 72,  "y_base": 14},
            {"id": 50, "name": "ST mbaape",      "position": "ST",  "x_base": 92,  "y_base": 50},
            {"id": 51, "name": "CM jude bellingham",      "position": "ST",  "x_base": 92,  "y_base": 30},
        ]
    }
}


def _jitter(base, std=8, low=0, high=120):
    return float(np.clip(np.random.normal(base, std), low, high))


def generate_match_events(team_name: str, n_passes: int = 400, match_id: int = 1,
                          half_override: int = None) -> pd.DataFrame:
    """
    Generate a realistic event dataframe for a single team in a match.
    Mimics the structure of StatsBomb event data.
    """
    team = TEAMS[team_name]
    players = team["players"]
    style = team["style"]
    rng = np.random.default_rng(match_id * 42)
    rows = []
    minute = 1

    # ── Pass weight matrix by style ──────────────────────────────────────────
    def pass_weight(passer_pos, receiver_pos):
        positions = [p["position"] for p in players]
        pi, ri = positions.index(passer_pos), positions.index(receiver_pos)
        if style == "possession":
            # Short passes, midfield dominant
            if abs(pi - ri) <= 2:
                return 5.0
            return 1.0
        elif style == "counter":
            # Fast vertical passes
            if ri > pi:
                return 4.0
            return 0.5
        else:  # direct / long-ball
            # Favour long passes to strikers
            if receiver_pos in ("ST", "RW", "LW") and passer_pos in ("GK", "CB", "CDM"):
                return 6.0
            return 1.0

    for i in range(n_passes):
        minute += rng.integers(0, 2)
        minute = min(minute, 90)
        half = 1 if minute <= 45 else 2
        if half_override:
            half = half_override

        # Pick passer (weighted by involvement)
        if style == "possession":
            passer_weights = [3 if p["position"] in ("CM","CDM","CAM") else 1 for p in players]
        elif style == "counter":
            passer_weights = [4 if p["position"] in ("RW","LW","ST") else 1 for p in players]
        else:
            passer_weights = [5 if p["position"] in ("CB","GK","CDM") else 1 for p in players]

        passer_weights = np.array(passer_weights, dtype=float)
        passer_weights /= passer_weights.sum()
        passer = players[rng.choice(len(players), p=passer_weights)]

        # Pick receiver
        rec_w = np.array([
            pass_weight(passer["position"], p["position"]) if p["id"] != passer["id"] else 0
            for p in players
        ], dtype=float)
        if rec_w.sum() == 0:
            rec_w = np.ones(len(players)); rec_w[players.index(passer)] = 0
        rec_w /= rec_w.sum()
        receiver = players[rng.choice(len(players), p=rec_w)]

        # Positions with noise
        px = _jitter(passer["x_base"], std=10)
        py = _jitter(passer["y_base"], std=8, low=0, high=PITCH_WIDTH)
        ex = _jitter(receiver["x_base"], std=10)
        ey = _jitter(receiver["y_base"], std=8, low=0, high=PITCH_WIDTH)

        outcome = "Complete" if rng.random() > 0.15 else "Incomplete"

        rows.append({
            "match_id": match_id,
            "minute": int(minute),
            "second": int(rng.integers(0, 60)),
            "period": half,
            "type": "Pass",
            "player_id": passer["id"],
            "player_name": passer["name"],
            "position": passer["position"],
            "team": team_name,
            "x": round(px, 2),
            "y": round(py, 2),
            "end_x": round(ex, 2),
            "end_y": round(ey, 2),
            "recipient_id": receiver["id"],
            "recipient_name": receiver["name"],
            "outcome": outcome,
            "pass_length": round(np.sqrt((ex-px)**2 + (ey-py)**2), 2),
        })

        # Occasionally add shots
        if rng.random() < 0.03:
            shooter = players[-1]  # ST
            sx = _jitter(shooter["x_base"], std=5, low=80, high=119)
            sy = _jitter(shooter["y_base"], std=5, low=0, high=PITCH_WIDTH)
            is_goal = rng.random() < 0.2
            rows.append({
                "match_id": match_id,
                "minute": int(minute),
                "second": int(rng.integers(0, 60)),
                "period": half,
                "type": "Shot",
                "player_id": shooter["id"],
                "player_name": shooter["name"],
                "position": shooter["position"],
                "team": team_name,
                "x": round(sx, 2),
                "y": round(sy, 2),
                "end_x": 120.0,
                "end_y": round(_jitter(40, 3, 32, 48), 2),
                "recipient_id": None,
                "recipient_name": None,
                "outcome": "Goal" if is_goal else "Saved",
                "pass_length": None,
            })

    df = pd.DataFrame(rows)
    df["index"] = range(len(df))
    return df


def generate_all_teams(n_passes: int = 400) -> dict:
    """Generate event data for all teams."""
    return {team: generate_match_events(team, n_passes=n_passes) for team in TEAMS}


def get_team_players(team_name: str) -> list:
    return TEAMS[team_name]["players"]


def get_all_team_names() -> list:
    return list(TEAMS.keys())


if __name__ == "__main__":
    data = generate_all_teams()
    for team, df in data.items():
        print(f"{team}: {len(df)} events")
        print(df.head(3), "\n")
