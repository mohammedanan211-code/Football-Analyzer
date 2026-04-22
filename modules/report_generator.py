"""
report_generator.py
===================
Generates natural language tactical summary reports
from all computed metrics.
"""

from datetime import datetime


def generate_tactical_report(
    team_name: str,
    style: str,
    style_description: str,
    formation: str,
    features: dict,
    chain_summary: dict,
    pressing: dict,
    top_players: list[dict],
    metrics_df=None,
) -> str:
    """
    Generate a detailed natural language tactical report.
    """
    now = datetime.now().strftime("%B %d, %Y")

    flank = "left flank" if features.get("left_flank_pct", 0) > features.get("right_flank_pct", 0) \
            else "right flank" if features.get("right_flank_pct", 0) > features.get("left_flank_pct", 0) \
            else "central channels"

    progression = features.get("vertical_progression", 0)
    prog_desc = "aggressively advancing the ball" if progression > 5 \
                else "building patiently" if progression > 2 \
                else "staying compact and cautious"

    territory = features.get("territory_pct", 0.5)
    terr_desc = "dominant in the opponent's half" if territory > 0.55 \
                else "balanced across both halves" if territory > 0.40 \
                else "sitting deeper, defending first"

    pass_len = features.get("avg_pass_length", 20)
    pass_desc = "long, direct passes" if pass_len > 28 \
                else "predominantly short combinations" if pass_len < 18 \
                else "a mix of short and medium-range passing"

    playmakers = []
    if metrics_df is not None and not metrics_df.empty:
        top3 = metrics_df[metrics_df["is_playmaker"] == True]["player"].tolist()
        playmakers = top3

    pm_text = ""
    if playmakers:
        pm_text = f"The primary game-controllers are **{', '.join(playmakers[:2])}**, " \
                  f"who dominate ball circulation through their high betweenness centrality."

    chain_text = ""
    if chain_summary:
        chain_text = (
            f"Possession chains average **{chain_summary.get('avg_chain_length', 0):.1f} passes**, "
            f"with the longest sequence being {chain_summary.get('max_chain_length', 0)} passes. "
            f"{chain_summary.get('chains_reaching_final_third', 0)} chains successfully "
            f"penetrated the final third, and the preferred buildup flank is the "
            f"**{chain_summary.get('dominant_flank', 'central')}** side."
        )

    pressing_text = ""
    if pressing:
        pressing_text = (
            f"The team's pressing intensity is rated **{pressing.get('pressing_intensity', 'N/A')}**, "
            f"with {pressing.get('pressing_ratio', 0)*100:.0f}% of passes occurring in the "
            f"opponent's half."
        )

    report = f"""# ⚽ Tactical Analysis Report — {team_name}
*Generated: {now}*

---

## 🎯 Executive Summary
{team_name} employs a **{style}** approach. {style_description}
The team was identified in a **{formation}** formation, {prog_desc} and primarily attacking through the **{flank}**.

---

## 📊 Passing Profile
- **Total Passes Analysed:** {features.get('total_passes', 0)}
- **Completion Rate:** {features.get('completion_rate', 0)*100:.1f}%
- **Average Pass Length:** {features.get('avg_pass_length', 0):.1f} metres
- **Long Ball %:** {features.get('long_ball_pct', 0)*100:.1f}%
- **Short Pass %:** {features.get('short_pass_pct', 0)*100:.1f}%
- **Vertical Progression per pass:** +{features.get('vertical_progression', 0):.1f}m

The team favours **{pass_desc}**, and is **{terr_desc}** ({territory*100:.0f}% of actions in opp. half).

---

## 🧩 Tactical Patterns
{chain_text}

{pressing_text}

### Flank Distribution
| Zone         | % of Activity |
|--------------|---------------|
| Left Flank   | {features.get('left_flank_pct', 0)*100:.0f}% |
| Central      | {features.get('central_pct', 0)*100:.0f}% |
| Right Flank  | {features.get('right_flank_pct', 0)*100:.0f}% |

---

## 🌟 Key Players
{pm_text}

"""

    if top_players:
        report += "| Player | Role | Passes | Completion | Influence |\n"
        report += "|--------|------|--------|------------|-----------|\n"
        for p in top_players[:5]:
            report += (
                f"| {p.get('Player', p.get('player', 'N/A'))} "
                f"| {p.get('Role', p.get('role', 'N/A'))} "
                f"| {p.get('Passes', p.get('total_passes', 0))} "
                f"| {p.get('Completion %', p.get('completion_pct', 0))}% "
                f"| {p.get('influence', '-')} |\n"
            )

    report += f"""
---

## 💡 Scout Recommendations
Based on the analysis:

1. **{team_name}** can be disrupted by pressing their {flank} buildup triggers.
2. Against this team, focus defensive shape on cutting **{flank}** penetration.
3. Their **{pressing.get('pressing_intensity', 'N/A')}** suggests {'high energy pressing — target late match fatigue' if 'High' in pressing.get('pressing_intensity','') else 'a structured block — patience and wide play can exploit gaps'}.
4. {f"Counter the playmaking of **{', '.join(playmakers[:1])}** by applying man-marking or midfield shadow coverage." if playmakers else "Target the midfield press recovery time."}

---
*This report was auto-generated by the Football Strategy Analyzer.*
"""

    return report
