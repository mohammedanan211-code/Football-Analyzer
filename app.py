"""
app.py  —  Football Team Strategy Analyzer
==========================================
Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Local imports ────────────────────────────────────────────────────────────
from data.data_generator import (
    generate_all_teams, get_all_team_names, get_team_players, TEAMS
)
from modules.passing_network import (
    build_passing_network, plot_passing_network, get_key_corridors
)
from modules.tactical_analysis import (
    extract_tactical_features, classify_tactical_style,
    cluster_team_styles, extract_possession_chains, summarise_chains,
    detect_formation, detect_tactical_shifts, calculate_pressing_intensity
)
from modules.heatmap_analysis import (
    plot_team_heatmap, plot_player_heatmap, plot_shot_map,
    plot_progressive_passes, plot_zone_control
)
from modules.player_analysis import (
    get_team_player_stats, extract_player_features, plot_player_radar,
    plot_team_radar, compute_influence_index, classify_player_role
)
from modules.report_generator import generate_tactical_report


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Strategy Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1a2d 50%, #0a1520 100%);
}

h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px;
}

.metric-card {
    background: linear-gradient(135deg, #0d1a2d, #1a2a40);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'Rajdhani', sans-serif;
    color: #FFD700;
}
.metric-label {
    font-size: 0.75rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.style-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1a3a5c, #0d2a4a);
    border: 1px solid #4fc3f7;
    color: #4fc3f7;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px 0;
}

.insight-box {
    background: linear-gradient(135deg, #0d1a2d, #1a2a40);
    border-left: 3px solid #FFD700;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #ccd5e0;
}

.section-header {
    background: linear-gradient(90deg, #1a2a40, transparent);
    border-left: 4px solid #4fc3f7;
    padding: 8px 16px;
    margin: 16px 0 8px 0;
    border-radius: 0 4px 4px 0;
}

div[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif;
    color: #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Football Strategy Analyzer")
    st.markdown("*Powered by Graph Theory, ML & Spatial Analysis*")
    st.divider()

    st.markdown("### 🔧 Configuration")
    selected_team = st.selectbox("Select Team", get_all_team_names(), index=0)
    compare_mode  = st.checkbox("Compare Two Teams", value=False)
    if compare_mode:
        other_team = st.selectbox(
            "Compare With",
            [t for t in get_all_team_names() if t != selected_team],
            index=0
        )

    st.divider()
    n_passes = st.slider("Simulated Passes (Data Size)", 200, 800, 450, step=50)
    min_passes_network = st.slider("Min Passes for Network Edge", 1, 8, 3)

#     st.divider()
#     st.markdown("### 📂 Use Real StatsBomb Data")
#     st.code("""
# # Install: pip install statsbombpy
# from statsbombpy import sb
# events = sb.events(match_id=3773369)
#     """, language="python")
#     st.caption("Replace synthetic data with real StatsBomb events.")

    st.divider()
    st.markdown("### 📤 Data Upload")
    
    if 'data_unlocked' not in st.session_state:
        st.session_state.data_unlocked = False

    u_sidebar = st.file_uploader("Upload CSV Data", type=["csv"], key="sidebar_uploader")
    if u_sidebar:
        st.session_state.data_unlocked = True

    if st.session_state.data_unlocked:
        st.success("✅ Analysis Unlocked")
        if st.button("🔄 Reset / Upload New"):
            st.session_state.data_unlocked = False
            st.rerun()
            
        st.divider()
        st.markdown("### 📑 Navigation")
        page = st.radio("View", [
            "🏠 Overview",
            "🕸️ Passing Network",
            "🔥 Heatmaps",
            "⚡ Tactical Analysis",
            "👤 Player Analysis",
            "🆚 Team Comparison",
            "📄 Full Report",
        ])
    else:
        st.info("Please upload a file to unlock the dashboard.")


# ─── Main Logic: Upload Gate ──────────────────────────────────────────────────

if not st.session_state.get('data_unlocked', False):
    st.markdown("""
    <div style="text-align: center; padding: 60px 0 20px 0;">
        <h1 style="font-size: 4rem; color: #FFD700; margin-bottom: 0; font-family: 'Rajdhani', sans-serif;">⚽ TACTICAL ANALYZER</h1>
        <p style="font-size: 1.1rem; color: #4fc3f7; letter-spacing: 4px; text-transform: uppercase; font-weight: 600;">Advanced Football Spatial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_l, col_m, col_r = st.columns([1, 1.8, 1])
    with col_m:
        st.markdown("""
        <div class="metric-card" style="border: 1px solid #4fc3f7; background: rgba(13, 26, 45, 0.8);">
            <h3 style="color: #ffffff; margin-top: 0;">Ready to Analyze?</h3>
            <p style="color: #8899aa; font-size: 0.95rem;">Upload your match events CSV to unlock the full tactical dashboard, including passing networks, influence maps, and player performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        main_upload = st.file_uploader("Drop your match data CSV here", type=["csv"], key="main_uploader", label_visibility="collapsed")
        
        if main_upload:
            st.session_state.data_unlocked = True
            st.success("✅ Analysis Unlocked! Initializing systems...")
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Pro Tip:** Use the `dummy_data.csv` file in the project directory to explore the features immediately.")
        
    st.divider()
    st.markdown("<div style='text-align: center; color: #445566; font-size: 0.8rem;'>POWERED BY GRAPH THEORY & SPATIAL DATA SCIENCE</div>", unsafe_allow_html=True)
    st.stop()


# ─── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(n):
    return generate_all_teams(n_passes=n)

with st.spinner("⚙️ Loading match data..."):
    all_data = load_data(n_passes)

events = all_data[selected_team]
if compare_mode:
    events_other = all_data[other_team]


# ─── Computed metrics (cached per team) ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_team_metrics(team_name: str, n: int):
    ev = all_data[team_name]
    G, node_pos, metrics_df = build_passing_network(ev, min_passes=min_passes_network)
    features = extract_tactical_features(ev, team_name)
    style, style_desc, style_scores = classify_tactical_style(features)
    formation, conf = detect_formation(ev, team_name)
    chains = extract_possession_chains(ev, team_name)
    chain_summary = summarise_chains(chains)
    pressing = calculate_pressing_intensity(ev, team_name)
    shifts_df = detect_tactical_shifts(ev, team_name)
    return {
        "G": G, "node_pos": node_pos, "metrics_df": metrics_df,
        "features": features, "style": style, "style_desc": style_desc,
        "style_scores": style_scores, "formation": formation,
        "formation_conf": conf, "chains": chains,
        "chain_summary": chain_summary, "pressing": pressing,
        "shifts_df": shifts_df,
    }

with st.spinner(f"🧠 Analysing {selected_team}..."):
    m = compute_team_metrics(selected_team, n_passes)

if compare_mode:
    with st.spinner(f"🧠 Analysing {other_team}..."):
        m2 = compute_team_metrics(other_team, n_passes)
    f2 = m2["features"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    if not compare_mode:
        st.markdown(f"# ⚽ {selected_team}")
        st.markdown(f"<div class='style-badge'>🎯 {m['style']}</div> "
                    f"<div class='style-badge'>📐 Formation: {m['formation']} "
                    f"({m['formation_conf']*100:.0f}% confidence)</div> "
                    f"<div class='style-badge'>🔥 {m['pressing']['pressing_intensity']}</div>",
                    unsafe_allow_html=True)

        st.divider()
        f = m["features"]

        # ── Key metrics row ───────────────────────────────────────────────────────
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        with col1: st.metric("Total Passes", f.get("total_passes", 0))
        with col2: st.metric("Completion %", f"{f.get('completion_rate',0)*100:.1f}%")
        with col3: st.metric("Avg Pass Length", f"{f.get('avg_pass_length',0):.1f}m")
        with col4: st.metric("Territory %", f"{f.get('territory_pct',0)*100:.1f}%")
        with col5: st.metric("Avg Chain", m["chain_summary"].get("avg_chain_length", "-"))
        with col6: st.metric("Pressing", m["pressing"]["pressing_intensity"])

        st.divider()
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("### 🕸️ Quick — Passing Network")
            fig = plot_passing_network(m["G"], m["node_pos"], m["metrics_df"], selected_team, figsize=(12,7))
            st.pyplot(fig); plt.close(fig)

        with col_right:
            st.markdown("### 📊 Style Signature")
            style_df = pd.DataFrame([{"Metric": k, "Value": f"{v*100:.1f}%"} for k, v in {
                "Possession (Territory)": f.get("territory_pct",0),
                "Short Passes": f.get("short_pass_pct",0),
                "Long Balls": f.get("long_ball_pct",0),
                "Forward Passes": f.get("forward_pass_pct",0),
            }.items()])
            st.dataframe(style_df, use_container_width=True, hide_index=True)
            st.markdown("### 🎯 Tactical Description")
            st.info(m["style_desc"])
    else:
        # Comparison Overview
        st.markdown(f"# 🆚 {selected_team} vs {other_team}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {selected_team}")
            st.markdown(f"<div class='style-badge'>🎯 {m['style']}</div> <div class='style-badge'>📐 {m['formation']}</div>", unsafe_allow_html=True)
            f = m["features"]
            st.metric("Total Passes", f.get("total_passes", 0))
            st.metric("Completion %", f"{f.get('completion_rate',0)*100:.1f}%")
            st.metric("Territory %", f"{f.get('territory_pct',0)*100:.1f}%")
            fig = plot_passing_network(m["G"], m["node_pos"], m["metrics_df"], selected_team, figsize=(8,6))
            st.pyplot(fig); plt.close(fig)
        
        with c2:
            st.markdown(f"### {other_team}")
            st.markdown(f"<div class='style-badge'>🎯 {m2['style']}</div> <div class='style-badge'>📐 {m2['formation']}</div>", unsafe_allow_html=True)
            f2 = m2["features"]
            st.metric("Total Passes", f2.get("total_passes", 0))
            st.metric("Completion %", f"{f2.get('completion_rate',0)*100:.1f}%")
            st.metric("Territory %", f"{f2.get('territory_pct',0)*100:.1f}%")
            fig2 = plot_passing_network(m2["G"], m2["node_pos"], m2["metrics_df"], other_team, figsize=(8,6))
            st.pyplot(fig2); plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Passing Network
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🕸️ Passing Network":
    if not compare_mode:
        st.markdown(f"# 🕸️ Passing Network — {selected_team}")
        st.caption("Node size = betweenness centrality | Gold = playmaker | Arrow thickness = pass frequency")
        fig = plot_passing_network(m["G"], m["node_pos"], m["metrics_df"], selected_team, figsize=(14, 9))
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.markdown(f"# 🕸️ Passing Network Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {selected_team}")
            fig = plot_passing_network(m["G"], m["node_pos"], m["metrics_df"], selected_team, figsize=(9, 8))
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            st.markdown(f"#### {other_team}")
            fig2 = plot_passing_network(m2["G"], m2["node_pos"], m2["metrics_df"], other_team, figsize=(9, 8))
            st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🌟 Network Metrics")
        if not m["metrics_df"].empty:
            display_cols = ["player", "betweenness_centrality", "pagerank",
                             "passes_sent", "passes_received", "is_playmaker"]
            display_df = m["metrics_df"][display_cols].copy()
            display_df.columns = ["Player", "Betweenness", "PageRank",
                                    "Sent", "Received", "Playmaker"]
            display_df["Betweenness"] = display_df["Betweenness"].round(3)
            display_df["PageRank"] = display_df["PageRank"].round(4)

            def highlight_playmaker(row):
                if row["Playmaker"]:
                    return ["background-color: #1a3a1a"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(highlight_playmaker, axis=1),
                use_container_width=True, hide_index=True
            )

    with c2:
        st.markdown("### 🛣️ Key Passing Corridors")
        corridors = get_key_corridors(m["G"], m["node_pos"], top_n=8)
        if not corridors.empty:
            st.dataframe(corridors, use_container_width=True, hide_index=True)

        st.markdown("### 📐 Network Stats")
        G = m["G"]
        if G.number_of_nodes() > 0:
            st.metric("Total Nodes (Players)", G.number_of_nodes())
            st.metric("Total Edges (Pass Links)", G.number_of_edges())
            try:
                import networkx as nx
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔥 Heatmaps":
    title = f"# 🔥 Heatmaps — {selected_team}" if not compare_mode else "# 🔥 Heatmap Comparison"
    st.markdown(title)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Team Pass Heatmap",
        "🎯 Shot Map",
        "⬆️ Progressive Passes",
        "👤 Player Heatmap",
    ])

    with tab1:
        st.caption("Shows where the team initiates passing actions across the pitch")
        if not compare_mode:
            fig = plot_team_heatmap(events, selected_team, "Pass", figsize=(13, 8))
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            c1, c2 = st.columns(2)
            with c1: st.pyplot(plot_team_heatmap(events, selected_team, "Pass", figsize=(9,7))); plt.close()
            with c2: st.pyplot(plot_team_heatmap(events_other, other_team, "Pass", figsize=(9,7))); plt.close()

    with tab2:
        st.caption("Shot locations, outcomes, and goal mouth diagram")
        if not compare_mode:
            fig = plot_shot_map(events, selected_team, figsize=(14, 8))
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            c1, c2 = st.columns(2)
            with c1: st.pyplot(plot_shot_map(events, selected_team, figsize=(9,7))); plt.close()
            with c2: st.pyplot(plot_shot_map(events_other, other_team, figsize=(9,7))); plt.close()

        shots = events[events["type"] == "Shot"]
        goals = shots[shots["outcome"] == "Goal"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Shots", len(shots))
        c2.metric("Goals", len(goals))
        c3.metric("Shot Conversion", f"{len(goals)/max(len(shots),1)*100:.0f}%")

    with tab3:
        st.caption("Passes that advance the ball ≥10m towards goal")
        if not compare_mode:
            fig = plot_progressive_passes(events, selected_team, figsize=(13, 8))
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            c1, c2 = st.columns(2)
            with c1: st.pyplot(plot_progressive_passes(events, selected_team, figsize=(9,7))); plt.close()
            with c2: st.pyplot(plot_progressive_passes(events_other, other_team, figsize=(9,7))); plt.close()

    with tab4:
        st.caption("Individual player position heatmap")
        players_in_team = events["player_name"].unique().tolist()
        selected_player = st.selectbox("Select Player", sorted(players_in_team))
        if selected_player:
            with st.spinner(f"Loading {selected_player}..."):
                fig = plot_player_heatmap(events, selected_player, figsize=(13, 8))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Tactical Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚡ Tactical Analysis":
    title = f"# ⚡ Tactical Analysis — {selected_team}" if not compare_mode else "# ⚡ Tactical Comparison"
    st.markdown(title)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Style Detection",
        "⛓️ Possession Chains",
        "🔄 Tactical Shifts",
        "📊 Multi-Team Clustering",
    ])

    with tab1:
        st.markdown("### 🎯 Detected Tactical Style")
        if not compare_mode:
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.markdown(f"<div class='style-badge' style='font-size:1.1rem;padding:10px 20px;'>"
                             f"🏆 {m['style']}</div>", unsafe_allow_html=True)
                st.markdown(f"\n{m['style_desc']}\n")
                st.metric("Formation", m["formation"], f"Conf: {m['formation_conf']*100:.0f}%")
                st.metric("Pressing", m["pressing"]["pressing_intensity"])
            with col2:
                feat_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in m["features"].items() if isinstance(v, (int, float))])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {selected_team}")
                st.markdown(f"<div class='style-badge'>🏆 {m['style']}</div>", unsafe_allow_html=True)
                st.info(m["style_desc"])
                st.metric("Formation", m["formation"])
                st.metric("Pressing", m["pressing"]["pressing_intensity"])
            with c2:
                st.markdown(f"#### {other_team}")
                st.markdown(f"<div class='style-badge'>🏆 {m2['style']}</div>", unsafe_allow_html=True)
                st.info(m2["style_desc"])
                st.metric("Formation", m2["formation"])
                st.metric("Pressing", m2["pressing"]["pressing_intensity"])

    with tab2:
        st.markdown("### ⛓️ Possession Chain Analysis")
        if not compare_mode:
            chains = m["chains"]
            if chains:
                cs = m["chain_summary"]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Chains", cs.get("total_chains", 0))
                c2.metric("Avg Length", cs.get("avg_chain_length", 0))
                c3.metric("Max Chain", cs.get("max_chain_length", 0))
                c4.metric("→ Final Third", cs.get("chains_reaching_final_third", 0))

                chains_df = pd.DataFrame(chains)
                cols_show = ["length", "start_zone", "end_zone", "x_gain", "flank", "reaches_final_third", "minute"]
                cols_show = [c for c in cols_show if c in chains_df.columns]
                st.dataframe(chains_df[cols_show].sort_values("x_gain", ascending=False).head(30), use_container_width=True, hide_index=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {selected_team}")
                cs = m["chain_summary"]
                st.metric("Avg Chain Length", cs.get("avg_chain_length", 0))
                st.metric("Final Third Entries", cs.get("chains_reaching_final_third", 0))
            with c2:
                st.markdown(f"#### {other_team}")
                cs2 = m2["chain_summary"]
                st.metric("Avg Chain Length", cs2.get("avg_chain_length", 0))
                st.metric("Final Third Entries", cs2.get("chains_reaching_final_third", 0))

            chains_df = pd.DataFrame(chains)
            cols_show = ["length", "start_zone", "end_zone", "x_gain",
                          "flank", "reaches_final_third", "sequence", "minute"]
            cols_show = [c for c in cols_show if c in chains_df.columns]

            st.markdown("#### 🔍 All Chains (sorted by x_gain)")
            st.dataframe(
                chains_df[cols_show].sort_values("x_gain", ascending=False).head(30),
                use_container_width=True, hide_index=True
            )

            # Chain length distribution
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1a2d")
            for ax in axes:
                ax.set_facecolor("#0d1a2d")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("#2a3a5c")

            axes[0].hist(chains_df["length"], bins=15, color="#4fc3f7", edgecolor="#0d1a2d", alpha=0.8)
            axes[0].set_title("Chain Length Distribution", color="white", fontsize=10)
            axes[0].set_xlabel("Passes per chain", color="#8899aa")
            axes[0].set_ylabel("Count", color="#8899aa")

            flank_counts = chains_df["flank"].value_counts() if "flank" in chains_df else pd.Series()
            if not flank_counts.empty:
                axes[1].pie(flank_counts.values, labels=flank_counts.index,
                             colors=["#ff6b6b","#4fc3f7","#FFD700"],
                             autopct="%1.0f%%", textprops={"color":"white"})
                axes[1].set_title("Chain Flank Distribution", color="white", fontsize=10)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab3:
        st.markdown("### 🔄 Mid-Match Tactical Shifts")
        shifts_df = m["shifts_df"]
        if not shifts_df.empty and len(shifts_df) > 2:
            shift_events = shifts_df[shifts_df.get("is_shift", pd.Series(False))]
            if not shift_events.empty:
                st.warning(f"⚠️ {len(shift_events)} tactical shift(s) detected!")
                for _, row in shift_events.iterrows():
                    st.markdown(
                        f"<div class='insight-box'>⏱️ Minute {int(row['minute_bin'])} — "
                        f"{row.get('shift_label','Shift')} "
                        f"(Score: {row.get('shift_score',0):.1f})</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.success("✅ No significant tactical shifts detected — consistent throughout.")

            # Tactical timeline
            fig, axes = plt.subplots(3, 1, figsize=(13, 8), facecolor="#0d1a2d", sharex=True)
            metrics_plot = [
                ("avg_x", "Avg X Position", "#4fc3f7"),
                ("territory", "Territory %", "#FFD700"),
                ("avg_length", "Avg Pass Length", "#ff6b6b"),
            ]
            for ax, (col, label, color) in zip(axes, metrics_plot):
                if col in shifts_df.columns:
                    ax.set_facecolor("#0d1a2d")
                    ax.plot(shifts_df["minute_bin"], shifts_df[col],
                             color=color, lw=2)
                    ax.fill_between(shifts_df["minute_bin"], shifts_df[col],
                                     alpha=0.15, color=color)
                    if not shift_events.empty:
                        for _, sr in shift_events.iterrows():
                            ax.axvline(sr["minute_bin"], color="#FF4500",
                                        lw=1.5, linestyle="--", alpha=0.7)
                    ax.set_ylabel(label, color="white", fontsize=8)
                    ax.tick_params(colors="white", labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_color("#2a3a5c")

            axes[-1].set_xlabel("Match Minute", color="white")
            fig.suptitle(f"{selected_team} — Tactical Timeline",
                          color="white", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Not enough data for shift detection (need >4 time bins).")

    with tab4:
        st.markdown("### 📊 Multi-Team Tactical Clustering (K-Means + PCA)")
        all_features = []
        for tn in get_all_team_names():
            f = extract_tactical_features(all_data[tn], tn)
            if f:
                all_features.append(f)

        if len(all_features) >= 2:
            cluster_df = cluster_team_styles(all_features, n_clusters=min(3, len(all_features)))

            # PCA scatter plot
            fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d1a2d")
            ax.set_facecolor("#0d1a2d")
            colors = ["#FFD700","#4fc3f7","#ff6b6b","#90ee90"]
            for i, row in cluster_df.iterrows():
                c = colors[int(row["cluster"]) % len(colors)]
                ax.scatter(row["pca_x"], row["pca_y"], s=220, c=c,
                            edgecolors="white", lw=1.5, zorder=5)
                ax.annotate(row["team"], (row["pca_x"], row["pca_y"]),
                             textcoords="offset points", xytext=(8, 5),
                             color="white", fontsize=9, fontweight="bold")

            ax.set_xlabel("PC1", color="white"); ax.set_ylabel("PC2", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_color("#2a3a5c")
            ax.set_title(
                f"PCA Cluster Map  ({cluster_df['pca_var_explained'].iloc[0]*100:.0f}% variance explained)",
                color="white", fontsize=11
            )
            ax.grid(color="#2a3a5c", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### Cluster Assignments")
            show_cols = ["team","cluster_label","completion_rate","long_ball_pct",
                          "territory_pct","tempo"]
            show_cols = [c for c in show_cols if c in cluster_df.columns]
            st.dataframe(cluster_df[show_cols].round(3),
                          use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Player Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Player Analysis":
    st.markdown(f"# 👤 Player Analysis — {selected_team}")

    tab1, tab2 = st.tabs(["📋 Squad Overview", "🕵️ Player Deep Dive"])

    with tab1:
        player_stats = get_team_player_stats(events, selected_team)
        # Add influence index
        if not player_stats.empty:
            player_stats["Influence"] = player_stats["Player"].apply(
                lambda p: compute_influence_index(events, p, m["G"], m["metrics_df"])
            )
            player_stats = player_stats.sort_values("Influence", ascending=False)

            st.dataframe(
                player_stats.style.background_gradient(
                    subset=["Passes","Completion %","Influence"],
                    cmap="YlOrRd"
                ),
                use_container_width=True, hide_index=True
            )

    with tab2:
        players = sorted(events["player_name"].unique().tolist())
        p1 = st.selectbox("Player 1", players, index=0)
        p2 = st.selectbox("Player 2 (compare)", players,
                           index=min(1, len(players)-1))

        col1, col2 = st.columns(2)

        def player_card(player_name, col):
            feats = extract_player_features(events, player_name)
            role  = classify_player_role(feats)
            infl  = compute_influence_index(events, player_name, m["G"], m["metrics_df"])
            with col:
                st.markdown(f"### {player_name}")
                st.markdown(f"<div class='style-badge'>{role}</div>", unsafe_allow_html=True)
                st.metric("Influence Index", f"{infl}/100")
                st.metric("Passes", feats["total_passes"])
                st.metric("Completion", f"{feats['completion_pct']:.1f}%")
                st.metric("Progressive %", f"{feats['progressive_pct']:.1f}%")
                st.metric("Goals", feats["goals"])

        player_card(p1, col1)
        player_card(p2, col2)

        st.markdown("### 🕸️ Radar Comparison")
        f1 = extract_player_features(events, p1)
        f2 = extract_player_features(events, p2)
        fig = plot_player_radar([f1, f2], [p1, p2], figsize=(8, 8))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Player heatmap comparison
        st.markdown("### 🔥 Position Heatmaps")
        hc1, hc2 = st.columns(2)
        with hc1:
            fig1 = plot_player_heatmap(events, p1, figsize=(7, 5))
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        with hc2:
            fig2 = plot_player_heatmap(events, p2, figsize=(7, 5))
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Team Comparison
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🆚 Team Comparison":
    st.markdown("# 🆚 Team Comparison")

    teams_to_compare = get_all_team_names()
    t1 = st.selectbox("Team A", teams_to_compare, index=0)
    t2 = st.selectbox("Team B", teams_to_compare, index=1)

    if t1 == t2:
        st.warning("Please select two different teams.")
    else:
        ev1 = all_data[t1]
        ev2 = all_data[t2]

        m1 = compute_team_metrics(t1, n_passes)
        m2 = compute_team_metrics(t2, n_passes)
        f1 = m1["features"]
        f2 = m2["features"]

        # ── Team Radar DNA Comparison ──────────────────────────────────────────
        st.markdown("### 🕸️ Tactical DNA Comparison")
        col_radar_l, col_radar_r = st.columns([1.2, 1])
        with col_radar_l:
            fig_radar = plot_team_radar([f1, f2], [t1, t2], figsize=(9, 8))
            st.pyplot(fig_radar, use_container_width=True)
            plt.close(fig_radar)
        with col_radar_r:
            st.markdown("#### Tactical Summary")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"**{t1}**")
                st.markdown(f"<div class='style-badge'>{m1['style']}</div>", unsafe_allow_html=True)
                st.caption(m1["style_desc"])
                st.markdown(f"Formation: `{m1['formation']}`")
            with pc2:
                st.markdown(f"**{t2}**")
                st.markdown(f"<div class='style-badge'>{m2['style']}</div>", unsafe_allow_html=True)
                st.caption(m2["style_desc"])
                st.markdown(f"Formation: `{m2['formation']}`")
            
            st.info("The radar chart scales metrics based on league averages to show relative strengths in ball retention, verticality, and intensity.")

        st.divider()

        # ── Stats comparison ──────────────────────────────────────────────────
        st.markdown("### 📊 Head-to-Head Stats")
        compare_metrics = [
            ("Passes", "total_passes", 1),
            ("Completion %", "completion_rate", 100),
            ("Long Ball %", "long_ball_pct", 100),
            ("Short Pass %", "short_pass_pct", 100),
            ("Territory %", "territory_pct", 100),
            ("Avg Pass Length", "avg_pass_length", 1),
            ("Vertical Progression", "vertical_progression", 1),
            ("Tempo", "tempo", 1),
            ("Shot Rate", "shot_rate", 1000),
        ]
        rows = []
        for label, key, mult in compare_metrics:
            v1 = round(f1.get(key, 0) * mult, 1)
            v2 = round(f2.get(key, 0) * mult, 1)
            rows.append({
                t1[:18]: v1,
                "Metric": label,
                t2[:18]: v2,
                "Winner": t1 if v1 > v2 else (t2 if v2 > v1 else "Draw")
            })
        cmp_df = pd.DataFrame(rows)[["Metric", t1[:18], t2[:18], "Winner"]]
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Side-by-side heatmaps ─────────────────────────────────────────────
        st.markdown("### 🔥 Heatmap Comparison")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"**{t1}**")
            fig1 = plot_team_heatmap(ev1, t1, figsize=(8, 5))
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        with hc2:
            st.markdown(f"**{t2}**")
            fig2 = plot_team_heatmap(ev2, t2, figsize=(8, 5))
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.divider()

        # ── Side-by-side Shot Maps ───────────────────────────────────────────
        st.markdown("### 🎯 Shot Map Comparison")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**{t1}**")
            fig_s1 = plot_shot_map(ev1, t1, figsize=(8, 5))
            st.pyplot(fig_s1, use_container_width=True)
            plt.close(fig_s1)
        with sc2:
            st.markdown(f"**{t2}**")
            fig_s2 = plot_shot_map(ev2, t2, figsize=(8, 5))
            st.pyplot(fig_s2, use_container_width=True)
            plt.close(fig_s2)

        st.divider()

        # ── Side-by-side Progressive Passes ─────────────────────────────────
        st.markdown("### ⬆️ Progressive Pass Comparison")
        ppc1, ppc2 = st.columns(2)
        with ppc1:
            st.markdown(f"**{t1}**")
            fig_pp1 = plot_progressive_passes(ev1, t1, figsize=(8, 5))
            st.pyplot(fig_pp1, use_container_width=True)
            plt.close(fig_pp1)
        with ppc2:
            st.markdown(f"**{t2}**")
            fig_pp2 = plot_progressive_passes(ev2, t2, figsize=(8, 5))
            st.pyplot(fig_pp2, use_container_width=True)
            plt.close(fig_pp2)

        st.divider()

        # ── Zone control ─────────────────────────────────────────────────────
        st.markdown("### 🗺️ Zone Control Comparison")
        fig_zone = plot_zone_control(ev1, ev2, t1, t2, figsize=(14, 5))
        st.pyplot(fig_zone, use_container_width=True)
        plt.close(fig_zone)

        st.divider()

        # ── Network comparison ────────────────────────────────────────────────
        st.markdown("### 🕸️ Passing Network Comparison")
        nc1, nc2 = st.columns(2)
        with nc1:
            st.markdown(f"**{t1}**")
            fig_n1 = plot_passing_network(m1["G"], m1["node_pos"], m1["metrics_df"], t1, figsize=(8,6))
            st.pyplot(fig_n1, use_container_width=True)
            plt.close(fig_n1)
        with nc2:
            st.markdown(f"**{t2}**")
            fig_n2 = plot_passing_network(m2["G"], m2["node_pos"], m2["metrics_df"], t2, figsize=(8,6))
            st.pyplot(fig_n2, use_container_width=True)
            plt.close(fig_n2)

        st.divider()

        # ── Possession Chain Comparison ─────────────────────────────────────
        st.markdown("### ⛓️ Possession Chain Comparison")
        cc1, cc2 = st.columns(2)
        
        def plot_mini_chain_dist(chains, team_name, color):
            if not chains: return None
            df = pd.DataFrame(chains)
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0d1a2d")
            ax.set_facecolor("#0d1a2d")
            ax.hist(df["length"], bins=10, color=color, alpha=0.7, edgecolor="#0d1a2d")
            ax.set_title(f"{team_name} — Chain Lengths", color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=7)
            for s in ax.spines.values(): s.set_color("#2a3a5c")
            plt.tight_layout()
            return fig

        with cc1:
            st.markdown(f"**{t1}**")
            fig_c1 = plot_mini_chain_dist(m1["chains"], t1, "#4fc3f7")
            if fig_c1:
                st.pyplot(fig_c1, use_container_width=True)
                plt.close(fig_c1)
            cs1 = m1["chain_summary"]
            st.metric("Avg Chain Length", cs1.get("avg_chain_length", 0))
            st.metric("Final Third Entries", cs1.get("chains_reaching_final_third", 0))

        with cc2:
            st.markdown(f"**{t2}**")
            fig_c2 = plot_mini_chain_dist(m2["chains"], t2, "#ff8c00")
            if fig_c2:
                st.pyplot(fig_c2, use_container_width=True)
                plt.close(fig_c2)
            cs2 = m2["chain_summary"]
            st.metric("Avg Chain Length", cs2.get("avg_chain_length", 0))
            st.metric("Final Third Entries", cs2.get("chains_reaching_final_third", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Full Report
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Full Report":
    st.markdown(f"# 📄 Full Tactical Report — {selected_team}")

    player_stats = get_team_player_stats(events, selected_team)
    player_stats["influence"] = player_stats["Player"].apply(
        lambda p: compute_influence_index(events, p, m["G"], m["metrics_df"])
    )
    top_players = player_stats.head(6).to_dict("records")

    report = generate_tactical_report(
        team_name=selected_team,
        style=m["style"],
        style_description=m["style_desc"],
        formation=m["formation"],
        features=m["features"],
        chain_summary=m["chain_summary"],
        pressing=m["pressing"],
        top_players=top_players,
        metrics_df=m["metrics_df"],
    )

    st.markdown(report)

    st.divider()
    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=report,
        file_name=f"{selected_team.replace(' ','_')}_tactical_report.md",
        mime="text/markdown",
    )

    # Export data
    st.markdown("### 📦 Export Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = events.to_csv(index=False)
        st.download_button(
            "⬇️ Download Events CSV",
            data=csv,
            file_name=f"{selected_team}_events.csv",
            mime="text/csv"
        )
    with col2:
        if not m["metrics_df"].empty:
            net_csv = m["metrics_df"].to_csv(index=False)
            st.download_button(
                "⬇️ Download Network Metrics CSV",
                data=net_csv,
                file_name=f"{selected_team}_network_metrics.csv",
                mime="text/csv"
            )
