import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="F1 Predictor Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0rem;
        color: #ff1801; /* F1 Red */
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #ff1801;
    }
</style>
""", unsafe_allow_html=True)

PREDICTIONS_DIR = Path("predictions")

@st.cache_data
def get_available_races():
    if not PREDICTIONS_DIR.exists():
        return []
    files = list(PREDICTIONS_DIR.glob("*_simulations.json"))
    races = []
    for f in files:
        # Extract year and round from filename, e.g., 2026_R04_simulations.json
        parts = f.name.split("_")
        if len(parts) >= 2:
            year = parts[0]
            rnd = parts[1].replace("R", "")
            races.append((int(year), int(rnd), f))
    return sorted(races, reverse=True)

@st.cache_data
def load_simulation_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def main():
    st.markdown('<p class="main-header">F1 Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monte Carlo Race Simulation Dashboard</p>', unsafe_allow_html=True)

    races = get_available_races()
    
    if not races:
        st.warning("No simulation data found in the 'predictions/' directory.")
        return

    # Sidebar selection
    st.sidebar.header("Race Selection")
    
    race_options = {f"{year} Round {rnd}": f_path for year, rnd, f_path in races}
    selected_race_label = st.sidebar.selectbox(
        "Select a race to visualize:",
        list(race_options.keys())
    )
    
    selected_file = race_options[selected_race_label]
    
    # Load data
    data = load_simulation_data(selected_file)
    year = data.get("year", "N/A")
    rnd = data.get("round", "N/A")
    event = data.get("event", "Unknown Event")
    circuit = data.get("circuit", "Unknown Circuit")
    num_sims = data.get("num_simulations", 10000)
    
    results = data.get("results", [])
    if not results:
        st.error("No results found in the selected simulation file.")
        return
        
    df = pd.DataFrame(results)
    
    # Header Info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Event:** {event}")
    with col2:
        st.markdown(f"**Circuit:** {circuit}")
    with col3:
        st.markdown(f"**Season:** {year} | **Round:** {rnd}")
    with col4:
        st.markdown(f"**Simulations:** {num_sims:,}")
        
    st.divider()

    # Top 3 Predictions
    st.subheader("Top 3 Favorites")
    c1, c2, c3 = st.columns(3)
    
    top3 = df.head(3)
    for i, (idx, row) in enumerate(top3.iterrows()):
        col = [c1, c2, c3][i]
        with col:
            win_pct = row['SimWinProb'] * 100
            pod_pct = row['SimPodiumProb'] * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin:0; padding:0;">P{i+1}: {row['Driver']}</h2>
                <p style="margin:0; color:#aaa;">{row['TeamName']}</p>
                <hr style="margin:10px 0; border-color:#333;">
                <p style="margin:0;"><b>Win Probability:</b> {win_pct:.1f}%</p>
                <p style="margin:0;"><b>Podium Probability:</b> {pod_pct:.1f}%</p>
                <p style="margin:0;"><b>Expected Finish:</b> {row['SimExpectedPos']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("")
    
    # Visualization: Heatmap of positions
    st.subheader("Monte Carlo Position Distribution")
    
    # Prepare data for heatmap
    # rows: Drivers, cols: Positions 1-20 + DNF
    heatmap_data = []
    drivers = []
    
    # Sort by Expected Position
    df_sorted = df.sort_values("SimExpectedPos")
    
    pos_cols = [str(i) for i in range(1, 21)] + ["DNF"]
    
    for _, row in df_sorted.iterrows():
        drivers.append(row["Driver"])
        dist = row["PositionDistribution"]
        # Convert to percentages
        driver_dist = [dist.get(pos, 0.0) * 100 for pos in pos_cols]
        heatmap_data.append(driver_dist)
        
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=pos_cols,
        y=drivers,
        colorscale="Inferno",
        hoverongaps=False,
        hovertemplate="Driver: %{y}<br>Position: %{x}<br>Probability: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Probability of Finishing in Each Position (%)",
        xaxis_title="Finishing Position",
        yaxis_title="Driver (Sorted by Expected Finish)",
        height=700,
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Data Table
    st.subheader("Detailed Simulation Metrics")
    
    # Format table for display
    display_df = df.copy()
    display_df = display_df[["Rank", "Driver", "TeamName", "PredictedQualiPos", 
                            "SimExpectedPos", "SimWinProb", "SimPodiumProb", "SimPointsProb", "SimDNFProb"]]
    
    display_df["SimWinProb"] = (display_df["SimWinProb"] * 100).apply(lambda x: f"{x:.1f}%")
    display_df["SimPodiumProb"] = (display_df["SimPodiumProb"] * 100).apply(lambda x: f"{x:.1f}%")
    display_df["SimPointsProb"] = (display_df["SimPointsProb"] * 100).apply(lambda x: f"{x:.1f}%")
    display_df["SimDNFProb"] = (display_df["SimDNFProb"] * 100).apply(lambda x: f"{x:.1f}%")
    
    display_df.columns = ["Rank", "Driver", "Team", "Pred. Quali", 
                         "Exp. Finish", "Win %", "Podium %", "Points %", "DNF %"]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
