import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MLB PrizePicks Simulator", layout="wide")

# --- Helper Functions ---
def simulate_binary(prob, sims=10000):
    outcomes = np.random.binomial(1, prob, sims)
    win_rate = outcomes.mean()
    return win_rate

def simulate_count_poisson(mean_val, threshold, sims=10000):
    outcomes = np.random.poisson(mean_val, sims)
    win_rate = (outcomes >= threshold).mean()
    return win_rate

def simulate_fantasy(mean, stddev=4.0, threshold=8.5, sims=10000):
    outcomes = np.random.normal(mean, stddev, sims)
    win_rate = (outcomes >= threshold).mean()
    return win_rate

def calculate_ev(prob, payout=1.0):
    return prob * payout - (1 - prob)

# --- UI Layout ---
st.title("📊 MLB PrizePicks Simulator")
st.markdown("Upload your **BallparkPal Batters & Pitchers** files to simulate HRs, Hits, Fantasy Score, and more.")

col1, col2 = st.columns(2)
batters_file = col1.file_uploader("📥 Upload Batters Excel File", type=["xlsx"])
pitchers_file = col2.file_uploader("📥 Upload Pitchers Excel File", type=["xlsx"])

if batters_file and pitchers_file:
    batters = pd.read_excel(batters_file)
    pitchers = pd.read_excel(pitchers_file)

    # Preprocess pitcher info
    pitcher_subset = pitchers[['GamePk', 'Team', 'HomeRunsAllowed']].rename(columns={
        'Team': 'PitcherTeam',
        'HomeRunsAllowed': 'HR_Allowed_By_Pitcher'
    })

    batters = batters.rename(columns={'Opponent': 'PitcherTeam'})
    merged = pd.merge(batters, pitcher_subset, on=['GamePk', 'PitcherTeam'], how='left')

    # Adjust HR prob using pitcher HR allowed
    mean_hr_allowed = merged['HR_Allowed_By_Pitcher'].mean()
    merged['Adjusted_HR_Prob'] = merged['HomeRunProbability'] * (merged['HR_Allowed_By_Pitcher'] / mean_hr_allowed)

    # --- Simulate Props ---
    st.header("📈 Simulated PrizePicks Props")

    results = []
    for _, row in merged.iterrows():
        hr_win = simulate_binary(row['Adjusted_HR_Prob'])
        hit_win = simulate_binary(row['HitProbability'])
        run_win = simulate_count_poisson(row['Runs'], threshold=1)
        rbi_win = simulate_count_poisson(row['RBIs'], threshold=1)
        sb_win = simulate_binary(row['StolenBaseProbability'])
        fp_win = simulate_fantasy(row['PointsDK'])

        results.append({
            "Player": row['FullName'],
            "Team": row['Team'],
            "Opponent": row['PitcherTeam'],
            "Adj HR Prob": row['Adjusted_HR_Prob'],
            "HR Win%": round(hr_win * 100, 1),
            "Hits Win%": round(hit_win * 100, 1),
            "Runs Win%": round(run_win * 100, 1),
            "RBIs Win%": round(rbi_win * 100, 1),
            "SB Win%": round(sb_win * 100, 1),
            "FP Win% (8.5)": round(fp_win * 100, 1),
            "HR EV": round(calculate_ev(hr_win), 3),
            "FP EV": round(calculate_ev(fp_win), 3),
            "HR +EV": "✅" if calculate_ev(hr_win) > 0 else "❌",
            "FP +EV": "✅" if calculate_ev(fp_win) > 0 else "❌",
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results.sort_values(by="HR EV", ascending=False), use_container_width=True)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="simulated_prizepicks_props.csv")

else:
    st.info("Please upload both batter and pitcher Excel files to begin.")
