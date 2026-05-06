import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

class TeamStateUpdater:
    """state tracking untuk setiap tim dengan moving window stats."""
    def __init__(self):
        #1. inisialisasi state
        self.elo = {}
        self.points = defaultdict(list)
        self.goals_scored = defaultdict(list)
        self.goals_conceded = defaultdict(list)
        self.outcomes = defaultdict(list)
        self.last_match_date = {}
        self.h2h = defaultdict(dict)
        
    def _dapatkan_k_factor(self, tournament):
        """K-factor bergantung tournament importance."""
        t_lower = str(tournament).lower()
        if 'world cup' in t_lower and 'qualification' not in t_lower:
            return 60
        elif 'continental' in t_lower or 'copa america' in t_lower or 'euro' in t_lower or 'african' in t_lower:
            return 50 if 'qualification' not in t_lower else 40
        elif 'friendly' in t_lower:
            return 20
        elif 'qualification' in t_lower:
            return 40
        return 30
            
    def _dapatkan_gd_multiplier(self, margin):
        """goal difference multiplier untuk elo update."""
        margin = abs(margin)
        if margin <= 1:
            return 1.0
        elif margin == 2:
            return 1.5
        elif margin == 3:
            return 1.75
        return 1.75 + ((margin - 3) / 8.0)
            
    def update_and_extract(self, date_str, team, opp, team_goals, opp_goals, is_home, neutral, tournament):
        """
        extract fitur sebelum match (pre-match state), update state setelah match.
        """
        current_date = pd.to_datetime(date_str)
        days_team = (current_date - self.last_match_date.get(team, current_date)).days
        days_opp = (current_date - self.last_match_date.get(opp, current_date)).days
        
        #2. ekstraksi fitur PRE-MATCH
        elo_team = self.elo.get(team, 1500.0)
        elo_opp = self.elo.get(opp, 1500.0)
        h2h_list = self.h2h[team].get(opp, [])
        
        fitur = {
            'elo_team_calc': elo_team,
            'elo_opponent_calc': elo_opp,
            'elo_diff_calc': elo_team - elo_opp,
            'team_points_last5_calc': sum(self.points[team][-5:]) if self.points[team] else 0,
            'team_gd_last5_calc': sum(self.goals_scored[team][-5:]) - sum(self.goals_conceded[team][-5:]) if self.goals_scored[team] else 0,
            'team_win_rate_last10_calc': np.mean([1 if x==1 else 0 for x in self.outcomes[team][-10:]]) if self.outcomes[team] else 0,
            'opp_points_last5_calc': sum(self.points[opp][-5:]) if self.points[opp] else 0,
            'opp_gd_last5_calc': sum(self.goals_scored[opp][-5:]) - sum(self.goals_conceded[opp][-5:]) if self.goals_scored[opp] else 0,
            'opp_win_rate_last10_calc': np.mean([1 if x==1 else 0 for x in self.outcomes[opp][-10:]]) if self.outcomes[opp] else 0,
            'h2h_points_last5_calc': sum(h2h_list[-5:]) if h2h_list else 0,
            'days_since_last_match_team_calc': days_team if team in self.last_match_date else -1,
            'days_since_last_match_opp_calc': days_opp if opp in self.last_match_date else -1,
        }
        
        #3. jika target kosong (test set), return fitur saja
        if pd.isna(team_goals) or pd.isna(opp_goals):
            return fitur
            
        #4. kalkulasi outcome & update elo
        if team_goals > opp_goals:
            w_team, w_opp = 1.0, 0.0
            pts_team, pts_opp = 3, 0
        elif team_goals == opp_goals:
            w_team, w_opp = 0.5, 0.5
            pts_team, pts_opp = 1, 1
        else:
            w_team, w_opp = 0.0, 1.0
            pts_team, pts_opp = 0, 3
            
        margin = abs(team_goals - opp_goals)
        dr = elo_team - elo_opp
        if neutral == 0:
            dr += 100 if is_home == 1 else -100
                
        we_team = 1 / (10 ** (-dr / 400.0) + 1)
        k_total = self._dapatkan_k_factor(tournament) * self._dapatkan_gd_multiplier(margin)
        
        self.elo[team] = elo_team + k_total * (w_team - we_team)
        self.elo[opp] = elo_opp + k_total * (w_opp - (1 - we_team))
        
        #5. update rolling window
        self.points[team].append(pts_team)
        self.points[opp].append(pts_opp)
        self.goals_scored[team].append(team_goals)
        self.goals_conceded[team].append(opp_goals)
        self.goals_scored[opp].append(opp_goals)
        self.goals_conceded[opp].append(team_goals)
        self.outcomes[team].append(w_team)
        self.outcomes[opp].append(w_opp)
        
        if opp not in self.h2h[team]:
            self.h2h[team][opp] = []
        if team not in self.h2h[opp]:
            self.h2h[opp][team] = []
        self.h2h[team][opp].append(pts_team)
        self.h2h[opp][team].append(pts_opp)
        
        self.last_match_date[team] = current_date
        self.last_match_date[opp] = current_date
        
        return fitur

#6. load cleaned training data
print("loading cleaned training data...")
train_df = pd.read_csv('./data/processed/train_cleaned.csv')
print(f"shape: {train_df.shape}")

#7. sort chronologically dan filter data
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year
train_df = train_df[train_df['year'] >= 1950].copy()
train_df = train_df.sort_values('date').reset_index(drop=True)

print(f"date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"total matches (filtered year >= 1950): {len(train_df)}\n")

#8. rekonstruksi state historis
print("reconstructing historical state & extracting features...")
tracker = TeamStateUpdater()
hasil_fitur = []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="State Reconstruction"):
    fitur = tracker.update_and_extract(
        date_str=row['date'],
        team=row['team'],
        opp=row['opponent'],
        team_goals=row['team_goals'],
        opp_goals=row['opp_goals'],
        is_home=row['is_home'],
        neutral=row['neutral'],
        tournament=row['tournament']
    )
    hasil_fitur.append(fitur)

fitur_df = pd.DataFrame(hasil_fitur)

#9. merge features dengan training data
train_engineered = pd.concat([train_df.reset_index(drop=True), fitur_df], axis=1)

print(f"\nfeatures extracted: {fitur_df.shape[1]}")
print(f"engineered training set: {train_engineered.shape}\n")

#10. verifikasi
missing_per_col = train_engineered.iloc[:, -fitur_df.shape[1]:].isna().sum()
if missing_per_col.sum() == 0:
    print("[PASS] 0 NaN di engineered features")
else:
    print(f"[WARNING] {missing_per_col.sum()} NaN ditemukan")
    print(missing_per_col[missing_per_col > 0])

#11. export results
print("\nexporting engineered training data...")
train_engineered.to_csv('./data/processed/train_engineered.csv', index=False)
print(f"[SAVED] ./data/processed/train_engineered.csv")

print("exporting state tracker for Phase 5...")
with open('./data/processed/state_tracker.pkl', 'wb') as f:
    pickle.dump(tracker, f)
print(f"[SAVED] ./data/processed/state_tracker.pkl")

print("\nFASE 3 SELESAI!")
