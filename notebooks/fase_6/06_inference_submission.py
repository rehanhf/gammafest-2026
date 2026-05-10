import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import sys
import os
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.optimizer import optimizer
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
    from src.optimizer import optimizer

def default_elo():
    return 1500.0
class TeamStateUpdater:
    def __init__(self):
        #inisialisasi memory stat historis
        self.elo = defaultdict(default_elo)
        self.goals_scored = defaultdict(list)
        self.goals_conceded = defaultdict(list)
        self.outcomes = defaultdict(list)
        self.last_match_date = {}
        
    def _dapatkan_k_factor(self, tournament):
        t_lower = str(tournament).lower()
        if 'world cup' in t_lower and 'qualification' not in t_lower: return 60
        if 'continental' in t_lower or 'copa america' in t_lower or 'euro' in t_lower or 'african' in t_lower:
            return 40 if 'qualification' in t_lower else 50
        if 'friendly' in t_lower: return 20
        if 'qualification' in t_lower: return 40
        return 30
            
    def _dapatkan_gd_multiplier(self, margin):
        margin = abs(margin)
        if margin <= 1: return 1.0
        if margin == 2: return 1.5
        if margin == 3: return 1.75
        return 1.75 + ((margin - 3) / 8.0)

    def update_and_extract(self, date_str, team, opp, team_goals, opp_goals, is_home, neutral, tournament):
        current_date = pd.to_datetime(date_str)
        
        #3.4 rest days (fatigue proxy)
        days_team = (current_date - self.last_match_date.get(team, current_date)).days
        days_opp = (current_date - self.last_match_date.get(opp, current_date)).days
        
        #3.1 elo probability derivation
        elo_t = self.elo[team]
        elo_o = self.elo[opp]
        elo_diff = elo_t - elo_o
        elo_win_prob = 1 / (1 + 10 ** (-elo_diff / 400.0))
        
        #3.3 rolling form extraction (3, 5, 10 windows)
        def get_roll(arr, w): return np.mean(arr[-w:]) if len(arr) > 0 else 0.0
        
        team_gs_3, team_gs_5, team_gs_10 = get_roll(self.goals_scored[team], 3), get_roll(self.goals_scored[team], 5), get_roll(self.goals_scored[team], 10)
        team_gc_3, team_gc_5, team_gc_10 = get_roll(self.goals_conceded[team], 3), get_roll(self.goals_conceded[team], 5), get_roll(self.goals_conceded[team], 10)
        opp_gs_3, opp_gs_5, opp_gs_10 = get_roll(self.goals_scored[opp], 3), get_roll(self.goals_scored[opp], 5), get_roll(self.goals_scored[opp], 10)
        opp_gc_3, opp_gc_5, opp_gc_10 = get_roll(self.goals_conceded[opp], 3), get_roll(self.goals_conceded[opp], 5), get_roll(self.goals_conceded[opp], 10)
        
        team_win_rate_10 = get_roll(self.outcomes[team], 10)
        opp_win_rate_10 = get_roll(self.outcomes[opp], 10)
        
        fitur = {
            'elo_team': elo_t,
            'elo_opp': elo_o,
            'elo_diff': elo_diff,
            'elo_win_prob': elo_win_prob,
            
            'roll_avg_goals_scored_team_3': team_gs_3,
            'roll_avg_goals_scored_team_10': team_gs_10,
            'roll_avg_goals_conceded_team_5': team_gc_5,
            'roll_win_rate_team_10': team_win_rate_10,
            
            'roll_avg_goals_scored_opp_3': opp_gs_3,
            'roll_avg_goals_scored_opp_10': opp_gs_10,
            'roll_avg_goals_conceded_opp_5': opp_gc_5,
            'roll_win_rate_opp_10': opp_win_rate_10,
            
            'form_trend_team': team_gs_3 - team_gs_10,
            'form_trend_opp': opp_gs_3 - opp_gs_10,
            
            'days_since_last_match_team': days_team if team in self.last_match_date else -1,
            'days_since_last_match_opp': days_opp if opp in self.last_match_date else -1
        }
        
        if pd.isna(team_goals) or pd.isna(opp_goals):
            return fitur
            
        if team_goals > opp_goals: w_t, w_o = 1.0, 0.0
        elif team_goals == opp_goals: w_t, w_o = 0.5, 0.5
        else: w_t, w_o = 0.0, 1.0
            
        dr_adj = elo_diff + (100 if neutral == 0 and is_home == 1 else (-100 if neutral == 0 and is_home == 0 else 0))
        we_team = 1 / (1 + 10 ** (-dr_adj / 400.0))
        
        margin = abs(team_goals - opp_goals)
        k_total = self._dapatkan_k_factor(tournament) * self._dapatkan_gd_multiplier(margin)
        
        self.elo[team] += k_total * (w_t - we_team)
        self.elo[opp] += k_total * (w_o - (1 - we_team))
        
        self.goals_scored[team].append(team_goals); self.goals_conceded[team].append(opp_goals)
        self.goals_scored[opp].append(opp_goals); self.goals_conceded[opp].append(team_goals)
        self.outcomes[team].append(1 if w_t == 1.0 else 0); self.outcomes[opp].append(1 if w_o == 1.0 else 0)
        
        self.last_match_date[team] = current_date
        self.last_match_date[opp] = current_date
        
        return fitur

# 1. Load Data dan State Tracker
test_df = pd.read_csv('./data/raw/test.csv')
test_df['date'] = pd.to_datetime(test_df['date'])
test_df = test_df.sort_values('date').reset_index(drop=True)

with open('./data/processed/state_tracker.pkl', 'rb') as f:
    tracker = pickle.load(f)

# 2. Load Ensemble Models
n_splits = 5
models_t =[]
models_o =[]
for fold in range(n_splits):
    with open(f'../models/lgbm_team_fold{fold}.pkl', 'rb') as f:
        models_t.append(pickle.load(f))
    with open(f'../models/lgbm_opp_fold{fold}.pkl', 'rb') as f:
        models_o.append(pickle.load(f))

# 3. Ekstraksi urutan fitur yang sesuai dengan X_train
train_engineered = pd.read_csv('./data/processed/train_engineered.csv', nrows=1)
drop_cols =['Id', 'match_id', 'date', 'team', 'opponent', 'tournament', 'team_goals', 'opp_goals']
fitur_df = train_engineered.drop(columns=[c for c in drop_cols if c in train_engineered.columns], errors='ignore')
fitur_df = fitur_df.select_dtypes(include=[np.number])
feature_names = fitur_df.columns.tolist()

pred_team_goals = []
pred_opp_goals =[]

print("Memulai Sequential Inference pada Test Set (2011-2026)...")

# 4. Sequential Inference Loop
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # Ekstrak state sebelum pertandingan
    state_features = tracker.update_and_extract(
        date_str=row['date'],
        team=row['team'],
        opp=row['opponent'],
        team_goals=np.nan,
        opp_goals=np.nan,
        is_home=row['is_home'],
        neutral=row['neutral'],
        tournament=row['tournament']
    )
    
    # Gabungkan fitur statis dan stateful
    current_features = {}
    
    # Fitur statis dari row
    kondisi_t = [
        'World Cup' in str(row['tournament']) and 'Qualification' not in str(row['tournament']),
        any(x in str(row['tournament']) for x in['AFC', 'Copa America', 'Euro', 'African']) and 'Qualification' not in str(row['tournament']),
        'Friendly' in str(row['tournament'])
    ]
    current_features['tournament_weight'] = np.select(kondisi_t, [2.00, 1.80, 0.96], default=1.20).item()
    current_features['is_home'] = row['is_home']
    current_features['neutral'] = row['neutral']
    
    # Update dictionary dengan state features
    current_features.update(state_features)
    
    # Konstruksi array 1D sesuai urutan feature_names
    x_input = np.array([current_features.get(f, 0.0) for f in feature_names]).reshape(1, -1)
    
    # Bungkus x_input ke dalam DataFrame untuk menghilangkan UserWarning
    x_input_df = pd.DataFrame(x_input, columns=feature_names)
    
    # Ensemble Lambda Prediksi
    lam_t_folds =[np.clip(m.predict(x_input_df)[0], 0.001, 15.0) for m in models_t]
    lam_o_folds =[np.clip(m.predict(x_input_df)[0], 0.001, 15.0) for m in models_o]
    
    # Rata-rata Lambda
    lam_t_mean = np.mean(lam_t_folds)
    lam_o_mean = np.mean(lam_o_folds)
    
    # Expected Loss Optimization
    pt, po = optimizer.optimize_prediction(lam_t_mean, lam_o_mean, rho=0.0)
    
    pred_team_goals.append(pt)
    pred_opp_goals.append(po)
    
    # Update tracker dengan prediksi optimal agar berantai ke pertandingan masa depan
    _ = tracker.update_and_extract(
        date_str=row['date'],
        team=row['team'],
        opp=row['opponent'],
        team_goals=pt,
        opp_goals=po,
        is_home=row['is_home'],
        neutral=row['neutral'],
        tournament=row['tournament']
    )

# 5. Ekspor Format Kaggle
submission = test_df[['Id']].copy()
submission['team_goals'] = pred_team_goals
submission['opp_goals'] = pred_opp_goals

os.makedirs('./submissions', exist_ok=True)
submission.to_csv('./submissions/final_submission_awmae.csv', index=False)
print("Inference Selesai. File submission tersimpan di ../submissions/final_submission_awmae.csv")