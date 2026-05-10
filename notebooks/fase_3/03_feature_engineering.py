import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

#define value default buat self.elo
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

#eksekusi utama pipeline
train_df = pd.read_csv('./data/processed/train_cleaned.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year'] = train_df['date'].dt.year

#pemotongan structural break pre-1950
train_df = train_df[train_df['year'] >= 1950].sort_values('date').reset_index(drop=True)

tracker = TeamStateUpdater()
hasil_fitur =[]

for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Rekonstruksi State"):
    fitur = tracker.update_and_extract(
        date_str=row['date'], team=row['team'], opp=row['opponent'],
        team_goals=row['team_goals'], opp_goals=row['opp_goals'],
        is_home=row['is_home'], neutral=row['neutral'], tournament=row['tournament']
    )
    hasil_fitur.append(fitur)

train_engineered = pd.concat([train_df, pd.DataFrame(hasil_fitur)], axis=1)

#3.2 rekayasa rasio dan diferensial
train_engineered['gdp_ratio'] = train_engineered['gdp_per_capita_team'] / (train_engineered['gdp_per_capita_opp'] + 1)
train_engineered['pop_ratio'] = train_engineered['population_team'] / (train_engineered['population_opp'] + 1)
train_engineered['travel_advantage'] = train_engineered['distance_travel_opp'] - train_engineered['distance_travel_team']

#3.4 injeksi konteks turnamen dan kandang
kondisi_t = [
    train_engineered['tournament'].str.contains('World Cup', case=False, na=False) & ~train_engineered['tournament'].str.contains('Qualification', case=False, na=False),
    train_engineered['tournament'].str.contains('AFC|Copa America|Euro|African', case=False, na=False) & ~train_engineered['tournament'].str.contains('Qualification', case=False, na=False),
    train_engineered['tournament'].str.contains('Friendly', case=False, na=False)
]
train_engineered['tournament_weight'] = np.select(kondisi_t, [2.00, 1.80, 0.96], default=1.20)

train_engineered['home_tournament_interaction'] = train_engineered['is_home'] * train_engineered['tournament_weight']
train_engineered['is_high_stakes'] = train_engineered['tournament'].str.contains('World Cup|Continental', case=False, na=False).astype(int)

#3.5 penyiapan target probabilitas W/D/L untuk fase 4A
train_engineered['target_win'] = (train_engineered['team_goals'] > train_engineered['opp_goals']).astype(int)
train_engineered['target_draw'] = (train_engineered['team_goals'] == train_engineered['opp_goals']).astype(int)
train_engineered['target_lose'] = (train_engineered['team_goals'] < train_engineered['opp_goals']).astype(int)

#3.6 pembersihan h2h features
print("\ncleaning h2h features...")
#fillna dengan 0 (semantik: tidak ada riwayat = 0, bukan median)
train_engineered['h2h_gd_last5'] = train_engineered['h2h_gd_last5'].fillna(0)
train_engineered['h2h_points_last5'] = train_engineered['h2h_points_last5'].fillna(0)
#flag: apakah ada riwayat H2H?
train_engineered['has_h2h_history'] = (
    train_engineered['h2h_gd_last5'].abs() > 0
).astype(int)
print("  h2h_gd_last5 fillna(0) OK")
print("  h2h_points_last5 fillna(0) OK")
print("  has_h2h_history flag added OK")

#3.7 drop dan rename kolom duplikat
print("\nfixing duplicate columns...")
cols_to_drop = [
    'elo_team',
    'elo_opponent',
    'days_since_last_match_team',
    'days_since_last_match_opp',
]
train_engineered = train_engineered.drop(columns=cols_to_drop, errors='ignore')

rename_map = {
    'elo_team.1': 'elo_team_final',
    'elo_opp': 'elo_opponent_final',
    'days_since_last_match_team.1': 'rest_days_team',
    'days_since_last_match_opp.1': 'rest_days_opp',
}
train_engineered = train_engineered.rename(columns=rename_map)
print(f"  dropped {len(cols_to_drop)} old columns")
print(f"  renamed {len(rename_map)} columns to final versions")

#3.8 categorical encoding absolut
#konversi gender ke biner
train_engineered['gender_encode'] = train_engineered['gender'].map({'M': 1, 'W': 0}).fillna(1)

#konversi confederation ke ordinal numeric
semua_conf = pd.concat([train_engineered['confederation_team'], train_engineered['confederation_opp']]).dropna().unique()
conf_map = {k: i for i, k in enumerate(semua_conf)}

train_engineered['conf_team_encode'] = train_engineered['confederation_team'].map(conf_map).fillna(-1)
train_engineered['conf_opp_encode'] = train_engineered['confederation_opp'].map(conf_map).fillna(-1)

#pemusnahan kolom teks mentah
kolom_sampah =['gender', 'confederation_team', 'confederation_opp', 'venue_country']
train_engineered = train_engineered.drop(columns=kolom_sampah, errors='ignore')

#ekspor akhir
train_engineered.to_csv('./data/processed/train_engineered.csv', index=False)
#penghapusan kolom redundan dan ekspor
train_engineered.to_csv('./data/processed/train_engineered.csv', index=False)
with open('./data/processed/state_tracker.pkl', 'wb') as f:
    pickle.dump(tracker, f)