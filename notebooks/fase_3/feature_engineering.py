"""
=============================================================
FASE 2 — FEATURE ENGINEERING
Gammafest 2026 | Football Match Score Prediction
=============================================================
Strategi:
  - Train: 1872–2011  |  Test: 2011–2026 (masa depan)
  - Test TIDAK punya ELO, rank, rolling stats → harus direkonstruksi
  - Solusi: gabung train+test secara kronologis, hitung semua fitur
    dari nol menggunakan riwayat SEBELUM setiap pertandingan
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("FASE 2 — FEATURE ENGINEERING")
print("=" * 60)

TRAIN_PATH = './data/processed/train_cleaned.csv'
TEST_PATH  = './data/processed/test_cleaned.csv'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])
train['is_train'] = 1
test['is_train']  = 0

print(f"\n[LOAD] Train: {len(train):,} baris | Test: {len(test):,} baris")
print(f"       Train date: {train['date'].min().date()} → {train['date'].max().date()}")
print(f"       Test  date: {test['date'].min().date()} → {test['date'].max().date()}")

# ─────────────────────────────────────────────
# 1. GABUNGKAN & URUTKAN KRONOLOGIS
# ─────────────────────────────────────────────
# Kolom target hanya ada di train; isi NaN untuk test
for col in ['team_goals','opp_goals']:
    if col not in test.columns:
        test[col] = np.nan

df = pd.concat([train, test], ignore_index=True, sort=False)
df = df.sort_values(['date','match_id'], ascending=True).reset_index(drop=True)

print(f"\n[GABUNG] Total baris gabungan: {len(df):,}")

# ─────────────────────────────────────────────
# 2. REKONSTRUKSI ELO DARI NOL
# ─────────────────────────────────────────────
print("\n[ELO] Merekonstruksi ELO dari 1872 secara kronologis...")

ELO_INIT  = 1500.0
ELO_K     = 32      # learning rate (agresif untuk data panjang)
ELO_HOME  = 50      # bonus kandang sebelum kalkulasi
elo_ratings = {}    # {team: elo_value}

elo_team_list = []
elo_opp_list  = []

for _, row in df.iterrows():
    team = row['team']
    opp  = row['opponent']

    # ambil ELO sebelum pertandingan
    elo_t = elo_ratings.get(team, ELO_INIT)
    elo_o = elo_ratings.get(opp,  ELO_INIT)

    elo_team_list.append(elo_t)
    elo_opp_list.append(elo_o)

    # update ELO hanya jika ada hasil (train)
    if pd.notna(row['team_goals']) and pd.notna(row['opp_goals']):
        g_team = row['team_goals']
        g_opp  = row['opp_goals']

        # bonus home advantage dalam ELO kalkulasi
        elo_t_adj = elo_t + (ELO_HOME if row['is_home'] == 1 else 0)
        elo_o_adj = elo_o + (ELO_HOME if row['is_home'] == 0 else 0)

        # expected score
        exp_t = 1 / (1 + 10 ** ((elo_o_adj - elo_t_adj) / 400))
        exp_o = 1 - exp_t

        # actual score (1=menang, 0.5=seri, 0=kalah)
        if g_team > g_opp:
            act_t, act_o = 1.0, 0.0
        elif g_team == g_opp:
            act_t, act_o = 0.5, 0.5
        else:
            act_t, act_o = 0.0, 1.0

        # margin of victory multiplier
        gd = abs(g_team - g_opp)
        mov = np.log(gd + 1) + 1 if gd > 0 else 1

        # update
        elo_ratings[team] = elo_t + ELO_K * mov * (act_t - exp_t)
        elo_ratings[opp]  = elo_o + ELO_K * mov * (act_o - exp_o)

df['elo_reconstructed_team'] = elo_team_list
df['elo_reconstructed_opp']  = elo_opp_list
df['elo_diff_reconstructed'] = df['elo_reconstructed_team'] - df['elo_reconstructed_opp']
df['elo_win_prob']            = 1 / (1 + 10 ** (-df['elo_diff_reconstructed'] / 400))

# Gunakan ELO rekonstruksi jika ELO asli tidak ada (test set)
df['elo_team_final']     = df['elo_team'].combine_first(df['elo_reconstructed_team'])
df['elo_opponent_final'] = df['elo_opponent'].combine_first(df['elo_reconstructed_opp'])
df['elo_diff']           = df['elo_team_final'] - df['elo_opponent_final']

print(f"       ✓ ELO rekonstruksi selesai. ELO tim tertinggi: "
      f"{max(elo_ratings.values()):.0f} | terendah: {min(elo_ratings.values()):.0f}")

# ─────────────────────────────────────────────
# 3. ROLLING STATS PER TIM (dari riwayat sebelumnya)
# ─────────────────────────────────────────────
print("\n[ROLLING] Menghitung rolling stats kronologis per tim...")

WINDOWS = [5, 10]

# Simpan riwayat per tim: {team: list of (date, goals_scored, goals_conceded, points)}
from collections import defaultdict
team_history = defaultdict(list)

roll_feats = {
    'roll_goals_scored_5': [], 'roll_goals_conceded_5': [],
    'roll_goals_scored_10': [], 'roll_goals_conceded_10': [],
    'roll_points_5': [], 'roll_points_10': [],
    'roll_win_rate_5': [], 'roll_win_rate_10': [],
    'roll_gd_5': [], 'roll_gd_10': [],
    'opp_roll_goals_scored_5': [], 'opp_roll_goals_conceded_5': [],
    'opp_roll_points_5': [], 'opp_roll_win_rate_5': [], 'opp_roll_gd_5': [],
}

def get_rolling(history, window):
    """Ambil statistik dari N pertandingan terakhir."""
    recent = history[-window:] if len(history) >= window else history
    if not recent:
        return {
            'goals_scored': np.nan, 'goals_conceded': np.nan,
            'points': np.nan, 'win_rate': np.nan, 'gd': np.nan
        }
    goals_s = np.mean([r['gs'] for r in recent])
    goals_c = np.mean([r['gc'] for r in recent])
    pts     = np.mean([r['pts'] for r in recent])
    wr      = np.mean([r['win'] for r in recent])
    gd      = np.mean([r['gd'] for r in recent])
    return {'goals_scored': goals_s, 'goals_conceded': goals_c,
            'points': pts, 'win_rate': wr, 'gd': gd}

for idx, row in df.iterrows():
    team = row['team']
    opp  = row['opponent']

    # ambil statistik SEBELUM pertandingan ini
    t5  = get_rolling(team_history[team], 5)
    t10 = get_rolling(team_history[team], 10)
    o5  = get_rolling(team_history[opp],  5)

    roll_feats['roll_goals_scored_5'].append(t5['goals_scored'])
    roll_feats['roll_goals_conceded_5'].append(t5['goals_conceded'])
    roll_feats['roll_points_5'].append(t5['points'])
    roll_feats['roll_win_rate_5'].append(t5['win_rate'])
    roll_feats['roll_gd_5'].append(t5['gd'])

    roll_feats['roll_goals_scored_10'].append(t10['goals_scored'])
    roll_feats['roll_goals_conceded_10'].append(t10['goals_conceded'])
    roll_feats['roll_points_10'].append(t10['points'])
    roll_feats['roll_win_rate_10'].append(t10['win_rate'])
    roll_feats['roll_gd_10'].append(t10['gd'])

    roll_feats['opp_roll_goals_scored_5'].append(o5['goals_scored'])
    roll_feats['opp_roll_goals_conceded_5'].append(o5['goals_conceded'])
    roll_feats['opp_roll_points_5'].append(o5['points'])
    roll_feats['opp_roll_win_rate_5'].append(o5['win_rate'])
    roll_feats['opp_roll_gd_5'].append(o5['gd'])

    # update riwayat hanya jika ada hasil nyata
    if pd.notna(row['team_goals']) and pd.notna(row['opp_goals']):
        gt, go = row['team_goals'], row['opp_goals']

        if gt > go:
            pts_t, pts_o, win_t, win_o = 3, 0, 1, 0
        elif gt == go:
            pts_t, pts_o, win_t, win_o = 1, 1, 0, 0
        else:
            pts_t, pts_o, win_t, win_o = 0, 3, 0, 1

        team_history[team].append({'gs': gt, 'gc': go, 'pts': pts_t, 'win': win_t, 'gd': gt-go})
        team_history[opp].append({'gs': go, 'gc': gt, 'pts': pts_o, 'win': win_o, 'gd': go-gt})

for k, v in roll_feats.items():
    df[k] = v

# Isi NaN rolling dengan median (tim baru / cold start)
roll_cols = list(roll_feats.keys())
for col in roll_cols:
    med = df[col].median()
    df[col] = df[col].fillna(med)

print(f"       ✓ Rolling stats (window 5 & 10) selesai untuk {len(df):,} baris")

# ─────────────────────────────────────────────
# 4. TOURNAMENT WEIGHT
# ─────────────────────────────────────────────
print("\n[TOURNAMENT] Mapping bobot turnamen...")

# Semakin penting turnamen → skor lebih serius → prediksi lebih sulit
TOURNAMENT_WEIGHT = {
    # Tier 1 – Kompetisi resmi tertinggi (bobot penuh)
    'FIFA World Cup': 1.00,
    'Copa América': 0.95,
    'UEFA Euro': 0.95,
    'AFC Asian Cup': 0.90,
    'Africa Cup of Nations': 0.90,
    'African Cup of Nations': 0.90,
    'CONCACAF Gold Cup': 0.90,
    'OFC Nations Cup': 0.85,

    # Tier 2 – Kualifikasi resmi
    'FIFA World Cup qualification': 0.85,
    'UEFA Euro qualification': 0.80,
    'AFC Asian Cup qualification': 0.75,
    'African Cup of Nations qualification': 0.75,
    'CONCACAF Gold Cup qualification': 0.70,
    'OFC Nations Cup qualification': 0.65,
    'Copa América qualification': 0.70,
    'FIFA World Cup qualification (intercontinental)': 0.75,

    # Tier 3 – Kompetisi regional
    'UEFA Nations League': 0.80,
    'CONCACAF Nations League': 0.75,
    'AFC Challenge Cup': 0.65,
    'COSAFA Cup': 0.60,
    'CECAFA Cup': 0.60,
    'CFU Caribbean Cup': 0.60,
    'UNCAF Nations Cup': 0.60,
    'WAFU Cup of Nations': 0.60,
    'South Asian Football Federation Cup': 0.60,

    # Tier 4 – Kompetisi minor / invitasi
    'Algarve Cup': 0.45,
    'SheBelieves Cup': 0.45,
    'King Cup': 0.45,
    'Merdeka Tournament': 0.45,
    'Island Games': 0.40,
    'CONIFA World Football Cup': 0.40,
    'CONIFA European Football Cup': 0.40,
    'British Home Championship': 0.50,
    'Asian Games': 0.55,

    # Tier 5 – Friendly
    'Friendly': 0.30,
}

df['tournament_weight'] = df['tournament'].map(TOURNAMENT_WEIGHT).fillna(0.50)
print(f"       ✓ Tournament weight: {df['tournament_weight'].value_counts().head(5).to_dict()}")

# ─────────────────────────────────────────────
# 5. HOME ADVANTAGE INTERACTION
# ─────────────────────────────────────────────
print("\n[HOME] Membuat fitur interaksi home advantage...")

df['home_elo_boost']       = df['elo_diff'] + (df['is_home'] * 100)
df['home_form_advantage']  = df['roll_points_5'] * df['is_home']
df['home_goals_avg']       = df['roll_goals_scored_5'] * (1 + 0.15 * df['is_home'])
df['away_fatigue']         = df['distance_travel_team'] * (1 - df['is_home'])
df['neutral_elo_pure']     = df['elo_diff'] * df['neutral']
df['home_win_prob']        = df['elo_win_prob'] + (0.05 * df['is_home']) - (0.05 * (1-df['is_home']))
df['home_win_prob']        = df['home_win_prob'].clip(0.01, 0.99)

print(f"       ✓ 6 fitur home advantage dibuat")

# ─────────────────────────────────────────────
# 6. ATTACK vs DEFENSE INTERACTION
# ─────────────────────────────────────────────
print("\n[ATTACK/DEF] Fitur perbandingan serangan vs pertahanan...")

df['attack_vs_opp_defense']  = df['roll_goals_scored_5'] - df['opp_roll_goals_conceded_5']
df['opp_attack_vs_defense']  = df['opp_roll_goals_scored_5'] - df['roll_goals_conceded_5']
df['form_diff_5']            = df['roll_points_5'] - df['opp_roll_points_5']
df['form_diff_10']           = df['roll_points_10'] - df['opp_roll_points_10']
df['gd_form_diff']           = df['roll_gd_5'] - df['opp_roll_gd_5']
df['win_rate_diff']          = df['roll_win_rate_5'] - df['opp_roll_win_rate_5']
df['goals_ratio']            = (df['roll_goals_scored_5'] + 0.5) / (df['opp_roll_goals_conceded_5'] + 0.5)
df['concede_ratio']          = (df['roll_goals_conceded_5'] + 0.5) / (df['opp_roll_goals_scored_5'] + 0.5)

print(f"       ✓ 8 fitur attack/defense interaction dibuat")

# ─────────────────────────────────────────────
# 7. CONFEDERATION ENCODING
# ─────────────────────────────────────────────
print("\n[CONFEDERATION] Encoding konfederasi...")

CONF_STRENGTH = {
    'UEFA': 6,    # strongest
    'CONMEBOL': 5,
    'CAF': 4,
    'CONCACAF': 3,
    'AFC': 2,
    'OFC': 1,
    'Unknown': 0
}

df['conf_strength_team']  = df['confederation_team'].map(CONF_STRENGTH).fillna(0)
df['conf_strength_opp']   = df['confederation_opp'].map(CONF_STRENGTH).fillna(0)
df['conf_diff']           = df['conf_strength_team'] - df['conf_strength_opp']
df['same_confederation']  = (df['confederation_team'] == df['confederation_opp']).astype(int)
df['intra_conf_match']    = df['same_confederation'] * df['conf_strength_team']

print(f"       ✓ 5 fitur confederation dibuat")

# ─────────────────────────────────────────────
# 8. FITUR GEOGRAFIS & SOSIO-EKONOMI
# ─────────────────────────────────────────────
print("\n[GEO/ECON] Fitur geografis dan ekonomi...")

df['altitude_bucket']      = pd.cut(df['altitude_venue'],
                                    bins=[-1, 500, 1500, 2500, 9999],
                                    labels=[0, 1, 2, 3]).astype(float)
df['high_altitude']        = (df['altitude_venue'] > 2000).astype(int)
df['gdp_ratio']            = np.log1p(df['gdp_per_capita_team']) - np.log1p(df['gdp_per_capita_opp'])
df['population_ratio']     = np.log1p(df['population_team']) - np.log1p(df['population_opp'])
df['distance_advantage']   = df['distance_travel_opp'] - df['distance_travel_team']
df['temp_extreme']         = (df['temperature_venue'].abs() > 30).astype(int)

print(f"       ✓ 6 fitur geo/ekonomi dibuat")

# ─────────────────────────────────────────────
# 9. TEMPORAL FEATURES
# ─────────────────────────────────────────────
print("\n[TEMPORAL] Fitur waktu...")

df['year']          = df['date'].dt.year
df['month']         = df['date'].dt.month
df['decade']        = (df['date'].dt.year // 10) * 10
df['is_modern_era'] = (df['year'] >= 1990).astype(int)
df['season_q']      = df['month'].apply(lambda m: (m - 1) // 3 + 1)  # quarter

print(f"       ✓ 5 fitur temporal dibuat")

# ─────────────────────────────────────────────
# 10. GENDER ENCODING
# ─────────────────────────────────────────────
df['gender_encoded'] = (df['gender'] == 'M').astype(int)

# ─────────────────────────────────────────────
# 11. GABUNGKAN DENGAN FITUR ORIGINAL TRAIN
# ─────────────────────────────────────────────
# Untuk train, isi kolom rolling asli yang sudah ada (fallback ke rekonstruksi jika NaN)
ROLL_ORIG_MAP = {
    'team_points_last5':     'roll_points_5',
    'team_avg_goals_last5':  'roll_goals_scored_5',
    'team_avg_conceded_last5': 'roll_goals_conceded_5',
    'team_win_rate_last10':  'roll_win_rate_10',
    'team_gd_last5':         'roll_gd_5',
}
for orig, reco in ROLL_ORIG_MAP.items():
    if orig in df.columns:
        df[orig] = df[orig].combine_first(df[reco])

# ─────────────────────────────────────────────
# 12. FINAL FEATURE LIST
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # ELO
    'elo_team_final', 'elo_opponent_final', 'elo_diff', 'elo_win_prob',
    'elo_reconstructed_team', 'elo_reconstructed_opp', 'elo_diff_reconstructed',

    # Rolling Stats (rekonstruksi)
    'roll_goals_scored_5', 'roll_goals_conceded_5', 'roll_points_5',
    'roll_win_rate_5', 'roll_gd_5',
    'roll_goals_scored_10', 'roll_goals_conceded_10', 'roll_points_10',
    'roll_win_rate_10', 'roll_gd_10',
    'opp_roll_goals_scored_5', 'opp_roll_goals_conceded_5',
    'opp_roll_points_5', 'opp_roll_win_rate_5', 'opp_roll_gd_5',

    # Home advantage
    'is_home', 'neutral', 'home_elo_boost', 'home_form_advantage',
    'home_goals_avg', 'away_fatigue', 'home_win_prob',

    # Attack/Defense
    'attack_vs_opp_defense', 'opp_attack_vs_defense',
    'form_diff_5', 'form_diff_10', 'gd_form_diff', 'win_rate_diff',
    'goals_ratio', 'concede_ratio',

    # Tournament
    'tournament_weight',

    # Confederation
    'conf_strength_team', 'conf_strength_opp', 'conf_diff',
    'same_confederation', 'intra_conf_match',

    # Geo/Econ
    'altitude_venue', 'altitude_bucket', 'high_altitude',
    'gdp_ratio', 'population_ratio', 'distance_advantage',
    'distance_travel_team', 'distance_travel_opp', 'temperature_venue',
    'temp_extreme',

    # Temporal
    'year', 'month', 'decade', 'is_modern_era', 'season_q',

    # Gender
    'gender_encoded',

    # Original rolling dari train (kalau tersedia)
    'team_points_last5', 'team_avg_goals_last5', 'team_gd_last5',
    'h2h_points_last5', 'h2h_gd_last5',
    'days_since_last_match_team',
]

# Filter hanya kolom yang ada
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
META_COLS    = ['Id', 'match_id', 'date', 'team', 'opponent', 'tournament',
                'confederation_team', 'confederation_opp',
                'team_goals', 'opp_goals', 'is_train']

all_cols = META_COLS + [c for c in FEATURE_COLS if c not in META_COLS]

print(f"\n[FITUR] Total fitur siap: {len(FEATURE_COLS)}")
print(f"        Fitur: {FEATURE_COLS[:10]} ... (+{len(FEATURE_COLS)-10} lainnya)")

# ─────────────────────────────────────────────
# 13. SPLIT KEMBALI & SIMPAN
# ─────────────────────────────────────────────
df_out       = df[[c for c in all_cols if c in df.columns]].copy()
train_fe     = df_out[df_out['is_train'] == 1].drop(columns=['is_train'])
test_fe      = df_out[df_out['is_train'] == 0].drop(columns=['is_train', 'team_goals', 'opp_goals'])

train_fe.to_csv('./data/processed/train_features.csv', index=False)
test_fe.to_csv('./data/processed/test_features.csv', index=False)

print(f"\n[SIMPAN] train_features.csv → {len(train_fe):,} baris × {len(train_fe.columns)} kolom")
print(f"         test_features.csv  → {len(test_fe):,} baris × {len(test_fe.columns)} kolom")

# ─────────────────────────────────────────────
# 14. RINGKASAN & VALIDASI
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RINGKASAN FEATURE ENGINEERING")
print("="*60)

feat_only = [c for c in FEATURE_COLS if c in train_fe.columns]
nan_pct = train_fe[feat_only].isna().mean() * 100
high_nan = nan_pct[nan_pct > 5]
if len(high_nan) > 0:
    print(f"\n⚠️  Fitur dengan NaN >5% di train:")
    for col, pct in high_nan.items():
        print(f"   {col}: {pct:.1f}%")
else:
    print("\n✓  Tidak ada fitur dengan NaN >5%")

# Korelasi fitur dengan target
print("\n[KORELASI] Top 10 fitur vs team_goals (train):")
corr = train_fe[feat_only + ['team_goals']].corr()['team_goals'].drop('team_goals')
top_corr = corr.abs().sort_values(ascending=False).head(10)
for feat, val in top_corr.items():
    print(f"   {feat:40s}: {corr[feat]:+.4f}")

print("\n" + "="*60)
print("STATUS: FASE 2 SELESAI")
print("Output: data/processed/train_features.csv")
print("        data/processed/test_features.csv")
print("="*60)