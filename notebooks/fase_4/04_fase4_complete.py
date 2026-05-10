import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.metrics import kalkulasi_aw_mae

#konfigurasi global: parameter dasar untuk fase 4 baseline computation
#mendefinisikan konstanta yang dipakai di seluruh script untuk consistency dan reproducibility
N_SPLITS    = 5
MODELS_DIR  = 'models'
ROLL_WINDOWS = [3, 5, 10]

os.makedirs(MODELS_DIR,       exist_ok=True)
os.makedirs('submissions',    exist_ok=True)
os.makedirs('reports',        exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

START_TIME = datetime.now()

#daftar lokasi file test mentah yang dicek secara berurutan
#script akan mencoba setiap path hingga menemukan file yang valid
#urutan prioritas: dari path yang paling standard ke alternatif backup
RAW_TEST_CANDIDATES = [
    'data/raw/test.csv',
    'data/test.csv',
    'data/raw/test_raw.csv',
    'test.csv',
]

def separator(title=""):
    print(f"{'=' * 60}")
    if title:
        pad = (58 - len(title)) // 2
        print(f"{'=' * pad} {title} {'=' * pad}")
        print(f"{'=' * 60}")

def elapsed():
    secs = (datetime.now() - START_TIME).seconds
    return f"{secs // 60}m {secs % 60}s"

print("FASE 4 — BASELINE COMPUTATION")
print(f"  mulai    : {START_TIME.strftime('%H:%M:%S')}")
print("  tujuan   : establish baseline AW-MAE metric untuk reference phase 5 models")
print("  strategi : evaluate 4 simple baseline strategies untuk set success criterion")

#step 4A: load engineered training data dan perform basic data validation
#tahap ini memastikan semua required columns ada, missing values ditangani dengan proper semantic
#kami menggunakan fillna(0) untuk h2h karena semantik tidak ada riwayat = 0, bukan median
print("load train_engineered.csv ...")
train = pd.read_csv('data/processed/train_engineered.csv')
print(f"shape: {train.shape[0]:,} x {train.shape[1]} columns")

#validasi: pastikan kolom yang dibutuhkan untuk AW-MAE metric ada di dataframe
#kami butuh team_goals dan opp_goals (target), tournament_weight (untuk weighting)
required_cols = ['team_goals', 'opp_goals', 'tournament_weight', 'team', 'opponent']
missing = [c for c in required_cols if c not in train.columns]
if missing:
    raise ValueError(f"missing kolom required: {missing}")
print("semua kolom required ada")

#validasi: pastikan semua kolom numerik ada di dataframe
h2h_cols = ['h2h_gd_last5', 'h2h_points_last5']
for col in h2h_cols:
    if col in train.columns:
        n = train[col].isna().sum()
        train[col] = train[col].fillna(0)
        print(f"{col}: {n:,} null → 0")

#validasi: pastikan semua kolom rank ada di dataframe
rank_cols = ['rank_team', 'rank_opponent']
for col in rank_cols:
    if col in train.columns:
        n = train[col].isna().sum()
        med = train[col].median()
        train[col] = train[col].fillna(med)
        print(f"{col}: {n:,} null → {med:.0f}")

#validasi: pastikan tournament_weight ada di dataframe
if 'tournament_weight' in train.columns and train['tournament_weight'].isna().sum() > 0:
    train['tournament_weight'] = train['tournament_weight'].fillna(1.2)
    print(f"tournament_weight: filled with 1.2 (default)")

#step 4B: validasi AW-MAE metric logic, lalu compute multiple baseline strategies
#sanity check memastikan metrik berfungsi sebagaimana didesain
#baseline strategies yang kami test: flat 1-1, flat 1-0, global mean, per-team mean
#score terbaik menjadi threshold: phase 5 model harus beat baseline ini
y_true = train[['team_goals', 'opp_goals']].values
w = np.asarray(train['tournament_weight'].values, dtype=float)

print("struktur data yang akan kami gunakan:")
print(f"y_true: {y_true.shape} (actual match scores dari training)")
print(f"weights: {w.shape} (tournament importance untuk weighted loss)")
print(f"weight range: [{w.min():.2f}, {w.max():.2f}] (dari 0.96 friendly hingga 2.00 world cup)")


y_perfect = y_true.copy().astype(float)
awmae_perfect = kalkulasi_aw_mae(y_true, y_perfect, w)
assert awmae_perfect < 1e-6, f"BUG: Perfect pred should be ~0, got {awmae_perfect}"
print(f"OK - perfect prediction (prediksi tepat 100%) → {awmae_perfect:.6f} (near zero)")

y_worst = np.zeros_like(y_true, dtype=float)
y_worst[:, 0] = 10
awmae_worst = kalkulasi_aw_mae(y_true, y_worst, w)
assert awmae_worst > 5.0, f"BUG: Worst pred should be > 5, got {awmae_worst}"
print(f"OK - worst prediction (selalu prediksi 10-0) → {awmae_worst:.4f} (sangat tinggi)")

y_c = y_true[:100].copy().astype(float) + 1.0
y_f = y_true[:100].copy().astype(float) + 10.0
assert kalkulasi_aw_mae(y_true[:100], y_c, w[:100]) < kalkulasi_aw_mae(y_true[:100], y_f, w[:100])
print("OK - metric monotonicity verified (prediksi lebih dekat = score lebih rendah)")

baselines = []
global_mean_team = train['team_goals'].mean()
global_mean_opp = train['opp_goals'].mean()

#baseline 1: prediksi flat score 1-1 untuk semua match
#1-1 adalah skor paling sering terjadi dalam data (draw dengan low scoring)
pred_1_1 = np.ones((len(train), 2), dtype=float)
awmae_1_1 = kalkulasi_aw_mae(y_true, pred_1_1, w)
baselines.append({'model': 'flat_always_1_1', 'awmae': round(awmae_1_1, 4), 'notes': 'skor paling umum'})
print(f"flat_always_1_1 → {awmae_1_1:.4f}")

#baseline 2: prediksi flat score 1-0 untuk semua match
#1-0 reflect home team advantage hypothesis (tim kandang menang 1-0)
pred_1_0 = np.column_stack([np.ones(len(train)), np.zeros(len(train))]).astype(float)
awmae_1_0 = kalkulasi_aw_mae(y_true, pred_1_0, w)
baselines.append({'model': 'flat_always_1_0', 'awmae': round(awmae_1_0, 4), 'notes': 'hipotesis home advantage'})
print(f"flat_always_1_0 → {awmae_1_0:.4f}")

#baseline 3: prediksi rata-rata gol global dari seluruh dataset
#strategi ini menggunakan informasi agregat tanpa mempertimbangkan karakteristik tim individual
pred_global = np.full((len(train), 2), [round(global_mean_team), round(global_mean_opp)], dtype=float)
awmae_global = kalkulasi_aw_mae(y_true, pred_global, w)
baselines.append({'model': 'flat_global_mean', 'awmae': round(awmae_global, 4), 
                  'notes': f'rata-rata global: ({round(global_mean_team)}, {round(global_mean_opp)})'})
print(f"flat_global_mean → {awmae_global:.4f}")

# Calculate expanding historical mean shifted by 1 (strict past only)
train['team_past_mean'] = train.groupby('team')['team_goals'].transform(lambda x: x.shift(1).expanding().mean())
train['opp_past_mean'] = train.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).expanding().mean())

pred_team_avg = np.column_stack((
    train['team_past_mean'].fillna(global_mean_team).round().to_numpy(dtype=float),
    train['opp_past_mean'].fillna(global_mean_opp).round().to_numpy(dtype=float)
))

awmae_team_avg = kalkulasi_aw_mae(y_true, pred_team_avg, w)
baselines.append({'model': 'per_team_mean_proper', 'awmae': round(awmae_team_avg, 4), 
                  'notes': 'Expanding mean (past only, no leakage)'})
print(f"per_team_mean_proper (time-aware) → {awmae_team_avg:.4f}")

#simpan hasil baseline ke CSV untuk dokumentasi dan comparison dengan phase 5 model
baseline_df = pd.DataFrame(baselines)
baseline_df.to_csv('data/processed/baseline_scores_detailed.csv', index=False)
print("baseline_scores_detailed.csv tersimpan")
print(baseline_df.to_string(index=False))

#identifikasi baseline terbaik (AW-MAE terendah) yang akan menjadi success criterion
#phase 5 model harus mengalahkan baseline ini agar dianggap lebih baik dari strategy sederhana
best_baseline = baseline_df.loc[baseline_df['awmae'].idxmin()]
print(f"baseline terbaik: {best_baseline['model']} = {best_baseline['awmae']:.4f}")
print(f"→ phase 5 model HARUS score di bawah {best_baseline['awmae']:.4f} untuk lebih baik dari baseline")

#ringkasan hasil phase 4: baseline metrics yang akan dipakai untuk evaluasi phase 5
#metrik ini established sebagai benchmark untuk model development di fase berikutnya
print("SUMMARY PHASE 4")
print("variable target: team_goals, opp_goals (count data, Poisson)")
print(f"training data: {len(train):,} matches")
print("Loss function: AW-MAE (weighted, discrete penalty)")
print("hasil:")
for _, row in baseline_df.iterrows():
    print(f"{row['model']:<35} → {row['awmae']:>6.4f}")
print("  kriteria sukses untuk phase 5:")
print(f"  → LightGBM model AW-MAE harus < {best_baseline['awmae']:.4f} untuk lebih baik dari baseline")