# =============================================================================
# notebooks/fase_4/04b_sanity_check_baseline.py
# STEP 4B — AW-MAE Sanity Check + Flat Baselines
# Tujuan: Verifikasi metrik berjalan benar, hitung floor reference baselines
# Output: data/processed/baseline_scores_summary.csv
# Estimasi waktu: ~20 menit
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np

from src.metrics import kalkulasi_aw_mae

# ---------------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------------
TRAIN_PATH  = 'data/processed/train_aligned.csv'
OUTPUT_PATH = 'data/processed/baseline_scores_summary.csv'

print("=" * 60)
print("STEP 4B — AW-MAE SANITY CHECK + FLAT BASELINES")
print("=" * 60)

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
print(f"\n[1/5] Loading {TRAIN_PATH} ...")
train   = pd.read_csv(TRAIN_PATH)
train   = train.sort_values('date').reset_index(drop=True)
y_true  = train[['team_goals', 'opp_goals']].values
weights = train['tournament_weight'].values
print(f"      {len(train):,} baris loaded")

# ---------------------------------------------------------------------------
# SANITY CHECK #1: Perfect prediction → AW-MAE harus = 0.0
# ---------------------------------------------------------------------------
print(f"\n[2/5] Sanity checks AW-MAE ...")

y_perfect = y_true.copy().astype(float)
score_perfect = kalkulasi_aw_mae(y_true, y_perfect, weights)
assert score_perfect == 0.0, f"BUG: Perfect pred menghasilkan {score_perfect}, bukan 0!"
print(f"      [OK] Perfect prediction  → AW-MAE = {score_perfect:.4f}  (expected: 0.0000)")

# SANITY CHECK #2: Semua prediksi salah besar → harus > 5.0
y_worst       = np.zeros_like(y_true, dtype=float)
y_worst[:, 0] = 10
y_worst[:, 1] = 0
score_worst = kalkulasi_aw_mae(y_true, y_worst, weights)
assert score_worst > 5.0, f"BUG: Worst pred menghasilkan {score_worst}, terlalu kecil!"
print(f"      [OK] Always predict 10-0 → AW-MAE = {score_worst:.4f}  (expected: >5.0)")

# SANITY CHECK #3: Prediksi lebih dekat ke benar harus < prediksi jauh salah
y_close    = y_true[:5].copy().astype(float)
y_close   += 1.0                              # off by 1 dari true
y_far      = y_true[:5].copy().astype(float)
y_far     += 10.0                             # off by 10 dari true
w_sub      = weights[:5]
score_close = kalkulasi_aw_mae(y_true[:5], y_close, w_sub)
score_far   = kalkulasi_aw_mae(y_true[:5], y_far,   w_sub)
assert score_close < score_far, "BUG: Prediksi dekat harus lebih baik dari prediksi jauh"
print(f"      [OK] Close pred ({'+1'}):    AW-MAE = {score_close:.4f}"
      f"  < far pred (+10): {score_far:.4f}")

print(f"\n      ✓ Semua sanity checks PASSED — metrik berjalan benar")

# ---------------------------------------------------------------------------
# HITUNG FLAT BASELINES
# ---------------------------------------------------------------------------
print(f"\n[3/5] Hitung flat baselines ...")

results = []

# B1: Always 1-1 (skor paling umum, 9.66% frekuensi historis)
b1      = np.ones((len(train), 2), dtype=float)
s1      = kalkulasi_aw_mae(y_true, b1, weights)
results.append({'model': 'flat_always_1_1', 'awmae': round(s1, 4),
                'notes': 'Skor paling umum (9.66%)'})
print(f"      B1 (always 1-1): AW-MAE = {s1:.4f}")

# B2: Always 1-0
b2      = np.column_stack([np.ones(len(train)), np.zeros(len(train))]).astype(float)
s2      = kalkulasi_aw_mae(y_true, b2, weights)
results.append({'model': 'flat_always_1_0', 'awmae': round(s2, 4),
                'notes': 'Home win bias'})
print(f"      B2 (always 1-0): AW-MAE = {s2:.4f}")

# B3: Always 0-0
b3      = np.zeros((len(train), 2), dtype=float)
s3      = kalkulasi_aw_mae(y_true, b3, weights)
results.append({'model': 'flat_always_0_0', 'awmae': round(s3, 4),
                'notes': 'Ultra-defensive baseline'})
print(f"      B3 (always 0-0): AW-MAE = {s3:.4f}")

# B4: Global mean (rounded)
global_mean_t = round(train['team_goals'].mean())
global_mean_o = round(train['opp_goals'].mean())
b4            = np.column_stack([
    np.full(len(train), global_mean_t),
    np.full(len(train), global_mean_o)
]).astype(float)
s4 = kalkulasi_aw_mae(y_true, b4, weights)
results.append({'model': f'flat_global_mean_{global_mean_t}_{global_mean_o}',
                'awmae': round(s4, 4),
                'notes': f'Global mean: {global_mean_t}-{global_mean_o}'})
print(f"      B4 (global mean {global_mean_t}-{global_mean_o}): AW-MAE = {s4:.4f}")

# ---------------------------------------------------------------------------
# PER-TEAM MEAN BASELINE
# ---------------------------------------------------------------------------
print(f"\n[4/5] Hitung per-team mean baseline ...")

global_mean = train['team_goals'].mean()

# Hitung rata-rata per tim (hanya data historis yang cukup)
team_avg_scored   = train.groupby('team')['team_goals'].mean()
team_avg_conceded = train.groupby('team')['opp_goals'].mean()

pred_team = train['team'].map(team_avg_scored).fillna(global_mean)
pred_opp  = train['opponent'].map(team_avg_scored).fillna(global_mean)

# Blend: rata-rata attack tim + rata-rata yang biasa kemasukan lawan
pred_team_blended = (pred_team + train['opponent'].map(team_avg_conceded).fillna(global_mean)) / 2
pred_opp_blended  = (pred_opp  + train['team'].map(team_avg_conceded).fillna(global_mean)) / 2

b5 = np.column_stack([
    np.round(pred_team_blended).clip(0).astype(int),
    np.round(pred_opp_blended).clip(0).astype(int)
]).astype(float)
s5 = kalkulasi_aw_mae(y_true, b5, weights)
results.append({'model': 'per_team_mean_blended', 'awmae': round(s5, 4),
                'notes': 'Blend attack+defense per tim'})
print(f"      B5 (per-team mean blended): AW-MAE = {s5:.4f}")

# Simple per-team (tanpa blend)
b5b = np.column_stack([
    np.round(pred_team).clip(0).astype(int),
    np.round(pred_opp).clip(0).astype(int)
]).astype(float)
s5b = kalkulasi_aw_mae(y_true, b5b, weights)
results.append({'model': 'per_team_mean_simple', 'awmae': round(s5b, 4),
                'notes': 'Rata-rata gol tim (simple)'})
print(f"      B5b (per-team mean simple): AW-MAE = {s5b:.4f}")

# ---------------------------------------------------------------------------
# DISTRIBUSI TARGET (informasi untuk kalibrasi model nanti)
# ---------------------------------------------------------------------------
print(f"\n[5/5] Statistik distribusi target ...")
print(f"      Team goals — mean: {train['team_goals'].mean():.3f} | "
      f"median: {train['team_goals'].median():.0f} | "
      f"max: {train['team_goals'].max():.0f}")
print(f"      Opp  goals — mean: {train['opp_goals'].mean():.3f} | "
      f"median: {train['opp_goals'].median():.0f} | "
      f"max: {train['opp_goals'].max():.0f}")
win_rate  = (train['team_goals'] > train['opp_goals']).mean()
draw_rate = (train['team_goals'] == train['opp_goals']).mean()
loss_rate = (train['team_goals'] < train['opp_goals']).mean()
print(f"      Outcome dist — Win: {win_rate:.2%} | Draw: {draw_rate:.2%} | Loss: {loss_rate:.2%}")

top10 = (train.groupby(['team_goals','opp_goals'])
         .size().reset_index(name='count')
         .sort_values('count', ascending=False)
         .head(10))
top10['pct'] = (top10['count'] / len(train) * 100).round(2)
print(f"\n      Top 10 skor terbanyak:")
for _, row in top10.iterrows():
    print(f"        {int(row['team_goals'])}-{int(row['opp_goals'])}: "
          f"{int(row['count']):,} match ({row['pct']:.2f}%)")

# ---------------------------------------------------------------------------
# SIMPAN HASIL
# ---------------------------------------------------------------------------
df_results = pd.DataFrame(results)
df_results['rank'] = df_results['awmae'].rank().astype(int)
df_results = df_results.sort_values('awmae')
df_results.to_csv(OUTPUT_PATH, index=False)

# ---------------------------------------------------------------------------
# RINGKASAN
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"STEP 4B SELESAI")
print(f"{'=' * 60}")
print(f"\n  TABEL BASELINE SCORES (diurutkan terbaik):")
print(f"  {'Model':<35} {'AW-MAE':>8}  Notes")
print(f"  {'-'*65}")
for _, row in df_results.iterrows():
    print(f"  {row['model']:<35} {row['awmae']:>8.4f}  {row['notes']}")
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"\n  Target model Fase 4 (LightGBM): AW-MAE < 2.75")
print(f"  Improvement vs best baseline  : "
      f"~{((df_results['awmae'].min() - 2.70) / df_results['awmae'].min() * 100):.1f}% needed")
print(f"\n  → Lanjut ke STEP 4C: 04c_lgbm_baseline.py")
