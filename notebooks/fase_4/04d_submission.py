# =============================================================================
# notebooks/fase_4/04d_submission.py
# STEP 4D — Generate Test Predictions & Submission
# Tujuan: Load test, rename kolom, ensemble 5 fold, submit
# Output: submissions/sub_baseline_lgbm_v1.csv
# Estimasi waktu: ~30 menit
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import pickle

from src.feature_names import (
    FINAL_FEATURES,
    TEST_RENAME,
    ELO_FORCE_WIN_THRESHOLD,
    ELO_FORCE_LOSS_THRESHOLD,
)

# ---------------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------------
TEST_PATH    = 'data/processed/test_features.csv'
MODELS_DIR   = 'models'
OUTPUT_PATH  = 'submissions/sub_baseline_lgbm_v1.csv'
N_FOLDS      = 5
os.makedirs('submissions', exist_ok=True)

print("=" * 60)
print("STEP 4D — GENERATE SUBMISSION")
print("=" * 60)

# ---------------------------------------------------------------------------
# LOAD TEST
# ---------------------------------------------------------------------------
print(f"\n[1/6] Loading {TEST_PATH} ...")
test = pd.read_csv(TEST_PATH)
print(f"      Test shape : {test.shape[0]:,} baris x {test.shape[1]} kolom")
print(f"      Date range : {test['date'].min()} → {test['date'].max()}")
print(f"      Unique match_ids: {test['match_id'].nunique():,}")

# ---------------------------------------------------------------------------
# RENAME KOLOM TEST → NAMA STANDAR FINAL_FEATURES
# ---------------------------------------------------------------------------
print(f"\n[2/6] Rename kolom test sesuai FINAL_FEATURES ...")
test = test.rename(columns=TEST_RENAME)

renamed_done   = [f"{k} → {v}" for k, v in TEST_RENAME.items() if v in test.columns]
renamed_failed = [f"{k} (not found)" for k, v in TEST_RENAME.items() if v not in test.columns]
for r in renamed_done:   print(f"      ✓ {r}")
for r in renamed_failed: print(f"      ✗ {r}")

# ---------------------------------------------------------------------------
# VALIDASI & FILLNA FINAL_FEATURES DI TEST
# ---------------------------------------------------------------------------
print(f"\n[3/6] Validasi dan fillna FINAL_FEATURES di test ...")
for f in FINAL_FEATURES:
    if f not in test.columns:
        test[f] = 0.0
        print(f"      WARNING: '{f}' tidak ada di test → diisi 0.0")
    elif test[f].isna().any():
        n_null = test[f].isna().sum()
        fill_val = test[f].median() if test[f].notna().any() else 0.0
        test[f] = test[f].fillna(fill_val)
        print(f"      INFO: '{f}' ada {n_null} null → fillna({fill_val:.3f})")
    else:
        pass  # OK

print(f"      Semua {len(FINAL_FEATURES)} FINAL_FEATURES siap di test")
X_test = test[FINAL_FEATURES].values
print(f"      X_test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# LOAD MODELS & ENSEMBLE
# ---------------------------------------------------------------------------
print(f"\n[4/6] Load {N_FOLDS} fold models & ensemble ...")
lambda_team_sum = np.zeros(len(test))
lambda_opp_sum  = np.zeros(len(test))
loaded_folds    = 0

for fold in range(N_FOLDS):
    path_t = os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl')
    path_o = os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl')

    if not os.path.exists(path_t) or not os.path.exists(path_o):
        print(f"      WARNING: Model fold {fold} tidak ditemukan, skip")
        continue

    with open(path_t, 'rb') as f: mt = pickle.load(f)
    with open(path_o, 'rb') as f: mo = pickle.load(f)

    lam_t = mt.predict(X_test).clip(0)
    lam_o = mo.predict(X_test).clip(0)

    lambda_team_sum += lam_t
    lambda_opp_sum  += lam_o
    loaded_folds    += 1
    print(f"      Fold {fold+1}: lambda_team mean={lam_t.mean():.3f} | "
          f"lambda_opp mean={lam_o.mean():.3f}")

assert loaded_folds > 0, "ERROR: Tidak ada model yang berhasil diload!"
lambda_team_avg = lambda_team_sum / loaded_folds
lambda_opp_avg  = lambda_opp_sum  / loaded_folds
print(f"      Ensemble ({loaded_folds} folds): "
      f"team={lambda_team_avg.mean():.3f} | opp={lambda_opp_avg.mean():.3f}")

# ---------------------------------------------------------------------------
# ROUND KE INTEGER
# ---------------------------------------------------------------------------
print(f"\n[5/6] Round ke integer + post-processing ...")
pred_team = np.round(lambda_team_avg).astype(int)
pred_opp  = np.round(lambda_opp_avg).astype(int)

# Post-processing: Outcome force untuk extreme ELO
if 'elo_diff' in test.columns:
    elo         = test['elo_diff'].values
    mask_home   = (elo >  ELO_FORCE_WIN_THRESHOLD)  & (pred_team <= pred_opp)
    mask_away   = (elo <  ELO_FORCE_LOSS_THRESHOLD) & (pred_opp  <= pred_team)
    forced_home = mask_home.sum()
    forced_away = mask_away.sum()
    pred_team[mask_home] = pred_opp[mask_home]  + 1
    pred_opp[mask_away]  = pred_team[mask_away] + 1
    print(f"      ELO force — home win forced: {forced_home} | away win forced: {forced_away}")

# Pastikan tidak ada negative
pred_team = np.maximum(pred_team, 0)
pred_opp  = np.maximum(pred_opp,  0)

# ---------------------------------------------------------------------------
# FORMAT & VALIDASI SUBMISSION
# ---------------------------------------------------------------------------
sub = test[['Id']].copy()
sub['team_goals'] = pred_team
sub['opp_goals']  = pred_opp

# Validasi
assert not sub.isnull().any().any(), "ERROR: Ada null di submission!"
assert (sub['team_goals'] >= 0).all(), "ERROR: Ada negative team_goals!"
assert (sub['opp_goals']  >= 0).all(), "ERROR: Ada negative opp_goals!"
assert sub['Id'].is_unique,            "ERROR: Id tidak unique!"
print(f"      Validasi format: PASSED")

# Distribusi prediksi
win_rate  = (sub['team_goals'] >  sub['opp_goals']).mean()
draw_rate = (sub['team_goals'] == sub['opp_goals']).mean()
loss_rate = (sub['team_goals'] <  sub['opp_goals']).mean()
print(f"\n      Distribusi prediksi test:")
print(f"        Mean team_goals : {sub['team_goals'].mean():.3f}")
print(f"        Mean opp_goals  : {sub['opp_goals'].mean():.3f}")
print(f"        Win  rate pred  : {win_rate:.2%}")
print(f"        Draw rate pred  : {draw_rate:.2%}")
print(f"        Loss rate pred  : {loss_rate:.2%}")
print(f"        Max team_goals  : {sub['team_goals'].max()}")
print(f"        Max opp_goals   : {sub['opp_goals'].max()}")

# Top 10 skor prediksi
top_pred = (sub.groupby(['team_goals','opp_goals'])
            .size().reset_index(name='count')
            .sort_values('count', ascending=False).head(10))
top_pred['pct'] = (top_pred['count'] / len(sub) * 100).round(2)
print(f"\n      Top 10 skor prediksi:")
for _, row in top_pred.iterrows():
    print(f"        {int(row['team_goals'])}-{int(row['opp_goals'])}: "
          f"{int(row['count']):,} ({row['pct']:.2f}%)")

# ---------------------------------------------------------------------------
# SIMPAN
# ---------------------------------------------------------------------------
print(f"\n[6/6] Simpan submission → {OUTPUT_PATH} ...")
sub.to_csv(OUTPUT_PATH, index=False)
print(f"      {len(sub):,} baris tersimpan")
print(f"\n      Preview 5 baris pertama:")
print(sub.head().to_string(index=False))

# ---------------------------------------------------------------------------
# RINGKASAN
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"STEP 4D SELESAI")
print(f"{'=' * 60}")
print(f"  Submission : {OUTPUT_PATH}")
print(f"  Baris      : {len(sub):,}")
print(f"  Folds used : {loaded_folds}/{N_FOLDS}")
print(f"  Win/Draw/Loss: {win_rate:.1%} / {draw_rate:.1%} / {loss_rate:.1%}")
print(f"\n  ⚡ SIAP SUBMIT KE LEADERBOARD!")
print(f"  → Catat public score setelah submit")
print(f"  → Lanjut ke STEP 4E: 04e_dokumentasi.py")
