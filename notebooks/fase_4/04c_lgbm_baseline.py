# =============================================================================
# notebooks/fase_4/04c_lgbm_baseline.py
# STEP 4C — Main Baseline Model: LightGBM Dual Poisson
# Tujuan: Train model utama dengan TimeSeriesSplit 5-fold, simpan OOF + models
# Output: data/processed/oof_baseline_lgbm.csv
#         models/lgbm_team_fold{0-4}.pkl
#         models/lgbm_opp_fold{0-4}.pkl
# Estimasi waktu: ~2-3 jam
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import kalkulasi_aw_mae
from src.feature_names import FINAL_FEATURES, LGB_PARAMS, ELO_FORCE_WIN_THRESHOLD, ELO_FORCE_LOSS_THRESHOLD

# ---------------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------------
TRAIN_PATH  = 'data/processed/train_aligned.csv'
OOF_PATH    = 'data/processed/oof_baseline_lgbm.csv'
MODELS_DIR  = 'models'
N_SPLITS    = 5
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4C — LIGHTGBM DUAL POISSON BASELINE")
print("=" * 60)
print(f"  Features : {len(FINAL_FEATURES)} fitur")
print(f"  CV Folds : {N_SPLITS}-fold TimeSeriesSplit")
print(f"  Objective: Poisson (count data regression)")

# ---------------------------------------------------------------------------
# LOAD & PREPARE
# ---------------------------------------------------------------------------
print(f"\n[1/5] Loading {TRAIN_PATH} ...")
train = pd.read_csv(TRAIN_PATH)
train = train.sort_values('date').reset_index(drop=True)
print(f"      {len(train):,} baris | date: {train['date'].min()} → {train['date'].max()}")

# Fillna semua FINAL_FEATURES dengan 0 (sudah handled di 4A, ini safety net)
for f in FINAL_FEATURES:
    if train[f].isna().any():
        n = train[f].isna().sum()
        print(f"      WARNING: {f} masih ada {n} null → fillna(0)")
        train[f] = train[f].fillna(0)

X  = train[FINAL_FEATURES].values
yt = train['team_goals'].values.astype(float)
yo = train['opp_goals'].values.astype(float)
w  = train['tournament_weight'].values

print(f"      X shape: {X.shape} | yt mean: {yt.mean():.3f} | yo mean: {yo.mean():.3f}")

# ---------------------------------------------------------------------------
# INISIALISASI OOF STORAGE
# ---------------------------------------------------------------------------
oof_team        = np.zeros(len(train))
oof_opp         = np.zeros(len(train))
oof_lambda_team = np.zeros(len(train))   # lambda sebelum rounding
oof_lambda_opp  = np.zeros(len(train))
models_team     = []
models_opp      = []
cv_scores       = []
fold_details    = []

# ---------------------------------------------------------------------------
# TRAINING LOOP: 5-FOLD TIMESERIES CV
# ---------------------------------------------------------------------------
print(f"\n[2/5] Training {N_SPLITS}-fold TimeSeriesSplit ...")
print(f"      {'─'*55}")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val     = X[tr_idx], X[val_idx]
    yt_tr, yt_val   = yt[tr_idx], yt[val_idx]
    yo_tr, yo_val   = yo[tr_idx], yo[val_idx]
    w_val           = w[val_idx]
    n_tr, n_val     = len(tr_idx), len(val_idx)

    print(f"\n      Fold {fold+1}/{N_SPLITS} | train={n_tr:,} | val={n_val:,}")

    # --- Train team_goals model ---
    mt = lgb.LGBMRegressor(**LGB_PARAMS)
    mt.fit(
        X_tr, yt_tr,
        eval_set=[(X_val, yt_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ]
    )

    # --- Train opp_goals model ---
    mo = lgb.LGBMRegressor(**LGB_PARAMS)
    mo.fit(
        X_tr, yo_tr,
        eval_set=[(X_val, yo_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ]
    )

    # --- Predict: lambda (expected goals, float) ---
    lam_t = mt.predict(X_val).clip(0)
    lam_o = mo.predict(X_val).clip(0)

    # --- Round ke integer ---
    pred_t = np.round(lam_t).astype(int)
    pred_o = np.round(lam_o).astype(int)

    # --- Post-processing: Outcome force untuk extreme ELO ---
    elo_idx      = FINAL_FEATURES.index('elo_diff')
    elo_val      = X_val[:, elo_idx]
    mask_home    = (elo_val >  ELO_FORCE_WIN_THRESHOLD)  & (pred_t <= pred_o)
    mask_away    = (elo_val <  ELO_FORCE_LOSS_THRESHOLD) & (pred_o <= pred_t)
    forced_home  = mask_home.sum()
    forced_away  = mask_away.sum()
    pred_t[mask_home] = pred_o[mask_home] + 1
    pred_o[mask_away] = pred_t[mask_away] + 1

    # --- Simpan OOF ---
    oof_team[val_idx]        = pred_t
    oof_opp[val_idx]         = pred_o
    oof_lambda_team[val_idx] = lam_t
    oof_lambda_opp[val_idx]  = lam_o

    models_team.append(mt)
    models_opp.append(mo)

    # --- Hitung AW-MAE fold ---
    y_true_val = np.column_stack([yt_val, yo_val])
    y_pred_val = np.column_stack([pred_t, pred_o]).astype(float)
    score      = kalkulasi_aw_mae(y_true_val, y_pred_val, w_val)
    cv_scores.append(score)

    # Kalkulasi outcome accuracy untuk insight
    gd_true = yt_val - yo_val
    gd_pred = pred_t - pred_o
    sign_t   = np.sign(gd_true)
    sign_p   = np.sign(gd_pred)
    outcome_acc = (sign_t == sign_p).mean()

    fold_details.append({
        'fold':           fold + 1,
        'n_train':        n_tr,
        'n_val':          n_val,
        'awmae':          round(score, 4),
        'outcome_acc':    round(outcome_acc, 4),
        'best_iter_team': mt.best_iteration_,
        'best_iter_opp':  mo.best_iteration_,
        'forced_home':    int(forced_home),
        'forced_away':    int(forced_away),
    })

    print(f"      AW-MAE: {score:.4f} | Outcome Acc: {outcome_acc:.2%} | "
          f"Best iter T={mt.best_iteration_} O={mo.best_iteration_} | "
          f"ELO forced: +{forced_home} -{forced_away}")

    # --- Simpan model ---
    model_path_t = os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl')
    model_path_o = os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl')
    with open(model_path_t, 'wb') as f: pickle.dump(mt, f)
    with open(model_path_o, 'wb') as f: pickle.dump(mo, f)

print(f"\n      {'─'*55}")

# ---------------------------------------------------------------------------
# RINGKASAN CV
# ---------------------------------------------------------------------------
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)

print(f"\n[3/5] Hasil Cross-Validation ...")
print(f"      CV AW-MAE: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"      Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

if cv_mean < 2.5:
    verdict = "🏆 SANGAT BAGUS — Jauh di atas target"
elif cv_mean < 2.6:
    verdict = "✅ BAGUS — Di atas target"
elif cv_mean < 2.75:
    verdict = "✅ OK — Memenuhi target fase 4"
else:
    verdict = "⚠️  Di bawah target — cek feature engineering"
print(f"      Verdict: {verdict}")

# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
print(f"\n[4/5] Feature importance (rata-rata semua fold) ...")
all_importance = np.zeros(len(FINAL_FEATURES))
for m in models_team:
    all_importance += m.feature_importances_
all_importance /= len(models_team)

fi_df = pd.DataFrame({
    'feature':    FINAL_FEATURES,
    'importance': all_importance
}).sort_values('importance', ascending=False)

print(f"      {'Rank':<5} {'Feature':<45} {'Importance':>10}")
print(f"      {'─'*63}")
for rank, (_, row) in enumerate(fi_df.iterrows(), 1):
    print(f"      {rank:<5} {row['feature']:<45} {row['importance']:>10.0f}")

# ---------------------------------------------------------------------------
# SIMPAN OOF
# ---------------------------------------------------------------------------
print(f"\n[5/5] Simpan OOF predictions → {OOF_PATH} ...")
oof_df = train[['Id', 'match_id', 'team', 'opponent', 'date',
                'team_goals', 'opp_goals', 'tournament_weight']].copy()
oof_df['pred_team_goals']   = oof_team.astype(int)
oof_df['pred_opp_goals']    = oof_opp.astype(int)
oof_df['lambda_team_goals'] = np.round(oof_lambda_team, 4)
oof_df['lambda_opp_goals']  = np.round(oof_lambda_opp, 4)

# Error analysis columns (berguna untuk debugging di Fase 5-6)
oof_df['error_team']     = (oof_df['team_goals'] - oof_df['pred_team_goals']).abs()
oof_df['error_opp']      = (oof_df['opp_goals']  - oof_df['pred_opp_goals']).abs()
oof_df['outcome_true']   = np.sign(oof_df['team_goals']  - oof_df['opp_goals'])
oof_df['outcome_pred']   = np.sign(oof_df['pred_team_goals'] - oof_df['pred_opp_goals'])
oof_df['outcome_correct']= (oof_df['outcome_true'] == oof_df['outcome_pred']).astype(int)
oof_df.to_csv(OOF_PATH, index=False)

# Global OOF score
y_true_all  = train[['team_goals','opp_goals']].values
y_pred_all  = np.column_stack([oof_team, oof_opp]).astype(float)
oof_score   = kalkulasi_aw_mae(y_true_all, y_pred_all, w)
outcome_acc_global = oof_df['outcome_correct'].mean()

# ---------------------------------------------------------------------------
# SIMPAN FOLD DETAILS
# ---------------------------------------------------------------------------
fold_df = pd.DataFrame(fold_details)
fold_df.to_csv('data/processed/cv_fold_details.csv', index=False)

# ---------------------------------------------------------------------------
# RINGKASAN AKHIR
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"STEP 4C SELESAI")
print(f"{'=' * 60}")
print(f"  CV AW-MAE (per fold) : {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  OOF AW-MAE (global)  : {oof_score:.4f}")
print(f"  Outcome accuracy OOF : {outcome_acc_global:.2%}")
print(f"  Models saved         : {N_SPLITS * 2} files ({MODELS_DIR}/lgbm_*_fold*.pkl)")
print(f"  OOF saved            : {OOF_PATH}")
print(f"  Fold details saved   : data/processed/cv_fold_details.csv")
print(f"\n  Improvement vs baseline (per-team blended 3.5682):")
print(f"    → {((3.5682 - cv_mean) / 3.5682 * 100):.1f}% lebih baik")
print(f"\n  → Lanjut ke STEP 4D: 04d_submission.py")
