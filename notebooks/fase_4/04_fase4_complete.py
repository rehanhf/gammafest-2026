# =============================================================================
# notebooks/fase_4/04_fase4_complete.py
# COMBINED — Fase 4 Baseline Model (All Steps: 4A → 4B → 4C → 4D → 4E)
# =============================================================================
# Jalankan file ini untuk eksekusi seluruh Fase 4 sekaligus dari awal sampai
# submission. Identik dengan menjalankan 04a → 04b → 04c → 04d → 04e
# secara berurutan.
#
# Usage:
#   cd gammafest-2026
#   python notebooks/fase_4/04_fase4_complete.py
#
# Estimasi waktu total: 3-4 jam (didominasi Step 4C training)
# Output akhir: submissions/sub_baseline_lgbm_v1.csv
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

from src.metrics import kalkulasi_aw_mae
from src.feature_names import (
    FINAL_FEATURES,
    TEST_RENAME,
    TRAIN_COLS_DROP,
    TRAIN_COLS_RENAME,
    LGB_PARAMS,
    ELO_FORCE_WIN_THRESHOLD,
    ELO_FORCE_LOSS_THRESHOLD,
)

# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------
N_SPLITS       = 5
MODELS_DIR     = 'models'
os.makedirs(MODELS_DIR,       exist_ok=True)
os.makedirs('submissions',    exist_ok=True)
os.makedirs('reports',        exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

START_TIME = datetime.now()

def separator(title=""):
    print(f"\n{'=' * 60}")
    if title:
        pad = (58 - len(title)) // 2
        print(f"{'=' * pad} {title} {'=' * pad}")
        print(f"{'=' * 60}")

def elapsed():
    return f"{(datetime.now() - START_TIME).seconds // 60}m {(datetime.now() - START_TIME).seconds % 60}s"

separator("FASE 4 — BASELINE MODEL (COMPLETE)")
print(f"  Start: {START_TIME.strftime('%H:%M:%S')}")
print(f"  Steps: 4A → 4B → 4C → 4D → 4E")
print(f"  Target CV AW-MAE: < 2.75")

# =============================================================================
# ███████╗████████╗███████╗██████╗      ██╗  ██╗ █████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██║  ██║██╔══██╗
# ███████╗   ██║   █████╗  ██████╔╝    ███████║███████║
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ╚════██║██╔══██║
# ███████║   ██║   ███████╗██║              ██║██║  ██║
# ╚══════╝   ╚═╝   ╚══════╝╚═╝             ╚═╝╚═╝  ╚═╝
# =============================================================================
separator("STEP 4A — Feature Alignment")
print(f"  [{elapsed()}] Mulai ...")

print("\n  [4A-1] Loading train_engineered.csv ...")
train = pd.read_csv('data/processed/train_engineered.csv')
print(f"          Shape awal: {train.shape[0]:,} x {train.shape[1]}")

print("\n  [4A-2] Drop kolom duplikat lama ...")
before = train.shape[1]
train  = train.drop(columns=TRAIN_COLS_DROP, errors='ignore')
print(f"          Kolom: {before} → {train.shape[1]}")

print("\n  [4A-3] Rename kolom .1 ke nama bersih ...")
renamed = {k: v for k, v in TRAIN_COLS_RENAME.items() if k in train.columns}
train   = train.rename(columns=renamed)
remaining_dupes = [c for c in train.columns if '.1' in c]
if remaining_dupes:
    print(f"          WARNING: Masih ada .1: {remaining_dupes}")
else:
    print(f"          OK: Tidak ada kolom .1 tersisa")

print("\n  [4A-4] fillna H2H ...")
for col in ['h2h_gd_last5', 'h2h_points_last5']:
    if col in train.columns:
        n = train[col].isna().sum()
        train[col] = train[col].fillna(0)
        print(f"          {col}: {n:,} null → 0")

print("\n  [4A-5] Buat flag has_h2h_history ...")
if 'h2h_gd_last5' in train.columns:
    train['has_h2h_history'] = (
        (train['h2h_gd_last5'] != 0) | (train['h2h_points_last5'] != 0)
    ).astype(int)
    print(f"          has_h2h_history: {train['has_h2h_history'].mean():.1%} punya riwayat")

print("\n  [4A-6] fillna rank + flag ...")
for col, flag in [('rank_team', 'rank_available_team'), ('rank_opponent', 'rank_available_opp')]:
    if col in train.columns:
        n   = train[col].isna().sum()
        med = train[col].median()
        train[flag] = train[col].notna().astype(int)
        train[col]  = train[col].fillna(med)
        print(f"          {col}: {n:,} null → fillna({med:.0f})")

print("\n  [4A-7] Validasi FINAL_FEATURES ...")
missing = [f for f in FINAL_FEATURES if f not in train.columns]
if missing:
    raise ValueError(f"FINAL_FEATURES hilang: {missing}")
print(f"          OK: Semua {len(FINAL_FEATURES)} fitur tersedia")

print("\n  [4A-8] Simpan train_aligned.csv ...")
train.to_csv('data/processed/train_aligned.csv', index=False)
print(f"          Saved: {train.shape[0]:,} x {train.shape[1]}")

print(f"\n  ✅ STEP 4A SELESAI [{elapsed()}]")

# =============================================================================
# ███████╗████████╗███████╗██████╗      ██╗  ██╗██████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██║  ██║██╔══██╗
# ███████╗   ██║   █████╗  ██████╔╝    ███████║██████╔╝
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ╚════██║██╔══██╗
# ███████║   ██║   ███████╗██║              ██║██████╔╝
# ╚══════╝   ╚═╝   ╚══════╝╚═╝             ╚═╝╚═════╝
# =============================================================================
separator("STEP 4B — Sanity Check + Flat Baselines")
print(f"  [{elapsed()}] Mulai ...")

train_b  = pd.read_csv('data/processed/train_aligned.csv')
train_b  = train_b.sort_values('date').reset_index(drop=True)
y_true_b = train_b[['team_goals', 'opp_goals']].values
w_b      = train_b['tournament_weight'].values

# Sanity checks
print("\n  [4B-1] Sanity checks AW-MAE ...")
y_perfect = y_true_b.copy().astype(float)
assert kalkulasi_aw_mae(y_true_b, y_perfect, w_b) == 0.0, "BUG: Perfect pred != 0"
print(f"          [OK] Perfect prediction → 0.0000")

y_worst = np.zeros_like(y_true_b, dtype=float); y_worst[:, 0] = 10
score_worst = kalkulasi_aw_mae(y_true_b, y_worst, w_b)
assert score_worst > 5.0, "BUG: Worst pred terlalu kecil"
print(f"          [OK] Worst pred (10-0)  → {score_worst:.4f}")

y_c = y_true_b[:5].copy().astype(float) + 1.0
y_f = y_true_b[:5].copy().astype(float) + 10.0
assert kalkulasi_aw_mae(y_true_b[:5], y_c, w_b[:5]) < kalkulasi_aw_mae(y_true_b[:5], y_f, w_b[:5])
print(f"          [OK] Close pred < Far pred — metrik monoton")

# Flat baselines
print("\n  [4B-2] Hitung flat baselines ...")
baselines = []
global_mean = train_b['team_goals'].mean()

configs = [
    (np.ones((len(train_b), 2)),                                           'flat_always_1_1',         'Skor paling umum (9.66%)'),
    (np.column_stack([np.ones(len(train_b)), np.zeros(len(train_b))]),    'flat_always_1_0',         'Home win bias'),
    (np.zeros((len(train_b), 2)),                                          'flat_always_0_0',         'Ultra-defensive'),
    (np.full((len(train_b), 2), [[round(global_mean), round(global_mean)]]), f'flat_global_mean_{round(global_mean)}_{round(global_mean)}', f'Global mean {round(global_mean)}-{round(global_mean)}'),
]
for pred, name, note in configs:
    s = kalkulasi_aw_mae(y_true_b, pred.astype(float), w_b)
    baselines.append({'model': name, 'awmae': round(s, 4), 'notes': note})
    print(f"          {name:<35} → {s:.4f}")

# Per-team baseline
print("\n  [4B-3] Per-team mean baseline ...")
team_avg_s   = train_b.groupby('team')['team_goals'].mean()
team_avg_c   = train_b.groupby('team')['opp_goals'].mean()
pt_blend     = (train_b['team'].map(team_avg_s).fillna(global_mean) +
                train_b['opponent'].map(team_avg_c).fillna(global_mean)) / 2
po_blend     = (train_b['opponent'].map(team_avg_s).fillna(global_mean) +
                train_b['team'].map(team_avg_c).fillna(global_mean)) / 2
b_blend      = np.column_stack([np.round(pt_blend).clip(0), np.round(po_blend).clip(0)]).astype(float)
s_blend      = kalkulasi_aw_mae(y_true_b, b_blend, w_b)
baselines.append({'model': 'per_team_mean_blended', 'awmae': round(s_blend, 4), 'notes': 'Blend attack+defense'})
print(f"          per_team_mean_blended            → {s_blend:.4f}")

pd.DataFrame(baselines).to_csv('data/processed/baseline_scores_summary.csv', index=False)
print(f"\n          Saved: data/processed/baseline_scores_summary.csv")
print(f"\n  ✅ STEP 4B SELESAI [{elapsed()}]")

# =============================================================================
# ███████╗████████╗███████╗██████╗      ██╗  ██╗ ██████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██║  ██║██╔════╝
# ███████╗   ██║   █████╗  ██████╔╝    ███████║██║
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ╚════██║██║
# ███████║   ██║   ███████╗██║              ██║╚██████╗
# ╚══════╝   ╚═╝   ╚══════╝╚═╝             ╚═╝ ╚═════╝
# =============================================================================
separator("STEP 4C — LightGBM Dual Poisson Training")
print(f"  [{elapsed()}] Mulai ... (estimasi 2-3 jam)")
print(f"  Features  : {len(FINAL_FEATURES)}")
print(f"  CV Folds  : {N_SPLITS}-fold TimeSeriesSplit")
print(f"  Objective : Poisson")

train_c = pd.read_csv('data/processed/train_aligned.csv')
train_c = train_c.sort_values('date').reset_index(drop=True)
for f in FINAL_FEATURES:
    train_c[f] = train_c[f].fillna(0)

X  = train_c[FINAL_FEATURES].values
yt = train_c['team_goals'].values.astype(float)
yo = train_c['opp_goals'].values.astype(float)
w  = train_c['tournament_weight'].values

oof_team        = np.zeros(len(train_c))
oof_opp         = np.zeros(len(train_c))
oof_lambda_team = np.zeros(len(train_c))
oof_lambda_opp  = np.zeros(len(train_c))
models_team, models_opp = [], []
cv_scores, fold_details  = [], []

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
print(f"\n  {'─' * 55}")

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val   = X[tr_idx], X[val_idx]
    yt_tr, yt_val = yt[tr_idx], yt[val_idx]
    yo_tr, yo_val = yo[tr_idx], yo[val_idx]
    w_val         = w[val_idx]

    print(f"\n  Fold {fold+1}/{N_SPLITS} | train={len(tr_idx):,} | val={len(val_idx):,} | [{elapsed()}]")

    mt = lgb.LGBMRegressor(**LGB_PARAMS)
    mt.fit(X_tr, yt_tr, eval_set=[(X_val, yt_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    mo = lgb.LGBMRegressor(**LGB_PARAMS)
    mo.fit(X_tr, yo_tr, eval_set=[(X_val, yo_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    lam_t  = mt.predict(X_val).clip(0)
    lam_o  = mo.predict(X_val).clip(0)
    pred_t = np.round(lam_t).astype(int)
    pred_o = np.round(lam_o).astype(int)

    elo_idx    = FINAL_FEATURES.index('elo_diff')
    elo_val    = X_val[:, elo_idx]
    mask_home  = (elo_val >  ELO_FORCE_WIN_THRESHOLD)  & (pred_t <= pred_o)
    mask_away  = (elo_val <  ELO_FORCE_LOSS_THRESHOLD) & (pred_o <= pred_t)
    pred_t[mask_home] = pred_o[mask_home] + 1
    pred_o[mask_away] = pred_t[mask_away] + 1

    oof_team[val_idx]        = pred_t
    oof_opp[val_idx]         = pred_o
    oof_lambda_team[val_idx] = lam_t
    oof_lambda_opp[val_idx]  = lam_o
    models_team.append(mt)
    models_opp.append(mo)

    score       = kalkulasi_aw_mae(
        np.column_stack([yt_val, yo_val]),
        np.column_stack([pred_t, pred_o]).astype(float),
        w_val
    )
    outcome_acc = (np.sign(yt_val - yo_val) == np.sign(pred_t - pred_o)).mean()
    cv_scores.append(score)
    fold_details.append({
        'fold': fold+1, 'n_train': len(tr_idx), 'n_val': len(val_idx),
        'awmae': round(score, 4), 'outcome_acc': round(outcome_acc, 4),
        'best_iter_team': mt.best_iteration_, 'best_iter_opp': mo.best_iteration_,
        'forced_home': int(mask_home.sum()), 'forced_away': int(mask_away.sum()),
    })

    print(f"  AW-MAE: {score:.4f} | Outcome Acc: {outcome_acc:.2%} | "
          f"Iter T={mt.best_iteration_} O={mo.best_iteration_} | "
          f"ELO +{mask_home.sum()} -{mask_away.sum()}")

    with open(os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl'), 'wb') as f: pickle.dump(mt, f)
    with open(os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl'), 'wb') as f: pickle.dump(mo, f)

print(f"\n  {'─' * 55}")
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores)
print(f"\n  CV AW-MAE : {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Folds     : {[f'{s:.4f}' for s in cv_scores]}")

oof_df = train_c[['Id','match_id','team','opponent','date','team_goals','opp_goals','tournament_weight']].copy()
oof_df['pred_team_goals']    = oof_team.astype(int)
oof_df['pred_opp_goals']     = oof_opp.astype(int)
oof_df['lambda_team_goals']  = np.round(oof_lambda_team, 4)
oof_df['lambda_opp_goals']   = np.round(oof_lambda_opp,  4)
oof_df['error_team']         = (oof_df['team_goals'] - oof_df['pred_team_goals']).abs()
oof_df['error_opp']          = (oof_df['opp_goals']  - oof_df['pred_opp_goals']).abs()
oof_df['outcome_true']       = np.sign(oof_df['team_goals']  - oof_df['opp_goals'])
oof_df['outcome_pred']       = np.sign(oof_df['pred_team_goals'] - oof_df['pred_opp_goals'])
oof_df['outcome_correct']    = (oof_df['outcome_true'] == oof_df['outcome_pred']).astype(int)
oof_df.to_csv('data/processed/oof_baseline_lgbm.csv', index=False)

pd.DataFrame(fold_details).to_csv('data/processed/cv_fold_details.csv', index=False)

print(f"\n  OOF saved : data/processed/oof_baseline_lgbm.csv")
print(f"  Models    : {N_SPLITS * 2} files di {MODELS_DIR}/")
print(f"\n  ✅ STEP 4C SELESAI [{elapsed()}]")

# =============================================================================
# ███████╗████████╗███████╗██████╗      ██╗  ██╗██████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██║  ██║██╔══██╗
# ███████╗   ██║   █████╗  ██████╔╝    ███████║██║  ██║
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ╚════██║██║  ██║
# ███████║   ██║   ███████╗██║              ██║██████╔╝
# ╚══════╝   ╚═╝   ╚══════╝╚═╝             ╚═╝╚═════╝
# =============================================================================
separator("STEP 4D — Generate Submission")
print(f"  [{elapsed()}] Mulai ...")

test = pd.read_csv('data/processed/test_features.csv')
print(f"\n  [4D-1] Test: {test.shape[0]:,} baris x {test.shape[1]} kolom")

test = test.rename(columns=TEST_RENAME)
print(f"  [4D-2] Rename kolom: {len(TEST_RENAME)} kolom di-map")

for f in FINAL_FEATURES:
    if f not in test.columns:
        test[f] = 0.0
    test[f] = test[f].fillna(0.0)
X_test = test[FINAL_FEATURES].values
print(f"  [4D-3] X_test shape: {X_test.shape}")

print(f"\n  [4D-4] Ensemble {N_SPLITS} fold models ...")
lam_t_sum = np.zeros(len(test))
lam_o_sum = np.zeros(len(test))
for fold in range(N_SPLITS):
    with open(os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl'), 'rb') as f: mt = pickle.load(f)
    with open(os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl'), 'rb') as f: mo = pickle.load(f)
    lt = mt.predict(X_test).clip(0)
    lo = mo.predict(X_test).clip(0)
    lam_t_sum += lt
    lam_o_sum += lo
    print(f"          Fold {fold+1}: λ_team={lt.mean():.3f} | λ_opp={lo.mean():.3f}")

lam_t_avg  = lam_t_sum / N_SPLITS
lam_o_avg  = lam_o_sum / N_SPLITS
pred_team  = np.round(lam_t_avg).astype(int)
pred_opp   = np.round(lam_o_avg).astype(int)

elo        = test['elo_diff'].values
mask_h     = (elo >  ELO_FORCE_WIN_THRESHOLD)  & (pred_team <= pred_opp)
mask_a     = (elo <  ELO_FORCE_LOSS_THRESHOLD) & (pred_opp  <= pred_team)
pred_team[mask_h] = pred_opp[mask_h]  + 1
pred_opp[mask_a]  = pred_team[mask_a] + 1
print(f"\n  [4D-5] ELO force: +{mask_h.sum()} home | -{mask_a.sum()} away")

pred_team = np.maximum(pred_team, 0)
pred_opp  = np.maximum(pred_opp,  0)

sub = test[['Id']].copy()
sub['team_goals'] = pred_team
sub['opp_goals']  = pred_opp

assert not sub.isnull().any().any(), "ERROR: Ada null!"
assert (sub['team_goals'] >= 0).all() and (sub['opp_goals'] >= 0).all(), "ERROR: Negative!"
assert sub['Id'].is_unique, "ERROR: Id duplikat!"

win_sub  = (sub['team_goals'] >  sub['opp_goals']).mean()
draw_sub = (sub['team_goals'] == sub['opp_goals']).mean()
loss_sub = (sub['team_goals'] <  sub['opp_goals']).mean()

sub.to_csv('submissions/sub_baseline_lgbm_v1.csv', index=False)
print(f"\n  [4D-6] Submission saved: {len(sub):,} baris")
print(f"          Win/Draw/Loss: {win_sub:.1%} / {draw_sub:.1%} / {loss_sub:.1%}")
print(f"          Mean goals  : team={sub['team_goals'].mean():.3f} | opp={sub['opp_goals'].mean():.3f}")
print(f"\n  ✅ STEP 4D SELESAI [{elapsed()}]")

# =============================================================================
# ███████╗████████╗███████╗██████╗      ██╗  ██╗███████╗
# ██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██║  ██║██╔════╝
# ███████╗   ██║   █████╗  ██████╔╝    ███████║█████╗
# ╚════██║   ██║   ██╔══╝  ██╔═══╝     ╚════██║██╔══╝
# ███████║   ██║   ███████╗██║              ██║███████╗
# ╚══════╝   ╚═╝   ╚══════╝╚═╝             ╚═╝╚══════╝
# =============================================================================
separator("STEP 4E — Dokumentasi")
print(f"  [{elapsed()}] Mulai ...")

oof_e     = pd.read_csv('data/processed/oof_baseline_lgbm.csv')
folds_e   = pd.read_csv('data/processed/cv_fold_details.csv')
train_e   = pd.read_csv('data/processed/train_aligned.csv').sort_values('date').reset_index(drop=True)

all_fi    = [pickle.load(open(f'{MODELS_DIR}/lgbm_team_fold{i}.pkl','rb')).feature_importances_ for i in range(N_SPLITS)]
avg_fi    = np.mean(all_fi, axis=0)
fi_df     = pd.DataFrame({'feature': FINAL_FEATURES, 'importance': avg_fi}).sort_values('importance', ascending=False)

cv_m, cv_s = folds_e['awmae'].mean(), folds_e['awmae'].std()
out_acc    = oof_e['outcome_correct'].mean() if 'outcome_correct' in oof_e.columns else folds_e['outcome_acc'].mean()

best_bl    = pd.read_csv('data/processed/baseline_scores_summary.csv')['awmae'].min()
improv     = (best_bl - cv_m) / best_bl * 100

if cv_m < 2.5:   verdict_e = "🏆 SANGAT BAGUS"
elif cv_m < 2.6: verdict_e = "✅ BAGUS"
elif cv_m < 2.75:verdict_e = "✅ OK (memenuhi target)"
else:            verdict_e = "⚠️  Di bawah target"

lines = [
    "# 📊 Fase 4 — Baseline Model Summary Report",
    f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Duration: {elapsed()}",
    "",
    "---",
    "",
    "## 🎯 Overview",
    "",
    "| Item | Detail |",
    "|------|--------|",
    f"| Model | LightGBM Dual Poisson |",
    f"| Fitur | {len(FINAL_FEATURES)} (TIER1 + TIER2 aligned) |",
    f"| CV | {N_SPLITS}-fold TimeSeriesSplit |",
    f"| CV AW-MAE | **`{cv_m:.4f} ± {cv_s:.4f}`** |",
    f"| Outcome Acc | {out_acc:.2%} |",
    f"| Verdict | {verdict_e} |",
    f"| Submission | `submissions/sub_baseline_lgbm_v1.csv` |",
    "",
    "---",
    "",
    "## 📈 Fold Detail",
    "",
    "| Fold | N Train | N Val | AW-MAE | Outcome Acc | Best Iter T/O |",
    "|------|---------|-------|--------|-------------|---------------|",
]
for _, r in folds_e.iterrows():
    lines.append(f"| {int(r['fold'])} | {int(r['n_train']):,} | {int(r['n_val']):,} | "
                 f"`{r['awmae']:.4f}` | {r['outcome_acc']:.2%} | "
                 f"{int(r['best_iter_team'])}/{int(r['best_iter_opp'])} |")
lines += [
    f"| **Mean** | — | — | **`{cv_m:.4f}`** | **{folds_e['outcome_acc'].mean():.2%}** | — |",
    "",
    "---",
    "",
    "## 🔑 Feature Importance (Avg 5 Fold — Team Model)",
    "",
    "| Rank | Feature | Importance |",
    "|------|---------|-----------|",
]
for rank, (_, r) in enumerate(fi_df.iterrows(), 1):
    lines.append(f"| {rank} | `{r['feature']}` | {r['importance']:.0f} |")

lines += [
    "",
    "---",
    "",
    "## 📉 Distribusi Prediksi vs Aktual",
    "",
    "| Outcome | Train Aktual | Test Prediksi |",
    "|---------|-------------|---------------|",
    f"| Win  | {(train_e['team_goals']>train_e['opp_goals']).mean():.2%} | {win_sub:.2%} |",
    f"| Draw | {(train_e['team_goals']==train_e['opp_goals']).mean():.2%} | {draw_sub:.2%} |",
    f"| Loss | {(train_e['team_goals']<train_e['opp_goals']).mean():.2%} | {loss_sub:.2%} |",
    "",
    "---",
    "",
    "## 📁 Output Files",
    "",
    "| File | Keterangan |",
    "|------|------------|",
    "| `data/processed/train_aligned.csv` | Train setelah alignment |",
    "| `data/processed/baseline_scores_summary.csv` | Flat baseline scores |",
    "| `data/processed/oof_baseline_lgbm.csv` | OOF predictions |",
    "| `data/processed/cv_fold_details.csv` | Detail per fold |",
    "| `models/lgbm_[team|opp]_fold[0-4].pkl` | 10 model files |",
    "| `submissions/sub_baseline_lgbm_v1.csv` | **Submission file** |",
    "",
    "---",
    "",
    "## 🚀 Prioritas Fase 5",
    "",
    "1. **Re-engineer TIER4 features ke train** → `xg_proxy`, `momentum`, `roll_wr`, `streak`",
    "2. **Clipping goals test** → `clip(8)` vs `clip(10)` vs no clip",
    "3. **Hyperparameter tuning** dengan Optuna (100 trials)",
    "4. **XGBoost Poisson** sebagai model kedua",
    "5. **Conditional model** → prediksi `total_goals + goal_diff` lalu derive skor",
    "",
    f"> Target Fase 5: CV AW-MAE < 2.40 (saat ini: `{cv_m:.4f}`)",
    "",
    "---",
    "*Auto-generated by `notebooks/fase_4/04_fase4_complete.py`*",
]

with open('reports/fase_4_summary.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"\n  Laporan saved: reports/fase_4_summary.md")
print(f"\n  ✅ STEP 4E SELESAI [{elapsed()}]")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
separator("FASE 4 — COMPLETE SUMMARY")
print(f"""
  ┌──────────────────────────────────────────────────────┐
  │              HASIL AKHIR FASE 4                      │
  ├──────────────────────────────────────────────────────┤
  │  CV AW-MAE       : {cv_m:.4f} ± {cv_s:.4f}              │
  │  Best fold       : Fold {folds_e.loc[folds_e['awmae'].idxmin(),'fold']} ({folds_e['awmae'].min():.4f})                    │
  │  Worst fold      : Fold {folds_e.loc[folds_e['awmae'].idxmax(),'fold']} ({folds_e['awmae'].max():.4f})                    │
  │  Outcome acc     : {out_acc:.2%}                          │
  │  vs best baseline: +{improv:.1f}% improvement              │
  │  Total waktu     : {elapsed()}                           │
  ├──────────────────────────────────────────────────────┤
  │  Verdict: {verdict_e:<42}│
  ├──────────────────────────────────────────────────────┤
  │  OUTPUT:                                             │
  │  ✓ data/processed/train_aligned.csv                  │
  │  ✓ data/processed/oof_baseline_lgbm.csv              │
  │  ✓ models/ (10 pkl files)                            │
  │  ✓ submissions/sub_baseline_lgbm_v1.csv ← SUBMIT!   │
  │  ✓ reports/fase_4_summary.md                         │
  └──────────────────────────────────────────────────────┘

  ⚡ SUBMIT sub_baseline_lgbm_v1.csv ke leaderboard!
  → Catat public score, lanjut ke FASE 5
""")
