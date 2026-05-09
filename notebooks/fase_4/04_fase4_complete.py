# =============================================================================
# notebooks/fase_4/04_fase4_complete.py  — IMPROVED v2
# Perubahan utama vs v1:
#   + Step 4A+: Rolling features (form, attack/defense strength, Poisson proxy)
#   + Step 4D:  Auto-generate test_features.csv dari raw test data
#   + Fix: variable 'f' collision → pakai 'fh'
#   + Fix: ELO force race condition → pakai snapshot copy
#   + Fix: std() ddof konsisten antara 4C dan 4E
#   + Fix: fold number float → int di final summary
#   + Fix: resource leak pickle di 4E
#   + Fix: feature name warning → fit/predict pakai DataFrame konsisten
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
import warnings
warnings.filterwarnings('ignore')

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
N_SPLITS    = 5
MODELS_DIR  = 'models'
ROLL_WINDOWS = [3, 5, 10]

os.makedirs(MODELS_DIR,       exist_ok=True)
os.makedirs('submissions',    exist_ok=True)
os.makedirs('reports',        exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

START_TIME = datetime.now()

# Kandidat lokasi raw test data — urutan prioritas
RAW_TEST_CANDIDATES = [
    'data/raw/test.csv',
    'data/test.csv',
    'data/raw/test_raw.csv',
    'test.csv',
]

def separator(title=""):
    print(f"\n{'=' * 60}")
    if title:
        pad = (58 - len(title)) // 2
        print(f"{'=' * pad} {title} {'=' * pad}")
        print(f"{'=' * 60}")

def elapsed():
    secs = (datetime.now() - START_TIME).seconds
    return f"{secs // 60}m {secs % 60}s"

# =============================================================================
# ═══════════════════════ HELPER FUNCTIONS ════════════════════════════════════
# =============================================================================

def compute_rolling_features(df, windows=None):
    """
    Hitung rolling per-team stats dari match history.
    df harus punya kolom: team, opponent, date, team_goals, opp_goals.
    Pakai shift(1) untuk cegah data leakage.
    """
    if windows is None:
        windows = ROLL_WINDOWS

    df = df.copy()
    df['_dt'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', '_dt']).reset_index(drop=True)

    # Kolom bantu
    df['_pts'] = np.where(df['team_goals'] > df['opp_goals'], 3,
                 np.where(df['team_goals'] == df['opp_goals'], 1, 0))
    df['_gd']  = df['team_goals'] - df['opp_goals']

    # ── Rolling per window ──────────────────────────────────────────────────
    for w in windows:
        g = df.groupby('team')
        df[f'goals_scored_last{w}']   = g['team_goals'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'goals_conceded_last{w}'] = g['opp_goals'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'points_last{w}']         = g['_pts'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f'gd_last{w}']             = g['_gd'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    # Fillna dengan global mean untuk baris pertama tiap tim (belum ada history)
    global_scored    = df['team_goals'].mean()
    global_conceded  = df['opp_goals'].mean()
    for w in windows:
        df[f'goals_scored_last{w}']   = df[f'goals_scored_last{w}'].fillna(global_scored)
        df[f'goals_conceded_last{w}'] = df[f'goals_conceded_last{w}'].fillna(global_conceded)
        df[f'points_last{w}']         = df[f'points_last{w}'].fillna(1.0)   # prior netral
        df[f'gd_last{w}']             = df[f'gd_last{w}'].fillna(0.0)

    # ── Fitur turunan ────────────────────────────────────────────────────────
    # Form trend: bandingkan short-term vs long-term
    df['form_momentum']  = df['points_last3']       - df['points_last10']
    df['scoring_trend']  = df['goals_scored_last3'] - df['goals_scored_last10']
    df['defense_trend']  = df['goals_conceded_last3']- df['goals_conceded_last10']

    # Over/under proxy
    df['total_goals_last5'] = df['goals_scored_last5'] + df['goals_conceded_last5']

    # Relative attack/defense strength vs global
    gm = max(global_scored, 0.1)
    df['attack_strength']  = df['goals_scored_last5']   / gm
    df['defense_weakness'] = df['goals_conceded_last5'] / gm

    # Win rate (points sebagai proxy: 1.0 = menang semua)
    df['win_rate_last5']  = df['points_last5']  / 3.0
    df['win_rate_last10'] = df['points_last10'] / 3.0

    # H2H × form interaction
    if 'h2h_gd_last5' in df.columns:
        df['h2h_form_ix'] = df['h2h_gd_last5'] * df['points_last5']
    if 'h2h_points_last5' in df.columns:
        df['h2h_pts_form_ix'] = df['h2h_points_last5'] * df['form_momentum']

    # ── Opponent rolling stats (merge on opponent + date) ────────────────────
    opp_src = ['goals_scored_last5', 'goals_conceded_last5',
                'points_last5', 'attack_strength', 'defense_weakness',
                'form_momentum', 'win_rate_last5']
    opp_src = [c for c in opp_src if c in df.columns]

    opp_lookup = df[['team', '_dt'] + opp_src].copy()
    opp_lookup = opp_lookup.rename(columns={'team': 'opponent'})
    opp_lookup = opp_lookup.rename(columns={c: f'opp_{c}' for c in opp_src})

    # Drop kolom duplikat sebelum merge untuk jaga-jaga
    existing_opp_cols = [c for c in [f'opp_{s}' for s in opp_src] if c in df.columns]
    df = df.drop(columns=existing_opp_cols, errors='ignore')
    df = df.merge(opp_lookup, on=['opponent', '_dt'], how='left')

    # Fillna opp stats dengan global mean kalau tim opp tidak ada di lookup
    for c in opp_src:
        oc = f'opp_{c}'
        if oc in df.columns:
            df[oc] = df[oc].fillna(df[c].mean() if c in df.columns else 0)

    # ── Fitur Poisson proxy & advantage ─────────────────────────────────────
    if 'opp_defense_weakness' in df.columns:
        df['poisson_lambda_team'] = df['attack_strength'] * df['opp_defense_weakness']
    if 'opp_attack_strength' in df.columns:
        df['poisson_lambda_opp']  = df['opp_attack_strength'] * df['defense_weakness']

    if 'opp_points_last5' in df.columns:
        df['points_advantage']  = df['points_last5']   - df['opp_points_last5']
    if 'opp_attack_strength' in df.columns:
        df['attack_advantage']  = df['attack_strength'] - df['opp_attack_strength']
    if 'opp_defense_weakness' in df.columns:
        df['defense_advantage'] = df['opp_defense_weakness'] - df['defense_weakness']

    # ── Cleanup temp cols ────────────────────────────────────────────────────
    df = df.drop(columns=['_pts', '_gd', '_dt'], errors='ignore')

    return df


def build_team_snapshot(train_df):
    """
    Buat snapshot statistik tiap tim dari data training (N games terakhir).
    Dipakai untuk enrich test data yang tidak punya match history sendiri.
    """
    snap = {}
    train_sorted = train_df.sort_values('date')
    pts_map = np.where(
        train_sorted['team_goals'] > train_sorted['opp_goals'], 3,
        np.where(train_sorted['team_goals'] == train_sorted['opp_goals'], 1, 0)
    )
    train_sorted = train_sorted.assign(_pts=pts_map)

    global_scored   = train_df['team_goals'].mean()
    global_conceded = train_df['opp_goals'].mean()
    global_pts      = pts_map.mean()
    global_gd       = (train_df['team_goals'] - train_df['opp_goals']).mean()
    gm = max(global_scored, 0.1)

    for team, grp in train_sorted.groupby('team'):
        grp = grp.sort_values('date')
        for w in [3, 5, 10]:
            tail = grp.tail(w)
            snap.setdefault(team, {})[f'goals_scored_last{w}']   = tail['team_goals'].mean()
            snap.setdefault(team, {})[f'goals_conceded_last{w}'] = tail['opp_goals'].mean()
            snap.setdefault(team, {})[f'points_last{w}']         = tail['_pts'].mean()
            snap.setdefault(team, {})[f'gd_last{w}']             = (tail['team_goals'] - tail['opp_goals']).mean()

        s5 = snap[team]
        s5['form_momentum']    = s5['points_last3']        - s5['points_last10']
        s5['scoring_trend']    = s5['goals_scored_last3']  - s5['goals_scored_last10']
        s5['defense_trend']    = s5['goals_conceded_last3']- s5['goals_conceded_last10']
        s5['total_goals_last5']= s5['goals_scored_last5']  + s5['goals_conceded_last5']
        s5['attack_strength']  = s5['goals_scored_last5']  / gm
        s5['defense_weakness'] = s5['goals_conceded_last5']/ gm
        s5['win_rate_last5']   = s5['points_last5']  / 3.0
        s5['win_rate_last10']  = s5['points_last10'] / 3.0

    # Global fallback untuk tim yang tidak ada di training
    global_snap = {
        **{f'goals_scored_last{w}':   global_scored   for w in [3,5,10]},
        **{f'goals_conceded_last{w}': global_conceded for w in [3,5,10]},
        **{f'points_last{w}':         global_pts      for w in [3,5,10]},
        **{f'gd_last{w}':             global_gd       for w in [3,5,10]},
        'form_momentum': 0, 'scoring_trend': 0, 'defense_trend': 0,
        'total_goals_last5': global_scored + global_conceded,
        'attack_strength': 1.0, 'defense_weakness': 1.0,
        'win_rate_last5': global_pts/3.0, 'win_rate_last10': global_pts/3.0,
    }
    return snap, global_snap


def apply_snapshot_to_test(test_df, snap, global_snap, h2h_cols=None):
    """
    Map snapshot statistik ke test dataframe.
    """
    stat_cols = list(next(iter(snap.values())).keys())

    for col in stat_cols:
        test_df[col]         = test_df['team'].map({t: s[col] for t, s in snap.items()}).fillna(global_snap[col])
        test_df[f'opp_{col}']= test_df['opponent'].map({t: s[col] for t, s in snap.items()}).fillna(global_snap[col])

    # Poisson proxy
    if 'opp_defense_weakness' in test_df.columns:
        test_df['poisson_lambda_team'] = test_df['attack_strength'] * test_df['opp_defense_weakness']
    if 'opp_attack_strength' in test_df.columns:
        test_df['poisson_lambda_opp']  = test_df['opp_attack_strength'] * test_df['defense_weakness']

    # Advantage features
    if 'opp_points_last5' in test_df.columns:
        test_df['points_advantage']  = test_df['points_last5']   - test_df['opp_points_last5']
    if 'opp_attack_strength' in test_df.columns:
        test_df['attack_advantage']  = test_df['attack_strength'] - test_df['opp_attack_strength']
    if 'opp_defense_weakness' in test_df.columns:
        test_df['defense_advantage'] = test_df['opp_defense_weakness'] - test_df['defense_weakness']

    # H2H interactions (fillna 0 kalau tidak ada)
    if 'h2h_gd_last5' in test_df.columns:
        test_df['h2h_form_ix']     = test_df['h2h_gd_last5'].fillna(0)  * test_df['points_last5']
    if 'h2h_points_last5' in test_df.columns:
        test_df['h2h_pts_form_ix'] = test_df['h2h_points_last5'].fillna(0) * test_df['form_momentum']

    return test_df


# =============================================================================
separator("FASE 4 — BASELINE MODEL (COMPLETE) v2")
print(f"  Start    : {START_TIME.strftime('%H:%M:%S')}")
print(f"  Steps    : 4A → 4A+ → 4B → 4C → 4D → 4E")
print(f"  Target   : CV AW-MAE < 2.60  (v1 baseline: 2.7064)")
print(f"  Features : {len(FINAL_FEATURES)} base → extended dengan rolling stats")

# =============================================================================
#  STEP 4A — Feature Alignment (sama seperti v1)
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
missing_base = [f for f in FINAL_FEATURES if f not in train.columns]
if missing_base:
    raise ValueError(f"FINAL_FEATURES hilang: {missing_base}")
print(f"          OK: Semua {len(FINAL_FEATURES)} base fitur tersedia")

train.to_csv('data/processed/train_aligned.csv', index=False)
print(f"\n  [4A-8] Saved train_aligned.csv: {train.shape[0]:,} x {train.shape[1]}")
print(f"\n  ✅ STEP 4A SELESAI [{elapsed()}]")

# =============================================================================
#  STEP 4A+ — Rich Feature Engineering (BARU)
# =============================================================================
separator("STEP 4A+ — Rich Feature Engineering")
print(f"  [{elapsed()}] Mulai ...")
print(f"  Hitung rolling stats: {ROLL_WINDOWS} games terakhir per tim")

train_a = pd.read_csv('data/processed/train_aligned.csv')
train_a = train_a.sort_values('date').reset_index(drop=True)
n_before = train_a.shape[1]

print(f"\n  [4A+-1] compute_rolling_features() ...")
train_a = compute_rolling_features(train_a, windows=ROLL_WINDOWS)
n_after = train_a.shape[1]
new_cols = n_after - n_before
print(f"          Kolom baru: {new_cols} | Total: {n_after}")

# Daftar fitur rolling yang berhasil dibuat
rolling_new = [
    c for c in train_a.columns
    if any(c.endswith(f'last{w}') for w in ROLL_WINDOWS)
    or c in ['form_momentum','scoring_trend','defense_trend','total_goals_last5',
             'attack_strength','defense_weakness','win_rate_last5','win_rate_last10',
             'poisson_lambda_team','poisson_lambda_opp',
             'points_advantage','attack_advantage','defense_advantage',
             'h2h_form_ix','h2h_pts_form_ix']
    or c.startswith('opp_')
]
print(f"          Rolling features: {len(rolling_new)}")
for c in sorted(rolling_new)[:5]:
    print(f"            {c}: mean={train_a[c].mean():.3f} | null={train_a[c].isna().sum()}")
print(f"            ... ({len(rolling_new)-5} lainnya)")

# Finalize EXTENDED_FEATURES
EXTENDED_FEATURES = FINAL_FEATURES.copy()
for c in rolling_new:
    if c not in EXTENDED_FEATURES and c in train_a.columns:
        EXTENDED_FEATURES.append(c)

# Final fillna safety net
for f in EXTENDED_FEATURES:
    if f in train_a.columns and train_a[f].isna().any():
        train_a[f] = train_a[f].fillna(0)

print(f"\n  [4A+-2] EXTENDED_FEATURES: {len(EXTENDED_FEATURES)} fitur")
print(f"          ({len(EXTENDED_FEATURES) - len(FINAL_FEATURES)} baru dari rolling)")

train_a.to_csv('data/processed/train_extended.csv', index=False)
print(f"  [4A+-3] Saved train_extended.csv")
print(f"\n  ✅ STEP 4A+ SELESAI [{elapsed()}]")

# =============================================================================
#  STEP 4B — Sanity Check + Flat Baselines
# =============================================================================
separator("STEP 4B — Sanity Check + Flat Baselines")
print(f"  [{elapsed()}] Mulai ...")

train_b  = train_a.copy()
y_true_b = train_b[['team_goals', 'opp_goals']].values
w_b      = train_b['tournament_weight'].values

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

print("\n  [4B-2] Hitung flat baselines ...")
baselines    = []
global_mean  = train_b['team_goals'].mean()
configs = [
    (np.ones((len(train_b), 2)),
     'flat_always_1_1', 'Skor paling umum'),
    (np.column_stack([np.ones(len(train_b)), np.zeros(len(train_b))]),
     'flat_always_1_0', 'Home win bias'),
    (np.full((len(train_b), 2), [[round(global_mean), round(global_mean)]]),
     f'flat_global_mean', f'Global mean'),
]
for pred, name, note in configs:
    s = kalkulasi_aw_mae(y_true_b, pred.astype(float), w_b)
    baselines.append({'model': name, 'awmae': round(s, 4), 'notes': note})
    print(f"          {name:<35} → {s:.4f}")

print("\n  [4B-3] Per-team mean baseline ...")
team_avg_s  = train_b.groupby('team')['team_goals'].mean()
team_avg_c  = train_b.groupby('team')['opp_goals'].mean()
pt_blend    = (train_b['team'].map(team_avg_s).fillna(global_mean) +
               train_b['opponent'].map(team_avg_c).fillna(global_mean)) / 2
po_blend    = (train_b['opponent'].map(team_avg_s).fillna(global_mean) +
               train_b['team'].map(team_avg_c).fillna(global_mean)) / 2
b_blend     = np.column_stack([np.round(pt_blend).clip(0),
                                np.round(po_blend).clip(0)]).astype(float)
s_blend     = kalkulasi_aw_mae(y_true_b, b_blend, w_b)
baselines.append({'model': 'per_team_mean_blended', 'awmae': round(s_blend, 4),
                  'notes': 'Blend attack+defense'})
print(f"          per_team_mean_blended            → {s_blend:.4f}")

pd.DataFrame(baselines).to_csv('data/processed/baseline_scores_summary.csv', index=False)
print(f"\n  ✅ STEP 4B SELESAI [{elapsed()}]")

# =============================================================================
#  STEP 4C — LightGBM Dual Poisson Training
# =============================================================================
separator("STEP 4C — LightGBM Dual Poisson Training")
print(f"  [{elapsed()}] Mulai ...")
print(f"  Features  : {len(EXTENDED_FEATURES)} (base {len(FINAL_FEATURES)} + rolling {len(EXTENDED_FEATURES)-len(FINAL_FEATURES)})")
print(f"  CV Folds  : {N_SPLITS}-fold TimeSeriesSplit")
print(f"  Objective : Poisson")

train_c = train_a.copy().sort_values('date').reset_index(drop=True)
for f in EXTENDED_FEATURES:
    if f not in train_c.columns:
        train_c[f] = 0.0
    train_c[f] = train_c[f].fillna(0)

# ── Pakai DataFrame agar feature names konsisten → hilangkan UserWarning ──────
X_df = train_c[EXTENDED_FEATURES]
yt   = train_c['team_goals'].values.astype(float)
yo   = train_c['opp_goals'].values.astype(float)
w    = train_c['tournament_weight'].values

oof_team        = np.zeros(len(train_c))
oof_opp         = np.zeros(len(train_c))
oof_lambda_team = np.zeros(len(train_c))
oof_lambda_opp  = np.zeros(len(train_c))
models_team, models_opp = [], []
cv_scores, fold_details = [], []

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
print(f"\n  {'─' * 55}")

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_df)):
    X_tr  = X_df.iloc[tr_idx]
    X_val = X_df.iloc[val_idx]
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

    # ── ELO force: snapshot dulu untuk hindari race condition (Bug #5 fix) ──
    elo_vals  = X_val['elo_diff'].values
    mask_home = (elo_vals >  ELO_FORCE_WIN_THRESHOLD)  & (pred_t <= pred_o)
    mask_away = (elo_vals <  ELO_FORCE_LOSS_THRESHOLD) & (pred_o <= pred_t)
    pred_t_snap = pred_t.copy()
    pred_o_snap = pred_o.copy()
    pred_t[mask_home] = pred_o_snap[mask_home] + 1   # pakai snapshot pred_o
    pred_o[mask_away] = pred_t_snap[mask_away] + 1   # pakai snapshot pred_t

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
        'fold': fold + 1, 'n_train': len(tr_idx), 'n_val': len(val_idx),
        'awmae': round(score, 4), 'outcome_acc': round(outcome_acc, 4),
        'best_iter_team': mt.best_iteration_, 'best_iter_opp': mo.best_iteration_,
        'forced_home': int(mask_home.sum()), 'forced_away': int(mask_away.sum()),
    })

    print(f"  AW-MAE: {score:.4f} | Outcome Acc: {outcome_acc:.2%} | "
          f"Iter T={mt.best_iteration_} O={mo.best_iteration_} | "
          f"ELO +{mask_home.sum()} -{mask_away.sum()}")

    # ── Simpan model — pakai 'fh' bukan 'f' untuk hindari variable collision ──
    for tag, mdl in [('team', mt), ('opp', mo)]:
        with open(os.path.join(MODELS_DIR, f'lgbm_{tag}_fold{fold}.pkl'), 'wb') as fh:
            pickle.dump(mdl, fh)

print(f"\n  {'─' * 55}")

# ── Hitung std dengan ddof=0 (population) agar konsisten dengan Step 4E ──────
cv_mean = np.mean(cv_scores)
cv_std  = np.std(cv_scores, ddof=0)
print(f"\n  CV AW-MAE : {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Folds     : {[f'{s:.4f}' for s in cv_scores]}")

if cv_mean < 2.5:   verdict = "🏆 SANGAT BAGUS"
elif cv_mean < 2.6: verdict = "✅ BAGUS"
elif cv_mean < 2.75:verdict = "✅ OK (memenuhi target)"
else:               verdict = "⚠️  Di bawah target"
print(f"  Verdict   : {verdict}")

oof_df = train_c[['Id','match_id','team','opponent','date',
                   'team_goals','opp_goals','tournament_weight']].copy()
oof_df['pred_team_goals']   = oof_team.astype(int)
oof_df['pred_opp_goals']    = oof_opp.astype(int)
oof_df['lambda_team_goals'] = np.round(oof_lambda_team, 4)
oof_df['lambda_opp_goals']  = np.round(oof_lambda_opp,  4)
oof_df['error_team']        = (oof_df['team_goals'] - oof_df['pred_team_goals']).abs()
oof_df['error_opp']         = (oof_df['opp_goals']  - oof_df['pred_opp_goals']).abs()
oof_df['outcome_true']      = np.sign(oof_df['team_goals']  - oof_df['opp_goals'])
oof_df['outcome_pred']      = np.sign(oof_df['pred_team_goals'] - oof_df['pred_opp_goals'])
oof_df['outcome_correct']   = (oof_df['outcome_true'] == oof_df['outcome_pred']).astype(int)
oof_df.to_csv('data/processed/oof_baseline_lgbm.csv', index=False)

pd.DataFrame(fold_details).to_csv('data/processed/cv_fold_details.csv', index=False)
print(f"\n  OOF saved : data/processed/oof_baseline_lgbm.csv")
print(f"  Models    : {N_SPLITS * 2} files di {MODELS_DIR}/")
print(f"\n  ✅ STEP 4C SELESAI [{elapsed()}]")

# =============================================================================
#  STEP 4D — Generate Test Features + Submission
# =============================================================================
separator("STEP 4D — Generate Test Features + Submission")
print(f"  [{elapsed()}] Mulai ...")

# ── 4D-1: Cari / buat test_features.csv ──────────────────────────────────────
TEST_FEATURES_PATH = 'data/processed/test_features.csv'

if os.path.exists(TEST_FEATURES_PATH):
    print(f"\n  [4D-1] test_features.csv ditemukan → load langsung")
    test = pd.read_csv(TEST_FEATURES_PATH)
else:
    print(f"\n  [4D-1] test_features.csv TIDAK ADA → generate dari raw test ...")
    raw_test_path = None
    for candidate in RAW_TEST_CANDIDATES:
        if os.path.exists(candidate):
            raw_test_path = candidate
            break

    if raw_test_path is None:
        raise FileNotFoundError(
            f"\n{'!'*60}\n"
            f"  Raw test file tidak ditemukan!\n"
            f"  Pastikan salah satu file berikut ada:\n"
            + "\n".join(f"    • {c}" for c in RAW_TEST_CANDIDATES) +
            f"\n{'!'*60}"
        )

    print(f"          Raw test ditemukan: {raw_test_path}")
    test_raw = pd.read_csv(raw_test_path)
    print(f"          Shape raw: {test_raw.shape[0]:,} x {test_raw.shape[1]}")

    # Apply TEST_RENAME (mapping dari src/feature_names)
    test_raw = test_raw.rename(columns=TEST_RENAME)

    # Buat snapshot dari training data
    print(f"          Build team snapshot dari training ...")
    snap, global_snap = build_team_snapshot(train_c)
    print(f"          Teams in snapshot: {len(snap)}")

    # fillna rank + flag (mirror 4A-6)
    for col, flag in [('rank_team', 'rank_available_team'),
                       ('rank_opponent', 'rank_available_opp')]:
        if col in test_raw.columns:
            med = train_c[col].median() if col in train_c.columns else 65
            test_raw[flag] = test_raw[col].notna().astype(int)
            test_raw[col]  = test_raw[col].fillna(med)

    # fillna H2H
    for col in ['h2h_gd_last5', 'h2h_points_last5']:
        if col in test_raw.columns:
            test_raw[col] = test_raw[col].fillna(0)

    # has_h2h_history
    if 'h2h_gd_last5' in test_raw.columns:
        test_raw['has_h2h_history'] = (
            (test_raw['h2h_gd_last5'] != 0) | (test_raw['h2h_points_last5'] != 0)
        ).astype(int)

    # Apply rolling snapshot
    test_raw = apply_snapshot_to_test(test_raw, snap, global_snap)

    # Simpan
    test_raw.to_csv(TEST_FEATURES_PATH, index=False)
    test = test_raw
    print(f"          Saved: {TEST_FEATURES_PATH} ({test.shape[0]:,} x {test.shape[1]})")

print(f"\n  [4D-2] Test shape: {test.shape[0]:,} x {test.shape[1]}")

# ── 4D-3: Pastikan semua EXTENDED_FEATURES ada di test ───────────────────────
print(f"\n  [4D-3] Align {len(EXTENDED_FEATURES)} EXTENDED_FEATURES ke test ...")
missing_test = []
for f in EXTENDED_FEATURES:
    if f not in test.columns:
        test[f] = 0.0
        missing_test.append(f)
    test[f] = test[f].fillna(0.0)
if missing_test:
    print(f"          WARNING: {len(missing_test)} fitur di-fillna 0: {missing_test[:5]} ...")
else:
    print(f"          OK: Semua {len(EXTENDED_FEATURES)} fitur tersedia")

X_test = test[EXTENDED_FEATURES]
print(f"  [4D-4] X_test shape: {X_test.shape}")

# ── 4D-5: Ensemble 5-fold ─────────────────────────────────────────────────────
print(f"\n  [4D-5] Ensemble {N_SPLITS} fold models ...")
lam_t_sum = np.zeros(len(test))
lam_o_sum = np.zeros(len(test))

for fold in range(N_SPLITS):
    with open(os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl'), 'rb') as fh:
        mt = pickle.load(fh)
    with open(os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl'), 'rb') as fh:
        mo = pickle.load(fh)
    lt = mt.predict(X_test).clip(0)
    lo = mo.predict(X_test).clip(0)
    lam_t_sum += lt
    lam_o_sum += lo
    print(f"          Fold {fold+1}: λ_team={lt.mean():.3f} | λ_opp={lo.mean():.3f}")

lam_t_avg = lam_t_sum / N_SPLITS
lam_o_avg = lam_o_sum / N_SPLITS
pred_team = np.round(lam_t_avg).astype(int)
pred_opp  = np.round(lam_o_avg).astype(int)

# ── ELO force dengan snapshot (bug fix) ──────────────────────────────────────
elo_test  = X_test['elo_diff'].values
mask_h    = (elo_test >  ELO_FORCE_WIN_THRESHOLD)  & (pred_team <= pred_opp)
mask_a    = (elo_test <  ELO_FORCE_LOSS_THRESHOLD) & (pred_opp  <= pred_team)
pt_snap   = pred_team.copy()
po_snap   = pred_opp.copy()
pred_team[mask_h] = po_snap[mask_h]  + 1
pred_opp[mask_a]  = pt_snap[mask_a]  + 1
print(f"\n  [4D-6] ELO force: +{mask_h.sum()} home | -{mask_a.sum()} away")

pred_team = np.maximum(pred_team, 0)
pred_opp  = np.maximum(pred_opp, 0)

sub = test[['Id']].copy()
sub['team_goals'] = pred_team
sub['opp_goals']  = pred_opp

# Validasi
assert not sub.isnull().any().any(), "ERROR: Ada null di submission!"
assert (sub['team_goals'] >= 0).all() and (sub['opp_goals'] >= 0).all(), "ERROR: Nilai negatif!"
assert sub['Id'].is_unique, "ERROR: Id duplikat!"

win_sub  = (sub['team_goals'] >  sub['opp_goals']).mean()
draw_sub = (sub['team_goals'] == sub['opp_goals']).mean()
loss_sub = (sub['team_goals'] <  sub['opp_goals']).mean()

sub.to_csv('submissions/sub_baseline_lgbm_v2.csv', index=False)
print(f"\n  [4D-7] Submission saved: {len(sub):,} baris")
print(f"          Win/Draw/Loss: {win_sub:.1%} / {draw_sub:.1%} / {loss_sub:.1%}")
print(f"          Mean goals   : team={sub['team_goals'].mean():.3f} | opp={sub['opp_goals'].mean():.3f}")
print(f"\n  ✅ STEP 4D SELESAI [{elapsed()}]")

# =============================================================================
#  STEP 4E — Dokumentasi
# =============================================================================
separator("STEP 4E — Dokumentasi")
print(f"  [{elapsed()}] Mulai ...")

oof_e   = pd.read_csv('data/processed/oof_baseline_lgbm.csv')
folds_e = pd.read_csv('data/processed/cv_fold_details.csv')
train_e = pd.read_csv('data/processed/train_aligned.csv').sort_values('date').reset_index(drop=True)

# ── Feature importance (pakai context manager — fix resource leak) ──────────
all_fi = []
for i in range(N_SPLITS):
    with open(f'{MODELS_DIR}/lgbm_team_fold{i}.pkl', 'rb') as fh:
        all_fi.append(pickle.load(fh).feature_importances_)
avg_fi = np.mean(all_fi, axis=0)
fi_df  = pd.DataFrame({'feature': EXTENDED_FEATURES, 'importance': avg_fi})\
           .sort_values('importance', ascending=False)

# ── Pakai ddof=0 agar konsisten dengan Step 4C ───────────────────────────────
cv_m   = folds_e['awmae'].mean()
cv_s   = folds_e['awmae'].std(ddof=0)
out_acc = oof_e['outcome_correct'].mean() if 'outcome_correct' in oof_e.columns \
          else folds_e['outcome_acc'].mean()

best_bl = pd.read_csv('data/processed/baseline_scores_summary.csv')['awmae'].min()
improv  = (best_bl - cv_m) / best_bl * 100

if cv_m < 2.5:   verdict_e = "🏆 SANGAT BAGUS"
elif cv_m < 2.6: verdict_e = "✅ BAGUS"
elif cv_m < 2.75:verdict_e = "✅ OK (memenuhi target)"
else:            verdict_e = "⚠️  Di bawah target"

lines = [
    "# 📊 Fase 4 — Baseline Model Summary Report (v2)",
    f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Duration: {elapsed()}",
    "",
    "---",
    "## 🎯 Overview",
    "",
    "| Item | Detail |",
    "|------|--------|",
    f"| Model | LightGBM Dual Poisson |",
    f"| Fitur | **{len(EXTENDED_FEATURES)}** ({len(FINAL_FEATURES)} base + {len(EXTENDED_FEATURES)-len(FINAL_FEATURES)} rolling) |",
    f"| CV | {N_SPLITS}-fold TimeSeriesSplit |",
    f"| CV AW-MAE | **`{cv_m:.4f} ± {cv_s:.4f}`** |",
    f"| Outcome Acc | {out_acc:.2%} |",
    f"| Verdict | {verdict_e} |",
    f"| Submission | `submissions/sub_baseline_lgbm_v2.csv` |",
    "",
    "---",
    "## 📈 Fold Detail",
    "",
    "| Fold | N Train | N Val | AW-MAE | Outcome Acc | Best Iter T/O |",
    "|------|---------|-------|--------|-------------|---------------|",
]
for _, r in folds_e.iterrows():
    lines.append(
        f"| {int(r['fold'])} | {int(r['n_train']):,} | {int(r['n_val']):,} | "
        f"`{r['awmae']:.4f}` | {r['outcome_acc']:.2%} | "
        f"{int(r['best_iter_team'])}/{int(r['best_iter_opp'])} |"
    )
lines += [
    f"| **Mean** | — | — | **`{cv_m:.4f}`** | **{folds_e['outcome_acc'].mean():.2%}** | — |",
    "",
    "---",
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
    "## 📉 Distribusi Prediksi vs Aktual",
    "",
    "| Outcome | Train Aktual | Test Prediksi |",
    "|---------|-------------|---------------|",
    f"| Win  | {(train_e['team_goals']>train_e['opp_goals']).mean():.2%} | {win_sub:.2%} |",
    f"| Draw | {(train_e['team_goals']==train_e['opp_goals']).mean():.2%} | {draw_sub:.2%} |",
    f"| Loss | {(train_e['team_goals']<train_e['opp_goals']).mean():.2%} | {loss_sub:.2%} |",
    "",
    "---",
    "## 📁 Output Files",
    "",
    "| File | Keterangan |",
    "|------|------------|",
    "| `data/processed/train_aligned.csv` | Train setelah base alignment |",
    "| `data/processed/train_extended.csv` | Train + rolling features |",
    "| `data/processed/test_features.csv` | Test features (auto-generated) |",
    "| `data/processed/baseline_scores_summary.csv` | Flat baseline scores |",
    "| `data/processed/oof_baseline_lgbm.csv` | OOF predictions |",
    "| `data/processed/cv_fold_details.csv` | Detail per fold |",
    "| `models/lgbm_[team|opp]_fold[0-4].pkl` | 10 model files |",
    "| `submissions/sub_baseline_lgbm_v2.csv` | **Submission file** |",
    "",
    "---",
    "## 🚀 Prioritas Fase 5",
    "",
    "1. **Hyperparameter tuning** dengan Optuna (100 trials)",
    "2. **XGBoost Poisson** sebagai model kedua untuk ensemble",
    "3. **Conditional model** → prediksi `total_goals + goal_diff` lalu derive skor",
    "4. **Clipping goals test** → `clip(8)` vs `clip(10)` vs no clip",
    "5. **xG proxy** dari shot data kalau tersedia",
    "",
    f"> Target Fase 5: CV AW-MAE < 2.40 (saat ini: `{cv_m:.4f}`)",
    "",
    "---",
    "*Auto-generated by `notebooks/fase_4/04_fase4_complete.py` v2*",
]

with open('reports/fase_4_summary.md', 'w', encoding='utf-8') as fh:
    fh.write("\n".join(lines))
print(f"\n  Laporan saved: reports/fase_4_summary.md")
print(f"\n  ✅ STEP 4E SELESAI [{elapsed()}]")

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
separator("FASE 4 — COMPLETE SUMMARY v2")

best_fold_num  = int(folds_e.loc[folds_e['awmae'].idxmin(), 'fold'])   # fix: int()
worst_fold_num = int(folds_e.loc[folds_e['awmae'].idxmax(), 'fold'])   # fix: int()
best_fold_mae  = folds_e['awmae'].min()
worst_fold_mae = folds_e['awmae'].max()

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │              HASIL AKHIR FASE 4 (v2)                 │
  ├──────────────────────────────────────────────────────┤
  │  CV AW-MAE       : {cv_m:.4f} ± {cv_s:.4f}              │
  │  Best fold       : Fold {best_fold_num} ({best_fold_mae:.4f})                    │
  │  Worst fold      : Fold {worst_fold_num} ({worst_fold_mae:.4f})                    │
  │  Outcome acc     : {out_acc:.2%}                          │
  │  vs best baseline: +{improv:.1f}% improvement              │
  │  Features        : {len(EXTENDED_FEATURES)} ({len(FINAL_FEATURES)} base + {len(EXTENDED_FEATURES)-len(FINAL_FEATURES)} rolling)           │
  │  Total waktu     : {elapsed()}                           │
  ├──────────────────────────────────────────────────────┤
  │  Verdict: {verdict_e:<42}│
  ├──────────────────────────────────────────────────────┤
  │  OUTPUT:                                             │
  │  ✓ data/processed/train_extended.csv                 │
  │  ✓ data/processed/test_features.csv                  │
  │  ✓ data/processed/oof_baseline_lgbm.csv              │
  │  ✓ models/ (10 pkl files)                            │
  │  ✓ submissions/sub_baseline_lgbm_v2.csv ← SUBMIT!   │
  │  ✓ reports/fase_4_summary.md                         │
  └──────────────────────────────────────────────────────┘

  ⚡ SUBMIT sub_baseline_lgbm_v2.csv ke leaderboard!
  → Catat public score, lanjut ke FASE 5
""")