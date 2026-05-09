# =============================================================================
# src/feature_names.py
# Single Source of Truth untuk semua nama fitur di Fase 4
# GammaFest 2026 — Jangan ubah tanpa koordinasi tim!
# =============================================================================

# ---------------------------------------------------------------------------
# FINAL_FEATURES: 14 fitur yang tersedia di KEDUA train & test
# Urutan ini harus konsisten di semua notebook
# ---------------------------------------------------------------------------
FINAL_FEATURES = [
    # ELO — top predictors (korelasi 0.38–0.41)
    'elo_diff',
    'elo_win_prob',
    # Venue context
    'is_home',
    'neutral',
    # Rolling form — Team (TIER 2: ada di train, nama beda di test)
    'roll_avg_goals_scored_team_3',    # = roll_gs_5  di test
    'roll_avg_goals_scored_team_10',   # = roll_gs_10 di test
    'roll_avg_goals_conceded_team_5',  # = roll_gc_5  di test
    # Rolling form — Opponent (rank #1 di feature importance!)
    'roll_avg_goals_conceded_opp_5',   # = opp_roll_gc_5  di test
    'roll_avg_goals_scored_opp_3',     # = opp_roll_gs_5  di test
    'roll_avg_goals_scored_opp_10',    # = opp_roll_gs_10 di test
    # Head-to-Head history
    'h2h_gd_last5',
    'h2h_points_last5',
    # Match context
    'tournament_weight',
    'gdp_ratio',
]

# ---------------------------------------------------------------------------
# TEST_RENAME: Map nama kolom di test_features.csv → nama standar FINAL_FEATURES
# Dipakai di 04d_submission.py saat load test
# ---------------------------------------------------------------------------
TEST_RENAME = {
    'roll_gs_5':      'roll_avg_goals_scored_team_3',
    'roll_gs_10':     'roll_avg_goals_scored_team_10',
    'roll_gc_5':      'roll_avg_goals_conceded_team_5',
    'opp_roll_gc_5':  'roll_avg_goals_conceded_opp_5',
    'opp_roll_gs_5':  'roll_avg_goals_scored_opp_3',
    'opp_roll_gs_10': 'roll_avg_goals_scored_opp_10',
}

# ---------------------------------------------------------------------------
# TRAIN_COLS_DROP: Kolom duplikat yang harus di-drop dari train_engineered
# Versi tanpa .1 adalah versi lama (pre-ELO update), simpan yang .1
# ---------------------------------------------------------------------------
TRAIN_COLS_DROP = [
    'elo_team',                    # diganti elo_team.1 (post-update)
    'elo_opponent',                # diganti elo_opp (post-update)
    'days_since_last_match_team',  # diganti days_since_last_match_team.1
    'days_since_last_match_opp',   # diganti days_since_last_match_opp.1
]

# ---------------------------------------------------------------------------
# TRAIN_COLS_RENAME: Rename kolom .1 ke nama bersih
# ---------------------------------------------------------------------------
TRAIN_COLS_RENAME = {
    'elo_team.1':                   'elo_team_final',
    'elo_opp':                      'elo_opp_final',
    'days_since_last_match_team.1': 'rest_days_team',
    'days_since_last_match_opp.1':  'rest_days_opp',
}

# ---------------------------------------------------------------------------
# LGB_PARAMS: Hyperparameter LightGBM baseline (dipakai di 04c dan 04_complete)
# ---------------------------------------------------------------------------
LGB_PARAMS = {
    'objective':         'poisson',
    'n_estimators':      600,
    'learning_rate':     0.05,
    'max_depth':         6,
    'num_leaves':        50,
    'min_child_samples': 50,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        0.3,
    'random_state':      42,
    'verbose':           -1,
    'n_jobs':            -1,
}

# ---------------------------------------------------------------------------
# ELO OUTCOME FORCE THRESHOLDS
# Jika elo_diff sangat ekstrem, paksa outcome sesuai ELO
# ---------------------------------------------------------------------------
ELO_FORCE_WIN_THRESHOLD  =  300   # elo_diff > 300  → paksa win
ELO_FORCE_LOSS_THRESHOLD = -300   # elo_diff < -300 → paksa lose
