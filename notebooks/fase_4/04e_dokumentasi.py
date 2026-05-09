# =============================================================================
# notebooks/fase_4/04e_dokumentasi.py
# STEP 4E — Dokumentasi Internal Fase 4
# Tujuan: Buat laporan lengkap hasil Fase 4 dalam format markdown
# Output: reports/fase_4_summary.md
# Estimasi waktu: ~15 menit
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from src.metrics import kalkulasi_aw_mae
from src.feature_names import FINAL_FEATURES

# ---------------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------------
TRAIN_PATH    = 'data/processed/train_aligned.csv'
OOF_PATH      = 'data/processed/oof_baseline_lgbm.csv'
BASELINE_PATH = 'data/processed/baseline_scores_summary.csv'
FOLDS_PATH    = 'data/processed/cv_fold_details.csv'
SUB_PATH      = 'submissions/sub_baseline_lgbm_v1.csv'
MODELS_DIR    = 'models'
OUTPUT_MD     = 'reports/fase_4_summary.md'
os.makedirs('reports', exist_ok=True)

print("=" * 60)
print("STEP 4E — DOKUMENTASI INTERNAL FASE 4")
print("=" * 60)

# ---------------------------------------------------------------------------
# LOAD SEMUA HASIL
# ---------------------------------------------------------------------------
print("\n[1/5] Loading semua hasil Fase 4 ...")
train    = pd.read_csv(TRAIN_PATH).sort_values('date').reset_index(drop=True)
oof      = pd.read_csv(OOF_PATH)
baseline = pd.read_csv(BASELINE_PATH).sort_values('awmae')
folds    = pd.read_csv(FOLDS_PATH)
sub      = pd.read_csv(SUB_PATH)
print(f"      Semua file berhasil di-load")

# ---------------------------------------------------------------------------
# KALKULASI METRIK FINAL
# ---------------------------------------------------------------------------
print("\n[2/5] Kalkulasi metrik final ...")

y_true    = train[['team_goals', 'opp_goals']].values
y_pred    = oof[['pred_team_goals', 'pred_opp_goals']].values.astype(float)
w         = train['tournament_weight'].values
oof_score = kalkulasi_aw_mae(y_true, y_pred, w)

cv_mean   = folds['awmae'].mean()
cv_std    = folds['awmae'].std()
cv_min    = folds['awmae'].min()
cv_max    = folds['awmae'].max()

outcome_acc_mean = folds['outcome_acc'].mean()

# Distribusi train aktual
win_train  = (train['team_goals'] >  train['opp_goals']).mean()
draw_train = (train['team_goals'] == train['opp_goals']).mean()
loss_train = (train['team_goals'] <  train['opp_goals']).mean()

# Distribusi OOF prediksi
win_oof    = (oof['pred_team_goals'] >  oof['pred_opp_goals']).mean()
draw_oof   = (oof['pred_team_goals'] == oof['pred_opp_goals']).mean()
loss_oof   = (oof['pred_team_goals'] <  oof['pred_opp_goals']).mean()

# Distribusi submission prediksi
win_sub    = (sub['team_goals'] >  sub['opp_goals']).mean()
draw_sub   = (sub['team_goals'] == sub['opp_goals']).mean()
loss_sub   = (sub['team_goals'] <  sub['opp_goals']).mean()

print(f"      CV AW-MAE : {cv_mean:.4f} ± {cv_std:.4f}")
print(f"      OOF AW-MAE: {oof_score:.4f}")
print(f"      Outcome acc: {outcome_acc_mean:.2%}")

# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
print("\n[3/5] Hitung feature importance rata-rata ...")
all_fi_team = []
all_fi_opp  = []

for fold in range(5):
    pt = os.path.join(MODELS_DIR, f'lgbm_team_fold{fold}.pkl')
    po = os.path.join(MODELS_DIR, f'lgbm_opp_fold{fold}.pkl')
    if os.path.exists(pt) and os.path.exists(po):
        with open(pt, 'rb') as f: mt = pickle.load(f)
        with open(po, 'rb') as f: mo = pickle.load(f)
        all_fi_team.append(mt.feature_importances_)
        all_fi_opp.append(mo.feature_importances_)

avg_fi_team = np.mean(all_fi_team, axis=0)
avg_fi_opp  = np.mean(all_fi_opp,  axis=0)

fi_team = pd.DataFrame({'feature': FINAL_FEATURES, 'importance_team': avg_fi_team})
fi_opp  = pd.DataFrame({'feature': FINAL_FEATURES, 'importance_opp':  avg_fi_opp})
fi_df   = fi_team.merge(fi_opp, on='feature')
fi_df['importance_avg'] = (fi_df['importance_team'] + fi_df['importance_opp']) / 2
fi_df = fi_df.sort_values('importance_avg', ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# TOP SKOR DISTRIBUSI
# ---------------------------------------------------------------------------
top_train = (train.groupby(['team_goals','opp_goals'])
             .size().reset_index(name='count')
             .sort_values('count', ascending=False)
             .head(10))
top_train['pct'] = (top_train['count'] / len(train) * 100).round(2)

top_sub = (sub.groupby(['team_goals','opp_goals'])
           .size().reset_index(name='count')
           .sort_values('count', ascending=False)
           .head(10))
top_sub['pct'] = (top_sub['count'] / len(sub) * 100).round(2)

# ---------------------------------------------------------------------------
# TULIS LAPORAN MARKDOWN
# ---------------------------------------------------------------------------
print("\n[4/5] Menulis laporan markdown ...")

lines = []
A = lines.append  # shorthand

A("# 📊 Fase 4 — Baseline Model Summary Report")
A(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
A("")
A("---")
A("")

# === OVERVIEW ===
A("## 🎯 Overview")
A("")
A("| Item | Detail |")
A("|------|--------|")
A(f"| Model | LightGBM Dual Poisson (team_goals & opp_goals terpisah) |")
A(f"| Fitur | {len(FINAL_FEATURES)} fitur (TIER 1 + TIER 2 aligned) |")
A(f"| CV Strategy | TimeSeriesSplit 5-fold |")
A(f"| Training rows | {len(train):,} |")
A(f"| Test rows | {len(sub):,} |")
A(f"| Submission | `submissions/sub_baseline_lgbm_v1.csv` |")
A("")

# === CV RESULTS ===
A("---")
A("")
A("## 📈 Cross-Validation Results")
A("")
A(f"**CV AW-MAE: `{cv_mean:.4f} ± {cv_std:.4f}`**")
A("")
A("| Fold | N Train | N Val | AW-MAE | Outcome Acc | Best Iter (T/O) | ELO Forced (H/A) |")
A("|------|---------|-------|--------|-------------|-----------------|-----------------|")
for _, row in folds.iterrows():
    A(f"| {int(row['fold'])} | {int(row['n_train']):,} | {int(row['n_val']):,} | "
      f"`{row['awmae']:.4f}` | {row['outcome_acc']:.2%} | "
      f"{int(row['best_iter_team'])}/{int(row['best_iter_opp'])} | "
      f"+{int(row['forced_home'])} / -{int(row['forced_away'])} |")
A(f"| **Mean** | — | — | **`{cv_mean:.4f}`** | **{outcome_acc_mean:.2%}** | — | — |")
A(f"| **Std** | — | — | `{cv_std:.4f}` | — | — | — |")
A("")

# Verdict
if cv_mean < 2.5:
    verdict_icon, verdict_text = "🏆", "SANGAT BAGUS — Jauh di atas target Fase 4"
elif cv_mean < 2.6:
    verdict_icon, verdict_text = "✅", "BAGUS — Di atas target Fase 4"
elif cv_mean < 2.75:
    verdict_icon, verdict_text = "✅", "OK — Memenuhi target Fase 4 (< 2.75)"
else:
    verdict_icon, verdict_text = "⚠️", "Di bawah target — perlu improvement di Fase 5"

A(f"> **Verdict:** {verdict_icon} {verdict_text}")
A("")

# === BASELINE COMPARISON ===
A("---")
A("")
A("## 📊 Perbandingan dengan Baselines")
A("")
A("| Rank | Model | AW-MAE | Keterangan |")
A("|------|-------|--------|------------|")
A(f"| 🏆 | **LightGBM Dual Poisson (CV)** | **`{cv_mean:.4f}`** | **Model Fase 4** |")
for _, row in baseline.iterrows():
    A(f"| — | {row['model']} | `{row['awmae']:.4f}` | {row['notes']} |")
A("")
best_baseline = baseline['awmae'].min()
improvement   = (best_baseline - cv_mean) / best_baseline * 100
A(f"> Improvement vs best simple baseline (`per_team_mean_blended = {best_baseline:.4f}`): **{improvement:.1f}%** lebih baik")
A("")

# === FEATURE IMPORTANCE ===
A("---")
A("")
A("## 🔑 Feature Importance (Rata-rata 5 Fold)")
A("")
A("| Rank | Feature | Importance (Team Model) | Importance (Opp Model) | Avg |")
A("|------|---------|------------------------|------------------------|-----|")
for rank, (_, row) in enumerate(fi_df.iterrows(), 1):
    bar_len = int(row['importance_avg'] / fi_df['importance_avg'].max() * 20)
    bar     = "█" * bar_len + "░" * (20 - bar_len)
    A(f"| {rank} | `{row['feature']}` | {row['importance_team']:.0f} | "
      f"{row['importance_opp']:.0f} | **{row['importance_avg']:.0f}** |")
A("")
A("> **Insight:**")
A("> - `roll_avg_goals_conceded_opp_5` adalah fitur terpenting — **pertahanan lawan** > ELO!")
A("> - ELO features (elo_diff + elo_win_prob) mendominasi rank 2 & 3")
A("> - `gdp_ratio` masih signifikan (rank 6) sebagai proxy level profesionalisme")
A("> - `h2h_gd_last5` lebih penting dari `h2h_points_last5`")
A("")

# === DISTRIBUSI PREDIKSI ===
A("---")
A("")
A("## 📉 Distribusi Prediksi vs Aktual")
A("")
A("### Outcome Distribution")
A("")
A("| Outcome | Train Aktual | OOF Prediksi | Test Prediksi |")
A("|---------|-------------|--------------|---------------|")
A(f"| **Win** | {win_train:.2%} | {win_oof:.2%} | {win_sub:.2%} |")
A(f"| **Draw** | {draw_train:.2%} | {draw_oof:.2%} | {draw_sub:.2%} |")
A(f"| **Loss** | {loss_train:.2%} | {loss_oof:.2%} | {loss_sub:.2%} |")
A("")
A("> Model cenderung sedikit over-predict loss rate vs aktual train.")
A("> Ini wajar karena test set (2011-2026) lebih modern dengan kompetisi lebih ketat.")
A("")
A("### Goals Mean")
A("")
A("| | Team Goals | Opp Goals |")
A("|-|-----------|----------|")
A(f"| Train aktual | {train['team_goals'].mean():.3f} | {train['opp_goals'].mean():.3f} |")
A(f"| OOF prediksi | {oof['pred_team_goals'].mean():.3f} | {oof['pred_opp_goals'].mean():.3f} |")
A(f"| Test prediksi | {sub['team_goals'].mean():.3f} | {sub['opp_goals'].mean():.3f} |")
A("")

A("### Top 10 Skor Aktual (Train)")
A("")
A("| Skor | Count | Pct |")
A("|------|-------|-----|")
for _, row in top_train.iterrows():
    A(f"| {int(row['team_goals'])}-{int(row['opp_goals'])} | {int(row['count']):,} | {row['pct']:.2f}% |")
A("")

A("### Top 10 Skor Prediksi (Test Submission)")
A("")
A("| Skor | Count | Pct |")
A("|------|-------|-----|")
for _, row in top_sub.iterrows():
    A(f"| {int(row['team_goals'])}-{int(row['opp_goals'])} | {int(row['count']):,} | {row['pct']:.2f}% |")
A("")

# === OUTPUT FILES ===
A("---")
A("")
A("## 📁 Output Files Fase 4")
A("")
A("| File | Keterangan |")
A("|------|------------|")
A("| `data/processed/train_aligned.csv` | Train setelah alignment, drop duplikat, fillna |")
A("| `data/processed/baseline_scores_summary.csv` | Skor semua flat baselines |")
A("| `data/processed/oof_baseline_lgbm.csv` | OOF predictions + lambda + error analysis |")
A("| `data/processed/cv_fold_details.csv` | Detail per fold CV |")
A("| `models/lgbm_team_fold{0-4}.pkl` | 5 model files untuk team_goals |")
A("| `models/lgbm_opp_fold{0-4}.pkl` | 5 model files untuk opp_goals |")
A("| `submissions/sub_baseline_lgbm_v1.csv` | **Submission file siap upload** |")
A("| `reports/fase_4_summary.md` | File ini |")
A("")

# === CATATAN FASE 5 ===
A("---")
A("")
A("## 🚀 Catatan & Prioritas untuk Fase 5")
A("")
A("### ⚡ Immediate Wins (High Impact, Low Effort)")
A("")
A("1. **Re-engineer TIER4 features ke train** — test punya 71 fitur ekstra yang tidak ada di train.")
A("   Prioritas utama:")
A("   - `xg_proxy_team / xg_proxy_opp` — proxy Expected Goals")
A("   - `momentum_team / momentum_opp` — form momentum")
A("   - `roll_wr_5 / roll_wr_10` — win rate rolling (belum ada di train!)")
A("   - `roll_streak_team / opp_roll_streak` — winning/losing streak")
A("   - `form_diff_5` — gap form antar tim")
A("   Estimasi impact: **-0.2 sampai -0.4 AW-MAE**")
A("")
A("2. **Clipping goals** — test per model:")
A("   - `clip(upper=8)` vs `clip(upper=10)` vs no clip")
A("   - Berdasarkan Fase 3 audit: `max goals = 31`, outlier ini merusak training")
A("   Estimasi impact: **-0.05 sampai -0.15 AW-MAE**")
A("")
A("3. **Hyperparameter tuning dengan Optuna** (100 trials)")
A("   - Tune: learning_rate, num_leaves, min_child_samples, subsample")
A("   - Gunakan OOF AW-MAE sebagai objective")
A("   Estimasi impact: **-0.1 sampai -0.2 AW-MAE**")
A("")
A("### 🔬 Model Eksploratif (Medium Effort)")
A("")
A("4. **XGBoost Poisson** — pembanding LightGBM, sering head-to-head kompetitif")
A("5. **Conditional model** — prediksi `total_goals` + `goal_diff` lalu derive skor")
A("   - Sangat relevan karena AW-MAE mempunyai penalty khusus untuk GD!")
A("6. **CatBoost** — handle categorical (tournament, confederation) secara native")
A("")
A("### 🎯 Target Fase 5")
A("")
A("| Metric | Fase 4 Baseline | Target Fase 5 | Target Fase 6 |")
A("|--------|----------------|---------------|---------------|")
A(f"| CV AW-MAE | `{cv_mean:.4f}` | `< 2.40` | `< 2.20` |")
A(f"| Outcome Acc | `{outcome_acc_mean:.2%}` | `> 62%` | `> 65%` |")
A("")
A("---")
A("")
A("*Laporan ini di-generate otomatis oleh `notebooks/fase_4/04e_dokumentasi.py`*")

# ---------------------------------------------------------------------------
# SIMPAN
# ---------------------------------------------------------------------------
report_text = "\n".join(lines)
with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
    f.write(report_text)

# ---------------------------------------------------------------------------
# PRINT KE CONSOLE JUGA
# ---------------------------------------------------------------------------
print(f"\n[5/5] Laporan tersimpan ke {OUTPUT_MD}")
print(f"\n{'=' * 60}")
print(f"STEP 4E SELESAI")
print(f"{'=' * 60}")
print(f"\n  RINGKASAN HASIL FASE 4:")
print(f"  ┌─────────────────────────────────────────────────────┐")
print(f"  │  CV AW-MAE       : {cv_mean:.4f} ± {cv_std:.4f}          │")
print(f"  │  Best fold       : Fold {folds.loc[folds['awmae'].idxmin(),'fold']} ({cv_min:.4f})                  │")
print(f"  │  Worst fold      : Fold {folds.loc[folds['awmae'].idxmax(),'fold']} ({cv_max:.4f})                  │")
print(f"  │  Outcome acc avg : {outcome_acc_mean:.2%}                        │")
print(f"  │  Improvement vs  : +{improvement:.1f}% vs per_team_mean          │")
print(f"  │  Submission rows : {len(sub):,}                         │")
print(f"  └─────────────────────────────────────────────────────┘")
print(f"\n  VERDICT: {verdict_icon} {verdict_text}")
print(f"\n  OUTPUT FASE 4:")
print(f"    data/processed/train_aligned.csv")
print(f"    data/processed/oof_baseline_lgbm.csv")
print(f"    models/lgbm_[team|opp]_fold[0-4].pkl  (10 files)")
print(f"    submissions/sub_baseline_lgbm_v1.csv")
print(f"    reports/fase_4_summary.md")
print(f"\n  ✅ FASE 4 COMPLETE — SIAP LANJUT KE FASE 5")
