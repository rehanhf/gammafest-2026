# ============================================================
# FASE 2 — COMPREHENSIVE EDA (02a + 02b + 02c)
# Integrated Distribution, Correlation, and Temporal Analysis
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, spearmanr, pearsonr, kendalltau

# ── 0. Setup ──────────────────────────────────────────────────
os.makedirs('./reports/figures', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')

# ── 1. Load data (once) ───────────────────────────────────────
train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})
print(f"Dataset Shape: {train.shape}")
print(f"\nDataset Info:")
print(f"  Date range: {train['date'].min()} to {train['date'].max()}")
print(f"  Columns: {train.shape[1]}")

# =============================================================
# PART 1: DISTRIBUTION ANALYSIS (02a)
# =============================================================
print("\n" + "="*70)
print("PART 1: DISTRIBUTION ANALYSIS — Gol Distribution")
print("="*70)

print(f"\nMissing values in ['team_goals', 'opp_goals']: {train[['team_goals', 'opp_goals']].isna().sum().to_dict()}")
print(train[['team_goals', 'opp_goals']].describe().round(3))

# ── 1.1 Statistik distribusi ──────────────────────────────────
for col in ['team_goals', 'opp_goals']:
    s = train[col].dropna()
    skewness = s.skew()
    kurtosis = s.kurtosis()
    skew_label = '(right-skewed)' if skewness > 0.5 else '(approx normal)' if abs(skewness) < 0.5 else '(left-skewed)'
    
    print(f"\n{'─'*50}")
    print(f"Kolom       : {col}")
    print(f"Count       : {len(s)} (missing: {train[col].isna().sum()})")
    print(f"Min         : {s.min():.0f}")
    print(f"Mean        : {s.mean():.3f}")
    print(f"Median      : {s.median():.3f}")
    mode_val = s.mode()
    print(f"Mode        : {mode_val.values[0]:.0f}" if len(mode_val) > 0 else "Mode        : N/A")
    print(f"Std         : {s.std():.3f}")
    print(f"P25 (25%)   : {s.quantile(0.25):.3f}")
    print(f"P75 (75%)   : {s.quantile(0.75):.3f}")
    print(f"P90 (90%)   : {s.quantile(0.90):.3f}")
    print(f"P95 (95%)   : {s.quantile(0.95):.3f}")
    print(f"Max         : {s.max():.0f}")
    print(f"Skewness    : {skewness:.3f}  {skew_label}")
    print(f"Kurtosis    : {kurtosis:.3f}")
    
    # Normality test
    stat, p_val = shapiro(s)
    is_normal = '✓ Data appears normal (fail to reject H0)' if p_val > 0.05 else '✗ Data is NOT normal (reject H0)'
    print(f"Shapiro-Wilk Test: stat={stat:.4f}, p-value={p_val:.4e}")
    print(f"  → {is_normal}")

# ── 1.2 Deteksi outlier dengan IQR ────────────────────────────
def deteksi_outlier_iqr(series, label):
    series_clean = series.dropna()
    Q1  = series_clean.quantile(0.25)
    Q3  = series_clean.quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas  = Q3 + 1.5 * IQR
    outlier = series_clean[(series_clean < batas_bawah) | (series_clean > batas_atas)]
    outlier_below = series_clean[series_clean < batas_bawah]
    outlier_above = series_clean[series_clean > batas_atas]
    pct_outlier = len(outlier)/len(series_clean)*100
    
    print(f"\n[OUTLIER DETECTION] {label}")
    print(f"  IQR           : {IQR:.3f}")
    print(f"  Lower bound   : {batas_bawah:.3f}")
    print(f"  Upper bound   : {batas_atas:.3f}")
    print(f"  Total outliers: {len(outlier)} ({pct_outlier:.2f}%)")
    print(f"    - Below lower bound: {len(outlier_below)} ({len(outlier_below)/len(series_clean)*100:.2f}%)")
    print(f"    - Above upper bound: {len(outlier_above)} ({len(outlier_above)/len(series_clean)*100:.2f}%)")
    if len(outlier_above) > 0:
        print(f"  Top outliers (above): {sorted(outlier_above.unique(), reverse=True)[:10]}")
    return batas_atas, pct_outlier

batas_team, pct_team = deteksi_outlier_iqr(train['team_goals'], 'team_goals')
batas_opp, pct_opp   = deteksi_outlier_iqr(train['opp_goals'],  'opp_goals')

# ── 1.2b Hard clipping limits (99.9th percentile for Phase 5) ───
p999_team = int(np.ceil(train['team_goals'].quantile(0.999)))
p999_opp = int(np.ceil(train['opp_goals'].quantile(0.999)))

print(f"\n[HARD CLIPPING LIMITS] — Use in Phase 5 Feature Engineering")
print(f"  team_goals 99.9th percentile: {p999_team} goals")
print(f"  opp_goals 99.9th percentile : {p999_opp} goals")
print(f"  Rationale: Clip extreme outliers without removing Poisson tail events")

# ── 1.3 Visualisasi Distribusi ────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('PART 1: Distribusi Gol — Fase 2 EDA', fontsize=16, fontweight='bold', y=0.995)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

warna = {'team_goals': '#2196F3', 'opp_goals': '#FF5722'}

for i, col in enumerate(['team_goals', 'opp_goals']):
    label = 'Tim' if col == 'team_goals' else 'Lawan'
    c     = warna[col]

    # -- histogram + KDE
    ax1 = fig.add_subplot(gs[0, i])
    train[col].plot.hist(bins=list(range(0, int(train[col].max()) + 2)),
                         ax=ax1, color=c, alpha=0.75, edgecolor='white')
    ax1_twin = ax1.twinx()
    train[col].plot.kde(ax=ax1_twin, color=c, linewidth=2)
    ax1_twin.set_ylabel('Density', fontsize=9)
    ax1.set_title(f'Histogram + KDE — Gol {label}', fontweight='bold')
    ax1.set_xlabel('Jumlah Gol')
    ax1.set_ylabel('Frekuensi')
    ax1.axvline(train[col].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Mean={train[col].mean():.2f}')
    ax1.axvline(train[col].median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median={train[col].median():.0f}')
    ax1.legend(fontsize=8)

    # -- boxplot
    ax2 = fig.add_subplot(gs[1, i])
    ax2.boxplot(train[col].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor=c, alpha=0.6),
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.4))
    ax2.set_title(f'Boxplot — Gol {label}', fontweight='bold')
    ax2.set_xlabel('Jumlah Gol')

# -- total gol per pertandingan
train['total_goals'] = train['team_goals'] + train['opp_goals']
ax3 = fig.add_subplot(gs[2, 0])
train['total_goals'].plot.hist(bins=list(range(0, int(train['total_goals'].max()) + 2)),
                               ax=ax3, color='#4CAF50', alpha=0.75, edgecolor='white')
ax3.set_title('Histogram — Total Gol per Pertandingan', fontweight='bold')
ax3.set_xlabel('Total Gol')
ax3.set_ylabel('Frekuensi')
ax3.axvline(train['total_goals'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f"Mean={train['total_goals'].mean():.2f}")
ax3.legend(fontsize=8)

# -- Q-Q plot
ax4 = fig.add_subplot(gs[2, 1])
stats.probplot(train['team_goals'], dist='norm', plot=ax4)
ax4.set_title('Q-Q Plot — Gol Tim (vs Normal)', fontweight='bold')
ax4.get_lines()[0].set(markersize=2, alpha=0.4, color='#2196F3')
ax4.get_lines()[1].set(color='red', linewidth=1.5)

plt.savefig('./reports/figures/02_part1_distribusi_gol.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] ./reports/figures/02_part1_distribusi_gol.png")

# Summary Part 1
print("\n" + "═"*60)
print("SUMMARY: DISTRIBUSI GOL")
print("═"*60)
print(f"→ Skewness team_goals      : {train['team_goals'].skew():.3f} (right-skewed)")
print(f"→ Skewness opp_goals       : {train['opp_goals'].skew():.3f}")
print(f"→ Outlier atas team        : > {batas_team:.1f} gol ({pct_team:.2f}%)")
print(f"→ Outlier atas opp         : > {batas_opp:.1f} gol ({pct_opp:.2f}%)")
print(f"→ Rata-rata total gol      : {train['total_goals'].mean():.2f} per pertandingan")
print(f"\nDATA QUALITY NOTES:")
print(f"  • Both variables are NOT normally distributed")
print(f"  • Right-skewed: more low-scoring matches")
print(f"  • {pct_team:.1f}% outliers in team_goals, {pct_opp:.1f}% in opp_goals")

# =============================================================
# PART 2: CORRELATION ANALYSIS (02b)
# =============================================================
print("\n" + "="*70)
print("PART 2: CORRELATION ANALYSIS — Feature Relationships")
print("="*70)

# ── 2.1 Pilih kolom numerik ──────────────────────────────────
kolom_numerik = [
    'team_goals', 'opp_goals',
    'altitude_venue', 'temperature_venue',
    'distance_travel_team', 'distance_travel_opp',
    'gdp_per_capita_team', 'gdp_per_capita_opp',
    'population_team', 'population_opp'
]

kolom_numerik = [c for c in kolom_numerik if c in train.columns]
print(f"\nKolom yang digunakan ({len(kolom_numerik)}): {kolom_numerik}")

df_num = train[kolom_numerik].copy()

# Check missing values
print(f"\nMissing values per kolom:")
missing = df_num.isna().sum()
for col in missing[missing > 0].index:
    pct = missing[col] / len(df_num) * 100
    print(f"  {col:<30} : {missing[col]:>5} ({pct:>5.2f}%)")
if missing.sum() == 0:
    print("  → Tidak ada missing values")

# ── 2.2 Matriks korelasi ──────────────────────────────────────
corr_pearson  = df_num.corr(method='pearson')
corr_spearman = df_num.corr(method='spearman')

# ── 2.3 P-values untuk korelasi Spearman ─────────────────────
def get_pvalues_matrix(data, method='spearman'):
    """Compute p-value matrix for correlations"""
    cols = data.columns
    pval_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), 
                               index=cols, columns=cols)
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                pval_matrix.iloc[i, j] = 0
            else:
                if method == 'spearman':
                    _, pval = spearmanr(data.iloc[:, i], data.iloc[:, j])
                else:
                    _, pval = pearsonr(data.iloc[:, i], data.iloc[:, j])
                pval_matrix.iloc[i, j] = float(pval)
    return pval_matrix

pval_spearman = get_pvalues_matrix(df_num, method='spearman')

# ── 2.4 Korelasi dengan target (dengan significance) ─────────
print("\n─────────────────────────────────────────────────────────")
print("── Korelasi Spearman vs team_goals (p-value < 0.05) ──")
print("─────────────────────────────────────────────────────────")

target_corr = corr_spearman['team_goals'].drop('team_goals').copy()
target_pval = pval_spearman['team_goals'].drop('team_goals').copy()

# Filter significant correlations only
significant_mask = target_pval < 0.05
sig_corr = target_corr[significant_mask].sort_values(key=abs, ascending=False)

if len(sig_corr) > 0:
    for fitur, corr_val in sig_corr.items():
        pval = target_pval[fitur]
        direction = "↑" if corr_val > 0 else "↓"
        print(f"  {fitur:<30} r={float(corr_val):>7.4f}  p={float(pval):.4e}  {direction}")
else:
    print("  → Tidak ada korelasi yang signifikan (p < 0.05)")

print(f"\n  Total correlations tested: {len(target_corr)}")
print(f"  Significant (p < 0.05)   : {(target_pval < 0.05).sum()}")
print(f"  Non-significant (p ≥ 0.05): {(target_pval >= 0.05).sum()}")

# ── 2.5 Visualisasi Korelasi ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('PART 2: Heatmap Korelasi Fitur Numerik — Fase 2 EDA', fontsize=15, fontweight='bold')

mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

for ax, corr, metode in zip(axes, [corr_pearson, corr_spearman], ['Pearson', 'Spearman']):
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title(f'Korelasi {metode}', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig('./reports/figures/02_part2a_heatmap_korelasi.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] ./reports/figures/02_part2a_heatmap_korelasi.png")

# ── 2.6 Plot bar: korelasi fitur vs team_goals ────────────────
fig2, ax = plt.subplots(figsize=(12, 6))

korelasi_target = corr_spearman['team_goals'].drop('team_goals').sort_values()
colors_list = ['#FF5722' if v < 0 else '#2196F3' for v in korelasi_target.values]

ax.barh(range(len(korelasi_target)), korelasi_target.values, 
        color=colors_list, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(range(len(korelasi_target)))
ax.set_yticklabels(korelasi_target.index)
ax.set_title('Korelasi Spearman — Fitur vs team_goals', fontweight='bold', fontsize=12)
ax.set_xlabel('Korelasi')
ax.set_xlim(-1, 1)

# Add value labels and significance markers
for i, (idx, val) in enumerate(korelasi_target.items()):
    pval = target_pval[idx]
    sig_marker = '**' if pval < 0.01 else '*' if pval < 0.05 else '(ns)'
    ax.text(float(val) + (0.02 if val >= 0 else -0.02), i,
            f'{float(val):.3f} {sig_marker}',
            va='center', ha='left' if val >= 0 else 'right',
            fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('./reports/figures/02_part2b_korelasi_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] ./reports/figures/02_part2b_korelasi_bar.png")
print("Note: * = p<0.05, ** = p<0.01, (ns) = not significant")

# Summary Part 2
print("\n" + "═"*60)
print("SUMMARY: HEATMAP KORELASI")
print("═"*60)

top3 = corr_spearman['team_goals'].drop('team_goals').abs().sort_values(ascending=False).head(3)
print(f"\nTop 3 Correlations dengan team_goals:")
for idx, (fitur, val) in enumerate(top3.items(), 1):
    corr_val = corr_spearman['team_goals'][fitur]
    pval = target_pval[fitur]
    is_sig = "✓ Sig" if pval < 0.05 else "✗ Not sig"
    direction = "(+)" if corr_val > 0 else "(-)"
    print(f"  {idx}. {fitur:<30} r={float(corr_val):>7.4f} {direction} p={float(pval):.2e} {is_sig}")

# Multicollinearity check
print(f"\n─────────────────────────────────────────────────────────")
print(f"Multikolinaritas (r > 0.85 antara fitur non-target):")
print(f"─────────────────────────────────────────────────────────")

multikolinear = []
cols = corr_pearson.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        val = float(abs(corr_pearson.iloc[i, j]))
        if val > 0.85 and cols[i] not in ['team_goals', 'opp_goals'] and cols[j] not in ['team_goals', 'opp_goals']:
            multikolinear.append((cols[i], cols[j], val))

if multikolinear:
    print(f"Pasangan fitur yang berkorelasi tinggi:")
    for a, b, v in sorted(multikolinear, key=lambda x: x[2], reverse=True):
        print(f"   {a} ↔ {b} : r={v:.3f}")
    print("\nRekomendasi:")
    print("  • Feature selection untuk menghindari multicollinearity")
    print("  • Gunakan PCA atau supervised methods untuk dimensionality reduction")
else:
    print("✓ Tidak ada multikolinearitas ekstrem terdeteksi (r > 0.85)")

# =============================================================
# PART 3: TEMPORAL ANALYSIS (02c)
# =============================================================
print("\n" + "="*70)
print("PART 3: TEMPORAL ANALYSIS — Time Series Patterns")
print("="*70)

# ── 3.1 Ekstrak fitur temporal ────────────────────────────────
train['year']   = train['date'].str[:4].astype(int)
train['month']  = train['date'].str[5:7].astype(int)
train['decade'] = (train['year'] // 10) * 10

print(f"Rentang tahun : {train['year'].min()} — {train['year'].max()}")
print(f"Jumlah dekade : {train['decade'].nunique()}")
print(train.groupby('decade').size().rename('jumlah_pertandingan').to_string())

# ── 3.2 Agregasi per dekade ───────────────────────────────────
per_dekade = train.groupby('decade').agg(
    jumlah_pertandingan = ('total_goals', 'count'),
    rata_team_goals     = ('team_goals',  'mean'),
    rata_opp_goals      = ('opp_goals',   'mean'),
    rata_total_goals    = ('total_goals', 'mean'),
    median_total_goals  = ('total_goals', 'median'),
    std_total_goals     = ('total_goals', 'std'),
).round(3).reset_index()

print("\n── Agregasi per Dekade ──")
print(per_dekade.to_string(index=False))

# ── 3.3 Agregasi per tahun ───────────────────────────────────
per_tahun = train.groupby('year').agg(
    jumlah_pertandingan = ('total_goals', 'count'),
    rata_team_goals     = ('team_goals',  'mean'),
    rata_opp_goals      = ('opp_goals',   'mean'),
    rata_total_goals    = ('total_goals', 'mean'),
).round(3).reset_index()

# ── 3.4 Agregasi per bulan ───────────────────────────────────
nama_bulan = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
              7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}

per_bulan = train.groupby('month').agg(
    rata_total_goals = ('total_goals', 'mean'),
    std_total_goals = ('total_goals', 'std'),
    jumlah_pertandingan = ('total_goals', 'count'),
).round(3).reset_index()
per_bulan['nama_bulan'] = per_bulan['month'].map(nama_bulan)

# ── 3.5 Trend analysis — Mann-Kendall test ───────────────────
print("\n" + "─"*60)
print("TREND ANALYSIS — Temporal Patterns")
print("─"*60)

slope, p_val = kendalltau(range(len(per_tahun)), per_tahun['rata_total_goals'].values)
trend_result = 'Significant upward trend' if float(slope) > 0 and float(p_val) < 0.05 else \
               'Significant downward trend' if float(slope) < 0 and float(p_val) < 0.05 else \
               'No significant trend'

print(f"\nMann-Kendall Trend Test (Yearly Average Goals):")
print(f"  Slope (τ)   : {float(slope):.4f}")
print(f"  P-value     : {float(p_val):.4e}")
print(f"  Result      : {trend_result}")

# Volume analysis
min_games_per_year = per_tahun['jumlah_pertandingan'].min()
max_games_per_year = per_tahun['jumlah_pertandingan'].max()
sparse_years = per_tahun[per_tahun['jumlah_pertandingan'] < per_tahun['jumlah_pertandingan'].median()]

print(f"\nData Volume Notes:")
print(f"  Min games/year   : {min_games_per_year:.0f}")
print(f"  Max games/year   : {max_games_per_year:.0f}")
print(f"  Sparse years     : {len(sparse_years)} (below median)")
if len(sparse_years) > 0:
    print(f"    Years: {sparse_years['year'].values.astype(int).tolist()}")
    print(f"    ⚠️  Consider weighting by sample size in analysis")

# ── 3.6 Visualisasi Temporal ──────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('PART 3: Analisis Temporal — Fase 2 EDA', fontsize=15, fontweight='bold', y=0.995)

# -- 3.6a. Rata-rata total gol per dekade (bar)
ax = axes[0, 0]
bars = ax.bar(per_dekade['decade'].astype(str), per_dekade['rata_total_goals'],
              color='#2196F3', alpha=0.8, edgecolor='white', width=0.6)
ax.errorbar(range(len(per_dekade)), per_dekade['rata_total_goals'],
            yerr=per_dekade['std_total_goals'], fmt='none',
            color='black', capsize=4, linewidth=1.2)
ax.set_title('Rata-rata Total Gol per Dekade', fontweight='bold')
ax.set_xlabel('Dekade')
ax.set_ylabel('Rata-rata Gol')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, per_dekade['rata_total_goals']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# -- 3.6b. Stacked bar: team vs opp goals per dekade
ax = axes[0, 1]
x = np.arange(len(per_dekade))
ax.bar(x, per_dekade['rata_team_goals'], label='Gol Tim',   color='#2196F3', alpha=0.85, edgecolor='white')
ax.bar(x, per_dekade['rata_opp_goals'],  label='Gol Lawan', color='#FF5722', alpha=0.85, edgecolor='white',
       bottom=per_dekade['rata_team_goals'])
ax.set_xticks(x)
ax.set_xticklabels(per_dekade['decade'].astype(str), rotation=45)
ax.set_title('Komposisi Gol Tim vs Lawan per Dekade', fontweight='bold')
ax.set_xlabel('Dekade')
ax.set_ylabel('Rata-rata Gol')
ax.legend(fontsize=9)

# -- 3.6c. Tren rata-rata total gol per tahun (line)
ax = axes[1, 0]
ax.plot(per_tahun['year'], per_tahun['rata_total_goals'],
        color='#4CAF50', linewidth=1.5, alpha=0.7)
# rolling mean 10 tahun
rolling = per_tahun.set_index('year')['rata_total_goals'].rolling(10, center=True).mean()
ax.plot(rolling.index, rolling.values, color='#E91E63', linewidth=2.5,
        linestyle='--', label='Rolling Mean (10 thn)')
ax.set_title('Tren Rata-rata Total Gol per Tahun', fontweight='bold')
ax.set_xlabel('Tahun')
ax.set_ylabel('Rata-rata Gol')
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.tick_params(axis='x', rotation=45)

# -- 3.6d. Jumlah pertandingan per tahun (volume data)
ax = axes[1, 1]
ax.fill_between(per_tahun['year'], per_tahun['jumlah_pertandingan'],
                color='#9C27B0', alpha=0.4)
ax.plot(per_tahun['year'], per_tahun['jumlah_pertandingan'],
        color='#9C27B0', linewidth=1.5)
ax.set_title('Volume Pertandingan per Tahun', fontweight='bold')
ax.set_xlabel('Tahun')
ax.set_ylabel('Jumlah Pertandingan')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.tick_params(axis='x', rotation=45)

# -- 3.6e. Seasonality: rata-rata gol per bulan
ax = axes[2, 0]
ax.bar(per_bulan['nama_bulan'], per_bulan['rata_total_goals'],
       color='#FF9800', alpha=0.8, edgecolor='white')
# Add error bars (standard deviation)
ax.errorbar(range(len(per_bulan)), per_bulan['rata_total_goals'].values,
            yerr=per_bulan['std_total_goals'].values, fmt='none',
            color='black', capsize=3, linewidth=1, alpha=0.6)
ax.set_title('Seasonality — Rata-rata Total Gol per Bulan (with Std Dev)', fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Rata-rata Gol')
for i, (_, row) in enumerate(per_bulan.iterrows()):
    ax.text(i, row['rata_total_goals'] + 0.03, f"{row['rata_total_goals']:.2f}",
            ha='center', va='bottom', fontsize=7.5)

# -- 3.6f. Jumlah pertandingan per bulan
ax = axes[2, 1]
ax.bar(per_bulan['nama_bulan'], per_bulan['jumlah_pertandingan'],
       color='#607D8B', alpha=0.8, edgecolor='white')
ax.set_title('Volume Pertandingan per Bulan', fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Pertandingan')

plt.tight_layout()
plt.savefig('./reports/figures/02_part3_analisis_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] ./reports/figures/02_part3_analisis_temporal.png")

# Summary Part 3
print("\n" + "═"*60)
print("SUMMARY: ANALISIS TEMPORAL")
print("═"*60)

dekade_tertinggi = per_dekade.loc[per_dekade['rata_total_goals'].idxmax()]
dekade_terendah  = per_dekade.loc[per_dekade['rata_total_goals'].idxmin()]
bulan_tersibuk   = per_bulan.loc[per_bulan['jumlah_pertandingan'].idxmax()]

print(f"\nDECADE ANALYSIS:")
print(f"  → Dekade gol tertinggi : {int(dekade_tertinggi['decade'])}an "
      f"(rata-rata {dekade_tertinggi['rata_total_goals']:.2f} gol/match)")
print(f"  → Dekade gol terendah  : {int(dekade_terendah['decade'])}an "
      f"(rata-rata {dekade_terendah['rata_total_goals']:.2f} gol/match)")

print(f"\nSEASONALITY:")
print(f"  → Bulan paling sibuk   : {bulan_tersibuk['nama_bulan']} "
      f"({int(bulan_tersibuk['jumlah_pertandingan'])} pertandingan)")
print(f"  → Variance across months: σ={per_bulan['std_total_goals'].mean():.3f}")

print(f"\nVOLUME TREND:")
volume_threshold = per_dekade['jumlah_pertandingan'].median()
vol_recovery_decade = int(per_dekade.loc[per_dekade['jumlah_pertandingan'] > volume_threshold, 'decade'].min())
print(f"  → Data volume increases from {vol_recovery_decade}0 onwards")
print(f"  → Median matches/decade: {volume_threshold:.0f}")
print(f"\n[CRITICAL FOR PHASE 3]: Filter train.csv to exclude pre-{vol_recovery_decade}0 matches")
print(f"  Reason: Severe structural drift detected in early decades")

# =============================================================
# FINAL RECOMMENDATIONS
# =============================================================
print("\n" + "="*70)
print("FINAL RECOMMENDATIONS — MINIMAL & LOSS-AWARE")
print("="*70)

print(f"""
MINIMAL FIXES FOR PHASE 3-5:
═══════════════════════════════════════════════════════════════════════════════

1. ✗ NULLIFIED: Never transform, log-scale, or standardize target variables.
   REASON: Goal counts must remain raw integers to train Poisson/Negative Binomial.
   
2. HARD CLIPPING (99.9th percentile):
   team_goals: CLIP at {p999_team} goals  
   opp_goals:  CLIP at {p999_opp} goals
   → Use these exact integers in Phase 5 feature engineering

3. VOL_RECOVERY_DECADE for Phase 3 (Elo Reconstruction):
   vol_recovery_decade = {vol_recovery_decade}
   → FILTER train.csv: Exclude all matches before {vol_recovery_decade}0
   → REASON: Pre-{vol_recovery_decade}0 shows structural drift + low volume
   
4. MODEL REQUIREMENTS (AW-MAE Loss):
   ✓ Poisson/Negative Binomial regression (count-aware link)
   ✓ Preserve original integer count space
   ✓ Validate with exact-match rate + L1 distance distribution
""")


# =============================================================
# PART 4: OUTCOME DISTRIBUTION ANALYSIS (02d)
# Prerequisites untuk Post-Processing Threshold
# =============================================================
print("\n" + "="*70)
print("PART 4: OUTCOME DISTRIBUTION — W/D/L & Goal Difference Analysis")
print("="*70)

# ── 4.0 Derive outcome & goal difference ─────────────────────
train['goal_diff']  = train['team_goals'] - train['opp_goals']
train['outcome']    = np.where(train['team_goals'] > train['opp_goals'], 'Win',
                     np.where(train['team_goals'] == train['opp_goals'], 'Draw', 'Loss'))
train['outcome_num'] = np.where(train['team_goals'] > train['opp_goals'], 2,
                      np.where(train['team_goals'] == train['opp_goals'], 1, 0))

print(f"\nOutcome distribution (global):")
outcome_global = train['outcome'].value_counts()
outcome_pct    = train['outcome'].value_counts(normalize=True) * 100
for o in ['Win', 'Draw', 'Loss']:
    print(f"  {o:<6}: {outcome_global[o]:>6} ({outcome_pct[o]:.1f}%)")

# ── 4.1 W/D/L per turnamen ───────────────────────────────────
print("\n── W/D/L Distribution per Tournament ──")

# Ambil top 15 turnamen by volume
top_tournaments = train['tournament'].value_counts().head(15).index.tolist()

# Tambahkan World Cup secara eksplisit jika belum masuk
for wc_name in ['FIFA World Cup', 'World Cup']:
    if wc_name in train['tournament'].values and wc_name not in top_tournaments:
        top_tournaments.append(wc_name)

outcome_per_tournament = (
    train[train['tournament'].isin(top_tournaments)]
    .groupby(['tournament', 'outcome'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=['Win', 'Draw', 'Loss'], fill_value=0)
)

# Tambah total dan persentase
outcome_per_tournament['Total']   = outcome_per_tournament.sum(axis=1)
outcome_per_tournament['Win%']    = (outcome_per_tournament['Win']  / outcome_per_tournament['Total'] * 100).round(1)
outcome_per_tournament['Draw%']   = (outcome_per_tournament['Draw'] / outcome_per_tournament['Total'] * 100).round(1)
outcome_per_tournament['Loss%']   = (outcome_per_tournament['Loss'] / outcome_per_tournament['Total'] * 100).round(1)

outcome_per_tournament = outcome_per_tournament.sort_values('Total', ascending=False)
print(outcome_per_tournament[['Win%', 'Draw%', 'Loss%', 'Total']].to_string())

# ── 4.2 Home win rate per turnamen ───────────────────────────
print("\n── Home Win Rate per Tournament ──")

home_matches = train[(train['is_home'] == 1) & (train['neutral'] == 0)]
home_wr = (
    home_matches[home_matches['tournament'].isin(top_tournaments)]
    .groupby('tournament')
    .agg(
        total_home   = ('outcome', 'count'),
        home_wins    = ('outcome', lambda x: (x == 'Win').sum()),
        home_draws   = ('outcome', lambda x: (x == 'Draw').sum()),
        home_losses  = ('outcome', lambda x: (x == 'Loss').sum()),
    )
)
home_wr['home_win_rate']  = (home_wr['home_wins']   / home_wr['total_home'] * 100).round(1)
home_wr['home_draw_rate'] = (home_wr['home_draws']  / home_wr['total_home'] * 100).round(1)
home_wr = home_wr.sort_values('home_win_rate', ascending=False)
print(home_wr[['total_home', 'home_win_rate', 'home_draw_rate']].to_string())

# ── 4.3 Distribusi selisih gol ───────────────────────────────
print("\n── Goal Difference Distribution ──")
gd_dist = train['goal_diff'].value_counts().sort_index()
gd_pct  = train['goal_diff'].value_counts(normalize=True).sort_index() * 100

print(f"{'GD':>5} | {'Count':>7} | {'Pct':>6}")
print("─" * 25)
for gd, count in gd_dist.items():
    bar = '█' * int(gd_pct[gd])
    print(f"{gd:>5} | {count:>7} | {gd_pct[gd]:>5.1f}% {bar}")

# Kalkulasi threshold penting
pct_gd_zero  = gd_pct.get(0, 0)
pct_gd_one   = (gd_pct.get(1, 0) + gd_pct.get(-1, 0))
pct_gd_small = sum(gd_pct.get(g, 0) for g in [-1, 0, 1])

print(f"\n[THRESHOLD INSIGHTS — untuk Post-Processing]")
print(f"  Draw (GD=0)          : {pct_gd_zero:.1f}% pertandingan")
print(f"  Selisih 1 gol        : {pct_gd_one:.1f}% pertandingan")
print(f"  Selisih ≤ 1 gol      : {pct_gd_small:.1f}% pertandingan")
print(f"  → Mayoritas match berakhir ketat — model harus konservatif")

# ── 4.4 Analisis khusus World Cup ────────────────────────────
print("\n── World Cup Specific Analysis ──")

wc_mask = train['tournament'].str.contains('World Cup', case=False, na=False)
train_wc = train[wc_mask]

if len(train_wc) > 0:
    print(f"Total pertandingan World Cup : {len(train_wc)}")
    print(f"\nDistribusi Outcome WC:")
    wc_outcome = train_wc['outcome'].value_counts(normalize=True) * 100
    for o in ['Win', 'Draw', 'Loss']:
        if o in wc_outcome:
            print(f"  {o:<6}: {wc_outcome[o]:.1f}%")

    print(f"\nStatistik Gol World Cup:")
    print(f"  Rata-rata team_goals : {train_wc['team_goals'].mean():.3f}")
    print(f"  Rata-rata opp_goals  : {train_wc['opp_goals'].mean():.3f}")
    print(f"  Rata-rata total gol  : {train_wc['total_goals'].mean():.3f}")
    print(f"  Median total gol     : {train_wc['total_goals'].median():.1f}")
    print(f"  % Draw di WC         : {(train_wc['outcome'] == 'Draw').mean()*100:.1f}%")
    print(f"  % Draw global        : {(train['outcome'] == 'Draw').mean()*100:.1f}%")

    wc_gd_zero = (train_wc['goal_diff'] == 0).mean() * 100
    wc_gd_one  = (train_wc['goal_diff'].abs() == 1).mean() * 100
    print(f"  % Selisih 0 gol      : {wc_gd_zero:.1f}%")
    print(f"  % Selisih 1 gol      : {wc_gd_one:.1f}%")
else:
    print("  ⚠️ Tidak ada data World Cup di train set — cek nama kolom tournament")

# ── 4.5 ELO vs Outcome validation ────────────────────────────
print("\n── ELO vs Outcome Validation ──")

if 'elo_team' in train.columns and 'elo_opponent' in train.columns:
    train['elo_diff_match'] = train['elo_team'] - train['elo_opponent']

    # Binning ELO diff
    bins   = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
    labels = ['<-200', '-200:-100', '-100:-50', '-50:0',
              '0:50',  '50:100',   '100:200',  '>200']
    train['elo_bin'] = pd.cut(train['elo_diff_match'], bins=bins, labels=labels)

    elo_outcome = train.groupby('elo_bin', observed=True)['outcome_num'].agg(
        win_rate  = lambda x: (x == 2).mean() * 100,
        draw_rate = lambda x: (x == 1).mean() * 100,
        loss_rate = lambda x: (x == 0).mean() * 100,
        count     = 'count'
    ).round(1)

    print(f"\n{'ELO Diff':<15} | {'Win%':>6} | {'Draw%':>6} | {'Loss%':>6} | {'N':>6}")
    print("─" * 50)
    for idx, row in elo_outcome.iterrows():
        print(f"{str(idx):<15} | {row['win_rate']:>6.1f} | {row['draw_rate']:>6.1f} | {row['loss_rate']:>6.1f} | {int(row['count']):>6}")

    print(f"\n[ELO VALIDATION]")
    high_elo_winrate = elo_outcome.loc['>200', 'win_rate'] if '>200' in elo_outcome.index else 'N/A'
    low_elo_winrate  = elo_outcome.loc['<-200', 'win_rate'] if '<-200' in elo_outcome.index else 'N/A'
    print(f"  ELO diff > 200  → Win rate: {high_elo_winrate}%")
    print(f"  ELO diff < -200 → Win rate: {low_elo_winrate}%")
    print(f"  → Semakin besar ELO diff, semakin prediktif hasilnya")
    print(f"  → ELO diff kecil (-50 to 50): zona ambiguous → draw lebih likely")

# ── 4.6 Visualisasi Part 4 ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('PART 4: Outcome Distribution Analysis — Fase 2 EDA',
             fontsize=15, fontweight='bold', y=0.99)

# -- 4.6a W/D/L per tournament (stacked bar)
ax = axes[0, 0]
plot_data = outcome_per_tournament[['Win%', 'Draw%', 'Loss%']].head(12)
plot_data.plot(kind='barh', ax=ax, stacked=True,
               color=['#4CAF50', '#FF9800', '#F44336'],
               edgecolor='white', alpha=0.85)
ax.set_title('W/D/L % per Tournament (Top 12)', fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.legend(loc='lower right', fontsize=8)
ax.axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# -- 4.6b Goal difference distribution
ax = axes[0, 1]
gd_range  = range(int(train['goal_diff'].min()), int(train['goal_diff'].max()) + 1)
gd_counts = [gd_dist.get(g, 0) for g in gd_range]
colors_gd = ['#F44336' if g < 0 else '#FF9800' if g == 0 else '#4CAF50' for g in gd_range]
ax.bar(list(gd_range), gd_counts, color=colors_gd, edgecolor='white', alpha=0.85)
ax.set_title('Distribusi Selisih Gol (Goal Difference)', fontweight='bold')
ax.set_xlabel('Selisih Gol (team - opp)')
ax.set_ylabel('Frekuensi')
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.text(0.5, 0.95, f"Draw: {pct_gd_zero:.1f}%",
        transform=ax.transAxes, ha='center', fontsize=10,
        color='#FF9800', fontweight='bold')

# -- 4.6c Home win rate per tournament
ax = axes[1, 0]
home_plot = home_wr.head(12)[['home_win_rate', 'home_draw_rate']].copy()
home_plot['home_loss_rate'] = 100 - home_plot['home_win_rate'] - home_plot['home_draw_rate']
home_plot.plot(kind='barh', ax=ax, stacked=True,
               color=['#4CAF50', '#FF9800', '#F44336'],
               edgecolor='white', alpha=0.85)
ax.set_title('Home Win/Draw/Loss Rate per Tournament', fontweight='bold')
ax.set_xlabel('Persentase (%)')
ax.axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(['Home Win%', 'Home Draw%', 'Home Loss%'], fontsize=8, loc='lower right')

# -- 4.6d ELO diff vs Win rate
ax = axes[1, 1]
if 'elo_diff_match' in train.columns:
    ax.bar(range(len(elo_outcome)), elo_outcome['win_rate'].values,
           color='#2196F3', alpha=0.8, edgecolor='white', label='Win%')
    ax.bar(range(len(elo_outcome)), elo_outcome['draw_rate'].values,
           bottom=elo_outcome['win_rate'].values,
           color='#FF9800', alpha=0.8, edgecolor='white', label='Draw%')
    ax.bar(range(len(elo_outcome)), elo_outcome['loss_rate'].values,
           bottom=(elo_outcome['win_rate'] + elo_outcome['draw_rate']).values,
           color='#F44336', alpha=0.8, edgecolor='white', label='Loss%')
    ax.set_xticks(range(len(elo_outcome)))
    ax.set_xticklabels(elo_outcome.index, rotation=30, ha='right', fontsize=8)
    ax.set_title('Win/Draw/Loss Rate per ELO Difference Bin', fontweight='bold')
    ax.set_xlabel('ELO Difference (team - opp)')
    ax.set_ylabel('Persentase (%)')
    ax.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./reports/figures/02_part4_outcome_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] ./reports/figures/02_part4_outcome_distribution.png")

# ── 4.7 Threshold Summary — langsung dipakai di post-processing ──
print("\n" + "═"*65)
print("THRESHOLD SUMMARY — SIMPAN INI UNTUK POST-PROCESSING FASE 5")
print("═"*65)

draw_rate_global = (train['outcome'] == 'Draw').mean() * 100
win_rate_global  = (train['outcome'] == 'Win').mean() * 100
loss_rate_global = (train['outcome'] == 'Loss').mean() * 100

draw_rate_wc = (train_wc['outcome'] == 'Draw').mean() * 100 if len(train_wc) > 0 else None
pct_gd_le1   = sum(gd_pct.get(g, 0) for g in [-1, 0, 1])

print(f"""
GLOBAL BASELINE:
  Win rate  : {win_rate_global:.1f}%
  Draw rate : {draw_rate_global:.1f}%
  Loss rate : {loss_rate_global:.1f}%

WORLD CUP BASELINE:
  Draw rate : {draw_rate_wc:.1f}% (vs global {draw_rate_global:.1f}%)

GOAL DIFFERENCE:
  % GD = 0  : {pct_gd_zero:.1f}%  → threshold draw correction
  % |GD| ≤ 1: {pct_gd_small:.1f}%  → majority of matches tight

POST-PROCESSING RULES (gunakan di Fase 5):
  Rule 1: if prob_draw > {draw_rate_global/100:.2f} and |pred_team - pred_opp| > 1
           → samakan skor (force draw)
  Rule 2: if prob_win > 0.70 and pred_team <= pred_opp
           → pred_team = pred_opp + 1
  Rule 3: if prob_lose > 0.70 and pred_team >= pred_opp
           → pred_opp = pred_team + 1
  Rule 4: World Cup → gunakan threshold draw lebih tinggi ({(draw_rate_wc or draw_rate_global)/100:.2f})
           karena draw rate WC lebih {'tinggi' if draw_rate_wc and draw_rate_wc > draw_rate_global else 'rendah'} dari global

ELO AMBIGUOUS ZONE:
  ELO diff antara -50 sampai 50 → draw lebih likely
  → Pertimbangkan prediksi 1-1 atau 0-0 sebagai fallback
""")

print("\n" + "="*70)
print("EDA ANALYSIS COMPLETE!")
print("="*70)
print(f"All visualizations saved to: ./reports/figures/")
print(f"Figures generated:")
print(f"  - 02_part1_distribusi_gol.png")
print(f"  - 02_part2a_heatmap_korelasi.png")
print(f"  - 02_part2b_korelasi_bar.png")
print(f"  - 02_part3_analisis_temporal.png")
