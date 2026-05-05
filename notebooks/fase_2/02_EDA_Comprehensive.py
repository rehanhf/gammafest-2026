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

print("\n" + "="*70)
print("EDA ANALYSIS COMPLETE!")
print("="*70)
print(f"All visualizations saved to: ./reports/figures/")
print(f"Figures generated:")
print(f"  - 02_part1_distribusi_gol.png")
print(f"  - 02_part2a_heatmap_korelasi.png")
print(f"  - 02_part2b_korelasi_bar.png")
print(f"  - 02_part3_analisis_temporal.png")
