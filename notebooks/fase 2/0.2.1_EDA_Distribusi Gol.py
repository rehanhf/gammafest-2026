import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})

print(f"Shape: {train.shape}")
print(train[['team_goals', 'opp_goals']].describe().round(3))


for col in ['team_goals', 'opp_goals']:
    s = train[col]
    print(f"\n{'─'*40}")
    print(f"Kolom       : {col}")
    print(f"Mean        : {s.mean():.3f}")
    print(f"Median      : {s.median():.3f}")
    print(f"Std         : {s.std():.3f}")
    print(f"Skewness    : {s.skew():.3f}  {'(right-skewed)' if s.skew() > 0.5 else '(approx normal)' if abs(s.skew()) < 0.5 else '(left-skewed)'}")
    print(f"Kurtosis    : {s.kurtosis():.3f}")
    print(f"Max         : {s.max()}")


def deteksi_outlier_iqr(series, label):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas  = Q3 + 1.5 * IQR
    outlier = series[(series < batas_bawah) | (series > batas_atas)]
    print(f"\n[OUTLIER] {label}")
    print(f"  IQR         : {IQR}")
    print(f"  Batas bawah : {batas_bawah}")
    print(f"  Batas atas  : {batas_atas}")
    print(f"  Jumlah outlier: {len(outlier)} ({len(outlier)/len(series)*100:.2f}%)")
    print(f"  Nilai outlier atas: {sorted(outlier[outlier > batas_atas].unique(), reverse=True)[:10]}")
    return batas_atas

batas_team = deteksi_outlier_iqr(train['team_goals'], 'team_goals')
batas_opp  = deteksi_outlier_iqr(train['opp_goals'],  'opp_goals')


fig = plt.figure(figsize=(18, 14))
fig.suptitle('Distribusi Gol — Fase 2 EDA', fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

warna = {'team_goals': '#2196F3', 'opp_goals': '#FF5722'}

for i, col in enumerate(['team_goals', 'opp_goals']):
    label = 'Tim' if col == 'team_goals' else 'Lawan'
    c     = warna[col]

    # -- histogram + KDE
    ax1 = fig.add_subplot(gs[0, i])
    train[col].plot.hist(bins=range(0, int(train[col].max()) + 2),
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

# -- distribusi gabungan (total gol per pertandingan)
train['total_goals'] = train['team_goals'] + train['opp_goals']
ax3 = fig.add_subplot(gs[2, 0])
train['total_goals'].plot.hist(bins=range(0, int(train['total_goals'].max()) + 2),
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

plt.savefig('./reports/figures/02a_distribusi_gol.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[SAVED] ./reports/figures/02a_distribusi_gol.png")

# ── 5. Ringkasan temuan ───────────────────────────────────────
print("\n" + "═"*50)
print("RINGKASAN TEMUAN — DISTRIBUSI GOL")
print("═"*50)
print(f"→ Skewness team_goals : {train['team_goals'].skew():.3f} (right-skewed, ekor panjang ke kanan)")
print(f"→ Skewness opp_goals  : {train['opp_goals'].skew():.3f}")
print(f"→ Outlier atas team   : > {batas_team:.1f} gol")
print(f"→ Outlier atas opp    : > {batas_opp:.1f} gol")
print(f"→ Rata-rata total gol : {train['total_goals'].mean():.2f} per pertandingan")
print(f"→ Rekomendasi         : Pertimbangkan log1p transform atau clipping di fase feature engineering")