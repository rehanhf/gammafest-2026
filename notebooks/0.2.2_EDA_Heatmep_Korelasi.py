import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('./reports/figures', exist_ok=True)


train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})


kolom_numerik = [
    'team_goals', 'opp_goals',
    'altitude_venue', 'temperature_venue',
    'distance_travel_team', 'distance_travel_opp',
    'gdp_per_capita_team', 'gdp_per_capita_opp',
    'population_team', 'population_opp'
]

# filter hanya kolom yang benar-benar ada di dataframe
kolom_numerik = [c for c in kolom_numerik if c in train.columns]
print(f"Kolom yang digunakan ({len(kolom_numerik)}): {kolom_numerik}")

df_num = train[kolom_numerik].copy()


corr_pearson  = df_num.corr(method='pearson')
corr_spearman = df_num.corr(method='spearman')


print("\n── Korelasi Pearson vs team_goals ──")
print(corr_pearson['team_goals'].drop('team_goals').sort_values(key=abs, ascending=False).round(4))

print("\n── Korelasi Spearman vs team_goals ──")
print(corr_spearman['team_goals'].drop('team_goals').sort_values(key=abs, ascending=False).round(4))

#Visualisasi 
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('Heatmap Korelasi Fitur Numerik — Fase 2 EDA', fontsize=15, fontweight='bold')

mask = np.triu(np.ones_like(corr_pearson, dtype=bool))  # sembunyikan segitiga atas (duplikat)

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
plt.savefig('./reports/figures/02b_heatmap_korelasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[SAVED] ./reports/figures/02b_heatmap_korelasi.png")

#Plot bar: korelasi fitur vs team_goals 
fig2, ax = plt.subplots(figsize=(10, 5))

korelasi_target = corr_spearman['team_goals'].drop('team_goals').sort_values()
warna = ['#FF5722' if v < 0 else '#2196F3' for v in korelasi_target]

korelasi_target.plot.barh(ax=ax, color=warna, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Korelasi Spearman — Fitur vs team_goals', fontweight='bold')
ax.set_xlabel('Korelasi')
ax.set_xlim(-1, 1)

for i, (val, patch) in enumerate(zip(korelasi_target, ax.patches)):
    ax.text(val + (0.02 if val >= 0 else -0.02), i,
            f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right',
            fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('./reports/figures/02b_korelasi_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("[SAVED] ./reports/figures/02b_korelasi_bar.png")

#Ringkasan temuan 
print("\n" + "═"*55)
print("RINGKASAN TEMUAN — HEATMAP KORELASI")
print("═"*55)

top3 = corr_spearman['team_goals'].drop('team_goals').abs().sort_values(ascending=False).head(3)
for fitur, val in top3.items():
    arah = "positif" if corr_spearman['team_goals'][fitur] > 0 else "negatif"
    print(f"→ {fitur:<30} r={val:.3f} ({arah})")

multikolinear = []
cols = corr_pearson.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        val = abs(corr_pearson.iloc[i, j])
        if val > 0.85 and cols[i] not in ['team_goals', 'opp_goals'] and cols[j] not in ['team_goals', 'opp_goals']:
            multikolinear.append((cols[i], cols[j], round(val, 3)))

if multikolinear:
    print(f"\n→ Pasangan fitur multikolinear (r > 0.85):")
    for a, b, v in multikolinear:
        print(f"   {a} ↔ {b} : {v}")
    print("   Rekomendasi: Pertimbangkan drop salah satu di fase feature engineering.")
else:
    print("\n→ Tidak ada multikolinearitas ekstrem terdeteksi (r > 0.85).")