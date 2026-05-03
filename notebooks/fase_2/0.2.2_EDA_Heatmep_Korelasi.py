import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('./reports/figures', exist_ok=True)


train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})

#mendefinisikan kolom-kolom numerik untuk analisis korelasi
#kolom-kolom ini mencakup fitur gol, venue, travel distance, ekonomi, dan populasi
#kami akan menganalisis seberapa kuat hubungan linear antara fitur-fitur ini
kolom_numerik = [
    'team_goals', 'opp_goals',
    'altitude_venue', 'temperature_venue',
    'distance_travel_team', 'distance_travel_opp',
    'gdp_per_capita_team', 'gdp_per_capita_opp',
    'population_team', 'population_opp'
]

#filter hanya kolom yang benar-benar ada di dataframe
#step ini penting untuk menghindari error jika ada kolom yang hilang atau dihapus
kolom_numerik = [c for c in kolom_numerik if c in train.columns]
print(f"Kolom yang digunakan ({len(kolom_numerik)}): {kolom_numerik}")

df_num = train[kolom_numerik].copy()

#menghitung matrix korelasi menggunakan method Pearson dan Spearman
#Pearson mengukur korelasi linear (hubungan garis lurus)
#Spearman mengukur korelasi rank-based (hubungan monotonic)
#kedua metode ini membantu kami memahami hubungan antar fitur
corr_pearson  = df_num.corr(method='pearson')
corr_spearman = df_num.corr(method='spearman')


print("\n── Korelasi Pearson vs team_goals ──")
print(corr_pearson['team_goals'].drop('team_goals').sort_values(key=abs, ascending=False).round(4))

print("\n── Korelasi Spearman vs team_goals ──")
print(corr_spearman['team_goals'].drop('team_goals').sort_values(key=abs, ascending=False).round(4))

#visualisasi heatmap korelasi dengan Pearson dan Spearman
#heatmap menunjukkan korelasi antara semua pasangan fitur dengan color intensity
#warna lebih merah berarti korelasi positif yang kuat, warna biru berarti negatif
#angka di setiap cell menunjukkan nilai korelasi yang tepat (-1 hingga 1)
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('Heatmap Korelasi Fitur Numerik — Fase 2 EDA', fontsize=15, fontweight='bold')

#sembunyikan segitiga atas karena merupakan duplikat (mirror) dari bawah
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))  #sembunyikan segitiga atas (duplikat)

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

#plot bar untuk menampilkan korelasi fitur vs team_goals (target variable)
#ini adalah fitur korelasi yang paling penting untuk prediksi model
#warna biru menunjukkan korelasi positif (lebih banyak feature → lebih banyak gol)
#warna orange menunjukkan korelasi negatif (lebih banyak feature → lebih sedikit gol)
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

#ringkasan temuan dari analisis korelasi
#bagian ini mengidentifikasi fitur-fitur paling penting yang berkorelasi dengan target
#serta mendeteksi multikolinearitas yang dapat merusak model
print("\n" + "═"*55)
print("RINGKASAN TEMUAN — HEATMAP KORELASI")
print("═"*55)

top3 = corr_spearman['team_goals'].drop('team_goals').abs().sort_values(ascending=False).head(3)
for fitur, val in top3.items():
    arah = "positif" if corr_spearman['team_goals'][fitur] > 0 else "negatif"
    print(f"→ {fitur:<30} r={val:.3f} ({arah})")

#deteksi multikolinearity antar fitur
#multikolinearity terjadi ketika dua fitur sangat berkorelasi satu sama lain
#ini dapat menyebabkan masalah dalam model karena redundansi informasi
#threshold 0.85 menunjukkan korelasi sangat kuat yang perlu dipertimbangkan
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