import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('./reports/figures', exist_ok=True)
train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})

train['year']   = train['date'].str[:4].astype(int)
train['month']  = train['date'].str[5:7].astype(int)
train['decade'] = (train['year'] // 10) * 10
train['total_goals'] = train['team_goals'] + train['opp_goals']

print(f"Rentang tahun : {train['year'].min()} — {train['year'].max()}")
print(f"Jumlah dekade : {train['decade'].nunique()}")
print(train.groupby('decade').size().rename('jumlah_pertandingan').to_string())


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


per_tahun = train.groupby('year').agg(
    jumlah_pertandingan = ('total_goals', 'count'),
    rata_team_goals     = ('team_goals',  'mean'),
    rata_opp_goals      = ('opp_goals',   'mean'),
    rata_total_goals    = ('total_goals', 'mean'),
).round(3).reset_index()


nama_bulan = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
              7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}

per_bulan = train.groupby('month').agg(
    rata_total_goals = ('total_goals', 'mean'),
    jumlah_pertandingan = ('total_goals', 'count'),
).round(3).reset_index()
per_bulan['nama_bulan'] = per_bulan['month'].map(nama_bulan)

#Visualisasi 
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('Analisis Temporal — Fase 2 EDA', fontsize=15, fontweight='bold', y=0.99)

#Rata-rata total gol per dekade (bar)
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

#Stacked bar: team vs opp goals per dekade
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

# Tren rata-rata total gol per tahun (line)
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

#Jumlah pertandingan per tahun (volume data)
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

#Seasonality: rata-rata gol per bulan
ax = axes[2, 0]
ax.bar(per_bulan['nama_bulan'], per_bulan['rata_total_goals'],
       color='#FF9800', alpha=0.8, edgecolor='white')
ax.set_title('Seasonality — Rata-rata Total Gol per Bulan', fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Rata-rata Gol')
for i, (_, row) in enumerate(per_bulan.iterrows()):
    ax.text(i, row['rata_total_goals'] + 0.03, f"{row['rata_total_goals']:.2f}",
            ha='center', va='bottom', fontsize=7.5)

#Jumlah pertandingan per bulan
ax = axes[2, 1]
ax.bar(per_bulan['nama_bulan'], per_bulan['jumlah_pertandingan'],
       color='#607D8B', alpha=0.8, edgecolor='white')
ax.set_title('Volume Pertandingan per Bulan', fontweight='bold')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Pertandingan')

plt.tight_layout()
plt.savefig('./reports/figures/02c_analisis_temporal.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[SAVED] ./reports/figures/02c_analisis_temporal.png")

# ── 7. Ringkasan temuan ───────────────────────────────────────
print("\n" + "═"*55)
print("RINGKASAN TEMUAN — ANALISIS TEMPORAL")
print("═"*55)

dekade_tertinggi = per_dekade.loc[per_dekade['rata_total_goals'].idxmax()]
dekade_terendah  = per_dekade.loc[per_dekade['rata_total_goals'].idxmin()]
bulan_tersibuk   = per_bulan.loc[per_bulan['jumlah_pertandingan'].idxmax()]

print(f"→ Dekade gol tertinggi : {int(dekade_tertinggi['decade'])}an "
      f"(rata-rata {dekade_tertinggi['rata_total_goals']:.2f} gol/match)")
print(f"→ Dekade gol terendah  : {int(dekade_terendah['decade'])}an "
      f"(rata-rata {dekade_terendah['rata_total_goals']:.2f} gol/match)")
print(f"→ Bulan paling sibuk   : {bulan_tersibuk['nama_bulan']} "
      f"({int(bulan_tersibuk['jumlah_pertandingan'])} pertandingan)")
print(f"→ Volume data meningkat signifikan mulai dekade: "
      f"{int(per_dekade.loc[per_dekade['jumlah_pertandingan'] > per_dekade['jumlah_pertandingan'].median(), 'decade'].min())}an")
print(f"→ Rekomendasi: Tambahkan fitur 'decade' dan 'month' sebagai fitur kategorik di fase feature engineering.")