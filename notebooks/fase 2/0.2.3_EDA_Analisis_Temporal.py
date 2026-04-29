#analisis tren temporal (time-series) untuk fitur tanggal dan target.
#input: dataframe (df), kolom_tanggal (str), kolom_target (str).
#output: plot tren agregat harian/bulanan.

import pandas as pd
import matplotlib.pyplot as plt

def analisis_temporal(df, kolom_tanggal='date', kolom_target='target'):
    df_temp = df.copy()
    df_temp[kolom_tanggal] = pd.to_datetime(df_temp[kolom_tanggal], errors='coerce')
    df_temp = df_temp.dropna(subset=[kolom_tanggal])
    df_temp = df_temp.sort_values(kolom_tanggal)

    #agregasi harian
    daily = df_temp.groupby(kolom_tanggal)[kolom_target].mean()

    plt.figure(figsize=(12, 5))
    daily.plot(marker='o', markersize=2, linewidth=1, color='steelblue')
    plt.title(f"Tren Temporal Harian - {kolom_target}")
    plt.xlabel("tanggal")
    plt.ylabel(kolom_target)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
