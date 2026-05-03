import pandas as pd
import numpy as np
import re

#1. muat data yang sudah dibersihkan
train_clean = pd.read_csv('./data/processed/train_cleaned.csv')
test_clean = pd.read_csv('./data/processed/test_cleaned.csv')

def verifikasi_fase_1(train_df, test_df):
    """
    menjalankan asersi strict untuk memastikan data cleaning 100% tuntas.
    jika script ini error, fase 1 belum selesai.
    """
    print("VERIFIKASI FASE 1...\n")
    
    #2. cek sisa nan pada kolom esensial
    kolom_wajib =[
        'altitude_venue', 'temperature_venue', 
        'distance_travel_team', 'distance_travel_opp',
        'gdp_per_capita_team', 'gdp_per_capita_opp',
        'population_team', 'population_opp'
    ]
    
    sisa_nan_train = train_df[kolom_wajib].isna().sum().sum()
    sisa_nan_test = test_df[kolom_wajib].isna().sum().sum()
    
    assert sisa_nan_train == 0, f"Gagal: masih ada {sisa_nan_train} NaN di train set."
    assert sisa_nan_test == 0, f"Gagal: masih ada {sisa_nan_test} NaN di test set."
    print("[PASS] 0 NaN tersisa di kolom imputasi numerik.")
    
    #3. cek anomali encoding teks (pastikan ascii clean)
    semua_tim = pd.concat([train_df['team'], train_df['opponent'], test_df['team'], test_df['opponent']]).unique()
    non_ascii =[tim for tim in semua_tim if pd.notna(tim) and not re.match(r'^[\x00-\x7F]+$', tim)]
    
    assert len(non_ascii) == 0, f"Gagal: ditemukan {len(non_ascii)} entitas non-ASCII: {non_ascii}"
    print("[PASS] Encoding nama tim 100% clean.")
    
    #4. deteksi outlier ekstrem (perlu dicatat untuk pemodelan)
    max_gol_train = train_df[['team_goals', 'opp_goals']].max().max()
    print(f"[INFO] Outlier gol maksimal di train: {max_gol_train}")
    if max_gol_train > 10:
        print("-> Peringatan: Outlier > 10 gol terdeteksi. Pertimbangkan clipping target di fase 5.")
        
    print("\nSTATUS: FASE 1 SELESAI DAN TERKUNCI.")
    return True

#5. eksekusi verifikasi
verifikasi_fase_1(train_clean, test_clean)