import pandas as pd
import numpy as np
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

#1. muat data yang sudah dibersihkan
train_clean = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str})  # ← paksa date dibaca sebagai string
test_clean  = pd.read_csv('./data/processed/test_cleaned.csv',  dtype={'date': str})

def standarisasi_tanggal(df, kolom_tanggal='date'):
    """
    menstandarisasi format tanggal campuran ke format ISO YYYY-MM-DD (string).
    menangani dua format:
      - YYYY-MM-DD  → sudah benar, tidak diubah       (contoh: 1899-03-18)
      - DD/MM/YYYY  → dikonversi ke YYYY-MM-DD         (contoh: 03/02/1900)
    """
    if kolom_tanggal not in df.columns:
        print(f"[WARNING] kolom '{kolom_tanggal}' tidak ditemukan, dilewati.")
        return df

    def konversi_satu_tanggal(val):
        if pd.isna(val) or str(val).strip() == '':
            return np.nan
        val = str(val).strip()

        if re.match(r'^\d{4}-\d{2}-\d{2}$', val):   # sudah YYYY-MM-DD
            return val
        if re.match(r'^\d{2}/\d{2}/\d{4}$', val):   # DD/MM/YYYY
            dd, mm, yyyy = val.split('/')
            return f"{yyyy}-{mm}-{dd}"

        print(f"[WARNING] format tanggal tidak dikenali: '{val}'")
        return val

    df[kolom_tanggal] = df[kolom_tanggal].apply(konversi_satu_tanggal)
    return df

def verifikasi_fase_1(train_df, test_df, kolom_tanggal='date'):
    """
    menjalankan asersi strict untuk memastikan data cleaning 100% tuntas.
    jika script ini error, fase 1 belum selesai.
    """
    print("VERIFIKASI FASE 1...\n")
    
    #2. cek sisa nan pada kolom esensial
    kolom_wajib = [
        'altitude_venue', 'temperature_venue', 
        'distance_travel_team', 'distance_travel_opp',
        'gdp_per_capita_team', 'gdp_per_capita_opp',
        'population_team', 'population_opp'
    ]
    
    sisa_nan_train = train_df[kolom_wajib].isna().sum().sum()
    sisa_nan_test  = test_df[kolom_wajib].isna().sum().sum()
    
    assert sisa_nan_train == 0, f"Gagal: masih ada {sisa_nan_train} NaN di train set."
    assert sisa_nan_test  == 0, f"Gagal: masih ada {sisa_nan_test} NaN di test set."
    print("[PASS] 0 NaN tersisa di kolom imputasi numerik.")
    
    #3. cek anomali encoding teks (pastikan ascii clean)
    semua_tim = pd.concat([
        train_df['team'], train_df['opponent'],
        test_df['team'],  test_df['opponent']
    ]).unique()
    non_ascii = [tim for tim in semua_tim if pd.notna(tim) and not re.match(r'^[\x00-\x7F]+$', tim)]
    
    assert len(non_ascii) == 0, f"Gagal: ditemukan {len(non_ascii)} entitas non-ASCII: {non_ascii}"
    print("[PASS] Encoding nama tim 100% clean.")

    # ════════════════════════════════════════════════════
    #4. ← BARU: verifikasi format tanggal seragam YYYY-MM-DD
    # ════════════════════════════════════════════════════
    if kolom_tanggal in train_df.columns:
        pola_iso = r'^\d{4}-\d{2}-\d{2}$'

        tidak_valid_train = train_df[kolom_tanggal].dropna()
        tidak_valid_train = tidak_valid_train[~tidak_valid_train.astype(str).str.match(pola_iso)]

        tidak_valid_test  = test_df[kolom_tanggal].dropna()
        tidak_valid_test  = tidak_valid_test[~tidak_valid_test.astype(str).str.match(pola_iso)]

        if len(tidak_valid_train) > 0 or len(tidak_valid_test) > 0:
            print(f"[WARNING] Ditemukan {len(tidak_valid_train)} tanggal tidak valid di train:")
            print(tidak_valid_train.value_counts().head(10).to_string())
            print(f"[WARNING] Ditemukan {len(tidak_valid_test)} tanggal tidak valid di test:")
            print(tidak_valid_test.value_counts().head(10).to_string())

            # auto-fix: jalankan ulang standarisasi
            print("\n[AUTO-FIX] Menjalankan ulang standarisasi_tanggal...")
            train_df = standarisasi_tanggal(train_df, kolom_tanggal)
            test_df  = standarisasi_tanggal(test_df,  kolom_tanggal)

            # simpan ulang hasil fix
            train_df.to_csv('./data/processed/train_cleaned.csv', index=False)
            test_df.to_csv('./data/processed/test_cleaned.csv',   index=False)
            print("[AUTO-FIX] File disimpan ulang ke processed/.")
        else:
            print(f"[PASS] Semua tanggal sudah dalam format YYYY-MM-DD.")
    else:
        print(f"[SKIP] Kolom '{kolom_tanggal}' tidak ditemukan, cek verifikasi tanggal dilewati.")

    #5. deteksi outlier ekstrem (perlu dicatat untuk pemodelan)
    max_gol_train = train_df[['team_goals', 'opp_goals']].max().max()
    print(f"\n[INFO] Outlier gol maksimal di train: {max_gol_train}")
    if max_gol_train > 10:
        print("-> Peringatan: Outlier > 10 gol terdeteksi. Pertimbangkan clipping target di fase 5.")
        
    print("\nSTATUS: FASE 1 SELESAI DAN TERKUNCI.")
    return train_df, test_df


#6. eksekusi verifikasi
train_clean, test_clean = verifikasi_fase_1(train_clean, test_clean)