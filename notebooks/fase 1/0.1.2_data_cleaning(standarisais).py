import pandas as pd
import numpy as np
import re

import sys
sys.stdout.reconfigure(encoding='utf-8')

#1. muat raw data
train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')

def standarisasi_tanggal(df, kolom_tanggal='date'):
    """
    menstandarisasi format tanggal campuran ke format ISO YYYY-MM-DD (string).
    menangani dua format:
      - YYYY-MM-DD  → sudah benar, tidak diubah       (contoh: 1899-03-18)
      - DD/MM/YYYY  → dikonversi ke YYYY-MM-DD         (contoh: 03/02/1900)
    semua disimpan sebagai string karena pandas tidak support tanggal < 1900.
    """
    if kolom_tanggal not in df.columns:
        print(f"[WARNING] kolom '{kolom_tanggal}' tidak ditemukan, dilewati.")
        return df

    def konversi_satu_tanggal(val):
        if pd.isna(val) or str(val).strip() == '':
            return np.nan
        
        val = str(val).strip()
        
        # format YYYY-MM-DD (sudah benar)
        if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
            return val
        
        # format DD/MM/YYYY → YYYY-MM-DD
        if re.match(r'^\d{2}/\d{2}/\d{4}$', val):
            dd, mm, yyyy = val.split('/')
            return f"{yyyy}-{mm}-{dd}"
        
        # format MM/DD/YYYY → YYYY-MM-DD (fallback jika ada)
        if re.match(r'^\d{2}/\d{2}/\d{4}$', val):
            mm, dd, yyyy = val.split('/')
            return f"{yyyy}-{mm}-{dd}"

        print(f"[WARNING] format tanggal tidak dikenali: '{val}'")
        return val

    sebelum = df[kolom_tanggal].isna().sum()
    df[kolom_tanggal] = df[kolom_tanggal].apply(konversi_satu_tanggal)
    sesudah = df[kolom_tanggal].isna().sum()

    print(f"[standarisasi_tanggal] kolom='{kolom_tanggal}' | "
          f"NaN sebelum={sebelum}, sesudah={sesudah}")
    return df

def standarisasi_nama_tim(df):
    """
    menghapus aksen non-ascii untuk mencegah entity fragmentation pada phase 3.
    """
    mapping_karakter = {
        'Curaçao': 'Curacao', 'Réunion': 'Reunion', 
        'São Tomé and Príncipe': 'Sao Tome and Principe',
        'Ynys Môn': 'Ynys Mon', 'Åland Islands': 'Aland Islands',
        'Frøya': 'Froya', 'Åland': 'Aland', 'Găgăuzia': 'Gagauzia',
        'Sápmi': 'Sapmi', 'Saint Barthélemy': 'Saint Barthelemy',
        'Székely Land': 'Szekely Land', 'Felvidék': 'Felvidek',
        'Délvidék': 'Delvidek', 'Kárpátalja': 'Karpatalja',
        'Ryūkyū': 'Ryukyu'
    }
    
    #2. terapkan mapping ke team dan opponent
    df['team'] = df['team'].replace(mapping_karakter)
    df['opponent'] = df['opponent'].replace(mapping_karakter)
    return df

def imputasi_missing_values(df):
    """
    mengisi missing features dengan global median dan contextual logic.
    tree-based models (XGBoost/LightGBM) robust terhadap median imputation.
    """
    #3. distance_travel bernilai 0 jika tim bermain di kandang
    df.loc[df['is_home'] == 1, 'distance_travel_team'] = df.loc[df['is_home'] == 1, 'distance_travel_team'].fillna(0)
    df.loc[(df['is_home'] == 0) & (df['neutral'] == 0), 'distance_travel_opp'] = df.loc[(df['is_home'] == 0) & (df['neutral'] == 0), 'distance_travel_opp'].fillna(0)
    
    kolom_numerik = [
        'altitude_venue', 'temperature_venue', 
        'distance_travel_team', 'distance_travel_opp',
        'gdp_per_capita_team', 'gdp_per_capita_opp',
        'population_team', 'population_opp'
    ]
    
    #4. isi sisa NaN dengan median dari masing-masing kolom
    for col in kolom_numerik:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    return df

#5. eksekusi pembersihan
train = standarisasi_tanggal(train, kolom_tanggal='date')   # ← ganti 'date' jika nama kolom berbeda
test  = standarisasi_tanggal(test,  kolom_tanggal='date')

train = standarisasi_nama_tim(train)
test  = standarisasi_nama_tim(test)

train = imputasi_missing_values(train)
test  = imputasi_missing_values(test)

#6. verifikasi sisa cold start teams setelah standardisasi
tim_train = set(train['team'].unique()).union(set(train['opponent'].unique()))
tim_test  = set(test['team'].unique()).union(set(test['opponent'].unique()))
genuine_new_teams = tim_test - tim_train

print(f"jumlah genuine new teams di test set: {len(genuine_new_teams)}")

#7. simpan data yang sudah bersih sebagai checkpoint untuk phase 2 dan 3
train.to_csv('./data/processed/train_cleaned.csv', index=False)
test.to_csv('./data/processed/test_cleaned.csv', index=False)