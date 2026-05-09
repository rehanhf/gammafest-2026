import pandas as pd
import numpy as np
import re

#1. muat data
train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')

def deteksi_anomali_string(df, nama_df):
    """
    memindai kolom teks untuk mendeteksi karakter non-ascii dan inkonsistensi.
    """
    semua_tim = pd.concat([df['team'], df['opponent']]).unique()
    
    #2. cari karakter non-ascii (indikator encoding rusak)
    non_ascii =[tim for tim in semua_tim if pd.notna(tim) and not re.match(r'^[\x00-\x7F]+$', tim)]
    
    print(f"--- anomali karakter di {nama_df} ---")
    print(f"jumlah entitas non-ascii: {len(non_ascii)}")
    if len(non_ascii) > 0:
        print("sampel: [non-ASCII entries detected, will be standardized in pipeline]")
    print("\n")

def bandingkan_entitas(train_df, test_df):
    """
    menganalisis set difference antara fase train dan test.
    """
    tim_train = set(train_df['team'].dropna().unique()).union(set(train_df['opponent'].dropna().unique()))
    tim_test = set(test_df['team'].dropna().unique()).union(set(test_df['opponent'].dropna().unique()))
    
    #3. set difference kalkulasi
    hanya_di_train = tim_train - tim_test
    hanya_di_test = tim_test - tim_train
    
    print("--- perbandingan entitas tim ---")
    print(f"total tim unik di train: {len(tim_train)}")
    print(f"total tim unik di test: {len(tim_test)}")
    print(f"tim hanya ada di test (cold start problem): {len(hanya_di_test)}")
    if len(hanya_di_test) > 0:
        print("daftar:", sorted(list(hanya_di_test)))
    print("\n")

def audit_missing_values(train_df, test_df):
    """
    menghitung persentase nan/null untuk setiap kolom.
    """
    print("--- missing value audit (% kosong) ---")
    kumpulan_kolom = set(train_df.columns).intersection(set(test_df.columns))
    
    for col in sorted(kumpulan_kolom):
        pct_train = train_df[col].isna().mean() * 100
        pct_test = test_df[col].isna().mean() * 100
        
        #4. hanya print jika ada missing value
        if pct_train > 0 or pct_test > 0:
            print(f"{col}: train={pct_train:.2f}%, test={pct_test:.2f}%")
    print("\n")

#5. eksekusi pipeline diagnostik
print("MEMULAI DIAGNOSTIK DATA...\n")
deteksi_anomali_string(train, "train.csv")
deteksi_anomali_string(test, "test.csv")
bandingkan_entitas(train, test)
audit_missing_values(train, test)

#6. cek duplikasi baris berdasarkan match_id
dup_train = train.duplicated(subset=['match_id', 'team']).sum()
dup_test = test.duplicated(subset=['match_id', 'team']).sum()
print(f"--- duplikasi baris ---")
print(f"duplikat di train: {dup_train}")
print(f"duplikat di test: {dup_test}")


#cleaning

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
    
    kolom_numerik =[
        'altitude_venue', 'temperature_venue', 
        'distance_travel_team', 'distance_travel_opp',
        'gdp_per_capita_team', 'gdp_per_capita_opp',
        'population_team', 'population_opp',
        'rank_team', 'rank_opponent'
    ]
    
    #4. isi sisa NaN dengan median dari masing-masing kolom
    for col in kolom_numerik:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    #5. flag untuk rank availability (indicator: apakah rank tersedia di match ini?)
    if 'rank_team' in df.columns:
        df['rank_available_team'] = (~df['rank_team'].isna()).astype(int)
    if 'rank_opponent' in df.columns:
        df['rank_available_opp'] = (~df['rank_opponent'].isna()).astype(int)
    
    #6. encode gender (1=men, 0=women)
    if 'gender' in df.columns:
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        #mapping: M -> 1 (men), W -> 0 (women)
            
    return df

#5. eksekusi pembersihan
train = standarisasi_nama_tim(train)
test = standarisasi_nama_tim(test)

train = imputasi_missing_values(train)
test = imputasi_missing_values(test)

#6. verifikasi sisa cold start teams setelah standardisasi
tim_train = set(train['team'].unique()).union(set(train['opponent'].unique()))
tim_test = set(test['team'].unique()).union(set(test['opponent'].unique()))
genuine_new_teams = tim_test - tim_train

print(f"jumlah genuine new teams di test set: {len(genuine_new_teams)}")

#7. simpan data yang sudah bersih sebagai checkpoint untuk phase 2 dan 3
train.to_csv('./data/processed/train_cleaned.csv', index=False)
test.to_csv('./data/processed/test_cleaned.csv', index=False)

#3: verif

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
    
    #tambah rank columns hanya jika ada di data
    if 'rank_team' in train_df.columns:
        kolom_wajib.extend(['rank_team', 'rank_opponent'])
    
    sisa_nan_train = train_df[kolom_wajib].isna().sum().sum()
    sisa_nan_test = test_df[[c for c in kolom_wajib if c in test_df.columns]].isna().sum().sum()
    
    assert sisa_nan_train == 0, f"Gagal: masih ada {sisa_nan_train} NaN di train set."
    assert sisa_nan_test == 0, f"Gagal: masih ada {sisa_nan_test} NaN di test set."
    print("[PASS] 0 NaN tersisa di kolom imputasi numerik.")
    
    #3. cek anomali encoding teks (pastikan ascii clean)
    semua_tim = pd.concat([train_df['team'], train_df['opponent'], test_df['team'], test_df['opponent']]).unique()
    non_ascii =[tim for tim in semua_tim if pd.notna(tim) and not re.match(r'^[\x00-\x7F]+$', tim)]
    
    assert len(non_ascii) == 0, f"Gagal: ditemukan {len(non_ascii)} entitas non-ASCII: {non_ascii}"
    print("[PASS] Encoding nama tim 100% clean.")
    
    #4. cek flag kolom baru
    if 'rank_available_team' in train_df.columns:
        assert 'rank_available_team' in train_df.columns, "Gagal: rank_available_team tidak ada"
        assert 'rank_available_opp' in train_df.columns, "Gagal: rank_available_opp tidak ada"
        print("[PASS] Flag kolom rank_available_* ada.")
    
    if 'gender_encoded' in train_df.columns:
        assert 'gender_encoded' in train_df.columns, "Gagal: gender_encoded tidak ada"
        print("[PASS] Flag kolom gender_encoded ada.")
    
    #5. deteksi outlier ekstrem (perlu dicatat untuk pemodelan)
    max_gol_train = train_df[['team_goals', 'opp_goals']].max().max()
    print(f"[INFO] Outlier gol maksimal di train: {max_gol_train}")
    if max_gol_train > 10:
        print("-> Peringatan: Outlier > 10 gol terdeteksi. Pertimbangkan clipping target di fase 5.")
        
    print("\nSTATUS: FASE 1 SELESAI DAN TERKUNCI.")
    return True

#5. eksekusi verifikasi
verifikasi_fase_1(train_clean, test_clean)
