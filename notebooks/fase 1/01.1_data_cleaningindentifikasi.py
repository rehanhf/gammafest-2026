import pandas as pd
import re

import sys
sys.stdout.reconfigure(encoding='utf-8')

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
        print("sampel:", non_ascii[:10])
    print("\n")

def bandingkan_entitas(train_df, test_df):
    """
    menganalisis set difference antara fase train dan test.
    """
    tim_train = set(train_df['team'].dropna().unique()).union(set(train_df['opponent'].dropna().unique()))
    tim_test = set(test_df['team'].dropna().unique()).union(set(test_df['opponent'].dropna().unique()))
    
    #3. set difference kalkulasi
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
print("--- duplikasi baris ---")
print(f"duplikat di train: {dup_train}")
print(f"duplikat di test: {dup_test}")