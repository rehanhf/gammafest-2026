# =============================================================================
# notebooks/fase_4/04a_feature_alignment.py
# STEP 4A — Feature Alignment
# Tujuan: Bersihkan train_engineered.csv → train_aligned.csv
#         Selaraskan nama kolom, drop duplikat, fillna kritis
# Output: data/processed/train_aligned.csv
# Estimasi waktu: ~30 menit
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np

from src.feature_names import (
    TRAIN_COLS_DROP,
    TRAIN_COLS_RENAME,
    FINAL_FEATURES,
)

# ---------------------------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------------------------
INPUT_PATH  = 'data/processed/train_engineered.csv'
OUTPUT_PATH = 'data/processed/train_aligned.csv'

print("=" * 60)
print("STEP 4A — FEATURE ALIGNMENT")
print("=" * 60)

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print(f"\n[1/8] Loading {INPUT_PATH} ...")
train = pd.read_csv(INPUT_PATH)
print(f"      Shape awal: {train.shape[0]:,} baris x {train.shape[1]} kolom")

# Cek kolom duplikat .1
dupes = [c for c in train.columns if '.1' in c]
print(f"      Kolom .1 ditemukan: {dupes}")

# ---------------------------------------------------------------------------
# DROP KOLOM DUPLIKAT (versi lama, sebelum ELO update)
# ---------------------------------------------------------------------------
print(f"\n[2/8] Drop kolom duplikat lama ...")
before = train.shape[1]
train = train.drop(columns=TRAIN_COLS_DROP, errors='ignore')
dropped = [c for c in TRAIN_COLS_DROP if c in pd.read_csv(INPUT_PATH, nrows=0).columns]
print(f"      Dropped: {dropped}")
print(f"      Kolom: {before} → {train.shape[1]}")

# ---------------------------------------------------------------------------
# RENAME KOLOM .1 KE NAMA BERSIH
# ---------------------------------------------------------------------------
print(f"\n[3/8] Rename kolom .1 ke nama bersih ...")
renamed = {k: v for k, v in TRAIN_COLS_RENAME.items() if k in train.columns}
train = train.rename(columns=renamed)
for old, new in renamed.items():
    print(f"      {old:40s} → {new}")

# Pastikan tidak ada .1 tersisa
remaining_dupes = [c for c in train.columns if '.1' in c]
if remaining_dupes:
    print(f"      WARNING: Masih ada kolom .1: {remaining_dupes}")
else:
    print(f"      OK: Tidak ada kolom .1 tersisa")

# ---------------------------------------------------------------------------
# FILLNA H2H (null = belum pernah ketemu = riwayat nol)
# ---------------------------------------------------------------------------
print(f"\n[4/8] fillna H2H features ...")
h2h_cols = ['h2h_gd_last5', 'h2h_points_last5']
for col in h2h_cols:
    if col in train.columns:
        null_before = train[col].isna().sum()
        train[col] = train[col].fillna(0)
        print(f"      {col}: {null_before:,} null → filled dengan 0")
    else:
        print(f"      WARNING: {col} tidak ditemukan di dataframe!")

# ---------------------------------------------------------------------------
# BUAT FLAG has_h2h_history
# ---------------------------------------------------------------------------
print(f"\n[5/8] Buat flag has_h2h_history ...")
if 'h2h_gd_last5' in train.columns and 'h2h_points_last5' in train.columns:
    train['has_h2h_history'] = (
        (train['h2h_gd_last5'] != 0) | (train['h2h_points_last5'] != 0)
    ).astype(int)
    pct = train['has_h2h_history'].mean() * 100
    print(f"      has_h2h_history: {pct:.1f}% match punya riwayat H2H")

# ---------------------------------------------------------------------------
# FILLNA RANK + FLAG rank_available
# ---------------------------------------------------------------------------
print(f"\n[6/8] fillna rank features ...")
rank_cols_map = {
    'rank_team':     'rank_available_team',
    'rank_opponent': 'rank_available_opp',
}
for col, flag_col in rank_cols_map.items():
    if col in train.columns:
        null_before = train[col].isna().sum()
        pct_null    = null_before / len(train) * 100
        train[flag_col] = train[col].notna().astype(int)
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        print(f"      {col}: {null_before:,} null ({pct_null:.1f}%) → fillna({median_val:.1f})")
        print(f"      {flag_col}: dibuat (1=ada, 0=tidak ada)")
    else:
        print(f"      INFO: {col} tidak ada di dataframe, skip")

# ---------------------------------------------------------------------------
# VALIDASI FINAL_FEATURES TERSEDIA SEMUA
# ---------------------------------------------------------------------------
print(f"\n[7/8] Validasi FINAL_FEATURES tersedia ...")
missing_feats = [f for f in FINAL_FEATURES if f not in train.columns]
if missing_feats:
    print(f"      ERROR: Fitur berikut TIDAK ADA di train:")
    for f in missing_feats:
        print(f"        - {f}")
    raise ValueError(f"Ada {len(missing_feats)} fitur yang hilang! Cek ulang Fase 3.")
else:
    print(f"      OK: Semua {len(FINAL_FEATURES)} FINAL_FEATURES tersedia")

# Cek null di FINAL_FEATURES
print(f"\n      Null check pada FINAL_FEATURES:")
for f in FINAL_FEATURES:
    n = train[f].isna().sum()
    status = "OK" if n == 0 else f"WARNING: {n:,} null"
    print(f"      {f:45s} {status}")

# ---------------------------------------------------------------------------
# SIMPAN OUTPUT
# ---------------------------------------------------------------------------
print(f"\n[8/8] Simpan ke {OUTPUT_PATH} ...")
train.to_csv(OUTPUT_PATH, index=False)
print(f"      Shape akhir: {train.shape[0]:,} baris x {train.shape[1]} kolom")
print(f"      Saved: {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# RINGKASAN
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"STEP 4A SELESAI")
print(f"{'=' * 60}")
print(f"  Input : {INPUT_PATH}  ({pd.read_csv(INPUT_PATH, nrows=0).shape[1]} kolom)")
print(f"  Output: {OUTPUT_PATH} ({train.shape[1]} kolom)")
print(f"  Baris : {train.shape[0]:,}")
print(f"  FINAL_FEATURES siap: {len(FINAL_FEATURES)} fitur")
print(f"  Kolom .1 tersisa   : {len([c for c in train.columns if '.1' in c])}")
print(f"\n  → Lanjut ke STEP 4B: 04b_sanity_check_baseline.py")
