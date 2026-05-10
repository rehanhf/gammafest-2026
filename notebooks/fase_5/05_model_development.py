import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pickle
import sys
import os
from tqdm import tqdm

# Pastikan path menunjuk ke root direktori
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from src.metrics import kalkulasi_aw_mae
    from src.optimizer import optimizer
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
    from src.metrics import kalkulasi_aw_mae
    from src.optimizer import optimizer

os.makedirs('./model', exist_ok=True)

# 1. Load Data Fase 3
train_df = pd.read_csv('./data/processed/train_engineered.csv')
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.sort_values('date').reset_index(drop=True)

# 2. Ekstraksi Target dan Fitur Numerik Absolut
y_team = train_df['team_goals'].values.astype(float)
y_opp = train_df['opp_goals'].values.astype(float)
weights = train_df['tournament_weight'].values

drop_cols =['Id', 'match_id', 'date', 'team', 'opponent', 'tournament', 'team_goals', 'opp_goals']
# Filter otomatis: abaikan kolom identifier dan pastikan hanya mengambil tipe numerik
fitur_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
fitur_df = fitur_df.select_dtypes(include=[np.number])

features = fitur_df.columns.tolist()
X = fitur_df.values

# 3. Setup TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

oof_team_pred = np.zeros(len(train_df))
oof_opp_pred = np.zeros(len(train_df))
oof_team_lambda = np.zeros(len(train_df))
oof_opp_lambda = np.zeros(len(train_df))

cv_scores =[]

print("FASE 5: TRAINING LIGHTGBM POISSON & EXPECTED LOSS OPTIMIZATION")
print(f"Jumlah fitur: {len(features)}")

# 4. Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    yt_train, yt_val = y_team[train_idx], y_team[val_idx]
    yo_train, yo_val = y_opp[train_idx], y_opp[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]
    
    params = {
        'objective': 'poisson',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_estimators': 1000
    }
    
    # Train Model Team
    model_t = lgb.LGBMRegressor(**params)
    model_t.fit(
        X_train, yt_train,
        sample_weight=w_train,
        eval_set=[(X_val, yt_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Train Model Opponent
    model_o = lgb.LGBMRegressor(**params)
    model_o.fit(
        X_train, yo_train,
        sample_weight=w_train,
        eval_set=[(X_val, yo_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Prediksi Lambdas (Dengan explicit cast untuk Pylance/NumPy compliance)
    lam_t = np.clip(np.array(model_t.predict(X_val)), 0.001, 15.0)
    lam_o = np.clip(np.array(model_o.predict(X_val)), 0.001, 15.0)
    
    oof_team_lambda[val_idx] = lam_t
    oof_opp_lambda[val_idx] = lam_o
    
    pred_t = np.zeros(len(val_idx))
    pred_o = np.zeros(len(val_idx))
    
    # Optimasi Prediksi Integer menggunakan Expected Loss AW-MAE
    for i in tqdm(range(len(val_idx)), desc=f"Fold {fold+1} Optimization"):
        opt_t, opt_o = optimizer.optimize_prediction(lam_t[i], lam_o[i], rho=0.0)
        pred_t[i] = opt_t
        pred_o[i] = opt_o
        
    oof_team_pred[val_idx] = pred_t
    oof_opp_pred[val_idx] = pred_o
    
    # Perhitungan AW-MAE
    y_true_val = np.column_stack((np.array(yt_val), np.array(yo_val)))
    y_pred_val = np.column_stack((np.array(pred_t), np.array(pred_o)))
    fold_score = kalkulasi_aw_mae(y_true_val, y_pred_val, w_val)
    
    print(f"Fold {fold+1} AW-MAE Score: {fold_score:.4f}")
    cv_scores.append(fold_score)
    
    # Ekspor Model
    with open(f'./model/lgbm_team_fold{fold}.pkl', 'wb') as f:
        pickle.dump(model_t, f)
    with open(f'./model/lgbm_opp_fold{fold}.pkl', 'wb') as f:
        pickle.dump(model_o, f)

# 5. Evaluasi OOF
cv_mean = np.mean(cv_scores)
print("\n" + "="*50)
print(f"Mean CV AW-MAE Score: {cv_mean:.4f}")
if cv_mean < 4.6375:
    print(f"STATUS: BERHASIL MENGALAHKAN BASELINE (4.6375). Improvement: {4.6375 - cv_mean:.4f}")
else:
    print(f"STATUS: GAGAL MENGALAHKAN BASELINE. Periksa overfitting.")

# 6. Penyimpanan Prediksi OOF
oof_df = train_df[['Id', 'date', 'team', 'opponent', 'team_goals', 'opp_goals', 'tournament_weight']].copy()
oof_df['lambda_team'] = oof_team_lambda
oof_df['lambda_opp'] = oof_opp_lambda
oof_df['pred_team_goals'] = oof_team_pred
oof_df['pred_opp_goals'] = oof_opp_pred

# Filter hanya fold validasi
oof_df = oof_df[oof_team_lambda > 0]
oof_df.to_csv('./data/processed/oof_predictions_lgbm.csv', index=False)