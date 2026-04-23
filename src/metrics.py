import numpy as np

def kalkulasi_aw_mae(y_true, y_pred, weights=None):
    """
    menghitung mean absolute error yang diperbarui dengan bobot (AW-MAE).
    
    y_true: numpy array dengan bentuk (n, 2) -> [team_goals, opp_goals]
    y_pred: numpy array dengan bentuk (n, 2) -> [team_goals, opp_goals]
    weights: numpy array dengan bentuk (n,) -> match weights
    
    kode ini menghitung MAE dengan mempertimbangkan beberapa komponen tambahan seperti:
    - Kesalahan dasar (MAE)
    - Komponen Denda
    - Multiplier Outcome
    - Skala Non-Linear
    - Bobot Turnamen
    
    fungsi ini mengembalikan nilai AW-MAE yang dihitung.
    """
    assert y_true.ndim == 2 and y_true.shape[1] == 2, "y_true harus memiliki bentuk (n, 2)"
    assert y_pred.ndim == 2 and y_pred.shape[1] == 2, "y_pred harus memiliki bentuk (n, 2)"
    
    if weights is None:
        weights = np.ones(len(y_true))
    
    team_true, opp_true = y_true[:, 0], y_true[:, 1]
    team_pred, opp_pred = y_pred[:, 0], y_pred[:, 1]
    
    #1. base error (MAE)
    mae = (np.abs(team_true - team_pred) + np.abs(opp_true - opp_pred)) / 2.0
    
    #2. komponen penalty 
    exact = ((team_true == team_pred) & (opp_true == opp_pred)).astype(int)
    
     #catatan: np.sign(0) akan menghasilkan 0. Prediksi yang benar (0 == 0 -> hasil 1)
    sign_true = np.sign(team_true - opp_true)
    sign_pred = np.sign(team_pred - opp_pred)
    outcome = (sign_true == sign_pred).astype(int)
    
    gd_true = team_true - opp_true
    gd_pred = team_pred - opp_pred
    gd = (gd_true == gd_pred).astype(int)
    
    #menghitung penalty berdasarkan kondisi
    penalty = 0.30 * (1 - exact) + 0.25 * (1 - outcome) + 0.15 * (1 - gd)
    
    #3. Multiplier Outcome 
    #memberikan penalti dua kali untuk hasil yang salah sesuai dengan aturan kompetisi
    multiplier = np.where(outcome == 1, 1.0, 1.5)
    
    #4. Non-Linear Scaling
    raw_loss = mae + penalty
    loss = (raw_loss * multiplier) ** 1.5
    
    #5. Tournament Weighting
    aw_mae = np.sum(loss * weights) / np.sum(weights)
    
    return aw_mae