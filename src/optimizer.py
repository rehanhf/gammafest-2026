import numpy as np
from scipy.stats import poisson
import sys
import os

#import fungsi base metric
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import kalkulasi_aw_mae

class AWMAE_Optimizer:
    def __init__(self, max_goals=15):
        self.max_goals = max_goals
        self.loss_tensor = self._precompute_loss_tensor()
        
    def _precompute_loss_tensor(self):
        """
        Precomputes the exact AW-MAE loss for all possible (pred_team, pred_opp, true_team, true_opp) combinations.
        Shape: (16, 16, 16, 16) ->[pred_t, pred_o, true_t, true_o]
        """
        print(f"Precomputing AW-MAE loss tensor up to {self.max_goals} goals...")
        dim = self.max_goals + 1
        tensor = np.zeros((dim, dim, dim, dim))
        
        #buat dummy weight array of size 1 karena kalkulasi_aw_mae butuh array
        w = np.ones(1)
        
        for pt in range(dim):
            for po in range(dim):
                y_pred = np.array([[pt, po]], dtype=float)
                for tt in range(dim):
                    for to in range(dim):
                        y_true = np.array([[tt, to]], dtype=float)
                        #Hitung exact loss tanpa iterasi ulang saat inference
                        tensor[pt, po, tt, to] = kalkulasi_aw_mae(y_true, y_pred, w)
                        
        return tensor

    def optimize_prediction(self, lambda_team, lambda_opp, rho=0.0):
        """
        ngubah output lambda Poisson dari model ML (LightGBM/XGBoost) menjadi prediksi 
        integer (team_goals, opp_goals) yang meminimalkan EXPECTED AW-MAE LOSS.
        
        rho: Parameter Dixon-Coles untuk meningkatkan probabilitas hasil seri (draw).
        """
        dim = self.max_goals + 1
        
        #1. generate Probability Mass Functions (PMF)
        pmf_t = poisson.pmf(np.arange(dim), lambda_team)
        pmf_o = poisson.pmf(np.arange(dim), lambda_opp)
        
        #2. buat Joint Probability Matrix P(team, opp)
        P = np.outer(pmf_t, pmf_o)
        
        #3. Dixon-Coles adjustment (inflasi Draw probability 0-0, 1-1, 0-1, 1-0)
        if rho != 0.0:
            P[0,0] *= (1 - lambda_team*lambda_opp*rho)
            P[0,1] *= (1 + lambda_team*rho)
            P[1,0] *= (1 + lambda_opp*rho)
            P[1,1] *= (1 - rho)
            P = np.clip(P, 0, 1)
            P /= np.sum(P) #normalize ulang
            
        #4. itung Expected Loss untuk setiap kemungkinan prediksi (u, v)
        #tensor dot product: E_loss[u, v] = Sum_{i, j} (Loss[u, v, i, j] * P[i, j])
        expected_loss_matrix = np.tensordot(self.loss_tensor, P, axes=([2, 3], [0, 1]))
        
        #5. cari index (prediksi) dengan Expected Loss terkecil
        optimal_pt, optimal_po = np.unravel_index(np.argmin(expected_loss_matrix), expected_loss_matrix.shape)
        
        return optimal_pt, optimal_po

#inisialisasi global optimizer agar precompute hanya berjalan sekali saat di-import
optimizer = AWMAE_Optimizer(max_goals=15)