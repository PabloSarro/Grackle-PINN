import torch
import numpy as np
from data_utils import DEBUG_helper

SEC_PER_YEAR = 86400*365.256363004 # Seconds in a year

class PhysicsLossManager:
    """Manages Physics Loss Computation."""
    def __init__(self, model, in_mean, in_std, tg_mean, tg_std):
        self.model = model
        
        self.in_mean = in_mean
        self.in_std = in_std

        self.tg_mean = tg_mean
        self.tg_std = tg_std

    def mass_conserv(self, batch_x, preds, print_samples=False):
        """
        Computes the residuals for all species.
            
            batch_x: [log(y(t))], for y = T, nHI, nHII
            preds: [log(y(t+dt)/y(t))], for y = T, nHI, nHII
        """
        DEBUG_helper("mass_conserv: Input batch_x", batch_x)
        DEBUG_helper("mass_conserv: Input preds", preds)

        # =============================================================================
        # ================== INPUT: [log(y(t))] for y = T, nHI, nHII ==================
        # ============= OUTPUT: [log(y(t+dt)/T(t))], for y = T, nHI, nHII =============
        # =============================================================================


        # ==============================================================================
        # ========================= 1. EXTRACT INPUTS & OUTPUTS ========================
        # ==============================================================================

        # Denormalise inputs and outputs for physics loss computations.
        input_destand = batch_x*(self.in_std + 1e-8) + self.in_mean
        output_destand = preds*(self.tg_std + 1e-8) + self.tg_mean
        
        # Linearise (remove log-space)
        log_nHI = input_destand[:, 1:2]
        log_nHII = input_destand[:, 2:3]

        log_nHI_ratio = output_destand[:, 1:2]
        log_nHII_ratio = output_destand[:, 2:3]
        
        
        # ========================================================================
        # ========================= 2. MASS CONSERVATION =========================
        # ========================================================================
        # ========================== [n_HI + n_HII = C] ==========================
        # ========================================================================

        # nH_initial = nHI + nHII                                           # ---> we want it in LOG-SPACE!
        # log(nH_initial) = log( exp(log_nHI) + exp(log_nHII) )             # Bound to gradient explosion...
        log_nH_initial = torch.logsumexp(torch.cat([log_nHI, log_nHII], dim=1), dim=1, keepdim=True) # Not anymore!
        
        # nH_next = nHI_next + nHII_next                                    # ---> same problem --> LOG-SPACE!
        log_nHI_next_pred = log_nHI + log_nHI_ratio
        log_nHII_next_pred = log_nHII + log_nHII_ratio
        # log(nH_next) = log( exp(log_nHI_next) + exp(log_nHII_next) )      # We use the same trick
        log_nH_next_pred = torch.logsumexp(torch.cat([log_nHI_next_pred, log_nHII_next_pred], dim=1), dim=1, keepdim=True)
        DEBUG_helper("log_nH_initial", log_nH_initial)
        DEBUG_helper("log_nH_next_pred", log_nH_next_pred)


        # ======================================================================
        # =============== 3. PHYSICAL LOSS COMPUTATION (LOG-MSE) ===============
        # ======================================================================
        # ================= (ALTERNATIVE: USE NORMALISED LOSS) =================
        # ======================================================================
        
        log_mass_mse = (log_nH_next_pred - log_nH_initial)**2
        DEBUG_helper("Physics Loss (log-MSE): nHI, nHII", log_mass_mse)
        
        # --- DIAGNOSTIC PRINT BLOCK ---
        if print_samples:
            batch_size = 20
            print(f"\n--- Batch Physics Comparison ({batch_size} samples) ---")
            for i in range(batch_size):
                print(f"  Mass : Initial nH   = {log_nH_initial[i, 0].item():12.4e} | Next nH = {log_nH_next_pred[i, 0].item():12.4e} | Log-Mass-MSE = {log_mass_mse[i, 0].item():12.4e}")
                print("-" * 55)
            print(f"  Mean-Log-Mass-MSE = {torch.mean(log_mass_mse):12.4e}")

        return torch.mean(log_mass_mse)