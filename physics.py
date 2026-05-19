import torch
import numpy as np
from data_utils import DEBUG_helper

SEC_PER_YEAR = 86400*365.256363004 # Seconds in a year

class GrackleRates:
    """Analytical fits for Grackle primordial chemistry and cooling."""
    def __init__(self, tiny=1e-30):
        self.tiny = tiny
        self.k_B = 1.380649e-16       # Boltzmann constant [erg/K]

    def compute_chemical_rates(self, T):
        """
        Calculates all 8 chemical reaction rates based on temperature T (in Kelvin).
        For now, some of them are commented out, since Helium is assumed to be 0.
        """
        T = torch.clamp(T, min=1.0) # Avoid unphysically low temperatures.
        T_ev = T / 11605.0          # Convert from K to eV
        logT_ev = torch.log(T_ev)

        # k1: HI + e -> HII + 2e (Collisional Ionization)
        # Can be found in line 35 of grackle/src/clib/rate_functions.c
        k1 = torch.exp(-32.71396786375 
                                + 13.53655609057 * logT_ev 
                                - 5.739328757388 * torch.pow(logT_ev, 2)
                                + 1.563154982022 * torch.pow(logT_ev, 3) 
                                - 0.2877056004391 * torch.pow(logT_ev, 4)
                                + 0.03482559773736999 * torch.pow(logT_ev, 5) 
                                - 0.00263197617559 * torch.pow(logT_ev, 6)
                                + 0.0001119543953861 * torch.pow(logT_ev, 7) 
                                - 2.039149852002e-6 * torch.pow(logT_ev, 8))
        k1 = torch.where(T_ev > 0.8, k1, torch.clamp(k1, min=self.tiny))
        
        # k2: HII + e -> HI + photon (Radiative Recombination - Case B)
        # Can be found in line 97 of grackle/src/clib/rate_functions.c
        k2 = (4.881357e-6 * torch.pow(T, -1.5) * torch.pow((1.0 + 1.14813e2 * torch.pow(T, -0.407)), -2.242))
        k2 = torch.where(T < 1.0e9, k2, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE IT HAPPENS TO ADD NOISE TO THE PHYSICS LOSS! ----- \\
        # k7 (k57 in source): HI + HI -> HII + HI + e
        # Can be found in line 678 of grackle/src/clib/rate_functions.c
        k7 = 1.2e-17 * torch.pow(T, 1.2) * torch.exp(-157800.0 / T)
        k7 = torch.where(T > 3000.0, k7, torch.full_like(T, self.tiny))

        return k1, k2, k7

    
    def compute_cooling_rates(self, T, nHI, nHII, ne):
        """Calculates the 4 primary cooling processes in [erg s^-1 cm^-3]."""
        
        # // ----- 1. Collisional Excitation (Line Emission) - Cen (1992) fit ----- \\
        # Can be found in line 755 of grackle/src/clib/rate_functions.c
        ceHI_rate = 7.5e-19 * torch.exp(-118348.0 / T) / (1.0 + torch.sqrt(T/1.0e5))
        lambda_ce = ceHI_rate * nHI * ne

        # // ----- 2. Collisional Ionization (Energy lost per k1 event) ------ \\
        # Can be found in line 799 of grackle/src/clib/rate_functions.c
        T = torch.clamp(T, min=1.0)
        T_ev = T / 11605.0
        logT_ev = torch.log(T_ev)

        k1 = torch.exp(-32.71396786375 
                                + 13.53655609057 * logT_ev 
                                - 5.739328757388 * torch.pow(logT_ev, 2)
                                + 1.563154982022 * torch.pow(logT_ev, 3) 
                                - 0.2877056004391 * torch.pow(logT_ev, 4)
                                + 0.03482559773736999 * torch.pow(logT_ev, 5) 
                                - 0.00263197617559 * torch.pow(logT_ev, 6)
                                + 0.0001119543953861 * torch.pow(logT_ev, 7) 
                                - 2.039149852002e-6 * torch.pow(logT_ev, 8))
        k1 = torch.where(T_ev > 0.8, k1, torch.clamp(k1, min=self.tiny))
        ciHI_rate = 2.18e-11 * k1
        lambda_ci = ciHI_rate * nHI * ne

        # // ----- 3. Recombination Cooling (Case B) ----- \\
        # Can be found in line 832 of grackle/src/clib/rate_functions.c
        lambdaHI = 2.0 * 157807.0 / T
        reHII_rate = 3.435e-30 * T * torch.pow(lambdaHI, 1.970) / torch.pow(1.0 + torch.pow(lambdaHI/2.25, 0.376), 3.720)
        lambda_re = reHII_rate * nHII * ne

        # // ----- 4. Bremsstrahlung (Free-Free) ----- \\
        # Can be found in line 910 of grackle/src/clib/rate_functions.c
        brem_rate = 1.43e-27 * torch.sqrt(T) * (1.1 + 0.34 * torch.exp(-torch.pow(5.5 - torch.log10(T), 2) / 3.0))
        lambda_br = brem_rate * nHII * ne
        
        return lambda_ce + lambda_ci + lambda_re + lambda_br
    

class PhysicsLossManager:
    """Manages Automatic Differentiation and Residual computation."""
    def __init__(self, model, grackle_phys, in_mean, in_std, tg_mean, tg_std):
        self.model = model
        self.phys = grackle_phys
        
        self.in_mean = in_mean
        self.in_std = in_std

        self.tg_mean = tg_mean
        self.tg_std = tg_std

    def get_residuals(self, batch_x, preds, print_samples=False):
        """
        Computes the residuals for all species.
            
            batch_x: [log(y(t))], for y = T, nHI, nHII
            preds: [log(y(t+dt)/y(t))], for y = T, nHI, nHII
        """
        DEBUG_helper("get_residuals: Input batch_x", batch_x)
        DEBUG_helper("get_residuals: Input preds", preds)

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