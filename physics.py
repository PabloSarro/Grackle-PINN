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
        # =================== COMPUTATION OF PHYSICAL LOSS FUNCTIONS ===================
        # ==============================================================================



        # ==============================================================================
        # ========================= 1. EXTRACT INPUTS & OUTPUTS ========================
        # ==============================================================================

        # Denormalise inputs and outputs for physics loss computations.
        input_destand = batch_x*(self.in_std + 1e-8) + self.in_mean
        output_destand = preds*(self.tg_std + 1e-8) + self.tg_mean
        
        # Linearise (remove log-space)
        # dt = 100.0 * SEC_PER_YEAR
        # log_T = input_destand[:, 0:1]
        log_nHI = input_destand[:, 1:2]
        log_nHII = input_destand[:, 2:3]

        # log_T_ratio = output_destand[:, 0:1] # torch.clamp([...], min=-10.0, max=10.0)
        log_nHI_ratio = output_destand[:, 1:2]
        log_nHII_ratio = output_destand[:, 2:3]

        # T = torch.exp(log_T)
        # nHI = torch.exp(log_nHI)   # Instead of using this, for 2.1 we will use torch.logsumexp(...)
        # nHII = torch.exp(log_nHII) # Instead of using this, for 2.1 we will use torch.logsumexp(...)

        # Prevent wild predictions from causing overflow in Grackle analytical fits
        # T_next = T * torch.exp(log_T_ratio) # torch.clamp([...], min=1.0, max=1e9)
        # nHI_next = nHI * torch.exp(log_nHI_ratio) # torch.clamp([...], min=1e-16, max=1e5)
        # nHII_next = nHII * torch.exp(log_nHII_ratio) # torch.clamp([...], min=1e-16, max=1e5)

        # DEBUG_helper("T/dt:", T/dt)
        # DEBUG_helper("nHI/dt:", nHI/dt)
        # DEBUG_helper("nHII/dt:", nHII/dt)
        # DEBUG_helper("expm1:", torch.expm1(log_T_ratio))
        

        # ========================================================================
        # ========================== NO LONGER NEEDED ??? ========================
        # ========================================================================

        # dT_dt = (T/dt)*torch.expm1(log_T_ratio)           # = (T_next-T)/dt ~ dT/dt                  # DEBUG: Not used???
        # dnHI_dt = (nHI/dt)*torch.expm1(log_nHI_ratio)     # = (nHI_next-nHI)/dt ~ dnHI/dt
        # dnHII_dt = (nHII/dt)*torch.expm1(log_nHII_ratio)  # = (nHII_next-nHII)/dt ~ dnHII/dt
        # DEBUG_helper("dT/dt", dT_dt)
        # DEBUG_helper("dnHI/dt", dnHI_dt)
        # DEBUG_helper("dnHII/dt", dnHII_dt)


        
        
        # ========================================================================
        # ========================= 2.1 MASS CONSERVATION ========================
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


        # ========================================================================
        # ====================== 2.1 CHEMICAL RATE EQUATIONS =====================
        # ========================================================================
        # ================= (STIFFNESS / NUMERICAL CANCELLATION) =================
        # ========================================================================

        # dn/dt = gain(n) - loss(n)

        # Goal: Compute gain(n), loss(n), where:
        #   gain(n) / loss(n): k_{ij}*n_i*n_j, where
        #       k_{ij}: rate at which species i,j can chemically react (dep. on T).
        #       n_i, n_j: species density for the species i,j, respectively.

        # # 2.1.1. Compute the chemical rates.
        # k1, k2, _ = self.phys.compute_chemical_rates(T_next) # Again, implicit formulation
        # DEBUG_helper("k1_next", k1)
        # DEBUG_helper("k2_next", k2)
        # # k7 = rates['k7']

        # # 2.1.2. Compute the gain and loss terms (both with an implicit formulation).
        # ionisation = k1*nHI_next*ne_next
        # recombination = k2*nHII_next*ne_next
        # DEBUG_helper("ionisation", ionisation)
        # DEBUG_helper("recombination", recombination)

        # # 2.1.3. Compute dnHI/dt, dnHII/dt
        # nHI_react = recombination - ionisation # k2*nHII*ne - k1*nHI*ne
        # nHII_react = ionisation - recombination # k1*nHI*ne - k2*nHII*ne
        # DEBUG_helper("nHI_react", nHI_react)
        # DEBUG_helper("nHII_react", nHII_react)


        # ========================================================================
        # ===================== 2.2 LAGRANGIAN ENERGY EQUATION ===================
        # ========================================================================
        # =========================== [du/dt = -e_cool] ==========================
        # ========================================================================

        # Goal: Isolate dT/dt out of the Lagrangian Energy Equation:
        #       du/dt = -e_cool + e_heat
        
        # We neglect heating, and assume that cooling is dominated by hydrogen line emission:
        #       du/dt = -e_cool

        # But at the same time, we have
        #       u = 3/2·n_{tot}·k_B·T
        #       du/dt = (u_{next}-u_{curr}) / dt, where
        #           u_{next} = 3/2·n_{tot, next}·k_B·T_{next}
        #           u_{curr} = 3/2·n_{tot, curr}·k_B·T_{curr}
        #           u_{next}-u_{curr} = 3/2·k_B(n_{tot, next}·T_{next} - n_{tot, curr}-T_{curr})

        # Hence, our physics equation becomes:
        #       -e_cool = (u_{next}-u_{curr}) / dt

        # 2.2.1. Compute LHS.
        # ne = nHII # + nHeII + 2 * nHeIII, but these last were assumed to be 0 --> cooling box model
        # ne_next = nHII_next
        # e_cool = self.phys.compute_cooling_rates(T_next, nHI_next, nHII_next, ne_next) # Implicit formulation

        # # 2.2.2. Compute RHS.
        # n_tot = nHI + nHII + ne
        # n_tot_next = nHI_next + nHII_next + ne_next
        
        # du_dt = (3/2*self.phys.k_B)/dt * (n_tot_next*T_next - n_tot*T)
        
        # DEBUG_helper("T", T, min_val=0.0, max_val=1e15)
        # DEBUG_helper("e_cool", e_cool)
        # DEBUG_helper("n_tot", n_tot)
        # DEBUG_helper("du/dt", du_dt)



        # ======================================================================
        # =============== 3. PHYSICAL LOSS COMPUTATION (LOG-MSE) ===============
        # ======================================================================
        # ================= (ALTERNATIVE: USE NORMALISED LOSS) =================
        # ======================================================================
        
        # def log_signed_mse(lhs, rhs, weight_sign, eps):
        #     log_abs_lhs = torch.log10(torch.abs(lhs) + eps)
        #     log_abs_rhs = torch.log10(torch.abs(rhs) + eps)
        #     magn_loss = (log_abs_lhs - log_abs_rhs)**2
                
        #     # 2. Sign Penalty (0 if signs match, 'weight_sign' if opposite)
        #     sign_mismatch = (torch.sign(lhs) != torch.sign(rhs)).float()
        #     return magn_loss + weight_sign*sign_mismatch
        

        log_mass_mse = (log_nH_next_pred - log_nH_initial)**2                                   # 3.1. Species (nHI, nHII)
        # log_cool_mse = log_signed_mse(lhs=du_dt, rhs=-e_cool, weight_sign = 10.0, eps = 1e-35)  # 3.2. Temperature
        DEBUG_helper("Physics Loss (log-MSE): nHI, nHII", log_mass_mse)
        # DEBUG_helper("Physics Loss (log-MSE): T", log_cool_mse)
        

        # --- DIAGNOSTIC PRINT BLOCK ---
        if print_samples:
            batch_size = 20
            print(f"\n--- Batch Physics Comparison ({batch_size} samples) ---")
            for i in range(batch_size):
                # print(f"Sample {i+1} (T: {T[i,0].item():.2e}):")
                print(f"  Mass : Initial nH   = {log_nH_initial[i, 0].item():12.4e} | Next nH = {log_nH_next_pred[i, 0].item():12.4e} | Log-Mass-MSE = {log_mass_mse[i, 0].item():12.4e}")
                # print(f"  Temp :  -e_cool     = {-e_cool[i, 0].item():12.4e} |  du_dt  = {du_dt[i, 0].item():12.4e} | Log-Cool-MSE = {log_cool_mse[i, 0].item():12.4e}")
                print("-" * 55)
            print(f"  Mean-Log-Mass-MSE = {torch.mean(log_mass_mse):12.4e}")
            # print(f"  Mean-Log-Cool-MSE = {torch.mean(log_cool_mse):12.4e}")

        # physics_loss = log_mass_mse + log_cool_mse
        return torch.mean(log_mass_mse)#, torch.mean(log_cool_mse)