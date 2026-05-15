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

        return k1, k2

    
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
            
            batch_x: [dt, log(y(t))], for y = T, nHI, nHII
            preds: [log(y(t+dt)/y(t))], for y = T, nHI, nHII
        """
        DEBUG_helper("get_residuals: Input batch_x", batch_x)
        DEBUG_helper("get_residuals: Input preds", preds)

        # --------------------------------------------------------------------------- #
        # ------------ INPUT: [dt, log(T(t)), log(nHI(t)), log(nHII(t))] ------------ #
        # ------------ OUTPUT: [log(y(t+dt)/T(t))], for y = T, nHI, nHII ------------ #
        # --------------------------------------------------------------------------- #

        # // ----- COMPUTATION OF PHYSICAL LOSS FUNCTIONS ----- \\
        # The rate equations are in linear space (dy/dt = ...)
        # Our inputs and outputs are standardised and in log-space.
        # Hence we first linearise them.
        input_destand = batch_x*(self.in_std + 1e-8) + self.in_mean
        output_destand = preds*(self.tg_std + 1e-8) + self.tg_mean

        #       // ----- 1. LHS: dy/dt ----- \\
        # Linearise (remove log-space)
        dt = input_destand[:, 0:1] * SEC_PER_YEAR
        log_T_t = input_destand[:, 1:2]
        log_nHI_t = input_destand[:, 2:3]
        log_nHII_t = input_destand[:, 3:4]

        log_T_ratio = torch.clamp(output_destand[:, 0:1], min=-10.0, max=10.0)
        log_nHI_ratio = torch.clamp(output_destand[:, 1:2], min=-15.0, max=15.0)
        log_nHII_ratio = torch.clamp(output_destand[:, 2:3], min=-15.0, max=15.0)

        T = torch.exp(log_T_t)
        nHI = torch.exp(log_nHI_t)
        nHII = torch.exp(log_nHII_t)

        T_next = T * torch.exp(log_T_ratio)
        nHI_next = nHI * torch.exp(log_nHI_ratio)
        nHII_next = nHII * torch.exp(log_nHII_ratio)

        DEBUG_helper("T/dt:", T/dt)
        DEBUG_helper("nHI/dt:", nHI/dt)
        DEBUG_helper("nHII/dt:", nHII/dt)
        DEBUG_helper("expm1:", torch.expm1(log_T_ratio))
        
        dT_dt = (T/dt)*torch.expm1(log_T_ratio)           # = (T_next-T)/dt ~ dT/dt                  # DEBUG: Not used???
        dnHI_dt = (nHI/dt)*torch.expm1(log_nHI_ratio)     # = (nHI_next-nHI)/dt ~ dnHI/dt
        dnHII_dt = (nHII/dt)*torch.expm1(log_nHII_ratio)  # = (nHII_next-nHII)/dt ~ dnHII/dt
        DEBUG_helper("dT/dt", dT_dt)
        DEBUG_helper("dnHI/dt", dnHI_dt)
        DEBUG_helper("dnHII/dt", dnHII_dt)

        

        # // ---------- 2.1 RHS: Lagrangian energy eq ---------- \\

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

        # 2.1.1. Compute LHS.
        ne = nHII # + nHeII + 2 * nHeIII, but these last were assumed to be 0 --> cooling box model
        ne_next = nHII_next
        e_cool = self.phys.compute_cooling_rates(T_next, nHI_next, nHII_next, ne_next) # Implicit formulation

        # 2.1.2. Compute RHS.
        n_tot = nHI + nHII + ne
        n_tot_next = nHI_next + nHII_next + ne_next
        
        du_dt = (3/2*self.phys.k_B)/dt * (n_tot_next*T_next - n_tot*T)
        
        DEBUG_helper("T", T, min_val=0.0, max_val=1e15)
        DEBUG_helper("e_cool", e_cool)
        DEBUG_helper("n_tot", n_tot)
        DEBUG_helper("du/dt", du_dt)


        #    // ---------- 2.2 RHS: chemical rate eq ---------- \\

        # dn/dt = gain(n) - loss(n)

        # Goal: Compute gain(n), loss(n), where:
        #   gain(n) / loss(n): k_{ij}*n_i*n_j, where
        #       k_{ij}: rate at which species i,j can chemically react (dep. on T).
        #       n_i, n_j: species density for the species i,j, respectively.

        # 2.2.1. Compute the chemical rates.
        k1, k2 = self.phys.compute_chemical_rates(T_next) # Again, implicit formulation
        DEBUG_helper("k1_next", k1)
        DEBUG_helper("k2_next", k2)
        # k7 = rates['k7']

        # 2.2.2. Compute the gain and loss terms (both with an implicit formulation).
        ionisation = k1*nHI_next*ne_next
        recombination = k2*nHII_next*ne_next
        DEBUG_helper("ionisation", ionisation)
        DEBUG_helper("recombination", recombination)

        # 2.2.3. Compute dnHI/dt, dnHII/dt
        nHI_react = recombination - ionisation # k2*nHII*ne - k1*nHI*ne
        nHII_react = ionisation - recombination # k1*nHI*ne - k2*nHII*ne
        DEBUG_helper("nHI_react", nHI_react)
        DEBUG_helper("nHII_react", nHII_react)


        # // ----- 3. Physical Loss Computation --> log-MSE = (log(LHS)-log(RHS))^2 ----- \\
        # Alternative: USE NORMALISED LOSS
        eps = 1e-35

        # 3.1. Temperature
        log_cool = torch.log10(torch.abs(e_cool) + eps)
        log_du_dt = torch.log10(torch.abs(du_dt) + eps)
        log_T_mse = (log_cool - log_du_dt)**2
        DEBUG_helper("log_cool", log_cool)
        DEBUG_helper("log_du_dt", log_du_dt)
        DEBUG_helper("Physics Loss (log-MSE): nT", log_T_mse)

        # 3.2. Species (nHI)
        log_dnHI_dt = torch.log10(torch.abs(dnHI_dt) + eps)
        log_nHI_react = torch.log10(torch.abs(nHI_react) + eps)
        log_nHI_mse = (log_dnHI_dt - log_nHI_react)**2
        DEBUG_helper("log_dnHI_dt", log_dnHI_dt)
        DEBUG_helper("log_nHI_react", log_nHI_react)
        DEBUG_helper("Physics Loss (log-MSE): nHI", log_nHI_mse)

        # 3.3. Species (nHII)
        log_dnHII_dt = torch.log10(torch.abs(dnHII_dt) + eps)
        log_nHII_react = torch.log10(torch.abs(nHII_react) + eps)
        log_nHII_mse = (log_dnHII_dt - log_nHII_react)**2
        DEBUG_helper("log_dnHII_dt", log_dnHII_dt)
        DEBUG_helper("log_nHII_react", log_nHII_react)
        DEBUG_helper("Physics Loss (log-MSE): nHII", log_nHII_mse)

        # --- DIAGNOSTIC PRINT BLOCK ---
        if print_samples:
            batch_size = batch_x.shape[0]
            print(f"\n--- Batch Physics Comparison ({batch_size} samples) ---")
            for i in range(batch_size):
                print(f"Sample {i+1} (T: {T[i,0]}):")
                print(f"  Temp: LHS (e_cool)   = {e_cool[i, 0].item():12.4e} | RHS (du_dt)      = {du_dt[i, 0].item():12.4e} | log-MSE     = {log_T_mse[i, 0].item():12.4e}")
                print(f"  nHI : LHS (dnHI_dt)  = {dnHI_dt[i, 0].item():12.4e} | RHS (nHI_react)  = {nHI_react[i, 0].item():12.4e} | log-MSE     = {log_nHI_mse[i, 0].item():12.4e}")
                print(f"  nHII: LHS (dnHII_dt) = {dnHII_dt[i, 0].item():12.4e} | RHS (nHII_react) = {nHII_react[i, 0].item():12.4e} | log-MSE     = {log_nHII_mse[i, 0].item():12.4e}")
            print("-" * 55)

        physics_loss = log_T_mse + log_nHI_mse + log_nHII_mse
        return torch.mean(physics_loss)


# ---------------------------------------------------------
# Ground Truth Verification Block
# Append to the bottom of physics.py
# ---------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import random
    import glob
    import os
    torch.set_default_dtype(torch.float64)

    print("--- Verifying GrackleRates against Ground Truth Data ---")
    
    DATA_FOLDER = "Test_100yr/"
    NUM_SAMPLES = 20

    device = torch.device("cpu")

    def compute_global_scaling(folder_path, pattern="output_*.dat"):
        """
        Computes global mean and std from all files in the directory.
        """
        files = glob.glob(os.path.join(folder_path, pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching {pattern} in {folder_path}")
            
        print(f"Loading {len(files)} files...")
        all_inputs, all_targets = [], []
        
        for f in files:
            data = pd.read_csv(f, comment='#', sep='\s+', header=None)
            
            # Extract columns
            dt = data.iloc[1:, 2].values.astype(np.float64)
            T = data.iloc[:, 3].values.astype(np.float64)
            nHI = data.iloc[:, 6].values.astype(np.float64)
            nHII = data.iloc[:, 7].values.astype(np.float64)
            
            # State vectors
            T_curr, T_next = T[:-1], T[1:]
            nHI_curr, nHI_next = nHI[:-1], nHI[1:]
            nHII_curr, nHII_next = nHII[:-1], nHII[1:]
            
            # Formulate Log-Inputs and Log-Ratios
            inputs = np.stack([dt, np.log(T_curr), np.log(nHI_curr), np.log(nHII_curr)], axis=1)
            targets = np.stack([
                np.log(T_next) - np.log(T_curr),
                np.log(nHI_next) - np.log(nHI_curr),
                np.log(nHII_next) - np.log(nHII_curr)
            ], axis=1)
            
            all_inputs.append(inputs)
            all_targets.append(targets)

        all_inputs = torch.from_numpy(np.concatenate(all_inputs, axis=0)).double()
        all_targets = torch.from_numpy(np.concatenate(all_targets, axis=0)).double()

        in_mean = all_inputs.mean(dim=0).to(device)
        in_std = all_inputs.std(dim=0).to(device)
        tg_mean = all_targets.mean(dim=0).to(device)
        tg_std = all_targets.std(dim=0).to(device)

        print("Dataset Loaded.")
        
        return all_inputs, all_targets, in_mean, in_std, tg_mean, tg_std

    try:
        all_inputs, all_targets, in_mean, in_std, tg_mean, tg_std = compute_global_scaling(DATA_FOLDER)

        grackle_phys = GrackleRates()
        phys_manager = PhysicsLossManager(
            model=None, 
            grackle_phys=grackle_phys, 
            in_mean=in_mean, 
            in_std=in_std,
            tg_mean=tg_mean,
            tg_std=tg_std
        )

        # 3. Standardize the entire dataset
        inputs_scaled = (all_inputs.to(device) - in_mean) / (in_std + 1e-8)
        targets_scaled = (all_targets.to(device) - tg_mean) / (tg_std + 1e-8)
        
        total_rows = len(inputs_scaled)
        indices = sorted(random.sample(range(total_rows), min(NUM_SAMPLES, total_rows)))
        
        batch_x = inputs_scaled[indices]
        batch_y = targets_scaled[indices]
        
        with torch.no_grad():
            # Set print_samples=True to trigger the diagnostic block
            loss = phys_manager.get_residuals(batch_x, batch_y, print_samples=True)
            print(f"\n[✓] Total Log-MSE Physics Loss on Sampled Ground Truth: {loss.item():.4e}")

    except Exception as e:
        print(f"\n[!] Error during verification: {e}")