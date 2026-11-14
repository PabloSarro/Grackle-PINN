import numpy as np

# Number of desired valid configurations
target_sims = 1000

# Physical bounds based on Grackle limits
T_min, T_max = 7.0, 8.5     # log10(Kelvin) -> 10^7 to 10^8.5
n_min, n_max = 1.0, 2.0     # log10(cm^-3) -> 10 to 100 particles/cm3
x_min, x_max = 0.0, 1.0     # Ionization fraction (nHII / nH_total)

generated = 0

with open("params_active.txt", "w") as f:
    while generated < target_sims:
        # 1. Sample candidate initial conditions
        T0 = 10**np.random.uniform(T_min, T_max)
        nH = 10**np.random.uniform(n_min, n_max)
        xHII = np.random.uniform(x_min, x_max)
        
        # 2. Derive physical densities to check Active_3 boundaries
        nHI = nH * (1.0 - xHII)
        nHII = nH * xHII
        
        # 3. Apply Active_3 Piecewise Selection Rules
        if T0 > 2.5e7:
            # High-Temperature Regime
            if nHI < 25.00:
                continue
        else:
            # Low-Temperature Regime
            if nHI < 10.00:
                continue
            
            # Check conditional nHII floor based on the 2.34e7 K threshold
            if T0 >= 2.34e7:
                if nHII < 0.14:
                    continue
            else:
                if nHII < 1.0:
                    continue
                    
        # 4. Write if valid and increment counter
        f.write(f"{T0:.4e} {nH:.4e} {xHII:.4e}\n")
        generated += 1

print(f"[✓] Successfully generated {target_sims} parameters matching Active_3 in params_3.txt")