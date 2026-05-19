import numpy as np

# Number of independent simulations
num_sims = 1000

# Physical bounds based on Grackle limits
T_min, T_max = 2.0, 8.0     # log10(Kelvin) -> 100K to 100MK
n_min, n_max = -4.0, 2.0    # log10(cm^-3) -> 0.0001 to 100 particles/cm3
x_min, x_max = 0.0, 1.0     # Ionization fraction (nHII / nH_total)

with open("params.txt", "w") as f:
    for i in range(num_sims):
        T0 = 10**np.random.uniform(T_min, T_max)
        nH = 10**np.random.uniform(n_min, n_max)
        xHII = np.random.uniform(x_min, x_max)
        # Format: T0 total_H_density nHII_fraction
        f.write(f"{T0:.4e} {nH:.4e} {xHII:.4e}\n")