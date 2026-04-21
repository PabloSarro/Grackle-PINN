import torch
import numpy as np

class GrackleRates:
    """Analytical fits for Grackle primordial chemistry and cooling."""
    def __init__(self, units=1.0, tiny=1e-30):
        self.units = units
        self.tiny = tiny
        self.k_B = 1.380649e-16       # Boltzmann constant [erg/K]
        self.eV_to_erg = 1.60218e-12  # Energy conversion
        self.tevk = 11605.0           # K to eV conversion factor

    def compute_chemical_rates(self, T):
        """
        Calculates all 8 chemical reaction rates based on temperature T (in Kelvin).
        For now, some of them are commented out, since Helium is assumed to be 0.
        """
        T = torch.clamp(T, min=1.0) # Avoid unphysically low temperatures.
        T_ev = T / self.tevk
        logT_ev = torch.log(T_ev)
        
        rates = {}
        
        # k1: HI + e -> HII + 2e (Collisional Ionization)
        k1 = torch.exp(-32.71396786375 
                                + 13.53655609057 * logT_ev 
                                - 5.739328757388 * torch.pow(logT_ev, 2)
                                + 1.563154982022 * torch.pow(logT_ev, 3) 
                                - 0.2877056004391 * torch.pow(logT_ev, 4)
                                + 0.03482559773736999 * torch.pow(logT_ev, 5) 
                                - 0.00263197617559 * torch.pow(logT_ev, 6)
                                + 0.0001119543953861 * torch.pow(logT_ev, 7) 
                                - 2.039149852002e-6 * torch.pow(logT_ev, 8)) / self.units
        rates['k1'] = torch.where(T_ev > 0.8, k1, torch.clamp(k1, min=self.tiny))
        
        # k2: HII + e -> HI + photon (Case B Recombination)
        k2 = (4.881357e-6 * torch.pow(T, -1.5) * torch.pow((1.0 + 1.14813e2 * torch.pow(T, -0.407)), -2.242)) / self.units
        rates['k2'] = torch.where(T < 1.0e9, k2, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE INVOLVES HELIUM ----- \\
        # # k3: HeI + e -> HeII + 2e
        # k3 = torch.exp(-44.09864886561001 
        #                    + 23.91596563469 * logT_ev 
        #                    - 10.75323019821 * torch.pow(logT_ev, 2)
        #                    + 3.058038757198 * torch.pow(logT_ev, 3) 
        #                    - 0.5685118909884001 * torch.pow(logT_ev, 4)
        #                    + 0.06795391233790001 * torch.pow(logT_ev, 5) 
        #                    - 0.005009056101857001 * torch.pow(logT_ev, 6)
        #                    + 0.0002067236157507 * torch.pow(logT_ev, 7) 
        #                    - 3.649161410833e-6 * torch.pow(logT_ev, 8)) / self.units
        # rates['k3'] = torch.where(T_ev > 0.8, k3, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE INVOLVES HELIUM ----- \\
        # # k4: HeII + e -> HeI + photon (Case B Recombination)
        # k4 = (1.26e-14 * torch.pow(5.7067e5 / T, 0.75)) / self.units
        # rates['k4'] = k4

        # // ----- NOT CONSIDERED FOR NOW, SINCE INVOLVES HELIUM ----- \\
        # # k5: HeII + e -> HeIII + 2e
        # k5 = torch.exp(-68.71040990212001 
        #                    + 43.93347632635 * logT_ev 
        #                    - 18.48066993568 * torch.pow(logT_ev, 2)
        #                    + 4.701626486759002 * torch.pow(logT_ev, 3) 
        #                    - 0.7692466334492 * torch.pow(logT_ev, 4)
        #                    + 0.08113042097303 * torch.pow(logT_ev, 5) 
        #                    - 0.005324020628287001 * torch.pow(logT_ev, 6)
        #                    + 0.0001975705312221 * torch.pow(logT_ev, 7) 
        #                    - 3.165581065665e-6 * torch.pow(logT_ev, 8)) / self.units
        # rates['k5'] = torch.where(T_ev > 0.8, k5, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE INVOLVES HELIUM ----- \\
        # # k6: HeIII + e -> HeII + photon (Case B Recombination)
        # k6 = (7.8155e-5 * torch.pow(T, -1.5) * torch.pow((1.0 + 2.0189e2 * torch.pow(T, -0.407)), -2.242)) / self.units
        # rates['k6'] = torch.where(T < 1.0e9, k6, torch.full_like(T, self.tiny))

        # k7 (k57 in source): HI + HI -> HII + HI + e
        k7 = 1.2e-17 * torch.pow(T, 1.2) * torch.exp(-157800.0 / T) / self.units
        rates['k7'] = torch.where(T > 3000.0, k7, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE INVOLVES HELIUM ----- \\
        # # k8 (k58 analogue): HI + HeI -> HII + HeI + e
        # # Based on Lenzuni et al. (1991) heavy particle fit
        # k8 = 1.75e-17 * torch.pow(T, 1.3) * torch.exp(-157800.0 / T) / self.units
        # rates['k8'] = torch.where(T > 1.0e4, k8, torch.full_like(T, self.tiny))

        return rates

    def compute_cooling_rates(self, T, nHI, nHII, ne, rates):
        """Calculates the 4 primary cooling processes in [erg s^-1 cm^-3] units."""
        
        # // ----- 1. Collisional Excitation (Line Emission) - Cen (1992) fit ----- \\
        # ceHI(i)*HI(i,j,k)*de(i,j,k)
        lambda_ceHI = 7.5e-19 * torch.exp(-118348.0 / T) / (1.0 + torch.sqrt(T/1e5))
        cooling_ce = lambda_ceHI * nHI * ne

        # // ----- 2. Collisional Ionization (Energy lost per k1 event) ------ \\
        # ciHI(i)*HI(i,j,k)*de(i,j,k)
        cooling_ci = rates['k1'] * nHI * ne * (13.6 * self.eV_to_erg) # 13.6 eV is the ionization potential of Hydrogen

        # // ----- 3. Recombination Cooling (Case B) ----- \\
        # reHII(i)*HII(i,j,k)*de(i,j,k)
        lambda_reHII = 3.48e-26 * torch.sqrt(T) * torch.pow(T/1e3, -0.2) / (1.0 + torch.pow(T/1e6, 0.7))
        cooling_re = lambda_reHII * nHII * ne

        # // ----- 4. Bremsstrahlung (Free-Free) ----- \\
        # brem(i)*HII(i,j,k)*de(i,j,k)
        cooling_br = 1.42e-27 * 1.3 * torch.sqrt(T) * nHII * ne
        
        return cooling_ce + cooling_ci + cooling_re + cooling_br
    

class PhysicsLossManager:
    """Manages Automatic Differentiation and Residual computation."""
    def __init__(self, model, grackle_phys, x_min, x_max, y_min, y_max):
        self.model = model
        self.phys = grackle_phys
        
        self.x_min = x_min
        self.x_max = x_max
        self.S_x = x_max - x_min

        self.y_min = y_min
        self.y_max = y_max
        self.S_y = y_max - y_min

    def get_residuals(self, batch_x_norm, preds_norm):
        """
        Computes the residuals for all species.
        batch_x: [log10_t, log10_T0, log10_HI0, log10_HII0]
        """
        if torch.isnan(preds_norm).any():
            print(f"DEBUG: 'preds_norm' contains NaNs from the model output.")

        # Recover physical (log) values from normalized inputs
        preds_log = preds_norm * self.S_y + self.y_min

        # Forward pass
        T_log, HI_log, HII_log = torch.clamp(preds_log[:, 0:1], min=0.0, max=9.0), torch.clamp(preds_log[:, 1:2], min=-20.0, max=5.0), torch.clamp(preds_log[:, 2:3], min=-20.0, max=5.0)
        # HeI_log, HeII_log, HeIII_log = preds_log[:, 3:4], preds_log[:, 4:5], preds_log[:, 5:6]
        
        # Convert to linear scale for physics calculations
        T = 10**T_log
        nHI = 10**HI_log
        nHII = 10**HII_log
        # nHeI = 10**HeI_log
        # nHeII = 10**HeII_log
        # nHeIII = 10**HeIII_log
        
        # Derived electron density
        ne = nHII # + nHeII + 2 * nHeIII



        # // ----- COMPUTATION OF PHYSICAL LOSS FUNCTIONS ----- \\

        #       // ----- 1. LHS (dn/dt and dT/dt)) ----- \\

        # Function for d(log10_y)/dt
        def get_phys_grad(y_log, target_idx):
            # grad_norm is d(y_log_norm) / d(t_lin_norm)
            grad_norm = torch.autograd.grad(
                y_log, batch_x_norm, grad_outputs=torch.ones_like(y_log),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]

            # Chain rule: d(y_log)/dt = (S_y / S_t) * d(y_norm)/d(t_norm)
            # self.S_x[0] is the scaling factor for Time (t_max - t_min)
            return (grad_norm * self.S_y[target_idx]) / self.S_x[0]

        # With the previous, we have converted the log derivatives (since PINN 
        # is in log space) to linear derivatives (for physical loss functions):
        dlogT_dt = get_phys_grad(T_log, target_idx=0) # dT/dt
        dlogHI_dt = get_phys_grad(HI_log, target_idx=1) # dnHI/dt
        dlogHII_dt = get_phys_grad(HII_log, target_idx=2) # dnHII/dt


        #       // ----- 2. RHS (k_{ij}*n_i*n_j) ----- \\

        # // --- 2.1. Chemical residuals --- \\
        rates = self.phys.compute_chemical_rates(T)

        HII_gain = rates['k1']*nHI*ne + rates['k7']*(nHI**2) # Terms contributing to the production of HII (and loss of HI)
        HI_gain = rates['k2']*nHII*ne # Term contributing to the production of HI (and loss of HII)

        rhs_HI_lin = HI_gain - HII_gain # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        rhs_HII_lin = HII_gain - HI_gain # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI


        # // --- 2.2. Thermal (Energy) residuals --- \\
        lambda_tot = self.phys.compute_cooling_rates(T, nHI, nHII, ne, rates)
        n_tot = nHI + nHII + ne
        
        # E = rho · e (E: volumetric internal energy)
        # rho = n_{tot}·mu·m_H (rho: density)
        # e = (kT)/((gamma-1)·mu·m_H) (e: specific internal energy)
        # => E = rho · e = n_{tot}·(kT)/(gamma-1)
        # => dE/dt = n_{tot}·(k/(gamma-1))·dT/dt (assuming n_{tot} is locally constant over the timestep)
        # => dE/dt = -lambda_tot (energy lost due to cooling)
        # => -lambda_tot = n_{tot}·(k/(gamma-1))·dT/dt => dT/dt = - (gamma-1)·lambda_tot / (n_{tot}·k)
        gamma = 5.0/3.0
        # mu = 1.0 // Ignored by now, but Lessandre noted that, since mu(T), the eqn. would need to be solved iteratively.

        rhs_T_lin = - (gamma-1) * lambda_tot / (n_tot * self.phys.k_B)

        # # --- THE TOTAL INSPECTION SYSTEM ---
        # with torch.no_grad():
        #     # 1. Check raw log-predictions
        #     if T_log.max() > 15.0 or HI_log.max() > 10.0:
        #         print(f"\n[!] PREDICTION OVERFLOW: T_log={T_log.max():.2f}, HI_log={HI_log.max():.2f}")

        #     # 2. Check linear values (The values that actually enter the rates)
        #     if T.max() > 1e15:
        #         print(f"[!] LINEAR TEMP EXPLOSION: T={T.max():.2e}")

        #     # 3. Check the "Scaling Multipliers" (t/y)
        #     # This is often where the 10^28 comes from
        #     T_scale = (t_lin / (T + 1e-20)).max()
        #     HI_scale = (t_lin / (nHI + 1e-20)).max()
        #     if T_scale > 1e20:
        #         print(f"[!] MULTIPLIER EXPLOSION: t/T scale = {T_scale:.2e}")

        #     # 4. Check the Linear RHS (The Grackle physics output)
        #     if rhs_T_lin.abs().max() > 1e20:
        #         print(f"[!] PHYSICS RHS EXPLOSION: rhs_T={rhs_T_lin.abs().max():.2e}")
        
        # # --- END INSPECTION ---

        
        # // ----- 3. Log-Space residuals ----- \\
        # Multiply linear RHS by (t/y) to match the LHS log-derivatives.
        eps = 1e-8
        ln10 = np.log(10.0)

        # Transform RHS: dy_log/dt = (1 / (y_lin * ln10)) * dy_lin/dt
        rhs_HI_log  = rhs_HI_lin  / (torch.clamp(nHI, min=1e-10) * ln10)
        rhs_HII_log = rhs_HII_lin / (torch.clamp(nHII, min=1e-10) * ln10)
        rhs_T_log   = rhs_T_lin   / (torch.clamp(T, min=1.0) * ln10)

        loss_HI = torch.mean(((dlogHI_dt - rhs_HI_log) / (torch.abs(rhs_HI_log) + eps))**2)
        loss_HII = torch.mean(((dlogHII_dt - rhs_HII_log) / (torch.abs(rhs_HII_log) + eps))**2)
        loss_T = torch.mean(((dlogT_dt - rhs_T_log) / (torch.abs(rhs_T_log) + eps))**2)

        return loss_HI + loss_HII + loss_T