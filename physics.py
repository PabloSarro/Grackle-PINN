import torch
import numpy as np
from data_utils import DEBUG_helper

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

    def get_residuals(self, batch_x, preds):
        """
        Computes the residuals for all species.
            
            batch_x: [t, log10_T0, log10_HI0, log10_HII0] (all normalised)
            preds: [log10_T, log10_HI, log10_HII] (all normalised)
        """
        DEBUG_helper("get_residuals: Input batch_x", batch_x)
        DEBUG_helper("get_residuals: Input preds", preds)

        # // ----- COMPUTATION OF PHYSICAL LOSS FUNCTIONS ----- \\
        # The rate equations are in linear space (dy/dt = ...)
        # Our t is normalised (t_norm) and y is additionally in log space (log_y_norm)
        # 
        # We will rewrite the physical loss equations by using the chain rule:
            # d(log_y_norm)/dt_norm = d(log_y_norm)/dlog_y * dlog_y/dy *  dy/dt * dt/dt_norm
            #         [---]         =      [1/cm^3}]       *   [---]   *[cm^3/s]*    [s]

        #       // ----- 1. LHS: d(log_y_norm)/dt_norm ----- \\

        def get_LHS(log_y_norm):
            return torch.autograd.grad(
                log_y_norm, batch_x,
                grad_outputs=torch.ones_like(log_y_norm), 
                create_graph=True, retain_graph=True
            )[0][:, 0:1]
        
        T_LHS = get_LHS(preds[:, 0:1]) # d(log_T_norm)/dt_norm
        HI_LHS = get_LHS(preds[:, 1:2]) # d(log_HI_norm)/dt_norm
        HII_LHS = get_LHS(preds[:, 2:3]) # d(log_HII_norm)/dt_norm
        DEBUG_helper("LHS: T_LHS", T_LHS)
        DEBUG_helper("LHS: HI_LHS", HI_LHS)
        DEBUG_helper("LHS: HII_LHS", HII_LHS)

        #   // ----- 2. RHS: d(log_y_norm)/dlog_y * dlog_y/dy * dy/dt * dt/dt_norm ----- \\

        #     // ----- 2.1: d(log_y_norm)/dlog_y ----- \\

        # log_y_norm = (log_y - y_min) / S_y --> d(log_y_norm)/d(log_y) = 1 / S_y

        T_RHS_1 = 1.0 / self.S_y[0]
        HI_RHS_1 = 1.0 / self.S_y[1]
        HII_RHS_1 = 1.0 / self.S_y[2]
        DEBUG_helper("RHS_1", T_RHS_1)
        DEBUG_helper("RHS_1", HI_RHS_1)
        DEBUG_helper("RHS_1", HII_RHS_1)


        #    // ---------- 2.2: dlog_y/dy ---------- \\

        # log_y = log10(y) --> dlog_y/dy = 1 / (y * ln(10))
        
        # 2.2.1. Denormalise values
        preds_denorm = preds * self.S_y + self.y_min

        # Clamp to avoid numerical issues in exponential (2.2.2) and division (2.2.3)
        T_log = torch.clamp(preds_denorm[:, 0:1], min=0.0, max=self.y_max[0]+1.0)
        HI_log = torch.clamp(preds_denorm[:, 1:2], min=-10.0, max=self.y_max[1]+1.0)
        HII_log = torch.clamp(preds_denorm[:, 2:3], min=-10.0, max=self.y_max[2]+1.0)
        # HeI_log, HeII_log, HeIII_log = preds_denorm[:, 3:4], preds_denorm[:, 4:5], preds_denorm[:, 5:6]

        # 2.2.2. Linearise values
        T = torch.pow(10, T_log)
        nHI = torch.pow(10, HI_log)
        nHII = torch.pow(10, HII_log)
        # nHeI = torch.pow(10, HeI_log)
        # nHeII = torch.pow(10, HeII_log)
        # nHeIII = torch.pow(10, HeIII_log)
        DEBUG_helper("RHS_2: Linear T", T)
        DEBUG_helper("RHS_2: Linear nHI", nHI)
        DEBUG_helper("RHS_2: Linear nHII", nHII)

        # 2.2.3. Compute second term in RHS
        ln10 = np.log(10.0)
        T_RHS_2 = 1 / (T * ln10)
        HI_RHS_2  = 1 / (nHI * ln10)
        HII_RHS_2 = 1 / (nHII * ln10)
        DEBUG_helper("RHS_2: T_RHS_2", T_RHS_2)
        DEBUG_helper("RHS_2: HI_RHS_2", HI_RHS_2)
        DEBUG_helper("RHS_2: HII_RHS_2", HII_RHS_2)


        #    // ---------- 2.3: dy/dt ---------- \\

        # This is where our cooling rate and chemical rate equations come in, since now both variables are linear and denormalised.

        # // ----- 2.3.1: dT/dt ----- \\
        
        # E = rho · e (E: volumetric internal energy)
        # rho = n_{tot}·mu·m_H (rho: density)
        # e = (kT)/((gamma-1)·mu·m_H) (e: specific internal energy)
        # => E = rho · e = n_{tot}·(kT)/(gamma-1)
        # => dE/dt = n_{tot}·k/(gamma-1)·dT/dt (assuming n_{tot} is locally constant over the timestep)
        # => dE/dt = -lambda_tot (energy lost due to cooling)
        # => -lambda_tot = n_{tot}·k/(gamma-1)·dT/dt => dT/dt = - (gamma-1)·lambda_tot / (n_{tot}·k)

        ne = nHII # + nHeII + 2 * nHeIII
        n_tot = nHI + nHII + ne
        DEBUG_helper("RHS_3: T", T, min_val=0.0, max_val=1e15)
        rates = self.phys.compute_chemical_rates(T)
        lambda_tot = self.phys.compute_cooling_rates(T, nHI, nHII, ne, rates)
        gamma = 5.0/3.0
        # mu = 1.0 // Not relevant in my analysis, but Lessandre noted that, since mu(T), the eqn. would need to be solved iteratively.
        DEBUG_helper("RHS_3: lambda_tot", lambda_tot)
        DEBUG_helper("RHS_3: n_tot", n_tot)
        T_RHS_3 = - (gamma-1) * lambda_tot / (n_tot * self.phys.k_B)
        DEBUG_helper("RHS_3: T_RHS_3", T_RHS_3)


        # // ----- 2.3.2: dn/dt ----- \\

        # dn/dt = gain(n) - loss(n),
        # where gain and loss are computed from chemical reaction rates of the form k_{ij}*n_i*n_j

        # 2.3.2.1. Compute the chemical rates.
        k1 = rates['k1']
        k2 = rates['k2']
        k7 = rates['k7']

        # 2.3.2.2. Compute the gain and loss terms.
        # Terms contributing to the production of HII (and loss of HI)
        HII_gain = k1*nHI*ne + k7*(nHI**2)
        HI_loss = HII_gain
        # Term contributing to the production of HI (and loss of HII)
        HI_gain = k2*nHII*ne 
        HII_loss = HI_gain

        # 2.3.2.3. Compute third term in RHS.
        HI_RHS_3 = HI_gain - HI_loss # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        HII_RHS_3 = HII_gain - HII_loss # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI


        #       // ----- 2.4: dt/dt_norm ----- \\
        # t_norm = (t - t_min) / S_x --> t = t_norm·S_x + t_min --> dt/dt_norm = S_x
        # This must be in seconds, since term 2.3 computes the rates per second (cm^3/s)

        SEC_PER_YEAR = 86400*365.256363004 # Seconds per year
        RHS_4 = self.S_x[0]*SEC_PER_YEAR
        DEBUG_helper("RHS_4", RHS_4)


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

        
        # // ----- 3. Physical Loss Computation: ((LHS - RHS)/(RHS + eps))^2 ----- \\
        eps = 0.1

        T_RHS   = T_RHS_1 * T_RHS_2 * T_RHS_3 * RHS_4
        HI_RHS  = HI_RHS_1 * HI_RHS_2 * HI_RHS_3 * RHS_4
        HII_RHS = HII_RHS_1 * HII_RHS_2 * HII_RHS_3 * RHS_4
        DEBUG_helper("Total RHS: T_RHS", T_RHS)
        DEBUG_helper("Total RHS: HI_RHS", HI_RHS)
        DEBUG_helper("Total RHS: HII_RHS", HII_RHS)
        # Previously, denom. with:  / (torch.abs(T_RHS) + eps)
        loss_T   = torch.mean(((T_LHS - T_RHS) / (torch.abs(T_RHS) + eps))**2) # DEBUG: In the denominator, usually a scale > 0 is preferred (rather than T_RHS).
        loss_HI  = torch.mean(((HI_LHS - HI_RHS) / (torch.abs(HI_RHS) + eps))**2) # Same for these 2 below.
        loss_HII = torch.mean(((HII_LHS - HII_RHS) / (torch.abs(HII_RHS) + eps))**2)
        # print(f"loss_T={loss_T}")
        # print(f"loss_HI={loss_HI}")
        # print(f"loss_HII={loss_HII}")
        DEBUG_helper("Physics Loss: loss_T", loss_T)
        DEBUG_helper("Physics Loss: loss_HI", loss_HI)
        DEBUG_helper("Physics Loss: loss_HII", loss_HII)

        return loss_T + loss_HI + loss_HII