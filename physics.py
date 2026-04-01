import torch

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
    def __init__(self, model, grackle_phys):
        self.model = model
        self.phys = grackle_phys

    def get_residuals(self, batch_x):
        """
        Computes the residuals for all species.
        batch_x: [log10_t, log10_T0, log10_HI0, log10_HII0]
        """
        batch_x.requires_grad = True
        t_log = batch_x[:, 0:1] # Extract log10(t)
        t_lin = 10**t_log # Convert to linear time for physical rate calculations

        # Forward pass
        pred_log = self.model(batch_x)
        T_log, HI_log, HII_log = pred_log[:, 0:1], pred_log[:, 1:2], pred_log[:, 2:3]
        # HeI_log, HeII_log, HeIII_log = pred_log[:, 3:4], pred_log[:, 4:5], pred_log[:, 5:6]

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

        # Function for d(log10_y)/d(log10_t)
        def get_log_grad(y_log):
            return torch.autograd.grad(
                y_log, batch_x, grad_outputs=torch.ones_like(y_log),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]

        # Convert log derivatives (since PINN is in log space) to linear derivatives (for physical loss functions):
        # Chain rule: dy/dt = (y/t) * dlogy/dlogt
        dT_dt_net = (T / t_lin) * get_log_grad(T_log) # dT/dt
        dHI_dt_net = (nHI / t_lin) * get_log_grad(HI_log) # dnHI/dt
        dHII_dt_net = (nHII / t_lin) * get_log_grad(HII_log) # dnHII/dt


        #       // ----- 2. RHS (k_{ij}*n_i*n_j) ----- \\

        # // --- 2.1. Chemical residuals --- \\
        rates = self.phys.compute_chemical_rates(T)

        HII_gain = rates['k1']*nHI*ne + rates['k7']*(nHI**2) # Terms contributing to the production of HII (and loss of HI)
        HI_gain = rates['k2']*nHII*ne # Term contributing to the production of HI (and loss of HII)

        rhs_HI = HI_gain - HII_gain # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        rhs_HII = HII_gain - HI_gain # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI


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

        rhs_T = - (gamma-1) * lambda_tot / (n_tot * self.phys.k_B)

        
        # // ----- 3. Final computation of residuals (MSE) ----- \\
        loss_HI = torch.mean((dHI_dt_net - rhs_HI)**2)
        loss_HII = torch.mean((dHII_dt_net - rhs_HII)**2)
        loss_T = torch.mean((dT_dt_net - rhs_T)**2)

        return loss_HI + loss_HII + loss_T