import torch
import numpy as np
from data_utils import DEBUG_helper

SEC_PER_YEAR = 86400*365.256363004 # Seconds in a year

class GrackleRates:
    """Analytical fits for Grackle primordial chemistry and cooling."""
    def __init__(self, units=1.0, tiny=1e-30):
        self.units = units
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
        
        rates = {}
        
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
                                - 2.039149852002e-6 * torch.pow(logT_ev, 8)) / self.units
        rates['k1'] = torch.where(T_ev > 0.8, k1, torch.clamp(k1, min=self.tiny))
        
        # k2: HII + e -> HI + photon (Case B Recombination)
        # Can be found in line 97 of grackle/src/clib/rate_functions.c
        k2 = (4.881357e-6 * torch.pow(T, -1.5) * torch.pow((1.0 + 1.14813e2 * torch.pow(T, -0.407)), -2.242)) / self.units
        rates['k2'] = torch.where(T < 1.0e9, k2, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE HELIUM IS SET TO 0 ----- \\
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

        # // ----- NOT CONSIDERED FOR NOW, SINCE HELIUM IS SET TO 0 ----- \\
        # # k4: HeII + e -> HeI + photon (Case B Recombination)
        # k4 = (1.26e-14 * torch.pow(5.7067e5 / T, 0.75)) / self.units
        # rates['k4'] = k4

        # // ----- NOT CONSIDERED FOR NOW, SINCE HELIUM IS SET TO 0 ----- \\
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

        # // ----- NOT CONSIDERED FOR NOW, SINCE HELIUM IS SET TO 0 ----- \\
        # # k6: HeIII + e -> HeII + photon (Case B Recombination)
        # k6 = (7.8155e-5 * torch.pow(T, -1.5) * torch.pow((1.0 + 2.0189e2 * torch.pow(T, -0.407)), -2.242)) / self.units
        # rates['k6'] = torch.where(T < 1.0e9, k6, torch.full_like(T, self.tiny))

        # k7 (k57 in source): HI + HI -> HII + HI + e
        # Can be found in line 678 of grackle/src/clib/rate_functions.c
        k7 = 1.2e-17 * torch.pow(T, 1.2) * torch.exp(-157800.0 / T) / self.units
        rates['k7'] = torch.where(T > 3000.0, k7, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE HELIUM IS SET TO 0 ----- \\
        # # k8 (k58 in source): HI + HeI -> HII + HeI + e
        # # Based on Lenzuni et al. (1991) heavy particle fit
        # k8 = 1.75e-17 * torch.pow(T, 1.3) * torch.exp(-157800.0 / T) / self.units
        # rates['k8'] = torch.where(T > 1.0e4, k8, torch.full_like(T, self.tiny))

        # // ----- NOT CONSIDERED FOR NOW, SINCE RT_HI_ionization_rate IS SET TO 0 ----- \\
        # k9: HI + photon -> HII + e
        # ...

        return rates

    def compute_cooling_rates(self, T, nHI, nHII, ne, rates):
        """Calculates the 4 primary cooling processes in [erg s^-1 cm^-3] units."""
        
        # // ----- 1. Collisional Excitation (Line Emission) - Cen (1992) fit ----- \\
        # Can be found in line 755 of grackle/src/clib/rate_functions.c
 
        # ceHI(i)*HI(i,j,k)*de(i,j,k)
        ceHI_rate = 7.5e-19 * torch.exp(-118348.0 / T) / (1.0 + torch.sqrt(T/1.0e5))
        lambda_ce = ceHI_rate * nHI * ne

        # // ----- 2. Collisional Ionization (Energy lost per k1 event) ------ \\
        # Can be found in line 799 of grackle/src/clib/rate_functions.c

        # ciHI(i)*HI(i,j,k)*de(i,j,k)
        ciHI_rate = 2.18e-11 * rates['k1']
        lambda_ci = ciHI_rate * nHI * ne
        # Previously: rates['k1'] * nHI * ne * (13.6 * self.eV_to_erg) # 13.6 eV is the ionization potential of Hydrogen (self.eV_to_erg = 1.60218e-12  # Energy conversion, defined in __init__)

        # // ----- 3. Recombination Cooling (Case B) ----- \\
        # Can be found in line 832 of grackle/src/clib/rate_functions.c

        # reHII(i)*HII(i,j,k)*de(i,j,k)
        lambdaHI = 2.0 * 157807.0 / T
        reHII_rate = 3.435e-30 * T * torch.pow(lambdaHI, 1.970) / torch.pow( 1.0 + torch.pow(lambdaHI/2.25, 0.376), 3.720)
        lambda_re = reHII_rate * nHII * ne
        # Previously: reHII_rate = 3.48e-26 * torch.sqrt(T) * torch.pow(T/1e3, -0.2) / (1.0 + torch.pow(T/1e6, 0.7))

        # // ----- 4. Bremsstrahlung (Free-Free) ----- \\
        # Can be found in line 910 of grackle/src/clib/rate_functions.c

        # brem(i)*HII(i,j,k)*de(i,j,k)
        brem_rate = 1.43e-27 * torch.sqrt(T) * (1.1 + 0.34 * torch.exp(-torch.pow(5.5 - torch.log10(T), 2) / 3.0))
        lambda_br = brem_rate * nHII * ne
        # Previously: lambda_br = 1.42e-27 * 1.3 * torch.sqrt(T) * nHII * ne
        
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

    def get_residuals(self, batch_x, preds):
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

        DEBUG_helper("T/dt:", T/dt)
        DEBUG_helper("expm1:", torch.expm1(log_T_ratio))
        
        dT_dt = (T/dt)*torch.expm1(log_T_ratio)           # = (T_next-T)/dt ~ dT/dt
        dnHI_dt = (nHI/dt)*torch.expm1(log_nHI_ratio)     # = (nHI_next-nHI)/dt ~ dnHI/dt
        dnHII_dt = (nHII/dt)*torch.expm1(log_nHII_ratio)  # = (nHII_next-nHII)/dt ~ dnHII/dt
        DEBUG_helper("dT/dt", dT_dt)
        DEBUG_helper("dnHI/dt", dnHI_dt)
        DEBUG_helper("dnHII/dt", dnHII_dt)

        

        # // ---------- 2.1 RHS: Lagrangian energy eq ---------- \\

        # Goal: Isolate dT/dt out of the Lagrangian Energy Equation:
        #       de/dt = -e_cool + e_heat
        
        # We neglect heating, and assume that cooling is dominated by hydrogen line emission:
        #       de/dt = -e_cool

        # Remember that
        #   E = rho · e, where
        #       E --> volumetric internal energy
        #       rho = n_{tot}·mu·m_H --> density
        #       e --> specific internal energy
        # Also, it is well known that
        #   e = (kT)/((gamma-1)·mu·m_H), where
        #       k --> Boltzmann constant
        #       T --> temperature
        #       gamma --> adiabatic index of an ideal gas
        #       mu --> mean molecular weight
        #       m_H --> mass of a Hydrogen atom
        # From this, one can derive
        #   E = rho · e = n_{tot}·(kT)/(gamma-1)
        #   dE/dt = n_{tot}·k/(gamma-1)·dT/dt, since
        #       k, gamma --> constant
        #       n_{tot} --> assumption that this is locally constant over the timestep
        # But remember that
        #   -e_cool = dE/dt = n_{tot}·k/(gamma-1)·dT/dt (energy lost due to cooling) # DEBUG!!!
        #   dT/dt = - (gamma-1)·e_cool / (n_{tot}·k)

        # 2.1.1. Compute quantities in RHS.
        gamma = 5.0/3.0
        ne = nHII # + nHeII + 2 * nHeIII, but these last were assumed to be 0 --> cooling box model
        n_tot = nHI + nHII + ne # DEBUG: ne also included in n_{tot}?
        
        rates = self.phys.compute_chemical_rates(T)
        e_cool = self.phys.compute_cooling_rates(T, nHI, nHII, ne, rates)
        
        
        DEBUG_helper("T", T, min_val=0.0, max_val=1e15)
        DEBUG_helper("e_cool", e_cool)
        DEBUG_helper("n_tot", n_tot)
        # mu = 1.0 // DEBUG: Not relevant in my analysis, but Lessandre noted that, since mu(T), the eqn. would need to be solved iteratively.
        
        # 2.1.2. Compute dT/dt
        T_physics = - (gamma-1) * e_cool / (n_tot * self.phys.k_B)
        DEBUG_helper("T_physics", T_physics)


        #    // ---------- 2.2 RHS: chemical rate eq ---------- \\

        # dn/dt = gain(n) - loss(n)

        # Goal: Compute gain(n), loss(n), where:
        #   gain(n) / loss(n): k_{ij}*n_i*n_j, where
        #       k_{ij}: rate at which species i,j can chemically react (dep. on T).
        #       n_i, n_j: species density for the species i,j, respectively.

        # 2.2.1. Compute the chemical rates.
        k1 = rates['k1']
        k2 = rates['k2']
        k7 = rates['k7']

        # 2.2.2. Compute the gain and loss terms.
        # Terms contributing to the production of HII (and loss of HI)
        HII_gain = k1*nHI*ne + k7*(nHI**2)
        HI_loss = HII_gain
        # Term contributing to the production of HI (and loss of HII)
        HI_gain = k2*nHII*ne 
        HII_loss = HI_gain

        # 2.2.3. Compute dnHI/dt, dnHII/dt
        nHI_physics = HI_gain - HI_loss # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        nHII_physics = HII_gain - HII_loss # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI
        DEBUG_helper("nHI_physics", nHI_physics)
        DEBUG_helper("nHII_physics", nHII_physics)

        # // ----- 3. Physical Loss Computation: ((LHS - RHS)/(RHS + eps))^2 ----- \\
        eps = 0.1
        loss_T   = torch.mean(((dT_dt - T_physics) / (torch.abs(T_physics) + eps))**2) # DEBUG: In the denominator, usually a scale > 0 is preferred (rather than y_physics).
        loss_HI  = torch.mean(((dnHI_dt - nHI_physics) / (torch.abs(nHI_physics) + eps))**2) # Same for these 2 below.
        loss_HII = torch.mean(((dnHII_dt - nHII_physics) / (torch.abs(nHII_physics) + eps))**2)
        # print(f"loss_T={loss_T}")
        # print(f"loss_HI={loss_HI}")
        # print(f"loss_HII={loss_HII}")
        DEBUG_helper("Physics Loss: loss_T", loss_T)
        DEBUG_helper("Physics Loss: loss_HI", loss_HI)
        DEBUG_helper("Physics Loss: loss_HII", loss_HII)

        return loss_T + loss_HI + loss_HII





        # ------------------------------------------------------------------------- #
        # ------------------- INPUT: [dt, T(t), nHI(t), nHII(t)] ------------------ #
        # ---------------- OUTPUT: [T(t+dt), nHI(t+dt), nHII(t+dt)] --------------- #
        # ------------------------------------------------------------------------- #

        # # // ----- COMPUTATION OF PHYSICAL LOSS FUNCTIONS ----- \\
        # # The rate equations are in linear space (dy/dt = ...)
        # # Our t and y are normalised (t_norm, y_norm)
        # # Hence we first denormalise
        # input_denorm = batch_x*(self.S_x + 1e-8) + self.x_min
        # output_denorm = preds*(self.S_y + 1e-8) + self.y_min

        # #       // ----- 1. LHS: dy/dt ----- \\
        # dt = input_denorm[:, 0:1]
        # T = output_denorm[:, 0:1]
        # nHI = output_denorm[:, 1:2]
        # nHII = output_denorm[:, 2:3]
        
        # dT_dt = (T[1:]-T[:-1])/dt          # dT/dt
        # dnHI_dt = (nHI[1:]-nHI[:-1])/dt    # dnHI/dt
        # dnHII_dt = (nHII[1:]-nHII[:-1])/dt # dnHII/dt
        # DEBUG_helper("LHS: T_LHS", dT_dt)
        # DEBUG_helper("LHS: HI_LHS", dnHI_dt)
        # DEBUG_helper("LHS: HII_LHS", dnHII_dt)

        

        # # // ---------- 2.1 RHS: Lagrangian energy eq ---------- \\

        # # Goal: Isolate dT/dt out of the Lagrangian Energy Equation:
        # #       de/dt = -e_cool + e_heat
        
        # # We neglect heating, and assume that cooling is dominated by hydrogen line emission:
        # #       de/dt = -e_cool

        # # Remember that
        # #   E = rho · e, where
        # #       E --> volumetric internal energy
        # #       rho = n_{tot}·mu·m_H --> density
        # #       e --> specific internal energy
        # # Also, it is well known that
        # #   e = (kT)/((gamma-1)·mu·m_H), where
        # #       k --> Boltzmann constant
        # #       T --> temperature
        # #       gamma --> adiabatic index of an ideal gas
        # #       mu --> mean molecular weight
        # #       m_H --> mass of a Hydrogen atom
        # # From this, one can derive
        # #   E = rho · e = n_{tot}·(kT)/(gamma-1)
        # #   dE/dt = n_{tot}·k/(gamma-1)·dT/dt, since
        # #       k, gamma --> constant
        # #       n_{tot} --> assumption that this is locally constant over the timestep
        # # But remember that
        # #   -e_cool = dE/dt = n_{tot}·k/(gamma-1)·dT/dt (energy lost due to cooling) # DEBUG!!!
        # #   dT/dt = - (gamma-1)·e_cool / (n_{tot}·k)

        # # 2.1.1. Compute quantities in RHS.
        # gamma = 5.0/3.0
        
        # rates = self.phys.compute_chemical_rates(T)
        # e_cool = self.phys.compute_cooling_rates(T, nHI, nHII, ne, rates)
        
        # ne = nHII # + nHeII + 2 * nHeIII, but these last were assumed to be 0 --> cooling box model
        # n_tot = nHI + nHII + ne # DEBUG: ne also included in n_{tot}?
        
        # DEBUG_helper("RHS_3: T", T, min_val=0.0, max_val=1e15)
        # DEBUG_helper("RHS_3: e_cool", e_cool)
        # DEBUG_helper("RHS_3: n_tot", n_tot)
        # # mu = 1.0 // DEBUG: Not relevant in my analysis, but Lessandre noted that, since mu(T), the eqn. would need to be solved iteratively.
        
        # # 2.1.2. Compute dT/dt
        # T_physics = - (gamma-1) * e_cool / (n_tot * self.phys.k_B)
        # DEBUG_helper("RHS_3: T_RHS_3", T_physics)


        # #    // ---------- 2.2 RHS: chemical rate eq ---------- \\

        # # dn/dt = gain(n) - loss(n)

        # # Goal: Compute gain(n), loss(n), where:
        # #   gain(n) / loss(n): k_{ij}*n_i*n_j, where
        # #       k_{ij}: rate at which species i,j can chemically react (dep. on T).
        # #       n_i, n_j: species density for the species i,j, respectively.

        # # 2.2.1. Compute the chemical rates.
        # k1 = rates['k1']
        # k2 = rates['k2']
        # k7 = rates['k7']

        # # 2.2.2. Compute the gain and loss terms.
        # # Terms contributing to the production of HII (and loss of HI)
        # HII_gain = k1*nHI*ne + k7*(nHI**2)
        # HI_loss = HII_gain
        # # Term contributing to the production of HI (and loss of HII)
        # HI_gain = k2*nHII*ne 
        # HII_loss = HI_gain

        # # 2.2.3. Compute dnHI/dt, dnHII/dt
        # nHI_physics = HI_gain - HI_loss # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        # nHII_physics = HII_gain - HII_loss # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI


        # # // ----- 3. Physical Loss Computation: ((LHS - RHS)/(RHS + eps))^2 ----- \\
        # eps = 0.1
        # loss_T   = torch.mean(((dT_dt - T_physics) / (torch.abs(T_physics) + eps))**2) # DEBUG: In the denominator, usually a scale > 0 is preferred (rather than T_RHS).
        # loss_HI  = torch.mean(((dnHI_dt - nHI_physics) / (torch.abs(nHI_physics) + eps))**2) # Same for these 2 below.
        # loss_HII = torch.mean(((dnHII_dt - nHII_physics) / (torch.abs(nHII_physics) + eps))**2)
        # # print(f"loss_T={loss_T}")
        # # print(f"loss_HI={loss_HI}")
        # # print(f"loss_HII={loss_HII}")
        # DEBUG_helper("Physics Loss: loss_T", loss_T)
        # DEBUG_helper("Physics Loss: loss_HI", loss_HI)
        # DEBUG_helper("Physics Loss: loss_HII", loss_HII)

        # return loss_T + loss_HI + loss_HII


        # ------------------------------------------------------------------------- #
        # ---------------- NO LOGS BUT INPUT: [t, T0, nHI0, nHII0] ---------------- #
        # ------------------------------------------------------------------------- #
        
        # # // ----- COMPUTATION OF PHYSICAL LOSS FUNCTIONS ----- \\
        # # The rate equations are in linear space (dy/dt = ...)
        # # Our t and y are normalised (t_norm, y_norm)
        # # Hence we first denormalise
        # output_denorm = preds*(self.S_y + 1e-8) + self.y_min
        # input_denorm = batch_x*(self.S_x + 1e-8) + self.x_min

        # T = output_denorm[:, 0:1]
        # nHI = output_denorm[:, 1:2]
        # nHII = output_denorm[:, 2:3]

        # #       // ----- 1. LHS: dy/dt ----- \\

        # def get_LHS(y):
        #     return torch.autograd.grad(
        #         y, input_denorm,
        #         grad_outputs=torch.ones_like(y), 
        #         create_graph=True, retain_graph=True
        #     )[0][:, 0:1]
        
        # dT_dt = get_LHS(T) # dT/dt
        # dnHI_dt = get_LHS(nHI) # dnHI/dt
        # dnHII_dt = get_LHS(nHII) # dnHII/dt
        # DEBUG_helper("LHS: T_LHS", dT_dt)
        # DEBUG_helper("LHS: HI_LHS", dnHI_dt)
        # DEBUG_helper("LHS: HII_LHS", dnHII_dt)



        
        # ------------------------------------------------------------------------- #
        # ----------- LOGS WITH INPUT: [t, log_T0, log_nHI0, log_nHII0] ----------- #
        # ------------------------------------------------------------------------- #

        # # We will write the physical loss equations by:
        #     # d(y_norm)/d(t_norm) = d(y_norm)/dy *   dy/dt  * dt/d(t_norm)
        #     #         [---]       =    [1/cm^3]  * [cm^3/s] *     [s]

        # #       // ----- 1. LHS: d(y_norm)/d(t_norm) ----- \\

        # def get_LHS(y_norm):
        #     return torch.autograd.grad(
        #         y_norm, batch_x,
        #         grad_outputs=torch.ones_like(y_norm), 
        #         create_graph=True, retain_graph=True
        #     )[0][:, 0:1]
        
        # T_LHS = get_LHS(preds[:, 0:1]) # d(T_norm)/d(t_norm)
        # HI_LHS = get_LHS(preds[:, 1:2]) # d(HI_norm)/d(t_norm)
        # HII_LHS = get_LHS(preds[:, 2:3]) # d(HII_norm)/d(t_norm)
        # DEBUG_helper("LHS: T_LHS", T_LHS)
        # DEBUG_helper("LHS: HI_LHS", HI_LHS)
        # DEBUG_helper("LHS: HII_LHS", HII_LHS)

        # #   // ----- 2. RHS: d(log_y_norm)/dlog_y * dlog_y/dy * dy/dt * dt/dt_norm ----- \\

        # #     // ----- 2.1: d(log_y_norm)/dlog_y ----- \\

        # # log_y_norm = (log_y - y_min) / S_y --> d(log_y_norm)/d(log_y) = 1 / S_y

        # T_RHS_1 = 1.0 / self.S_y[0]
        # HI_RHS_1 = 1.0 / self.S_y[1]
        # HII_RHS_1 = 1.0 / self.S_y[2]
        # DEBUG_helper("RHS_1", T_RHS_1)
        # DEBUG_helper("RHS_1", HI_RHS_1)
        # DEBUG_helper("RHS_1", HII_RHS_1)


        # #    // ---------- 2.2: dlog_y/dy ---------- \\

        # # log_y = log10(y) --> dlog_y/dy = 1 / (y * ln(10))
        
        # # 2.2.1. Denormalise values
        # preds_denorm = preds * self.S_y + self.y_min

        # # Clamp to avoid numerical issues in exponential (2.2.2) and division (2.2.3)
        # T_log = torch.clamp(preds_denorm[:, 0:1], min=0.0, max=self.y_max[0]+1.0)
        # HI_log = torch.clamp(preds_denorm[:, 1:2], min=-10.0, max=self.y_max[1]+1.0)
        # HII_log = torch.clamp(preds_denorm[:, 2:3], min=-10.0, max=self.y_max[2]+1.0)
        # # HeI_log, HeII_log, HeIII_log = preds_denorm[:, 3:4], preds_denorm[:, 4:5], preds_denorm[:, 5:6]

        # # 2.2.2. Linearise values
        # T = torch.pow(10, T_log)
        # nHI = torch.pow(10, HI_log)
        # nHII = torch.pow(10, HII_log)
        # # nHeI = torch.pow(10, HeI_log)
        # # nHeII = torch.pow(10, HeII_log)
        # # nHeIII = torch.pow(10, HeIII_log)
        # DEBUG_helper("RHS_2: Linear T", T)
        # DEBUG_helper("RHS_2: Linear nHI", nHI)
        # DEBUG_helper("RHS_2: Linear nHII", nHII)

        # # 2.2.3. Compute second term in RHS
        # ln10 = np.log(10.0)
        # T_RHS_2 = 1 / (T * ln10)
        # HI_RHS_2  = 1 / (nHI * ln10)
        # HII_RHS_2 = 1 / (nHII * ln10)
        # DEBUG_helper("RHS_2: T_RHS_2", T_RHS_2)
        # DEBUG_helper("RHS_2: HI_RHS_2", HI_RHS_2)
        # DEBUG_helper("RHS_2: HII_RHS_2", HII_RHS_2)


        # #    // ---------- 2.3: dy/dt ---------- \\

        # # This is where our cooling rate and chemical rate equations come in, since now both variables are linear and denormalised.

        # # // ----- 2.3.1: dT/dt ----- \\
        
        # # E = rho · e (E: volumetric internal energy)
        # # rho = n_{tot}·mu·m_H (rho: density)
        # # e = (kT)/((gamma-1)·mu·m_H) (e: specific internal energy)
        # # => E = rho · e = n_{tot}·(kT)/(gamma-1)
        # # => dE/dt = n_{tot}·k/(gamma-1)·dT/dt (assuming n_{tot} is locally constant over the timestep)
        # # => dE/dt = -lambda_tot (energy lost due to cooling)
        # # => -lambda_tot = n_{tot}·k/(gamma-1)·dT/dt => dT/dt = - (gamma-1)·lambda_tot / (n_{tot}·k)

        # ne = nHII # + nHeII + 2 * nHeIII
        # n_tot = nHI + nHII + ne
        # DEBUG_helper("RHS_3: T", T, min_val=0.0, max_val=1e15)
        # rates = self.phys.compute_chemical_rates(T)
        # lambda_tot = self.phys.compute_cooling_rates(T, nHI, nHII, ne, rates)
        # gamma = 5.0/3.0
        # # mu = 1.0 // Not relevant in my analysis, but Lessandre noted that, since mu(T), the eqn. would need to be solved iteratively.
        # DEBUG_helper("RHS_3: lambda_tot", lambda_tot)
        # DEBUG_helper("RHS_3: n_tot", n_tot)
        # T_RHS_3 = - (gamma-1) * lambda_tot / (n_tot * self.phys.k_B)
        # DEBUG_helper("RHS_3: T_RHS_3", T_RHS_3)


        # # // ----- 2.3.2: dn/dt ----- \\

        # # dn/dt = gain(n) - loss(n),
        # # where gain and loss are computed from chemical reaction rates of the form k_{ij}*n_i*n_j

        # # 2.3.2.1. Compute the chemical rates.
        # k1 = rates['k1']
        # k2 = rates['k2']
        # k7 = rates['k7']

        # # 2.3.2.2. Compute the gain and loss terms.
        # # Terms contributing to the production of HII (and loss of HI)
        # HII_gain = k1*nHI*ne + k7*(nHI**2)
        # HI_loss = HII_gain
        # # Term contributing to the production of HI (and loss of HII)
        # HI_gain = k2*nHII*ne 
        # HII_loss = HI_gain

        # # 2.3.2.3. Compute third term in RHS.
        # HI_RHS_3 = HI_gain - HI_loss # k7*nHI*nHI - k1*nHI*ne - k2*nHII*ne
        # HII_RHS_3 = HII_gain - HII_loss # k1*nHI*ne + k2*nHII*ne - k7*nHI*nHI


        # #       // ----- 2.4: dt/dt_norm ----- \\
        # # t_norm = (t - t_min) / S_x --> t = t_norm·S_x + t_min --> dt/dt_norm = S_x
        # # This must be in seconds, since term 2.3 computes the rates per second (cm^3/s)

        # SEC_PER_YEAR = 86400*365.256363004 # Seconds per year
        # RHS_4 = self.S_x[0]*SEC_PER_YEAR
        # DEBUG_helper("RHS_4", RHS_4)


        # # # --- THE TOTAL INSPECTION SYSTEM ---
        # # with torch.no_grad():
        # #     # 1. Check raw log-predictions
        # #     if T_log.max() > 15.0 or HI_log.max() > 10.0:
        # #         print(f"\n[!] PREDICTION OVERFLOW: T_log={T_log.max():.2f}, HI_log={HI_log.max():.2f}")

        # #     # 2. Check linear values (The values that actually enter the rates)
        # #     if T.max() > 1e15:
        # #         print(f"[!] LINEAR TEMP EXPLOSION: T={T.max():.2e}")

        # #     # 3. Check the "Scaling Multipliers" (t/y)
        # #     # This is often where the 10^28 comes from
        # #     T_scale = (t_lin / (T + 1e-20)).max()
        # #     HI_scale = (t_lin / (nHI + 1e-20)).max()
        # #     if T_scale > 1e20:
        # #         print(f"[!] MULTIPLIER EXPLOSION: t/T scale = {T_scale:.2e}")

        # #     # 4. Check the Linear RHS (The Grackle physics output)
        # #     if rhs_T_lin.abs().max() > 1e20:
        # #         print(f"[!] PHYSICS RHS EXPLOSION: rhs_T={rhs_T_lin.abs().max():.2e}")
        
        # # # --- END INSPECTION ---

        
        # # // ----- 3. Physical Loss Computation: ((LHS - RHS)/(RHS + eps))^2 ----- \\
        # eps = 0.1

        # T_RHS   = T_RHS_1 * T_RHS_2 * T_RHS_3 * RHS_4
        # HI_RHS  = HI_RHS_1 * HI_RHS_2 * HI_RHS_3 * RHS_4
        # HII_RHS = HII_RHS_1 * HII_RHS_2 * HII_RHS_3 * RHS_4
        # DEBUG_helper("Total RHS: T_RHS", T_RHS)
        # DEBUG_helper("Total RHS: HI_RHS", HI_RHS)
        # DEBUG_helper("Total RHS: HII_RHS", HII_RHS)
        # # Previously, denom. with:  / (torch.abs(T_RHS) + eps)
        # loss_T   = torch.mean(((T_LHS - T_RHS) / (torch.abs(T_RHS) + eps))**2) # DEBUG: In the denominator, usually a scale > 0 is preferred (rather than T_RHS).
        # loss_HI  = torch.mean(((HI_LHS - HI_RHS) / (torch.abs(HI_RHS) + eps))**2) # Same for these 2 below.
        # loss_HII = torch.mean(((HII_LHS - HII_RHS) / (torch.abs(HII_RHS) + eps))**2)
        # # print(f"loss_T={loss_T}")
        # # print(f"loss_HI={loss_HI}")
        # # print(f"loss_HII={loss_HII}")
        # DEBUG_helper("Physics Loss: loss_T", loss_T)
        # DEBUG_helper("Physics Loss: loss_HI", loss_HI)
        # DEBUG_helper("Physics Loss: loss_HII", loss_HII)

        # return loss_T + loss_HI + loss_HII