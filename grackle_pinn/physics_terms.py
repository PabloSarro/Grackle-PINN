"""B3: physics residuals of the hydrogen-only Grackle network (P3 to P6) as functions of one 100-yr step.

Inputs are natural-log states `ls = ln [T, nHI, nHII]` of the current row (cur) and of the predicted next row (nxt).
Each term returns a dimensionless residual per pair; the loss is mean(residual^2). The time derivative is taken
discretely, over the fixed step dt = 100 yr, which is how the ground truth is stored (rows one step apart). For a
perfect integrator the residual is zero, so the truth itself gives the consistency test (tests/test_physics_terms.py).

Chemistry (electrons ne = nHII, hydrogen only), written for ln nHII (flat-regime sensitive):
    d nHII/dt = k1 nHI nHII - k2 nHII^2 + C7 k7 nHI^2        (k1 collisional ionisation, k2 case-B recombination,
                                                             k7 HI+HI ionisation; C7 = 1 includes it)
Energy (Lagrangian, heating neglected): d U/dt = -e_cool, U = 3/2 k_B n_tot T, n_tot = nHI + 2 nHII.
Rate fits and cooling terms: Grackle 'rate_functions.c' (fits as in commit 475ca3b, physics.py; k7 restored).
"""
import math
import torch

SEC_PER_YEAR = 86400 * 365.256363004
DT = 100.0 * SEC_PER_YEAR
K_B = 1.380649e-16
TINY = 1e-30


def k1(T):
    T = torch.clamp(T, min=1.0)
    lt = torch.log(T / 11605.0)
    poly = (-32.71396786375 + 13.53655609057 * lt - 5.739328757388 * lt**2 + 1.563154982022 * lt**3
            - 0.2877056004391 * lt**4 + 0.03482559773736999 * lt**5 - 0.00263197617559 * lt**6
            + 0.0001119543953861 * lt**7 - 2.039149852002e-6 * lt**8)
    k = torch.exp(poly)
    return torch.where(T / 11605.0 > 0.8, k, torch.clamp(k, min=TINY))


def k2(T):
    T = torch.clamp(T, min=1.0)
    k = 4.881357e-6 * T.pow(-1.5) * (1.0 + 1.14813e2 * T.pow(-0.407)).pow(-2.242)
    return torch.where(T < 1.0e9, k, torch.full_like(T, TINY))


def k7(T):
    T = torch.clamp(T, min=1.0)
    k = 1.2e-17 * T.pow(1.2) * torch.exp(-157800.0 / T)
    return torch.where(T > 3000.0, k, torch.full_like(T, TINY))


def cooling(T, nHI, nHII):
    """e_cool [erg s^-1 cm^-3]: collisional excitation, collisional ionisation, recombination, bremsstrahlung."""
    T = torch.clamp(T, min=1.0)
    ne = nHII
    ce = 7.5e-19 * torch.exp(-118348.0 / T) / (1.0 + torch.sqrt(T / 1.0e5)) * nHI * ne
    ci = 2.18e-11 * k1(T) * nHI * ne
    lHI = 2.0 * 157807.0 / T
    re = 3.435e-30 * T * lHI.pow(1.970) / (1.0 + (lHI / 2.25).pow(0.376)).pow(3.720) * nHII * ne
    br = 1.43e-27 * torch.sqrt(T) * (1.1 + 0.34 * torch.exp(-(5.5 - torch.log10(T)).pow(2) / 3.0)) * nHII * ne
    return ce + ci + re + br


def _split(ls):
    return torch.exp(ls[..., 0]), torch.exp(ls[..., 1]), torch.exp(ls[..., 2])


def dlnII_dt(T, nHI, nHII):
    """d ln nHII / dt = k1 nHI - k2 nHII (electrons = nHII). Finite in the flat regime, unlike the rate of nHII.
    The HI+HI channel (k7) is left out here: its term nHI^2 / nHII diverges as nHII -> 0 and is negligible
    on the regenerated data (tests/test_physics_terms.py)."""
    return k1(T) * nHI - k2(T) * nHII


def p4_chemistry(cur, nxt, dt=DT):
    """P4 (log form): ln nHII_next - ln nHII_cur = dt * <d ln nHII/dt>, trapezoid in time. Dimensionless."""
    T0, I0, II0 = _split(cur)
    T1, I1, II1 = _split(nxt)
    rate = 0.5 * (dlnII_dt(T0, I0, II0) + dlnII_dt(T1, I1, II1))
    return (nxt[..., 2] - cur[..., 2]) - dt * rate


def p4_chemistry_implicit(cur, nxt, dt=DT):
    """P4 variant, rate at the next state only (backward Euler)."""
    T1, I1, II1 = _split(nxt)
    return (nxt[..., 2] - cur[..., 2]) - dt * dlnII_dt(T1, I1, II1)


def energy_density(T, nHI, nHII):
    return 1.5 * K_B * (nHI + 2.0 * nHII) * T


def p6_energy(cur, nxt, dt=DT):
    """P6 (log form): ln U_next - ln U_cur = -dt * <e_cool / U>, trapezoid. Dimensionless."""
    T0, I0, II0 = _split(cur)
    T1, I1, II1 = _split(nxt)
    U0, U1 = energy_density(T0, I0, II0), energy_density(T1, I1, II1)
    rate = 0.5 * (cooling(T0, I0, II0) / U0 + cooling(T1, I1, II1) / U1)
    return (torch.log(U1) - torch.log(U0)) + dt * rate


def p3_energy_monotone(cur, nxt):
    """P3: penalty on predicted increases of the specific energy e ~ T (nHI + 2 nHII) / nH (zero for the truth)."""
    T0, I0, II0 = _split(cur)
    T1, I1, II1 = _split(nxt)
    le0 = torch.log(T0) + torch.log((I0 + 2 * II0) / (I0 + II0))
    le1 = torch.log(T1) + torch.log((I1 + 2 * II1) / (I1 + II1))
    return torch.relu(le1 - le0)


def x_equilibrium(T):
    """Ionisation equilibrium of the chemistry (C7 channel neglected): x_eq = k1 / (k1 + k2)."""
    a, b = k1(T), k2(T)
    return a / (a + b + TINY)


def p5_equilibrium(cur, nxt, dt=DT):
    """P5: x = x_eq(T) where the relaxation time is far below the step; weighted by sigma = 1 / (1 + (t_rel/dt)^2)."""
    T1, I1, II1 = _split(nxt)
    nH = I1 + II1
    t_rel = 1.0 / (nH * (k1(T1) + k2(T1)) + TINY)
    sigma = 1.0 / (1.0 + (t_rel / dt) ** 2)
    x = II1 / (nH + TINY)
    xe = x_equilibrium(T1)
    logit = lambda p: torch.log(p + TINY) - torch.log1p(-p + TINY)
    return sigma * (logit(x) - logit(xe.clamp(1e-12, 1 - 1e-12)))
