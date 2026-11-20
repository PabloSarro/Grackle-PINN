"""B3 unit tests for the physics residuals (grackle_pinn/physics_terms.py). CPU, float64.
Run from the repo root:  python tests/test_physics_terms.py
Checks: (1) consistency with the regenerated Grackle output (truth gives small residuals), (2) sensitivity (a drift
of the size that the report's flat regime accumulates is detected), (3) sign and monotonicity, (4) gradients,
(5) agreement with the parameterisation (targets reproduce the truth residual exactly).
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.getcwd())
import grackle_pinn.physics_terms as P
from grackle_pinn.param import Constrained, LogRate, Report

torch.set_default_dtype(torch.float64)
z = np.load("data/datasets/report8.npz")["logstate"]
cur = torch.from_numpy(z[:, :-1].reshape(-1, 3))
nxt = torch.from_numpy(z[:, 1:].reshape(-1, 3))
T0, I0, II0 = [torch.exp(cur[:, i]) for i in range(3)]
T1, I1, II1 = [torch.exp(nxt[:, i]) for i in range(3)]
DT = P.DT


def med(x):
    return float(x.abs().median())


def p99(x):
    return float(torch.quantile(x.abs(), 0.99))


def test_rates_finite_and_positive():
    T = torch.logspace(0, 9, 400)
    for k in (P.k1(T), P.k2(T), P.k7(T)):
        assert torch.isfinite(k).all() and (k > 0).all()
    print("rates: finite and positive on T in [1, 1e9] K")


def test_truth_is_consistent():
    r4 = P.p4_chemistry(cur, nxt)
    r6 = P.p6_energy(cur, nxt)
    r3 = P.p3_energy_monotone(cur, nxt)
    print(f"truth: P4 median {med(r4):.2e} p99 {p99(r4):.2e} | P6 median {med(r6):.2e} p99 {p99(r6):.2e} | "
          f"P3 max {float(r3.max()):.1e}")
    assert med(r4) < 1e-7 and p99(r4) < 1e-5, "chemistry residual on the truth is too large"
    assert med(r6) < 1e-7 and p99(r6) < 1e-5, "energy residual on the truth is too large"
    assert float(r3.max()) == 0.0, "the truth must not increase the specific energy"


def test_flat_regime_sensitivity():
    """A per-step bias of 6e-6 in ln nHII (the size of the report's flat-regime drift) must stand out."""
    base = P.p4_chemistry(cur, nxt)
    pert = nxt.clone(); pert[:, 2] += 6e-6
    r = P.p4_chemistry(cur, pert) - base
    # the residual moves by -6e-6 plus the recombination feedback (dt k2 nHII), which is tiny in the flat regime
    assert abs(float(r.median()) - 6e-6) < 1e-3 * 6e-6, "P4 must respond to ln nHII with slope +1 in the flat regime"
    flat = (II1 - II0).abs() < 1e-3 * (I1 + II1)
    print(f"sensitivity: flat steps {int(flat.sum())}; median truth residual there {med(base[flat]):.2e}, "
          f"bias 6e-6 gives residual {6e-6:.1e} ({6e-6 / max(med(base[flat]), 1e-30):.0f} x the truth median)")
    assert 6e-6 > 100 * med(base[flat]), "a flat-regime bias of 6e-6 should be far above the truth residual"


def test_energy_sign_and_sensitivity():
    pert = nxt.clone(); pert[:, 0] += 1e-4                         # hotter than the truth: energy rises
    r = P.p6_energy(cur, pert) - P.p6_energy(cur, nxt)
    # ln U moves by exactly the change in ln T; the cooling term also depends on T, so only the median is fixed
    assert abs(float(r.median()) - 1e-4) < 2e-5
    print(f"energy: +1e-4 in ln T shifts P6 by exactly {float(r.median()):.2e}")


def test_monotone_penalty_detects_increase():
    pert = nxt.clone(); pert[:, 0] += 0.01
    r3 = P.p3_energy_monotone(cur, pert)
    assert float(r3.mean()) > 1e-3 and float((r3 > 0).float().mean()) > 0.9


def test_gradients_finite():
    x = nxt.clone().requires_grad_(True)
    for f in (P.p4_chemistry, P.p6_energy, P.p3_energy_monotone, P.p5_equilibrium):
        out = f(cur[:200], x[:200])
        g, = torch.autograd.grad(out.pow(2).mean(), x, retain_graph=True)
        assert torch.isfinite(g).all(), f.__name__
    print("gradients: finite for P3, P4, P5, P6")


def test_targets_reproduce_truth_residual():
    """Through the parameterisations: advance(targets) returns the truth, so the residuals equal the truth residuals."""
    for par in (Report(), Constrained(lmax=25.0), LogRate(lo=(-40, -40), hi=(40, 40))):
        t = par.targets(cur, nxt)
        if isinstance(par, LogRate):
            t = par.to_physical(t / (1 + 1e-8), {"tg_mean": torch.zeros(2), "tg_std": torch.ones(2)})
        pred = par.advance(cur, t)
        assert float((pred - nxt).abs().max()) < 1e-6, par.name
    print("parameterisations: advance(targets) reproduces the next state (report, constrained, lograte)")


def test_p5_where_equilibrium_holds():
    nH = I1 + II1
    t_rel = 1.0 / (nH * (P.k1(T1) + P.k2(T1)))
    fast = t_rel < 0.05 * DT
    r5 = P.p5_equilibrium(cur, nxt)
    print(f"P5: fast-relaxation steps {int(fast.sum())}; median |residual| there {med(r5[fast]) if fast.any() else float('nan'):.2e}")
    if fast.any():
        assert med(r5[fast]) < 1e-3, "equilibrium term must be near zero on the truth where relaxation is fast"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name, flush=True)
