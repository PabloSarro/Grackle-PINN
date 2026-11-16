"""CPU tests for A4 (parameterisations) and A6 (resume). Run from the repo root:
    python tests/test_param_and_resume.py
"""
import json, os, subprocess, sys, tempfile
import numpy as np, torch

sys.path.insert(0, os.getcwd())
from grackle_pinn.param import Constrained, Report, LogRate
from grackle_pinn.model import PINN

torch.set_default_dtype(torch.float64)
z = np.load("data/datasets/report8.npz")["logstate"]
ls = torch.from_numpy(z[:, ::250].reshape(-1, 3))       # 8 trajectories, 100 states each


def test_roundtrip():
    c = Constrained()
    back = c.from_z(c.to_z(ls))
    err = (back - ls).abs().max().item()
    print("round trip |d ln| max", err); assert err < 1e-12


def test_targets_reproduce_next_state():
    c = Constrained()
    cur, nxt = torch.from_numpy(z[:, :-1].reshape(-1, 3)), torch.from_numpy(z[:, 1:].reshape(-1, 3))
    t = c.targets(cur, nxt)
    rec = c.advance(cur, t)
    err = (rec - nxt).abs().max().item()
    print("advance(targets) vs next state, max |d ln|", err); assert err < 1e-9
    assert (t[:, 0] <= 0).all(), "energy increment should be <= 0 for the ground truth"


def test_hard_constraints_for_any_network():
    c = Constrained(lmax=25.0)
    torch.manual_seed(1)
    m = PINN(3, 2, hidden_dim=32, hidden_layers=2)
    sc = {"in_mean": torch.zeros(3), "in_std": torch.ones(3), "tg_mean": torch.zeros(2), "tg_std": torch.ones(2)}
    s = ls.clone()
    e0 = torch.exp(c.to_z(s)[:, 0]); nH0 = torch.exp(s[:, 1]) + torch.exp(s[:, 2])
    worst_mass, e_up = 0.0, 0
    for _ in range(50):
        for p in m.parameters():                              # a wild, untrained network each time
            p.data = torch.randn_like(p) * 3
        s2 = c.step(m, sc, s)
        nH = torch.exp(s2[:, 1]) + torch.exp(s2[:, 2])
        worst_mass = max(worst_mass, (nH / nH0 - 1).abs().max().item())
        e1 = torch.exp(c.to_z(s2)[:, 0])
        e_up += int((e1 > e0 * (1 + 1e-12)).sum())
        assert torch.isfinite(s2).all()
    print("worst mass violation", worst_mass, "energy increases", e_up)
    assert worst_mass < 1e-13 and e_up == 0


def test_lograte():
    c = LogRate()
    cur, nxt = torch.from_numpy(z[:, :-1].reshape(-1, 3)), torch.from_numpy(z[:, 1:].reshape(-1, 3))
    t = c.targets(cur, nxt)
    sc = {"in_mean": torch.zeros(3), "in_std": torch.ones(3), "tg_mean": torch.zeros(2), "tg_std": torch.ones(2)}
    rec = c.advance(cur, c.to_physical(t / (1 + 1e-8), sc))   # inverse of the target transform (the loader divides by std + 1e-8)
    err = (rec - nxt).abs().max().item()
    print("lograte: advance(inverse(targets)) vs next state, max |d ln|", err,
          "| target ranges", t.min(0).values.tolist(), t.max(0).values.tolist()); assert err < 1e-9
    m = PINN(3, 2, hidden_dim=32, hidden_layers=2)
    s = ls.clone(); e0 = torch.exp(c.to_z(s)[:, 0]); nH0 = torch.exp(s[:, 1]) + torch.exp(s[:, 2])
    worst, up = 0.0, 0
    for _ in range(50):
        for p in m.parameters():
            p.data = torch.randn_like(p) * 3
        s2 = c.step(m, sc, s)
        assert torch.isfinite(s2).all()
        worst = max(worst, (torch.exp(s2[:, 1]) + torch.exp(s2[:, 2]) - nH0).abs().div(nH0).max().item())
        up += int((torch.exp(c.to_z(s2)[:, 0]) > e0 * (1 + 1e-12)).sum())
    print("lograte: worst mass violation", worst, "energy increases", up); assert worst < 1e-13 and up == 0


def test_report_unchanged():
    r = Report()
    cur, nxt = torch.from_numpy(z[:, :-1].reshape(-1, 3)), torch.from_numpy(z[:, 1:].reshape(-1, 3))
    assert torch.equal(r.targets(cur, nxt), nxt - cur) and torch.equal(r.features(cur), cur)


def test_resume_reproduces_uninterrupted():
    with tempfile.TemporaryDirectory() as d:
        base = dict(train_path="data/datasets/report8.npz", valid_path="data/datasets/report8.npz", out_dir=d,
                    hidden_dim=32, hidden_layers=2, epochs=6, check_every=2, batch_size=8192, seed=3,
                    rollout_trajectories=4, rollout_window=100, rollout_starts=[0, 1000], weight_factor=20.0)
        for param, extra in (("report", {}), ("constrained", {}), ("lograte", {}),
                             ("lograte", {"phys_weight": 1.0}), ("lograte", {"weight_physical": True, "weight_factor": 20.0})):
            tagname = param + "".join(sorted(extra)).replace("_", "")
            ws = {}
            for tag, stop in (("full", None), ("part", 3)):
                cfg = dict(base, **extra, run_name=f"{tagname}_{tag}", parameterisation=param, stop_after_epochs=stop)
                path = os.path.join(d, f"{cfg['run_name']}.json"); json.dump(cfg, open(path, "w"))
                subprocess.run([sys.executable, "-m", "grackle_pinn.train", "--config", path], check=True, capture_output=True)
            ck = os.path.join(d, f"{tagname}_part", f"PINN_{tagname}_part.pth")
            assert torch.load(ck, weights_only=False)["epoch"] == 3
            cfg = dict(base, **extra, run_name=f"{tagname}_part", parameterisation=param)
            path = os.path.join(d, "resume.json"); json.dump(cfg, open(path, "w"))
            subprocess.run([sys.executable, "-m", "grackle_pinn.train", "--config", path, "--model", ck], check=True, capture_output=True)
            a = torch.load(os.path.join(d, f"{tagname}_full", f"PINN_{tagname}_full.pth"), weights_only=False)["model"]
            b = torch.load(os.path.join(d, f"{tagname}_part", f"PINN_{tagname}_part.pth"), weights_only=False)["model"]
            diff = max((a[k] - b[k]).abs().max().item() for k in a)
            lines = open(os.path.join(d, f"{tagname}_part", f"losses_{tagname}_part.txt")).read().strip().splitlines()
            print(tagname, "resumed vs uninterrupted: max weight difference", diff, "| loss file lines", len(lines))
            assert diff == 0.0 and len(lines) == 7      # header + 6 epochs, appended not overwritten


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
