"""Evaluation harness: free rollout from the initial condition only.

Rule (TASKS.md, section 0.2): every headline number comes from a free rollout in which the model sees
only the initial condition and its own previous predictions. Teacher-forced one-step errors are
computed as diagnostics only and are stored under the key "diagnostics_teacher_forced".

Usage:
    python -m grackle_pinn.evaluate run --ckpt PINN_BEST_x.pth [--scaling scaling_params_x.pth] \
        --set report|grid --label NAME [--dtype float64] [--max-per-family N] [--plot]
    python -m grackle_pinn.evaluate table results/eval/*.json > results/baseline.md
"""
import os
import re
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import torch

from grackle_pinn.model import PINN
from grackle_pinn.param import build as build_param

LN10 = np.log(10.0)
COL_T, COL_HI, COL_HII = 3, 6, 7          # columns of the Grackle output files
CLIFF_THRESHOLD = 0.05                    # |d ln T| per step that defines a cooling cliff
STEP_YR = 100.0
DEFAULT_FAMILY_CSV = "scripts/datagen/families/Trajectory_Families.csv"


# --------------------------------------------------------------------------- model
def load_model(ckpt_path, scaling_path=None, device="cpu", dtype=torch.float64):
    """Build the network from a checkpoint. New checkpoints (A6) carry architecture, scaling and parameterisation;
    legacy ones are handled by inferring width/depth from the tensors and reading the sibling scaling file."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = d["model"] if "model" in d else d
    param = build_param(d.get("param")) if "param" in d else build_param(None)
    if "arch" in d:
        hidden, hidden_layers = d["arch"]["hidden_dim"], d["arch"]["hidden_layers"]
    else:
        weights = [k for k in sd if k.endswith("weight")]
        hidden, hidden_layers = sd[weights[0]].shape[0], len(weights) - 2
    model = PINN(param.in_dim, param.out_dim, hidden_dim=hidden, hidden_layers=hidden_layers)
    model.load_state_dict(sd)
    model = model.to(device=device, dtype=dtype).eval()

    scaling = d.get("scaling") if "model" in d else None
    if scaling is None:
        if scaling_path is None:
            base = os.path.basename(ckpt_path)
            name = re.sub(r"^PINN_(BEST_)?", "", base)
            scaling_path = os.path.join(os.path.dirname(ckpt_path), f"scaling_params_{name}")
        scaling = torch.load(scaling_path, map_location="cpu", weights_only=True)
    sc = {k: scaling[k].to(device=device, dtype=dtype) for k in ("in_mean", "in_std", "tg_mean", "tg_std")}
    return model, sc, {"hidden": hidden, "hidden_layers": hidden_layers, "scaling_file": scaling_path,
                       "param": param, "param_config": param.config()}


def step(model, sc, state, param=None):
    """One model step on log-states (B, 3): returns the new log-state."""
    return (param or build_param(None)).step(model, sc, state)


@torch.no_grad()
def rollout(model, sc, state0, n_steps, param=None):
    """Free rollout. state0: (B, 3) log [T, nHI, nHII]. Returns (B, n_steps + 1, 3) on the CPU as float64.

    Only state0 and the model's own previous output are used.
    """
    param = param or build_param(None)
    out = torch.empty(state0.shape[0], n_steps + 1, 3, dtype=state0.dtype, device=state0.device)
    out[:, 0] = state0
    s = state0
    for i in range(n_steps):
        s = param.step(model, sc, s)
        out[:, i + 1] = s
    return out.cpu().double().numpy()


@torch.no_grad()
def window_rollout_score(model, param, sc, truth_ls, window, starts, cap_dex=50.0):
    """Checkpoint-selection score: free rollouts of `window` steps started at the true state of rows `starts`
    (the first state of each window is the only ground truth used). Mean |d log10| over time, the three variables,
    windows and trajectories, each element capped at `cap_dex`. Lower is better.

    truth_ls: tensor (n_traj, n_rows, 3) of ln states on the model device."""
    was_training = model.training
    model.eval()
    starts = [s for s in starts if s + window < truth_ls.shape[1]]
    s0 = torch.cat([truth_ls[:, s] for s in starts])
    pred = torch.empty(s0.shape[0], window + 1, 3, dtype=s0.dtype, device=s0.device)
    pred[:, 0] = s0
    cur = s0
    for i in range(window):
        cur = param.step(model, sc, cur)
        pred[:, i + 1] = cur
    truth = torch.cat([truth_ls[:, s:s + window + 1] for s in starts])
    err = (pred - truth).abs() / LN10
    err = torch.nan_to_num(err, nan=cap_dex, posinf=cap_dex).clamp(max=cap_dex)
    model.train(was_training)
    return err.mean().item()


@torch.no_grad()
def one_step_teacher_forced(model, sc, truth_log, param=None):
    """Diagnostic only: one-step prediction from the TRUE state. truth_log: (B, n+1, 3) -> (B, n, 3)."""
    param = param or build_param(None)
    B, n1, _ = truth_log.shape
    s = torch.from_numpy(truth_log[:, :-1].reshape(-1, 3)).to(sc["in_mean"])
    chunks = []
    for part in torch.split(s, 1 << 18):
        chunks.append((param.step(model, sc, part) - part).cpu().double())
    return torch.cat(chunks).numpy().reshape(B, n1 - 1, 3)


# --------------------------------------------------------------------------- data
def load_trajectory(path):
    """Log-state trajectory (n_rows, 3) = ln [T, nHI, nHII] from a Grackle output file."""
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None)
    return np.log(df[[COL_T, COL_HI, COL_HII]].values.astype(np.float64))


def report_set(data_dir="data/Plot_Files"):
    """The 8 trajectories behind the report figures: (name, family, path)."""
    items = []
    for kind in ("Train", "Valid"):
        for fam in range(4):
            p = os.path.join(data_dir, f"{kind}_Fam_{fam}.dat")
            if os.path.exists(p):
                items.append((f"{kind}_Fam_{fam}", fam, p))
    return items


def grid_set(grid_dir="data/families/Grid_Files", family_csv=DEFAULT_FAMILY_CSV, max_per_family=None, seed=0):
    """Trajectories of the family grid with their family label from Trajectory_Families.csv."""
    fam = pd.read_csv(family_csv).set_index("Filename")["Family_ID"].to_dict()
    rng = np.random.default_rng(seed)
    by_fam = {}
    for p in sorted(glob.glob(os.path.join(grid_dir, "output_*.dat"))):
        f = fam.get(os.path.basename(p))
        if f is not None:
            by_fam.setdefault(int(f), []).append(p)
    items = []
    for f, paths in sorted(by_fam.items()):
        if max_per_family and len(paths) > max_per_family:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_per_family, replace=False))]
        items += [(os.path.basename(p), f, p) for p in paths]
    return items


def family_oracle(family_csv=DEFAULT_FAMILY_CSV):
    """kNN (k=3) on the initial (ln T, ln nHI, ln nHII) trained on the grid families, as in the report."""
    from sklearn.neighbors import KNeighborsClassifier
    df = pd.read_csv(family_csv)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df[["T_0(log)", "nHI_0(log)", "nHII_0(log)"]].values, df["Family_ID"].values)
    return knn


def npz_set(path, max_per_family=None, seed=0):
    """Trajectories from a compact .npz (build_datasets.py). Family from the kNN oracle on the initial state.
    Returns (name, family, log-state array) items."""
    z = np.load(path)
    ls, names = z["logstate"], z["names"]
    knn = family_oracle()
    fam = knn.predict(ls[:, 0] / LN10)       # the oracle works on log10 values
    rng = np.random.default_rng(seed)
    idx = []
    for f in sorted(set(fam)):
        ii = np.where(fam == f)[0]
        if max_per_family and len(ii) > max_per_family:
            ii = np.sort(rng.choice(ii, max_per_family, replace=False))
        idx += list(ii)
    return [(str(names[i]), int(fam[i]), ls[i]) for i in idx]


# --------------------------------------------------------------------------- metrics
@np.errstate(all="ignore")
def specific_energy(logs):
    """Specific internal energy up to a constant at fixed nH: e ~ T (nHI + 2 nHII) / nH. logs (..., 3)."""
    T, HI, HII = np.exp(logs[..., 0]), np.exp(logs[..., 1]), np.exp(logs[..., 2])
    return T * (HI + 2 * HII) / (HI + HII)


def trajectory_metrics(truth, pred):
    """Metrics of one free rollout. truth, pred: (n+1, 3) natural logs of [T, nHI, nHII]."""
    err = np.abs(pred - truth) / LN10                      # |d log10| per step and variable
    finite = np.isfinite(err).all(axis=1)
    m = {"finite_fraction": float(finite.mean())}
    err = np.where(np.isfinite(err), err, np.inf)
    for j, name in enumerate(("T", "nHI", "nHII")):
        m[f"{name}_median"] = float(np.median(err[:, j]))
        m[f"{name}_p95"] = float(np.percentile(err[:, j], 95))
        m[f"{name}_final"] = float(err[-1, j])

    dT_true, dT_pred = np.diff(truth[:, 0]), np.diff(pred[:, 0])
    # cliff time = row of the largest |d ln T| after the first step, if it exceeds the threshold (A2 definition of
    # 2026-11-16). The first step is excluded: Family 3 starts with a one-step relaxation (|d ln T| up to 0.35 in the
    # report set) that is not the cooling cliff; "first row above the threshold" and "global argmax" both picked it.
    def steepest(d):
        d = np.abs(d[1:])
        return int(np.argmax(d)) + 1 if d.size and d.max() > CLIFF_THRESHOLD else None
    i_true = steepest(dT_true)
    i_pred = steepest(dT_pred) if np.isfinite(dT_pred).all() else None
    m["has_cliff"] = i_true is not None
    if m["has_cliff"]:
        pred_has = i_pred is not None
        m["cliff_time_true_yr"] = i_true * STEP_YR
        m["cliff_missed"] = not pred_has
        m["cliff_time_error_yr"] = abs(i_pred - i_true) * STEP_YR if pred_has else None
        m["cliff_time_error_rel"] = abs(i_pred - i_true) / i_true if (pred_has and i_true > 0) else None

    nH0 = np.log(np.exp(truth[0, 1]) + np.exp(truth[0, 2]))
    nH_pred = np.logaddexp(pred[:, 1], pred[:, 2])
    m["mass_violation_max"] = float(np.max(np.abs(nH_pred - nH0))) if np.isfinite(nH_pred).all() else float("inf")

    e = specific_energy(pred)
    m["T_below_1K_steps"] = int(np.sum(pred[:, 0] < 0.0))
    m["energy_increase_fraction"] = float(np.mean(np.diff(e) > 1e-12 * e[:-1])) if np.isfinite(e).all() else float("nan")
    e_true = specific_energy(truth)
    m["energy_increase_fraction_truth"] = float(np.mean(np.diff(e_true) > 1e-12 * e_true[:-1]))
    return m


def diagnostics_one_step(truth_batch, inc_pred):
    """Teacher-forced one-step RMS error of d ln T binned by decade of the true |d ln T|. Diagnostic only."""
    inc_true = np.diff(truth_batch, axis=1)
    a, b = np.abs(inc_true[..., 0]).ravel(), (inc_pred[..., 0] - inc_true[..., 0]).ravel()
    edges = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (a >= lo) & (a < hi)
        if sel.any():
            rows.append({"lo": lo, "hi": None if hi == np.inf else hi, "count": int(sel.sum()),
                         "rms_error": float(np.sqrt(np.mean(b[sel] ** 2))),
                         "mean_error": float(np.mean(b[sel]))})
    return rows


def aggregate(per_traj):
    """Median over trajectories of each scalar metric (inf = blown up, NaN values are skipped), plus counts."""
    out = {"n": len(per_traj)}
    keys = sorted({k for r in per_traj for k in r} - {"name", "family", "has_cliff", "cliff_missed"})
    for k in keys:
        vals = [r[k] for r in per_traj if r.get(k) is not None and not np.isnan(r[k])]
        if vals:
            out[k] = float(np.median(vals))
    cl = [r for r in per_traj if r["has_cliff"]]
    out["n_cliff"] = len(cl)
    out["n_cliff_missed"] = sum(1 for r in cl if r["cliff_missed"])
    return out


# --------------------------------------------------------------------------- driver
def evaluate(ckpt, scaling, items, device, dtype, label, set_name, max_steps=None, plot_dir=None):
    model, sc, info = load_model(ckpt, scaling, device, dtype)
    trajs = {}
    for name, fam, path in items:
        tr = load_trajectory(path) if isinstance(path, str) else path
        if max_steps:
            tr = tr[: max_steps + 1]
        trajs[name] = (fam, tr)
    # batch by length (all files should have equal length; this keeps it safe)
    by_len = {}
    for name, (fam, tr) in trajs.items():
        by_len.setdefault(len(tr), []).append(name)
    per_traj, diag_rows = [], []
    for n_rows, names in by_len.items():
        truth = np.stack([trajs[n][1] for n in names])
        state0 = torch.from_numpy(truth[:, 0]).to(device=device, dtype=dtype)
        pred = rollout(model, sc, state0, n_rows - 1, info["param"])
        for i, n in enumerate(names):
            m = trajectory_metrics(truth[i], pred[i])
            m.update(name=n, family=trajs[n][0])
            per_traj.append(m)
            if plot_dir:
                plot_rollout(n, truth[i], pred[i], os.path.join(plot_dir, f"{label}_{n}.png"))
        diag_rows.append(diagnostics_one_step(truth, one_step_teacher_forced(model, sc, truth, info["param"])))
    fams = sorted({r["family"] for r in per_traj})
    return {
        "label": label, "set": set_name, "ckpt": ckpt, "dtype": str(dtype).replace("torch.", ""),
        "architecture": {k: v for k, v in info.items() if k != "param"}, "rollout_free": True,
        "overall": aggregate(per_traj),
        "per_family": {str(f): aggregate([r for r in per_traj if r["family"] == f]) for f in fams},
        "per_trajectory": per_traj,
        "diagnostics_teacher_forced": diag_rows[0] if len(diag_rows) == 1 else diag_rows,
    }


def plot_rollout(name, truth, pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(truth)) * STEP_YR / 1e6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, j, title in zip(ax, range(3), ("T [K]", "nHI [cm^-3]", "nHII [cm^-3]")):
        a.plot(t, np.exp(truth[:, j]), "k--", label="Grackle")
        a.plot(t, np.exp(np.clip(pred[:, j], -700, 700)), label="free rollout")
        a.set_yscale("log"); a.set_xlabel("t [Myr]"); a.set_title(f"{name}: {title}")
    ax[0].legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def fmt(x, f="{:.3g}"):
    if x is None:
        return "-"
    return f.format(x)


def make_table(paths):
    """Markdown table from result JSON files (one row per label and set, overall + per family)."""
    rows = [json.load(open(p)) for p in paths]
    lines = ["| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["set"], r["label"], r["dtype"])):
        pt = r["per_trajectory"]
        parts = [("all", aggregate(pt))] + [(str(f), aggregate([x for x in pt if x["family"] == f]))
                                            for f in sorted({x["family"] for x in pt})]
        for fam, a in parts:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                r["label"], r["set"], r["dtype"], fam, a["n"], fmt(a.get("T_median")), fmt(a.get("T_p95")), fmt(a.get("T_final")),
                fmt(a.get("nHI_final")), fmt(a.get("nHII_final")), fmt(a.get("cliff_time_error_yr")),
                a["n_cliff_missed"], a["n_cliff"], fmt(a.get("mass_violation_max")), fmt(a.get("energy_increase_fraction"))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--scaling", default=None)
    r.add_argument("--set", default="report", help="report | grid | npz:<file.npz> (label of the set = file stem)")
    r.add_argument("--label", required=True)
    r.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    r.add_argument("--max-per-family", type=int, default=None)
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--out-dir", default="results/eval")
    r.add_argument("--plot", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    args = ap.parse_args()

    if args.cmd == "table":
        print(make_table(args.files))
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    if args.set == "report":
        items = report_set()
    elif args.set == "grid":
        items = grid_set(max_per_family=args.max_per_family)
    else:
        items = npz_set(args.set[4:], args.max_per_family)
        args.set = os.path.splitext(os.path.basename(args.set))[0]
    res = evaluate(args.ckpt, args.scaling, items, device, dtype, args.label, args.set, args.max_steps,
                   os.path.join(args.out_dir, "plots") if args.plot else None)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.label}__{args.set}__{args.dtype}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
