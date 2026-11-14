"""Builds the compact training/validation arrays (.npz, float64) from the regenerated ground truth.

Each .npz holds `logstate` (n_traj, n_rows, 3) = ln [T, nHI, nHII] and `names`. The training code pairs
adjacent rows (one 100-yr step) exactly like the text-file loader, so both give the same pairs.

Datasets (A3):
  REF-B   balanced_train / balanced_valid: kNN-balanced, 250 per family (800 train, 200 valid).
  REF-A   variance-stratified selection from the 15,000 random ICs (re-implemented from the report):
          score = sum over T, nHI, nHII of Var[ d ln y ] of the file; 800 most active, 250 moderate,
          250 flat, split 700/150/150 train and 100/100/100 validation.
          The June boundaries between "moderate" and "flat" are not recorded. Choice made here:
          moderate = random draw among the files ranked 801..4000 by score, flat = random draw among
          the files ranked below 4000 (seed 0).
  eval    report8 and balanced_valid copied as arrays under /home (scratch is purged).

Usage: python scripts/datasets/build_datasets.py [--gt /scratch/izar/sarro/ground_truth] [--out data/datasets]
"""
import os
import glob
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

COLS = (3, 6, 7)


def load_log_state(path):
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, usecols=COLS)
    return np.log(df[list(COLS)].values.astype(np.float64))


def score_file(path):
    ls = load_log_state(path)
    return path, float(np.var(np.diff(ls, axis=0), axis=0).sum()), ls.shape[0]


def save(out_dir, name, paths, n_rows=25001):
    states, names = [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        for p, ls in zip(paths, ex.map(load_log_state, paths, chunksize=8)):
            if ls.shape[0] != n_rows:
                print(f"  skipped {p}: {ls.shape[0]} rows")
                continue
            states.append(ls)
            names.append(os.path.basename(p))
    arr = np.stack(states)
    np.savez(os.path.join(out_dir, f"{name}.npz"), logstate=arr, names=np.array(names))
    print(f"{name}: {arr.shape} ({arr.nbytes / 1e6:.0f} MB)", flush=True)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/scratch/izar/sarro/ground_truth")
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--selection", default="configs/datasets/refA_selection.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.selection), exist_ok=True)
    ls = lambda sub: sorted(glob.glob(os.path.join(args.gt, sub, "output_*.dat")))

    # REF-B and evaluation sets
    save(args.out, "refB_train", ls("balanced_train"))
    save(args.out, "refB_valid", ls("balanced_valid"))
    save(args.out, "report8", sorted(glob.glob(os.path.join(args.gt, "report8", "*.dat"))))

    # REF-A: variance stratification of the random ICs
    files = ls("random15000")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        scored = list(ex.map(score_file, files, chunksize=32))
    scored = [(p, s) for p, s, n in scored if n == 25001 and np.isfinite(s)]
    scored.sort(key=lambda x: -x[1])
    rng = np.random.default_rng(0)
    active = [p for p, _ in scored[:800]]
    mid = [p for p, _ in scored[800:4000]]
    low = [p for p, _ in scored[4000:]]
    moderate = [mid[i] for i in rng.choice(len(mid), 250, replace=False)]
    flat = [low[i] for i in rng.choice(len(low), 250, replace=False)]
    train, valid = [], []
    for group, n_tr in ((active, 700), (moderate, 150), (flat, 150)):
        order = rng.permutation(len(group))
        group = [group[i] for i in order]
        train += group[:n_tr]
        valid += group[n_tr:n_tr + 100]
    sc = dict(scored)
    sel = {"score_definition": "sum over T, nHI, nHII of Var[d ln y]", "n_scored": len(scored),
           "train": [[os.path.basename(p), sc[p]] for p in train], "valid": [[os.path.basename(p), sc[p]] for p in valid]}
    with open(args.selection, "w") as f:
        json.dump(sel, f)
    print(f"REF-A selection: {len(train)} train, {len(valid)} valid; score range active "
          f"{scored[799][1]:.3g}..{scored[0][1]:.3g}, rank 4000 {scored[3999][1]:.3g}, last {scored[-1][1]:.3g}")
    save(args.out, "refA_train", train)
    save(args.out, "refA_valid", valid)


if __name__ == "__main__":
    main()
