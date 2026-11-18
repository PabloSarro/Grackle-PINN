"""B7 step 1: dataset builders for the sampling strategies S3 to S6 (S1, S2 = REF-A, REF-B already exist).

Every strategy draws its training pairs from the same pool of trajectories and has the same budget of pairs
(default 20M, the size of REF-B), so differences come from the choice of pairs, not from the amount of data.

Pool: `--pool-size` random trajectories from `random15000` (seed 0), excluding every file that is in the REF-A or
REF-B validation sets. Pairs are (ln state_t, ln state_t+1) of one 100-yr step; activity a = max_j |d ln y_j|.

Strategies (output `.npz` with `cur` and `nxt`, (N, 3) float64; read by grackle_pinn.data_utils):
  S3  uniform random: whole trajectories drawn at random until the budget is reached (control).
  S4  per-decade balance: equal number of pairs in each decade of activity (log10 a in [-12, 1]).
  S5  state-space coverage: equal number of pairs in each cell of a grid in (ln T, logit x, ln nH).
  S6  corrected equilibrium filter: pairs with a < 1e-7 are kept with probability 1/100, then uniform sampling.
Usage: python scripts/datasets/sampling.py --gt /scratch/izar/sarro/ground_truth --out data/datasets
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

EQ_THRESHOLD = 1e-7          # S6: activity below this is "equilibrium"
EQ_KEEP = 0.01               # S6: fraction of equilibrium pairs kept
DECADES = np.arange(-12, 2)  # S4 bin edges in log10 activity
STATE_BINS = 12              # S5: bins per axis


def pairs_of(ls):
    """ls: (n_rows, 3) log-states -> cur, nxt (n_rows - 1, 3)."""
    return ls[:-1], ls[1:]


def activity(cur, nxt):
    return np.abs(nxt - cur).max(axis=1)


def state_cell(cur):
    """Cell index of each pair in a grid over (ln T, logit x, ln nH), with x = nHII / (nHI + nHII)."""
    lT, lI, lII = cur[:, 0], cur[:, 1], cur[:, 2]
    logit = lII - lI
    lnH = np.logaddexp(lI, lII)
    axes = [lT, logit, lnH]
    idx = []
    for a, lo, hi in zip(axes, (2.0, -12.0, -6.0), (9.0, 12.0, 4.0)):
        b = np.clip(((a - lo) / (hi - lo) * STATE_BINS).astype(int), 0, STATE_BINS - 1)
        idx.append(b)
    return np.ravel_multi_index(idx, (STATE_BINS,) * 3)


def sample_budget(rng, groups, budget):
    """Equal share per group (a list of index arrays), uniform within each group; groups smaller than the share
    contribute all their pairs and the rest of the budget goes to the other groups."""
    groups = [g for g in groups if g.size]
    out = []
    remaining = budget
    order = sorted(range(len(groups)), key=lambda i: groups[i].size)
    for k, i in enumerate(order):
        share = remaining // (len(order) - k)
        g = groups[i]
        if g.size <= share:
            out.append(g)
            remaining -= g.size
        else:
            out.append(rng.choice(g, share, replace=False))
            remaining -= share
    return np.concatenate(out)


def build_all(cur, nxt, traj_id, budget, rng):
    """Returns {name: index array into the pool pairs}. traj_id: trajectory index of each pair."""
    act = activity(cur, nxt)
    sets = {}
    # S3: whole trajectories at random until the budget is reached
    order = rng.permutation(int(traj_id.max()) + 1)
    counts = np.bincount(traj_id, minlength=int(traj_id.max()) + 1)
    chosen, total = [], 0
    for t in order:
        if total >= budget:
            break
        chosen.append(t); total += counts[t]
    pool_idx = np.flatnonzero(np.isin(traj_id, chosen))
    sets["S3"] = rng.choice(pool_idx, min(budget, pool_idx.size), replace=False)
    # S4: equal pairs per decade of activity
    dec = np.clip(np.floor(np.log10(np.maximum(act, 1e-300))).astype(int), DECADES[0], DECADES[-1] - 1) - DECADES[0]
    groups = [np.flatnonzero(dec == k) for k in range(len(DECADES) - 1)]
    sets["S4"] = sample_budget(rng, groups, budget)
    # S5: equal pairs per state-space cell
    cell = state_cell(cur)
    groups = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    sets["S5"] = sample_budget(rng, groups, budget)
    # S6: thin out equilibrium pairs, then uniform sampling
    keep = (act >= EQ_THRESHOLD) | (rng.random(act.size) < EQ_KEEP)
    cand = np.flatnonzero(keep)
    sets["S6"] = rng.choice(cand, min(budget, cand.size), replace=False)
    return sets


def save_pairs(path, cur, nxt, idx, extra=None):
    np.savez(path, cur=cur[idx], nxt=nxt[idx], **(extra or {}))
    print(f"{path}: {idx.size} pairs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/scratch/izar/sarro/ground_truth")
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import json
    from concurrent.futures import ProcessPoolExecutor
    from scripts.datasets.build_datasets import load_log_state

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.gt, "random15000", "output_*.dat")))
    excluded = set()
    for name in ("refA_valid", "refB_valid"):
        excluded |= set(np.load(os.path.join(args.out, f"{name}.npz"))["names"].tolist())
    files = [f for f in files if os.path.basename(f) not in excluded]
    pool = [files[i] for i in rng.choice(len(files), args.pool_size, replace=False)]
    print(f"pool: {len(pool)} trajectories from {len(files)} eligible random ICs", flush=True)

    cur_l, nxt_l, tid_l = [], [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
        for t, ls in enumerate(ex.map(load_log_state, pool, chunksize=8)):
            c, n = pairs_of(ls)
            cur_l.append(c); nxt_l.append(n); tid_l.append(np.full(c.shape[0], t, dtype=np.int32))
    cur, nxt, tid = np.concatenate(cur_l), np.concatenate(nxt_l), np.concatenate(tid_l)
    del cur_l, nxt_l, tid_l
    print(f"pool pairs: {cur.shape[0]}", flush=True)

    sets = build_all(cur, nxt, tid, args.budget, rng)
    os.makedirs(args.out, exist_ok=True)
    for name, idx in sets.items():
        save_pairs(os.path.join(args.out, f"sampling_{name}.npz"), cur, nxt, idx)
        a = activity(cur[idx], nxt[idx])
        print(f"  {name}: median activity {np.median(a):.3g}, fraction below {EQ_THRESHOLD:g} {np.mean(a < EQ_THRESHOLD):.3f}, "
              f"trajectories {np.unique(tid[idx]).size}", flush=True)
    with open(os.path.join("configs", "datasets", "sampling_pool.json"), "w") as f:
        json.dump({"pool_files": [os.path.basename(p) for p in pool], "seed": args.seed, "budget": args.budget,
                   "eq_threshold": EQ_THRESHOLD, "eq_keep": EQ_KEEP}, f)


if __name__ == "__main__":
    main()
