"""Checks every regenerated trajectory: rows, 100-yr spacing, mass and energy consistency, finite values.
Usage: python scripts/datasets/check_ground_truth.py <dir> [<dir> ...]"""
import os, sys, glob, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd


def check(path):
    d = pd.read_csv(path, comment="#", sep=r"\s+", header=None).values.astype(np.float64)
    t, dt_col, T, HI, HII, E = d[:, 1], d[:, 2], d[:, 3], d[:, 6], d[:, 7], d[:, 12]
    nH = HI + HII
    return dict(f=os.path.basename(path), rows=len(d), finite=bool(np.isfinite(d).all()),
                dt_err=float(np.abs(np.diff(t) - 100.0).max()), dtcol_err=float(np.abs(dt_col[1:] - 100.0).max()),
                mass=float(np.abs(nH / nH[0] - 1).max()), e_up=int((np.diff(E) > 1e-12 * E[:-1]).sum()),
                x_err=float(np.abs(d[:, 13] - HII / nH).max()) if d.shape[1] > 13 else None)


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(folder, "*.dat")))
        with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))) as ex:
            r = list(ex.map(check, files, chunksize=16))
        bad = [x for x in r if x["rows"] != 25001 or not x["finite"]]
        print(f"{folder}: {len(r)} files; bad rows/non-finite: {len(bad)}; max |dt-100| = {max(x['dt_err'] for x in r):.2e} yr; "
              f"max dt column error {max(x['dtcol_err'] for x in r):.2e}; max mass drift {max(x['mass'] for x in r):.2e}; "
              f"files with increasing energy: {sum(x['e_up'] > 0 for x in r)} (rows {sum(x['e_up'] for x in r)}); "
              f"max |x - nHII/nH| {max(x['x_err'] for x in r):.2e}", flush=True)
        if bad:
            print("  examples:", bad[:3])
