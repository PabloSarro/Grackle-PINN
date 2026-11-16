"""Regenerates every table of results/baseline.md from results/eval/*.json (keeps the text around them).
Each table is rebuilt from the files that match the heading it sits under:
  legacy report/float64, legacy grid, legacy report/float32, and all non-legacy runs on `report` and `refB_valid`.
Usage: python scripts/make_baseline.py
"""
import glob, os, sys
sys.path.insert(0, os.getcwd())
from grackle_pinn.evaluate import make_table

def files(legacy, tag):
    allf = sorted(glob.glob(f"results/eval/*__{tag}.json"))
    return [f for f in allf if os.path.basename(f).startswith("legacy_") == legacy]

SECTIONS = [("Report-figure set (8 trajectories: Train", True, "report__float64"),
            ("Family grid subset", True, "grid__float64"),
            ("Report-figure set, float32", True, "report__float32"),
            ("Report-figure set (8 trajectories), float64", False, "report__float64"),
            ("Per-family balanced validation set", False, "refB_valid__float64")]

lines = open("results/baseline.md").read().split("\n")
out, i, current = [], 0, None
while i < len(lines):
    ln = lines[i]
    if ln.startswith("#"):
        current = next((s for s in SECTIONS if ln.lstrip("# ").startswith(s[0])), None)
    if ln.startswith("| run |") and current:
        while i < len(lines) and lines[i].startswith("|"):
            i += 1
        out.append(make_table(files(current[1], current[2])).rstrip("\n"))
        continue
    out.append(ln); i += 1
open("results/baseline.md", "w").write("\n".join(out))
