# Grackle-PINN

Neural emulator of the Grackle hydrogen chemistry and cooling network (PINN-style MLP predicting log-increments of T, nHI, nHII over one 100-yr step). Work plan, decisions and results log: [`TASKS.md`](TASKS.md). The June 2026 state behind the report is tagged `report-june2026`.

## Layout

| Path | Content |
| :-- | :-- |
| `grackle_pinn/` | package: `model.py`, `data_utils.py`, `physics.py`, `metrics.py`, `train.py` |
| `configs/` | one JSON file per training run |
| `scripts/datagen/` | Grackle driver (`.c`), IC generators, ground-truth generation; `families/` holds the family grid and kNN balancing |
| `scripts/datasets/` | dataset selection, splitting and filtering scripts |
| `scripts/plots/` | plotting scripts |
| `jobs/` | Slurm job files |
| `docs/` | `report/` (LaTeX source of the June report), `Plots_Report/` |
| `data/`, `results/`, `archive/` | git-ignored: datasets and IC lists, run outputs (`results/legacy/` = June runs), old files |

## Setup

The virtual environment lives outside the repository:

```bash
python3 -m venv ~/venvs/grackle-pinn
~/venvs/grackle-pinn/bin/pip install -r requirements.txt
```

## Data generation

The Grackle driver `scripts/datagen/cooling-full-integration-100yr.c` is built against the Grackle install of the `Lessandre` project (see `TASKS.md`, task A3, which also rebuilds the datasets). Initial conditions: `python scripts/datagen/params.py`, then `sbatch scripts/datagen/generate_data.sh`.

## Training

Choose the Slurm account and QOS as described in `TASKS.md` section 3.2, then:

```bash
sbatch --account=<acc> --qos=<qos> --export=ALL,CONFIG=configs/<run>.json jobs/train.job
# resume: add RESUME=results/runs/<run>/PINN_<run>.pth to --export
```

Run outputs (checkpoints, scaling file, loss history, copy of the config) go to `results/runs/<run_name>/`; Slurm logs to `results/logs/`. Locally: `python -m grackle_pinn.train --config configs/<run>.json` from the repository root.

## Evaluation

The evaluation harness (`grackle_pinn/evaluate.py`, free rollout from the initial condition only) is task A2 and does not exist yet. The old `visualise.py` and `ComputeMSE.py` are in `archive/`.
