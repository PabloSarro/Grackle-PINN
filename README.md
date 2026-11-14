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

```bash
bash scripts/datagen/build_driver.sh                      # builds the Grackle driver (needs ../../Lessandre/grackle-install)
sbatch --account=<acc> --qos=<qos> jobs/generate_ground_truth.job   # 22,633 trajectories to /scratch/izar/sarro/ground_truth (about 5 min on 30 cores)
sbatch --account=<acc> --qos=<qos> jobs/build_datasets.job          # checks them and writes data/datasets/*.npz (REF-A, REF-B, report8)
```

Initial-condition lists are in `configs/ics/` (`grid` 5,625, `balanced_train`/`balanced_valid` 800/200, `random15000`, `report8`); `scripts/datagen/families/ICs_grid.py` regenerates the grid list. The REF-A selection is in `configs/datasets/refA_selection.json`. Training reads the `.npz` files (`train_path`/`valid_path` in a config may be a `.npz`).

## Training

Choose the Slurm account and QOS as described in `TASKS.md` section 3.2, then:

```bash
sbatch --account=<acc> --qos=<qos> --export=ALL,CONFIG=configs/<run>.json jobs/train.job
# resume: add RESUME=results/runs/<run>/PINN_<run>.pth to --export
```

Run outputs (checkpoints, scaling file, loss history, copy of the config) go to `results/runs/<run_name>/`; Slurm logs to `results/logs/`. Locally: `python -m grackle_pinn.train --config configs/<run>.json` from the repository root.

Config keys are listed in `DEFAULTS` of `grackle_pinn/train.py` (parameterisation `report` or `constrained`, ramp, selection, seed, dtype). Resumed runs reproduce uninterrupted ones exactly. Tests: `sbatch --account=<acc> --qos=debug jobs/test.job`.

## Evaluation

`python -m grackle_pinn.evaluate run --ckpt <checkpoint> --label <name> --set report|grid|npz:<file>` runs free rollouts from the initial condition only and writes metrics to `results/eval/`; `... table results/eval/*.json` prints the Markdown table (`results/baseline.md`). The old `visualise.py` and `ComputeMSE.py` are in `archive/`.
