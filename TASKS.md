# TASKS — improving the Grackle-PINN results

Living record of the improvement work on branch `improvements-nov2026`. It lists what has been done, what is in progress and what is left, with the outcome of every step.

## 0. How this file works

### 0.1 Status markers
- `[X]` done
- `[~]` in progress or partly done
- `[-]` not started

### 0.2 Rules for whoever continues this work
- Work on one task ID (`A1`, `B1`, ...) at a time, on Pablo's command: implement, test, commit, push, report, then wait for confirmation.
- Commits are authored by the configured git user only; no co-author lines.
- Update this file in the same commit as the work it describes:
  - flip the markers of the task and of its steps;
  - add a dated line to the task's **Log** for every thing tried, including failed attempts, ablations, small configuration tests and fine-tunings;
  - add every training run to section 8 (run registry), even aborted ones;
  - add the metrics table row to `results/baseline.md` once A2 exists.
- Log line format: `YYYY-MM-DD | what was tried | config / commit / run id | result with numbers | decision taken`.
- Never delete a log line. If a result is later found wrong, add a new line that says so.
- Evidence labels: **[measured]** a check was run and the number is reported; **[read]** taken from code, logs or the report; **[inferred]** interpretation, to be confirmed by the task that tests it.

### 0.3 Status at a glance
- Review and planning
  - `[X]` R1 Read the report
  - `[X]` R2 Review the repository code and git state
  - `[X]` R3 Check the ground-truth data files
  - `[X]` R4 Test the density-time scaling
  - `[X]` R5 Test the analytic rates against the data
  - `[X]` R6 Review `../../Lessandre` and `../../OldFiles`
  - `[X]` R7 Check cluster accounts
- Quick wins
  - `[-]` A1 Simplify the repository structure
  - `[-]` A2 Evaluation harness and numeric baseline
  - `[-]` A3 Clean ground-truth generation
  - `[-]` A4 State variables with built-in constraints
  - `[-]` A5 Target and loss that resolve the flat regime
  - `[-]` A6 Training-loop fixes
- Larger experiments
  - `[-]` B1 Flow map with scaled time as input
  - `[-]` B2 Rollout-aware training
  - `[-]` B3 Physics-informed terms that respect stiffness
  - `[-]` B4 Reference baselines and speed
  - `[-]` B5 Report future-work items, re-assessed
  - `[-]` B6 Scope extensions
- Next task to start: **A1** (not started; waiting for Pablo's go-ahead).

## 1. Baseline: what the report states

### 1.1 Method
- Hydrogen-only Grackle network (`primordial_chemistry=1`, He = 0, no UV background, no heating), 2.5 Myr per trajectory, output every 100 yr (25,000 rows).
- MLP 12 x 512, tanh, float64; input `[ln T, ln nHI, ln nHII]`, output the three log-increments over one 100-yr step, both standardised.
- Loss: weighted MSE `(1 + 20|y|)(y_pred - y)^2` plus a log-space mass-conservation penalty, weight ramped linearly from 0.
- Model A: variance-stratified dataset (25.0M points). Model B: kNN-balanced dataset, 250 trajectories per family (20.0M points).

### 1.2 Results (all qualitative; the report contains no error number)
- Model A: follows Family 3 (Fig. 3 top, Fig. 5); cooling cliff about 0.15 Myr early on another Family 3 case (Fig. 6, roughly 2.10 vs 2.25 Myr); invents cooling on Families 0, 1, 2 (Fig. 3 bottom ends near 1e-28 K; Fig. 7 drifts +0.07 dex in T).
- Model B: straight log-linear drift on every family (Fig. 4), about 2 dex off in T at 2.5 Myr on Family 3.

### 1.3 Limitations and future work stated in the report
- "Activity bias" (Model A) and "equilibrium bias" (Model B), both attributed to the active/flat balance of the training set.
- Rate-equation and energy-equation physics losses abandoned (training instability; rates not matching finite differences).
- Future work: 4.1 equilibrium filter, 4.2 Mixture of Experts per family, 4.3 time-adaptive data with `dt` as input.

## 2. Decisions taken

- 2026-11-14 | Cluster account (Q1): before every job submission, compare the four accounts and use the one with the best queue position. Procedure in section 3.
- 2026-11-14 | Model identity (Q2): the June checkpoints in `ResultsToDownload/` cannot be trusted to be the report's Model A and B (Pablo: models were mixed up at the time). Consequences:
  - no attempt to map checkpoints to the report;
  - comparison against the report uses the report's own figures (section 1.2) as "Model A / Model B as reported";
  - numeric baselines come from new reference runs with known provenance (task A2, runs `REF-A`, `REF-B`);
  - the June checkpoints are scored once as "legacy, identity uncertain" and never presented as Model A or B.
- 2026-11-14 | Clean-up (Q7): regenerable or unnecessary files may be deleted and `EnvPablo/` taken out of the repository. Applies to task A1.
- 2026-11-14 | This file is the single detailed record of done and to-do work (section 0.2).

## 3. Cluster accounts and job submission

### 3.1 What exists
- **[measured, 2026-11-14, on `jed2`]** Four Slurm accounts for user `sarro` on cluster `jed`:
  - `cs-302`: QOS `academic`, `cs-302` (default `cs-302`), max wall 1 day on the association
  - `env540`: QOS `academic`, max wall 1 day
  - `math-454`: QOS `math-454`, max wall 1 day
  - `master`: QOS `academic`, max wall 3 days
- **[measured]** QOS on jed: `cs-302` priority 10000 but max wall 3 h and 5 submitted jobs per user; `math-454` priority 5000; `academic` priority 0, 1,152 CPUs per user.
- **[measured]** Fair-share snapshot on jed (higher is better): `env540` 0.991, `cs-302` 0.728, `math-454` 0.670, `master` 0.451. Raw usage is 0 on all but `math-454` (6).
- **[measured]** `jed` has no GPU partition (partitions: `standard`, `bigmem`, `hugemem`, `academic`, `infiniband`). The June training ran on `izar` (`--partition=gpu`). Only jed's accounting is visible from jed; the GPU cluster keeps its own associations and fair-share.
- **[inferred]** `/home` is shared between clusters: a June training log written on a GPU node shows the same `/home/sarro/SemProj/...` path.

### 3.2 Procedure before every job submission
- Run on the cluster where the job will be submitted:
  - `sacctmgr -P show assoc user=$USER format=cluster,account,partition,qos,defaultqos,maxwall`
  - `sshare -U -u $USER -o Account,RawUsage,EffectvUsage,FairShare`
  - `sacctmgr -P show qos format=name,priority,maxwall,maxtrespu,maxsubmitpu`
- Keep only the account/QOS pairs that allow the job (partition, GPU, requested wall time, job-count limits).
- Among those, pick the highest QOS priority, then the highest `FairShare` (equivalently the lowest usage).
- Write the chosen account, QOS and the fair-share values in the run registry entry (section 8).
- With today's jed numbers: a job under 3 h goes to `cs-302` with QOS `cs-302`; a longer one to `env540` with QOS `academic`.

### 3.3 Not settled
- Which GPU cluster to use and how to reach it from a session (see Q11).

## 4. Completed review steps and their outcomes

### R1. Read the report `[X]`
- Outcome: summary in section 1. The report contains no quantitative metric.
- Log:
  - 2026-11-14 | Extracted text and page images of the 16-page PDF | PyMuPDF in a temporary folder (no PDF tool on the system) | all sections and Figs. 1 to 9 read | baseline is qualitative only, so A2 must create the numbers.

### R2. Review the repository code and git state `[X]`
- Outcome (reproducibility):
  - **[read]** Last commit is 2026-05-19. Five tracked files are modified and 14 files/folders are untracked, including `train_modelB.py`, `Families/` and `filter_dataset.py`. The code behind Model B is not in git.
  - **[read]** The script that produced the report's rollout figures is gone. `visualise.out` (9 June) shows a multi-file version taking a model name; the `visualise.py` on disk is the 19 May version (file reverted on 13 June). It hardcodes a 256 x 8 model and the `noClamp_Long` scaling file.
  - **[read]** That `visualise.py` runs the rollout in float32 although training is float64. At ln T = 16 a float32 step is about 2e-6, larger than most true increments. Whether the lost version did the same is unknown.
  - **[read]** Training data paths point to `/scratch/izar/sarro/...`, not reachable from `jed2`.
  - **[read]** Local data: `Active_Files/` (1,000 trajectories generated from `params_active.txt`; 700 train, 300 val), `Families/Grid_Files/` (507), `Plot_Files/` (8, used for report figures).
- Outcome (training details):
  - **[read]** The mass-loss ramp is `1e7 * epoch / (EPOCHS - 1)`. With `EPOCHS = 10000` and a stop at epoch 956 on wall-time (`modelB_long`), the weight reached under 10% of its nominal maximum.
  - **[read]** Report and code disagree: scheduler patience 30 epochs (report) vs 5 checks x 10 epochs = 50 (code); weight factor 20 (report, `train_modelB.py`) vs 10 (`train.py`).
  - **[read]** The best checkpoint is chosen on unweighted one-step MSE, which does not track rollout quality.
  - **[read]** Resuming with `--model` restarts the epoch counter (ramp restarts at 0) and overwrites the loss file.
  - **[read]** `train.py` and `train_modelB.py` differ only in about ten constants. Every start re-parses 20 to 25M text rows with pandas.
- Outcome (equilibrium filter, report 4.1):
  - **[read]** `filter_dataset.py` drops rows inside each file, then `GrackleDataset` pairs adjacent rows as one 100-yr step. In flat regions 1 row in 100 is kept, so 10,000-yr increments are labelled as 100-yr steps. It also rewrites the data with `%.6e`, which zeroes increments below 1e-6.
  - **[read]** `Plot_Files/Plots_filt/BEST_Valid_Fam_3.png` shows a one-step HII increment of -0.04 after the cliff where the truth is near 0, consistent with this. The filter has not had a fair test.
- Log:
  - 2026-11-14 | Read all tracked and untracked scripts, job files, logs and loss files | working tree at `d0e52f8` plus uncommitted changes | findings above; legacy runs listed in section 8.1 | tasks A1, A2, A6 and B5 created from them.

### R3. Check the ground-truth data files `[X]`
- Outcome:
  - **[measured]** Rows are not always 100 yr apart. The driver writes at the first sub-step past each 100-yr mark, and sub-steps shrink to 0.1 x cooling time. In `Plot_Files/Train_Fam_3.dat` the intervals around the cliff are 142, 72, 105, 80, 141 yr and the first interval is 105.2 yr (105.8 yr in `Valid_Fam_3.dat`). The `dt` column is hard-coded to 100.
  - **[inferred]** The network has no `dt` input, so this is label noise of up to about 40% on the samples the loss weights most.
  - **[read]** The driver caps sub-steps at 10 yr before 0.5 Myr and 100 yr after (left over from the Iliev test).
  - **[read]** The time column has 6 significant digits (10-yr resolution after 1 Myr).
  - **[measured]** `nHI + nHII = nH` holds to 4e-16. Specific internal energy never increases (0 rows out of 25,000 in each of 5 files). Temperature does increase (every row in the Family 1 and 2 samples).
  - **[measured]** Median |d ln T| per step: 1.3e-9 (Family 0 sample), 6e-9 (Family 2), 1.9e-6 (Family 1), 3e-6 (Family 3); maximum at the cliff 0.41. Fraction of rows with |d ln T| > 1e-5: 1.1% (Family 0), 0% (Family 1), 1.2% (Family 2), 38 to 40% (Family 3).
- Outcome (why rollouts drift):
  - **[read]** Target std is about 3e-3 for ln T, so flat-regime targets sit near 1e-6 sigma. Best validation MSE of the legacy runs: 2.0e-3 (`1`) and 1.4e-2 (`modelB_long`).
  - **[measured from Fig. 7]** The Family 0 drift is ln(2.48/2.13) = 0.15 over 25,000 steps: a bias of 6e-6 per step, about 0.002 sigma.
  - **[inferred]** "Activity bias" and "equilibrium bias" are the same defect: the loss does not constrain increments in the flat regime, so the residual bias there is integrated 25,000 times. Rebalancing the data does not address it; changing the target, the loss or the number of steps does. Tested by A5 and B1.
- Log:
  - 2026-11-14 | Row-interval, sub-step and increment statistics | 5 files in `Plot_Files/` (`Train_Fam_0..3`, `Valid_Fam_3`) | numbers above | tasks A3, A4, A5 created.

### R4. Test the density-time scaling `[X]`
- Outcome:
  - **[inferred from the equations]** With no radiation every term is two-body, so d(state)/dt = nH x f(T, x): the evolution depends on nH and t only through nH x t.
  - **[measured]** Grid trajectories with the same T0 and x0 = 0.5 and densities 10x or 100x apart (T0 = 1e4 to 1e7 K), comparing the denser one at t with the other at 10t or 100t: max |d ln T| is 1e-6 to 1e-2 against total changes of 0.3 to 5. Two pairs reach 0.2 at the cliff, where a one-row time offset dominates.
  - Not checked: other x0 values, T0 = 1e8 K, T0 below 1e4 K.
  - **[inferred]** Consequences: the state is two-dimensional (T or energy, ionised fraction) plus one scaled time nH x dt; the four families mostly encode how much nH x t fits in 2.5 Myr; making `dt` an input (report 4.3) merges with the density input.
- Log:
  - 2026-11-14 | Pairwise comparison of trajectories at rescaled times | `Families/Grid_Files/`, 78 pairs printed | agreement to about 1% of the total change in ln T | task B1 created; B5 recommends against the Mixture of Experts.

### R5. Test the analytic rates against the data `[X]`
- Outcome:
  - **[measured]** Analytic k1, k2 reproduce the finite-difference d nHII/dt with median relative error 1e-4 to 5e-3 wherever ionisation and recombination do not cancel. They fail (factor 1e2 to 1e3) only where the ionisation state relaxes in 0.03 to 0.2 yr and the two terms cancel to 1e-5 (ionisation equilibrium, first 1e4 yr of the Family 3 samples).
  - **[read]** `helpers/physics_inconsistency_proof.png` shows the analytic dT/dt on top of the finite-difference one over a whole trajectory.
  - **[inferred]** The rates are right; the explicit residual is the wrong form in the stiff regime.
- Log:
  - 2026-11-14 | Explicit and trapezoid rate vs finite difference, cancellation ratio, relaxation time | 4 files in `Plot_Files/` | numbers above | task B3 created.

### R6. Review `../../Lessandre` and `../../OldFiles` `[X]`
- `Lessandre` (previous project, reference [10] of the report):
  - Reuse: `grackle/` source (3.4.2-dev) and `grackle-install/` (headers, `libgrackle.a`) for rebuilding the driver (A3) and as reference for analytic rates (B3); `export_onnx.py` with its note on SWIFT's `ml_onnx_infer.c` (B6); `cooling-main.c`, a driver that integrates one arbitrary `dt` per input line (B1, B4); the commented-out 3-step unrolled loss in `train_test.py` (B2) and softmax-enforced HI + HII in `model.py` (A4).
  - Compare against: it already had `dt` as an input and evaluated (initial state, dt) to state in one shot, the idea B1 returns to. `losses_NN_HII_1e7_Pablo_v2.txt` is flat at 0.112 from the first epoch, so that run did not learn and is not an accuracy baseline.
  - Fix or avoid: MinMax scaling of log-ratio targets, ReLU activations, the constants flagged in `generate_input.py` (mu = 1, gamma = 2/3), and `create_train_test_set` loading `output_file2` twice.
- `OldFiles`: `OldSimulations/Sim_0`, `Sim_1` (early checkpoints, logs, plots) and `OldProblems/*.log` (three early time-encoding attempts). Nothing to reuse in code.
- Log:
  - 2026-11-14 | Read code, job files, loss files, listed data | both folders | points above | no task depends on `OldFiles`.

### R7. Check cluster accounts `[X]`
- Outcome: section 3.
- Log:
  - 2026-11-14 | `sacctmgr`, `sshare`, `sinfo`, QOS table | on `jed2` | 4 accounts, fair-share snapshot, no GPU partition on jed | procedure 3.2 fixed; Q11 opened for the GPU cluster.

## 5. Tasks: quick wins

### A1. Simplify the repository structure `[-]`
- Goal: one obvious place for code, configs, jobs, data and results; everything needed to reproduce a run is in git.
- Why: R2. The code behind the report is uncommitted and the evaluation script is lost. The root holds 49 entries and 38 GB (14 GB virtual environment, 15 GB `Active_Files`, 7.3 GB `Families`, about 1,550 regenerable PNGs).
- Steps:
  - `[-]` Commit the current working tree as it is and tag it `report-june2026`.
  - `[-]` Move code to: `grackle_pinn/` (`model.py`, `data_utils.py`, `physics.py`, one `train.py` replacing `train.py` + `train_modelB.py`, `evaluate.py`), `configs/` (one file per experiment), `scripts/datagen/` (`create_Grackle/*`, `Families/*.py`), `scripts/datasets/` (`active.py`, `split.py`, `extract_fam3.py`, `extract_plot_files.py`, `filter_dataset.py`), `scripts/plots/` (the six `plot_*.py` merged into one CLI), `jobs/`, `docs/` (report PDF, `Plots_Report/`).
  - `[-]` Move data to `data/` and June outputs to `results/legacy/<run>/` (both git-ignored).
  - `[-]` Move to `archive/`: `helpers/`, loose root files (`ac_traj.dat`, `eq_traj.dat`, `output_1.dat`, `*_analysis.png`), `Eval_Performance/`, `Family3_Files/`.
  - `[-]` Delete (allowed, section 2): `Plots/`, `Active_Files/Plots_Subfolder/`, `Families/Grid_Files/Plots_Subfolder/`, `__pycache__/`, empty `.err` files.
  - `[-]` Virtual environment: recreate outside the repo from `requirements.txt` and delete `EnvPablo/` (a venv cannot simply be moved, its scripts contain absolute paths). Check first that the new one imports torch and runs one batch.
  - `[-]` `.gitignore`: stop ignoring `*.job`, `*.sh` and the IC lists.
  - `[-]` Rewrite `README.md` (it references `generate_data.sh` at the root and `pinn_performance.png`, which do not exist).
- Effort: 0.5 to 1 day.
- Success: fresh clone + README reproduces data generation, one training launch and one evaluation; root has under 12 entries; `git status` clean; same loss on one fixed batch before and after the move.
- Log: none yet.

### A2. Evaluation harness and numeric baseline `[-]`
- Goal: a single `evaluate.py` that turns any checkpoint into a table of numbers, and reference numbers to compare against.
- Why: section 1.2 and R2. The report has figures only, so "improved" cannot be measured yet. Per section 2, the June checkpoints are not accepted as Model A and B.
- Steps:
  - `[-]` Rollout in float64, batched over trajectories; architecture and scaling read from the checkpoint (stored there from now on).
  - `[-]` Metrics per family and overall: rollout error |d log10| for T, nHI, nHII (median, 95th percentile over time, value at 2.5 Myr); cliff-time error (time of steepest d ln T); mass-conservation violation max |ln(nH_pred / nH_0)|; one-step error binned by decade of the true increment; count of unphysical outputs (T below 1 K, energy increasing).
  - `[-]` Fixed evaluation sets: the 8 `Plot_Files` trajectories and a per-family set regenerated from `Families/ICs_valid.txt` (200 ICs).
  - `[-]` "As reported" row: numbers read off the report figures (section 1.2), labelled as such.
  - `[-]` Legacy checkpoints: score all seven runs in `results/legacy/` once, labelled "legacy, identity uncertain". Also once in float32 to quantify the precision effect found in R2.
  - `[-]` Reference runs with known provenance, report configuration (12 x 512, loss weight 20, soft mass loss), trained through the cleaned pipeline:
    - `REF-B`: kNN-balanced dataset regenerated from `Families/ICs_train.txt` and `ICs_valid.txt` (exactly reproducible).
    - `REF-A`: variance-stratified dataset; needs the 15,000 trajectories of `params.txt` and the 800/250/250 selection, whose script is not in the repo (Q12).
  - `[-]` Write `results/baseline.md` with the table; regenerate the layout of Figs. 3 to 9 from the same script.
- Effort: 1 to 2 days for the harness (CPU); reference runs 24 GPU-hours each.
- Success: baseline table committed with rows "as reported", legacy, `REF-A`, `REF-B`; every later task reports the same columns next to it.
- Log: none yet.

### A3. Clean ground-truth generation `[-]`
- Goal: trajectories whose rows are exactly 100 yr apart, integrated the same way at all times, stored at full precision in a fast format.
- Why: R3. Timing jitter up to about 40% at the cliff bounds how well any fixed-step model can fit the cliff (Fig. 6 timing error).
- Steps:
  - `[-]` Driver: clip the last sub-step to land exactly on each output time; drop the 0.5 Myr switch; write the true interval in the `dt` column; print time at full precision; write internal energy and x directly.
  - `[-]` Convert trajectories once to `.npy`/`.npz` (float64) and load those in training.
  - `[-]` Regenerate the datasets from the existing IC lists (`params*.txt`, `Families/ICs_*.txt`). The existing binary links and runs on jed; 1,000 trajectories took under an hour on 16 cores in May.
  - `[-]` Regenerate the 8 evaluation trajectories; report A2 numbers on old and new truth.
- Effort: 1 day plus 1 to 2 h of CPU jobs.
- Success: all row intervals equal 100 yr to 1e-6 yr; dataset load time from minutes to seconds; A2 table recomputed on the new truth.
- Open point: A2's reference runs should train on the regenerated data, so A3 may have to run before the `REF` runs of A2. Decide when starting A2.
- Log: none yet.

### A4. State variables with built-in constraints `[-]`
- Goal: the network predicts in (internal energy, ionised fraction x) with nH carried through unchanged.
- Why: R3 and Fig. 3 bottom. Mass conservation becomes exact instead of a penalty with a weight to tune; energy can be forced non-increasing, which removes outputs like Fig. 4 bottom (T rising 20x) and bounds x to [0, 1].
- Steps:
  - `[-]` Inputs: ln e (or ln T), logit x, ln nH. Outputs: -softplus(.) for d ln e; a bounded update for logit x.
  - `[-]` Recover T, nHI, nHII analytically from (e, x, nH) for metrics and plots.
  - `[-]` Keep the soft mass loss as a switchable ablation so the report's variant stays comparable.
- Effort: 1 day plus one training run.
- Success: mass violation at float64 round-off; zero unphysical outputs; rollout metrics no worse than `REF` on Family 3.
- Log: none yet.

### A5. Target and loss that resolve the flat regime `[-]`
- Goal: test the diagnosis of R3 on the fixed-100-yr model, changing only target and loss.
- Why: Figs. 3 bottom, 4, 7, 8, 9. Most direct comparison with the report: same data, same network.
- Steps:
  - `[-]` Target: ln of the rate magnitude (e.g. ln(-d ln e / dt), sign fixed by A4) instead of the raw increment; for x, a signed-log or asinh transform.
  - `[-]` Loss: plain MSE in that space; drop the `1 + 20|y|` weight.
  - `[-]` Train on the balanced dataset first (where the report saw the worst drift), then the stratified one.
- Effort: 1 to 2 days plus two training runs of 12 to 24 GPU-hours.
- Success (provisional until A2 gives numbers): Families 0 to 2 final error in T under 0.02 dex at 2.5 Myr (report: 0.07 dex to tens of dex); Family 3 cliff time within 2% (report Fig. 6: about 7%); no regression on the Family 3 case of Fig. 5.
- Log: none yet.

### A6. Training-loop fixes `[-]`
- Goal: runs that finish what they are configured to do and select the right checkpoint.
- Why: R2. `modelB_long` trained with under 10% of the intended physics weight, and "BEST" is chosen on a metric unrelated to rollout.
- Steps:
  - `[-]` Ramp and scheduler defined on optimiser steps or fraction of the wall-time budget, not nominal epochs.
  - `[-]` Resume restores epoch and ramp position and appends to the loss file.
  - `[-]` Checkpoint selection on a short rollout metric from A2.
  - `[-]` Data on GPU as tensors instead of a 4-worker DataLoader; test float32 forward with float64 accumulation for speed.
  - `[-]` Fix seeds; save the config in the run folder.
- Effort: 0.5 to 1 day.
- Success: epoch time reduced (June: 72 to 89 s per epoch for 20 to 25M points); a resumed run reproduces an uninterrupted one; the selected checkpoint has the best rollout score among saved ones.
- Log: none yet.

## 6. Tasks: larger experiments

### B1. Flow map with scaled time as input (main bet) `[-]`
- Goal: a model state(t), nH x dt -> state(t + dt) valid for dt from 100 yr to 2.5 Myr, so a 2.5 Myr evolution takes one or a few evaluations instead of 25,000.
- Why: R3, R4 and report 4.3. Error accumulation, the report's central problem, is removed by construction. A SWIFT surrogate needs variable `dt` anyway.
- Steps:
  - `[-]` Verify the scaling over the whole grid (all x0, all T0) and quantify the residual; keep ln nH as a weak extra input if it is not negligible.
  - `[-]` Build training pairs (row i, row j > i) from existing trajectories, dt log-uniform; sample pairs uniformly in (T, x, ln(nH dt)).
  - `[-]` Network: inputs (ln e, logit x, ln(nH dt)), outputs as in A4, identity as dt -> 0 built in.
  - `[-]` Evaluate with one jump, with k jumps (k = 2 to 100), and against the fixed-step models of A5.
- Risk: the map is steep in dt near the cliff, so one-jump errors appear as cliff-time errors. Mitigation: sub-step near the cliff; B3's composition loss.
- Effort: 1 to 2 weeks including runs.
- Success: A2 metrics at or below A5 on all families with at most 100 evaluations per trajectory; speed-up over Grackle measured in B4.
- Log: none yet.

### B2. Rollout-aware training for the fixed-step model `[-]`
- Goal: train on what is evaluated, the multi-step rollout.
- Why: Fig. 6; one-step accuracy does not imply rollout stability. Less needed if B1 succeeds.
- Steps:
  - `[-]` Unrolled loss over K steps with gradient through the first step only, K increased during training.
  - `[-]` Noise injection on inputs at the scale of the model's own one-step error.
- Effort: 3 to 5 days plus runs.
- Success: rollout 95th-percentile error reduced against A5 at equal one-step error.
- Log: none yet.

### B3. Physics-informed terms that respect stiffness `[-]`
- Goal: bring back equation-based losses in a form that trains.
- Why: report section 2.3.2 (rate and energy losses abandoned) and R5 (the rates are correct outside ionisation equilibrium).
- Steps:
  - `[-]` Restore `GrackleRates` from commit `475ca3b`; unit-test k1, k2 and the cooling function against Grackle output.
  - `[-]` Composition loss for B1: map(dt1 + dt2) = map(dt2) after map(dt1), on unlabelled states.
  - `[-]` ODE residual for B1: derivative of the map with respect to dt equals the right-hand side at the predicted state, replaced by x = x_eq(T) where the ionisation relaxation time is far below dt.
  - `[-]` Collocation points in regions without data; repeat with 10x fewer trajectories.
- Effort: 1 to 2 weeks.
- Success: same A2 metrics with 10x less training data, or lower cliff-time error at equal data.
- Log: none yet.

### B4. Reference baselines and speed `[-]`
- Goal: know what the network has to beat.
- Why: the report motivates the work by speed but reports no timing and no non-neural baseline.
- Steps:
  - `[-]` Tabulated flow map on a (T, x, nH dt) grid with interpolation.
  - `[-]` Grackle wall-time per particle-step; network inference time on CPU and GPU, batched.
  - `[-]` Grackle at a coarser tolerance as an accuracy/time reference; measure the truth's own self-consistency (R4 suggests a floor of 1e-3 to 1e-2 in ln T).
- Effort: 2 to 3 days.
- Success: one table of accuracy vs cost for Grackle, table lookup, `REF-A/B`, A5, B1.
- Log: none yet.

### B5. Report future-work items, re-assessed `[-]`
- `[-]` Equilibrium filter (report 4.1), corrected: form (state, next state) pairs first, then subsample pairs by activity; keep float64. Effort 1 to 2 days. Compare with A5 on the A2 table. Low priority: R3 predicts it reduces but does not remove the drift.
- `[-]` Mixture of Experts (report 4.2): recommendation is not to start. Per R4 the families are not separate physics and gating adds discontinuities at family borders. The single-family legacy run `fam3_long` gets scored in A2 as a data point. Revisit only if A5 and B1 both fail on a specific family.
- Time-adaptive data (report 4.3): absorbed into B1.
- Log: none yet.

### B6. Scope extensions `[-]`
- `[-]` Helium species and electrons; UV background and heating (breaks the nH x t scaling, so density returns as a full input).
- `[-]` ONNX export and SWIFT coupling following `Lessandre/export_onnx.py`.
- Effort: open-ended; not planned in detail.
- Log: none yet.

## 7. Recommended order

- A1 -> A2 (harness) -> A3 -> A2 (reference runs) -> A4 -> A6 -> A5
- then B1 -> B4 -> B3 -> B2 -> B5 -> B6.
- A5 and B1 are the two tasks expected to change the results; the others make their comparison with the report valid.

## 8. Run registry

### 8.1 Legacy runs, June 2026 (identity uncertain, see section 2)
- Files in `ResultsToDownload/` (to become `results/legacy/` in A1): last and `BEST` checkpoint, scaling file, loss history and log per run. None has been scored yet (A2).
- `1` | 25.0M training points | 1,600 epochs, 32.8 h, completed | best one-step validation MSE 1.98e-3 at epoch 681 | about 72 s per epoch.
- `noClamp_Long` | 2.1M points | 6,500 epochs, 10.3 h (19 to 20 May) | no validation column | log shows a 256 x 8 era; final printed MSE 1.7e-5.
- `modelB` | 150 epochs | best validation MSE 1.15e-1 at epoch 146.
- `modelB_long` | 20.0M points | 956 epochs, stopped at the 24 h limit | best validation MSE 1.45e-2 at epoch 461 | about 89 s per epoch.
- `fam3` | 278 epochs | best validation MSE 4.80e-1 at epoch 271 | loss scale differs from the other runs.
- `fam3_long` | 5.0M points (Family 3 only) | 1,900 epochs | best validation MSE 6.99e-3 at epoch 1,021.
- `filt` | 3.64M points (filtered data, defect described in R2) | 336 epochs, stopped at 1.75 h | best validation MSE 2.01e-2 at epoch 336.
- Architecture of each run is not recorded in its files and has to be read from the checkpoint tensors in A2.

### 8.2 New runs
- Entry format: `run id | date | task | data | config | account, QOS, fair-share at submission, job id | status | key metrics | verdict`.
- None yet.

## 9. Open questions

### 9.1 Answered
- Q1 (accounts): answered 2026-11-14, see sections 2 and 3. Remaining part moved to Q11.
- Q2 (which checkpoints are Model A and B): answered 2026-11-14, see section 2.
- Q7 (deleting files, moving the environment): answered 2026-11-14, see section 2.

### 9.2 Open
- Q3. Is the multi-file `visualise.py` from 9 June available anywhere else (laptop, other clone)? Otherwise A2 rebuilds it.
- Q4. What is the deliverable (updated report, paper, thesis continuation) and is there a deadline?
- Q5. Must the model remain a "PINN" with an explicit physics loss term? A4 makes mass conservation exact by construction, which removes the mass loss; B3 adds physics terms back in another form.
- Q6. Is the fixed 100-yr step a requirement from the supervisor or the SWIFT use-case? Which range of time steps and densities does the target application need?
- Q8. May the Grackle driver be changed (A3)? The regenerated truth will differ slightly from the data behind the report.
- Q9. Was `Active_Files/` (1,000 `params_active.txt` trajectories) used for a model in the report, or only for the activity plots?
- Q10. After hydrogen-only works: helium and UV background first, or SWIFT coupling first?
- Q11. GPU cluster: jed has no GPU and only jed's accounts are visible from it. Which GPU cluster should training use (Izar as in June, or another), and should jobs be submitted by a session started on that cluster's login node or over `ssh` from jed? Are the Izar scratch datasets (`Full_Grackle_Data`, `Balanced_Dataset`, `Fam3_Dataset`, `Filtered_Data`) still there, or is everything regenerated from the IC lists?
- Q12. The script that built the variance-stratified dataset (800 / 250 / 250 files out of 15,000) is not in the repo; `active.py` only copies the top 3,000. Is it available, or should `REF-A` use a re-implementation from the report's description?
