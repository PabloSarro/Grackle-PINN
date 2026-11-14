# TASKS — improving the Grackle-PINN results

Plan written on 2026-11-14, branch `improvements-nov2026`. Analysis only: no code has been changed yet.

## 0. How to use this file

- One bullet point (task ID such as `A1`, `B1`) is worked on at a time: implement, test, commit, push, report, wait.
- Order of work is the order of section 5 ("Recommended order").
- Evidence labels used below:
  - **[measured]**: I ran a check on the files in this repo and report the number.
  - **[read]**: taken directly from the code, logs or report.
  - **[inferred]**: my interpretation; to be confirmed by the task that tests it.

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

## 2. Findings from this review that shape the plan

### 2.1 Reproducibility
- **[read]** Last commit is 2026-05-19. Five tracked files are modified and 14 files/folders are untracked, including `train_modelB.py`, `Families/` and `filter_dataset.py`. The code behind Model B is not in git.
- **[read]** The script that produced the report's rollout figures is gone. `visualise.out` (9 June) shows a multi-file version taking a model name; the `visualise.py` on disk is the 19 May version (file reverted on 13 June). It hardcodes a 256 x 8 model and the `noClamp_Long` scaling file.
- **[read]** That `visualise.py` runs the rollout in float32 (`.float()`, `dtype=torch.float32`) although training is float64. At ln T = 16 a float32 step is about 2e-6, larger than most true increments (see 2.3). Whether the lost version did the same is unknown.
- **[read]** Training data paths point to `/scratch/izar/sarro/...`. This session runs on `jed2` (no GPU), where that path does not exist. Local data: `Active_Files/` (1,000 trajectories from `params_active.txt`), `Families/Grid_Files/` (507), `Plot_Files/` (8, the ones used for report figures).
- **[read]** Checkpoints, scaling files, loss histories and logs for seven runs are in `ResultsToDownload/` (`1`, `noClamp_Long`, `modelB`, `modelB_long`, `fam3`, `fam3_long`, `filt`).

### 2.2 Ground-truth data
- **[measured]** Rows are not always 100 yr apart. The driver writes at the first sub-step past each 100-yr mark, and sub-steps shrink to 0.1 x cooling time. In `Plot_Files/Train_Fam_3.dat` the intervals around the cliff are 142, 72, 105, 80, 141 yr, and the first interval is 105.2 yr. The `dt` column is hard-coded to 100, so this is invisible in the files.
- **[inferred]** The network has no `dt` input, so this is label noise of up to about 40% on exactly the samples the loss weights most.
- **[read]** The driver caps sub-steps at 10 yr before 0.5 Myr and 100 yr after (left over from the Iliev test), so the same state is integrated differently depending on elapsed time.
- **[read]** The time column has 6 significant digits (10-yr resolution after 1 Myr).
- **[measured]** `nHI + nHII = nH` holds to 4e-16 in the data. Specific internal energy never increases (0 rows out of 25,000 in each of 5 files). Temperature does increase (every row in the Family 1 and 2 samples, because recombination raises the mean molecular weight).

### 2.3 Why the rollouts drift
- **[measured]** Median |d ln T| per 100-yr step: 1.3e-9 (Family 0 sample), 6e-9 (Family 2), 1.9e-6 (Family 1); maximum at the cliff 0.41 (Family 3). Eight orders of magnitude in one standardised target.
- **[read]** Target std is about 3e-3 for ln T (training logs), so flat-regime targets sit near 1e-6 sigma. Best validation MSE is 2.0e-3 (Model A, run `1`) and 1.4e-2 (Model B).
- **[measured from Fig. 7]** The Family 0 drift is ln(2.48/2.13) = 0.15 over 25,000 steps, a bias of 6e-6 per step, about 0.002 sigma. The loss cannot see an error that small.
- **[inferred]** The "activity bias" and the "equilibrium bias" are the same defect: the loss does not constrain increments in the flat regime, so whatever small bias training leaves there is integrated 25,000 times. Rebalancing the dataset (Models A, B, the filter, a Mixture of Experts) does not address this; changing the target, the loss, or the number of steps does.

### 2.4 The equilibrium filter (report 4.1) was not tested fairly
- **[read]** `filter_dataset.py` drops rows inside each file, then `GrackleDataset` pairs adjacent rows as one 100-yr step. In flat regions 1 row in 100 is kept, so 10,000-yr increments are labelled as 100-yr steps. It also rewrites the data with `%.6e`, which zeroes increments below 1e-6.
- **[read]** `Plot_Files/Plots_filt/BEST_Valid_Fam_3.png` shows a one-step HII increment of -0.04 after the cliff where the truth is near 0, consistent with this.

### 2.5 Density-time scaling
- **[inferred from the equations]** With no radiation, every term (collisional ionisation, recombination, all four cooling terms) is two-body, so d(state)/dt = nH x f(T, x). The evolution depends on nH and t only through nH x t.
- **[measured]** On grid trajectories with the same T0 and x0 = 0.5 and densities 10x or 100x apart (T0 = 1e4 to 1e7 K), comparing the denser one at time t with the other at 10t or 100t: max |d ln T| is 1e-6 to 1e-2 against total changes of 0.3 to 5. Two pairs reach 0.2 at the cliff, where a one-row time offset dominates. Other x0 values and T0 = 1e8 K were not checked.
- **[inferred]** Consequences:
  - The problem is two-dimensional in state (T or energy, ionised fraction) plus one "scaled time" nH x dt.
  - The four families mostly encode how much nH x t fits into 2.5 Myr, which weakens the case for a Mixture of Experts split by family (report 4.2).
  - Making `dt` an input (report 4.3) is not an extra burden: it merges with the density input.

### 2.6 Physics losses
- **[measured]** Analytic k1, k2 reproduce the finite-difference d nHII/dt with median relative error 1e-4 to 5e-3 wherever ionisation and recombination do not cancel. They fail (error factor 1e2 to 1e3) only where the ionisation state relaxes in 0.03 to 0.2 yr and the two terms cancel to 1e-5, i.e. ionisation equilibrium.
- **[read]** `helpers/physics_inconsistency_proof.png` shows the analytic dT/dt on top of the finite-difference one over the whole trajectory.
- **[inferred]** The rates are right. The explicit residual is the wrong form in the stiff regime; an equilibrium constraint there and an ODE residual elsewhere should work.

### 2.7 Training details
- **[read]** The mass-loss ramp is `1e7 * epoch / (EPOCHS - 1)` with `EPOCHS = 10000` for Model B, which stopped at epoch 956 on wall-time: the weight reached under 10% of its nominal maximum.
- **[read]** Report and code disagree: scheduler patience 30 epochs (report) vs 5 checks x 10 epochs = 50 (code); weight factor 20 (report, `train_modelB.py`) vs 10 (`train.py`).
- **[read]** The best checkpoint is chosen on unweighted one-step MSE, which does not track rollout quality.
- **[read]** Resuming with `--model` restarts the epoch counter (ramp restarts at 0) and overwrites the loss file.
- **[read]** `train.py` and `train_modelB.py` differ only in about ten constants (file names, data paths, wall-time, epochs, loss weight). Every start re-parses 20 to 25M text rows with pandas.

## 3. Reuse from `../../Lessandre` and `../../OldFiles`

### 3.1 `Lessandre` (previous project, reference [10])
- Reuse:
  - `grackle/` source (3.4.2-dev) and `grackle-install/` (headers, `libgrackle.a`): needed to rebuild the driver for task A3. `src/clib/rate_functions.c` is the reference for analytic rates in B3.
  - `export_onnx.py` and its note on SWIFT's `ml_onnx_infer.c`: the route to SWIFT coupling (B6).
  - `cooling-main.c`: driver that reads (T, nHI, nHII, dt) per line and integrates one arbitrary `dt`. Useful for generating variable-`dt` pairs directly (B1) and for timing Grackle (B4).
  - Commented-out 3-step unrolled loss in `train_test.py` and softmax-enforced HI + HII in `model.py`: earlier sketches of B2 and A4.
- Compare against:
  - Its setup already had `dt` as an input and evaluated (initial state, dt) to state in one shot, the idea B1 returns to. Report it as the prior baseline.
  - `losses_NN_HII_1e7_Pablo_v2.txt` is flat at 0.112 from the first epoch: that run did not learn, so it is not a usable accuracy baseline.
- Fix or avoid:
  - MinMax scaling of log-ratio targets and ReLU activations; `generate_input.py` flags its own wrong constants (mu = 1, gamma = 2/3).
  - `create_train_test_set` loads `output_file2` twice (file 3 is never read).

### 3.2 `OldFiles`
- `OldSimulations/Sim_0`, `Sim_1`: early checkpoints, logs and `pinn_performance_*.png`. `OldProblems/*.log`: debugging logs of three early time-encoding attempts.
- Nothing to reuse in code. Keep as history; candidates for the archive in A1.

## 4. Tasks

### 4.A Quick wins

#### A1. Simplify the repository structure (first implementation step)
- Goal: one obvious place for code, configs, jobs, data and results; everything needed to reproduce a run is in git.
- Why: section 2.1. Model B's code is uncommitted and the evaluation script is lost; no later comparison against the report is trustworthy until this is fixed. The root currently holds about 50 entries and 38 GB (14 GB virtual environment, 15 GB `Active_Files`, 7.3 GB `Families`, about 1,550 regenerable PNGs).
- Steps:
  - Commit the current working tree as it is and tag it `report-june2026`, so the report state is recoverable.
  - Move to this layout:
    - `grackle_pinn/`: `model.py`, `data_utils.py`, `physics.py`, one `train.py` (replaces `train.py` + `train_modelB.py`), `evaluate.py`.
    - `configs/`: one file per experiment (paths, sizes, loss weights) instead of edited constants.
    - `scripts/datagen/` (`create_Grackle/*`, `Families/*.py`), `scripts/datasets/` (`active.py`, `split.py`, `extract_fam3.py`, `extract_plot_files.py`, `filter_dataset.py`), `scripts/plots/` (the six `plot_*.py` merged into one CLI).
    - `jobs/`: SLURM files. `docs/`: report PDF and `Plots_Report/`.
    - `data/` and `results/<run>/` (both git-ignored): trajectories; checkpoint + scaling + losses + log + plots per run (from `ResultsToDownload/`).
    - `archive/`: `helpers/`, loose root files (`ac_traj.dat`, `eq_traj.dat`, `output_1.dat`, `*_analysis.png`), `Eval_Performance/`, `Family3_Files/`.
  - Remove regenerable output after confirmation (see Q7): `Plots/`, `Active_Files/Plots_Subfolder/` (1,000 PNG), `Families/Grid_Files/Plots_Subfolder/` (507 PNG), `__pycache__/`, empty `.err` files. Move `EnvPablo/` out of the repo folder.
  - `.gitignore`: stop ignoring `*.job`, `*.sh` and the IC lists (`*.txt` under `scripts/datagen/`), so runs are reproducible.
  - Rewrite `README.md` (it references `generate_data.sh` at the root and `pinn_performance.png`, neither of which exists).
- Effort: 0.5 to 1 day.
- Success: fresh clone + README reproduces data generation, one training launch and one evaluation; root has under 12 entries; `git status` clean; no behaviour change (same loss on one fixed batch before and after the move).

#### A2. Evaluation harness and numeric baseline
- Goal: a single `evaluate.py` that turns any checkpoint into a table of numbers, and that table for the report's Model A and Model B.
- Why: section 1.2 and 2.1. The report has figures only, so "improved" cannot be measured yet.
- Steps:
  - Rollout in float64, batched over trajectories; architecture and scaling read from the checkpoint (store them there from now on).
  - Metrics per family and overall:
    - rollout error |d log10| for T, nHI, nHII: median and 95th percentile over time, and value at 2.5 Myr;
    - cliff-time error for trajectories with a cliff (time of steepest d ln T);
    - mass-conservation violation, max |ln(nH_pred / nH_0)|;
    - one-step error binned by decade of the true increment (shows directly whether the flat regime is resolved);
    - count of unphysical outputs (T below 1 K, energy increasing).
  - Fixed evaluation sets: the 8 `Plot_Files` trajectories (report figures) and a larger per-family set (needs Q1 or regenerated data).
  - Score `ResultsToDownload/PINN_1` and `PINN_modelB_long` (last and BEST), plus `fam3_long` and `filt`. Also run once in float32 to quantify the effect of 2.1.
  - Write `results/baseline.md` with the table; regenerate Figs. 3 to 9 from the same script.
- Effort: 1 to 2 days (CPU is enough for evaluation).
- Success: baseline table committed; every later task reports the same columns next to it.

#### A3. Clean ground-truth generation
- Goal: trajectories whose rows are exactly 100 yr apart, integrated the same way at all times, stored at full precision in a fast format.
- Why: section 2.2. Timing jitter up to about 40% at the cliff bounds how well any fixed-`dt` model can fit the cliff (Fig. 6 timing error).
- Steps:
  - Driver: clip the last sub-step to land exactly on each output time; drop the 0.5 Myr switch (one sub-step cap); write the true interval in the `dt` column; print time with full precision; optionally write internal energy and x directly.
  - Convert trajectories once to `.npy`/`.npz` (float64) and load those in training.
  - Regenerate the datasets needed (A, B, grid) from the existing IC lists (`params*.txt`, `Families/ICs_*.txt`). The existing binary runs on this machine; 1,000 trajectories took under an hour on 16 cores.
  - Keep the 8 evaluation ICs fixed and regenerate them too; report A2 numbers on both old and new truth.
- Effort: 1 day plus 1 to 2 h of CPU jobs.
- Success: all row intervals equal 100 yr to 1e-6 yr; dataset load time from minutes to seconds; A2 table recomputed on the new truth.

#### A4. State variables with built-in constraints
- Goal: the network predicts in (internal energy, ionised fraction x) with nH carried through unchanged.
- Why: section 2.2 and Fig. 3 bottom. Mass conservation becomes exact instead of a penalty with a weight to tune; energy can be forced non-increasing (no heating in this setup), which removes outputs like Fig. 4 bottom (T rising 20x) and bounds x to [0, 1].
- Steps:
  - Inputs: ln e (or ln T), logit x, ln nH. Outputs: -softplus(.) for d ln e; a bounded update for logit x.
  - Recover T, nHI, nHII analytically from (e, x, nH) for metrics and plots.
  - Keep the soft mass loss as a switchable ablation, so the report's variant stays comparable.
- Effort: 1 day plus one training run.
- Success: mass violation at float64 round-off (baseline: from A2); zero unphysical outputs; rollout metrics no worse than baseline on Family 3.

#### A5. Target and loss that resolve the flat regime
- Goal: test the diagnosis of section 2.3 on the fixed-100-yr model, changing only target and loss.
- Why: Figs. 3 bottom, 4, 7, 8, 9. This is the most direct comparison with the report: same data, same network.
- Steps:
  - Target: ln of the rate magnitude (e.g. ln(-d ln e / dt), sign fixed by A4) instead of the raw increment, so 1e-9 and 0.4 become a range of about 20 in a smooth variable. For x, a signed-log or asinh transform.
  - Loss: plain MSE in that space (relative error on the rate); drop the `1 + 20|y|` weight.
  - Compare on Model B's dataset first (balanced, where the report saw the worst drift), then Model A's.
- Effort: 1 to 2 days plus two training runs of 12 to 24 GPU-hours each.
- Success (provisional, to be fixed once A2 gives numbers): Families 0 to 2 final error in T under 0.02 dex at 2.5 Myr (report: 0.07 dex to tens of dex); Family 3 cliff time within 2% (report Fig. 6: about 7%); no regression on the Family 3 case of Fig. 5.

#### A6. Training-loop fixes
- Goal: runs that finish what they are configured to do and select the right checkpoint.
- Why: section 2.7. Model B was trained with under 10% of the intended physics weight, and "BEST" is chosen on a metric unrelated to rollout.
- Steps:
  - Ramp and scheduler defined on optimiser steps or elapsed fraction of the wall-time budget, not nominal epochs.
  - Resume restores epoch, ramp position and appends to the loss file.
  - Checkpoint selection on a short rollout metric from A2 (a few validation trajectories every N epochs).
  - Data on GPU as tensors (20M x 6 float64 is 1 GB) instead of a 4-worker DataLoader; test float32 forward with float64 accumulation for speed.
  - Fix seeds; log the config into the run folder.
- Effort: 0.5 to 1 day.
- Success: epoch time reduced (baseline 72 to 89 s per epoch for 20 to 25M points); resumed run reproduces an uninterrupted one; selected checkpoint has the best rollout score among saved ones.

### 4.B Larger experiments

#### B1. Flow map with scaled time as input (main bet)
- Goal: a model state(t), nH x dt -> state(t + dt) valid for dt from 100 yr to 2.5 Myr, so a 2.5 Myr evolution takes one or a few evaluations instead of 25,000.
- Why: sections 2.3 and 2.5; report 4.3. Error accumulation, the report's central problem, is removed by construction rather than fought. A SWIFT surrogate needs variable `dt` anyway.
- Steps:
  - Verify the scaling over the whole grid (all x0, T0 up to 1e8 K) and quantify the residual; keep ln nH as a weak extra input if the residual is not negligible.
  - Build training pairs (row i, row j > i) from existing trajectories, dt log-uniform; no new Grackle runs beyond A3. Sample pairs uniformly in (T, x, ln(nH dt)) rather than per trajectory.
  - Network: inputs (ln e, logit x, ln(nH dt)), outputs as in A4; identity as dt -> 0 built in (multiply the update by a function vanishing at dt = 0).
  - Evaluate with one jump, with k jumps (k = 2 to 100), and against the fixed-step models of A5.
  - Risk: the map is steep in dt near the cliff, so one-jump errors appear as cliff-time errors. Mitigation: sub-step near the cliff; B3's composition loss.
- Effort: 1 to 2 weeks including runs.
- Success: A2 metrics at or below A5 on all families with at most 100 evaluations per trajectory; speed-up over Grackle measured in B4.

#### B2. Rollout-aware training for the fixed-step model
- Goal: train on what is evaluated, the multi-step rollout.
- Why: report section 1 and Fig. 6; one-step accuracy does not imply rollout stability. Complements A5; less needed if B1 succeeds.
- Steps:
  - Unrolled loss over K steps with gradient through the first step only (pushforward), K increased during training.
  - Noise injection on inputs at the scale of the model's own one-step error.
  - Reuse the 3-step sketch in `Lessandre/train_test.py` as a starting point.
- Effort: 3 to 5 days plus runs.
- Success: rollout 95th-percentile error reduced against A5 at equal one-step error.

#### B3. Physics-informed terms that respect stiffness
- Goal: bring back equation-based losses in a form that trains.
- Why: report section 2.3.2 (rate and energy losses abandoned) and finding 2.6 (the rates are correct outside ionisation equilibrium).
- Steps:
  - Restore `GrackleRates` from commit `475ca3b`; unit-test k1, k2 and the cooling function against Grackle output.
  - Composition loss for B1: map(dt1 + dt2) = map(dt2) after map(dt1), on unlabelled states. Needs no rates.
  - ODE residual for B1: derivative of the map with respect to dt (autograd) equals the right-hand side at the predicted state, in ln space, skipped or replaced by x = x_eq(T) where the ionisation relaxation time is far below dt.
  - Collocation points in regions without data (tests the "physics-informed" claim: accuracy with 10x fewer trajectories).
- Effort: 1 to 2 weeks.
- Success: same A2 metrics with 10x less training data, or lower cliff-time error at equal data.

#### B4. Reference baselines and speed
- Goal: know what the network has to beat.
- Why: the report motivates the work by speed but reports no timing and no non-neural baseline.
- Steps:
  - Tabulated flow map on a (T, x, nH dt) grid with interpolation (possible because of 2.5).
  - Grackle wall-time per particle-step for the same task; network inference time on CPU and GPU, batched.
  - Grackle with a coarser tolerance as an accuracy/time reference; measure the truth's own self-consistency (2.5 suggests an integration noise floor of 1e-3 to 1e-2 in ln T).
- Effort: 2 to 3 days.
- Success: one table of accuracy vs cost for Grackle, table lookup, Model A/B, A5, B1.

#### B5. Report future-work items, re-assessed
- Equilibrium filter (4.1), corrected:
  - Goal and why: test it properly, since 2.4 shows the June attempt paired non-adjacent rows.
  - Steps: form (state, next state) pairs first, then subsample pairs by activity; keep float64.
  - Effort: 1 to 2 days. Success: compare with A5 on the A2 table.
  - Priority: low. Section 2.3 predicts it reduces but does not remove the drift.
- Mixture of Experts (4.2):
  - Recommendation: do not start. Per 2.5 the families are not separate physics, gating adds discontinuities at family borders, and the single-family run `fam3_long` already exists as a data point (score it in A2).
  - Revisit only if A5 and B1 both fail on a specific family.
- Time-adaptive data (4.3): absorbed into B1.

#### B6. Scope extensions (after the hydrogen-only case works)
- Helium species and electrons (6 species); UV background and heating (breaks the nH x t scaling, so density returns as a full input).
- ONNX export and SWIFT coupling following `Lessandre/export_onnx.py`.
- Effort: open-ended. Not planned in detail yet.

## 5. Recommended order

- A1 (structure) -> A2 (numbers for the report's models) -> A3 (clean truth) -> A4 (constraints) -> A6 (training loop) -> A5 (flat-regime fix, the direct comparison with the report)
- then B1 (flow map) -> B4 (baselines and speed) -> B3 (physics terms) -> B2 -> B5 -> B6.
- A5 and B1 are the two tasks expected to change the results; A1 to A4 and A6 make their comparison with the report valid.

## 6. Open questions

- Q1. Compute and data: which cluster and account do I use for GPU runs (job files use `env540`, `cs-302`, `master` on Izar)? Do the Izar scratch datasets still exist (`Full_Grackle_Data` 15,000 files, `Balanced_Dataset`, `Fam3_Dataset`, `Filtered_Data`), or should everything be regenerated from the IC lists?
- Q2. Which checkpoints are the report's Model A and Model B? My reading of the logs: `PINN_1` (25.0M points, 1,600 epochs) and `PINN_modelB_long` (20.0M points, 956 epochs). Last or `BEST`?
- Q3. Do you still have the multi-file `visualise.py` from 9 June (laptop, other clone)? Otherwise A2 rebuilds it.
- Q4. What is the deliverable: an updated report, a paper, a thesis continuation? Is there a deadline?
- Q5. Must the model remain a "PINN" with an explicit physics loss term? A4 makes mass conservation exact by construction, which removes the mass loss; B3 adds physics terms back in another form.
- Q6. Is the fixed 100-yr step a requirement from your supervisor or from the SWIFT use-case? Which range of time steps and densities does the target application need?
- Q7. In A1, may regenerable folders be deleted (`Plots/`, the two `Plots_Subfolder/`), and `EnvPablo/` moved outside the repo, or should everything only be moved to `archive/`?
- Q8. May the Grackle driver be changed (A3)? The regenerated truth will differ slightly from the data behind the report.
- Q9. `Active_Files/` holds the 1,000 `params_active.txt` trajectories (hot, dense region). Was this used for a model in the report, or only for the activity plots?
- Q10. After hydrogen-only works: helium and UV background first, or SWIFT coupling first?
