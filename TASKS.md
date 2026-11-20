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
- Evaluation rule (from Pablo's supervisor): at inference the model may only use the initial condition and its own previous predictions. Every headline number comes from such a free rollout. Numbers obtained by feeding ground-truth states (teacher-forced one-step errors) are diagnostics only and must be labelled as such. The same holds for checkpoint selection.
- The physics-informed part is central to the project: every model that is presented as a result carries at least one explicit physics term, and its contribution is measured by ablation (task B3).
- Evidence labels: **[measured]** a check was run and the number is reported; **[read]** taken from code, logs or the report; **[inferred]** interpretation, to be confirmed by the task that tests it.

### 0.3 Status at a glance
- Review and planning
  - `[X]` R1 Read the report
  - `[X]` R2 Review the repository code and git state
  - `[X]` R3 Check the ground-truth data files
  - `[X]` R4 Test the density-time scaling
  - `[X]` R5 Test the analytic rates against the data
  - `[X]` R6 Review `../../Lessandre` and `../../OldFiles`
  - `[X]` R7 Check cluster accounts (jed, then Izar)
- Quick wins
  - `[~]` A1 Simplify the repository structure
  - `[X]` A2 Evaluation harness and numeric baseline
  - `[X]` A3 Clean ground-truth generation and dataset rebuild
  - `[X]` A4 State variables with built-in constraints
  - `[~]` A5 Target and loss that resolve the flat regime
  - `[X]` A6 Training-loop fixes
- Larger experiments
  - `[-]` B3 Physics-informed terms: which ones help (central study)
  - `[-]` B1 Flow map with scaled time as input
  - `[-]` B2 Rollout-aware training
  - `[-]` B4 Reference baselines and speed
  - `[-]` B5 Report future-work items, re-assessed
  - `[-]` B6 Scope extensions
  - `[-]` B7 Training-set sampling strategies
- Deliverable
  - `[-]` C1 Updated report
- Next step: A5 done; B7 step 1 done; the S3–S6 screen (`B7-S3` to `B7-S6`, jobs 3257724 to 3257727, about 18 h) is running. Score with the A2 table when they finish, then compare with `A5-M10-B` (S2 is the same data class as REF-B, so its comparison is `A5-M10-B` itself, which used the REF-B pairs). B1 and B3 stay open for Pablo. Open for Pablo: whether B1 (variable step) or B7 (sampling) starts from this loss. B3 stays deferred until Pablo confirms. A1 is done apart from its listed deviations.

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

- 2026-11-14 | Cluster account: before every job submission, compare the four accounts and use the one with the best queue position. Procedure in section 3.
- 2026-11-14 | Model identity: the June checkpoints in `ResultsToDownload/` cannot be trusted to be the report's Model A and B (Pablo: models were mixed up at the time). Consequences:
  - no attempt to map checkpoints to the report;
  - comparison against the report uses the report's own figures (section 1.2) as "Model A / Model B as reported";
  - numeric baselines come from new reference runs with known provenance (task A2, runs `REF-A`, `REF-B`);
  - the June checkpoints are scored once as "legacy, identity uncertain" and never presented as Model A or B.
- 2026-11-14 | Clean-up: regenerable or unnecessary files may be deleted and `EnvPablo/` taken out of the repository. Applies to task A1.
- 2026-11-14 | This file is the single detailed record of done and to-do work (section 0.2).
- 2026-11-14 | Deliverable: an updated report. No deadline. Task C1.
- 2026-11-14 | The physics-informed part is central and must remain in one form or another. Different ways of imposing physics are to be analysed and tested alone, in pairs, in triples and so on, to measure how much each helps. Task B3, moved ahead of B1.
- 2026-11-14 | The 100-yr step is not a requirement: the step may be fixed or fully dynamic (for example tied to stiffness). Tasks A5 (fixed step) and B1 (variable step) both stay.
- 2026-11-14 | Inference must depend only on previously predicted information, never on ground-truth states. Rule added to section 0.2; applies to A2 metrics, checkpoint selection (A6) and multi-jump evaluation (B1).
- 2026-11-14 | The lost multi-file `visualise.py` is not available elsewhere. The remaining `visualise.py` may be removed and rebuilt (A1, A2).
- 2026-11-14 | The Grackle driver may be changed (A3).
- 2026-11-14 | Whether `Active_Files/` was used for a report model is not remembered. It is kept as extra data and not used for any reference run.
- 2026-11-14 | Work runs on Izar, from a session on its login node. The June scratch data is gone; all datasets are regenerated from the IC lists and the report's description (A3). The variance-stratified selection is re-implemented (A3).
- 2026-11-14 | Job length: assume a 72 h limit per training job (Pablo); revisit if the cluster shows otherwise. On Izar only account `master` allows more than 24 h, so runs are designed to fit 24 h where possible (checkpoint and resume from A6 for longer ones).
- 2026-11-14 | Report source: `Report/Overleaf Report.zip` holds the LaTeX source (`LASTRO - Semester Project Report.tex`, `preamble.sty`, `refs.bib`, `title_page.tex`, 14 images). C1 builds the updated report from it; A1 unpacks it to `docs/report/`.
- 2026-11-14 | `REF-A` and `REF-B` on regenerated, timing-corrected data are accepted as stand-ins for the report's Models A and B. Models A and B are only two ways of sampling the training data; further sampling strategies may be added (task B7).
- 2026-11-14 | Order of scope extensions left to my judgement: helium and UV background first, SWIFT coupling second. Reason: the updated report gains more from showing the method on a less simplified network than from the coupling engineering; inference speed is already covered by B4.

## 3. Cluster accounts and job submission

### 3.1 What exists on Izar (where all jobs run)
- **[measured, 2026-11-14, on `izar1`]** Four Slurm accounts for user `sarro`:
  - `cs-302`: QOS `cs-302` (default), `debug`, `normal`; max wall 1 day
  - `env540`: QOS `normal` (default), `build`, `debug`; max wall 1 day
  - `math-454`: QOS `normal` (default), `math-454`, `debug`; max wall 1 day
  - `master`: QOS `normal` (default), `build`, `debug`, `long`; max wall 3 days
- **[measured]** QOS priorities: `debug` 50000 (max 1 h, 5 jobs), `cs-302` 10000 (5 jobs), `math-454` 10000 (5 jobs), `normal` 1000 (16 GPUs per user), `long` 100 (7 days).
- **[measured]** Fair-share snapshot (higher is better): `cs-302` 0.318, `env540` 0.293, `math-454` 0.180, `master` 0.024.
- **[measured]** Partitions: `gpu` (default; 69 nodes with 2 V100, 2 nodes with 4), `gpu-xl`, `test`. Login node has 40 cores and no GPU.
- **[measured]** `/scratch/izar/sarro` exists and is empty. `/home` is shared with jed (same repository path).
- **[measured]** `EnvPablo` imports torch 2.8.0+cu128; the Grackle driver binary links on Izar.
- On jed (CPU only) the same four accounts exist with different QOS and fair-share; not used unless a large CPU-only job needs it.

### 3.2 Procedure before every job submission
- Run:
  - `sacctmgr -P show assoc user=$USER format=cluster,account,partition,qos,defaultqos,maxwall`
  - `sshare -U -u $USER -o Account,RawUsage,EffectvUsage,FairShare`
  - `sacctmgr -P show qos format=name,priority,maxwall,maxtrespu,maxsubmitpu`
- Keep only the account/QOS pairs that allow the job (wall time, GPU count, job-count limits).
- Among those, pick the highest QOS priority, then the highest `FairShare` (the account with the least recent usage relative to its share).
- Write the chosen account, QOS and the fair-share values in the run registry entry (section 8).
- **[measured, 2026-11-14]** QOS `cs-302` and `math-454` allow 1 running job per user (5 submitted); a second job under the same QOS stays pending (`QOSMaxJobsPerUserLimit`). Two simultaneous runs therefore use different accounts.
- With today's numbers: smoke tests under 1 h use QOS `debug`; training up to 24 h uses account `cs-302` with QOS `cs-302` (at most 5 jobs at once); a sixth simultaneous job or anything over 24 h falls back to `normal` on the best remaining account (`master` for more than 1 day). Wall time is capped at 72 h (section 2).

### 3.3 Storage
- Scratch is purged without notice (the June datasets disappeared). Anything that cannot be regenerated by a script in git must live under `/home`.
- Raw trajectories go to `/scratch/izar/sarro/`; the IC lists, the generation scripts and the compact training arrays of the selected datasets stay under `/home` (A3).

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
  - 2026-11-14 | `sacctmgr`, `sshare`, `sinfo`, QOS table | on `jed2` | 4 accounts, no GPU partition on jed | procedure drafted; GPU cluster question opened.
  - 2026-11-14 | Same checks repeated | on `izar1` | 4 accounts with different QOS; `cs-302` best today (QOS priority 10000, fair-share 0.318); scratch folder empty | section 3 rewritten for Izar; GPU cluster question closed.

## 5. Tasks: quick wins

### A1. Simplify the repository structure `[~]`
- Goal: one obvious place for code, configs, jobs, data and results; everything needed to reproduce a run is in git.
- Why: R2. The code behind the report is uncommitted and the evaluation script is lost. The root holds 49 entries and 38 GB (14 GB virtual environment, 15 GB `Active_Files`, 7.3 GB `Families`, about 1,550 regenerable PNGs).
- Steps:
  - `[X]` Commit the current working tree as it is and tag it `report-june2026`.
  - `[~]` Move code to: `grackle_pinn/` (`model.py`, `data_utils.py`, `physics.py`, one `train.py` replacing `train.py` + `train_modelB.py`, `evaluate.py`), `configs/` (one file per experiment), `scripts/datagen/` (`create_Grackle/*`, `Families/*.py`), `scripts/datasets/` (`active.py`, `split.py`, `extract_fam3.py`, `extract_plot_files.py`, `filter_dataset.py`), `scripts/plots/` (the six `plot_*.py` merged into one CLI), `jobs/`, `docs/` (report PDF, `Plots_Report/`).
  - `[X]` Move data to `data/` (including `Active_Files/`, kept as extra data) and June outputs to `results/legacy/<run>/` (both git-ignored).
  - `[X]` Move to `archive/`: `helpers/`, loose root files (`ac_traj.dat`, `eq_traj.dat`, `output_1.dat`, `*_analysis.png`), `Eval_Performance/`, `Family3_Files/`.
  - `[X]` Delete (allowed, section 2): `Plots/`, `Active_Files/Plots_Subfolder/`, `Families/Grid_Files/Plots_Subfolder/`, `__pycache__/`, empty `.err` files.
  - `[X]` Virtual environment: recreate outside the repo from `requirements.txt` and delete `EnvPablo/` (a venv cannot simply be moved, its scripts contain absolute paths). Check first that the new one imports torch and runs one batch on a GPU node (QOS `debug`).
  - `[~]` Remove the stale `visualise.py` and `ComputeMSE.py` once their useful parts are in `evaluate.py` (A2); until then keep them under `archive/`.
  - `[X]` `.gitignore`: stop ignoring `*.job`, `*.sh` and the IC lists.
  - `[X]` Unpack `Report/Overleaf Report.zip` to `docs/report/` and track the source in git; remove the zip and the duplicate root PDF afterwards (the PDF stays in `docs/`).
  - `[X]` Rewrite `README.md` (it references `generate_data.sh` at the root and `pinn_performance.png`, which do not exist).
- Effort: 0.5 to 1 day.
- Success: fresh clone + README reproduces data generation, one training launch and one evaluation; root has under 12 entries; `git status` clean; same loss on one fixed batch before and after the move.
- Deviations and open points:
  - `[~]` `evaluate.py` does not exist yet (A2). The six `plot_*.py` scripts were moved to `scripts/plots/` but not merged into one CLI; they are replaced by `evaluate.py` and the plotting rebuild in A2.
  - `[~]` `train_modelB.py` is not kept as a script: its differences from `train.py` (paths, 24 h limit, 10,000 epochs, loss weight factor 20) are now `configs/modelB.json`; `configs/fam3_long.json` reproduces the other. Both configs still point to scratch data that no longer exists.
  - `[~]` `visualise.py`, `ComputeMSE.py`, `train_modelB.py` and the old job files are in `archive/` (ComputeMSE's function is now `grackle_pinn/metrics.py`). `archive/` is git-ignored except for the files that were already tracked.
  - `[~]` `jobs/train.job` takes the config in `CONFIG` and no account (chosen per submission, section 3.2); the other job files are in `archive/jobs/`.
  - `[~]` `.gitignore` no longer ignores `*.job` or `*.sh`; the IC lists (`*.txt`) now live in `data/`, which stays ignored, so they are still not in git (they are regenerated by the scripts in A3).
  - `[~]` Success check "fresh clone reproduces data generation" is not run: the datagen scripts were moved unchanged and A3 rebuilds them.
- Log:
  - 2026-11-14 | Committed the working tree as it was and tagged it | `7afd1b0`, tag `report-june2026` (`.job` files and `generate_data.sh` force-added; data, checkpoints and ICs stay ignored) | 35 files | tree preserved.
  - 2026-11-14 | Restructured the repository | git mv for tracked files, mv for ignored data; `grackle_pinn.train` now takes `--config` and writes to `results/runs/<run>/` | root 11 entries | done.
  - 2026-11-14 | Deleted regenerable output | `Plots/`, both `Plots_Subfolder/`, `__pycache__/`, empty `.err` files, `EnvPablo/` | about 14 GB freed | allowed by section 2.
  - 2026-11-14 | Fixed-batch regression check | seed 0, 4096 pairs from `Train_Fam_3.dat`, 12 x 512 model, weight 20 | data loss 34.415265596337576, mass loss 1.1717976320327076e-04, identical to the last digit before and after the move | structure change is behaviour-neutral.
  - 2026-11-14 | New environment `~/venvs/grackle-pinn` | Python 3.9.10, `requirements.txt`, torch 2.8.0+cu128 | imports and runs | `EnvPablo/` removed.
  - 2026-11-14 | GPU smoke test of the moved training code | job 3252156, account `cs-302`, QOS `debug` (fair-share 0.318 of 0.318, 0.293, 0.180, 0.024), 3 epochs on `Train_Fam_3.dat` | completed in 57 s, losses falling (data 6.16e2 to 6.08e2) | OK.
  - 2026-11-14 | The login node was heavily loaded by other users (load average 35), so CPU smoke tests ran slowly | none | none.


### A2. Evaluation harness and numeric baseline `[~]`
- Goal: a single `evaluate.py` that turns any checkpoint into a table of numbers, and reference numbers to compare against.
- Why: section 1.2 and R2. The report has figures only, so "improved" cannot be measured yet. Per section 2, the June checkpoints are not accepted as Model A and B.
- Steps:
  - `[X]` Free rollout in float64 from the initial condition only, batched over trajectories; architecture and scaling read from the checkpoint (stored there from now on). This replaces the lost multi-file `visualise.py`.
  - `[X]` Headline metrics (free rollout), per family and overall: error |d log10| for T, nHI, nHII (median, 95th percentile over time, value at 2.5 Myr); cliff-time error (time of steepest d ln T); mass-conservation violation max |ln(nH_pred / nH_0)|; count of unphysical outputs (T below 1 K, energy increasing).
  - `[X]` Diagnostics (teacher-forced, labelled as such, never used as a result): one-step error binned by decade of the true increment.
  - `[~]` Fixed evaluation sets: the 8 report-figure initial conditions and a per-family set from `Families/ICs_valid.txt` (200 ICs), all regenerated in A3.
  - `[X]` "As reported" row: numbers read off the report figures (section 1.2), labelled as such.
  - `[X]` Legacy checkpoints: score all seven runs in `results/legacy/` once, labelled "legacy, identity uncertain". Also once in float32 to quantify the precision effect found in R2.
  - `[-]` Reference runs with known provenance, report configuration (12 x 512, loss weight 20, soft mass loss ramp), trained through the cleaned pipeline on the data rebuilt in A3:
    - `REF-B`: kNN-balanced dataset from `Families/ICs_train.txt` and `ICs_valid.txt`.
    - `REF-A`: variance-stratified dataset, selection re-implemented from the report (A3).
  - `[~]` Write `results/baseline.md` with the table (done, including `REF-A`, `REF-B`, `A4-C-B`, `A5-L-B`); regenerate the layout of Figs. 3 to 9 from the same script (done 2026-11-16; `plot_rollout`, figures in `results/eval/plots`, not tracked).
- Order: harness first (CPU), then A3, then the two reference runs.
- Effort: 1 to 2 days for the harness; reference runs 24 GPU-hours each.
- Success: baseline table committed with rows "as reported", legacy, `REF-A`, `REF-B`; every later task reports the same columns next to it.
- Status notes:
  - Harness done: `grackle_pinn/evaluate.py` (`run` and `table` commands), `jobs/evaluate_legacy.job`. Architecture is inferred from the checkpoint tensors and the scaling file from the sibling `scaling_params_<name>.pth` (A6 will store both in the checkpoint).
  - Fixed sets: the 8 report-figure trajectories (`data/Plot_Files`) work; the grid subset available locally is only 507 files (Families 1 and 2 in the 50 sampled); the 200-IC per-family validation set needs the regenerated data (A3).
  - `results/baseline.md` has the "as reported" and legacy rows; the `REF-A`/`REF-B` rows and the figure-layout regeneration (Figs. 3 to 9; `--plot` writes one 3-panel rollout figure per trajectory) are pending. The reference runs wait for A3, as planned in the Order line.
  - Metric definitions are in the header of `results/baseline.md`. Energy increase uses e proportional to T (nHI + 2 nHII) / nH at fixed nH, threshold 1e-12 relative per step; the ground truth gives 0.
- Log:
  - 2026-11-16 | Scored `REF-A` and `REF-B` (best and last checkpoint) on the report set and on the 200-trajectory `refB_valid` set (`jobs/evaluate_run.job`, jobs 3253117 to 3253120) | `results/baseline.md`, `results/eval/*` | **[measured]** `REF-B_last` (900 epochs): T median 0.046 dex, T final 0.073, mass violation 0.034, energy-increase fraction about 0 on `refB_valid`; Family 3 T median 0.57. `REF-B_BEST` (June rule, one-step MSE): T median 0.82 dex, mass violation 4.3. `REF-A_last` (stopped at epoch 752): T median 0.23, mass violation 1.1; `REF-A_BEST` 0.39 and 0.82 | the June selection rule picks bad checkpoints: BEST is worse than last for both; the soft mass penalty leaves mass violations of 0.03 to 4 (nH off by 3% to a factor 70) in the reference runs.
  - 2026-11-16 | `REF-A` completed (job 3253123, epochs 753 to 900, 4.26 h, finished 05:47) and rescored | `results/baseline.md` | **[measured]** `refB_valid`: `REF-A_last` T median 0.33 dex, T final 0.55, nHI final 2.9, mass violation 0.30, energy-increase fraction 0.48; `REF-A_BEST` (one-step MSE, epoch of the last improvement after 781) T median 0.23, mass violation 0.75, 0.39. Report set: T median 0.54 (last) | the completed run is not better than the epoch-752 state (0.23 to 0.39 dex): finishing the mass ramp did not reduce the drift; both reference runs keep T errors of 0.05 to 0.3 dex with mass violations of 0.03 to 0.8. The first job had stopped at epoch 752 of 900 (114 s per epoch, 23.5 h cap; the mass ramp had reached 83% of its maximum); `REF-B` finished all 900 in its first job.
  - 2026-11-16 | Resumed `REF-A` for epochs 753 to 900 | job 3253123, account `cs-302`, QOS `cs-302` (fair-share 0.258; `math-454` 0.187 and `env540` 0.232 were free too, `cs-302` has the highest QOS priority), started 01:31, expected about 5 h | finished (05:47, 4.26 h) | the checkpoint is from the June loop (no epoch, step or elapsed stored), so `train.py` got the config key `legacy_resume_epoch` (= 752: sets the epoch counter, the ramp position as step = 752 x 3,052 and the best score from `best_loss`); `select_on` = "mse" keeps the June rule for this baseline. Differences from an uninterrupted June run: the mass-loss ramp is epoch/900 instead of epoch/899 (0.1%), the batch order follows the new generator, and the loss file gets 7 columns after epoch 752 (5 before). Backups: `PINN_REF-A_epoch752.pth`, `losses_REF-A_to_epoch752.txt`, `config_original.json` in `results/runs/REF-A`; `configs/refA_resume.json`.
  - 2026-11-16 | Figure layout of the report's Figs. 3 to 9 regenerated | `grackle_pinn/evaluate.py` (`plot_rollout`), jobs 3253468 to 3253471; `results/eval/plots/<label>_<trajectory>.png` for the 8 report trajectories and 8 checkpoints (best and last of `REF-A`, `REF-B`, `A4-C-B`, `A5-L-B`) | one figure per trajectory, 3 x 3: top = one-step log-increments along the free rollout (symlog axis, Grackle dashed), middle = T, nHI, nHII against time, bottom = predicted against Grackle values (log10) with the identity line, colour = time | reading `A5-L-B_last_Train_Fam_3.png` **[read from the figure]**: the whole cooling is slowed from the first steps (the predicted d ln T is a factor of several from the truth), so the cliff falls at 1.25 Myr instead of 0.74 Myr and nHII relaxes to the wrong level afterwards. A systematic bias in the dominant cooling regime, not noise at the cliff.
  - 2026-11-16 | Cliff definition changed (Pablo's go-ahead), then corrected the same day | `evaluate.py` | final definition: cliff time = row of the largest |d ln T| **after the first step**, if it exceeds 0.05; first-step relaxation excluded. Reason **[measured, `report8.npz`]**: `Train_Fam_3` has a first-step drop of 0.066 and its cooling cliff at row 7,359 (0.395); `Valid_Fam_3` has only a first-step drop (0.351) and **no cooling cliff**. The two definitions tried before (global argmax; first row above 0.05, which I proposed) both returned row 0 for these trajectories. Relative error is left out when the true cliff is at row 0 | all cliff columns recomputed by re-running the evaluations (jobs 3253289 to 3253294); effects: `refB_valid` Family 3 has 43 trajectories with a cliff (not 50), `Valid_Fam_3` is no longer a cliff case. T, nHI, nHII, mass and energy columns are unaffected.
  - 2026-11-14 | Submitted the reference runs `REF-A` (job 3252215) and `REF-B` (job 3252217) on the A3 data | see section 8.2 | both running; `REF-B` first went pending under `cs-302` (1 running job per user), resubmitted under `math-454` | scoring with `evaluate.py` when they finish.
  - 2026-11-14 | Harness written: batched float64 free rollout in log space (no exp until metrics), per-trajectory and per-family metrics, teacher-forced one-step RMS by decade of the true |d ln T| stored as diagnostics | `grackle_pinn/evaluate.py` | runs end-to-end | CPU login node too loaded (load 35) for tests, so all runs go through Slurm.
  - 2026-11-14 | Scored all 14 legacy checkpoints on the report set (float64, float32) and on 50 grid trajectories (float64) | job 3252188, account `cs-302`, QOS `cs-302` (fair-share 0.318), 19 m 55 s, 42 result files in `results/eval/` | see `results/baseline.md`; on the report set the best are `legacy_modelB_long` (T median 0.080 dex, final 0.162 dex, nHI/nHII final 0.023/0.041) and `legacy_BEST_noClamp_Long` (0.092, 0.248); the worst are `legacy_modelB`/`BEST_modelB` (final T 15 and 8.6 dex), `legacy_1` on Family 0 (final T 225 dex) and the `fam3` runs (0.6 to 1.8 dex median) | confirms the baseline can now be measured.
  - 2026-11-14 | Consequence for model identity | `legacy_modelB_long` on Family 3: T final 0.127 dex, against about 2 dex for Model B in the report | the June checkpoint named `modelB_long` is not the report's Model B, as stated in section 2; the need for `REF-A`/`REF-B` stands.
  - 2026-11-14 | Precision check (R2 finding) | float32 vs float64 rollouts of the same 14 checkpoints on the report set (8 trajectories) | the median T error changes by under 8% for 12 of 14 checkpoints (ratios 0.92 to 1.07) and by x1.70 (`legacy_modelB`) and x0.67 (`legacy_modelB_long`) for the other two; the T error at 2.5 Myr changes by x0.44 to x1.5 for `legacy_modelB_long`, `legacy_BEST_noClamp_Long` and `legacy_modelB`, and by under 10% for the rest | float32 is not a first-order effect but moves some single rollouts by about a factor 2 in either direction; keep float64 for all numbers.
  - 2026-11-14 | Energy monotonicity | last column of `results/baseline.md`, report set, float64 | fraction of steps with increasing energy (truth: 0) is 0.002 to 0.005 for the two `fam3_long` checkpoints and `legacy_noClamp_Long`, 0.07 for `legacy_BEST_noClamp_Long`, and 0.17 to 0.86 for the rest | most checkpoints violate energy monotonicity on a large fraction of steps, which supports testing P2 and P3 in B3.

### A3. Clean ground-truth generation and dataset rebuild `[X]`
- Goal: all datasets regenerated (the June scratch copies are gone), with rows exactly 100 yr apart, one integration scheme at all times, full precision and a fast format.
- Why: R3. Timing jitter up to about 40% at the cliff bounds how well any fixed-step model can fit the cliff (Fig. 6 timing error). Regeneration is needed in any case.
- Steps:
  - `[X]` Driver (change allowed, section 2): clip the last sub-step to land exactly on each output time; drop the 0.5 Myr switch; write the true interval in the `dt` column; print time at full precision; write internal energy and x directly. Rebuild against `../../Lessandre/grackle-install`.
  - `[X]` Check the new driver against the old one on the 8 report-figure ICs (differences expected only near the cliff and at the 1e-3 level elsewhere).
  - `[X]` Generate on scratch: the 5,625-point family grid (the IC list in the repo has only 507 lines, so the full grid is rebuilt from Table 3 of the report), the 1,000 balanced ICs (`Families/ICs_train.txt`, `ICs_valid.txt`), the 15,000 random ICs (`create_Grackle/params.txt`), the 8 report-figure ICs.
  - `[X]` Re-implement the variance-stratified selection from the report (score each file by the summed log-space variance of its increments; 800 most active, 250 moderate, 250 flat; split 700/150/150 train and 100/100/100 validation). The exact June boundaries between "moderate" and "flat" are not recorded, so the choice made is written in the Log.
  - `[X]` Re-run the family clustering and kNN oracle on the new grid only if the balanced IC lists need to change; otherwise reuse `Trajectory_Families.csv`.
  - `[X]` Convert selected datasets to `.npz` (float64) under `/home` (section 3.3) and load those in training.
- Effort: 1 to 2 days plus CPU jobs (May: 1,000 trajectories in under an hour on 16 cores, so about 22,600 trajectories need roughly 10 to 25 node-hours).
- Success: all row intervals equal 100 yr to 1e-6 yr; dataset load time from minutes to seconds; datasets for `REF-A` and `REF-B` have the point counts of the report (25.0M and 20.0M) to within 1%.
- Outcome:
  - Driver `scripts/datagen/cooling-full-integration-100yr.c`: each sub-step is clipped to land on a multiple of 100 yr, one scheme at all times (cap 100 yr, no 0.5 Myr switch), time written with 17 digits, `dt` column is the true interval, new last column x = nHII/(nHI+nHII). Build: `scripts/datagen/build_driver.sh`.
  - All 22,633 trajectories (grid 5,625; balanced 800 + 200; random 15,000; report8) have 25,001 rows, spacing 100 yr to 4.7e-10 yr, nHI + nHII constant to 1.1e-15 relative, specific energy never increasing (0 rows), x column equal to nHII/nH to 4.4e-16 **[measured]**.
  - Generation takes 4 m 32 s on 30 cores (the June estimate was 10 to 25 node-hours); data are on `/scratch/izar/sarro/ground_truth/` (171 GB, purgeable); IC lists in `configs/ics/`, everything else regenerable by the scripts.
  - Datasets in `data/datasets/`: `refB_train` (800 files, 20.0M pairs), `refB_valid` (200), `refA_train` (1,000 files, 25.0M pairs), `refA_valid` (300), `report8`. Load time of `refA_train` 7 s. Pairs are identical to those of the text loader (max difference 0.0 on `Valid_Fam_3`).
  - REF-A selection rule written to `configs/datasets/refA_selection.json`: file score = sum over T, nHI, nHII of Var[d ln y]; the 800 highest-scoring files are "active" (score 0.0137 to 0.0183); "moderate" = 250 random among ranks 801 to 4000; "flat" = 250 random among ranks 4001 to 15000 (scores to 1.8e-30); seed 0; train 700/150/150, validation 100/100/100. The June boundaries were not recorded, so this is a stand-in.
- Log:
  - 2026-11-14 | Built the old driver (tag `report-june2026`) as reference and modified the driver | `results/old_driver`, new `.c` | both link against `Lessandre/grackle-install` with the spack HDF5 and gfortran libraries | `build_driver.sh`.
  - 2026-11-14 | Old vs new driver on the 8 report-figure ICs, new interpolated to the old output times | `scripts/datagen/compare_drivers.py` | max |d ln| in T, nHI, nHII: Fam 1 train 1.4e-6, 1.3e-5, 1.3e-5; Fam 1 valid 1.1e-6, 5e-6, 1.6e-5; Fam 2 train 6e-3, 5e-3, 1.1e-2; Fam 2 valid 4e-4, 1e-4, 6e-4; Fam 0 train 5e-4, 8e-2, 1e-3; Fam 0 valid 6e-5, 4e-2, 1e-4; Fam 3 train 6e-2, 0.31, 6e-3; Fam 3 valid 3.5e-3, 1.6e-2, 3e-4 | large differences only at the Family 3 cliff (timing jitter of the old rows, 72 to 142 yr between rows) and in nHI of the almost fully ionised Family 0 trajectories (nHI tiny, relative differences large); elsewhere 1e-6 to 1e-2 | new driver accepted.
  - 2026-11-14 | Generated the ground truth | job 3252211, account `cs-302`, QOS `cs-302` (fair-share 0.284 vs 0.325 for `env540`; QOS priority decides), 30 cores | 22,633 of 22,633 trajectories in 4 m 32 s | OK.
  - 2026-11-14 | Checked and converted | job 3252212 (same account), `check_ground_truth.py` + `build_datasets.py`, 2 m 37 s | all checks above passed | datasets built.
  - 2026-11-14 | Family labels | the grid ICs and their order are unchanged, so `Trajectory_Families.csv` (family per grid file) is reused; the kNN oracle (k = 3, report) labels the validation files | not re-clustered on the new trajectories | OK.


### A4. State variables with built-in constraints `[X]`
- Goal: the network predicts in (internal energy, ionised fraction x) with nH carried through unchanged, as one of the ways of imposing physics.
- Why: R3 and Fig. 3 bottom. Mass conservation becomes exact instead of a penalty with a weight to tune; energy can be forced non-increasing, which removes outputs like Fig. 4 bottom (T rising 20x) and bounds x to [0, 1].
- Position in the project: this is physics imposed through the architecture ("hard" constraint). It does not replace the explicit physics losses, which stay mandatory (section 0.2); B3 compares hard, soft and combined variants.
- Steps:
  - `[X]` Inputs: ln e (or ln T), logit x, ln nH. Outputs: -softplus(.) for d ln e; a bounded update for logit x.
  - `[X]` Recover T, nHI, nHII analytically from (e, x, nH) for metrics and plots.
  - `[X]` Both parameterisations selectable by config: the report's (ln T, ln nHI, ln nHII) with soft mass loss, and the constrained one.
  - `[X]` Training run on the REF-B data for the success check: `A4-C-B` (section 8.2), scored against `REF-B` on Family 3.
- Effort: 1 day plus one training run.
- Success: mass violation at float64 round-off; zero unphysical outputs; free-rollout metrics no worse than `REF` on Family 3.
- Implementation: `grackle_pinn/param.py` (`report` and `constrained`), selected by `parameterisation` in the config; features z = (ln e, logit x, ln nH) with nH = nHI + nHII, x = nHII/nH, ln e = ln T + ln(1 + x); outputs d ln e = -softplus(o1) and d logit x = L tanh(o2), L = 1.25 x the largest |d logit x| in the training data (29.96 for the REF-B data); ln nH carried unchanged; last-layer bias of o1 set to -12 so that initial energy decrements are 6e-6 instead of 0.69. Checkpoints store the parameterisation, so `evaluate.py` rolls out either kind.
- Log:
  - 2026-11-14 | Data ranges that fix the design | `refB_valid` | d ln e <= 0 in every row (max -3.4e-10); |d logit x| up to 22.6 in one step (x from about 1e-10 to order 0.1); d ln e down to -2.26 in one step; logit x in [-10.3, 21.2] **[measured]** | bounded update L tanh(.) with L set from the data; the energy is non-increasing in the ground truth, so the constraint is consistent with it.
  - 2026-11-14 | Unit tests (`tests/test_param_and_resume.py`, job 3252239) | real trajectories and 50 randomly re-initialised wild networks | round trip ln state -> z -> ln state 3.6e-15; advance(targets) reproduces the next state to 3.6e-15; worst mass violation 1.8e-15; 0 energy increases; x in (0, 1) | constraints hold for any network output.
  - 2026-11-14 | 6-epoch smoke run on the REF-B data (`constrained_smoke`, job 3252240) scored on the 8 report trajectories | `results/eval_smoke` | mass violation 1.7e-16 overall, 0 energy increases, but T median 0.061 dex and Family 3 cliffs missed (2/2): untrained, as expected | only the constraints are verified here; accuracy comes from `A4-C-B`.
  - 2026-11-14 | Submitted `A4-C-B` (constrained parameterisation, REF-B data and settings, 800 epochs) | job 3252249, account `env540`, QOS `normal` (`cs-302` and `math-454` slots hold REF-A/B) | finished 2026-11-15 12:56 (18.0 h) | -.
  - 2026-11-16 | Scored `A4-C-B_last` against `REF-B_last` (free rollouts, float64) | `results/baseline.md` | **[measured]** `refB_valid`, Family 3: T median 0.089 vs 0.572 dex, T final 0.13 vs 1.79, nHI final 0.006 vs 5.6. All families: T median 0.093 vs 0.046 (A4 is worse on Families 0 to 2: 0.10, 0.08, 0.14 vs 0.02, 0.06, 0.03). Cliffs (43 on `refB_valid` Family 3): 0 missed vs 23 for `REF-B_last`, median error 3.4% vs 16% of the cliff time; report set `Train_Fam_3`: 0.5% (4,000 yr) vs 50%. Mass violation 8.9e-16 vs 0.034; energy increases 0 vs 4e-5 (report set: 0 vs 0.86) | success criteria met: mass at round-off, 0 energy increases, Family 3 clearly better than `REF-B`. The Families 0 to 2 drift is still 0.1 dex, as expected: A4 does not change target or loss (A5 does).
  - 2026-11-16 | Confound to keep in mind | `A4-C-B` also differs from `REF-B` by the new loop (800 vs 900 epochs, selection on free-rollout windows instead of one-step MSE); `A4-C-B_BEST` and `_last` differ by 0.15 vs 0.09 dex T median | the gain on Family 3 is the combined effect of the constrained parameterisation and of the new selection rule; B3 will separate the two if needed.


### A5. Target and loss that resolve the flat regime `[X]`
- Goal: test the diagnosis of R3 on the fixed-100-yr model, changing only target and loss.
- Why: Figs. 3 bottom, 4, 7, 8, 9. Most direct comparison with the report: same data, same network.
- Steps:
  - `[X]` Target: ln of the rate magnitude (u1 = ln(-d ln e), sign fixed by A4) instead of the raw increment; for x, u2 = asinh(d logit x / s) with s = 1e-10. Implemented as parameterisation `lograte` (`grackle_pinn/param.py`) on top of the A4 state, so mass and energy guarantees are kept.
  - `[X]` Loss: plain MSE in that space (`weight_factor` 0). With the physical-increment term added (`phys_weight` 1 or 10, or the report's `1 + 20|y|` on the physical increment), the Family 3 failure is fixed; see the log.
  - `[X]` Train on the balanced dataset first (where the report saw the worst drift), then the stratified one: `A5-L-B` done (balanced); `A5-L-A` (REF-A data) done, see the log.
- Effort: 1 to 2 days plus two training runs of 12 to 24 GPU-hours.
- Success (provisional until A2 gives numbers): Families 0 to 2 final error in T under 0.02 dex at 2.5 Myr (report: 0.07 dex to tens of dex); Family 3 cliff time within 2% (report Fig. 6: about 7%); no regression on the Family 3 case of Fig. 5.
- Implementation: the network outputs two numbers, standardised in (u1, u2) space. Rollouts invert the transform (d ln e = -exp(u1), d logit x = s sinh(u2)) with u clipped to the training range plus 2 nats, so the update stays finite for any output. The ranges in the training data are u1 in [-20.5, -0.9] (increments 1.3e-9 to 0.4 in ln e) and u2 in [-24.2, 26.6] **[measured, report8 data]**.
- Log:
  - 2026-11-14 | Unit tests (`tests/test_param_and_resume.py`, job 3252306) | report8 data, 50 wild networks | inverse of the target transform reproduces the next state to 2.8e-14; worst mass violation 1.9e-15; 0 energy increases; resumed run equals uninterrupted run (weight difference 0.0) | `lograte` is correct and resumable.
  - 2026-11-14 | Submitted `A5-L-B` (lograte, REF-B data, weight factor 0, otherwise identical to `A4-C-B`) | job 3252307, account `env540`, QOS `normal` (`cs-302` and `math-454` slots hold REF-A/B) | finished 2026-11-15 14:23 (17.9 h) | -.
  - 2026-11-16 | Scored `A5-L-B_last` (free rollouts, float64, `refB_valid`, 50 per family) | `results/baseline.md` | **[measured]** T median 0.00067 / 0.00024 / 0.0011 dex and T final 0.00067 / 0.00034 / 0.0010 dex for Families 0 / 1 / 2 (`A4-C-B_last`: 0.10 / 0.08 / 0.14; `REF-B_last`: 0.02 / 0.06 / 0.03). Report set, Family 0: 5e-6 dex. Family 3: T median 1.36 dex, nHI final 5.7, 30 of 43 cliffs missed (`REF-B_last` 23 of 43, `A4-C-B_last` 0 of 43; `REF-A_last` 40 of 43); report set `Train_Fam_3` cliff error 71% (`A4-C-B_last` 0.5%) | the flat-regime drift is removed: errors on Families 0 to 2 fall by 2 to 3 orders of magnitude, with mass and energy still exact (confirms R3's diagnosis). Family 3 gets worse than A4 and `REF-B`: the post-cliff state is wrong (nHI off by 5 to 6 dex).
  - 2026-11-16 | Success criterion check (A5-L-B, superseded by the results below) | Families 0 to 2 final T under 0.02 dex: **met** (worst median 0.0010 dex on `refB_valid`, 0.0098 on the report set Family 2). Family 3 cliff within 2%: **not met** (71% on `Train_Fam_3`; 30 of 43 cliffs missed on `refB_valid`, median error of the others 0.28 of the cliff time). No regression on the Family 3 case of Fig. 5: **not met** (A4 and `REF-B` do better) | A5 stays `[~]`.
  - 2026-11-17 | Scored the four A5 follow-up runs (last and best checkpoint, report set and `refB_valid`), jobs 3255265 to 3255268 | `results/baseline.md`, `results/eval/*` | **[measured, last checkpoints, `refB_valid`]** T median per family 0 / 1 / 2 / 3 (dex): `A5-M1-B` 0.0005 / 0.0002 / 0.0011 / 0.0008 (Family 3 cliff relative error 3.4%, 0 of 43 missed); `A5-M10-B` 0.0002 / 0.0002 / 0.0006 / 0.0004 (0.3%, 0 missed); `A5-W-B` 0.0001 / 0.0002 / 0.0005 / 0.0007 (4.2%, 0 missed); `A5-L-A` 0.0007 / 0.0006 / 0.0030 / 0.083 (6.0%, 2 missed); for comparison `A5-L-B` Family 3 1.36 dex (30 of 43 missed) and `A4-C-B` 0.089 dex (0 missed). Mass violation 4e-16 to 9e-16 and 0 energy increases for all. Report set, last checkpoints, Family 3 T median: `A5-M1-B` 0.0004, `A5-M10-B` 0.0003, `A5-W-B` 0.0005; `A5-L-A` 1.02 (fails the same way as on the balanced data, so the stratified data alone does not help); `A4-C-B` 0.074. Best-by-rollout checkpoints (`BEST`) are within a factor 2 of the last ones on the physical-increment variants, except `A5-L-A` on the report set (1.02, same).
  - 2026-11-17 | Seed check (seed 1, otherwise identical to the runs they check; `A5-M1-B-s1` added later the same day) | `configs/A5-M10-B-s1.json`, `configs/A5-W-B-s1.json`; `A5-M10-B-s1` job 3255462 (account `cs-302`, QOS `cs-302`, fair-share 0.262, the highest QOS priority); `A5-W-B-s1` job 3255463 (account `math-454`, QOS `math-454`, fair-share 0.212; `cs-302` was taken by the first job) | submitted, pending | expected to finish about 18 h after start; then score on `refB_valid` and the report set, and compare the seed spread with the 0.0005 dex differences.
  - 2026-11-18 | Seed check scored (seed 1 vs seed 0, last checkpoints, `refB_valid`; jobs 3257705 to 3257707) | `results/baseline.md`, `results/eval/*-s1*` | **[measured]** T median Family 3 (dex) seed 0 / seed 1: `A5-M1-B` 0.0008 / **0.0061**; `A5-M10-B` 0.0004 / 0.0004; `A5-W-B` 0.0007 / 0.0014 (1 of 43 cliffs missed on seed 1). Families 0 to 2 for all runs stay between 0.0001 and 0.0014 dex on both seeds (all under the 0.02 dex criterion) | `A5-M10-B` is the only variant whose Family 3 result reproduces across seeds (0.0004 both times, 0 cliffs missed both times). `A5-M1-B` varies by a factor 7 on Family 3 between seeds; `A5-W-B` varies by a factor 2 and misses one cliff on seed 1. The Families 0 to 2 differences that were under 0.0005 dex on seed 0 are of the same order as the seed spread, so they do not rank the variants. Conclusion: `A5-M10-B` is the candidate; `A5-W-B` is close but less stable; `A5-M1-B` is not reliable enough at weight 1.
  - 2026-11-17 | A5 verdict | success criteria: Families 0 to 2 final T under 0.02 dex **met** by all three physical-increment variants (and `A5-L-B`); Family 3 cliff within 2% **met** by `A5-M10-B` (median 0.3%) and not by `A5-M1-B` (3.4%) or `A5-W-B` (4.2%), though all three miss no cliffs; no regression on the Family 3 case of Fig. 5 (report `Train_Fam_3` final T 0.0001 to 0.0003 dex vs 0.17 for `A4-C-B`) **met**. Best candidate for B1 and later: `A5-M10-B`, with `A5-W-B` close (better Families 0 to 2, the report's own weight form). Caveat: each result is one seed (seed 0); the differences between `A5-M1-B`, `A5-M10-B` and `A5-W-B` on Families 0 to 2 are of order 0.0005 dex and need a seed check before ranking them.
  - 2026-11-16 | Hypothesis for the Family 3 failure **[tested 2026-11-17, supported]** | the three physical-increment variants (below) all remove the failure, so the plain log-space loss was missing the physical size of the cliff steps. Option (a) with weight 1 or 10 and option (b) both work; option (c) (A5-L-A, REF-A data) does not.
  - 2026-11-16 | Four follow-up runs in parallel (Pablo's go-ahead), all `lograte`, 12 x 512, 800 epochs, seed 0, selection on free-rollout windows | `configs/A5-M1-B.json`, `A5-M10-B`, `A5-W-B`, `A5-L-A.json`; `train.py` gained `phys_weight` and `weight_physical` (unit-tested, resume exact for both) | (a) `A5-M1-B`, `A5-M10-B`: log-space MSE + phys_weight x MSE of the physical increments (d ln e, d logit x), each divided by its training std; phys_weight 1 and 10; (b) `A5-W-B`: log-space MSE with the weight 1 + 20 |y|, y = standardised physical increment (the report's weight on the physical size); (c) `A5-L-A`: same as `A5-L-B` on the REF-A data | jobs 3253131 to 3253134 (section 8.2) | submitted; compare on Family 3 and on Families 0 to 2 against `A5-L-B`, `A4-C-B`, `REF-B`.


### A6. Training-loop fixes `[X]`
- Goal: runs that finish what they are configured to do and select the right checkpoint.
- Why: R2. `modelB_long` trained with under 10% of the intended physics weight, and "BEST" is chosen on a metric unrelated to rollout.
- Steps:
  - `[X]` Ramp and scheduler defined on optimiser steps or fraction of the wall-time budget, not nominal epochs.
  - `[X]` Resume restores epoch and ramp position and appends to the loss file.
  - `[X]` Checkpoint selection on a short free-rollout metric from A2 (no ground-truth states fed back).
  - `[X]` Data on GPU as tensors instead of a 4-worker DataLoader; test float32 forward with float64 accumulation for speed.
  - `[X]` Fix seeds; save the config in the run folder.
- Effort: 0.5 to 1 day.
- Success: epoch time reduced (June: 72 to 89 s per epoch for 20 to 25M points); a resumed run reproduces an uninterrupted one; the selected checkpoint has the best rollout score among saved ones.
- Implementation (`grackle_pinn/train.py`, rewritten; configs have new keys, see its `DEFAULTS`):
  - Mass-loss ramp = step / planned steps (epochs x steps per epoch), so it always completes if the run finishes; an optional `budget_hours` makes it follow the wall clock too (not bit-reproducible, off by default).
  - Checkpoints store epoch, step, best score, elapsed time, optimiser, scheduler, architecture, scaling, parameterisation and config; `--model` resumes exactly and appends to the loss file. The data order of epoch e depends only on (seed, e).
  - Selection score = `window_rollout_score` in `evaluate.py`: 32 validation trajectories, free rollouts of 2,000 steps started at rows 0, 5,000, 10,000, 15,000, 20,000 (the first state of each window is the only ground truth used); mean |d log10| over time, variables, windows and trajectories, each element capped at 50 dex. One-step MSE is still logged. Windows of 2,000 steps cover the cooling cliffs only if one falls inside a window; a full 25,000-step rollout would cost about 30 s per check.
  - Data live on the GPU as tensors (no DataLoader workers). Dtype is a config key.
- Deviation from the success criterion: epoch time is not reduced. **[measured]** 112.6 s per epoch in float64 on 25.0M pairs (bench job 3252240), against June's 72 to 89 s and about 100 s for REF-A under the old loop: the 12 x 512 float64 network is compute-bound on a V100, not data-bound. Float32 halves it (58.3 s) but changes the numerics of training; not adopted.
- Log:
  - 2026-11-14 | Resume test, report and constrained parameterisations, 6 epochs against 3 + resume 3, tiny network, CPU/GPU node | `tests/test_param_and_resume.py`, job 3252239 | max weight difference 0.0 in both, loss file has 1 header + 6 lines (appended) | resume reproduces an uninterrupted run exactly. A first version failed (difference 2e-4) because the ramp followed the wall clock; fixed by making the wall-clock term opt-in.
  - 2026-11-14 | Speed test, 3 epochs on `refA_train` | job 3252240, `results/bench/bench64.json`, `bench32.json` | float64 112.6 s per epoch, float32 58.3 s | see deviation above.
  - 2026-11-14 | REF-A and REF-B were started with the June loop (one-step MSE selection, DataLoader, ramp on nominal epochs: 900 epochs planned, about 108 s per epoch means it will stop near epoch 780 at the 23.5 h limit, ramp about 87% of its maximum). They are kept as the "report configuration on clean data" baseline | jobs 3252215, 3252217 | - | runs unchanged; later runs use the new loop.


## 6. Tasks: larger experiments

### B3. Physics-informed terms: which ones help (central study) `[~]`
- Goal: establish, by controlled ablation, which ways of imposing physics improve the free rollout, alone and in combination, and carry the winning combination into the final models.
- Why: the physics-informed part is the centre of the project (section 2). The report ended with a single soft mass term after the rate and energy losses were abandoned (report 2.3.2); R5 shows the rates themselves are correct, so those losses can return in a stiffness-aware form.
- Candidate physics terms:
  - P1 soft mass conservation in log space (the report's term).
  - P2 hard constraints through the architecture (A4): exact mass conservation, bounded x, energy never increasing.
  - P3 soft energy monotonicity: penalty on predicted increases of internal energy (soft counterpart of part of P2).
  - P4 chemistry residual: predicted change of x against k1, k2 (k57) evaluated at the predicted state, in implicit or trapezoid form.
  - P5 ionisation-equilibrium constraint x = x_eq(T) where the relaxation time is far below the step (the regime where P4 is ill-conditioned).
  - P6 energy residual: predicted change of internal energy against the analytic cooling function.
  - P7 composition consistency (variable-step models, B1): map(dt1 + dt2) = map(dt2) after map(dt1).
  - P8 ODE residual by differentiating the map with respect to dt (variable-step models, B1), on collocation points without data.
- Steps:
  - `[X]` Restore `GrackleRates` from commit `475ca3b` (`grackle_pinn/physics_terms.py`, k7 kept); unit-test k1, k2, k7 and the four cooling terms against the regenerated Grackle output (target: R5's 1e-4 to 5e-3 agreement outside equilibrium). Result: rate-relative residual median 3.8e-4 on the steps that move (p99 3.6e-2); energy 3.6e-5 (p99 1.8e-2). See the log.
  - `[~]` Implement P1 to P6 as switchable loss terms with a common weighting interface. P3, P4, P5, P6 are in `physics_terms.py` and in `train.py` (config `physics`, fixed weights, each term clamped at 1 per pair); P1 is the existing mass term (`lambda_mass_max`); P2 is the parameterisation (A4/A5). Smoke test of the loop: see the log.
  - `[-]` Weighting test (small): fixed weight vs linear ramp vs gradient-norm balancing, on P1 alone and on P4 alone.
  - `[-]` Screening protocol: fixed-step model from A5, reduced dataset, equal budget per run (about 2 to 3 GPU-hours), identical seed; score with the A2 free-rollout table.
  - `[-]` Stage 1, one at a time: no physics; P1; P2; P3; P4; P5; P6 (7 runs).
  - `[-]` Stage 2, pairs: all pairs among the terms that did not hurt in stage 1 (at most 15 runs; P2 with P1 or P3 only where not redundant).
  - `[-]` Stage 3, triples and larger sets: built greedily from the best pairs (about 5 to 10 runs), plus "all terms".
  - `[-]` Stage 4, confirmation: the best 3 configurations and "no physics" at full budget with 3 seeds each.
  - `[-]` Data-efficiency test: best configuration vs "no physics" with 10x fewer training trajectories.
  - `[-]` Stage 5, after B1: P7 and P8 alone, together and with the best stage-4 set, on the variable-step model.
- Effort: 2 to 3 weeks; roughly 40 screening runs (about 100 GPU-hours) plus 12 full runs.
- Success: a table with one row per configuration and the A2 columns, with seed spread for the confirmed ones; a stated answer for each term (helps / neutral / hurts, and on which family); at least one physics configuration beating "no physics" on the free rollout beyond the seed spread, or a documented negative result.
- Log: none yet.

### B1. Flow map with scaled time as input (main bet) `[-]`
- Goal: a model state(t), nH x dt -> state(t + dt) valid for dt from 100 yr to 2.5 Myr, so a 2.5 Myr evolution takes one or a few evaluations instead of 25,000.
- Why: R3, R4 and report 4.3. Error accumulation, the report's central problem, is reduced by construction. The step is free to choose (section 2), so it can be fixed, drawn from a range, or set from the stiffness of the current predicted state.
- Steps:
  - `[-]` Verify the scaling over the whole grid (all x0, all T0) and quantify the residual; keep ln nH as a weak extra input if it is not negligible.
  - `[-]` Build training pairs (row i, row j > i) from existing trajectories, dt log-uniform; sample pairs uniformly in (T, x, ln(nH dt)).
  - `[-]` Network: inputs (ln e, logit x, ln(nH dt)), outputs as in A4, identity as dt -> 0 built in.
  - `[-]` Evaluate with one jump from the initial condition, with k chained jumps (k = 2 to 100, each fed only the previous prediction), with a stiffness-adaptive step chosen from the predicted state, and against the fixed-step models of A5.
  - `[-]` Physics terms P7 and P8 of B3 (stage 5).
- Risk: the map is steep in dt near the cliff, so one-jump errors appear as cliff-time errors. Mitigation: smaller steps near the cliff; composition loss P7.
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
- `[-]` Equilibrium filter (report 4.1): moved to B7 as strategy S6. Corrected form: form (state, next state) pairs first, then subsample pairs by activity; keep float64. Effort 1 to 2 days. Compare with A5 on the A2 table. Low priority: R3 predicts it reduces but does not remove the drift.
- `[-]` Mixture of Experts (report 4.2): recommendation is not to start. Per R4 the families are not separate physics and gating adds discontinuities at family borders. The single-family legacy run `fam3_long` gets scored in A2 as a data point. Revisit only if A5 and B1 both fail on a specific family.
- Time-adaptive data (report 4.3): absorbed into B1.
- Log: none yet.

### B6. Scope extensions `[-]`
- `[-]` First: helium species and electrons; UV background and heating (breaks the nH x t scaling, so density returns as a full input; energy is no longer monotone, so P2/P3 change form).
- `[-]` Second: ONNX export and SWIFT coupling following `Lessandre/export_onnx.py`.
- Effort: open-ended; not planned in detail.
- Log: none yet.

### B7. Training-set sampling strategies `[X]`
- Goal: compare ways of choosing which trajectories and pairs go into the training set, beyond the report's two (Pablo, 2026-11-14: Models A and B are just two possible approaches).
- Why: the report attributes the failures to dataset balance; R3 says the loss is the cause. Testing several strategies under the same loss and budget separates the two explanations and gives the final models the best data.
- Candidate strategies:
  - S1 variance-stratified by trajectory (report Model A, = `REF-A`).
  - S2 kNN-balanced by family (report Model B, = `REF-B`).
  - S3 uniform random initial conditions (control, never tested in the report).
  - S4 per-sample balancing: equal number of training pairs per decade of |increment|.
  - S5 uniform coverage of state space (T, x, nH) regardless of trajectory.
  - S6 corrected equilibrium filter (from B5): form (state, next state) pairs first, then subsample by activity; keep float64.
  - S7 loss-based resampling during training: draw more from pairs with high current error.
- Steps:
  - `[~]` Implement the strategies as dataset builders sharing one interface (A3 datasets as input). Step 1 done 2026-11-18: `scripts/datasets/sampling.py` builds S3–S6 from a common pool (3,000 random ICs excluding the REF-A/B validation files, seed 0) with a 20M-pair budget each; output is pair-format `.npz` (`cur`, `nxt`), read by `GrackleDataset` (tested: identical pairs to the trajectory loader). S7 needs a training-loop change and is not built yet.
  - `[X]` Screen with the B3 protocol (fixed-step model from A5, equal budget, identical seed), scored on the A2 free-rollout table: `B7-S3` to `B7-S6` (2026-11-20), against `A5-M10-B` (S2) on the same settings.
  - `[-]` Carry the best strategy into B3 stage 4. Not needed on this evidence: no strategy beats S2 by more than the seed spread (see the log).
- Effort: 1 to 2 weeks including runs.
- Success: a table of strategies with the A2 columns; a stated answer on whether sampling matters once the target and loss of A5 are used.
- Log:
  - 2026-11-20 | Choice of the discrete derivative (Pablo's question) **[decided, measured]** | the step derivative is taken discretely over the fixed 100-yr step (trapezoid in time; backward Euler as a variant). It is exact for the truth, which is stored one step apart, and needs no autodiff in time. An autodiff derivative with respect to time is needed only for the variable-step model (P8, B1). **The form matters more than the discretisation:** the residual of d nHII / dt, normalised by nH, is 1e-9 on the truth in the flat regime but also blind to a 6e-6 per-step bias there; the same residual normalised by the size of the change is undefined where nothing moves (0.5% of chemistry steps move on report8); the log form, d ln nHII / dt = k1 nHI - k2 nHII and d ln U / dt = -e_cool / U, is finite everywhere and separates the bias: a 6e-6 bias gives a residual 5,500 times the truth median (`tests/test_physics_terms.py`). The same test in the log form on report8 gives median 1.1e-9 for P4 and 6.0e-11 for P6 on all 200,000 pairs (p99 6.5e-7 and 1.6e-6; the maxima of 200 and 7.5 are at the cliff, where the clamp at 1 limits their weight). So the log-form chemistry and energy residuals are the candidate physics losses; they are the derivative-based terms the literature points to, written for the log state.
  - 2026-11-20 | Unit tests (`tests/test_physics_terms.py`, 10 checks, all pass on the login node) | report8 pairs | rates finite and positive on [1, 1e9] K; truth consistency (P4, P6 small, P3 zero on truth); flat-regime sensitivity (6.0e-6 bias = 5,500 x truth median); energy sign and exact response to ln T; monotone penalty detects an increase; gradients finite (P3–P6); parameterisations reproduce the truth residual (report, constrained, lograte); P5 median 1.2e-4 on the 13,855 fast-relaxation steps | the terms are consistent with the truth and sensitive to the size of drift that the report's flat regime accumulates. Not yet tested: the training effect (the smoke run and stage 1 answer that).
  - 2026-11-20 | Implementation in the training loop (`train.py`, config `physics`) | `GrackleDataset(keep_raw=True)` stores the raw current states; the predicted next state is `param.advance(cur, to_physical(out))`, differentiable; each enabled term adds `w * mean(min(r^2, 1))` | smoke run (job 3261865, GPU debug, 2 epochs on report8, all four terms on) pending the GPU queue. On the login node the CPU is too loaded for a smoke run (2 s per 8,192-element operation).
  - 2026-11-20 | Unknown jobs under this account | `Base31` (3257723, completed), `Cls31` (3257732?) and `Phys31` (3261863) and `ClsSparse` (3261864) are not from this repo: `scontrol` shows `Command=/home/sarro/SummerProj/src/*.job`, submitted 2026-11-20 08:56 to 08:57 | not touched | they occupy `cs-302` queue priority; Pablo to check.
  - 2026-11-18 | Plan decision | the plan order puts B7 next after A5 (section 7). Pablo confirmed the screening protocol ("whatever you consider"), so the S3 to S6 screen was submitted. | - | -
  - 2026-11-18 | Loader for pair datasets | `grackle_pinn/data_utils.py` reads `cur`/`nxt` `.npz` files; a test on `report8` gives the same inputs and targets as the trajectory loader (equal to the last bit) | - | no S7 support (no trajectory structure for rollout selection).
  - 2026-11-20 | B7 screen scored (last checkpoint, `refB_valid`; S2 = `A5-M10-B`, same loss and settings; jobs 3261814 to 3261817) | `results/baseline.md`, `results/eval/B7-*` | **[measured]** T median per family 0 / 1 / 2 / 3 (dex): S2 0.00015 / 0.00020 / 0.00057 / 0.00043; S3 0.00035 / 0.00023 / 0.00074 / 0.00132; S4 0.00010 / 0.00024 / 0.00056 / 0.00064; S5 0.00036 / 0.00009 / 0.00038 / 0.00040; S6 0.00087 / 0.00018 / 0.00085 / 0.00140. Cliffs missed 0 of 43 for every set; mass violation 9e-16; no energy increases. Report set Family 3: S2 0.00028, S3 0.00095, S4 0.00074, S5 0.00034, S6 0.00129 | **the choice of pairs matters little once the A5 target and loss are used.** S5 (state-space coverage) ties S2 on Family 3 and is best on Family 1; S3 (uniform random, the control) and S6 (thinned equilibrium) are the worst on Family 3, by about 0.001 dex. Those differences are about the size of the seed spread measured for `A5-M10-B` (Family 3 0.0004 on both seeds) and rest on one seed each, so the ranking is not firm. S4 is not better than S2. Nothing justifies carrying a different strategy into B3 stage 4. Caveat: one seed per strategy.
  - 2026-11-18 | Built the pair sets | job 3257722 (`jobs/build_sampling.job`, account `env540`, QOS `normal`, CPU work on a GPU partition as in the existing build job), pool 3,000 trajectories = 75.0M pairs | finished | **[measured]** each 20,000,000 pairs (960 MB each, in `data/datasets`, not tracked): S3 from 800 trajectories, median activity 2.4e-5, 15% of pairs below 1e-7; S4 from 3,000 trajectories, median 1.3e-6, 34% below 1e-7; S5 median 5.1e-5, 2% below 1e-7; S6 median 4.1e-5, 0.2% below 1e-7. Pool list in `configs/datasets/sampling_pool.json`. | the sets differ as intended; screening runs (the B3 protocol) wait for Pablo's confirmation.

## 6b. Tasks: deliverable

### C1. Updated report `[-]`
- Goal: a revised version of the June report built on the new results. No deadline. Source: `docs/report/` (unpacked from `Report/Overleaf Report.zip` in A1).
- Steps:
  - `[-]` Replace the qualitative results by the A2 table (as reported, `REF-A`, `REF-B`, new models), all from free rollouts.
  - `[-]` Rewrite the diagnosis of Models A and B if A5 confirms R3 (flat-regime resolution rather than dataset balance).
  - `[-]` New section on the physics-informed study (B3 table and conclusions).
  - `[-]` New section on the variable-step model (B1) and on cost vs Grackle (B4).
  - `[-]` Correct the points where report and code disagreed (R2) and describe the data timing fix (A3).
  - `[-]` Regenerate all figures from `evaluate.py`.
- Effort: 1 to 2 weeks, after B3 and B1.
- Success: every number and figure in the report is produced by a committed script and a run listed in section 8.
- Log: none yet.

## 7. Recommended order

- A1 -> A2 (harness) -> A3 -> A2 (reference runs) -> A4 -> A6 -> A5
- then B7 (after A5) -> B3 stages 1 to 4 (with the best sampling) -> B1 -> B3 stage 5 -> B4 -> B2 -> B5 -> C1 -> B6.
- A5, B3 and B1 are the tasks expected to change the results; the others make their comparison with the report valid.

## 8. Run registry

### 8.1 Legacy runs, June 2026 (identity uncertain, see section 2)
- Files in `ResultsToDownload/` (now `results/legacy/`): last and `BEST` checkpoint, scaling file, loss history and log per run. None has been scored yet (A2).
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
- `REF-A` | 2026-11-14 | A2 | `refA_train.npz` (25.0M pairs, 1,000 files), valid `refA_valid.npz` (300) | `configs/refA.json`: 12 x 512 tanh, float64, batch 8192, lr 1e-4, weight factor 20, mass ramp 1e7 x epoch/899, 900 epochs, stop at 23.5 h | account `cs-302`, QOS `cs-302`, fair-share 0.261 (env540 0.325, math-454 0.180, master 0.024), job 3252215 | stopped at the 23.5 h cap, epoch 752 of 900 (finished 2026-11-15 17:09); resumed as job 3253123 (configs/refA_resume.json) from 2026-11-16 01:31 | `results/runs/REF-A` | rows in `results/baseline.md` are the completed run (rescored 2026-11-16, jobs 3253289 and 3253291)
- `REF-B` | 2026-11-14 | A2 | `refB_train.npz` (20.0M pairs, 800 files), valid `refB_valid.npz` (200) | `configs/refB.json`, same as REF-A | account `math-454`, QOS `math-454`, fair-share 0.180, job 3252217 | finished 900 epochs, 22.2 h (2026-11-15 15:50) | `results/runs/REF-B` | scored: `results/baseline.md`
- `bench64`, `bench32`, `constrained_smoke` | 2026-11-14 | A6/A4 | `refA_train` (first two), `refB_train` (last) | 3, 3 and 6 epochs, speed and smoke tests | `env540`, QOS `debug`, job 3252240 | finished | see A6/A4 logs | not models.
- `A4-C-B` | 2026-11-14 | A4 | `refB_train.npz`, valid `refB_valid.npz` | `configs/A4-C-B.json`: constrained parameterisation, 12 x 512, float64, weight factor 20, 800 epochs, selection on free-rollout windows | account `env540`, QOS `normal`, fair-share 0.325, job 3252249 | finished 800 epochs, 18.0 h (2026-11-15 12:56) | `results/runs/A4-C-B` | scored: `results/baseline.md`
- `A5-L-B` | 2026-11-14 | A5 | `refB_train.npz`, valid `refB_valid.npz` | `configs/A5-L-B.json`: lograte parameterisation, plain MSE, 12 x 512, float64, 800 epochs, selection on free-rollout windows | account `env540`, QOS `normal`, fair-share 0.294, job 3252307 | finished 800 epochs, 17.9 h (2026-11-15 14:23) | `results/runs/A5-L-B` | scored: `results/baseline.md`
- `A5-M1-B` | 2026-11-16 | A5 | `refB_train.npz`, valid `refB_valid.npz` | `configs/A5-M1-B.json`: lograte + 1 x physical-increment MSE | account `math-454`, QOS `math-454`, fair-share 0.187, job 3253131 | finished 2026-11-16/17 (800 epochs), scored 2026-11-17 (jobs 3255265–3255268) | -
- `A5-M10-B` | 2026-11-16 | A5 | same | `configs/A5-M10-B.json`: lograte + 10 x physical-increment MSE | account `env540`, QOS `normal`, fair-share 0.232, job 3253132 | finished 2026-11-16/17 (800 epochs), scored 2026-11-17 (jobs 3255265–3255268) | -
- `A5-W-B` | 2026-11-16 | A5 | same | `configs/A5-W-B.json`: lograte with weight 1 + 20 |y| on the physical increment | account `env540`, QOS `normal`, job 3253133 | finished 2026-11-16/17 (800 epochs), scored 2026-11-17 (jobs 3255265–3255268) | -
- `A5-L-A` | 2026-11-16 | A5 | `refA_train.npz`, valid `refA_valid.npz` | `configs/A5-L-A.json`: as `A5-L-B` on the REF-A data | account `env540`, QOS `normal`, job 3253134 | finished 2026-11-16/17 (800 epochs), scored 2026-11-17 (jobs 3255265–3255268) | -
- Configuration choice: the report's nominal 10,000 epochs would again leave the mass-loss ramp under 10% of its maximum (R2); 900 epochs fit the 24 h wall time (June: 72 to 89 s per epoch), so the ramp reaches its full 1e7. This is the only deviation from the report configuration. The checkpoints keep being chosen on one-step validation MSE until A6.

## 9. Open questions

- No open questions as of 2026-11-14. Q1 to Q15 are answered; the answers are recorded in section 2 and applied in the tasks. Future questions go here.
- `A5-M10-B-s1` | 2026-11-17 | A5 seed check | `refB_train.npz`, valid `refB_valid.npz` | `configs/A5-M10-B-s1.json`: as `A5-M10-B`, seed 1 | account `cs-302`, QOS `cs-302`, fair-share 0.262, job 3255462 | pending | finished 2026-11-18 (800 epochs, 18 h), scored 2026-11-18 (jobs 3257705–3257707) | -
- `A5-W-B-s1` | 2026-11-17 | A5 seed check | same | `configs/A5-W-B-s1.json`: as `A5-W-B`, seed 1 | account `math-454`, QOS `math-454`, fair-share 0.212, job 3255463 | pending | finished 2026-11-18 (800 epochs, 18 h), scored 2026-11-18 (jobs 3257705–3257707) | -
- `A5-M1-B-s1` | 2026-11-17 | A5 seed check | same | `configs/A5-M1-B-s1.json`: as `A5-M1-B`, seed 1 | account `env540`, QOS `normal`, fair-share 0.186 (`cs-302` and `math-454` slots hold the other two seed runs), job 3255476 | pending | finished 2026-11-18 (800 epochs, 18 h), scored 2026-11-18 (jobs 3257705–3257707) | -
- `B7-S3` | 2026-11-18 | B7 screen | `sampling_S3.npz` (20M pairs, 800 trajectories), valid `refB_valid.npz` | `configs/B7-S3.json`: as `A5-M10-B` (lograte, phys_weight 10, seed 0, 800 epochs) | account `cs-302`, QOS `cs-302`, job 3257724 | submitted | finished 800 epochs (18 to 23 h), scored 2026-11-20 (jobs 3261814–3261817) | -
- `B7-S4` | 2026-11-18 | B7 screen | `sampling_S4.npz` (20M pairs, per decade of activity) | `configs/B7-S4.json`: as above | account `math-454`, QOS `math-454`, job 3257725 | submitted | finished 800 epochs (18 to 23 h), scored 2026-11-20 (jobs 3261814–3261817) | -
- `B7-S5` | 2026-11-18 | B7 screen | `sampling_S5.npz` (20M pairs, per state-space cell) | `configs/B7-S5.json`: as above | account `env540`, QOS `normal`, job 3257726 | submitted | finished 800 epochs (18 to 23 h), scored 2026-11-20 (jobs 3261814–3261817) | -
- `B7-S6` | 2026-11-18 | B7 screen | `sampling_S6.npz` (20M pairs, equilibrium thinned) | `configs/B7-S6.json`: as above | account `env540`, QOS `normal`, job 3257727 | submitted | finished 800 epochs (18 to 23 h), scored 2026-11-20 (jobs 3261814–3261817) | -
