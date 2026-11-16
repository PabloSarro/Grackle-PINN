# Baseline table

All numbers are free rollouts from the initial condition only (see `TASKS.md` section 0.2), produced by `python -m grackle_pinn.evaluate` and `jobs/evaluate_legacy.job` (job 3252188, 2026-11-14). Errors are |d log10| of the predicted against the Grackle value over the 25,000 steps: `T med` and `T p95` are the median and 95th percentile over time, `final` is at 2.5 Myr; every column is the median over the trajectories of the group. `cliff err` is the time error of the steepest d ln T (only trajectories with a cliff; `cliffs missed` = no cliff above |d ln T| = 0.05 in the prediction / trajectories with a cliff). `mass viol.` = max |ln(nH_pred / nH_0)|. `E up frac.` = fraction of steps where the specific energy increases (truth: 0).

## As reported (June 2026 report, read off figures, no numbers published)

| model | what the report shows |
|---|---|
| Model A | follows Family 3 (Figs. 3 top, 5); cooling cliff about 0.15 Myr early on another Family 3 case (Fig. 6); invents cooling on Families 0, 1, 2 (Fig. 3 bottom, T near 1e-28 K); Fig. 7 drift +0.07 dex in T |
| Model B | straight log-linear drift on every family (Fig. 4); about 2 dex off in T at 2.5 Myr on Family 3 |

## Legacy checkpoints, identity uncertain (not the report's Models A and B, see `TASKS.md` section 2)

`legacy_<name>` = last checkpoint `PINN_<name>.pth`; `legacy_BEST_<name>` = `PINN_BEST_<name>.pth` (best one-step validation MSE). Architectures read from the checkpoints: `1` 128 x 10; `noClamp_Long` 256 x 8; all others 512 x 12. `legacy_filt` and `legacy_BEST_filt` are the same weights (best = last).

### Report-figure set (8 trajectories: Train and Valid for Families 0 to 3), float64

| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_1 | report | float64 | all | 8 | 0.0938 | 1.01 | 0.175 | 0.153 | 0.533 | 8.22e+04 | 0/2 | 0.203 | 0.386 |
| legacy_1 | report | float64 | 0 | 2 | 2.86 | 176 | 225 | 1.06e+03 | 417 | - | 0/0 | 960 | 0.765 |
| legacy_1 | report | float64 | 1 | 2 | 0.0846 | 0.169 | 0.179 | 0.191 | 0.0802 | - | 0/0 | 0.0732 | 0.193 |
| legacy_1 | report | float64 | 2 | 2 | 0.0938 | 0.167 | 0.175 | 0.0204 | 0.908 | - | 0/0 | 0.0748 | 0.976 |
| legacy_1 | report | float64 | 3 | 2 | 0.0418 | 1.77 | 0.0331 | 0.153 | 0.399 | 8.22e+04 | 0/2 | 0.382 | 0 |
| legacy_BEST_1 | report | float64 | all | 8 | 0.152 | 1.01 | 0.207 | 1.13 | 1.54 | 8.51e+04 | 0/2 | 2.54 | 0.614 |
| legacy_BEST_1 | report | float64 | 0 | 2 | 0.729 | 1.41 | 0.91 | 1.05 | 2.1 | - | 0/0 | 4.93 | 0.85 |
| legacy_BEST_1 | report | float64 | 1 | 2 | 0.251 | 0.475 | 0.5 | 1.33 | 0.345 | - | 0/0 | 1.56 | 0 |
| legacy_BEST_1 | report | float64 | 2 | 2 | 0.0197 | 0.0523 | 0.0576 | 1.06 | 2.22 | - | 0/0 | 2.45 | 0.838 |
| legacy_BEST_1 | report | float64 | 3 | 2 | 0.106 | 1.83 | 0.0955 | 1.14 | 0.62 | 8.51e+04 | 0/2 | 2.61 | 0.219 |
| legacy_BEST_fam3 | report | float64 | all | 8 | 0.647 | 1.37 | 1.32 | 3.24 | 1.52 | 3.48e+05 | 0/2 | 4.91 | 0.701 |
| legacy_BEST_fam3 | report | float64 | 0 | 2 | 0.647 | 1.15 | 1.2 | 3.24 | 3.07 | - | 0/0 | 7.08 | 0.957 |
| legacy_BEST_fam3 | report | float64 | 1 | 2 | 3.54 | 3.89 | 3.93 | 4.2 | 2.35 | - | 0/0 | 5.03 | 0.974 |
| legacy_BEST_fam3 | report | float64 | 2 | 2 | 0.37 | 0.578 | 0.583 | 0.603 | 0.413 | - | 0/0 | 1.38 | 0.0127 |
| legacy_BEST_fam3 | report | float64 | 3 | 2 | 0.615 | 2.19 | 1.39 | 2.8 | 0.893 | 3.48e+05 | 0/2 | 4.75 | 0.454 |
| legacy_BEST_fam3_long | report | float64 | all | 8 | 1.72 | 2.05 | 1.79 | 5.93 | 2.07 | 3.68e+05 | 0/2 | 8.71 | 0.00238 |
| legacy_BEST_fam3_long | report | float64 | 0 | 2 | 2.27 | 2.47 | 1.57 | 8.22 | 3.9 | - | 0/0 | 8.98 | 0.416 |
| legacy_BEST_fam3_long | report | float64 | 1 | 2 | 1.28 | 1.73 | 1.79 | 3.54 | 0.501 | - | 0/0 | 7.66 | 6e-05 |
| legacy_BEST_fam3_long | report | float64 | 2 | 2 | 0.555 | 0.854 | 0.794 | 0.731 | 0.462 | - | 0/0 | 1.77 | 0.411 |
| legacy_BEST_fam3_long | report | float64 | 3 | 2 | 5.31 | 13 | 13.9 | 624 | 48.6 | 3.68e+05 | 0/2 | 106 | 0.00234 |
| legacy_BEST_filt | report | float64 | all | 8 | 0.506 | 0.862 | 0.744 | 0.797 | 1.11 | 1.28e+05 | 0/2 | 1.4 | 0.494 |
| legacy_BEST_filt | report | float64 | 0 | 2 | 0.757 | 1.14 | 1.18 | 7.73 | 0.699 | - | 0/0 | 1.61 | 0.991 |
| legacy_BEST_filt | report | float64 | 1 | 2 | 0.41 | 0.975 | 1.05 | 0.716 | 1.36 | - | 0/0 | 1.17 | 0 |
| legacy_BEST_filt | report | float64 | 2 | 2 | 0.282 | 0.381 | 0.389 | 0.797 | 0.958 | - | 0/0 | 1.84 | 0.993 |
| legacy_BEST_filt | report | float64 | 3 | 2 | 0.506 | 1.51 | 0.744 | 0.265 | 1.73 | 1.28e+05 | 0/2 | 0.783 | 0 |
| legacy_BEST_modelB | report | float64 | all | 8 | 6.39 | 8.45 | 8.63 | 12.8 | 50.2 | 2.43e+05 | 0/2 | 29.8 | 0.825 |
| legacy_BEST_modelB | report | float64 | 0 | 2 | 2.08 | 3.48 | 3.62 | 4.91 | 0.823 | - | 0/0 | 1.89 | 0 |
| legacy_BEST_modelB | report | float64 | 1 | 2 | 8.36 | 10.3 | 10.4 | 12.8 | 57.1 | - | 0/0 | 30.1 | 0.883 |
| legacy_BEST_modelB | report | float64 | 2 | 2 | 8.28 | 10.1 | 10.2 | 14.6 | 51.5 | - | 0/0 | 33.6 | 0.993 |
| legacy_BEST_modelB | report | float64 | 3 | 2 | 4.08 | 6.04 | 6.22 | 11 | 35.7 | 2.43e+05 | 0/2 | 25.4 | 0.818 |
| legacy_BEST_modelB_long | report | float64 | all | 8 | 0.162 | 0.356 | 0.375 | 0.151 | 0.281 | 1.15e+05 | 0/2 | 0.372 | 0.17 |
| legacy_BEST_modelB_long | report | float64 | 0 | 2 | 0.0337 | 0.0569 | 0.0482 | 0.219 | 0.335 | - | 0/0 | 0.772 | 0.747 |
| legacy_BEST_modelB_long | report | float64 | 1 | 2 | 0.41 | 0.734 | 0.769 | 0.219 | 0.229 | - | 0/0 | 0.135 | 0 |
| legacy_BEST_modelB_long | report | float64 | 2 | 2 | 0.0555 | 0.208 | 0.228 | 0.0594 | 0.0392 | - | 0/0 | 0.332 | 0.17 |
| legacy_BEST_modelB_long | report | float64 | 3 | 2 | 0.275 | 1.45 | 0.574 | 2.32 | 8.77 | 1.15e+05 | 0/2 | 7.02 | 0.405 |
| legacy_BEST_noClamp_Long | report | float64 | all | 8 | 0.0922 | 0.393 | 0.248 | 0.0772 | 0.812 | 1.44e+05 | 0/2 | 0.281 | 0.0657 |
| legacy_BEST_noClamp_Long | report | float64 | 0 | 2 | 0.02 | 0.0384 | 0.0415 | 0.033 | 0.0112 | - | 0/0 | 0.133 | 0.957 |
| legacy_BEST_noClamp_Long | report | float64 | 1 | 2 | 20.2 | 20.2 | 20.2 | 3.94 | 4.05 | - | 0/0 | 8.78 | 0 |
| legacy_BEST_noClamp_Long | report | float64 | 2 | 2 | 0.174 | 0.393 | 0.415 | 0.383 | 1.97 | - | 0/0 | 0.894 | 0.0657 |
| legacy_BEST_noClamp_Long | report | float64 | 3 | 2 | 0.456 | 2.43 | 0.427 | 1.06 | 0.628 | 1.44e+05 | 0/2 | 7.55 | 0.479 |
| legacy_fam3 | report | float64 | all | 8 | 0.773 | 1.74 | 0.993 | 1.2 | 1.65 | 3.68e+05 | 0/2 | 3.51 | 0.471 |
| legacy_fam3 | report | float64 | 0 | 2 | 0.557 | 0.848 | 0.876 | 1.52 | 2.9 | - | 0/0 | 6.68 | 0.971 |
| legacy_fam3 | report | float64 | 1 | 2 | 3.68 | 3.83 | 3.85 | 6.08 | 2.28 | - | 0/0 | 6.24 | 1 |
| legacy_fam3 | report | float64 | 2 | 2 | 0.32 | 0.494 | 0.512 | 1.01 | 0.118 | - | 0/0 | 2.33 | 0 |
| legacy_fam3 | report | float64 | 3 | 2 | 1.38 | 3.3 | 1.42 | 0.984 | 1.05 | 3.68e+05 | 0/2 | 2.26 | 0 |
| legacy_fam3_long | report | float64 | all | 8 | 1.84 | 2.16 | 2.19 | 5.19 | 1.76 | 1.38e+05 | 0/2 | 8.26 | 0.00218 |
| legacy_fam3_long | report | float64 | 0 | 2 | 2.73 | 3.12 | 3.15 | 8.95 | 2.99 | - | 0/0 | 8.26 | 0.499 |
| legacy_fam3_long | report | float64 | 1 | 2 | 1.29 | 1.66 | 1.7 | 3.64 | 0.334 | - | 0/0 | 7.9 | 0.00218 |
| legacy_fam3_long | report | float64 | 2 | 2 | 0.507 | 0.723 | 0.653 | 0.865 | 0.237 | - | 0/0 | 1.99 | 0.392 |
| legacy_fam3_long | report | float64 | 3 | 2 | 7.02 | 15.8 | 16.8 | 650 | 35.7 | 1.38e+05 | 0/2 | 76.2 | 0 |
| legacy_filt | report | float64 | all | 8 | 0.506 | 0.862 | 0.744 | 0.797 | 1.11 | 1.28e+05 | 0/2 | 1.4 | 0.494 |
| legacy_filt | report | float64 | 0 | 2 | 0.757 | 1.14 | 1.18 | 7.73 | 0.699 | - | 0/0 | 1.61 | 0.991 |
| legacy_filt | report | float64 | 1 | 2 | 0.41 | 0.975 | 1.05 | 0.716 | 1.36 | - | 0/0 | 1.17 | 0 |
| legacy_filt | report | float64 | 2 | 2 | 0.282 | 0.381 | 0.389 | 0.797 | 0.958 | - | 0/0 | 1.84 | 0.993 |
| legacy_filt | report | float64 | 3 | 2 | 0.506 | 1.51 | 0.744 | 0.265 | 1.73 | 1.28e+05 | 0/2 | 0.783 | 0 |
| legacy_modelB | report | float64 | all | 8 | 1.1 | 12.9 | 15.4 | 37 | 113 | 3.56e+05 | 0/2 | 77 | 0.515 |
| legacy_modelB | report | float64 | 0 | 2 | 2.25 | 4.16 | 4.23 | 10.9 | 3.2 | - | 0/0 | 8.04 | 0.0117 |
| legacy_modelB | report | float64 | 1 | 2 | 0.547 | 33.3 | 38.5 | 91.5 | 351 | - | 0/0 | 211 | 0.631 |
| legacy_modelB | report | float64 | 2 | 2 | 1.61 | 24.4 | 29.5 | 70.1 | 257 | - | 0/0 | 161 | 0.762 |
| legacy_modelB | report | float64 | 3 | 2 | 0.447 | 2.32 | 0.415 | 4.03 | 2.17 | 3.56e+05 | 0/2 | 10.6 | 0.429 |
| legacy_modelB_long | report | float64 | all | 8 | 0.08 | 0.288 | 0.162 | 0.023 | 0.0412 | 9.6e+04 | 0/2 | 0.0525 | 0.865 |
| legacy_modelB_long | report | float64 | 0 | 2 | 0.033 | 0.056 | 0.037 | 0.0541 | 0.0178 | - | 0/0 | 0.041 | 0.477 |
| legacy_modelB_long | report | float64 | 1 | 2 | 0.08 | 0.154 | 0.162 | 0.0073 | 0.0356 | - | 0/0 | 0.0366 | 0.5 |
| legacy_modelB_long | report | float64 | 2 | 2 | 0.299 | 0.404 | 0.412 | 0.0221 | 0.372 | - | 0/0 | 0.051 | 0.993 |
| legacy_modelB_long | report | float64 | 3 | 2 | 0.146 | 1.34 | 0.127 | 0.685 | 2.49 | 9.6e+04 | 0/2 | 3.22 | 0.388 |
| legacy_noClamp_Long | report | float64 | all | 8 | 0.157 | 0.47 | 0.291 | 0.149 | 0.369 | 1.2e+05 | 0/2 | 0.489 | 0.0544 |
| legacy_noClamp_Long | report | float64 | 0 | 2 | 0.0936 | 0.275 | 0.277 | 0.515 | 0.176 | - | 0/0 | 0.426 | 0.435 |
| legacy_noClamp_Long | report | float64 | 1 | 2 | 20.7 | 20.8 | 20.8 | 4 | 4.2 | - | 0/0 | 8.97 | 0 |
| legacy_noClamp_Long | report | float64 | 2 | 2 | 0.192 | 0.419 | 0.442 | 0.385 | 1.98 | - | 0/0 | 0.896 | 0.054 |
| legacy_noClamp_Long | report | float64 | 3 | 2 | 0.0747 | 1.99 | 0.0736 | 0.117 | 0.0739 | 1.2e+05 | 0/2 | 0.486 | 0.0756 |

### Family grid subset (50 trajectories, Families 1 and 2 only), float64

Only 507 of the 5,625 grid files survive locally, and none of Families 0 and 3 are in the subset used here (25 per family sampled with seed 0). Regenerated in A3.

| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_1 | grid | float64 | all | 50 | 0.102 | 0.239 | 0.181 | 0.519 | 0.543 | 200 | 2/30 | 0.364 | 0.0158 |
| legacy_1 | grid | float64 | 1 | 25 | 0.0975 | 0.191 | 0.185 | 0.484 | 0.596 | 0 | 0/14 | 0.156 | 0.0132 |
| legacy_1 | grid | float64 | 2 | 25 | 0.123 | 0.605 | 0.178 | 0.545 | 0.531 | 300 | 2/16 | 0.412 | 0.0492 |
| legacy_BEST_1 | grid | float64 | all | 50 | 0.164 | 0.472 | 0.301 | 1.07 | 0.538 | 50 | 4/30 | 2.37 | 0.307 |
| legacy_BEST_1 | grid | float64 | 1 | 25 | 0.141 | 0.364 | 0.33 | 1.1 | 0.614 | 0 | 1/14 | 2.29 | 0.095 |
| legacy_BEST_1 | grid | float64 | 2 | 25 | 0.196 | 0.644 | 0.259 | 1.05 | 0.477 | 100 | 3/16 | 2.4 | 0.363 |
| legacy_BEST_fam3 | grid | float64 | all | 50 | 1.04 | 1.55 | 1.46 | 2.71 | 1.03 | 600 | 10/30 | 2.82 | 0.561 |
| legacy_BEST_fam3 | grid | float64 | 1 | 25 | 1.11 | 1.51 | 1.47 | 1.86 | 1.04 | 0 | 3/14 | 2.16 | 0.504 |
| legacy_BEST_fam3 | grid | float64 | 2 | 25 | 0.857 | 1.56 | 1.42 | 2.82 | 0.826 | 2.25e+04 | 7/16 | 3.15 | 0.563 |
| legacy_BEST_fam3_long | grid | float64 | all | 50 | 1.39 | 1.76 | 1.21 | 5.63 | 1.99 | 0 | 5/30 | 7.56 | 0.362 |
| legacy_BEST_fam3_long | grid | float64 | 1 | 25 | 1.12 | 1.47 | 1.23 | 5.03 | 1.87 | 0 | 1/14 | 9.07 | 0.0502 |
| legacy_BEST_fam3_long | grid | float64 | 2 | 25 | 1.86 | 2.09 | 0.956 | 6.49 | 2.04 | 0 | 4/16 | 5.62 | 0.397 |
| legacy_BEST_filt | grid | float64 | all | 50 | 0.497 | 0.746 | 0.736 | 0.685 | 1.18 | 0 | 1/30 | 0.757 | 0.00514 |
| legacy_BEST_filt | grid | float64 | 1 | 25 | 0.476 | 0.705 | 0.715 | 0.73 | 0.902 | 0 | 0/14 | 0.983 | 0.0164 |
| legacy_BEST_filt | grid | float64 | 2 | 25 | 0.519 | 0.811 | 0.745 | 0.542 | 1.46 | 0 | 1/16 | 0.577 | 0.00232 |
| legacy_BEST_modelB | grid | float64 | all | 50 | 3.8 | 5.66 | 5.58 | 7.79 | 30.5 | 0 | 1/30 | 20.4 | 0.828 |
| legacy_BEST_modelB | grid | float64 | 1 | 25 | 4.59 | 6.83 | 6.99 | 7.3 | 36.6 | 0 | 0/14 | 20.8 | 0.851 |
| legacy_BEST_modelB | grid | float64 | 2 | 25 | 2.97 | 4.52 | 4.68 | 7.9 | 24.7 | 0 | 1/16 | 20.2 | 0.774 |
| legacy_BEST_modelB_long | grid | float64 | all | 50 | 0.235 | 0.481 | 0.487 | 0.311 | 0.334 | 0 | 2/30 | 0.68 | 0.0018 |
| legacy_BEST_modelB_long | grid | float64 | 1 | 25 | 0.235 | 0.467 | 0.485 | 0.226 | 0.307 | 0 | 0/14 | 0.408 | 0.00296 |
| legacy_BEST_modelB_long | grid | float64 | 2 | 25 | 0.235 | 0.5 | 0.488 | 0.36 | 0.35 | 0 | 2/16 | 0.747 | 0.00088 |
| legacy_BEST_noClamp_Long | grid | float64 | all | 50 | 0.104 | 0.209 | 0.147 | 0.516 | 0.196 | 0 | 5/30 | 0.629 | 0.0272 |
| legacy_BEST_noClamp_Long | grid | float64 | 1 | 25 | 0.128 | 0.274 | 0.218 | 0.823 | 0.213 | 0 | 1/14 | 0.836 | 0.00144 |
| legacy_BEST_noClamp_Long | grid | float64 | 2 | 25 | 0.0616 | 0.164 | 0.0893 | 0.294 | 0.183 | 0 | 4/16 | 0.59 | 0.0863 |
| legacy_fam3 | grid | float64 | all | 50 | 0.885 | 1.25 | 1.14 | 1.1 | 1.17 | 0 | 10/30 | 2.36 | 0.797 |
| legacy_fam3 | grid | float64 | 1 | 25 | 0.848 | 1.27 | 1.08 | 0.924 | 0.852 | 0 | 3/14 | 2.04 | 0.705 |
| legacy_fam3 | grid | float64 | 2 | 25 | 0.897 | 1.24 | 1.2 | 1.21 | 1.37 | 0 | 7/16 | 2.79 | 0.956 |
| legacy_fam3_long | grid | float64 | all | 50 | 1.71 | 1.99 | 2.01 | 4.75 | 1.28 | 0 | 5/30 | 6.44 | 0.0081 |
| legacy_fam3_long | grid | float64 | 1 | 25 | 1.25 | 1.61 | 1.64 | 4.86 | 1.05 | 0 | 1/14 | 3.79 | 0.00688 |
| legacy_fam3_long | grid | float64 | 2 | 25 | 2.29 | 2.55 | 2.58 | 4.55 | 1.31 | 3e+03 | 4/16 | 6.56 | 0.015 |
| legacy_filt | grid | float64 | all | 50 | 0.497 | 0.746 | 0.736 | 0.685 | 1.18 | 0 | 1/30 | 0.757 | 0.00514 |
| legacy_filt | grid | float64 | 1 | 25 | 0.476 | 0.705 | 0.715 | 0.73 | 0.902 | 0 | 0/14 | 0.983 | 0.0164 |
| legacy_filt | grid | float64 | 2 | 25 | 0.519 | 0.811 | 0.745 | 0.542 | 1.46 | 0 | 1/16 | 0.577 | 0.00232 |
| legacy_modelB | grid | float64 | all | 50 | 2.03 | 3.64 | 3.18 | 6.66 | 3.19 | 0 | 0/30 | 9.77 | 0.445 |
| legacy_modelB | grid | float64 | 1 | 25 | 2.03 | 4.02 | 4.1 | 6.7 | 4.19 | 0 | 0/14 | 10.9 | 0.508 |
| legacy_modelB | grid | float64 | 2 | 25 | 1.93 | 3.34 | 2.89 | 6.63 | 2.29 | 0 | 0/16 | 9.15 | 0.387 |
| legacy_modelB_long | grid | float64 | all | 50 | 0.0619 | 0.12 | 0.0945 | 0.0687 | 0.058 | 0 | 1/30 | 0.0907 | 0.189 |
| legacy_modelB_long | grid | float64 | 1 | 25 | 0.076 | 0.136 | 0.129 | 0.0708 | 0.0576 | 0 | 0/14 | 0.116 | 0.195 |
| legacy_modelB_long | grid | float64 | 2 | 25 | 0.044 | 0.094 | 0.0561 | 0.0473 | 0.0585 | 0 | 1/16 | 0.0756 | 0.189 |
| legacy_noClamp_Long | grid | float64 | all | 50 | 0.118 | 0.277 | 0.167 | 0.61 | 0.255 | 0 | 5/30 | 0.692 | 0.00194 |
| legacy_noClamp_Long | grid | float64 | 1 | 25 | 0.147 | 0.305 | 0.259 | 0.764 | 0.311 | 0 | 1/14 | 1.19 | 0.00152 |
| legacy_noClamp_Long | grid | float64 | 2 | 25 | 0.0858 | 0.173 | 0.123 | 0.493 | 0.164 | 0 | 4/16 | 0.471 | 0.0323 |

### Report-figure set, float32 rollout (precision check)

| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_1 | report | float32 | all | 8 | 0.095 | 1.01 | 0.178 | 0.112 | 0.52 | 8.22e+04 | 0/2 | 0.203 | 0.422 |
| legacy_1 | report | float32 | 0 | 2 | 1.73 | 134 | 182 | 857 | 337 | - | 0/0 | 777 | 0.734 |
| legacy_1 | report | float32 | 1 | 2 | 0.0845 | 0.169 | 0.179 | 0.19 | 0.078 | - | 0/0 | 0.0739 | 0.211 |
| legacy_1 | report | float32 | 2 | 2 | 0.095 | 0.169 | 0.178 | 0.0204 | 0.907 | - | 0/0 | 0.0746 | 0.976 |
| legacy_1 | report | float32 | 3 | 2 | 0.0421 | 1.77 | 0.0421 | 0.112 | 0.361 | 8.22e+04 | 0/2 | 0.324 | 0 |
| legacy_BEST_1 | report | float32 | all | 8 | 0.158 | 0.854 | 0.212 | 1.15 | 1.32 | 8.5e+04 | 0/2 | 2.57 | 0.622 |
| legacy_BEST_1 | report | float32 | 0 | 2 | 1.01 | 1.26 | 1.28 | 1.5 | 1.74 | - | 0/0 | 4.11 | 0.856 |
| legacy_BEST_1 | report | float32 | 1 | 2 | 0.25 | 0.472 | 0.498 | 1.33 | 0.348 | - | 0/0 | 1.56 | 0 |
| legacy_BEST_1 | report | float32 | 2 | 2 | 0.0203 | 0.0544 | 0.0598 | 1.06 | 2.22 | - | 0/0 | 2.45 | 0.843 |
| legacy_BEST_1 | report | float32 | 3 | 2 | 0.115 | 1.83 | 0.106 | 1.15 | 0.651 | 8.5e+04 | 0/2 | 2.63 | 0.222 |
| legacy_BEST_fam3 | report | float32 | all | 8 | 0.644 | 1.42 | 1.32 | 3.23 | 1.5 | 3.48e+05 | 0/2 | 4.91 | 0.708 |
| legacy_BEST_fam3 | report | float32 | 0 | 2 | 0.644 | 1.14 | 1.19 | 3.23 | 3.07 | - | 0/0 | 7.08 | 0.958 |
| legacy_BEST_fam3 | report | float32 | 1 | 2 | 3.54 | 3.9 | 3.94 | 4.2 | 2.35 | - | 0/0 | 5.03 | 0.974 |
| legacy_BEST_fam3 | report | float32 | 2 | 2 | 0.369 | 0.576 | 0.581 | 0.604 | 0.413 | - | 0/0 | 1.38 | 0.0135 |
| legacy_BEST_fam3 | report | float32 | 3 | 2 | 0.614 | 2.24 | 1.39 | 2.88 | 0.882 | 3.48e+05 | 0/2 | 4.78 | 0.496 |
| legacy_BEST_fam3_long | report | float32 | all | 8 | 1.84 | 2.17 | 1.79 | 5.95 | 2.08 | 3.68e+05 | 0/2 | 8.73 | 0.0024 |
| legacy_BEST_fam3_long | report | float32 | 0 | 2 | 2.37 | 2.56 | 1.75 | 8.02 | 3.87 | - | 0/0 | 8.91 | 0.413 |
| legacy_BEST_fam3_long | report | float32 | 1 | 2 | 1.28 | 1.73 | 1.79 | 3.54 | 0.501 | - | 0/0 | 7.66 | 6e-05 |
| legacy_BEST_fam3_long | report | float32 | 2 | 2 | 0.555 | 0.854 | 0.793 | 0.732 | 0.461 | - | 0/0 | 1.77 | 0.412 |
| legacy_BEST_fam3_long | report | float32 | 3 | 2 | 5.31 | 13 | 13.9 | 624 | 48.6 | 3.68e+05 | 0/2 | 106 | 0.00236 |
| legacy_BEST_filt | report | float32 | all | 8 | 0.505 | 0.858 | 0.743 | 0.796 | 1.11 | 1.29e+05 | 0/2 | 1.4 | 0.494 |
| legacy_BEST_filt | report | float32 | 0 | 2 | 0.702 | 1.1 | 1.14 | 7.82 | 0.727 | - | 0/0 | 1.67 | 0.991 |
| legacy_BEST_filt | report | float32 | 1 | 2 | 0.409 | 0.974 | 1.05 | 0.718 | 1.35 | - | 0/0 | 1.17 | 0 |
| legacy_BEST_filt | report | float32 | 2 | 2 | 0.283 | 0.382 | 0.39 | 0.796 | 0.957 | - | 0/0 | 1.83 | 0.993 |
| legacy_BEST_filt | report | float32 | 3 | 2 | 0.505 | 1.51 | 0.743 | 0.265 | 1.73 | 1.29e+05 | 0/2 | 0.785 | 0 |
| legacy_BEST_modelB | report | float32 | all | 8 | 6.64 | 8.55 | 8.72 | 12.8 | 51.4 | 2.71e+05 | 0/2 | 30 | 0.825 |
| legacy_BEST_modelB | report | float32 | 0 | 2 | 2.08 | 3.49 | 3.62 | 4.92 | 0.824 | - | 0/0 | 1.89 | 0 |
| legacy_BEST_modelB | report | float32 | 1 | 2 | 8.36 | 10.2 | 10.4 | 12.8 | 57.1 | - | 0/0 | 30 | 0.882 |
| legacy_BEST_modelB | report | float32 | 2 | 2 | 8.46 | 10.3 | 10.4 | 14.9 | 52.7 | - | 0/0 | 34.4 | 0.993 |
| legacy_BEST_modelB | report | float32 | 3 | 2 | 4.27 | 6.06 | 6.23 | 11.1 | 35.8 | 2.71e+05 | 0/2 | 25.5 | 0.817 |
| legacy_BEST_modelB_long | report | float32 | all | 8 | 0.163 | 0.358 | 0.273 | 0.151 | 0.28 | 1.13e+05 | 0/2 | 0.372 | 0.168 |
| legacy_BEST_modelB_long | report | float32 | 0 | 2 | 0.0359 | 0.0588 | 0.0521 | 0.214 | 0.335 | - | 0/0 | 0.77 | 0.748 |
| legacy_BEST_modelB_long | report | float32 | 1 | 2 | 0.411 | 0.736 | 0.77 | 0.219 | 0.229 | - | 0/0 | 0.135 | 0 |
| legacy_BEST_modelB_long | report | float32 | 2 | 2 | 0.0556 | 0.21 | 0.23 | 0.0595 | 0.0391 | - | 0/0 | 0.332 | 0.168 |
| legacy_BEST_modelB_long | report | float32 | 3 | 2 | 0.28 | 1.31 | 0.374 | 0.724 | 2.47 | 1.13e+05 | 0/2 | 1.95 | 0.141 |
| legacy_BEST_noClamp_Long | report | float32 | all | 8 | 0.0894 | 0.394 | 0.111 | 0.0785 | 0.237 | 1.44e+05 | 0/2 | 0.282 | 0.00364 |
| legacy_BEST_noClamp_Long | report | float32 | 0 | 2 | 0.0212 | 0.044 | 0.0473 | 0.0436 | 0.0125 | - | 0/0 | 0.133 | 0.957 |
| legacy_BEST_noClamp_Long | report | float32 | 1 | 2 | 20.2 | 20.2 | 20.2 | 3.94 | 4.05 | - | 0/0 | 8.78 | 0 |
| legacy_BEST_noClamp_Long | report | float32 | 2 | 2 | 0.175 | 0.394 | 0.416 | 0.383 | 1.97 | - | 0/0 | 0.894 | 0.0652 |
| legacy_BEST_noClamp_Long | report | float32 | 3 | 2 | 0.0715 | 1.96 | 0.0709 | 0.129 | 0.054 | 1.44e+05 | 0/2 | 0.434 | 0.00364 |
| legacy_fam3 | report | float32 | all | 8 | 0.769 | 1.73 | 0.986 | 1.2 | 1.65 | 3.68e+05 | 0/2 | 3.51 | 0.471 |
| legacy_fam3 | report | float32 | 0 | 2 | 0.556 | 0.843 | 0.87 | 1.52 | 2.9 | - | 0/0 | 6.68 | 0.971 |
| legacy_fam3 | report | float32 | 1 | 2 | 3.67 | 3.83 | 3.84 | 6.08 | 2.28 | - | 0/0 | 6.24 | 0.999 |
| legacy_fam3 | report | float32 | 2 | 2 | 0.319 | 0.492 | 0.51 | 1.01 | 0.118 | - | 0/0 | 2.33 | 0 |
| legacy_fam3 | report | float32 | 3 | 2 | 1.38 | 3.3 | 1.42 | 0.984 | 1.05 | 3.68e+05 | 0/2 | 2.26 | 0 |
| legacy_fam3_long | report | float32 | all | 8 | 1.69 | 2.26 | 2.04 | 3.77 | 0.794 | 1.37e+05 | 0/2 | 6.96 | 0.00422 |
| legacy_fam3_long | report | float32 | 0 | 2 | 2.76 | 3.14 | 3.17 | 8.96 | 2.97 | - | 0/0 | 8.26 | 0.499 |
| legacy_fam3_long | report | float32 | 1 | 2 | 1.77 | 2.16 | 2.2 | 3.77 | 0.416 | - | 0/0 | 8.19 | 0.0022 |
| legacy_fam3_long | report | float32 | 2 | 2 | 0.506 | 0.723 | 0.652 | 0.865 | 0.236 | - | 0/0 | 1.99 | 0.392 |
| legacy_fam3_long | report | float32 | 3 | 2 | 3.78 | 9.05 | 8.79 | 344 | 19.6 | 1.37e+05 | 0/2 | 41.8 | 0.389 |
| legacy_filt | report | float32 | all | 8 | 0.505 | 0.858 | 0.743 | 0.796 | 1.11 | 1.29e+05 | 0/2 | 1.4 | 0.494 |
| legacy_filt | report | float32 | 0 | 2 | 0.702 | 1.1 | 1.14 | 7.82 | 0.727 | - | 0/0 | 1.67 | 0.991 |
| legacy_filt | report | float32 | 1 | 2 | 0.409 | 0.974 | 1.05 | 0.718 | 1.35 | - | 0/0 | 1.17 | 0 |
| legacy_filt | report | float32 | 2 | 2 | 0.283 | 0.382 | 0.39 | 0.796 | 0.957 | - | 0/0 | 1.83 | 0.993 |
| legacy_filt | report | float32 | 3 | 2 | 0.505 | 1.51 | 0.743 | 0.265 | 1.73 | 1.29e+05 | 0/2 | 0.785 | 0 |
| legacy_modelB | report | float32 | all | 8 | 1.86 | 20.8 | 23.4 | 55.6 | 191 | 3.56e+05 | 0/2 | 120 | 0.517 |
| legacy_modelB | report | float32 | 0 | 2 | 2.25 | 4.16 | 4.23 | 10.9 | 3.2 | - | 0/0 | 8.04 | 0.0117 |
| legacy_modelB | report | float32 | 1 | 2 | 6.17 | 48 | 53.4 | 128 | 488 | - | 0/0 | 294 | 0.716 |
| legacy_modelB | report | float32 | 2 | 2 | 2.03 | 40.8 | 46.2 | 112 | 417 | - | 0/0 | 258 | 0.919 |
| legacy_modelB | report | float32 | 3 | 2 | 0.288 | 2.32 | 0.22 | 3.95 | 1.77 | 3.56e+05 | 0/2 | 9.92 | 0.398 |
| legacy_modelB_long | report | float32 | all | 8 | 0.0534 | 0.29 | 0.0708 | 0.0234 | 0.0417 | 1.02e+05 | 0/2 | 0.0526 | 0.477 |
| legacy_modelB_long | report | float32 | 0 | 2 | 0.0316 | 0.053 | 0.0322 | 0.0482 | 0.0172 | - | 0/0 | 0.0397 | 0.477 |
| legacy_modelB_long | report | float32 | 1 | 2 | 0.0802 | 0.155 | 0.163 | 0.00783 | 0.0355 | - | 0/0 | 0.0359 | 0.5 |
| legacy_modelB_long | report | float32 | 2 | 2 | 0.298 | 0.402 | 0.41 | 0.0226 | 0.37 | - | 0/0 | 0.0522 | 0.993 |
| legacy_modelB_long | report | float32 | 3 | 2 | 0.0148 | 1.17 | 0.022 | 0.0531 | 0.0755 | 1.02e+05 | 0/2 | 0.132 | 0 |
| legacy_noClamp_Long | report | float32 | all | 8 | 0.158 | 0.472 | 0.293 | 0.15 | 0.369 | 1.2e+05 | 0/2 | 0.489 | 0.0262 |
| legacy_noClamp_Long | report | float32 | 0 | 2 | 0.0921 | 0.276 | 0.271 | 0.528 | 0.178 | - | 0/0 | 0.425 | 0.493 |
| legacy_noClamp_Long | report | float32 | 1 | 2 | 20.8 | 20.8 | 20.8 | 4 | 4.21 | - | 0/0 | 8.97 | 0 |
| legacy_noClamp_Long | report | float32 | 2 | 2 | 0.193 | 0.421 | 0.444 | 0.385 | 1.98 | - | 0/0 | 0.896 | 0.0536 |
| legacy_noClamp_Long | report | float32 | 3 | 2 | 0.0755 | 1.99 | 0.0747 | 0.105 | 0.0704 | 1.2e+05 | 0/2 | 0.461 | 0.0917 |

## Reference and new runs on the regenerated data (A2, A4, A5; scored 2026-11-16, jobs 3253117 to 3253120)

`BEST` = checkpoint chosen by the run's own rule (REF-A, REF-B: lowest one-step validation MSE, June loop; A4-C-B, A5-L-B: lowest free-rollout window score). `last` = final epoch. REF-A stopped at epoch 752 of 900 (23.5 h cap); REF-B finished 900 epochs.

### Report-figure set (8 trajectories), float64

| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A4-C-B_BEST | report | float64 | all | 8 | 0.153 | 0.34 | 0.351 | 0.0341 | 0.481 | 1.44e+04 | 0/2 | 8.88e-16 | 0 |
| A4-C-B_BEST | report | float64 | 0 | 2 | 0.107 | 0.189 | 0.198 | 0.174 | 1.25e-09 | - | 0/0 | 2.22e-15 | 0 |
| A4-C-B_BEST | report | float64 | 1 | 2 | 0.156 | 0.222 | 0.222 | 0.239 | 0.266 | - | 0/0 | 2.02e-14 | 0 |
| A4-C-B_BEST | report | float64 | 2 | 2 | 1.08 | 20.2 | 80.4 | 0.000473 | 99.1 | - | 0/0 | 2.78e-16 | 0 |
| A4-C-B_BEST | report | float64 | 3 | 2 | 0.141 | 0.34 | 0.351 | 0.00761 | 0.916 | 1.44e+04 | 0/2 | 8.88e-16 | 0 |
| A4-C-B_last | report | float64 | all | 8 | 0.108 | 0.201 | 0.161 | 0.0656 | 1.16 | 2e+03 | 0/2 | 7.77e-16 | 0 |
| A4-C-B_last | report | float64 | 0 | 2 | 0.0588 | 0.0988 | 0.103 | 0.17 | 1.63e-09 | - | 0/0 | 1.78e-15 | 0 |
| A4-C-B_last | report | float64 | 1 | 2 | 0.173 | 0.188 | 0.161 | 0.212 | 3.87 | - | 0/0 | 5.33e-15 | 0 |
| A4-C-B_last | report | float64 | 2 | 2 | 0.983 | 54.7 | 106 | 0.000431 | 102 | - | 0/0 | 5.55e-17 | 0 |
| A4-C-B_last | report | float64 | 3 | 2 | 0.0736 | 0.201 | 0.167 | 0.00579 | 1.02 | 2e+03 | 0/2 | 7.77e-16 | 0 |
| A5-L-B_BEST | report | float64 | all | 8 | 0.00052 | 0.00168 | 0.000603 | 0.00165 | 0.00457 | 1.39e+04 | 0/2 | 5.55e-16 | 0 |
| A5-L-B_BEST | report | float64 | 0 | 2 | 2.5e-06 | 2.73e-06 | 2.35e-06 | 0.00227 | 1.38e-11 | - | 0/0 | 1.78e-15 | 0 |
| A5-L-B_BEST | report | float64 | 1 | 2 | 0.000347 | 0.000577 | 0.000603 | 0.00289 | 0.00457 | - | 0/0 | 1.89e-14 | 0 |
| A5-L-B_BEST | report | float64 | 2 | 2 | 0.00979 | 0.0098 | 0.00978 | 8.12e-07 | 0.00582 | - | 0/0 | 5.55e-17 | 0 |
| A5-L-B_BEST | report | float64 | 3 | 2 | 0.948 | 1.01 | 0.942 | 3.44 | 1.06 | 1.39e+04 | 0/2 | 5.55e-16 | 0 |
| A5-L-B_last | report | float64 | all | 8 | 0.00204 | 0.00208 | 0.00103 | 0.000312 | 0.00134 | 2.6e+05 | 0/2 | 5.55e-16 | 0 |
| A5-L-B_last | report | float64 | 0 | 2 | 5.25e-06 | 5.43e-06 | 5.18e-06 | 0.000527 | 1.47e-12 | - | 0/0 | 8.88e-16 | 0 |
| A5-L-B_last | report | float64 | 1 | 2 | 0.000167 | 0.000264 | 0.000213 | 0.000624 | 0.0011 | - | 0/0 | 1.4e-14 | 0 |
| A5-L-B_last | report | float64 | 2 | 2 | 0.00902 | 0.00903 | 0.00902 | 1.15e-06 | 0.00458 | - | 0/0 | 5.55e-17 | 0 |
| A5-L-B_last | report | float64 | 3 | 2 | 0.976 | 2.39 | 0.962 | 3.57 | 1.13 | 2.6e+05 | 0/2 | 5.55e-16 | 0 |
| REF-A_BEST | report | float64 | all | 8 | 0.602 | 1.18 | 0.587 | 3.7 | 0.641 | 3.68e+05 | 0/2 | 0.804 | 0.256 |
| REF-A_BEST | report | float64 | 0 | 2 | 1.38 | 1.91 | 1.85 | 8.57 | 0.711 | - | 0/0 | 1.15 | 0.27 |
| REF-A_BEST | report | float64 | 1 | 2 | 0.0602 | 0.112 | 0.119 | 0.367 | 0.245 | - | 0/0 | 0.719 | 0.0629 |
| REF-A_BEST | report | float64 | 2 | 2 | 2.83 | 5.67 | 6.02 | 2.52 | 0.535 | - | 0/0 | 5.46 | 0.194 |
| REF-A_BEST | report | float64 | 3 | 2 | 1.47 | 2.16 | 1.67 | 7.3 | 2.43 | 3.68e+05 | 0/2 | 0.486 | 0.647 |
| REF-A_last | report | float64 | all | 8 | 0.754 | 1.21 | 0.816 | 0.723 | 0.416 | 3.68e+05 | 0/2 | 1.82 | 0.0788 |
| REF-A_last | report | float64 | 0 | 2 | 1.89 | 2.77 | 2.44 | 9.84 | 0.934 | - | 0/0 | 3.67 | 0.0874 |
| REF-A_last | report | float64 | 1 | 2 | 0.754 | 0.85 | 0.816 | 0.505 | 1.2 | - | 0/0 | 1.35 | 0.103 |
| REF-A_last | report | float64 | 2 | 2 | 2.35 | 4.71 | 4.98 | 2.43 | 0.192 | - | 0/0 | 4.66 | 0 |
| REF-A_last | report | float64 | 3 | 2 | 0.207 | 1.01 | 0.0305 | 0.53 | 0.264 | 3.68e+05 | 0/2 | 1.93 | 0.109 |
| REF-B_BEST | report | float64 | all | 8 | 0.652 | 2.25 | 2.02 | 2.71 | 1.84 | 4.29e+04 | 0/2 | 5.74 | 0 |
| REF-B_BEST | report | float64 | 0 | 2 | 1.05 | 1.84 | 1.94 | 1.32 | 0.0664 | - | 0/0 | 0.173 | 0 |
| REF-B_BEST | report | float64 | 1 | 2 | 0.613 | 1.11 | 1.34 | 2.71 | 1.42 | - | 0/0 | 5.74 | 0 |
| REF-B_BEST | report | float64 | 2 | 2 | 0.652 | 3.04 | 4.16 | 3.63 | 3.07 | - | 0/0 | 8.35 | 0 |
| REF-B_BEST | report | float64 | 3 | 2 | 1.09 | 4.38 | 4.66 | 4.4 | 2.78 | 4.29e+04 | 0/2 | 10.1 | 0.00012 |
| REF-B_last | report | float64 | all | 8 | 0.062 | 0.118 | 0.0928 | 0.0381 | 0.0955 | 1.85e+05 | 0/2 | 0.0329 | 0.861 |
| REF-B_last | report | float64 | 0 | 2 | 0.038 | 0.0637 | 0.0536 | 0.0504 | 0.00943 | - | 0/0 | 0.0217 | 0.477 |
| REF-B_last | report | float64 | 1 | 2 | 0.0884 | 0.164 | 0.172 | 0.0902 | 0.106 | - | 0/0 | 0.039 | 1 |
| REF-B_last | report | float64 | 2 | 2 | 0.0522 | 0.0658 | 0.0639 | 0.00234 | 0.0731 | - | 0/0 | 0.0089 | 0.87 |
| REF-B_last | report | float64 | 3 | 2 | 0.797 | 1.93 | 0.79 | 2.66 | 1.14 | 1.85e+05 | 0/2 | 0.0449 | 0 |

### Per-family balanced validation set (200 trajectories, 50 per family; `refB_valid`), float64

| run | set | dtype | family | n | T med | T p95 | T final | nHI final | nHII final | cliff err [yr] | cliffs missed | mass viol. | E up frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A4-C-B_BEST | refB_valid | float64 | all | 200 | 0.151 | 0.292 | 0.291 | 0.041 | 0.498 | 0 | 0/97 | 8.88e-16 | 0 |
| A4-C-B_BEST | refB_valid | float64 | 0 | 50 | 0.162 | 0.306 | 0.325 | 0.452 | 1.21e-08 | 0 | 0/27 | 0 | 0 |
| A4-C-B_BEST | refB_valid | float64 | 1 | 50 | 0.131 | 0.199 | 0.204 | 0.154 | 0.341 | 0 | 0/1 | 6.66e-15 | 0 |
| A4-C-B_BEST | refB_valid | float64 | 2 | 50 | 0.237 | 0.522 | 0.563 | 0.0048 | 0.99 | 0 | 0/19 | 8.88e-16 | 0 |
| A4-C-B_BEST | refB_valid | float64 | 3 | 50 | 0.123 | 0.353 | 0.298 | 0.00741 | 0.59 | 0 | 0/50 | 4.44e-16 | 0 |
| A4-C-B_last | refB_valid | float64 | all | 200 | 0.0928 | 0.174 | 0.143 | 0.0312 | 0.746 | 0 | 0/97 | 8.88e-16 | 0 |
| A4-C-B_last | refB_valid | float64 | 0 | 50 | 0.0971 | 0.178 | 0.187 | 0.283 | 4.31e-09 | 0 | 0/27 | 0 | 0 |
| A4-C-B_last | refB_valid | float64 | 1 | 50 | 0.0816 | 0.121 | 0.0819 | 0.171 | 3.78 | 0 | 0/1 | 3.55e-15 | 0 |
| A4-C-B_last | refB_valid | float64 | 2 | 50 | 0.141 | 0.242 | 0.248 | 0.00494 | 0.97 | 0 | 0/19 | 8.88e-16 | 0 |
| A4-C-B_last | refB_valid | float64 | 3 | 50 | 0.0888 | 0.318 | 0.129 | 0.00573 | 0.799 | 0 | 0/50 | 4.44e-16 | 0 |
| A5-L-B_BEST | refB_valid | float64 | all | 200 | 0.00169 | 0.00287 | 0.00151 | 0.003 | 0.00691 | 0 | 5/97 | 4.44e-16 | 0 |
| A5-L-B_BEST | refB_valid | float64 | 0 | 50 | 0.000725 | 0.00112 | 0.000903 | 0.00683 | 1.32e-10 | 0 | 0/27 | 0 | 0 |
| A5-L-B_BEST | refB_valid | float64 | 1 | 50 | 0.000379 | 0.000906 | 0.000679 | 0.00164 | 0.00278 | 0 | 0/1 | 8.88e-15 | 0 |
| A5-L-B_BEST | refB_valid | float64 | 2 | 50 | 0.0013 | 0.0018 | 0.00116 | 8.92e-05 | 0.00978 | 0 | 1/19 | 8.88e-16 | 0 |
| A5-L-B_BEST | refB_valid | float64 | 3 | 50 | 1.35 | 2.23 | 1.36 | 5.55 | 2.32 | 0 | 4/50 | 0 | 0 |
| A5-L-B_last | refB_valid | float64 | all | 200 | 0.0015 | 0.0032 | 0.00157 | 0.00123 | 0.00176 | 0 | 4/97 | 4.44e-16 | 0 |
| A5-L-B_last | refB_valid | float64 | 0 | 50 | 0.000669 | 0.000956 | 0.000671 | 0.0115 | 2.31e-10 | 0 | 0/27 | 0 | 0 |
| A5-L-B_last | refB_valid | float64 | 1 | 50 | 0.000241 | 0.000448 | 0.000342 | 0.000587 | 0.00139 | 0 | 0/1 | 1.11e-14 | 0 |
| A5-L-B_last | refB_valid | float64 | 2 | 50 | 0.00106 | 0.00175 | 0.00103 | 1.53e-05 | 0.00223 | 0 | 1/19 | 8.88e-16 | 0 |
| A5-L-B_last | refB_valid | float64 | 3 | 50 | 1.36 | 2.32 | 1.39 | 5.66 | 2.17 | 0 | 3/50 | 0 | 0 |
| REF-A_BEST | refB_valid | float64 | all | 200 | 0.392 | 1.06 | 0.913 | 1.95 | 0.806 | 0 | 4/97 | 0.824 | 0.799 |
| REF-A_BEST | refB_valid | float64 | 0 | 50 | 0.437 | 1.2 | 1.16 | 2.43 | 0.414 | 0 | 0/27 | 0.99 | 0.962 |
| REF-A_BEST | refB_valid | float64 | 1 | 50 | 0.101 | 0.243 | 0.157 | 0.559 | 0.398 | - | 1/1 | 0.94 | 0.345 |
| REF-A_BEST | refB_valid | float64 | 2 | 50 | 0.238 | 0.392 | 0.293 | 0.504 | 0.775 | 0 | 0/19 | 0.987 | 0.596 |
| REF-A_BEST | refB_valid | float64 | 3 | 50 | 2.38 | 3.04 | 3.08 | 11.3 | 2.68 | 0 | 3/50 | 0.442 | 0.905 |
| REF-A_last | refB_valid | float64 | all | 200 | 0.228 | 0.694 | 0.364 | 0.686 | 0.435 | 0 | 3/97 | 1.08 | 0.066 |
| REF-A_last | refB_valid | float64 | 0 | 50 | 0.196 | 0.376 | 0.395 | 1.14 | 0.379 | 0 | 0/27 | 0.857 | 0.762 |
| REF-A_last | refB_valid | float64 | 1 | 50 | 0.298 | 0.623 | 0.592 | 0.366 | 1.23 | 0 | 0/1 | 0.75 | 0.00576 |
| REF-A_last | refB_valid | float64 | 2 | 50 | 0.111 | 0.195 | 0.118 | 0.686 | 0.413 | 0 | 0/19 | 1.42 | 0.00212 |
| REF-A_last | refB_valid | float64 | 3 | 50 | 0.258 | 1.47 | 0.253 | 1.54 | 0.413 | 0 | 3/50 | 1.92 | 0.069 |
| REF-B_BEST | refB_valid | float64 | all | 200 | 0.82 | 2.69 | 3.26 | 2.94 | 1.77 | 0 | 0/97 | 4.25 | 0 |
| REF-B_BEST | refB_valid | float64 | 0 | 50 | 0.989 | 1.96 | 2.11 | 2.07 | 0.0959 | 0 | 0/27 | 0.335 | 4e-05 |
| REF-B_BEST | refB_valid | float64 | 1 | 50 | 0.579 | 3.72 | 4.74 | 6.18 | 2.36 | 0 | 0/1 | 13.1 | 0.00026 |
| REF-B_BEST | refB_valid | float64 | 2 | 50 | 0.67 | 2.6 | 3.7 | 3 | 2.81 | 0 | 0/19 | 6.75 | 0 |
| REF-B_BEST | refB_valid | float64 | 3 | 50 | 0.926 | 3.01 | 2.3 | 1.47 | 1.86 | 2.1e+04 | 0/50 | 5.44 | 0 |
| REF-B_last | refB_valid | float64 | all | 200 | 0.0462 | 0.087 | 0.0731 | 0.043 | 0.0628 | 0 | 5/97 | 0.0337 | 4e-05 |
| REF-B_last | refB_valid | float64 | 0 | 50 | 0.0216 | 0.0414 | 0.0409 | 0.0541 | 0.0189 | 0 | 2/27 | 0.051 | 0.00228 |
| REF-B_last | refB_valid | float64 | 1 | 50 | 0.058 | 0.087 | 0.0901 | 0.0341 | 0.06 | 0 | 0/1 | 0.0233 | 0.137 |
| REF-B_last | refB_valid | float64 | 2 | 50 | 0.0301 | 0.0464 | 0.0401 | 0.00701 | 0.0869 | 0 | 0/19 | 0.0192 | 0.374 |
| REF-B_last | refB_valid | float64 | 3 | 50 | 0.572 | 1.91 | 1.79 | 5.6 | 2.1 | 0 | 3/50 | 0.0541 | 0 |
