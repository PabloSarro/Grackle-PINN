"""Training loop (A6). One JSON config per run, see configs/.

Differences from the June script (TASKS.md, R2 and A6):
  - the mass-loss ramp and the stop are tied to optimiser steps / wall-time budget, not to a nominal epoch count;
  - resume restores epoch, step, ramp position, optimiser, scheduler, best score and the elapsed training time,
    and appends to the loss file; the data order of epoch e depends only on (seed, e), so a resumed run
    reproduces an uninterrupted one;
  - the best checkpoint is chosen on a short free-rollout score (the model only sees a window's first state and
    its own predictions), not on the one-step MSE (still logged);
  - data live on the GPU as tensors; seeds are fixed; the config, architecture, scaling and parameterisation are
    stored in the run folder and in every checkpoint.
"""
import os
import sys
import json
import time
import shutil
import argparse

import torch

from grackle_pinn.model import PINN
from grackle_pinn.param import build as build_param
from grackle_pinn import physics_terms as PTERMS
PHYS_FUNCS = {"P3": PTERMS.p3_energy_monotone, "P4": PTERMS.p4_chemistry, "P5": PTERMS.p5_equilibrium, "P6": PTERMS.p6_energy}
from grackle_pinn.data_utils import GrackleDataset
from grackle_pinn.physics import PhysicsLossManager
from grackle_pinn.evaluate import window_rollout_score

DEFAULTS = dict(
    run_name=None, train_path=None, valid_path=None, out_dir="results/runs",
    hidden_dim=512, hidden_layers=12, parameterisation="report",
    batch_size=8192, learning_rate=1.0e-4, epochs=900, check_every=10,
    weight_factor=10.0, lambda_mass_max=1.0e7, ramp_fraction=1.0,
    physics={},               # B3: {"P3": w, "P4": w, "P5": w, "P6": w} fixed weights on the residuals of physics_terms (P1 = report mass term, see lambda_mass_max)
    phys_weight=0.0,          # A5: weight of an extra MSE on the physical increments (d ln e, d logit x), each divided by its std
    weight_physical=False,    # A5: the 1 + weight_factor |y| weight uses the standardised physical increment instead of the training target
    legacy_resume_epoch=None,                         # only for a checkpoint of the June loop (no epoch stored): epochs already done
    max_runtime_hours=23.5, budget_hours=None,        # None: ramp follows optimiser steps only; a number: also follows that wall-clock budget
    scheduler_patience=5, scheduler_factor=0.5, min_lr=1e-6, grad_clip=50.0,
    seed=0, dtype="float64", stop_after_epochs=None,   # the last one is for tests of resume
    select_on="rollout",                              # "rollout" (free-rollout window score) or "mse"
    rollout_trajectories=32, rollout_window=2000, rollout_starts=(0, 5000, 10000, 15000, 20000),
)


def parse():
    ap = argparse.ArgumentParser(description="Train Grackle PINN")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="checkpoint to resume from (or to start from, if it has no epoch)")
    return ap.parse_args()


def main():
    args = parse()
    starting_time = time.time()
    cfg = dict(DEFAULTS)
    with open(args.config) as f:
        cfg.update(json.load(f))
    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    if os.path.abspath(args.config) != os.path.abspath(os.path.join(run_dir, "config.json")):
        shutil.copy(args.config, os.path.join(run_dir, "config.json"))
    out = lambda n: os.path.join(run_dir, n)
    MODEL_NAME, BEST_NAME = out(f"PINN_{cfg['run_name']}.pth"), out(f"PINN_BEST_{cfg['run_name']}.pth")
    SCALING_NAME, LOSSES_NAME = out(f"scaling_params_{cfg['run_name']}.pth"), out(f"losses_{cfg['run_name']}.txt")

    dtype = getattr(torch, cfg["dtype"])
    torch.set_default_dtype(dtype)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, precision {dtype}, parameterisation {cfg['parameterisation']}", flush=True)

    # ---- resume information (needed before the data to know the parameterisation settings)
    ckpt = None
    if args.model is not None:
        if not os.path.exists(args.model):
            print(f"[!] CRITICAL: '{args.model}' not found"); sys.exit(1)
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    param_cfg = {"name": cfg["parameterisation"]}
    if ckpt is not None and "param" in ckpt:
        param_cfg = ckpt["param"]
    param = build_param(param_cfg)

    # ---- data
    keep_raw = any(k != "P1" for k in cfg["physics"])
    train_set = GrackleDataset(cfg["train_path"], param=param, keep_raw=keep_raw)
    if param.name == "constrained" and ckpt is None:
        param = build_param({"name": "constrained", "lmax": 1.25 * float(train_set.target_absmax[1])})
        train_set = GrackleDataset(cfg["train_path"], param=param, keep_raw=keep_raw)   # same data, only the bound changes
        print(f"constrained: |d logit x| <= {param.lmax:.3f}", flush=True)
    if param.name == "lograte" and ckpt is None:
        m = 2.0   # nats of margin beyond the training range for the clip used in rollouts
        param = build_param({"name": "lograte", "lo": (train_set.target_min - m).tolist(), "hi": (train_set.target_max + m).tolist()})
        train_set = GrackleDataset(cfg["train_path"], param=param, keep_raw=keep_raw)
        print(f"lograte: clip range lo {param.lo} hi {param.hi}", flush=True)
    sc_cpu = {"in_mean": train_set.in_mean, "in_std": train_set.in_std, "tg_mean": train_set.tg_mean, "tg_std": train_set.tg_std}
    if ckpt is not None and "scaling" in ckpt:
        sc_cpu = ckpt["scaling"]
        train_set = GrackleDataset(cfg["train_path"], param=param, keep_raw=keep_raw, in_mean=sc_cpu["in_mean"], in_std=sc_cpu["in_std"],
                                   tg_mean=sc_cpu["tg_mean"], tg_std=sc_cpu["tg_std"])
    valid_set = GrackleDataset(cfg["valid_path"], param=param, **sc_cpu)
    torch.save(sc_cpu, SCALING_NAME)
    sc = {k: v.to(device) for k, v in sc_cpu.items()}
    X, Y = train_set.all_inputs.to(device), train_set.all_targets.to(device)
    PT_NAMES = [k for k in cfg["physics"] if k != "P1"]
    if PT_NAMES:                       # B3: current states of the training pairs (physical log-state)
        CUR = train_set.raw_cur.to(device)
    Xv, Yv = valid_set.all_inputs.to(device), valid_set.all_targets.to(device)
    N = X.shape[0]
    BS = cfg["batch_size"]
    steps_per_epoch = (N + BS - 1) // BS
    planned_steps = cfg["epochs"] * steps_per_epoch
    print(f"Training on {N} pairs, {steps_per_epoch} steps per epoch, {planned_steps} planned steps", flush=True)

    use_rollout = cfg["select_on"] == "rollout" and valid_set.logstate is not None
    if use_rollout:
        val_ls = torch.from_numpy(valid_set.logstate[:cfg["rollout_trajectories"]]).to(device=device, dtype=dtype)

    # ---- model, optimiser, scheduler
    model = PINN(param.in_dim, param.out_dim, hidden_dim=cfg["hidden_dim"], hidden_layers=cfg["hidden_layers"]).to(device)
    if ckpt is None:
        param.init_model(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"], min_lr=cfg["min_lr"])

    start_epoch, step, best_score, elapsed_before, resumed = 0, 0, float("inf"), 0.0, False
    if ckpt is not None:
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if "optimizer" in ckpt:
            optimiser.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:      # a checkpoint of this code: continue exactly
            start_epoch, step = ckpt["epoch"], ckpt["step"]
            best_score, elapsed_before, resumed = ckpt["best_score"], ckpt["elapsed"], True
        elif cfg["legacy_resume_epoch"] is not None:   # June-loop checkpoint: continue the epoch counter and the ramp by hand
            start_epoch = int(cfg["legacy_resume_epoch"])
            step = start_epoch * ((N + BS - 1) // BS)
            best_score, resumed = float(ckpt.get("best_loss", float("inf"))), True
        print(f"[✓] loaded {args.model}; resumed={resumed}, start epoch {start_epoch}, step {step}", flush=True)

    use_phys = cfg["phys_weight"] > 0 or cfg["weight_physical"]
    if use_phys:                       # physical increments of the training pairs and their std (constrained and lograte only)
        with torch.no_grad():
            P = torch.cat([param.to_physical(Y[i:i + (1 << 22)], sc) for i in range(0, N, 1 << 22)])
            p_std = P.std(dim=0)
            del P
        print(f"physical increment std {p_std.tolist()}", flush=True)

    phys = PhysicsLossManager(model, sc["in_mean"], sc["in_std"], sc["tg_mean"], sc["tg_std"])
    arch = {"hidden_dim": cfg["hidden_dim"], "hidden_layers": cfg["hidden_layers"],
            "input_dim": param.in_dim, "output_dim": param.out_dim}

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimizer": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "step": step, "best_score": best_score,
                    "elapsed": elapsed_before + time.time() - starting_time,
                    "arch": arch, "param": param.config(), "scaling": sc_cpu, "config": cfg}, path)

    if not resumed:
        with open(LOSSES_NAME, "w") as f:
            f.write("Epoch AvgDataLoss AvgMassLoss AvgTotalLoss ValidationMSE RolloutScore LR\n")

    @torch.no_grad()
    def validation_mse():
        model.eval()
        tot = 0.0
        for i in range(0, Xv.shape[0], 1 << 17):
            p = param.to_standardised(model(Xv[i:i + (1 << 17)]), sc)
            tot += ((p - Yv[i:i + (1 << 17)]) ** 2).sum().item()
        return tot / Yv.numel()

    budget_s = (cfg["budget_hours"] or cfg["max_runtime_hours"]) * 3600  # ramp follows it only if budget_hours is set
    stop_s = cfg["max_runtime_hours"] * 3600
    stopped_early = False
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(cfg["seed"] * 100003 + epoch)
        perm = torch.randperm(N, generator=g).to(device)
        tot = dat = mas = phy = 0.0
        for i in range(0, N, BS):
            idx = perm[i:i + BS]
            bx, by = X[idx], Y[idx]
            optimiser.zero_grad()
            out_raw = model(bx)
            pred = param.to_standardised(out_raw, sc)
            if use_phys:
                true_p = param.to_physical(by, sc)
                ys = true_p / p_std if cfg["weight_physical"] else by
            else:
                ys = by
            w = 1.0 + cfg["weight_factor"] * torch.abs(ys)
            loss_data = torch.mean(w * (pred - by) ** 2)
            if cfg["phys_weight"] > 0:
                loss_data = loss_data + cfg["phys_weight"] * torch.mean(((param.to_physical(out_raw, sc) - true_p) / p_std) ** 2)
            if param.name == "report":
                loss_mass = phys.mass_conserv(bx, pred)
                progress = step / (cfg["ramp_fraction"] * planned_steps)
                if cfg["budget_hours"]:      # optional: also follow the wall-clock budget (not reproducible bit by bit)
                    progress = max(progress, (elapsed_before + time.time() - starting_time) / (cfg["ramp_fraction"] * budget_s))
                progress = min(1.0, progress)
                lam = cfg["lambda_mass_max"] * progress
            else:   # mass conservation is exact by construction
                loss_mass, lam = torch.zeros((), device=device), 0.0
            loss = loss_data + lam * loss_mass
            if PT_NAMES:
                loss_phys = torch.zeros((), device=device)
                # residuals of the predicted next state (one 100-yr step); each term is clamped at 1 so a few
                # cliff steps cannot dominate the mean
                nxt_pred = param.advance(CUR[idx], param.to_physical(out_raw, sc))
                for k in PT_NAMES:
                    r = PHYS_FUNCS[k](CUR[idx], nxt_pred)
                    term = cfg["physics"][k] * torch.clamp(r ** 2, max=1.0).mean()
                    loss = loss + term; loss_phys = loss_phys + term.detach()
            if torch.isnan(loss):
                print(f"[!] NaN at epoch {epoch + 1}, step {step}: data {loss_data.item():.4e}, mass {loss_mass.item():.4e}")
                sys.exit(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimiser.step()
            step += 1
            tot += loss.item(); dat += loss_data.item(); mas += loss_mass.item()
            phy += float(loss_phys) if PT_NAMES else 0.0
        nb = steps_per_epoch
        mse_s, roll_s = "nan", "nan"
        if epoch % cfg["check_every"] == 0 or epoch == cfg["epochs"] - 1:
            mse = validation_mse()
            mse_s = f"{mse:.6e}"
            if use_rollout:
                score = window_rollout_score(model, param, sc, val_ls, cfg["rollout_window"], cfg["rollout_starts"])
                roll_s = f"{score:.6e}"
            else:
                score = mse
            scheduler.step(score)
            if score < best_score:
                best_score = score
                save(BEST_NAME, epoch + 1)
                print(f"[✓] new best ({cfg['select_on'] if use_rollout else 'mse'}) {score:.6e} at epoch {epoch + 1}, MSE {mse_s}", flush=True)
        lr = optimiser.param_groups[0]["lr"]
        print(f"Epoch [{epoch + 1}/{cfg['epochs']}] loss {tot / nb:.6e} data {dat / nb:.6e} mass {mas / nb:.6e} phys {phy / nb:.6e} "
              f"val {mse_s} roll {roll_s} lr {lr:.2e} ({time.time() - t0:.1f} s)", flush=True)
        with open(LOSSES_NAME, "a") as f:
            f.write(f"{epoch + 1} {dat / nb:.6e} {mas / nb:.6e} {tot / nb:.6e} {mse_s} {roll_s} {lr:.3e}\n")
        if (epoch + 1) % 20 == 0:
            save(MODEL_NAME, epoch + 1)
        if time.time() - starting_time > stop_s or (cfg["stop_after_epochs"] and epoch + 1 >= cfg["stop_after_epochs"]):
            save(MODEL_NAME, epoch + 1)
            print(f"[!] wall-time limit reached after epoch {epoch + 1}; checkpoint saved, resume with --model {MODEL_NAME}", flush=True)
            stopped_early = True
            break
    if not stopped_early:
        save(MODEL_NAME, cfg["epochs"])
    print(f"Training {'stopped' if stopped_early else 'complete'} in {(time.time() - starting_time) / 3600:.2f} h.", flush=True)


if __name__ == "__main__":
    main()
