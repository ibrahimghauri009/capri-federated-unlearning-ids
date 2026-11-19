"""Experiment runner.

Pass 1 (per scenario x seed): pre-deletion FedAvg model, CAPRI-Audit signals (computed
and saved before any unlearning runs), and baselines B0-B5. All parameters, test
predictions and metrics are logged.
Audit fitting: leave-one-seed-out binomial GLM on the pass-1 signal table.
Pass 2 (per scenario x seed): CAPRI-Guard on top of B1, B4, B5 with the audit's
flagged families, B8 exemplar fine-tuning, and the Guard/Audit ablations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import make_split, prepare, leakage_checks, select_targets, Clean
from .fl import FedTrainer, FLConfig
from .partition import partition, ownership
from .capri import leave_u_out_estimate, audit_signals
from .evaluate import evaluate_model
from . import unlearn as U

K, DEPART = 10, 0


@dataclass
class Ctx:
    tr: FedTrainer
    X: torch.Tensor
    y: torch.Tensor
    parts: list
    ex_fit: torch.Tensor
    ex_chk: torch.Tensor
    Xval: torch.Tensor
    yval: torch.Tensor
    ev: dict
    families: np.ndarray
    n: np.ndarray
    s: np.ndarray
    m: np.ndarray
    meta: dict


def build(cl: Clean, scenario: str, seed: int, device: str, k_ex: int = 50, cfg: FLConfig = FLConfig(),
          n_clients: int = K, alpha: float = 0.5, weighted: bool = True) -> Ctx:
    sp = make_split(cl, seed, k=k_ex)
    pr = prepare(cl, sp)
    lk = leakage_checks(cl, sp, pr)
    if not lk["passed"]:
        raise RuntimeError(f"leakage check failed: {lk}")
    tg = select_targets(cl, k=k_ex)
    targets = list(tg["targets"].values())
    C = len(pr.classes)
    ytr = pr.y["train"]
    parts_np = partition(ytr, n_clients, scenario, targets, seed, alpha=alpha)
    n, s, m = ownership(ytr, parts_np, C, DEPART)
    # class weights: inverse square root of the global frequency (federated label counts)
    tot = n.sum(0).astype(float)
    cw = np.where(tot > 0, 1.0 / np.sqrt(np.maximum(tot, 1) / tot.sum()), 0.0)
    cw = cw / cw[tot > 0].mean()
    cw[tot == 0] = 1.0
    if not weighted:      # sensitivity check: plain unweighted cross-entropy
        cw = np.ones_like(cw)
    # exemplar split: half fit, half check, per class
    rng = np.random.default_rng(seed + 31)
    yex = pr.y["ex"]
    fit, chk = [], []
    for c in np.unique(yex):
        idx = rng.permutation(np.flatnonzero(yex == c))
        h = len(idx) // 2
        fit += idx[:h].tolist()
        chk += idx[h:].tolist()
    ntr = len(ytr)
    dev = device
    X = torch.tensor(np.concatenate([pr.X["train"], pr.X["ex"]]), device=dev)
    y = torch.tensor(np.concatenate([ytr, yex]), device=dev)
    parts = [torch.tensor(p, device=dev) for p in parts_np]
    ex_fit = torch.tensor(np.array(fit, dtype=np.int64) + ntr, device=dev)
    ex_chk = torch.tensor(np.array(chk, dtype=np.int64) + ntr, device=dev)
    Xte = torch.tensor(pr.X["test"], device=dev)
    yte = torch.tensor(pr.y["test"], device=dev)
    # evaluable families: attack families with >= 30 test records
    sup = np.bincount(pr.y["test"], minlength=C)
    families = np.array([c for c in range(C) if c != pr.benign and sup[c] >= 30])
    # membership-inference sample: client u's records vs class-matched unseen test records
    from .metrics import class_matched_nonmembers
    y0 = ytr[parts_np[DEPART]]
    mem_rel, non = class_matched_nonmembers(y0, pr.y["test"], np.random.default_rng(seed + 41), 5000)
    ev = {"Xte": Xte, "yte": yte, "yte_np": pr.y["test"], "X": X, "y": y,
          "mem": parts[DEPART][torch.tensor(mem_rel, device=dev)], "non": torch.tensor(non, device=dev),
          "forget_all": parts[DEPART], "y_forget_np": y0}
    tr = FedTrainer(X.shape[1], C, torch.tensor(cw, dtype=torch.float32), dev, cfg)
    meta = {"scenario": scenario, "seed": seed, "classes": pr.classes, "targets": tg,
            "features": pr.features, "prep": pr.meta, "leakage": lk,
            "client_sizes": [int(len(p)) for p in parts_np], "class_weights": cw.tolist(),
            "test_support": sup.tolist(), "families_evaluable": families.tolist(),
            "y_test_checksum": int(np.sum(pr.y["test"] * np.arange(len(pr.y["test"])) % 1000003))}
    return Ctx(tr, X, y, parts, ex_fit, ex_chk, torch.tensor(pr.X["val"], device=dev),
               torch.tensor(pr.y["val"], device=dev), ev, families, n, s, m, meta)


def _save(out: Path, name: str, rec: dict, arrays: dict):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.json", "w") as f:
        json.dump(rec, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(out / f"{name}.npz", **arrays)


def run_pass1(cl: Clean, scenario: str, seed: int, out: Path, device: str, cfg: FLConfig = FLConfig(),
              log=print) -> dict:
    name = f"p1_{scenario}_s{seed}"
    if (out / f"{name}.json").exists():
        log(f"skip {name}")
        return json.load(open(out / f"{name}.json"))
    t0 = time.time()
    cx = build(cl, scenario, seed, device, cfg=cfg)
    tr, X, y = cx.tr, cx.X, cx.y
    theta0 = tr.init_theta(seed)
    timings = {}
    t = time.time()
    pre = tr.train(theta0, X, y, cx.parts, seed=seed, Xval=cx.Xval, yval=cx.yval)
    timings["pre"] = time.time() - t
    log(f"{name}: pre-deletion trained, rounds={pre.rounds_run} best={pre.best_round} "
        f"val_f1={max(pre.val_curve):.4f} ({timings['pre']:.0f}s)")
    # ---- CAPRI-Audit signals: computed and saved BEFORE any unlearning ----
    t = time.time()
    theta_est = leave_u_out_estimate(pre, DEPART)
    chk_y = y[cx.ex_chk].cpu().numpy()
    sig = audit_signals(tr, theta_est, X[cx.ex_chk], chk_y, cx.s, cx.m, cx.families.tolist())
    timings["audit_signals"] = time.time() - t
    _save(out, f"{name}_audit", {"signals": sig, "s": cx.s, "m": cx.m}, {"theta_est": theta_est.numpy()})
    thetas = {"B0": pre.theta}
    t = time.time(); r1 = U.retrain(tr, theta0, X, y, cx.parts, DEPART, seed, cx.Xval, cx.yval)
    thetas["B1"] = r1.theta; timings["B1"] = time.time() - t
    t = time.time(); r2 = U.lawful_oracle(tr, theta0, X, y, cx.parts, DEPART, cx.ex_fit, seed, cx.Xval, cx.yval)
    thetas["B2"] = r2.theta; timings["B2"] = time.time() - t
    t = time.time(); thetas["B3"] = U.continued(tr, pre.theta, X, y, cx.parts, DEPART, seed); timings["B3"] = time.time() - t
    t = time.time(); thetas["B4"], ga = U.gradient_ascent(tr, pre, X, y, cx.parts, DEPART, seed); timings["B4"] = time.time() - t
    t = time.time(); thetas["B5"] = U.federaser(tr, pre, X, y, cx.parts, DEPART, seed); timings["B5"] = time.time() - t
    # ---- evaluation ----
    res, preds, cms = {}, {}, {}
    pre_eval, preds["B0"], cms["B0"] = evaluate_model(tr, pre.theta, cx.ev, cx.families, seed=seed)
    target_acc = pre_eval["forget_acc"] - 0.01
    res["B0"] = pre_eval
    for b in ("B1", "B2", "B3", "B4", "B5"):
        res[b], preds[b], cms[b] = evaluate_model(tr, thetas[b], cx.ev, cx.families,
                                                  relearn_target=target_acc, seed=seed)
    timings["total"] = time.time() - t0
    rec = {**cx.meta, "n": cx.n, "s": cx.s, "m": cx.m, "results": res, "timings": timings,
           "pre_rounds": pre.rounds_run, "pre_best_round": pre.best_round, "pre_val_curve": pre.val_curve,
           "B1_rounds": r1.best_round, "B2_rounds": r2.best_round, "ga": ga, "audit_signals": sig}
    arrays = {f"theta_{k}": v.cpu().numpy() for k, v in thetas.items()}
    arrays.update({f"pred_{k}": v for k, v in preds.items()})
    arrays.update({f"cm_{k}": v for k, v in cms.items()})
    arrays["parts_sizes"] = np.array(cx.meta["client_sizes"])
    arrays["y_test"] = cx.ev["yte_np"].astype(np.int8)
    arrays["n_counts"] = cx.n
    _save(out, name, rec, arrays)
    log(f"{name}: done in {timings['total']:.0f}s | " + " ".join(
        f"{b}:acc={res[b]['accuracy']:.3f},DD={res[b]['dd']['rho0.8']['dd_count']}" for b in res))
    return rec
