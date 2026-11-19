"""Pass 3: sensitivity analyses and extension experiments (Deliverable 3, Steps 7 and 10).

Each variant changes one factor of the main design and re-runs, on scenario S3, the
pre-deletion training, the CAPRI-Audit signals, the retrain oracle (B1), exemplar
fine-tuning (B8) and CAPRI-Guard-S on B1. The audit model used for the flags is fitted
on the full pass-1 audit table of the same dataset (default design), so the variant
tests whether the fitted audit and the repair still work when the setting changes.

Variants
  alpha0.1, alpha1.0   Dirichlet label skew of non-target classes (default 0.5)
  K20                  20 clients instead of 10
  cnn                  1D-CNN detector instead of the MLP
  unweighted           plain cross-entropy instead of inverse-sqrt class weighting
  partial0.5           5 of 10 clients sampled per round (extension 2)
  sequential           a second client leaves 10 rounds after the first (extension 1)
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .capri import (SIGNAL_SETS, fit_audit, predict_audit, audit_labels, rule_audit, guard, leave_u_out_estimate,
                    audit_signals, REQUIRED, COLLATERAL)
from .evaluate import evaluate_model
from .experiment import build, DEPART, _save
from .fl import FLConfig
from .partition import ownership
from . import unlearn as U

RHO = 0.8
VARIANTS = {
    "alpha0.1": {"alpha": 0.1}, "alpha1.0": {"alpha": 1.0}, "K20": {"n_clients": 20},
    "cnn": {"cfg": {"arch": "cnn"}}, "unweighted": {"weighted": False},
    "partial0.5": {"cfg": {"participation": 0.5}}, "sequential": {},
}


def fitted_audit(audit_df: pd.DataFrame):
    sig = SIGNAL_SETS["all"]
    return fit_audit(audit_df.dropna(subset=sig + ["r_oracle"]), sig), sig


def flags_for(sig_rows, glm, sig):
    d = pd.DataFrame(sig_rows)
    rhat = predict_audit(glm, d.fillna(0), sig)
    lab = audit_labels(rhat, d.m.to_numpy(), RHO)
    return d.family[np.isin(lab, [REQUIRED, COLLATERAL])].astype(int).tolist(), rhat.tolist()


def run_variant(cl, variant: str, seed: int, out: Path, device: str, audit_df: pd.DataFrame,
                base_cfg: FLConfig = FLConfig(), log=print, scenario: str = "S3") -> dict:
    name = f"p3_{variant}_{scenario}_s{seed}"
    if (out / f"{name}.json").exists():
        log(f"skip {name}")
        return json.load(open(out / f"{name}.json"))
    t0 = time.time()
    spec = dict(VARIANTS[variant])
    cfg = replace(base_cfg, **spec.pop("cfg", {}))
    cx = build(cl, scenario, seed, device, cfg=cfg, **spec)
    tr, X, y = cx.tr, cx.X, cx.y
    glm, sig = fitted_audit(audit_df)
    theta0 = tr.init_theta(seed)
    pre = tr.train(theta0, X, y, cx.parts, seed=seed, Xval=cx.Xval, yval=cx.yval)
    theta_est = leave_u_out_estimate(pre, DEPART)
    chk_y = y[cx.ex_chk].cpu().numpy()
    sigs = audit_signals(tr, theta_est, X[cx.ex_chk], chk_y, cx.s, cx.m, cx.families.tolist())
    flagged, rhat = flags_for(sigs, glm, sig)
    res, th = {}, {"B0": pre.theta}
    th["B1"] = U.retrain(tr, theta0, X, y, cx.parts, DEPART, seed, cx.Xval, cx.yval).theta
    th["B8[B1]"] = U.finetune_exemplars(tr, th["B1"], X, y, cx.ex_fit, seed)
    th["Guard[B1]"] = guard(tr, th["B1"], X, y, cx.ex_fit, cx.ex_chk, flagged, seed)[0]
    extra = {}
    if variant == "sequential":
        # the remaining clients keep training for 10 rounds, then client 1 also leaves
        rest = [c if k != DEPART else c[:0] for k, c in enumerate(cx.parts)]
        mid = tr.train(th["Guard[B1]"], X, y, rest, seed=seed + 1500, rounds=10, early_stop=False,
                       keep_history=False).theta
        th["mid"] = mid
        rest2 = [c if k not in (DEPART, 1) else c[:0] for k, c in enumerate(cx.parts)]
        ytr = y[torch.cat([c for c in cx.parts])].cpu().numpy()
        # ownership of client 1 relative to the clients still present after the first deletion
        parts_np = [c.cpu().numpy() for c in rest]
        yall = y.cpu().numpy()
        n2 = np.stack([np.bincount(yall[p], minlength=tr.C) for p in parts_np])
        tot2 = n2.sum(0)
        s2 = np.where(tot2 > 0, n2[1] / np.maximum(tot2, 1), 0.0)
        m2 = tot2 - n2[1]
        rule2 = [int(c) for c in cx.families if rule_audit(np.array([s2[c]]), np.array([m2[c]]))[0]]
        flagged2 = sorted(set(flagged) | set(rule2))
        th["B1_2"] = tr.train(theta0, X, y, rest2, seed=seed + 1600, Xval=cx.Xval, yval=cx.yval,
                              keep_history=False).theta
        th["B8[B1_2]"] = U.finetune_exemplars(tr, th["B1_2"], X, y, cx.ex_fit, seed + 1)
        th["Guard[B1_2]"] = guard(tr, th["B1_2"], X, y, cx.ex_fit, cx.ex_chk, flagged2, seed + 1)[0]
        extra = {"flagged_second": flagged2, "s_client1": s2.tolist(), "m_client1": m2.tolist()}
    pre_eval, _, _ = evaluate_model(tr, pre.theta, cx.ev, cx.families, seed=seed)
    res["B0"] = pre_eval
    tgt = pre_eval["forget_acc"] - 0.01
    for k, v in th.items():
        if k != "B0":
            res[k] = evaluate_model(tr, v, cx.ev, cx.families, relearn_target=tgt, seed=seed)[0]
    rec = {**cx.meta, "variant": variant, "n": cx.n, "s": cx.s, "m": cx.m, "flagged": flagged, "rhat": rhat,
           "audit_signals": sigs, "results": res, "pre_rounds": pre.rounds_run, "pre_best_round": pre.best_round,
           "seconds": time.time() - t0, **extra}
    _save(out, name, rec, {})
    log(f"{name}: {time.time() - t0:.0f}s | " + " ".join(
        f"{k}:DD={v['dd']['rho0.8']['dd_count']},def={v['dd']['rho0.8']['dd_deficit']:.2f}" for k, v in res.items()))
    return rec
