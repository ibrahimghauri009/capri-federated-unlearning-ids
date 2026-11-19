"""Pass 2: audit fitting, CAPRI-Guard, B6, B8 and ablations.

The audit model is fitted leave-one-seed-out on the pass-1 signal table: predictions
for seed s come from a model fitted only on deletions simulated with other seeds
(each seed fixes its own split, partition and initialisation). The original plan
fitted on one dataset and tested on the other; with one dataset in this phase,
leave-one-seed-out is the closest non-circular alternative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .capri import (SIGNAL_SETS, fit_audit, predict_audit, audit_labels, rule_audit, guard,
                    REQUIRED, COLLATERAL, SAFE)
from .evaluate import evaluate_model
from .experiment import build, DEPART, _save
from .fl import FLConfig
from .model import get_flat
from . import unlearn as U

RHO = 0.8


# ----------------------------------------------------------------------------
def audit_table(out: Path, runs) -> pd.DataFrame:
    rows = []
    for scen, seed in runs:
        f = out / f"p1_{scen}_s{seed}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        res = r["results"]
        for sg in r["audit_signals"]:
            c = sg["family"]
            row = {"scenario": scen, "seed": seed, **sg,
                   "r_pre": res["B0"]["recall"][c], "r_oracle": res["B1"]["recall"][c],
                   "r_lawful": res["B2"]["recall"][c], "r_B3": res["B3"]["recall"][c],
                   "r_B4": res["B4"]["recall"][c], "r_B5": res["B5"]["recall"][c],
                   "support": r["test_support"][c], "family_name": r["classes"][c]}
            rows.append(row)
    return pd.DataFrame(rows)


def fit_audits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, sig in SIGNAL_SETS.items():
        pred = np.full(len(df), np.nan)
        for s in df.seed.unique():
            tr_m = (df.seed != s).to_numpy()
            te_m = ~tr_m
            fitd = df[tr_m].dropna(subset=sig + ["r_oracle"])
            if len(fitd) < len(sig) + 2:
                continue
            res = fit_audit(fitd, sig)
            pred[te_m] = predict_audit(res, df[te_m].fillna(0), sig)
        df[f"rhat_{name}"] = pred
        df[f"label_{name}"] = audit_labels(pred, df.m.to_numpy(), RHO)
    df["flag_rule"] = rule_audit(df.s.to_numpy(), df.m.to_numpy())
    df["at_risk"] = (df.r_oracle < RHO) & (df.r_pre >= RHO)
    return df


# ----------------------------------------------------------------------------
def distill_unlearn(tr, theta_est, theta_T, X, fit_idx, seed, epochs=20, lr=1e-3, batch=64, T=2.0):
    """B6: subtract client u's stored contributions, then recover utility by distilling
    the original global model on unlabeled server-side samples (Wu et al., 2022)."""
    teacher = tr.as_model(theta_T); teacher.eval()
    student = tr.as_model(theta_est)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1111)
    for _ in range(epochs):
        student.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=g).to(fit_idx.device)]
        for i in range(0, len(perm), batch):
            xb = X[perm[i:i + batch]]
            with torch.no_grad():
                pt = F.softmax(teacher(xb) / T, 1)
            loss = F.kl_div(F.log_softmax(student(xb) / T, 1), pt, reduction="batchmean") * T * T
            opt.zero_grad(); loss.backward(); opt.step()
    return get_flat(student)


def guard_f(tr, theta_u, cx, flagged, seed, rounds=3, **kw):
    """Guard-F: remaining clients train normally for a few federated rounds while the
    server adds the exemplar repair step after each aggregation."""
    rest = [c for k, c in enumerate(cx.parts) if k != DEPART]
    theta = theta_u
    info = []
    per_round = max(1, round(20 / rounds))      # same total repair epochs as Guard-S
    for r in range(rounds):
        theta = tr.train(theta, cx.X, cx.y, rest, seed=seed + 1200 + r, rounds=1, early_stop=False,
                         keep_history=False).theta
        theta, inf = guard(tr, theta, cx.X, cx.y, cx.ex_fit, cx.ex_chk, flagged, seed + r,
                           epochs=per_round, **kw)
        info.append(inf["trace"][-1])
    return theta, {"rounds": info}


def subset_exemplars(idx: torch.Tensor, y: torch.Tensor, per_class: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = []
    yc = y[idx]
    for c in torch.unique(yc):
        sel = idx[yc == c]
        out.append(sel[torch.randperm(len(sel), generator=g)[:per_class].to(sel.device)])
    return torch.cat(out)


# ----------------------------------------------------------------------------
def run_all(cl, runs, out: Path, device: str, cfg: FLConfig = FLConfig(), log=print,
            ablation_scenarios=("S3",), bases=("B1", "B4", "B5")):
    # the audit is fitted on every available pass-1 run (leave-one-seed-out), so the
    # flags do not depend on how runs are split across jobs
    all_runs = sorted({(f.stem.split("_")[1], int(f.stem.split("_")[2][1:]))
                       for f in out.glob("p1_*_s*.json") if not f.stem.endswith("_audit")})
    df = fit_audits(audit_table(out, all_runs))
    df.to_csv(out / "audit_table.csv", index=False)
    log(f"audit table: {len(df)} family-deletions")
    for scen, seed in runs:
        name = f"p2_{scen}_s{seed}"
        if (out / f"{name}.json").exists() or not (out / f"p1_{scen}_s{seed}.json").exists():
            continue
        t0 = time.time()
        p1 = json.load(open(out / f"p1_{scen}_s{seed}.json"))
        arr = np.load(out / f"p1_{scen}_s{seed}.npz")
        aud = np.load(out / f"p1_{scen}_s{seed}_audit.npz")
        cx = build(cl, scen, seed, device, cfg=cfg)
        assert cx.meta["y_test_checksum"] == p1["y_test_checksum"], "data mismatch between passes"
        tr = cx.tr
        th = {b: torch.tensor(arr[f"theta_{b}"], device=device) for b in ("B0", "B1", "B2", "B3", "B4", "B5")}
        target_acc = p1["results"]["B0"]["forget_acc"] - 0.01
        mia_lawful = p1["results"]["B2"]["mia_auc"]
        sub = df[(df.scenario == scen) & (df.seed == seed)]
        flag = {k: sub.family[sub[f"label_{k}"].isin([REQUIRED, COLLATERAL])].astype(int).tolist()
                for k in SIGNAL_SETS}
        flag["rule"] = sub.family[sub.flag_rule].astype(int).tolist()
        flag["blind"] = cx.families.tolist()
        A = flag["all"]
        res, preds = {}, {}

        def ev(key, theta):
            res[key], preds[key], _ = evaluate_model(tr, theta, cx.ev, cx.families,
                                                     relearn_target=target_acc, seed=seed)

        # B6 distillation-based unlearning
        th["B6"] = distill_unlearn(tr, torch.tensor(aud["theta_est"], device=device), th["B0"], cx.X, cx.ex_fit, seed)
        ev("B6", th["B6"])
        guard_info = {}
        for b in bases:
            ev(f"B8[{b}]", U.finetune_exemplars(tr, th[b], cx.X, cx.y, cx.ex_fit, seed))
            g_th, inf = guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed)
            ev(f"Guard[{b}]", g_th)
            gap = res[f"Guard[{b}]"]["mia_auc"] - mia_lawful
            inf["mia_gap"] = gap
            if gap > 0.02:   # safeguard: double the proximity weight and rerun once
                g_th, inf2 = guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, gamma=2e-2)
                ev(f"Guard[{b}]", g_th)
                inf["rerun"] = True
                inf["mia_gap_after"] = res[f"Guard[{b}]"]["mia_auc"] - mia_lawful
            guard_info[b] = {k: v for k, v in inf.items() if k != "trace"} | {"trace_last": inf["trace"][-1]}
            gf, _ = guard_f(tr, th[b], cx, A, seed)
            ev(f"GuardF[{b}]", gf)
        if scen in ablation_scenarios:
            for b in bases:
                ev(f"Abl-fixed[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, dual_ascent=False)[0])
                ev(f"Abl-noanchor[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, anchor=False)[0])
                ev(f"Abl-noprox[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, proximity=False)[0])
                ev(f"Abl-blind[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, flag["blind"], seed)[0])
                ev(f"Abl-ruleflags[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, flag["rule"], seed)[0])
            b = "B1"
            for k in (2, 5, 10):   # exemplars per class in each half (k_total = 2k)
                fit = subset_exemplars(cx.ex_fit, cx.y, k, seed + k)
                chk = subset_exemplars(cx.ex_chk, cx.y, k, seed + 50 + k)
                ev(f"Sweep-k{2 * k}[{b}]", guard(tr, th[b], cx.X, cx.y, fit, chk, A, seed)[0])
                ev(f"SweepB8-k{2 * k}[{b}]", U.finetune_exemplars(tr, th[b], cx.X, cx.y, fit, seed))
            for rho in (0.5, 0.7, 0.9):
                ev(f"Sweep-rho{rho}[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, rho=rho)[0])
            for gm in (1e-3, 1e-1):
                ev(f"Sweep-gamma{gm}[{b}]", guard(tr, th[b], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed, gamma=gm)[0])
            # exemplar label noise: flip a share of exemplar labels in the fit half
            for noise in (0.1, 0.2):
                y_noisy = cx.y.clone()
                g = torch.Generator().manual_seed(seed + int(noise * 100))
                fit = cx.ex_fit
                nflip = int(noise * len(fit))
                sel = fit[torch.randperm(len(fit), generator=g)[:nflip].to(fit.device)]
                y_noisy[sel] = torch.randint(0, tr.C, (nflip,), generator=g).to(fit.device)
                ev(f"Sweep-noise{noise}[{b}]", guard(tr, th[b], cx.X, y_noisy, cx.ex_fit, cx.ex_chk, A, seed)[0])
        rec = {"scenario": scen, "seed": seed, "flags": flag, "results": res, "guard_info": guard_info,
               "seconds": time.time() - t0}
        arrays = {f"pred_{k}": v for k, v in preds.items()}
        if seed == 0:   # penultimate embeddings of a stratified test sample for the failure analysis
            from .model import embeddings
            rng = np.random.default_rng(0)
            yte = cx.ev["yte_np"]
            samp = np.concatenate([rng.choice(np.flatnonzero(yte == c), min(300, (yte == c).sum()), replace=False)
                                   for c in np.unique(yte)])
            Xs = cx.ev["Xte"][torch.tensor(samp, device=device)]
            arrays["emb_idx"] = samp
            g_b1, _ = guard(tr, th["B1"], cx.X, cx.y, cx.ex_fit, cx.ex_chk, A, seed)
            for key, theta in (("B0", th["B0"]), ("B1", th["B1"]), ("Guard_B1", g_b1)):
                arrays[f"emb_{key}"] = embeddings(tr.as_model(theta), Xs).cpu().numpy().astype(np.float16)
        _save(out, name, rec, arrays)
        log(f"{name}: {len(res)} models in {rec['seconds']:.0f}s | flagged={A} | " +
            " ".join(f"{k}:DD={v['dd']['rho0.8']['dd_count']}" for k, v in res.items() if k.startswith(("Guard[", "B8[", "B6"))))
