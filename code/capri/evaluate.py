"""Evaluation of one model: capability on the global test set and forgetting of client u."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import per_family, detection_debt, mia_auc
from .model import predict, predict_logits


@torch.no_grad()
def per_sample_loss(model, X, y, batch=65536):
    out = []
    model.eval()
    for i in range(0, len(X), batch):
        out.append(F.cross_entropy(model(X[i:i + batch]), y[i:i + batch], reduction="none"))
    return torch.cat(out)


def relearn_steps(tr, theta, X, y, mem_idx, target_acc, seed, max_steps=300, lr=1e-3, batch=64, check_every=10):
    """Fine-tuning steps on client u's data needed to regain the pre-deletion accuracy
    on its records. Fast relearning signals residual knowledge."""
    model = tr.as_model(theta)
    Xm, ym = X[mem_idx], y[mem_idx]
    acc = float((predict(model, Xm) == ym).float().mean())
    if acc >= target_acc:
        return 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 909)
    step = 0
    while step < max_steps:
        model.train()
        perm = torch.randperm(len(Xm), generator=g).to(Xm.device)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(model(Xm[b]), ym[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % check_every == 0:
                if float((predict(model, Xm) == ym).float().mean()) >= target_acc:
                    return step
            if step >= max_steps:
                break
    return max_steps + 1      # did not relearn within the budget (right-censored)


def evaluate_model(tr, theta, ctx, families, rho=0.8, relearn_target=None, seed=0) -> dict:
    """ctx holds device tensors: Xte, yte, X, y, mem (member idx into X), non (non-member idx into Xte)."""
    model = tr.as_model(theta)
    pred = predict(model, ctx["Xte"]).cpu().numpy()
    yte = ctx["yte_np"]
    pf = per_family(yte, pred, tr.C)
    dd = {f"rho{r}": detection_debt(pf["recall"], families, r) for r in (0.5, 0.8, 0.9)}
    lm = per_sample_loss(model, ctx["X"][ctx["mem"]], ctx["y"][ctx["mem"]]).cpu().numpy()
    ln = per_sample_loss(model, ctx["Xte"][ctx["non"]], ctx["yte"][ctx["non"]]).cpu().numpy()
    mia = mia_auc(lm, ln)
    forget_pred = predict(model, ctx["X"][ctx["forget_all"]]).cpu().numpy()
    forget_acc = float((forget_pred == ctx["y_forget_np"]).mean())
    out = {
        "accuracy": pf["accuracy"], "macro_f1": pf["macro_f1"], "balanced_acc": pf["balanced_acc"],
        "benign_far": pf["benign_far"], "attack_detection_rate": pf["attack_detection_rate"],
        "recall": pf["recall"].tolist(), "precision": pf["precision"].tolist(), "f1": pf["f1"].tolist(),
        "silent_miss": pf["silent_miss"].tolist(), "dd": dd,
        "forget_acc": forget_acc, **mia,
    }
    if relearn_target is not None:
        out["relearn_steps"] = relearn_steps(tr, theta, ctx["X"], ctx["y"], ctx["mem"], relearn_target, seed)
    return out, pred.astype(np.int8), pf["cm"]
