"""Baseline unlearning methods (Deliverable 3, Step 6).

B1 retrain oracle      FedAvg from the same initialisation without client u.
B2 lawful oracle       as B1, plus the server's E_fit as an extra participant.
B3 continued training  remaining clients keep training from theta_T.
B4 gradient ascent     client u runs projected gradient ascent on its data inside an
                       L2 ball around the remaining clients' reference model
                       (Halimi et al., 2022), then remaining clients restore utility.
B5 FedEraser           stored updates calibrated by short local training of the
                       remaining clients (Liu et al., 2021).
B8 exemplar fine-tune  plain fine-tuning of a base model on E_fit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .fl import FedTrainer, FLResult
from .model import MLP, get_flat, set_flat


def retrain(tr: FedTrainer, theta0, X, y, clients, u, seed, Xval, yval) -> FLResult:
    rest = [c for k, c in enumerate(clients) if k != u]
    return tr.train(theta0, X, y, rest, seed=seed + 101, Xval=Xval, yval=yval, keep_history=False)


def lawful_oracle(tr: FedTrainer, theta0, X, y, clients, u, ex_fit_idx, seed, Xval, yval) -> FLResult:
    rest = [c for k, c in enumerate(clients) if k != u]
    n = torch.tensor([len(c) for c in rest], dtype=torch.float32)
    # the exemplar participant gets the weight of an average remaining client
    w = torch.cat([n / n.sum() * (len(rest) / (len(rest) + 1)), torch.tensor([1.0 / (len(rest) + 1)])])
    return tr.train(theta0, X, y, rest + [ex_fit_idx], seed=seed + 202, agg_weights=w,
                    Xval=Xval, yval=yval, keep_history=False)


def continued(tr: FedTrainer, thetaT, X, y, clients, u, seed, rounds=5) -> torch.Tensor:
    rest = [c for k, c in enumerate(clients) if k != u]
    return tr.train(thetaT, X, y, rest, seed=seed + 303, rounds=rounds, early_stop=False,
                    keep_history=False).theta


def gradient_ascent(tr: FedTrainer, pre: FLResult, X, y, clients, u, seed, restore_rounds=5,
                    lr=0.01, max_epochs=1, seed_rand=10) -> tuple[torch.Tensor, dict]:
    dev = tr.device
    w = pre.agg_weights.to(dev)
    # reference: global model after the last round, built without client u
    th_last = pre.thetas[-1].to(dev)
    U_last = pre.updates[-1].to(dev)
    keep = torch.ones(len(w), dtype=torch.bool, device=dev)
    keep[u] = False
    w_ref = th_last + (w[keep].unsqueeze(1) * U_last[keep]).sum(0) / w[keep].sum()
    # radius: one third of the mean distance from random models to the reference
    d = [torch.norm(tr.init_theta(seed + 9000 + i) - w_ref) for i in range(seed_rand)]
    delta = float(torch.stack(d).mean()) / 3
    model = tr.as_model(pre.theta)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    idx = clients[u]
    g = torch.Generator().manual_seed(seed + 404)
    steps = 0
    for _ in range(max_epochs):
        perm = idx[torch.randperm(len(idx), generator=g).to(dev)]
        for i in range(0, len(perm), tr.cfg.batch_size):
            b = perm[i:i + tr.cfg.batch_size]
            loss = -F.cross_entropy(model(X[b]), y[b], weight=tr.cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            with torch.no_grad():   # project onto the L2 ball around w_ref
                th = get_flat(model)
                diff = th - w_ref
                nrm = torch.norm(diff)
                if nrm > delta:
                    set_flat(model, w_ref + diff * (delta / nrm))
    theta_ga = get_flat(model)
    theta = continued(tr, theta_ga, X, y, clients, u, seed + 505, rounds=restore_rounds)
    return theta, {"ga_steps": steps, "delta": delta}


def federaser(tr: FedTrainer, pre: FLResult, X, y, clients, u, seed, interval=2, calib_epochs=1) -> torch.Tensor:
    dev = tr.device
    rest = [c for k, c in enumerate(clients) if k != u]
    keep = [k for k in range(len(clients)) if k != u]
    n = torch.tensor([len(c) for c in rest], dtype=torch.float32, device=dev)
    w = n / n.sum()
    gen = torch.Generator().manual_seed(seed + 606)
    theta = pre.thetas[0].to(dev).clone()
    for t in range(0, len(pre.updates), interval):
        U_old = pre.updates[t][keep].to(dev)                          # [K-1, P]
        Pk = tr.local_train(theta, X, y, rest, calib_epochs, gen)
        dU = Pk - theta.unsqueeze(0)
        cal = U_old.norm(dim=1, keepdim=True) * dU / dU.norm(dim=1, keepdim=True).clamp_min(1e-12)
        theta = theta + (w.unsqueeze(1) * cal).sum(0)
    return theta


def finetune_exemplars(tr: FedTrainer, theta_u, X, y, fit_idx, seed, epochs=20, lr=1e-3, batch=64) -> torch.Tensor:
    model = tr.as_model(theta_u)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 707)
    for _ in range(epochs):
        perm = fit_idx[torch.randperm(len(fit_idx), generator=g).to(fit_idx.device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(model(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return get_flat(model)
