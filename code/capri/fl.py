"""Federated training core (FedAvg, McMahan et al., 2017).

All K clients train in parallel on one GPU: client parameters are stacked
along a leading dimension and a single vmapped gradient call computes every
client's local step at once. This is a pure speed device: each client still
runs its own Adam optimiser on its own data, exactly as in sequential FedAvg.
Clients with fewer local steps are masked once they finish.

Per-round client updates U_{k,t} = theta_{k,t}^{local} - theta_t are kept
(needed by history-based unlearning and by CAPRI-Audit's leave-u-out probe).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .model import MLP, make_model, get_flat, set_flat, predict


@dataclass
class FLConfig:
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 10         # early stop on validation macro-F1
    min_rounds: int = 20
    # FedRS restricted softmax (Li & Zhan, KDD 2021): during local training, logits of
    # classes a client does not hold are scaled by rs_alpha (0 = fully restricted,
    # 1 = standard softmax). Prevents clients from suppressing classes they never saw.
    rs_alpha: float = 0.0
    weight_decay: float = 0.0
    arch: str = "mlp"           # "mlp" (default) or "cnn" (sensitivity analysis)
    participation: float = 1.0  # fraction of clients sampled per round (extension experiment)


@dataclass
class FLResult:
    theta: torch.Tensor                      # final global parameters (flat)
    updates: list = field(default_factory=list)   # per round: tensor [K, P] (CPU)
    thetas: list = field(default_factory=list)    # per round: global params before the round (CPU)
    agg_weights: torch.Tensor | None = None   # [K]
    best_round: int = 0
    val_curve: list = field(default_factory=list)
    rounds_run: int = 0
    round_weights: list = field(default_factory=list)   # per round: aggregation weights [K] (partial participation)


def macro_f1_torch(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> float:
    tp = torch.zeros(n_classes, device=y_true.device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tp.index_add_(0, y_true[y_true == y_pred], torch.ones((y_true == y_pred).sum(), device=y_true.device))
    wrong = y_true != y_pred
    fp.index_add_(0, y_pred[wrong], torch.ones(wrong.sum(), device=y_true.device))
    fn.index_add_(0, y_true[wrong], torch.ones(wrong.sum(), device=y_true.device))
    present = (tp + fn) > 0
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)
    return float(f1[present].mean())


class FedTrainer:
    """Runs FedAvg over index sets into a shared (X, y) tensor on one device."""

    def __init__(self, d_in: int, n_classes: int, class_weights: torch.Tensor, device: str,
                 cfg: FLConfig = FLConfig()):
        self.d_in, self.C, self.device, self.cfg = d_in, n_classes, device, cfg
        self.model = make_model(cfg.arch, d_in, n_classes).to(device)
        self.names = [n for n, _ in self.model.named_parameters()]
        self.shapes = [p.shape for _, p in self.model.named_parameters()]
        self.sizes = [p.numel() for _, p in self.model.named_parameters()]
        self.P = sum(self.sizes)
        self.cw = class_weights.to(device)

        def loss_fn(params, x, y, m, scale):
            logits = functional_call(self.model, params, (x,))
            logits = logits * scale
            l = F.cross_entropy(logits, y, reduction="none")
            w = self.cw[y] * m
            return (l * w).sum() / w.sum().clamp_min(1e-12)

        self._vgrad = vmap(grad(loss_fn), in_dims=(0, 0, 0, 0, 0), randomness="different")

    def init_theta(self, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        m = make_model(self.cfg.arch, self.d_in, self.C).to(self.device)
        return get_flat(m)

    def _unflat(self, Pk: torch.Tensor) -> dict:
        parts = torch.split(Pk, self.sizes, dim=1)
        return {n: t.reshape(Pk.shape[0], *s) for n, t, s in zip(self.names, parts, self.shapes)}

    def local_train(self, theta: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
                    clients: list[torch.Tensor], epochs: int, gen: torch.Generator,
                    lr: float | None = None) -> torch.Tensor:
        """Every client starts from theta and runs `epochs` local epochs of Adam.
        Returns the stacked local parameters [K, P]."""
        cfg = self.cfg
        lr = cfg.lr if lr is None else lr
        K, B, dev = len(clients), cfg.batch_size, self.device
        n = torch.tensor([len(c) for c in clients], device=dev)
        steps = torch.ceil(n.float() * epochs / B).long()
        T = int(steps.max())
        # index matrix [K, T*B]; -1 = padding
        I = torch.full((K, T * B), -1, dtype=torch.long, device=dev)
        for k, c in enumerate(clients):
            if len(c) == 0:
                continue
            seq = torch.cat([c[torch.randperm(len(c), generator=gen, device="cpu").to(dev)] for _ in range(epochs)])
            I[k, :len(seq)] = seq
        I = I.view(K, T, B)
        # per-client logit scale: 1 for locally held classes, rs_alpha for absent ones
        scale = torch.ones(K, self.C, device=dev)
        if cfg.rs_alpha != 1.0:
            for k, c in enumerate(clients):
                present = torch.zeros(self.C, dtype=torch.bool, device=dev)
                if len(c):
                    present[torch.unique(y[c])] = True
                scale[k, ~present] = cfg.rs_alpha
        scale = scale.unsqueeze(1)                       # [K, 1, C], broadcast over the batch
        Pk = theta.unsqueeze(0).repeat(K, 1).contiguous()
        m1 = torch.zeros_like(Pk)
        m2 = torch.zeros_like(Pk)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.model.train()
        for t in range(T):
            idx = I[:, t]                      # [K, B]
            mask = (idx >= 0).float()
            active = (mask.sum(1) > 0).float().unsqueeze(1)   # [K, 1]
            if active.sum() == 0:
                break
            safe = idx.clamp_min(0)
            xb, yb = X[safe], y[safe]
            g = self._vgrad(self._unflat(Pk), xb, yb, mask, scale)
            G = torch.cat([g[nm].reshape(K, -1) for nm in self.names], dim=1)
            if cfg.weight_decay:
                G = G + cfg.weight_decay * Pk
            m1 = torch.where(active.bool(), b1 * m1 + (1 - b1) * G, m1)
            m2 = torch.where(active.bool(), b2 * m2 + (1 - b2) * G * G, m2)
            tk = torch.clamp(torch.minimum(torch.full_like(steps, t + 1), steps), min=1).float().unsqueeze(1)
            mh = m1 / (1 - b1 ** tk)
            vh = m2 / (1 - b2 ** tk)
            Pk = Pk - lr * active * mh / (vh.sqrt() + eps)
        return Pk

    def train(self, theta0: torch.Tensor, X: torch.Tensor, y: torch.Tensor, clients: list[torch.Tensor],
              seed: int, rounds: int | None = None, epochs: int | None = None,
              agg_weights: torch.Tensor | None = None, Xval=None, yval=None,
              early_stop: bool = True, keep_history: bool = True, lr: float | None = None) -> FLResult:
        cfg = self.cfg
        rounds = cfg.rounds if rounds is None else rounds
        epochs = cfg.local_epochs if epochs is None else epochs
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        n = torch.tensor([len(c) for c in clients], dtype=torch.float32, device=self.device)
        w = (n / n.sum()) if agg_weights is None else agg_weights.to(self.device) / agg_weights.sum()
        theta = theta0.clone()
        res = FLResult(theta=theta, agg_weights=w.cpu())
        best, best_theta, best_r, bad = -1.0, theta.clone(), 0, 0
        K = len(clients)
        prng = np.random.default_rng(seed + 9973)
        for r in range(rounds):
            if cfg.participation < 1.0:
                # sample a fraction of the clients that hold data; others contribute nothing this round
                live = [k for k in range(K) if len(clients[k]) > 0]
                nsel = max(1, int(math.ceil(cfg.participation * len(live))))
                sel = sorted(prng.choice(live, size=nsel, replace=False).tolist())
                sub = [clients[k] if k in sel else clients[k][:0] for k in range(K)]
                Pk = self.local_train(theta, X, y, sub, epochs, gen, lr=lr)
                wr = torch.zeros_like(w)
                wr[sel] = w[sel] / w[sel].sum()
            else:
                Pk = self.local_train(theta, X, y, clients, epochs, gen, lr=lr)
                wr = w
            U = Pk - theta.unsqueeze(0)
            if keep_history:
                res.updates.append(U.cpu())
                res.thetas.append(theta.cpu())
                res.round_weights.append(wr.cpu())
            theta = theta + (wr.unsqueeze(1) * U).sum(0)
            res.rounds_run = r + 1
            if Xval is not None and early_stop:
                set_flat(self.model, theta)
                f1 = macro_f1_torch(yval, predict(self.model, Xval), self.C)
                res.val_curve.append(f1)
                if f1 > best + 1e-4:
                    best, best_theta, best_r, bad = f1, theta.clone(), r + 1, 0
                else:
                    bad += 1
                if bad >= cfg.patience and r + 1 >= cfg.min_rounds:
                    break
        if Xval is not None and early_stop:
            theta = best_theta
            res.best_round = best_r
            if keep_history:   # history must match the trajectory that produced theta
                res.updates = res.updates[:best_r]
                res.thetas = res.thetas[:best_r]
                res.round_weights = res.round_weights[:best_r]
        else:
            res.best_round = res.rounds_run
        res.theta = theta
        return res

    def as_model(self, theta: torch.Tensor):
        m = make_model(self.cfg.arch, self.d_in, self.C).to(self.device)
        set_flat(m, theta)
        return m
