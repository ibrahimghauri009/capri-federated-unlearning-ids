"""CAPRI: CApability-PReserving, rIsk-informed federated unlearning (Deliverable 3, Step 5).

CAPRI-Audit (before the deletion is approved)
  signals per attack family c:
    ownership    s_c, log(1 + m_c)              from per-client label counts
    contribution recall of a leave-u-out estimate theta_est on E_chk[c]
    separability prototype margin of family c under theta_est on E_chk
  theta_est replaces every stored round aggregate by the aggregate of the remaining
  clients (renormalised weights), i.e. it subtracts client u's weighted contributions
  from theta_T without any client interaction.
  A small logistic (binomial GLM) model maps the signals to predicted post-deletion
  recall; labels: REQUIRED (m_c = 0), COLLATERAL (m_c > 0 and r_hat < rho), SAFE.

CAPRI-Guard (after any base unlearner U)
  minimise  KL(theta_u || theta) on E_fit of non-flagged classes
          + gamma * ||theta - theta_u||^2
          + sum_{c in A} lambda_c * CE(theta, E_fit[c])
  lambda_c <- clip(lambda_c + eta * (rho - recall(theta, E_chk[c])), 0, lambda_max)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .fl import FedTrainer, FLResult
from .model import MLP, get_flat, set_flat, embeddings, predict

REQUIRED, COLLATERAL, SAFE = "Required", "Collateral", "Safe"


# ----------------------------------------------------------------------------
# CAPRI-Audit
# ----------------------------------------------------------------------------
def leave_u_out_estimate(pre: FLResult, u: int) -> torch.Tensor:
    w = pre.agg_weights
    keep = torch.ones(len(w), dtype=torch.bool)
    keep[u] = False
    theta = pre.theta.cpu().clone()
    rws = pre.round_weights if pre.round_weights else [w] * len(pre.updates)
    for U, wr in zip(pre.updates, rws):
        full = (wr.unsqueeze(1) * U).sum(0)
        if wr[keep].sum() <= 0:          # only client u took part in this round: drop the round
            theta -= full
            continue
        loo = (wr[keep].unsqueeze(1) * U[keep]).sum(0) / wr[keep].sum()
        theta += loo - full
    return theta


def prototype_margin(Z: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    protos = {c: Z[y == c].mean(0) for c in classes if (y == c).any()}
    out = {}
    for c, p in protos.items():
        spread = np.linalg.norm(Z[y == c] - p, axis=1).mean() + 1e-6
        others = [np.linalg.norm(p - q) for d, q in protos.items() if d != c]
        out[c] = float(min(others) / spread) if others else float("nan")
    return out


def audit_signals(tr: FedTrainer, theta_est: torch.Tensor, Xchk: torch.Tensor, ychk: np.ndarray,
                  s: np.ndarray, m: np.ndarray, families: list[int]) -> list[dict]:
    model = tr.as_model(theta_est.to(tr.device))
    pred = predict(model, Xchk).cpu().numpy()
    Z = embeddings(model, Xchk).cpu().numpy()
    margins = prototype_margin(Z, ychk, np.unique(ychk))
    rows = []
    for c in families:
        msk = ychk == c
        rows.append({"family": int(c), "s": float(s[c]), "m": int(m[c]),
                     "log_m": float(np.log1p(m[c])),
                     "r_probe": float((pred[msk] == c).mean()) if msk.any() else float("nan"),
                     "margin": margins.get(c, float("nan")), "n_chk": int(msk.sum())})
    return rows


SIGNAL_SETS = {
    "ownership": ["s", "log_m"],
    "ownership+contribution": ["s", "log_m", "r_probe"],
    "all": ["s", "log_m", "r_probe", "margin"],
}


def fit_audit(df_train, signals: list[str], target: str = "r_oracle"):
    """Binomial GLM (logit link) on recall proportions."""
    import statsmodels.api as sm
    Xd = sm.add_constant(df_train[signals].to_numpy(float), has_constant="add")
    yv = df_train[target].clip(0, 1).to_numpy(float)
    model = sm.GLM(yv, Xd, family=sm.families.Binomial())
    try:
        res = model.fit()
    except Exception:
        res = model.fit_regularized(alpha=1e-3, L1_wt=0.0)
    return res


def predict_audit(res, df, signals):
    import statsmodels.api as sm
    Xd = sm.add_constant(df[signals].to_numpy(float), has_constant="add")
    return np.clip(res.predict(Xd), 0, 1)


def audit_labels(r_hat: np.ndarray, m: np.ndarray, rho: float) -> np.ndarray:
    return np.where(m == 0, REQUIRED, np.where(r_hat < rho, COLLATERAL, SAFE))


def rule_audit(s: np.ndarray, m: np.ndarray, m_min: int = 50, s_thr: float = 0.5) -> np.ndarray:
    """Counts-only baseline audit: flag if few samples remain or the leaver dominated."""
    return (m < m_min) | (s > s_thr)


# ----------------------------------------------------------------------------
# CAPRI-Guard
# ----------------------------------------------------------------------------
def recall_on(model, X, y, c) -> float:
    msk = y == c
    if not bool(msk.any()):
        return float("nan")
    return float((predict(model, X[msk]) == c).float().mean())


def guard(tr: FedTrainer, theta_u: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
          fit_idx: torch.Tensor, chk_idx: torch.Tensor, flagged: list[int], seed: int,
          rho: float = 0.8, gamma: float = 1e-2, eta: float = 1.0, lam_max: float = 10.0,
          epochs: int = 20, lr: float = 1e-3, batch: int = 64,
          dual_ascent: bool = True, anchor: bool = True, proximity: bool = True) -> tuple[torch.Tensor, dict]:
    dev = tr.device
    teacher = tr.as_model(theta_u)
    teacher.eval()
    student = tr.as_model(theta_u)
    th_u = get_flat(teacher).detach()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    lam = torch.zeros(tr.C, device=dev)
    A = torch.zeros(tr.C, dtype=torch.bool, device=dev)
    if flagged:
        A[torch.tensor(flagged, device=dev)] = True
        lam[A] = 1.0
    Xc, yc = X[chk_idx], y[chk_idx]
    g = torch.Generator().manual_seed(seed + 808)
    trace = []
    for ep in range(epochs):
        student.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=g).to(dev)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb, yb = X[b], y[b]
            logits = student(xb)
            loss = torch.zeros((), device=dev)
            inA = A[yb]
            if anchor and (~inA).any():
                with torch.no_grad():
                    pt = F.softmax(teacher(xb[~inA]), 1)
                loss = loss + F.kl_div(F.log_softmax(logits[~inA], 1), pt, reduction="batchmean")
            if inA.any():
                ce = F.cross_entropy(logits[inA], yb[inA], reduction="none")
                loss = loss + (lam[yb[inA]] * ce).mean()
            if proximity and gamma > 0:
                params = torch.cat([p.reshape(-1) for p in student.parameters()])
                loss = loss + gamma * ((params - th_u) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec = {int(c): recall_on(student, Xc, yc, int(c)) for c in flagged}
        if dual_ascent:
            for c, r in rec.items():
                if not np.isnan(r):
                    lam[c] = float(np.clip(float(lam[c]) + eta * (rho - r), 0.0, lam_max))
        trace.append({"epoch": ep, "recall_chk": rec, "lambda": {int(c): float(lam[c]) for c in flagged}})
    return get_flat(student), {"trace": trace}
