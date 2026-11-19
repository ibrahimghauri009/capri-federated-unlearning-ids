"""Evaluation metrics (Deliverable 3, Step 8)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def confusion(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    return np.bincount(y_true * C + y_pred, minlength=C * C).reshape(C, C)


def per_family(y_true, y_pred, C, benign=0) -> dict:
    cm = confusion(np.asarray(y_true), np.asarray(y_pred), C)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    tp = np.diag(cm)
    recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
    precision = np.where(pred_n > 0, tp / np.maximum(pred_n, 1), np.nan)
    f1 = np.where(support > 0, 2 * tp / np.maximum(support + pred_n, 1), np.nan)
    silent = np.where(support > 0, cm[:, benign] / np.maximum(support, 1), np.nan)
    silent[benign] = np.nan
    present = support > 0
    attacks = present.copy()
    attacks[benign] = False
    return {
        "recall": recall, "precision": precision, "f1": f1, "support": support,
        "silent_miss": silent,
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(np.nanmean(f1[present])),
        "balanced_acc": float(np.nanmean(recall[present])),
        "benign_far": float(1 - recall[benign]) if support[benign] > 0 else float("nan"),
        "attack_detection_rate": float(cm[attacks][:, [c for c in range(C) if c != benign]].sum()
                                       / max(cm[attacks].sum(), 1)),
        "cm": cm,
    }


def detection_debt(recall: np.ndarray, families: np.ndarray, rho: float = 0.8) -> dict:
    r = recall[families]
    return {"dd_count": int((r < rho).sum()),
            "dd_deficit": float(np.clip(rho - r, 0, None).sum()),
            "worst_family_recall": float(np.nanmin(r)) if len(r) else float("nan")}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mia_auc(loss_members: np.ndarray, loss_nonmembers: np.ndarray) -> dict:
    """Loss-threshold membership inference: lower loss -> 'member'."""
    s = np.concatenate([-loss_members, -loss_nonmembers])
    t = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    auc = roc_auc_score(t, s)
    fpr, tpr, _ = roc_curve(t, s)
    tpr1 = float(np.interp(0.01, fpr, tpr))
    return {"mia_auc": float(auc), "mia_tpr_at_1fpr": tpr1}


def class_matched_nonmembers(y_members: np.ndarray, y_pool: np.ndarray, rng, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw members and non-members with the same class mix, so the attack measures
    membership rather than class differences. Returns (member_idx, nonmember_idx)."""
    mem, non = [], []
    for c in np.unique(y_members):
        a = np.flatnonzero(y_members == c)
        b = np.flatnonzero(y_pool == c)
        n = min(len(a), len(b))
        if n == 0:
            continue
        mem.append(rng.choice(a, n, replace=False))
        non.append(rng.choice(b, n, replace=False))
    mem, non = np.concatenate(mem), np.concatenate(non)
    if len(mem) > max_n:
        sel = rng.choice(len(mem), max_n, replace=False)
        mem, non = mem[sel], non[sel]
    return mem, non
