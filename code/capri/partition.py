"""Ownership scenarios (Deliverable 3, Step 7).

S0   IID: random equal split of all records.
S1   shared: target families split equally over all K clients (client 0 ~ 1/K).
S2x  dominant: client 0 holds a share s in {0.5, 0.8, 0.95} of each target family;
     the remainder is split equally over the other K-1 clients (held by >= 3 clients).
S3   sole holder: client 0 holds 100% of each target family (m_c = 0).
In S1-S3 every non-target class (benign included) follows a Dirichlet(alpha) label
skew, the standard non-IID setting of prior work.
"""
from __future__ import annotations

import numpy as np

SCENARIOS = {"S0": None, "S1": "equal", "S2a": 0.5, "S2b": 0.8, "S2c": 0.95, "S3": 1.0}


def partition(y: np.ndarray, K: int, scenario: str, targets: list[int], seed: int,
              alpha: float = 0.5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 7919)
    parts = [[] for _ in range(K)]
    if scenario == "S0":
        idx = rng.permutation(len(y))
        for k, chunk in enumerate(np.array_split(idx, K)):
            parts[k] = list(chunk)
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    share = SCENARIOS[scenario]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n = len(idx)
        if c in targets:
            if share == "equal":
                cuts = np.array_split(idx, K)
            else:
                n0 = int(round(share * n))
                rest = np.array_split(idx[n0:], K - 1)
                cuts = [idx[:n0]] + rest
        else:
            p = rng.dirichlet(alpha * np.ones(K))
            bounds = (np.cumsum(p) * n).round().astype(int)[:-1]
            cuts = np.split(idx, bounds)
        for k in range(K):
            parts[k].extend(cuts[k].tolist())
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


def ownership(y: np.ndarray, parts: list[np.ndarray], n_classes: int, u: int = 0):
    """n[k, c] label counts; s_c = n[u,c]/sum_k n[k,c]; m_c = sum_{k != u} n[k,c]."""
    n = np.stack([np.bincount(y[p], minlength=n_classes) for p in parts])
    tot = n.sum(0)
    s = np.where(tot > 0, n[u] / np.maximum(tot, 1), 0.0)
    m = tot - n[u]
    return n, s, m
