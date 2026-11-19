"""Statistics, tables and figures (Deliverable 3, Steps 9-11).

Every number in the report is produced here from the logged raw outputs.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCEN_ORDER = ["S0", "S1", "S2a", "S2b", "S2c", "S3"]
SCEN_SHARE = {"S0": 0.1, "S1": 0.1, "S2a": 0.5, "S2b": 0.8, "S2c": 0.95, "S3": 1.0}
RHO = 0.8


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load(res_dir: Path):
    p1, p2 = {}, {}
    for f in sorted(res_dir.glob("p1_*_s*.json")):
        if f.stem.endswith("_audit"):
            continue
        r = json.load(open(f))
        p1[(r["scenario"], r["seed"])] = r
    for f in sorted(res_dir.glob("p2_*_s*.json")):
        r = json.load(open(f))
        p2[(r["scenario"], r["seed"])] = r
    return p1, p2


def long_table(p1, p2) -> pd.DataFrame:
    """One row per (scenario, seed, method)."""
    rows = []
    for key, r in p1.items():
        scen, seed = key
        targets = r["targets"]["targets"]          # name -> class id
        fam = np.array(r["families_evaluable"])
        m = np.array(r["m"])
        res = dict(r["results"])
        if key in p2:
            res.update(p2[key]["results"])
        rb0 = np.array(res["B0"]["recall"], dtype=float)
        rb1 = np.array(res["B1"]["recall"], dtype=float)
        n0 = np.array(r["n"][0], dtype=float)       # departing client's records per class
        for meth, v in res.items():
            rec = np.array(v["recall"], dtype=float)
            sil = np.array(v["silent_miss"], dtype=float)
            shared = [c for c in fam if m[c] > 0]
            row = {"scenario": scen, "seed": seed, "method": meth,
                   "accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "balanced_acc": v["balanced_acc"],
                   "benign_far": v["benign_far"],
                   "dd": v["dd"]["rho0.8"]["dd_count"], "dd_deficit": v["dd"]["rho0.8"]["dd_deficit"],
                   "dd05": v["dd"]["rho0.5"]["dd_count"], "dd09": v["dd"]["rho0.9"]["dd_count"],
                   "wfr": v["dd"]["rho0.8"]["worst_family_recall"],
                   "col_vs_pre": float(np.nanmean(np.clip(rb0[shared] - rec[shared], 0, None))) if shared else np.nan,
                   "col_vs_oracle": float(np.nanmean(np.clip(rb1[shared] - rec[shared], 0, None))) if shared else np.nan,
                   "mia_auc": v["mia_auc"], "mia_tpr1": v["mia_tpr_at_1fpr"], "forget_acc": v["forget_acc"],
                   "relearn": v.get("relearn_steps", np.nan)}
            # class-matched generalisation accuracy: the accuracy the model would have on the departing
            # client's records if it treated them like unseen test records of the same classes.
            ok = ~np.isnan(rec) & (n0 > 0)
            row["cm_test_acc"] = float((n0[ok] * rec[ok]).sum() / n0[ok].sum())
            row["forget_gap"] = v["forget_acc"] - row["cm_test_acc"]
            for tname, c in targets.items():
                row[f"recall_{tname}"] = rec[c]
                row[f"silent_{tname}"] = sil[c]
                row[f"pre_{tname}"] = rb0[c]
            rows.append(row)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, methods, cols, scenarios=SCEN_ORDER) -> pd.DataFrame:
    d = df[df.method.isin(methods) & df.scenario.isin(scenarios)]
    g = d.groupby(["scenario", "method"])[cols]
    s = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    s = s.reset_index()
    s["scenario"] = pd.Categorical(s.scenario, scenarios, ordered=True)
    s["method"] = pd.Categorical(s.method, methods, ordered=True)
    return s.sort_values(["scenario", "method"])


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon(x, y, alternative="two-sided"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.allclose(x, y):
        return {"n": int(len(x)), "W": np.nan, "p": 1.0, "rbc": 0.0, "mean_diff": float(np.mean(x - y)) if len(x) else np.nan}
    w = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return {"n": int(len(x)), "W": float(w.statistic), "p": float(w.pvalue), "rbc": rank_biserial(x, y),
            "mean_diff": float(np.mean(x - y))}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def tost_paired(x, y, margin):
    """Two one-sided paired t-tests: equivalence if both p < 0.05."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "p": np.nan}
    se = d.std(ddof=1) / np.sqrt(n) + 1e-12
    t1 = (d.mean() + margin) / se
    t2 = (d.mean() - margin) / se
    p1 = 1 - stats.t.cdf(t1, n - 1)
    p2 = stats.t.cdf(t2, n - 1)
    ci = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=se)
    return {"n": n, "mean_diff": float(d.mean()), "p": float(max(p1, p2)), "ci90": [float(ci[0]), float(ci[1])],
            "equivalent": bool(max(p1, p2) < 0.05)}


def friedman_nemenyi(df: pd.DataFrame, methods, value: str, block_cols=("scenario", "seed")):
    piv = df[df.method.isin(methods)].pivot_table(index=list(block_cols), columns="method", values=value)
    piv = piv[methods].dropna()
    stat, p = stats.friedmanchisquare(*[piv[m] for m in methods])
    ranks = piv.rank(axis=1, method="average").mean()       # lower value = better = lower rank
    k, n = len(methods), len(piv)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
         11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}[k]
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": float(stat), "p": float(p), "n_blocks": n, "ranks": ranks.to_dict(), "cd": float(cd)}


def bootstrap_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


# ----------------------------------------------------------------------------
# audit (RQ2)
# ----------------------------------------------------------------------------
def prf(flag, truth):
    flag, truth = np.asarray(flag, bool), np.asarray(truth, bool)
    tp = (flag & truth).sum(); fp = (flag & ~truth).sum(); fn = (~flag & truth).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "n": int(len(flag)), "n_at_risk": int(truth.sum())}


def audit_report(at: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> dict:
    d = at[at.r_pre >= RHO].copy()          # families the detector could catch before deletion
    out = {}
    policies = {
        "CAPRI-Audit (all signals)": d["label_all"].isin(["Required", "Collateral"]),
        "Audit: ownership + contribution": d["label_ownership+contribution"].isin(["Required", "Collateral"]),
        "Audit: ownership only": d["label_ownership"].isin(["Required", "Collateral"]),
        "Rule-only (counts)": d["flag_rule"].astype(bool),
        "No warning": pd.Series(False, index=d.index),
    }
    for subset, msk in (("all families", np.ones(len(d), bool)), ("m_c > 0 (non-trivial)", (d.m > 0).to_numpy())):
        out[subset] = {name: prf(f.to_numpy()[msk], d.at_risk.to_numpy()[msk]) for name, f in policies.items()}
    # prediction quality of the learned audit
    for name in ("all", "ownership+contribution", "ownership"):
        e = d.dropna(subset=[f"rhat_{name}"])
        out[f"mae_{name}"] = float(np.abs(e[f"rhat_{name}"] - e.r_oracle).mean())
        loss_pred = e.r_pre - e[f"rhat_{name}"]
        loss_true = e.r_pre - e.r_oracle
        out[f"spearman_{name}"] = float(stats.spearmanr(loss_pred, loss_true).correlation)
        e2 = e[e.m > 0]
        out[f"mae_{name}_mc>0"] = float(np.abs(e2[f"rhat_{name}"] - e2.r_oracle).mean())
        out[f"spearman_{name}_mc>0"] = float(stats.spearmanr(e2.r_pre - e2[f"rhat_{name}"], e2.r_pre - e2.r_oracle).correlation)
    # paired permutation test: CAPRI-Audit F1 vs rule-only F1 on the non-trivial subset
    rng = np.random.default_rng(seed)
    msk = (d.m > 0).to_numpy()
    a = policies["CAPRI-Audit (all signals)"].to_numpy()[msk]
    b = policies["Rule-only (counts)"].to_numpy()[msk]
    t = d.at_risk.to_numpy()[msk]
    obs = prf(a, t)["f1"] - prf(b, t)["f1"]
    cnt = 0
    for _ in range(n_perm):
        sw = rng.random(len(a)) < 0.5
        a2, b2 = np.where(sw, b, a), np.where(sw, a, b)
        cnt += (prf(a2, t)["f1"] - prf(b2, t)["f1"]) >= obs - 1e-12
    out["perm_test_f1_audit_vs_rule_mc>0"] = {"diff": float(obs), "p_one_sided": float((cnt + 1) / (n_perm + 1))}
    # permutation test against chance (shuffled flags)
    cnt = 0
    for _ in range(n_perm):
        cnt += prf(rng.permutation(a), t)["f1"] >= prf(a, t)["f1"] - 1e-12
    out["perm_test_f1_audit_vs_chance_mc>0"] = {"p_one_sided": float((cnt + 1) / (n_perm + 1))}
    return out
