"""Extension experiments that need no new training (Deliverable 3, Step 7, items 3 and 5).

usage: python extensions_local.py <results_dir> <out_dir>

3. Privacy-protected audit: every client perturbs its own per-class label counts with Laplace
   noise of scale 1/epsilon (sensitivity 1 per record), rounds and clips at zero, and the server
   computes s_c and m_c from the noisy counts. The audit (leave-one-seed-out binomial GLM, all
   signals) and the counts rule are re-evaluated with the noisy ownership signals; 20 noise draws.
5. Before/after leakage test (P13-style): can an observer who compares the model before and
   after unlearning tell which families the departing client held? Score of family c = recall
   drop B0 -> unlearned model on the global test set; positives = families of which client 0
   held at least half (s_c >= 0.5). ROC AUC per method, pooled over scenarios S2a-S3 and seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from capri.capri import SIGNAL_SETS, fit_audit, predict_audit, audit_labels, rule_audit
from capri.analysis import prf

RHO = 0.8
res_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
at = pd.read_csv(res_dir / "audit_table.csv")
p1 = {(r["scenario"], r["seed"]): r for r in (json.load(open(f)) for f in sorted(res_dir.glob("p1_S*_s?.json")))}
p2 = {(r["scenario"], r["seed"]): r for r in (json.load(open(f)) for f in sorted(res_dir.glob("p2_S*_s?.json")))}


# ---------------------------------------------------------------- 3. privacy-protected audit
def noisy_signals(df, eps, rng):
    d = df.copy()
    for (sc, sd), g in d.groupby(["scenario", "seed"]):
        n = np.array(p1[(sc, sd)]["n"], float)
        if eps is not None:
            n = np.clip(np.round(n + rng.laplace(0, 1 / eps, n.shape)), 0, None)
        tot = n.sum(0)
        s = np.where(tot > 0, n[0] / np.maximum(tot, 1), 0.0)
        m = tot - n[0]
        fam = g.family.to_numpy()
        d.loc[g.index, "s"] = s[fam]
        d.loc[g.index, "m"] = m[fam]
        d.loc[g.index, "log_m"] = np.log1p(m[fam])
    return d


def evaluate(d):
    sig = SIGNAL_SETS["all"]
    pred = np.full(len(d), np.nan)
    for s in d.seed.unique():
        trm = (d.seed != s).to_numpy()
        res = fit_audit(d[trm].dropna(subset=sig + ["r_oracle"]), sig)
        pred[~trm] = predict_audit(res, d[~trm].fillna(0), sig)
    lab = audit_labels(pred, d.m.to_numpy(), RHO)
    flag = np.isin(lab, ["Required", "Collateral"])
    rule = rule_audit(d.s.to_numpy(), d.m.to_numpy())
    true_m = at.m.to_numpy()
    e = (at.r_pre >= RHO).to_numpy()
    risk = ((at.r_oracle < RHO) & (at.r_pre >= RHO)).to_numpy()
    rows = {}
    for subset, msk in (("all families", e), ("m_c > 0 (non-trivial)", e & (true_m > 0))):
        rows[(subset, "CAPRI-Audit")] = prf(flag[msk], risk[msk])
        rows[(subset, "Rule-only")] = prf(rule[msk], risk[msk])
    return rows


rng = np.random.default_rng(0)
priv = []
for eps in (None, 1.0, 0.5):
    draws = 1 if eps is None else 20
    acc = {}
    for _ in range(draws):
        for k, v in evaluate(noisy_signals(at, eps, rng)).items():
            acc.setdefault(k, []).append(v)
    for (subset, pol), vs in acc.items():
        priv.append({"epsilon": "none" if eps is None else eps, "subset": subset, "policy": pol, "draws": draws,
                     **{f"{m}_mean": float(np.mean([x[m] for x in vs])) for m in ("precision", "recall", "f1")},
                     "f1_sd": float(np.std([x["f1"] for x in vs]))})
pd.DataFrame(priv).to_csv(out / "table_ext_privacy_audit.csv", index=False)

# ---------------------------------------------------------------- 5. before/after leakage test
leak = []
methods = ["B1", "B2", "B3", "B4", "B5", "B6", "B8[B1]", "Guard[B1]"]
for meth in methods:
    sc_, lab_ = [], []
    for key, r in p1.items():
        if key[0] not in ("S2a", "S2b", "S2c", "S3"):
            continue
        res = dict(r["results"])
        if key in p2:
            res.update(p2[key]["results"])
        if meth not in res:
            continue
        r0 = np.array(res["B0"]["recall"], float)
        rm = np.array(res[meth]["recall"], float)
        for c in r["families_evaluable"]:
            sc_.append(r0[c] - rm[c])
            lab_.append(r["s"][c] >= 0.5)
    sc_, lab_ = np.array(sc_), np.array(lab_)
    ok = ~np.isnan(sc_)
    leak.append({"method": meth, "n_family_deletions": int(ok.sum()), "positives": int(lab_[ok].sum()),
                 "auc_recall_drop": float(roc_auc_score(lab_[ok], sc_[ok]))})
pd.DataFrame(leak).to_csv(out / "table_ext_before_after_leakage.csv", index=False)
print(pd.DataFrame(priv).round(3).to_string())
print(pd.DataFrame(leak).round(3).to_string())
