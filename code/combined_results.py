"""Cross-dataset analyses (Deliverable 3, Steps 9-11) and the pass-3 summary.

usage: python combined_results.py <edge_results> <cic_results> <edge_analysis> <cic_analysis> <pass3_results> <out_dir>

1. H1 pooled over both datasets (seed x target family x dataset units), Wilcoxon + Holm.
2. H3 pooled over both datasets (scenario x seed x dataset units, S1-S3), Wilcoxon + Holm.
3. Audit fitted on one dataset and evaluated on the other (the planned leave-one-dataset-out fit).
4. Pass 3: sensitivity analyses and extension experiments (Edge-IIoTset, S3, 5 seeds), compared
   with the default design on the same seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from capri.analysis import wilcoxon, holm, prf, bootstrap_ci
from capri.capri import SIGNAL_SETS, fit_audit, predict_audit, audit_labels, rule_audit

RHO = 0.8
edge_res, cic_res, edge_an, cic_an, p3_res, out = map(Path, sys.argv[1:7])
out.mkdir(parents=True, exist_ok=True)
T = {}
dfs = {"Edge-IIoTset": pd.read_csv(edge_an / "long_table.csv"), "CICIoT2023": pd.read_csv(cic_an / "long_table.csv")}
for k, d in dfs.items():
    d["dataset"] = k
df = pd.concat(dfs.values(), ignore_index=True)


# ---------------------------------------------------------------- 1. pooled H1
def losses(d, meth, scen):
    a = d[(d.method == meth) & (d.scenario == scen)].set_index("seed")
    b0 = d[(d.method == "B0") & (d.scenario == scen)].set_index("seed")
    rows = []
    for t in ("rare", "medium", "common"):
        l = b0[f"recall_{t}"] - a[f"recall_{t}"].reindex(b0.index)
        rows += [((s, t), v) for s, v in l.items()]
    return pd.Series(dict(rows))


h1 = []
for meth in ("B1", "B3", "B4", "B5", "B6"):
    a3, a1 = [], []
    for k, d in dfs.items():
        l3, l1 = losses(d, meth, "S3"), losses(d, meth, "S1")
        i = l3.index.intersection(l1.index)
        a3 += list(l3[i]); a1 += list(l1[i])
    w = wilcoxon(np.array(a3), np.array(a1), alternative="greater")
    h1.append({"method": meth, **w, "mean_loss_S3": float(np.nanmean(a3)), "mean_loss_S1": float(np.nanmean(a1)),
               "ci_loss_S3": bootstrap_ci(np.array(a3))})
for r, p in zip(h1, holm([r["p"] for r in h1])):
    r["p_holm"] = p
T["H1_pooled"] = h1
pd.DataFrame(h1).to_csv(out / "table_h1_pooled.csv", index=False)

# H1 restricted to units whose family the pre-deletion model had learned in S3 (B0 recall >= rho);
# the S1 unit of the same seed and family is its pair. Reported separately (not pre-specified).
h1l = []
for scope, keys in (("Edge-IIoTset", ["Edge-IIoTset"]), ("CICIoT2023", ["CICIoT2023"]), ("pooled", list(dfs))):
    for meth in ("B1", "B3", "B4", "B5", "B6"):
        a3, a1 = [], []
        for k in keys:
            d = dfs[k]
            l3, l1 = losses(d, meth, "S3"), losses(d, meth, "S1")
            b0 = d[(d.method == "B0") & (d.scenario == "S3")].set_index("seed")
            ok = [i for i in l3.index.intersection(l1.index) if b0.loc[i[0], f"recall_{i[1]}"] >= RHO]
            a3 += list(l3[ok]); a1 += list(l1[ok])
        w = wilcoxon(np.array(a3), np.array(a1), alternative="greater")
        h1l.append({"scope": scope, "method": meth, **w, "mean_loss_S3": float(np.nanmean(a3)),
                    "mean_loss_S1": float(np.nanmean(a1))})
for scope in ("Edge-IIoTset", "CICIoT2023", "pooled"):
    rr = [r for r in h1l if r["scope"] == scope]
    for r, p in zip(rr, holm([r["p"] for r in rr])):
        r["p_holm"] = p
T["H1_learned_only"] = h1l
pd.DataFrame(h1l).to_csv(out / "table_h1_learned_only.csv", index=False)

# ---------------------------------------------------------------- 2. pooled H3
units = df[df.scenario.isin(["S1", "S2a", "S2b", "S2c", "S3"])]


def paired(m1, m2, col):
    a = units[units.method == m1].set_index(["dataset", "scenario", "seed"])[col]
    b = units[units.method == m2].set_index(["dataset", "scenario", "seed"])[col]
    i = a.index.intersection(b.index)
    return a[i].values, b[i].values


comps = []
for b in ("B1", "B4", "B5"):
    comps += [(f"Guard[{b}]", b), (f"Guard[{b}]", f"B8[{b}]")]
comps += [("Guard[B1]", "B2"), ("Guard[B1]", "B3"), ("Guard[B1]", "B6")]
h3 = []
for m1, m2 in comps:
    x, y = paired(m1, m2, "dd_deficit")
    h3.append({"comparison": f"{m1} vs {m2}", **wilcoxon(x, y, alternative="less")})
for r, p in zip(h3, holm([r["p"] for r in h3])):
    r["p_holm"] = p
T["H3_pooled"] = h3
pd.DataFrame(h3).to_csv(out / "table_h3_pooled.csv", index=False)

# ---------------------------------------------------------------- 3. cross-dataset audit
at = {"Edge-IIoTset": pd.read_csv(edge_res / "audit_table.csv"), "CICIoT2023": pd.read_csv(cic_res / "audit_table.csv")}
xa = []
for fit_on, test_on in (("Edge-IIoTset", "CICIoT2023"), ("CICIoT2023", "Edge-IIoTset")):
    tr_, te_ = at[fit_on], at[test_on]
    risk = ((te_.r_oracle < RHO) & (te_.r_pre >= RHO)).to_numpy()
    ev = (te_.r_pre >= RHO).to_numpy()
    for name, sig in SIGNAL_SETS.items():
        res = fit_audit(tr_.dropna(subset=sig + ["r_oracle"]), sig)
        rhat = predict_audit(res, te_.fillna(0), sig)
        flag = np.isin(audit_labels(rhat, te_.m.to_numpy(), RHO), ["Required", "Collateral"])
        for subset, msk in (("all families", ev), ("m_c > 0 (non-trivial)", ev & (te_.m.to_numpy() > 0))):
            xa.append({"fit_on": fit_on, "test_on": test_on, "audit": name, "subset": subset,
                       **prf(flag[msk], risk[msk]),
                       "mae": float(np.nanmean(np.abs(rhat[msk] - te_.r_oracle.to_numpy()[msk])))})
        if name == "all":
            flag_all = flag
    rule = rule_audit(te_.s.to_numpy(), te_.m.to_numpy())
    for subset, msk in (("all families", ev), ("m_c > 0 (non-trivial)", ev & (te_.m.to_numpy() > 0))):
        xa.append({"fit_on": "-", "test_on": test_on, "audit": "rule", "subset": subset, **prf(rule[msk], risk[msk]),
                   "mae": np.nan})
    # paired permutation test (as within-dataset): transferred all-signal audit vs counts rule, m_c > 0
    msk = ev & (te_.m.to_numpy() > 0)
    a_, b_, t_ = flag_all[msk], rule[msk], risk[msk]
    obs = prf(a_, t_)["f1"] - prf(b_, t_)["f1"]
    rng = np.random.default_rng(0)
    cnt = 0
    for _ in range(5000):
        sw = rng.random(len(a_)) < 0.5
        cnt += (prf(np.where(sw, b_, a_), t_)["f1"] - prf(np.where(sw, a_, b_), t_)["f1"]) >= obs - 1e-12
    T[f"audit_cross_perm_{fit_on}_to_{test_on}"] = {"diff": float(obs), "p_one_sided": float((cnt + 1) / 5001)}
pd.DataFrame(xa).to_csv(out / "table_audit_cross_dataset.csv", index=False)
T["audit_cross_dataset"] = xa

# ---------------------------------------------------------------- 4. pass 3
p3 = [json.load(open(f)) for f in sorted(p3_res.glob("p3_*.json"))] if p3_res.exists() else []
if not p3:
    json.dump(T, open(out / "stats_combined.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(pd.DataFrame(h1).round(4).to_string()); print(pd.DataFrame(h1l).round(4).to_string())
    print(pd.DataFrame(h3)[["comparison", "n", "mean_diff", "rbc", "p_holm"]].round(4).to_string())
    print(pd.DataFrame(xa)[["fit_on", "test_on", "audit", "subset", "precision", "recall", "f1"]].round(3).to_string())
    sys.exit(0)
rows = []
for r in p3:
    for meth, v in r["results"].items():
        tg = r["targets"]["targets"]
        rows.append({"variant": r["variant"], "seed": r["seed"], "method": meth,
                     "accuracy": v["accuracy"], "dd": v["dd"]["rho0.8"]["dd_count"], "dd_deficit": v["dd"]["rho0.8"]["dd_deficit"],
                     "benign_far": v["benign_far"], "mia_auc": v["mia_auc"], "forget_acc": v["forget_acc"],
                     **{f"recall_{t}": v["recall"][c] for t, c in tg.items()}})
# default design on the same seeds (S3, seeds 0-4) from the main Edge-IIoTset runs
d0 = dfs["Edge-IIoTset"]
d0 = d0[(d0.scenario == "S3") & (d0.seed < 5) & d0.method.isin(["B0", "B1", "B8[B1]", "Guard[B1]"])]
for _, x in d0.iterrows():
    rows.append({"variant": "default", "seed": x.seed, "method": x.method, "accuracy": x.accuracy, "dd": x.dd,
                 "dd_deficit": x.dd_deficit, "benign_far": x.benign_far, "mia_auc": x.mia_auc, "forget_acc": x.forget_acc,
                 "recall_rare": x.recall_rare, "recall_medium": x.recall_medium, "recall_common": x.recall_common})
p3df = pd.DataFrame(rows)
p3df.to_csv(out / "pass3_long.csv", index=False)
summ = p3df.groupby(["variant", "method"])[["accuracy", "dd", "dd_deficit", "recall_rare", "recall_medium",
                                            "recall_common", "benign_far", "mia_auc"]].agg(["mean", "std"]).round(4)
summ.to_csv(out / "table_pass3_summary.csv")
# Guard vs B8 and Guard vs B1 within each variant (exploratory, 5 seeds each)
tests = []
for var in p3df.variant.unique():
    d = p3df[p3df.variant == var].set_index(["seed", "method"]).dd_deficit.unstack()
    for a, b in (("Guard[B1]", "B1"), ("Guard[B1]", "B8[B1]"), ("Guard[B1_2]", "B1_2"), ("Guard[B1_2]", "B8[B1_2]")):
        if a in d and b in d:
            w = wilcoxon(d[a].values, d[b].values, alternative="two-sided")
            tests.append({"variant": var, "comparison": f"{a} vs {b}", **w})
pd.DataFrame(tests).to_csv(out / "table_pass3_tests.csv", index=False)
T["pass3_tests"] = tests
json.dump(T, open(out / "stats_combined.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(pd.DataFrame(h1).round(4).to_string())
print(pd.DataFrame(h3)[["comparison", "n", "mean_diff", "rbc", "p_holm"]].round(4).to_string())
print(pd.DataFrame(xa)[["fit_on", "test_on", "audit", "subset", "precision", "recall", "f1"]].round(3).to_string())
print(summ.xs("mean", axis=1, level=1)[["dd_deficit", "recall_rare", "recall_medium", "recall_common", "accuracy"]].to_string())
