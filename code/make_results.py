"""Build every table, statistical test and figure of the Phase 3 report from logged outputs.

usage: python make_results.py <results_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from capri.analysis import (load, long_table, summary, wilcoxon, holm, tost_paired, friedman_nemenyi,
                            audit_report, bootstrap_ci, SCEN_ORDER, SCEN_SHARE, RHO)
from capri import figures as F

res_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
p1, p2 = load(res_dir)
df = long_table(p1, p2)
df.to_csv(out / "long_table.csv", index=False)
any_run = next(iter(p1.values()))
classes = any_run["classes"]
targets = any_run["targets"]["targets"]
T = {}   # all statistics, written to stats.json


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


md = []
# ---------------------------------------------------------------------------
# Dataset and setup
# ---------------------------------------------------------------------------
prep = any_run["prep"]
T["dataset"] = {"split_sizes": prep["split_sizes"], "n_features": prep["n_features"],
                "dropped_high_corr": prep["dropped_high_corr"], "dropped_low_variance": prep["dropped_low_variance"],
                "log_transformed": prep["log_transformed"], "classes": classes,
                "split_class_counts": prep["split_class_counts"], "targets": any_run["targets"]["target_names"],
                "runs": sorted([f"{k[0]}_s{k[1]}" for k in p1])}
cc = pd.DataFrame(prep["split_class_counts"], index=classes)[["train", "val", "test", "ex"]]
cc.index.name = "class"
cc.to_csv(out / "table_class_counts_seed0.csv")
md.append("## Class counts per split (seed 0)\n" + md_table(cc.reset_index()))

# ---------------------------------------------------------------------------
# Pre-deletion model quality (gate)
# ---------------------------------------------------------------------------
gate = df[df.method == "B0"].groupby("scenario")[["accuracy", "macro_f1", "pre_rare", "pre_medium", "pre_common"]].agg(["mean", "std"])
gate.to_csv(out / "table_predeletion_gate.csv")
rounds = pd.DataFrame([{"scenario": k[0], "seed": k[1], "best_round": r["pre_best_round"], "rounds_run": r["pre_rounds"],
                        "pre_seconds": r["timings"]["pre"], "total_seconds": r["timings"]["total"]} for k, r in p1.items()])
rounds.to_csv(out / "table_rounds_timing.csv", index=False)
T["timing"] = rounds.drop(columns=["scenario", "seed"]).describe().to_dict()

# ---------------------------------------------------------------------------
# RQ1: capability loss under shared vs sole ownership
# ---------------------------------------------------------------------------
base = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
cols = ["accuracy", "macro_f1", "dd", "dd_deficit", "wfr", "col_vs_pre", "col_vs_oracle",
        "recall_rare", "recall_medium", "recall_common", "silent_rare", "silent_medium", "silent_common",
        "benign_far", "mia_auc", "forget_acc", "relearn"]
main = df[df.method.isin(base)].groupby(["scenario", "method"])[cols].agg(["mean", "std"])
main.to_csv(out / "table_rq1_main.csv")
short = df[df.method.isin(base)].groupby(["scenario", "method"])[["accuracy", "dd", "recall_rare", "recall_medium", "recall_common", "col_vs_pre", "mia_auc"]].mean().round(3)
md.append("## RQ1 main results (means over seeds)\n" + md_table(short.reset_index()))

# H1: target-family capability loss, S1 vs S3, pooled units (seed x target family)
h1 = []
for meth in ["B1", "B3", "B4", "B5", "B6"]:
    def losses(scen):
        d = df[(df.method == meth) & (df.scenario == scen)].set_index("seed")
        d0 = df[(df.method == "B0") & (df.scenario == scen)].set_index("seed")
        rows = []
        for t in ("rare", "medium", "common"):
            l = (d0[f"recall_{t}"] - d[f"recall_{t}"].reindex(d0.index))
            rows += [(s, t, v) for s, v in l.items()]
        return pd.DataFrame(rows, columns=["seed", "target", "loss"]).set_index(["seed", "target"]).loss
    a, b = losses("S3"), losses("S1")
    idx = a.index.intersection(b.index)
    w = wilcoxon(a[idx].values, b[idx].values, alternative="greater")
    acc3 = df[(df.method == meth) & (df.scenario == "S3")].set_index("seed").accuracy
    acc0 = df[(df.method == "B0") & (df.scenario == "S3")].set_index("seed").accuracy
    h1.append({"method": meth, **w, "mean_loss_S3": float(a.mean()), "mean_loss_S1": float(b.mean()),
               "ci_loss_S3": bootstrap_ci(a.values), "acc_change_S3": float((acc3 - acc0).mean())})
for r, p in zip(h1, holm([r["p"] for r in h1])):
    r["p_holm"] = p
T["H1_wilcoxon_S3_vs_S1"] = h1
pd.DataFrame(h1).to_csv(out / "table_h1_tests.csv", index=False)
# robustness: leave-one-family-out recomputation of H1 (each target family dropped in turn)
lofo = []
for meth in ["B1", "B4", "B5"]:
    for drop in ("rare", "medium", "common"):
        keep = [t for t in ("rare", "medium", "common") if t != drop]
        def ls(scen):
            d = df[(df.method == meth) & (df.scenario == scen)].set_index("seed")
            d0 = df[(df.method == "B0") & (df.scenario == scen)].set_index("seed")
            return np.concatenate([(d0[f"recall_{t}"] - d[f"recall_{t}"].reindex(d0.index)).values for t in keep])
        w = wilcoxon(ls("S3"), ls("S1"), alternative="greater")
        lofo.append({"method": meth, "dropped": drop, **w})
T["robustness_leave_one_family_out"] = lofo
pd.DataFrame(lofo).to_csv(out / "table_h1_leave_one_family_out.csv", index=False)

# per-family loss vs ownership share: dose-response GLM (retrain oracle)
rows = []
for k, r in p1.items():
    sup = r["test_support"]
    for tname, c in targets.items():
        for meth in ("B1", "B4", "B5"):
            rec = r["results"][meth]["recall"][c]
            rows.append({"scenario": k[0], "seed": k[1], "target": tname, "method": meth, "s": r["s"][c],
                         "correct": round(rec * sup[c]), "n": sup[c], "r_pre": r["results"]["B0"]["recall"][c]})
dr = pd.DataFrame(rows)
dr = dr[dr.scenario != "S0"]
dose = {}
for meth in ("B1", "B4", "B5"):
    d = dr[dr.method == meth]
    X = pd.get_dummies(d[["s", "target"]], columns=["target"], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = np.column_stack([d.correct, d.n - d.correct])
    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.seed.to_numpy()})
        if np.isfinite(fit.bse["s"]):
            dose[meth] = {"coef_s": float(fit.params["s"]), "se_s": float(fit.bse["s"]), "p_s": float(fit.pvalues["s"]),
                          "odds_ratio_per_0.1_share": float(np.exp(0.1 * fit.params["s"]))}
        else:
            dose[meth] = {"error": "complete separation (recall is a step function of s); GLM slope not estimable"}
    except Exception as e:
        dose[meth] = {"error": str(e)}
T["dose_response_glm"] = dose

# Non-parametric ordered-trend test (Page's L): seeds are blocks, scenarios S1 < S2a < S2b < S2c < S3
# are ordered treatments, response = recall loss (B0 recall - method recall). Robust to the step shape.
from scipy.stats import rankdata


def page_perm(M, n_perm=20000, seed=0):
    """Page's L with average ranks for ties; p-value by permuting treatment labels within blocks
    (exact under ties, one-sided for increasing loss)."""
    R = rankdata(M, axis=1)
    w = np.arange(1, M.shape[1] + 1)
    L = float((R.sum(0) * w).sum())
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, *R.shape)), axis=2)
    Lp = (np.take_along_axis(np.broadcast_to(R, perm.shape), perm, axis=2).sum(1) * w).sum(1)
    return L, float((1 + (Lp >= L - 1e-9).sum()) / (n_perm + 1))


order = ["S1", "S2a", "S2b", "S2c", "S3"]
dr["loss"] = dr.r_pre - dr.correct / dr.n
trend = []
for meth in ("B1", "B4", "B5"):
    for tname in targets:
        d = dr[(dr.method == meth) & (dr.target == tname)]
        piv = d.pivot_table(index="seed", columns="scenario", values="loss").reindex(columns=order).dropna()
        mean_rec = (d.assign(rec=d.correct / d.n).groupby("scenario").rec.mean().reindex(order))
        below = [s for s in order if mean_rec[s] < RHO]
        row = {"method": meth, "target": tname, "blocks": len(piv),
               **{f"mean_recall_{s}": float(mean_rec[s]) for s in order},
               "first_scenario_below_rho": below[0] if below else "none",
               "first_share_below_rho": SCEN_SHARE[below[0]] if below else None}
        if len(piv) >= 2:
            L, pval = page_perm(piv.to_numpy())
            row.update({"page_L": L, "page_p": pval})
        trend.append(row)
for r, p in zip(trend, holm([r.get("page_p", 1.0) for r in trend])):
    r["page_p_holm"] = p
T["dose_response_trend"] = trend
pd.DataFrame(trend).to_csv(out / "table_dose_trend.csv", index=False)

# ---------------------------------------------------------------------------
# RQ2: audit
# ---------------------------------------------------------------------------
at_path = res_dir / "audit_table.csv"
if at_path.exists():
    at = pd.read_csv(at_path)
    T["audit"] = audit_report(at)
    rows = []
    for subset, pol in T["audit"].items():
        if isinstance(pol, dict) and "No warning" in pol:
            for name, m in pol.items():
                rows.append({"subset": subset, "policy": name, "precision": fmt(m["precision"]), "recall": fmt(m["recall"]),
                             "F1": fmt(m["f1"]), "n": m["n"], "at risk": m["n_at_risk"]})
    md.append("## RQ2 audit\n" + md_table(pd.DataFrame(rows)))
    pd.DataFrame(rows).to_csv(out / "table_rq2_audit.csv", index=False)
    F.audit_calibration(at, out)
    # audit failure analysis per family (all-signal audit, families with pre-deletion recall >= rho)
    e = at[at.r_pre >= RHO].copy()
    e["flag"] = e.label_all.isin(["Required", "Collateral"])
    e["at_risk"] = e.at_risk.astype(bool)
    per = e.groupby("family_name").apply(lambda g: pd.Series({
        "deletions": len(g), "at_risk": int(g.at_risk.sum()), "required": int((g.m == 0).sum()),
        "tp": int((g.flag & g.at_risk).sum()), "fp": int((g.flag & ~g.at_risk).sum()), "fn": int((~g.flag & g.at_risk).sum()),
        "mean_r_pre": float(g.r_pre.mean()), "mean_r_oracle": float(g.r_oracle.mean())}), include_groups=False)
    per.to_csv(out / "table_audit_by_family.csv")
    fn = e[~e.flag & e.at_risk][["scenario", "seed", "family_name", "s", "m", "r_probe", "rhat_all", "r_pre", "r_oracle"]]
    fn.to_csv(out / "table_audit_false_negatives.csv", index=False)

# ---------------------------------------------------------------------------
# RQ3: CAPRI-Guard
# ---------------------------------------------------------------------------
if p2:
    rq3_methods = ["B1", "B4", "B5", "B2", "B8[B1]", "B8[B4]", "B8[B5]", "Guard[B1]", "Guard[B4]", "Guard[B5]",
                   "GuardF[B1]", "GuardF[B4]", "GuardF[B5]", "B6"]
    rq3 = df[df.method.isin(rq3_methods)].groupby(["scenario", "method"])[
        ["accuracy", "dd", "dd_deficit", "wfr", "recall_rare", "recall_medium", "recall_common", "col_vs_pre",
         "benign_far", "mia_auc", "forget_acc", "relearn"]].agg(["mean", "std"])
    rq3.to_csv(out / "table_rq3_main.csv")
    short = df[df.method.isin(rq3_methods)].groupby(["scenario", "method"])[["dd", "dd_deficit", "recall_rare", "recall_medium", "recall_common", "benign_far", "mia_auc"]].mean().round(3)
    md.append("## RQ3 (means over seeds)\n" + md_table(short.reset_index()))
    units = df[df.scenario.isin(["S1", "S2a", "S2b", "S2c", "S3"])]
    def paired(m1, m2, col):
        a = units[units.method == m1].set_index(["scenario", "seed"])[col]
        b = units[units.method == m2].set_index(["scenario", "seed"])[col]
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
    T["H3_wilcoxon_dd_deficit"] = h3
    pd.DataFrame(h3).to_csv(out / "table_h3_tests.csv", index=False)
    tost = []
    for b in ("B1", "B4", "B5"):
        for col, margin in (("mia_auc", 0.02), ("forget_acc", 0.02), ("forget_gap", 0.02)):
            x, y = paired(f"Guard[{b}]", "B2", col)
            tost.append({"comparison": f"Guard[{b}] vs B2", "metric": col, "margin": margin, **tost_paired(x, y, margin)})
    T["H3_tost_forgetting"] = tost
    pd.DataFrame(tost).to_csv(out / "table_h3_tost.csv", index=False)
    # record-level memorisation: forget-set accuracy minus class-matched test accuracy (all methods)
    mem_methods = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B8[B1]", "Guard[B1]", "GuardF[B1]", "Guard[B4]", "Guard[B5]"]
    mem = (df[df.method.isin(mem_methods)].groupby(["scenario", "method"])[["forget_acc", "cm_test_acc", "forget_gap", "mia_auc"]]
           .agg(["mean", "std"]).round(4))
    mem.to_csv(out / "table_memorisation_gap.csv")
    # exploratory (not pre-specified): Guard vs B8 per scenario, dd_deficit and rare-target recall
    expl = []
    for b in ("B1", "B4", "B5"):
        for sc in ("S1", "S2a", "S2b", "S2c", "S3"):
            for col, alt in (("dd_deficit", "less"), ("recall_rare", "greater")):
                a = df[(df.scenario == sc) & (df.method == f"Guard[{b}]")].set_index("seed")[col]
                c = df[(df.scenario == sc) & (df.method == f"B8[{b}]")].set_index("seed")[col]
                i = a.index.intersection(c.index)
                w = wilcoxon(a[i].values, c[i].values, alternative=alt)
                expl.append({"base": b, "scenario": sc, "metric": col, "guard": float(a[i].mean()), "b8": float(c[i].mean()), **w})
    for r, p in zip(expl, holm([r["p"] for r in expl])):
        r["p_holm"] = p
    T["exploratory_guard_vs_b8_by_scenario"] = expl
    pd.DataFrame(expl).to_csv(out / "table_exploratory_guard_vs_b8.csv", index=False)
    # recovery ratio on the sole-held targets (S3)
    rr = []
    for b in ("B1", "B4", "B5"):
        for t in ("rare", "medium", "common"):
            g = df[(df.scenario == "S3") & (df.method == f"Guard[{b}]")].set_index("seed")[f"recall_{t}"]
            u = df[(df.scenario == "S3") & (df.method == b)].set_index("seed")[f"recall_{t}"]
            e = df[(df.scenario == "S3") & (df.method == f"B8[{b}]")].set_index("seed")[f"recall_{t}"]
            pr = df[(df.scenario == "S3") & (df.method == "B0")].set_index("seed")[f"recall_{t}"]
            den = (pr - u).clip(lower=1e-6)
            rr.append({"base": b, "target": t, "RR_guard": float(((g - u) / den).clip(0, 1).mean()),
                       "RR_B8": float(((e - u) / den).clip(0, 1).mean())})
    T["recovery_ratio_S3"] = rr
    pd.DataFrame(rr).to_csv(out / "table_recovery_ratio.csv", index=False)
    try:
        fr_methods = ["B1", "B4", "B5", "B6", "B8[B1]", "B8[B4]", "B8[B5]", "Guard[B1]", "Guard[B4]", "Guard[B5]"]
        fr = friedman_nemenyi(units, fr_methods, "dd_deficit")
        T["friedman_dd_deficit"] = fr
        F.cd_diagram(fr, out, title="Detection-debt deficit")
    except Exception as e:
        T["friedman_dd_deficit"] = {"error": str(e)}
    # ablations (S3)
    abl = [m for m in df.method.unique() if m.startswith("Abl-")]
    if abl:
        ab = df[(df.scenario == "S3") & df.method.isin(abl + ["Guard[B1]", "Guard[B4]", "Guard[B5]", "B1", "B4", "B5"])]
        ab = ab.groupby("method")[["dd", "dd_deficit", "recall_rare", "recall_medium", "recall_common", "col_vs_pre",
                                   "benign_far", "mia_auc", "accuracy"]].agg(["mean", "std"])
        ab.to_csv(out / "table_ablation_S3.csv")
        # paired two-sided Wilcoxon tests of each ablation against full Guard, units = seed x base (n = 30)
        s3 = df[df.scenario == "S3"]
        abt = []
        for a in ("Abl-fixed", "Abl-noanchor", "Abl-noprox", "Abl-blind", "Abl-ruleflags", "B8"):
            for col in ("dd_deficit", "col_vs_pre", "benign_far", "accuracy"):
                xs, ys = [], []
                for b in ("B1", "B4", "B5"):
                    x = s3[s3.method == f"{a}[{b}]"].set_index("seed")[col]
                    y = s3[s3.method == f"Guard[{b}]"].set_index("seed")[col]
                    i = x.index.intersection(y.index)
                    xs += list(x[i]); ys += list(y[i])
                abt.append({"variant": a, "metric": col, "variant_mean": float(np.mean(xs)), "guard_mean": float(np.mean(ys)),
                            **wilcoxon(np.array(xs), np.array(ys), alternative="two-sided")})
        for r, p in zip(abt, holm([r["p"] for r in abt])):
            r["p_holm"] = p
        T["ablation_tests"] = abt
        pd.DataFrame(abt).to_csv(out / "table_ablation_tests.csv", index=False)
        sw = df[(df.scenario == "S3") & df.method.str.startswith(("Sweep", "SweepB8"))]
        sw.groupby("method")[["dd", "dd_deficit", "recall_rare", "recall_medium", "recall_common", "benign_far", "mia_auc"]].agg(["mean", "std"]).to_csv(out / "table_sweeps_S3.csv")
        F.sweep_k(df, out)
        F.ablation_panels(df, ["B1", "Guard[B1]", "Abl-fixed[B1]", "Abl-noanchor[B1]", "Abl-noprox[B1]", "Abl-blind[B1]",
                               "Abl-ruleflags[B1]", "B8[B1]"], out)
    F.pareto(df, out, "S3")
    F.pareto(df, out, "S2c")
    F.dose_response(df, out, names=any_run['targets']['target_names'])
    # failure analysis on S3 seed 0
    k = ("S3", 0)
    if k in p2:
        a1 = np.load(res_dir / "p1_S3_s0.npz")
        a2 = np.load(res_dir / "p2_S3_s0.npz")
        yte = a1["y_test"].astype(int)
        C = len(classes)
        def cm(pred):
            return np.bincount(yte * C + pred.astype(int), minlength=C * C).reshape(C, C)
        cms = {"B0": cm(a1["pred_B0"]), "B1": cm(a1["pred_B1"]), "B8[B1]": cm(a2["pred_B8[B1]"]), "Guard[B1]": cm(a2["pred_Guard[B1]"])}
        for tname, c in targets.items():
            F.confusion_panels(cms, classes, c, out, f"fig_confusion_{tname}.png")
        if "emb_B0" in a2.files:
            idx = a2["emb_idx"]
            F.embedding_panels({"B0": a2["emb_B0"], "B1": a2["emb_B1"], "Guard[B1]": a2["emb_Guard_B1"]},
                               yte[idx], classes, list(targets.values()), out)
else:
    F.dose_response(df, out, methods=("B0", "B1", "B4", "B5", "B2"), names=any_run["targets"]["target_names"])

# robustness: leave-one-seed-out on the headline H1 effect (B1)
loo = []
for s in sorted(df.seed.unique()):
    d = df[df.seed != s]
    l3 = (d[(d.method == "B0") & (d.scenario == "S3")].recall_medium.mean() - d[(d.method == "B1") & (d.scenario == "S3")].recall_medium.mean())
    loo.append({"left_out_seed": int(s), "mean_loss_medium_S3_B1": float(l3)})
T["robustness_leave_one_seed_out"] = loo

# per-family recall and F1 (evaluable families), and macro-F1 restricted to Normal + evaluable families
pf_rows, mf_rows = [], []
for key, r in p1.items():
    res = dict(r["results"])
    if key in p2:
        res.update(p2[key]["results"])
    fam = list(r["families_evaluable"])
    cls = r["classes"]
    keep = [r.get("benign", 0) if isinstance(r.get("benign", 0), int) else 0] + fam
    for meth in ("B0", "B1", "B2", "B4", "B5", "B6", "B8[B1]", "Guard[B1]"):
        if meth not in res:
            continue
        v = res[meth]
        for c in fam:
            pf_rows.append({"scenario": key[0], "seed": key[1], "method": meth, "family": cls[c],
                            "recall": v["recall"][c], "f1": v["f1"][c], "silent_miss": v["silent_miss"][c]})
        f1 = np.array([np.nan if v["f1"][c] is None else v["f1"][c] for c in keep], float)
        mf_rows.append({"scenario": key[0], "seed": key[1], "method": meth, "macro_f1_evaluable": float(np.nanmean(f1))})
pf = pd.DataFrame(pf_rows)
pft = pf[pf.scenario.isin(["S1", "S3"])].pivot_table(index="family", columns=["scenario", "method"], values="recall", aggfunc="mean").round(3)
pft.to_csv(out / "table_per_family_recall.csv")
mf = pd.DataFrame(mf_rows).groupby(["scenario", "method"]).macro_f1_evaluable.agg(["mean", "std"]).round(4)
mf.to_csv(out / "table_macro_f1_evaluable.csv")
md.append("## Per-family recall, S1 and S3 (means over seeds)\n" + md_table(pft.reset_index()))

json.dump(T, open(out / "stats.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
(out / "tables.md").write_text("\n\n".join(md), encoding="utf-8")
print("wrote", out)
