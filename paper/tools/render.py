"""Render the paper: fill every <<expression>> in main_template.tex from the Phase 3 logged outputs.

usage: python tools/render.py <phase3_dir> main_template.tex main.tex

Every number and every results table in the paper is computed here from the Phase 3 result files
(results/full, results/full_cic, results/p3 and the analysis tables derived from them); none is typed by hand.
Numbers quoted from other publications are written in the template itself and marked with an asterisk.
The output is a single .tex file, as the Springer Nature template requires.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P3, TPL, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
R = P3 / "results"
RHO = 0.8
DS = {}
for key, res, an in (("edge", "full", "analysis"), ("cic", "full_cic", "analysis_cic")):
    DS[key] = {"df": pd.read_csv(R / an / "long_table.csv"), "st": json.load(open(R / an / "stats.json")),
               "p1": {f.stem: json.load(open(f)) for f in sorted((R / res).glob("p1_S*_s?.json"))},
               "p2": {f.stem: json.load(open(f)) for f in sorted((R / res).glob("p2_S*_s?.json"))},
               "an": R / an, "res": R / res}
COMB = R / "analysis_combined"
CST = json.load(open(COMB / "stats_combined.json"))
DNS = json.load(open(R / "edge_dns_field_check.json"))
df = st = p1 = p2 = an = None
LAB = {"edge": "Edge-IIoTset", "cic": "CICIoT2023"}
CODE_URL = r"\url{https://github.com/ibrahimghauri009/capri-federated-unlearning-ids}"


def use(key):
    global df, st, p1, p2, an
    d = DS[key]
    df, st, p1, p2, an = d["df"], d["st"], d["p1"], d["p2"], d["an"]
    return ""


use("edge")


# ---------------------------------------------------------------- formatting
def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def fp(p):
    """p-value in LaTeX."""
    if p >= 0.001:
        return f"{p:.3f}"
    m_, e = f"{p:.1e}".split("e")
    return f"${m_}\\times10^{{{int(e)}}}$"


def num(n):
    return f"{int(n):,}".replace(",", "{,}")


# ---------------------------------------------------------------- data access
def sel(sc, meth, ds=None):
    d = DS[ds]["df"] if ds else df
    return d[(d.scenario == sc) & (d.method == meth)]


def v(sc, meth, col, ds=None):
    return float(sel(sc, meth, ds)[col].mean())


def m(sc, meth, col, d=3, ds=None):
    return f(v(sc, meth, col, ds), d)


def tgt_series(sc, meth, ds=None):
    s = sel(sc, meth, ds)
    return s[["recall_rare", "recall_medium", "recall_common"]].mean(axis=1)


def tgt(sc, meth, ds=None):
    return float(tgt_series(sc, meth, ds).mean())


def tgt_pm(sc, meth, ds=None, d=3):
    x = tgt_series(sc, meth, ds)
    return f"{f(x.mean(), d)}\\,$\\pm$\\,{f(x.std(), d)}"


def pm(sc, meth, col, ds=None, d=3):
    x = sel(sc, meth, ds)[col]
    return f"{f(x.mean(), d)}\\,$\\pm$\\,{f(x.std(), d)}"


def nseeds(sc, ds=None):
    return int(sel(sc, "B0", ds).shape[0])


def csv(name, ds=None, **kw):
    return pd.read_csv((DS[ds]["an"] if ds else an) / name, **kw)


def comb(name, **kw):
    return pd.read_csv(COMB / name, **kw)


def TN(t, ds=None):
    r = next(iter((DS[ds]["p1"] if ds else p1).values()))["targets"]["target_names"]
    return r[t]


def tn(t, ds=None):
    return TN(t, ds).replace("_", "\\_")


def n_eval(ds=None):
    return len(next(iter((DS[ds]["p1"] if ds else p1).values()))["families_evaluable"])


def h1p(meth, field, scope="pooled"):
    t = comb("table_h1_learned_only.csv")
    return float(t[(t.scope == scope) & (t.method == meth)][field].iloc[0])


def h1all(meth, field):
    t = comb("table_h1_pooled.csv")
    return float(t[t.method == meth][field].iloc[0])


def h1ds(meth, field, ds):
    t = csv("table_h1_tests.csv", ds)
    return float(t[t.method == meth][field].iloc[0])


def h3(comp, field, ds=None):
    t = csv("table_h3_tests.csv", ds) if ds else comb("table_h3_pooled.csv")
    return float(t[t.comparison == comp][field].iloc[0])


def audit(sub, pol, field, ds):
    return DS[ds]["st"]["audit"][sub][pol][field]


def auditp(key, ds):
    return DS[ds]["st"]["audit"][key]


def xaud(test_on, a, sub, col):
    x = comb("table_audit_cross_dataset.csv")
    return float(x[(x.test_on == test_on) & (x.audit == a) & (x.subset == sub)][col].iloc[0])


def abt(variant, metric, field, ds):
    t = csv("table_ablation_tests.csv", ds)
    return float(t[(t.variant == variant) & (t.metric == metric)][field].iloc[0])


def fr(ds):
    return DS[ds]["st"]["friedman_dd_deficit"]


def p3v(var, meth, col, stat="mean"):
    d = comb("pass3_long.csv")
    return float(getattr(d[(d.variant == var) & (d.method == meth)][col], stat)())


def p3tgt(var, meth):
    return float(np.mean([p3v(var, meth, f"recall_{t}") for t in ("rare", "medium", "common")]))


def p3t(var, comp, col="p"):
    t = comb("table_pass3_tests.csv")
    return float(t[(t.variant == var) & (t.comparison == comp)][col].iloc[0])


def leak(ds, meth):
    t = pd.read_csv(DS[ds]["an"] / "table_ext_before_after_leakage.csv")
    return float(t[t.method == meth].auc_recall_drop.iloc[0])


def priv(ds, eps, pol, col, sub="m_c > 0 (non-trivial)"):
    t = pd.read_csv(DS[ds]["an"] / "table_ext_privacy_audit.csv")
    return float(t[(t.epsilon.astype(str) == eps) & (t.policy == pol) & (t.subset == sub)][col].iloc[0])


def mean_flags(ds, sc="S3"):
    return float(np.mean([len(r["flags"]["all"]) for r in DS[ds]["p2"].values() if r["scenario"] == sc]))


def timing(col, stat, ds):
    return DS[ds]["st"]["timing"][col][stat]


def split(ds, part):
    return int(DS[ds]["st"]["dataset"]["split_sizes"][part])


def nfeat(ds):
    return int(DS[ds]["st"]["dataset"]["n_features"])


def n_models(ds):
    return len(DS[ds]["df"])


def all_leak(ds):
    return all(r["leakage"]["passed"] for r in DS[ds]["p1"].values())


def guard_reruns():
    return sum(bool(g.get("rerun")) for k in DS for r in DS[k]["p2"].values() for g in r["guard_info"].values())


def guard_runs():
    return sum(len(r["guard_info"]) for k in DS for r in DS[k]["p2"].values())


def target_share(ds, seed=0):
    r = DS[ds]["p1"][f"p1_S3_s{seed}"]
    sup = r["test_support"]
    return sum(sup[c] for c in r["targets"]["targets"].values()) / sum(sup)


def total(ds):
    return sum(DS[ds]["st"]["dataset"]["split_sizes"].values())


def raw_edge():
    return int(pd.read_csv(R / "table_dedup_counts.csv").raw.sum())


def rel(new, old):
    """Relative change in percent."""
    return 100 * (new - old) / old


def pf(sc, meth, fam, ds=None):
    return float(csv("table_per_family_recall.csv", ds, header=[0, 1], index_col=0)[(sc, meth)][fam])


# ---------------------------------------------------------------- LaTeX tables
NAME = {"B0": "B0 No unlearning", "B1": "B1 Retrain oracle", "B2": "B2 Lawful oracle", "B3": "B3 Continued training",
        "B4": "B4 Gradient ascent~\\cite{halimi2022,kwon2026}", "B5": "B5 FedEraser~\\cite{liu2021federaser}",
        "B6": "B6 Distillation~\\cite{wu2022distill}", "B8[B1]": "B8 Exemplar fine-tuning",
        "Guard[B1]": "CAPRI-Guard-S", "GuardF[B1]": "CAPRI-Guard-F"}


def rows(rs):
    return " \\\\\n".join(" & ".join(map(str, r)) for r in rs) + " \\\\"


def T_rq1():
    out = []
    for meth in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        r = [NAME[meth]]
        for ds in ("edge", "cic"):
            r += [tgt_pm("S1", meth, ds), tgt_pm("S3", meth, ds)]
        r += ["--" if meth in ("B0", "B2") else fp(h1all(meth, "p_holm"))]
        out.append(r)
    return rows(out)


def T_audit():
    out = []
    A, RU = "CAPRI-Audit (all signals)", "Rule-only (counts)"
    for ds in ("edge", "cic"):
        a = DS[ds]["st"]["audit"]
        s1, s2 = "m_c > 0 (non-trivial)", "all families"
        out.append([f"{LAB[ds]} (within)",f(a[s1][A]["precision"]), f(a[s1][A]["recall"]), f(a[s1][A]["f1"]),
                    f(a[s1][RU]["f1"]), fp(a["perm_test_f1_audit_vs_rule_mc>0"]["p_one_sided"]),
                    f(a[s2][A]["f1"]), f(a[s2][RU]["f1"])])
    for fit_on, test_on, k in (("CICIoT2023", "Edge-IIoTset", "audit_cross_perm_CICIoT2023_to_Edge-IIoTset"),
                               ("Edge-IIoTset", "CICIoT2023", "audit_cross_perm_Edge-IIoTset_to_CICIoT2023")):
        s1, s2 = "m_c > 0 (non-trivial)", "all families"
        out.append([f"{fit_on.split('-')[0].replace('CICIoT2023', 'CICIoT')} $\\rightarrow$ {test_on.split('-')[0].replace('CICIoT2023', 'CICIoT')}",f(xaud(test_on, "all", s1, "precision")),
                    f(xaud(test_on, "all", s1, "recall")), f(xaud(test_on, "all", s1, "f1")),
                    f(xaud(test_on, "rule", s1, "f1")), fp(CST[k]["p_one_sided"]),
                    f(xaud(test_on, "all", s2, "f1")), f(xaud(test_on, "rule", s2, "f1"))])
    return rows(out)


def T_repair():
    out = []
    names = {"B1": "B1 Retrain oracle", "B8[B1]": "\\quad + B8 fine-tuning", "Guard[B1]": "\\quad + CAPRI-Guard-S",
             "GuardF[B1]": "\\quad + CAPRI-Guard-F", "B4": "B4 Gradient ascent", "B8[B4]": "\\quad + B8 fine-tuning",
             "Guard[B4]": "\\quad + CAPRI-Guard-S", "B5": "B5 FedEraser", "B8[B5]": "\\quad + B8 fine-tuning",
             "Guard[B5]": "\\quad + CAPRI-Guard-S", "B2": "B2 Lawful oracle"}
    for meth in ["B2", "B1", "B8[B1]", "Guard[B1]", "GuardF[B1]", "B4", "B8[B4]", "Guard[B4]", "B5", "B8[B5]", "Guard[B5]"]:
        r = [names[meth]]
        for ds in ("edge", "cic"):
            r += [f(tgt("S3", meth, ds)), pm("S3", meth, "dd_deficit", ds, 2), pct(v("S3", meth, "benign_far", ds), 1)]
        out.append(r)
    return rows(out)


def T_h3():
    out = []
    nm = {"B1": "B1", "B4": "B4", "B5": "B5", "B2": "B2", "B3": "B3", "B6": "B6"}
    for comp in ["Guard[B1] vs B1", "Guard[B4] vs B4", "Guard[B5] vs B5", "Guard[B1] vs B2", "Guard[B1] vs B3",
                 "Guard[B1] vs B6", "Guard[B1] vs B8[B1]", "Guard[B4] vs B8[B4]", "Guard[B5] vs B8[B5]"]:
        a, b = comp.split(" vs ")
        lab = f"Guard-S({a[6:8]}) vs {b.replace('[', '(').replace(']', ')')}"
        out.append([lab] + [f(h3(comp, "mean_diff", ds), 2) for ds in ("edge", "cic")] +
                   [f(h3(comp, "mean_diff"), 2), f(h3(comp, "rbc"), 2), fp(h3(comp, "p_holm"))])
    return rows(out)


ABL = [("Abl-fixed", "Fixed $\\lambda$ (no dual ascent)"), ("Abl-noanchor", "No KL anchor"),
       ("Abl-noprox", "No proximity term ($\\gamma=0$)"), ("Abl-blind", "All families flagged (no audit)"),
       ("Abl-ruleflags", "Flags from counts rule"), ("B8", "Plain fine-tuning (B8)")]


def sig(p):
    return "$^{\\dagger}$" if p < 0.05 else ""


def T_ablation():
    out = [["Full CAPRI-Guard-S"] + [x for ds in ("edge", "cic") for x in
                                      (f(abt("Abl-fixed", "dd_deficit", "guard_mean", ds), 2),
                                       f(abt("Abl-fixed", "col_vs_pre", "guard_mean", ds)),
                                       pct(abt("Abl-fixed", "benign_far", "guard_mean", ds), 1))]]
    for key, lab in ABL:
        r = [lab]
        for ds in ("edge", "cic"):
            r += [f(abt(key, "dd_deficit", "variant_mean", ds), 2) + sig(abt(key, "dd_deficit", "p_holm", ds)),
                  f(abt(key, "col_vs_pre", "variant_mean", ds)) + sig(abt(key, "col_vs_pre", "p_holm", ds)),
                  pct(abt(key, "benign_far", "variant_mean", ds), 1) + sig(abt(key, "benign_far", "p_holm", ds))]
        out.append(r)
    return rows(out)


def T_basecmp():
    out = []
    spec = [("B1", "Retrain oracle (reference)", "R"), ("B5", "FedEraser~\\cite{liu2021federaser}", "R"),
            ("B4", "Gradient ascent~\\cite{halimi2022}", "R"), ("B6", "KD unlearning~\\cite{wu2022distill}", "R"),
            ("B8[B1]", "Exemplar fine-tuning (B8)", "--"), ("Guard[B5]", "CAPRI (Guard-S on FedEraser)", "--"),
            ("Guard[B4]", "CAPRI (Guard-S on gradient ascent)", "--"), ("Guard[B1]", "CAPRI (Guard-S on retrain)", "--")]
    for meth, lab, mark in spec:
        r = [lab + ("$^{\\mathrm{R}}$" if mark == "R" else "")]
        for ds in ("edge", "cic"):
            r += [f(tgt("S1", meth, ds)), f(tgt("S3", meth, ds)), f(v("S3", meth, "dd_deficit", ds), 2),
                  f(v("S3", meth, "mia_auc", ds))]
        out.append(r)
    return rows(out)


def T_main_cmp():
    out = []
    spec = [("B1", "Retrain oracle (B1)"), ("B2", "Lawful oracle (B2)"),
            ("B5", "FedEraser~\\cite{liu2021federaser}$^{\\mathrm{R}}$ (B5)"), ("B8[B5]", "\\quad + B8 fine-tuning"),
            ("Guard[B5]", "\\quad + CAPRI-Guard-S"),
            ("B4", "Gradient ascent~\\cite{halimi2022}$^{\\mathrm{R}}$ (B4)"), ("B8[B4]", "\\quad + B8 fine-tuning"),
            ("Guard[B4]", "\\quad + CAPRI-Guard-S"),
            ("B6", "KD unlearning~\\cite{wu2022distill}$^{\\mathrm{R}}$ (B6)"),
            ("B8[B1]", "B1 + B8 fine-tuning"), ("Guard[B1]", "B1 + CAPRI-Guard-S"), ("GuardF[B1]", "B1 + CAPRI-Guard-F")]
    for meth, lab in spec:
        r = [lab]
        for ds in ("edge", "cic"):
            r += [f(tgt("S1", meth, ds)), f(tgt("S3", meth, ds)), f(v("S3", meth, "dd_deficit", ds), 2),
                  pct(v("S3", meth, "benign_far", ds), 1), f(v("S3", meth, "mia_auc", ds))]
        out.append(r)
    return rows(out)


def T_improve():
    out = []
    for base, guard, lab in (("B5", "Guard[B5]", "FedEraser~\\cite{liu2021federaser}"),
                             ("B4", "Guard[B4]", "Gradient ascent~\\cite{halimi2022}"),
                             ("B6", "Guard[B1]", "KD unlearning~\\cite{wu2022distill}"),
                             ("B8[B1]", "Guard[B1]", "Exemplar fine-tuning (B8)")):
        r = [lab]
        for ds in ("edge", "cic"):
            dt = 100 * (tgt("S3", guard, ds) - tgt("S3", base, ds))
            dd0, dd1 = v("S3", base, "dd_deficit", ds), v("S3", guard, "dd_deficit", ds)
            dfar = 100 * (v("S3", guard, "benign_far", ds) - v("S3", base, "benign_far", ds))
            r += [("+" if dt >= 0 else "$-$") + f"{abs(dt):.1f}", ("+" if dd1 >= dd0 else "") + f(dd1 - dd0, 2) + f" ({rel(dd1, dd0):+.1f}\\%)".replace("-", "$-$"),
                  ("+" if dfar >= 0 else "$-$") + f"{abs(dfar):.1f}"]
        out.append(r)
    return rows(out)


def T_pass3():
    lab = {"default": "Default design", "alpha0.1": "Label skew $\\alpha=0.1$",
           "alpha1.0": "Label skew $\\alpha=1.0$", "K20": "$K=20$ clients", "cnn": "1D-CNN detector",
           "unweighted": "Unweighted loss", "partial0.5": "Partial participation",
           "sequential": "Sequential (1st deletion)"}
    out = []
    for var in lab:
        out.append([lab[var], f(p3tgt(var, "B0")), f(p3tgt(var, "B1")), f(p3v(var, "B1", "dd_deficit"), 2),
                    f(p3v(var, "B8[B1]", "dd_deficit"), 2), f(p3v(var, "Guard[B1]", "dd_deficit"), 2),
                    fp(p3t(var, "Guard[B1] vs B8[B1]"))])
    return rows(out)


def T_ext():
    out = []
    for meth, lab in (("B1", "B1 Retrain oracle"), ("B4", "B4 Gradient ascent"), ("B5", "B5 FedEraser"),
                      ("B6", "B6 Distillation"), ("B8[B1]", "B8 Exemplar fine-tuning"), ("Guard[B1]", "CAPRI-Guard-S")):
        out.append([lab, f(leak("edge", meth)), f(leak("cic", meth))])
    return rows(out)


def T_perfam():
    t = csv("table_per_family_recall.csv", "edge", header=[0, 1], index_col=0)
    out = []
    for fam in t.index:
        out.append([fam.replace("_", "\\_")] + [f(t[(sc, mm)][fam], 2) for sc, mm in
                                                 (("S1", "B0"), ("S3", "B0"), ("S3", "B1"), ("S3", "B4"), ("S3", "B5"),
                                                  ("S3", "B8[B1]"), ("S3", "Guard[B1]"))])
    return rows(out)


# ---------------------------------------------------------------- fill
tpl = TPL.read_text(encoding="utf-8")
missing = []


def repl(mt):
    expr = mt.group(1)
    try:
        return str(eval(expr))
    except Exception as e:
        missing.append(f"{expr}: {e!r}")
        return "<<" + expr + ">>"


body = re.sub(r"<<(.+?)>>", repl, tpl, flags=re.S)
if missing:
    print("FAILED PLACEHOLDERS:\n" + "\n".join(missing))
    sys.exit(1)
OUT.write_text(body, encoding="utf-8")
print("wrote", OUT, "placeholders:", len(re.findall(r"<<(.+?)>>", tpl, flags=re.S)))
