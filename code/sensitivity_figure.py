"""Sensitivity line plots (Deliverable 3, Step 10): detection debt and membership inference against each factor.

usage: python sensitivity_figure.py <edge_analysis> <cic_analysis> <combined_dir>

Edge-IIoTset, S3, base B1. Exemplar budget, label noise, recall floor and proximity weight come from the main
runs (10 seeds); label skew, number of clients and detector architecture from pass 3 (5 seeds, default design on
the same seeds as reference); privacy noise on the audit counts from extensions_local.py (both datasets, audit
F1 on families other clients still hold). Points are means with 95% confidence intervals over seeds.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
edge_an, cic_an, comb = map(Path, sys.argv[1:4])
main = pd.read_csv(edge_an / "long_table.csv")
main = main[main.scenario == "S3"]
p3 = pd.read_csv(comb / "pass3_long.csv")
G, B8 = "#1f77b4", "#17becf"


def mci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean(), (1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def mm(meth, col):
    return main[main.method == meth][col]


def pv(var, meth, col):
    return p3[(p3.variant == var) & (p3.method == meth)][col]


# factor -> (x labels, x positions, {series: [values per x]})
def factors(col):
    f = {}
    f["Exemplars per class k"] = ([0, 4, 10, 20, 50], {
        "CAPRI-Guard-S": [mm("B1", col), mm("Sweep-k4[B1]", col), mm("Sweep-k10[B1]", col), mm("Sweep-k20[B1]", col),
                          mm("Guard[B1]", col)],
        "B8": [mm("B1", col), mm("SweepB8-k4[B1]", col), mm("SweepB8-k10[B1]", col), mm("SweepB8-k20[B1]", col),
               mm("B8[B1]", col)]})
    f["Exemplar label noise"] = (["0%", "10%", "20%"], {
        "CAPRI-Guard-S": [mm("Guard[B1]", col), mm("Sweep-noise0.1[B1]", col), mm("Sweep-noise0.2[B1]", col)]})
    f["Recall floor ρ"] = ([0.5, 0.7, 0.8, 0.9], {
        "CAPRI-Guard-S": [mm("Sweep-rho0.5[B1]", col), mm("Sweep-rho0.7[B1]", col), mm("Guard[B1]", col),
                          mm("Sweep-rho0.9[B1]", col)]})
    f["Proximity weight γ"] = (["10⁻³", "10⁻²", "10⁻¹"], {
        "CAPRI-Guard-S": [mm("Sweep-gamma0.001[B1]", col), mm("Guard[B1]", col), mm("Sweep-gamma0.1[B1]", col)]})
    for name, xs, vars_ in (("Label skew α", [0.1, 0.5, 1.0], ["alpha0.1", "default", "alpha1.0"]),
                            ("Number of clients K", ["10", "20"], ["default", "K20"]),
                            ("Detector", ["MLP", "1D-CNN"], ["default", "cnn"])):
        f[name] = (xs, {"CAPRI-Guard-S": [pv(v, "Guard[B1]", col) for v in vars_],
                        "B8": [pv(v, "B8[B1]", col) for v in vars_]})
    return f


def draw(ax, name, xs, series, ylabel=None, hline=None):
    pos = np.arange(len(xs))
    for s, vals in series.items():
        mu, ci = zip(*[mci(v) for v in vals])
        ax.errorbar(pos, mu, yerr=ci, marker="o", ms=3.5, capsize=2, lw=1.2, label=s, color=G if s.startswith("CAPRI") else B8)
    ax.set_xticks(pos, [str(x) for x in xs])
    ax.set_title(name, fontsize=8.5)
    if hline is not None:
        ax.axhline(hline, color="grey", lw=0.8, ls=":")
    if ylabel:
        ax.set_ylabel(ylabel)


fig, axes = plt.subplots(4, 4, figsize=(9.2, 9.0))
dd, mia = factors("dd_deficit"), factors("mia_auc")
for i, (name, (xs, s)) in enumerate(dd.items()):
    ax = axes[i // 4][i % 4]
    draw(ax, name, xs, s, ylabel="DD deficit (S3)" if i % 4 == 0 else None)
# privacy noise panel: audit F1 on the non-trivial subset, both datasets
ax = axes[1][3]
for an, lab, c in ((edge_an, "Edge-IIoTset", G), (cic_an, "CICIoT2023", "#d62728")):
    t = pd.read_csv(an / "table_ext_privacy_audit.csv")
    t = t[(t.subset == "m_c > 0 (non-trivial)") & (t.policy == "CAPRI-Audit")]
    order = ["none", "1.0", "0.5"]
    y = [float(t[t.epsilon.astype(str) == e].f1_mean.iloc[0]) for e in order]
    e = [1.96 * float(t[t.epsilon.astype(str) == e].f1_sd.iloc[0]) for e in order]
    ax.errorbar(np.arange(3), y, yerr=e, marker="o", ms=3.5, capsize=2, lw=1.2, color=c, label=lab)
ax.set_xticks(np.arange(3), ["none", "ε = 1", "ε = 0.5"])
ax.set_title("Privacy noise on audit counts", fontsize=8.5)
ax.set_ylabel("Audit F1 (m_c > 0)")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, fontsize=7)
for i, (name, (xs, s)) in enumerate(mia.items()):
    ax = axes[2 + i // 4][i % 4]
    draw(ax, name, xs, s, ylabel="MIA AUC (S3)" if i % 4 == 0 else None, hline=0.5)
    ax.set_ylim(0.45, 0.55)
axes[3][3].axis("off")
axes[0][0].legend(frameon=False, fontsize=7)
fig.text(0.5, 0.497, "(b) Membership-inference AUC on the departing client's records (0.5 = chance)", ha="center", fontsize=9)
fig.text(0.5, 1.0, "(a) Detection-debt deficit; privacy-noise panel: audit F1", ha="center", fontsize=9)
fig.tight_layout(h_pad=3.4)
fig.subplots_adjust(top=0.97)
fig.savefig(comb / "fig_sensitivity.png")
print("wrote", comb / "fig_sensitivity.png")
