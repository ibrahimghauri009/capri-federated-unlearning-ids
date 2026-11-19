"""Compact dose-response figure for the paper (one row of four panels, printed at full text width).

usage: python tools/paper_figures.py <phase3_dir> <figures_out>

Same data and statistics as the Phase 3 dose-response figure (results/analysis*/long_table.csv): mean target-family
recall and accuracy per ownership scenario, with 95% confidence intervals over seeds, drawn at the printed size
(6.3 in wide) so that all text prints at 7-8 pt.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p3, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(p3 / "code"))
from capri.figures import PAL, LABEL  # same colours and labels as the Phase 3 figures

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.8})
SCEN = ["S1", "S2a", "S2b", "S2c", "S3"]
SHARE = ["0.1", "0.5", "0.8", "0.95", "1"]
METHODS = ["B0", "B1", "B4", "B5", "B6", "B8[B1]", "Guard[B1]"]
LAB = dict(LABEL, **{"Guard[B1]": "CAPRI-Guard-S (on B1)", "B8[B1]": "B8 exemplar fine-tuning (on B1)"})


def mci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean(), (1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


for key, an in (("edge", "analysis"), ("cic", "analysis_cic")):
    df = pd.read_csv(p3 / "results" / an / "long_table.csv")
    run = json.load(open(next((p3 / "results" / ("full" if key == "edge" else "full_cic")).glob("p1_S3_s0.json"))))
    names = run["targets"]["target_names"]
    panels = [(f"recall_{t}", f"{t.capitalize()} target\n{names[t]}") for t in ("rare", "medium", "common")]
    panels.append(("accuracy", "Overall\naccuracy"))
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.2))
    x = np.arange(len(SCEN))
    for ax, (col, title) in zip(axes, panels):
        for m in METHODS:
            mu, ci = zip(*[mci(df[(df.scenario == s) & (df.method == m)][col]) for s in SCEN])
            ax.errorbar(x, mu, yerr=ci, marker="o", ms=2.6, lw=1.0, capsize=1.5, color=PAL.get(m, "k"),
                        label=LAB.get(m, m))
        if col != "accuracy":
            ax.axhline(0.8, color="grey", lw=0.6, ls=":")
            ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, pad=3)
        ax.set_xticks(x, SHARE)
        ax.set_xlabel("share $s_c$")
    axes[0].set_ylabel("Recall")
    axes[3].set_ylabel("Accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
               handlelength=1.6)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.6)
    fig.savefig(out / f"{key}_dose_compact.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", f"{key}_dose_compact.pdf")
