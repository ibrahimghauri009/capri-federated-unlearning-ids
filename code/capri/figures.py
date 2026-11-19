"""Figures for the Phase 3 report. All figures are generated from logged results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import SCEN_ORDER, SCEN_SHARE, RHO

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
PAL = {"B0": "#4d4d4d", "B1": "#d62728", "B2": "#9467bd", "B3": "#8c564b", "B4": "#ff7f0e",
       "B5": "#e377c2", "B6": "#bcbd22", "B8[B1]": "#17becf", "Guard[B1]": "#1f77b4",
       "Guard[B4]": "#2ca02c", "Guard[B5]": "#08519c", "GuardF[B1]": "#6baed6"}
PAL.update({"B8[B4]": "#7fd8e3", "B8[B5]": "#0e7c86", "GuardF[B4]": "#98df8a", "GuardF[B5]": "#9ecae1"})
LABEL = {"B0": "B0 no unlearning", "B1": "B1 retrain oracle", "B2": "B2 lawful oracle",
         "B3": "B3 continued training", "B4": "B4 gradient ascent", "B5": "B5 FedEraser",
         "B6": "B6 distillation", "B8[B1]": "B8 exemplar fine-tuning (on B1)",
         "Guard[B1]": "CAPRI-Guard (on B1)", "Guard[B4]": "CAPRI-Guard (on B4)",
         "Guard[B5]": "CAPRI-Guard (on B5)", "GuardF[B1]": "CAPRI-Guard-F (on B1)",
         "B8[B4]": "B8 exemplar fine-tuning (on B4)", "B8[B5]": "B8 exemplar fine-tuning (on B5)",
         "GuardF[B4]": "CAPRI-Guard-F (on B4)", "GuardF[B5]": "CAPRI-Guard-F (on B5)",
         "Abl-fixed[B1]": "Fixed λ (no dual ascent)", "Abl-noanchor[B1]": "No KL anchor",
         "Abl-noprox[B1]": "No proximity term", "Abl-blind[B1]": "All families flagged (no audit)",
         "Abl-ruleflags[B1]": "Rule-audit flags"}


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), 1.96 * se


def dose_response(df: pd.DataFrame, out: Path, methods=("B0", "B1", "B4", "B5", "B2", "B8[B1]", "Guard[B1]"),
                  names=None):
    names = names or {"rare": "MITM", "medium": "Port_Scanning", "common": "DDoS_TCP"}
    scen = [s for s in SCEN_ORDER if s != "S0" and s in df.scenario.unique()]
    pos = np.arange(len(scen))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharex=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, [("recall_rare", f"Rare target family ({names['rare']})"), ("recall_medium", f"Medium target family ({names['medium']})"),
                                       ("recall_common", f"Common target family ({names['common']})"), ("accuracy", "Overall accuracy")]):
        for j, m in enumerate(methods):
            d = df[df.method == m]
            if d.empty:
                continue
            mu, ci = zip(*[_mean_ci(d[d.scenario == s][col]) for s in scen])
            off = (j - (len(methods) - 1) / 2) * 0.045
            ax.errorbar(pos + off, mu, yerr=ci, marker="o", ms=3.2, lw=1.3, capsize=1.5, color=PAL.get(m), label=LABEL.get(m, m))
        if col != "accuracy":
            ax.axhline(RHO, ls="--", lw=0.8, color="grey")
            ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{s}\n{SCEN_SHARE[s]:g}" for s in scen], fontsize=7.5)
    axes[0].set_ylabel("Recall on global test set")
    axes[2].set_ylabel("Recall on global test set")
    axes[3].set_ylabel("Accuracy")
    fig.supxlabel("Ownership scenario and departing client's share of each target family ($s_c$)", fontsize=9, y=0.085)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_dose_response.png")
    plt.close(fig)


def pareto(df: pd.DataFrame, out: Path, scenario="S3", methods=None):
    d = df[df.scenario == scenario]
    methods = methods or [m for m in PAL if m in d.method.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for m in methods:
        dm = d[d.method == m]
        for ax, col in zip(axes, ("forget_acc", "forget_gap")):
            ax.scatter(dm[col].mean(), dm.dd_deficit.mean(), s=38, color=PAL.get(m), label=LABEL.get(m, m), zorder=3,
                       edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Accuracy on the departed client's records")
    axes[0].set_ylabel(f"Detection-debt deficit at ρ={RHO}")
    axes[0].set_title("(a) Forget-set accuracy", fontsize=8.5)
    axes[1].axvspan(-0.02, 0.02, color="0.92", zorder=0)
    axes[1].axvline(0, color="grey", lw=0.7)
    axes[1].set_xlim(-0.03, 0.03)
    axes[1].set_xlabel("Forget-set accuracy minus class-matched test accuracy")
    axes[1].set_title("(b) Record-level memorisation (band: ±0.02)", fontsize=8.5)
    axes[1].legend(fontsize=6.5, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Forgetting versus capability ({scenario}, means over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"fig_pareto_{scenario}.png")
    plt.close(fig)


def cd_diagram(fr: dict, out: Path, name="fig_cd.png", title="Average rank (lower is better)"):
    ranks = dict(sorted(fr["ranks"].items(), key=lambda kv: kv[1]))
    r = np.array(list(ranks.values()))
    k, cd = len(ranks), fr["cd"]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * k + 1.4))
    ax.set_xlim(0.8, k + 2.6)
    ax.set_ylim(k - 0.4, -1.4)
    for i, (m, v) in enumerate(ranks.items()):
        ax.plot([v], [i], "o", color=PAL.get(m, "k"), ms=6)
        ax.text(v + 0.12, i, f"{LABEL.get(m, m)}  ({v:.2f})", va="center", fontsize=7.5)
    # groups whose average ranks differ by less than the critical difference (not significantly different)
    groups, last = [], -1
    for i in range(k):
        j = int(np.max(np.where(r - r[i] < cd)[0]))
        if j > i and j > last:
            groups.append((i, j)); last = j
    for g, (i, j) in enumerate(groups):
        x = 0.95 + 0.12 * g
        ax.plot([x, x], [i - 0.15, j + 0.15], color="k", lw=2.5)
    ax.plot([r[0], r[0] + cd], [-0.9, -0.9], color="k", lw=2)
    ax.text(r[0] + cd + 0.1, -0.9, f"critical difference = {cd:.2f}", va="center", fontsize=7.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Average rank (lower is better); vertical bars join methods not significantly different (Nemenyi)", fontsize=8)
    ax.set_title(f"{title}: Friedman χ² = {fr['chi2']:.1f}, p = {fr['p']:.1g}, {fr['n_blocks']} scenario × seed blocks", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def audit_calibration(at: pd.DataFrame, out: Path):
    d = at[at.r_pre >= RHO].dropna(subset=["rhat_all"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, msk, title in ((axes[0], np.ones(len(d), bool), "All families"), (axes[1], (d.m > 0).to_numpy(), "Families other clients still hold (m_c > 0)")):
        e = d[msk]
        for name, col, mk in (("CAPRI-Audit (all signals)", "rhat_all", "o"), ("Ownership only", "rhat_ownership", "^")):
            ax.scatter(e[col], e.r_oracle, s=12, alpha=0.6, marker=mk, label=name)
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.axhline(RHO, ls="--", lw=0.7, color="grey"); ax.axvline(RHO, ls="--", lw=0.7, color="grey")
        ax.set_xlabel("Predicted post-deletion recall"); ax.set_ylabel("Actual recall (retrain oracle)")
        ax.set_title(title, fontsize=8.5)
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(out / "fig_audit_calibration.png")
    plt.close(fig)


def bars(df: pd.DataFrame, methods, col, out: Path, name, title, ylabel, scenario="S3"):
    d = df[(df.scenario == scenario) & df.method.isin(methods)]
    mu = [d[d.method == m][col].mean() for m in methods]
    sd = [d[d.method == m][col].std() for m in methods]
    fig, ax = plt.subplots(figsize=(0.75 * len(methods) + 2, 3.2))
    ax.bar(range(len(methods)), mu, yerr=sd, capsize=3, color=[PAL.get(m, "#6baed6") for m in methods])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABEL.get(m, m) for m in methods], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    fig.savefig(out / name)
    plt.close(fig)


def ablation_panels(df: pd.DataFrame, methods, out: Path, scenario="S3", name="fig_ablation.png"):
    d = df[(df.scenario == scenario) & df.method.isin(methods)]
    panels = [("dd_deficit", f"Detection-debt deficit (ρ={RHO})"), ("col_vs_pre", "Collateral loss vs pre-deletion"),
              ("benign_far", "Benign false-alarm rate")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, (col, t) in zip(axes, panels):
        mu = [d[d.method == m][col].mean() for m in methods]
        sd = [d[d.method == m][col].std() for m in methods]
        ax.barh(range(len(methods)), mu, xerr=sd, capsize=2, color=[PAL.get(m, "#6baed6") for m in methods])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([LABEL.get(m, m) for m in methods] if ax is axes[0] else [], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(t, fontsize=8.5)
    fig.suptitle(f"Ablation of CAPRI-Guard ({scenario}, base B1, mean ± s.d. over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def sweep_k(df: pd.DataFrame, out: Path, scenario="S3"):
    d = df[df.scenario == scenario]
    ks = [0, 4, 10, 20, 50]
    rows = {"CAPRI-Guard": [], "Exemplar fine-tuning (B8)": []}
    for k in ks:
        if k == 0:
            g = b8 = d[d.method == "B1"].dd_deficit
        elif k == 50:
            g, b8 = d[d.method == "Guard[B1]"].dd_deficit, d[d.method == "B8[B1]"].dd_deficit
        else:
            g, b8 = d[d.method == f"Sweep-k{k}[B1]"].dd_deficit, d[d.method == f"SweepB8-k{k}[B1]"].dd_deficit
        rows["CAPRI-Guard"].append(_mean_ci(g)); rows["Exemplar fine-tuning (B8)"].append(_mean_ci(b8))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (name, v), c in zip(rows.items(), ("#1f77b4", "#17becf")):
        mu, ci = zip(*v)
        ax.errorbar(ks, mu, yerr=ci, marker="o", capsize=2, label=name, color=c)
    ax.set_xlabel("Exemplars per class k (fit + check)")
    ax.set_ylabel(f"Detection-debt deficit at ρ={RHO}")
    ax.set_title(f"Exemplar budget ({scenario}, base B1)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(out / "fig_sweep_k.png")
    plt.close(fig)


def confusion_panels(cms: dict, classes, fam: int, out: Path, name="fig_confusion.png"):
    """Row of family `fam` across models: where its attacks go."""
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(cms) + 1.2))
    keys = list(cms)
    mat = np.array([cms[k][fam] / max(cms[k][fam].sum(), 1) for k in keys])
    top = np.argsort(-mat.sum(0))[:6]
    im = ax.imshow(mat[:, top], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top))); ax.set_xticklabels([classes[t] for t in top], rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([LABEL.get(k, k) for k in keys], fontsize=7.5)
    for i in range(len(keys)):
        for j in range(len(top)):
            ax.text(j, i, f"{mat[i, top[j]]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, top[j]] > 0.5 else "black")
    ax.set_title(f"Predicted class for test records of {classes[fam]}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(out / name)
    plt.close(fig)


def embedding_panels(embs: dict, y: np.ndarray, classes, highlight: list[int], out: Path, name="fig_embeddings.png"):
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, len(embs), figsize=(4 * len(embs), 3.6))
    for ax, (k, Z) in zip(np.atleast_1d(axes), embs.items()):
        P = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(Z.astype(np.float32))
        ax.scatter(P[:, 0], P[:, 1], s=2, c="lightgrey")
        for c, col in zip(highlight, ("#d62728", "#1f77b4", "#2ca02c")):
            msk = y == c
            ax.scatter(P[msk, 0], P[msk, 1], s=4, color=col, label=classes[c])
        ax.set_title(LABEL.get(k, k), fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=7, markerscale=3, frameon=False)
    fig.savefig(out / name)
    plt.close(fig)
