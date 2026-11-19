"""Regenerate the Phase 3 figures as vector PDFs for the paper.

usage: python tools/make_figures.py <phase3_dir> <figures_out>

Runs the Phase 3 analysis scripts unchanged on the logged Phase 3 outputs, with matplotlib's savefig redirected
so that every figure is written as PDF (vector) into <figures_out>, prefixed by dataset. Tables written by those
scripts go to a temporary directory and are discarded; the paper's numbers are read from Phase 3 results/.
"""
import runpy
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

p3, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(p3 / "code"))
_orig = matplotlib.figure.Figure.savefig
PREFIX = {"value": ""}


# printed width (inches) of each figure in the paper; text width = 6.3 in, column width = 3.0 in.
# The figure is re-laid out at this width so that fonts print at their nominal 7-9 pt without scaling.
WIDTH = {"fig_dose_response": 5.6, "fig_confusion_rare": 4.6, "fig_cd": 6.3, "fig_ablation": 6.3,
         "fig_embeddings": 6.3, "fig_sensitivity": 6.3}


def savefig(self, fname, *a, **kw):
    f = Path(str(fname))
    if f.suffix == ".png":
        kw.pop("dpi", None)
        if f.stem in WIDTH:
            w, h = self.get_size_inches()
            self.set_size_inches(WIDTH[f.stem], h * WIDTH[f.stem] / w * (1.15 if f.stem in ("fig_dose_response", "fig_sensitivity") else 1.0))
        for ax in self.axes:  # the paper writes the exemplar count as q
            if ax.get_title() == "Exemplars per class k":
                ax.set_title("Exemplars per class q", fontsize=ax.title.get_fontsize())
        if f.stem == "fig_ablation":  # three long panel titles: smaller title font so they do not collide
            for ax in self.axes:
                ax.title.set_fontsize(7.5)
        target = out / f"{PREFIX['value']}{f.stem}.pdf"
        _orig(self, target, *a, **kw)
        print("figure", target.name)
    else:
        _orig(self, fname, *a, **kw)


matplotlib.figure.Figure.savefig = savefig
tmp = Path(tempfile.mkdtemp())
for prefix, res in (("edge_", "results/full"), ("cic_", "results/full_cic")):
    PREFIX["value"] = prefix
    sys.argv = ["make_results.py", str(p3 / res), str(tmp / prefix)]
    runpy.run_path(str(p3 / "code" / "make_results.py"), run_name="__main__")
# combined sensitivity figure needs the analysis tables of both datasets and pass 3 (from Phase 3)
PREFIX["value"] = ""
sys.argv = ["sensitivity_figure.py", str(p3 / "results/analysis"), str(p3 / "results/analysis_cic"),
            str(p3 / "results/analysis_combined")]
runpy.run_path(str(p3 / "code" / "sensitivity_figure.py"), run_name="__main__")
