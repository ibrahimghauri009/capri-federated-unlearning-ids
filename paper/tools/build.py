"""Render main.tex from the template and compile it with MiKTeX (pdflatex, bibtex, pdflatex x2).

usage: python tools/build.py [<results_root>]   (default: repository root, else ../../Phase 3)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[1]
# results root: the repository root (code/ and results/ next to paper/) or, in the course folders, ../../Phase 3
if len(sys.argv) > 1:
    p3 = Path(sys.argv[1])
elif (here.parent / "results").exists():
    p3 = here.parent
else:
    p3 = here.parents[1] / "Phase 3"
subprocess.run([sys.executable, str(here / "tools" / "render.py"), str(p3), "main_template.tex", "main.tex"],
               cwd=here, check=True)
bindir = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
latex = [str(bindir / "pdflatex.exe"), "-interaction=nonstopmode", "main.tex"]
for cmd in (latex, [str(bindir / "bibtex.exe"), "main"], latex, latex):
    subprocess.run(cmd, cwd=here, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log = (here / "main.log").read_text(encoding="latin-1")
errors = re.findall(r"^! .*", log, flags=re.M)
overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*", log)
undefined = re.findall(r"(?:Citation|Reference) `[^']+' .*undefined", log)
pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
print("pages:", pages.group(1) if pages else "?", "| errors:", len(errors), "| undefined refs:", len(undefined),
      "| overfull > 2pt:", [o for o in overfull if float(o) > 2])
for e in errors[:10]:
    print("  ", e)
for u in sorted(set(undefined))[:10]:
    print("  ", u)
