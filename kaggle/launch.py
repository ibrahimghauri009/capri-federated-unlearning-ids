"""Generate and push a Kaggle kernel for a list of runs.

usage: python launch.py <slug> <job> '<runs json>' ['<extra json>'] [--sources slug1,slug2]
"""
import json, subprocess, sys
from pathlib import Path

USER = "muhammadibrahim9990"
here = Path(__file__).parent
slug, job, runs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else {}
kernel_sources = []
if "--sources" in sys.argv:
    kernel_sources = [f"{USER}/{s}" for s in sys.argv[sys.argv.index("--sources") + 1].split(",")]
d = here / "kernels" / slug
d.mkdir(parents=True, exist_ok=True)
src = (here / "kernel_template.py").read_text()
src = src.replace("__JOB__", job).replace("__RUNS__", json.dumps(runs)).replace("__EXTRA__", repr(extra))
(d / "run.py").write_text(src)
compile(src, str(d / "run.py"), "exec")   # refuse to push a script with a syntax error
meta = {
    "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": [f"{USER}/capri-code"] + ([f"{USER}/capri-ciciot2023-sample"] if extra.get("dataset") == "ciciot2023"
                                                 else ["mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"]),
    "competition_sources": [], "kernel_sources": kernel_sources,
}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d)], capture_output=True, text=True)
print(r.stdout.strip()[-300:], r.stderr.strip()[-300:])
