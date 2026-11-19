# CAPRI experiment runner on Kaggle (generated from kernel_template.py)
import glob, json, os, pickle, shutil, subprocess, sys, time, zipfile
from pathlib import Path

JOB = "__JOB__"            # pass1 | pass2
RUNS = __RUNS__            # list of [scenario, seed]
EXTRA = __EXTRA__          # dict of extra options

t_start = time.time()
work = Path("/kaggle/working")
# ---- locate and import the code ----
code_root = None
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    depth = root.count(os.sep)
    if depth <= 6:
        print("input:", root, files[:6], flush=True)
    if "experiment.py" in files and "fl.py" in files and code_root is None:
        shutil.copytree(root, work / "code" / "capri", dirs_exist_ok=True)
        code_root = str(work / "code")
    for f in files:
        if f == "capri.zip" and code_root is None:
            zipfile.ZipFile(os.path.join(root, f)).extractall(work / "code")
            code_root = str(work / "code")
if code_root is None:
    raise SystemExit("code not found under /kaggle/input")
sys.path.insert(0, code_root)
print("code:", code_root, flush=True)

import torch
from capri.data import load_edge_iiotset, load_ciciot2023
from capri.fl import FLConfig
from capri import experiment as E

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
DATASET = EXTRA.get("dataset", "edgeiiotset")
if DATASET == "ciciot2023":
    csv = [os.path.join(r, "ciciot2023_sample.csv") for r, _, fs in os.walk("/kaggle/input", followlinks=True) if "ciciot2023_sample.csv" in fs]
    info = json.load(open(os.path.join(os.path.dirname(csv[0]), "ciciot2023_sample.json")))
    print("data:", csv, flush=True)
    cl = load_ciciot2023(csv[0], expected_sha256=info["sample_sha256"])
else:
    csv = [os.path.join(r, "DNN-EdgeIIoT-dataset.csv") for r, _, fs in os.walk("/kaggle/input", followlinks=True) if "DNN-EdgeIIoT-dataset.csv" in fs]
    print("data:", csv, flush=True)
    cl = load_edge_iiotset(csv[0], verify_hash=True)
print("clean rows:", len(cl.y), "classes:", cl.classes, flush=True)

out = work / "results"
out.mkdir(exist_ok=True)
# resume from earlier outputs attached as inputs
for f in [os.path.join(r, x) for r, _, fs in os.walk("/kaggle/input", followlinks=True) for x in fs
          if x.startswith(("p1_", "p2_", "p3_")) or x == "audit_table.csv"]:
    shutil.copy(f, out)
cfg = FLConfig(**EXTRA.get("fl", {}))
logf = open(out / f"log_{JOB}.txt", "a")
def log(msg):
    line = f"[{time.time() - t_start:7.0f}s] {msg}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()

if JOB == "pass1":
    for scen, seed in RUNS:
        try:
            E.run_pass1(cl, scen, seed, out, dev, cfg=cfg, log=log)
        except Exception as e:
            import traceback; log(f"FAILED {scen} s{seed}: {e}\n{traceback.format_exc()}")
elif JOB == "pass2":
    from capri import experiment2 as E2
    E2.run_all(cl, RUNS, out, dev, cfg=cfg, log=log, **EXTRA.get("pass2", {}))
elif JOB == "pass3":
    import pandas as pd
    from capri import experiment3 as E3
    adf = pd.read_csv(out / "audit_table.csv")
    for variant, seed in RUNS:
        try:
            E3.run_variant(cl, variant, seed, out, dev, adf, base_cfg=cfg, log=log)
        except Exception as e:
            import traceback; log(f"FAILED {variant} s{seed}: {e}" + chr(10) + traceback.format_exc())
log("finished")
