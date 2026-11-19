"""Learnability gate and target-family selection (Section 2.6 of the report).

usage: python target_gate.py ciciot2023 <sample.csv> <sample.json> <out.json>
       python target_gate.py edgeiiotset <DNN-EdgeIIoT-dataset.csv> - <out.json>

Rule (fixed before any deletion result): among attack families with >= 30 expected test
records, keep those that a centralised reference model (class-balanced histogram gradient
boosting, learning rate 0.02, 600 iterations, seed-0 split) detects with validation recall >= 0.8; sort them by size and take the
median family of the smallest, middle and largest third as the rare, medium and common target.
For CICIoT2023 the size is the number of records in the full release (the sample is capped, so
sample counts would tie at the cap).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from capri.data import load_ciciot2023, load_edge_iiotset, make_split, prepare, select_targets

ds, sample, info_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
if ds == "ciciot2023":
    info = json.loads(info_path.read_text())
    cl = load_ciciot2023(sample, expected_sha256=info["sample_sha256"])
    size = info["class_counts_source"]
else:
    cl = load_edge_iiotset(sample, verify_hash=True)
    size = cl.log["class_counts"]
sp = make_split(cl, 0, k=50)
pr = prepare(cl, sp)
q = select_targets(cl, k=50, fixed=None)["qualifying"]
C = len(cl.classes)
cnt = np.bincount(pr.y["train"], minlength=C).astype(float)
w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)[pr.y["train"]]
# learning rate 0.02: at the default 0.1 training diverges on CICIoT2023 (training accuracy 0.23)
m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.02, early_stopping=False, random_state=0)
m.fit(pr.X["train"], pr.y["train"], sample_weight=w * len(w) / w.sum())
res_train_acc = float((m.predict(pr.X["train"]) == pr.y["train"]).mean())
pv = m.predict(pr.X["val"])
rec = {}
for c in range(C):
    sel = pr.y["val"] == c
    rec[cl.classes[c]] = round(float((pv[sel] == c).mean()), 3) if sel.any() else None
elig = sorted([f for f in q if rec[f] is not None and rec[f] >= 0.8], key=lambda f: (size[f], f))
thirds = [list(t) for t in np.array_split(np.array(elig, dtype=object), 3)]
targets = {name: t[len(t) // 2] for name, t in zip(("rare", "medium", "common"), thirds)}
res = {"dataset": ds, "reference_model": "HistGradientBoosting, lr 0.02, 600 iterations, class-balanced weights (centralised), seed-0 split, validation set",
       "train_accuracy": res_train_acc,
       "clean_class_counts": cl.log["class_counts"], "exact_duplicates_removed": cl.log["exact_duplicates_removed"],
       "rows_in_conflicting_groups": cl.log["rows_in_conflicting_groups"], "n_features": len(pr.features),
       "val_recall": rec, "val_support": np.bincount(pr.y["val"], minlength=C).tolist(),
       "qualifying": q, "eligible_sorted_by_size": elig, "thirds": thirds, "targets": targets,
       "excluded_not_separable": [f for f in q if rec[f] is None or rec[f] < 0.8]}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("targets", "excluded_not_separable", "thirds", "n_features")}, indent=1))
print({k: v for k, v in rec.items()})
