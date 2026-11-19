"""Data collection and leakage-safe preprocessing (Deliverable 3, Steps 2-4).

Order of operations (fixed to prevent leakage):
  1. drop identifier / shortcut columns, clean values, drop constant columns
  2. remove exact duplicates (features + label)
  3. group identical feature vectors; every group goes to exactly one split,
     so no feature vector is shared between exemplar, test, validation and
     training data (stronger than removing test rows identical to training rows)
  4. split: exemplar pool (k per class, from unique vectors) -> stratified
     20% test / 10% validation / 70% client pool
  5. fit encoders, log transform, imputation, scaling and feature filters on
     the client pool only. Mean, variance and correlation are computed from
     counts, sums, sums of squares and cross-products, i.e. the quantities a
     server can aggregate from clients, so they equal the federated statistics.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Edge-IIoTset (Ferrag et al., 2022), DNN-EdgeIIoT-dataset.csv
# ----------------------------------------------------------------------------
EDGE_LABEL = "Attack_type"
EDGE_DROP = [
    # identifiers of hosts, time and capture session
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "tcp.srcport", "tcp.dstport", "udp.port", "udp.stream", "tcp.ack_raw",
    "icmp.transmit_timestamp", "icmp.seq_le", "mbtcp.trans_id",
    # checksums (per-packet values that identify packets, not attacks)
    "icmp.checksum", "tcp.checksum",
    # raw payloads and free-text strings
    "http.file_data", "http.request.full_uri", "http.request.uri.query", "tcp.options",
    "tcp.payload", "mqtt.msg", "dns.qry.name",
    # binary label duplicates the target
    "Attack_label",
]
SHA256 = {"DNN-EdgeIIoT-dataset.csv": "1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7"}

# ----------------------------------------------------------------------------
# CICIoT2023 (Neto et al., 2023), official MERGED_CSV release (63 files, 45,019,243 records)
# ----------------------------------------------------------------------------
CIC_LABEL = "Label"
CIC_DROP: list[str] = []      # the merged flow-feature release contains no host, port or time identifiers
CIC_CAP = 20_000              # Deliverable 3, Step 2: majority classes capped per class, rare classes kept whole
DROP_LISTS = {"edgeiiotset": EDGE_DROP, "ciciot2023": CIC_DROP}


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@dataclass
class Clean:
    df: pd.DataFrame            # features (raw values, after cleaning)
    y: np.ndarray               # int labels
    classes: list[str]
    benign: int
    group: np.ndarray           # feature-vector group id per row
    log: dict = field(default_factory=dict)


def load_edge_iiotset(path: str | Path, verify_hash: bool = True) -> Clean:
    path = Path(path)
    log = {"file": path.name}
    if verify_hash and path.name in SHA256:
        h = sha256_of(path)
        log["sha256"] = h
        if h != SHA256[path.name]:
            raise RuntimeError(f"hash mismatch for {path.name}: {h}")
    df = pd.read_csv(path, low_memory=False)
    log["rows_raw"] = len(df)
    df = df.dropna(subset=[EDGE_LABEL])
    labels = df[EDGE_LABEL].astype(str).str.strip()
    df = df.drop(columns=[c for c in EDGE_DROP + [EDGE_LABEL] if c in df.columns])
    cl = _clean(df, labels, benign_name="Normal", log=log)
    cl.log["dataset"] = "edgeiiotset"
    return cl


def build_ciciot2023_sample(src_dir: str | Path, out_csv: str | Path, cap: int = CIC_CAP, seed: int = 0,
                            oversample: float = 3.0, chunksize: int = 1_000_000) -> dict:
    """Stratified sample of the CICIoT2023 merged CSVs (Deliverable 3, Step 2).

    Classes with at most `cap` unique records are kept whole. For larger classes a uniform
    random subset is drawn: a Bernoulli pre-sample at rate oversample*cap/N_c in one streaming
    pass, then exact duplicates are removed and `cap` unique records are drawn at random.
    The cap is therefore applied to unique records, so duplicate removal later cannot push a
    capped class below the cap.
    """
    src_dir = Path(src_dir)
    files = sorted(src_dir.glob("Merged*.csv"))
    rng = np.random.default_rng(seed)
    counts: dict = {}
    for f in files:   # pass 1: exact class counts
        for ch in pd.read_csv(f, usecols=[CIC_LABEL], chunksize=chunksize):
            for k_, v in ch[CIC_LABEL].value_counts().items():
                counts[k_] = counts.get(k_, 0) + int(v)
    rate = {c: min(1.0, oversample * cap / n) for c, n in counts.items()}
    parts = []
    for f in files:   # pass 2: Bernoulli pre-sample
        for ch in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            r = ch[CIC_LABEL].map(rate).to_numpy()
            parts.append(ch[rng.random(len(ch)) < r])
    df = pd.concat(parts, ignore_index=True)
    out = []
    for c, g in df.groupby(CIC_LABEL, sort=True):
        if counts[c] > cap:
            g = g.drop_duplicates()
            g = g.iloc[rng.permutation(len(g))[:cap]]
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False)
    info = {"source_files": len(files), "source_records": int(sum(counts.values())), "cap": cap, "seed": seed,
            "oversample": oversample, "class_counts_source": dict(sorted(counts.items())),
            "class_counts_sample": {k_: int(v) for k_, v in df[CIC_LABEL].value_counts().sort_index().items()},
            "sample_records": int(len(df)), "sample_file": out_csv.name, "sample_sha256": sha256_of(out_csv)}
    out_csv.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def load_ciciot2023(path: str | Path, expected_sha256: str | None = None) -> Clean:
    path = Path(path)
    log = {"file": path.name}
    if expected_sha256:
        h = sha256_of(path)
        log["sha256"] = h
        if h != expected_sha256:
            raise RuntimeError(f"hash mismatch for {path.name}: {h}")
    df = pd.read_csv(path, low_memory=False)
    log["rows_raw"] = len(df)
    df = df.dropna(subset=[CIC_LABEL])
    labels = df[CIC_LABEL].astype(str).str.strip()
    df = df.drop(columns=[c for c in CIC_DROP + [CIC_LABEL] if c in df.columns])
    cl = _clean(df, labels, benign_name="BENIGN", log=log)
    cl.log["dataset"] = "ciciot2023"
    return cl


def _clean(df: pd.DataFrame, labels: pd.Series, benign_name: str, log: dict) -> Clean:
    # normalise mixed-type columns: numeric only if every non-missing value parses as a
    # number; otherwise the column is categorical (text values such as HTTP methods
    # must not be silently turned into missing values)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            num = pd.to_numeric(s, errors="coerce")
            missing = s.isin(["", "nan", "NaN", "None"])
            if (num.notna() | missing).all():
                df[c] = num
            else:
                # canonical text for numeric-looking entries ("0", "0.0" -> "0")
                df[c] = np.where(num.notna(), num.map(lambda v: f"{v:g}" if pd.notna(v) else ""), s)
    df = df.replace([np.inf, -np.inf], np.nan)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    log["dropped_constant"] = const
    df[EDGE_LABEL] = labels.values
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["rows_after_dedup"] = len(df)
    log["exact_duplicates_removed"] = before - len(df)
    labels = df.pop(EDGE_LABEL)
    classes = sorted(labels.unique(), key=lambda s: (s != benign_name, s))
    cmap = {c: i for i, c in enumerate(classes)}
    y = labels.map(cmap).to_numpy(np.int64)
    # group identical feature vectors (possibly with different labels)
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    _, group = np.unique(h, return_inverse=True)
    log["feature_groups"] = int(group.max() + 1)
    log["rows_in_conflicting_groups"] = int(pd.Series(group).map(
        pd.Series(y).groupby(group).nunique()).gt(1).sum())
    log["class_counts"] = {classes[i]: int(n) for i, n in zip(*np.unique(y, return_counts=True))}
    log["columns"] = list(df.columns)
    return Clean(df=df, y=y, classes=classes, benign=0, group=group, log=log)


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
@dataclass
class Split:
    ex: np.ndarray      # row indices: exemplar pool
    test: np.ndarray
    val: np.ndarray
    train: np.ndarray   # client pool (70%)


def make_split(cl: Clean, seed: int, k: int = 50, test_frac: float = 0.2, val_frac: float = 0.1) -> Split:
    rng = np.random.default_rng(seed)
    y, g = cl.y, cl.group
    gsize = np.bincount(g)
    unique_rows = gsize[g] == 1
    ex = []
    for c in range(len(cl.classes)):
        cand = np.flatnonzero((y == c) & unique_rows)
        take = min(k, max(0, len(cand) // 4))       # never take more than a quarter of a class
        ex.extend(rng.choice(cand, size=take, replace=False).tolist())
    ex = np.array(sorted(ex), dtype=np.int64)
    taken = np.zeros(len(y), bool)
    taken[ex] = True
    # group-level stratified assignment of everything else
    rest_groups = np.unique(g[~taken])
    # label of a group = label of its first row (groups are almost always pure)
    first = pd.Series(np.arange(len(y))).groupby(g).first().to_numpy()
    glab = y[first[rest_groups]]
    assign = np.empty(len(rest_groups), dtype=np.int8)   # 0 train, 1 val, 2 test
    for c in np.unique(glab):
        idx = np.flatnonzero(glab == c)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        assign[idx[:n_test]] = 2
        assign[idx[n_test:n_test + n_val]] = 1
        assign[idx[n_test + n_val:]] = 0
    row_assign = np.full(len(y), -1, dtype=np.int8)
    gmap = np.full(g.max() + 1, -1, dtype=np.int8)
    gmap[rest_groups] = assign
    row_assign[~taken] = gmap[g[~taken]]
    return Split(ex=ex, test=np.flatnonzero(row_assign == 2), val=np.flatnonzero(row_assign == 1),
                 train=np.flatnonzero(row_assign == 0))


# ----------------------------------------------------------------------------
# Representation (fit on the client pool only)
# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    X: dict             # name -> float32 array
    y: dict             # name -> int64 array
    rows: dict          # name -> original row indices
    features: list
    classes: list
    benign: int
    meta: dict


def prepare(cl: Clean, sp: Split, max_categories: int = 20, min_category_count: int = 100,
            corr_thr: float = 0.95) -> Prepared:
    df = cl.df
    tr = df.iloc[sp.train]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if c not in cat_cols]
    blocks = {}
    names = []
    # numeric: log1p on heavy-tailed non-negative columns
    trn = tr[num_cols].astype(np.float64)
    nonneg = (trn.min() >= 0)
    heavy = nonneg & (trn.max() > 10)
    log_cols = [c for c in num_cols if heavy[c]]
    med = {}
    for c in num_cols:
        v = trn[c]
        if c in log_cols:
            v = np.log1p(v)
        med[c] = float(np.nanmedian(v)) if v.notna().any() else 0.0
    # categorical: categories fixed from the client pool
    cats = {}
    for c in cat_cols:
        vc = tr[c].value_counts()
        # rare values (e.g. one-off payload strings) fall into "other" so the model
        # cannot memorise individual attack payloads
        cats[c] = [v for v in vc.index[:max_categories] if vc[v] >= min_category_count]

    def transform(part: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in num_cols:
            v = part[c].astype(np.float64)
            if c in log_cols:
                v = np.log1p(v.clip(lower=0))
            cols.append(v.fillna(med[c]).to_numpy())
        for c in cat_cols:
            s = part[c]
            for val in cats[c]:
                cols.append((s == val).to_numpy(np.float64))
            cols.append((~s.isin(cats[c])).to_numpy(np.float64))
        return np.stack(cols, axis=1) if cols else np.zeros((len(part), 0))

    names = list(num_cols) + [f"{c}={v}" for c in cat_cols for v in cats[c]] + [f"{c}=other" for c in cat_cols]
    # reorder names to match transform()
    names = list(num_cols)
    for c in cat_cols:
        names += [f"{c}={v}" for v in cats[c]] + [f"{c}=other"]
    Xtr = transform(tr)
    # federated statistics: count, sum, sum of squares, cross-products
    n = Xtr.shape[0]
    s1 = Xtr.sum(0)
    s2 = (Xtr ** 2).sum(0)
    mean = s1 / n
    var = s2 / n - mean ** 2
    keep = var > 1e-8
    # correlation from cross-products (computed on kept columns)
    Xk = Xtr[:, keep]
    mk, sk = mean[keep], np.sqrt(var[keep])
    cross = (Xk.T @ Xk) / n
    cov = cross - np.outer(mk, mk)
    corr = cov / np.outer(sk, sk)
    kidx = np.flatnonzero(keep)
    drop = set()
    for i in range(len(kidx)):
        if kidx[i] in drop:
            continue
        for j in range(i + 1, len(kidx)):
            if kidx[j] not in drop and abs(corr[i, j]) > corr_thr:
                drop.add(kidx[j])
    final = np.array([i for i in range(len(names)) if keep[i] and i not in drop], dtype=np.int64)
    mu, sd = mean[final], np.sqrt(var[final])
    X, Y, R = {}, {}, {}
    for nm, idx in (("train", sp.train), ("val", sp.val), ("test", sp.test), ("ex", sp.ex)):
        part = df.iloc[idx]
        Z = (transform(part)[:, final] - mu) / sd
        X[nm] = Z.astype(np.float32)
        Y[nm] = cl.y[idx]
        R[nm] = idx
    meta = {
        "n_features_before_filter": len(names), "n_features": int(len(final)),
        "dropped_low_variance": [names[i] for i in range(len(names)) if not keep[i]],
        "dropped_high_corr": [names[i] for i in sorted(drop)],
        "log_transformed": log_cols, "categorical": {c: cats[c] for c in cat_cols},
        "split_sizes": {k: int(len(v)) for k, v in R.items()},
        "split_class_counts": {k: np.bincount(Y[k], minlength=len(cl.classes)).tolist() for k in Y},
    }
    return Prepared(X=X, y=Y, rows=R, features=[names[i] for i in final], classes=cl.classes,
                    benign=cl.benign, meta=meta)


def leakage_checks(cl: Clean, sp: Split, pr: Prepared) -> dict:
    """Automated checks run before every experiment (Deliverable 3, Step 3)."""
    sets = {"ex": sp.ex, "test": sp.test, "val": sp.val, "train": sp.train}
    out = {}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f"row_overlap_{a}_{b}"] = int(len(np.intersect1d(sets[a], sets[b])))
            out[f"feature_overlap_{a}_{b}"] = int(len(np.intersect1d(cl.group[sets[a]], cl.group[sets[b]])))
    out["identifier_columns_present"] = [c for c in DROP_LISTS.get(cl.log.get("dataset", "edgeiiotset"), EDGE_DROP)
                                         if c in cl.df.columns]
    out["train_feature_mean_abs_max"] = float(np.abs(pr.X["train"].mean(0)).max())
    out["passed"] = all(v == 0 for k, v in out.items() if k.startswith(("row_overlap", "feature_overlap"))) \
        and not out["identifier_columns_present"] and out["train_feature_mean_abs_max"] < 1e-3
    return out


# Target families for Edge-IIoTset, chosen by the rule in select_targets() applied to
# families a centralised reference model can learn (validation recall >= 0.8);
# see results/target_selection.json. Backdoor (0.71, confused with Ransomware) and
# Uploading (0.77) fail the learnability gate.
EDGE_TARGETS = {"rare": "MITM", "medium": "Port_Scanning", "common": "DDoS_TCP"}


# Target families per dataset (filled after the learnability gate; None = use the size rule)
# CICIoT2023: same rule (results/target_selection_ciciot2023.json); size = records in the full release.
# All web-based, brute-force and rare reconnaissance families fail the learnability gate.
CIC_TARGETS = {"rare": "DOS-HTTP_FLOOD", "medium": "MIRAI-UDPPLAIN", "common": "DDOS-UDP_FLOOD"}
DATASET_TARGETS: dict = {"edgeiiotset": EDGE_TARGETS, "ciciot2023": CIC_TARGETS}


def select_targets(cl: Clean, k: int = 50, min_test: int = 30, test_frac: float = 0.2,
                   fixed: dict | None = "auto") -> dict:
    """Three target attack families: median-sized family of the smallest, middle and
    largest third of qualifying families (expected test records >= min_test)."""
    counts = np.bincount(cl.y, minlength=len(cl.classes))
    qual = [c for c in range(len(cl.classes)) if c != cl.benign and (counts[c] - k) * test_frac >= min_test]
    qual = sorted(qual, key=lambda c: counts[c])
    thirds = np.array_split(np.array(qual), 3)
    picks = {name: int(t[len(t) // 2]) for name, t in zip(("rare", "medium", "common"), thirds)}
    if fixed == "auto":
        fixed = DATASET_TARGETS.get(cl.log.get("dataset", "edgeiiotset"))
    if fixed:
        picks = {name: cl.classes.index(fam) for name, fam in fixed.items()}
    excluded = [cl.classes[c] for c in range(len(cl.classes)) if c != cl.benign and c not in qual]
    return {"targets": picks, "target_names": {k_: cl.classes[v] for k_, v in picks.items()},
            "qualifying": [cl.classes[c] for c in qual], "excluded_from_sole_holder_tests": excluded}
