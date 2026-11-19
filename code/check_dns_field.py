"""Count Edge-IIoTset records whose DNS query-name length field is not numeric (Section 8.5).

usage: python check_dns_field.py <DNN-EdgeIIoT-dataset.csv> <out.json>
"""
import json
import sys

import pandas as pd

src, out = sys.argv[1:3]
df = pd.read_csv(src, usecols=["dns.qry.name.len", "Attack_type"], low_memory=False, dtype=str)
v = df["dns.qry.name.len"].str.strip()
bad = pd.to_numeric(v, errors="coerce").isna() & ~v.isna() & ~v.isin(["", "nan"])
res = {"rows": int(len(df)), "non_numeric": int(bad.sum()), "share": float(bad.mean()),
       "labels_of_non_numeric": df.loc[bad, "Attack_type"].value_counts().to_dict(),
       "examples": v[bad].head(5).tolist()}
json.dump(res, open(out, "w"), indent=1)
print(res)
