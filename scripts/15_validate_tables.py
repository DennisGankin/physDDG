#!/usr/bin/env python
"""Validate all data-HT tables: schema, 100% WT identity at (structure, chain, pdb_resnum), sign sanity, splits,
Ssym pairing. Exits non-zero on any failure. Run after scripts 11-14.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ht_common import ROOT, chain_residues, read_structure  # noqa: E402

PROC = ROOT / "data/processed"
REQ = ["dataset", "protein_id", "structure", "chain", "pdb_resnum", "wt", "mut", "ddg", "split", "cluster"]
TABLES = {"megascale": "megascale.parquet", "megascale_censored": "megascale_censored.csv", "fireprot": "fireprot.csv",
          "fireprot_dtm": "fireprot_dtm.csv", "s669": "s669.csv", "s461": "s461.csv", "ssym": "ssym.csv",
          "s2648": "s2648.csv"}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def main():
    fails = []
    cache = {}
    for name, fn in TABLES.items():
        p = PROC / fn
        df = pd.read_parquet(p) if fn.endswith(".parquet") else pd.read_csv(p, dtype={"pdb_resnum": str, "chain": str})
        miss = [c for c in REQ if c not in df.columns]
        if miss:
            fails.append(f"{name}: missing columns {miss}")
        if not (df.wt.isin(AA) & df.mut.isin(AA)).all():
            fails.append(f"{name}: non-standard amino acids")
        if (df.wt == df.mut).any():
            fails.append(f"{name}: synonymous rows")
        if df.cluster.isna().any() or (df.cluster == "").any():
            fails.append(f"{name}: empty cluster")
        bad = 0
        for (s, c), g in df.groupby(["structure", "chain"]):
            if (s, c) not in cache:
                path = Path(s) if s.startswith("/") else ROOT / s
                cache[(s, c)] = dict(chain_residues(read_structure(path), c))
            res = cache[(s, c)]
            bad += sum(res.get(r) != w for r, w in zip(g.pdb_resnum.astype(str), g.wt))
        if bad:
            fails.append(f"{name}: {bad} rows with WT != structure residue")
        key = ["protein_id", "chain", "pdb_resnum", "wt", "mut"] + (["direction"] if "direction" in df else [])
        dup = df.duplicated(key).sum()
        if dup:
            fails.append(f"{name}: {dup} duplicate mutation rows")
        d = df[df.ddg.notna()]
        msg = (f"{name:18s} rows={len(df):7d} proteins={df.protein_id.nunique():4d} WT-identity={1 - bad / len(df):.4%} "
               f"splits={df.split.value_counts().to_dict()}")
        if len(d):
            dd = d[d.get("direction", pd.Series("direct", index=d.index)).fillna("direct") == "direct"]
            msg += f" mean_ddg(direct)={dd.ddg.mean():+.2f}"
            if dd.ddg.mean() <= 0:
                fails.append(f"{name}: mean ddg <= 0 (sign?)")
        print(msg)
        if name == "ssym":
            pv = df.pivot_table(index="pair_id", columns="direction", values="ddg")
            if not ((pv.direct + pv.reverse).abs() < 1e-9).all() or pv.isna().any().any():
                fails.append("ssym: direct/reverse not antisymmetric or incomplete")
    for f in fails:
        print("FAIL:", f)
    print("ALL OK" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
