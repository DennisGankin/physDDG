#!/usr/bin/env python
"""Raw (unit-weight, unfitted) Stage-1 / Stage-2 dF vs measured ddG: a first sanity number, no fitting.
Megascale val/test (pooled Spearman, per-protein median Spearman, natural-domain subset) and FireProt-test
(Spearman, Pearson). Output: results/mutate/sanity_raw_dF.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
K = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]


def load(stage, name):
    f = pd.read_parquet(ROOT / f"data/features/phys_s{stage}_{name}.parquet")
    if name == "megascale":
        t = pd.read_parquet(ROOT / "data/processed/megascale.parquet", columns=K + ["ddg", "split", "domain_origin"])
    else:
        t = pd.read_csv(ROOT / f"data/processed/{name}.csv", low_memory=False)[K + ["ddg", "split"]]
    t = t.assign(pdb_resnum=t.pdb_resnum.astype(str), chain=t.chain.astype(str)).drop_duplicates(K)
    return f.merge(t, on=K)


rows = []
for stage in (1, 2):
    col = f"s{stage}_dF"
    m = load(stage, "megascale")
    for split in ("val", "test"):
        g = m[(m.split == split) & m.ddg.notna() & m[col].notna()]
        per = lambda d: float(np.nanmedian([spearmanr(x[col], x.ddg).statistic for _, x in d.groupby("protein_id") if len(x) >= 20]))
        nat = g[g.domain_origin.astype(str).str.lower().str.startswith("nat")] if "domain_origin" in g else g
        rows.append(dict(stage=stage, dataset="megascale", split=split, n=len(g), spearman=spearmanr(g[col], g.ddg).statistic,
                         pearson=pearsonr(g[col], g.ddg).statistic, spearman_prot_median=per(g),
                         spearman_prot_median_natural=per(nat) if len(nat) else np.nan))
    f = load(stage, "fireprot")
    g = f[(f.split == "test") & f.ddg.notna() & f[col].notna()]
    rows.append(dict(stage=stage, dataset="fireprot", split="test", n=len(g), spearman=spearmanr(g[col], g.ddg).statistic,
                     pearson=pearsonr(g[col], g.ddg).statistic))
df = pd.DataFrame(rows)
df.round(4).to_csv(ROOT / "results/mutate/sanity_raw_dF.csv", index=False)
print(df.round(3).to_string())

# Ssym antisymmetry of raw dF (direct on WT structure vs reverse on mutant structure)
anti = []
for stage in (1, 2):
    col = f"s{stage}_dF"
    f = pd.read_parquet(ROOT / f"data/features/phys_s{stage}_ssym.parquet")
    t = pd.read_csv(ROOT / "data/processed/ssym.csv")[K + ["structure", "direction", "pair_id"]]
    t = t.assign(pdb_resnum=t.pdb_resnum.astype(str), chain=t.chain.astype(str))
    m = f.merge(t, on=K + ["structure"])
    w = m.pivot_table(index="pair_id", columns="direction", values=col)
    w = w.dropna()
    anti.append(dict(stage=stage, n_pairs=len(w), pearson_dir_vs_minus_rev=pearsonr(w["direct"], -w["reverse"]).statistic,
                     mean_bias_kcal=float((w["direct"] + w["reverse"]).mean())))
a = pd.DataFrame(anti)
a.round(4).to_csv(ROOT / "results/mutate/sanity_ssym_antisymmetry.csv", index=False)
print(a.round(3).to_string())
