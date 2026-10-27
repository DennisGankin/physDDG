#!/usr/bin/env python
"""Audit 2 - independent re-computation of the model-card headline numbers.

Reads ONLY results/ladder_physics/preds_*.parquet and data/processed/*, never the reporting
scripts' outputs. Own metric code (scipy only). Prints a table; writes nothing.

  PYTHONPATH=. .venv/bin/python scripts/99b_audit2_metrics.py [--boot N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "ladder_physics"
PROC = ROOT / "data" / "processed"

MODELS = ["B6gj", "B6gjs1", "B6gjaf", "B6g", "B5", "FoldX", "FoldX-clip"]
JOINT = {"B6gj", "B6gjs1", "B6gjaf", "B4j", "B5j", "B4j+C", "B5j+C", "tmol-refit", "KORPM", "DDGun3D"}


# ------------------------------------------------------------------ metrics (own implementation)
def prot_rho(y, p, prot, min_n=10):
    d = pd.DataFrame({"y": y, "p": p, "g": prot})
    out = {}
    for g, s in d.groupby("g", sort=True):
        if len(s) >= min_n and s.p.nunique() > 1 and s.y.nunique() > 1:
            out[g] = spearmanr(s.p.to_numpy(), s.y.to_numpy())[0]
    return pd.Series(out, dtype=float)


def metrics(df, col, target):
    y = df["ddg"].to_numpy(float) if target == "ddg" else -df["score"].to_numpy(float)
    p = df[col].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, prot = y[ok], p[ok], df["protein_id"].to_numpy()[ok]
    r = prot_rho(y, p, prot)
    out = {"n": int(ok.sum()), "n_prot": len(np.unique(prot)), "n_prot_eval": int(r.notna().sum()),
           "med_rho": float(r.median()), "mean_rho": float(r.mean()),
           "pooled_rho": float(spearmanr(p, y)[0]), "pearson": float(pearsonr(p, y)[0])}
    out["rmse"] = float(np.sqrt(np.mean((p - y) ** 2))) if target == "ddg" else np.nan
    return out


# ------------------------------------------------------------------ data
def load(bench):
    b = pd.read_parquet(PRED / f"preds_{bench}.parquet")
    hp = {"megascale": PROC / "megascale.parquet", "domainome": PROC / "homology_domainome.csv"}
    src = hp.get(bench, PROC / f"{bench}.csv")
    if src.exists():
        t = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
        hc = [c for c in t.columns if c.startswith("homolog_in_")]
        if hc:
            key = ["protein_id"]
            t = t[key + hc].drop_duplicates(key)
            b = b.merge(t, on=key, how="left")
    if bench == "domainome":
        x = pd.read_parquet(ROOT / "data" / "features" / "extra_sites_domainome.parquet")
        x = x[["structure", "chain", "pdb_resnum", "dist_term"]].drop_duplicates(["structure", "chain", "pdb_resnum"])
        x["pdb_resnum"] = x["pdb_resnum"].astype(str)
        b["pdb_resnum"] = b["pdb_resnum"].astype(str)
        b["chain"] = b["chain"].astype(str)
        x["chain"] = x["chain"].astype(str)
        b = b.merge(x, on=["structure", "chain", "pdb_resnum"], how="left")
    return b


def subset(b, bench, view, model):
    d = b
    strict_cols = (["homolog_in_megascale_train", "homolog_in_fireprot_trainval"] if model in JOINT
                   else ["homolog_in_megascale_train"])
    if bench == "megascale":
        if view.startswith("nat"):
            d = d[d.domain_origin == "natural"]
        if "strict" in view:
            for c in strict_cols:
                d = d[d[c].fillna(0) == 0]
    elif bench in ("s669", "s461") and view == "strict":
        for c in strict_cols:
            d = d[d[c].fillna(0) == 0]
    elif bench == "domainome" and view == "flagkept":
        d = d[(d.plddt >= 70) & (d.dist_term >= 3)]
    return d


VIEWS = [("megascale", "nat", "MS nat"), ("megascale", "nat-strict", "MS nat-strict"),
         ("megascale", "all", "MS all"), ("fireprot", "all", "FireProt"),
         ("s669", "strict", "S669-str"), ("s461", "strict", "S461-str"),
         ("domainome", "all", "Domainome"), ("domainome", "flagkept", "Dom flag-kept"),
         ("abundance_dms", "all", "abundance")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for bench, view, label in VIEWS:
        b = load(bench)
        target = "ddg" if b.ddg.notna().any() else "score"
        have = [m for m in MODELS if m in b.columns and b[m].notna().any()]
        # identical rows: all models present with >=95% coverage must be non-null
        keep = [m for m in have if b[m].notna().mean() >= 0.95]
        for m in have:
            d = subset(b, bench, view, m)
            d = d[d[keep].notna().all(axis=1)] if keep else d
            r = metrics(d, m, target)
            rows.append({"view": label, "model": m, **r})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for label in t.view.unique():
        print(f"\n### {label}")
        print(t[t.view == label].drop(columns="view").to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
