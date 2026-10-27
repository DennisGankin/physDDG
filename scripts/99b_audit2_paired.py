#!/usr/bin/env python
"""Audit 2 - independent paired cluster-bootstrap (own code, not physddg.eval).

Resamples `cluster` with replacement. Per-protein rho is computed once on the full data and the
replicate's median/mean is taken over the resampled proteins (duplicating a protein's rows does not
change its own rank correlation), which is the same estimator the project's harness uses.

  PYTHONPATH=. .venv/bin/python -u scripts/99b_audit2_paired.py [--boot 2000]   # Domainome part is slow: ~1 h
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "ladder_physics"


class Boot:
    def __init__(self, d, col, target):
        y = d["ddg"].to_numpy(float) if target == "ddg" else -d["score"].to_numpy(float)
        p = d[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        self.y, self.p = y[ok], p[ok]
        self.target = target
        prot = d["protein_id"].to_numpy()[ok]
        cl = d["cluster"].astype(str).to_numpy()[ok]
        cl = np.where((cl == "nan") | (cl == ""), prot.astype(str), cl)
        self.prot = prot
        g = pd.Series(np.arange(len(prot))).groupby(cl, sort=True)
        self.rows = {k: v.to_numpy() for k, v in g}
        self.groups = np.array(sorted(self.rows))
        rho = {}
        for pr, ix in pd.Series(np.arange(len(prot))).groupby(prot, sort=True):
            ix = ix.to_numpy()
            if len(ix) >= 10 and len(np.unique(self.p[ix])) > 1 and len(np.unique(self.y[ix])) > 1:
                rho[pr] = spearmanr(self.p[ix], self.y[ix])[0]
        self.rho = pd.Series(rho, dtype=float)
        self.gprot = {k: pd.unique(prot[v]) for k, v in self.rows.items()}

    def stats(self, groups):
        ix = np.concatenate([self.rows[g] for g in groups])
        y, p = self.y[ix], self.p[ix]
        pv = self.rho.reindex(np.concatenate([self.gprot[g] for g in groups])).to_numpy()
        pv = pv[np.isfinite(pv)]
        out = {"med_rho": float(np.median(pv)) if len(pv) else np.nan,
               "mean_rho": float(np.mean(pv)) if len(pv) else np.nan,
               "pooled_rho": float(spearmanr(p, y)[0]), "pearson": float(pearsonr(p, y)[0])}
        out["rmse"] = float(np.sqrt(np.mean((p - y) ** 2))) if self.target == "ddg" else np.nan
        return out


def paired(d, a, b, target, nboot, seed=0):
    d = d[d[a].notna() & d[b].notna()]
    A, B = Boot(d, a, target), Boot(d, b, target)
    full = A.groups
    sa, sb = A.stats(full), B.stats(full)
    d0 = {k: sa[k] - sb[k] for k in sa}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(nboot):
        g = rng.choice(full, len(full))
        x, z = A.stats(g), B.stats(g)
        reps.append({k: x[k] - z[k] for k in d0})
    r = pd.DataFrame(reps)
    return len(d), {k: (d0[k], float(r[k].quantile(.025)), float(r[k].quantile(.975))) for k in d0}


def show(tag, n, out):
    print(f"{tag}  (n={n})")
    for k, (v, lo, hi) in out.items():
        if np.isfinite(v):
            print(f"   {k:11s} {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    m = pd.read_parquet(PRED / "preds_megascale.parquet")
    nat = m[m.domain_origin == "natural"]
    for A, B in [("B6gj", "FoldX-clip"), ("B6gj", "FoldX"), ("B6gj", "B5"), ("B6gjaf", "B6gj"), ("B6g", "B5")]:
        n, o = paired(nat, A, B, "ddg", a.boot)
        show(f"MS natural  {A} - {B}", n, o)
    n, o = paired(m, "B6gj", "FoldX-clip", "ddg", a.boot)
    show("MS all      B6gj - FoldX-clip", n, o)
    dom = pd.read_parquet(PRED / "preds_domainome.parquet")
    for A, B in [("B6gj", "FoldX-clip"), ("B6gj", "B5"), ("B6gjaf", "B6gj")]:
        n, o = paired(dom, A, B, "score", min(a.boot, 500))
        show(f"Domainome   {A} - {B}", n, o)
    fp = pd.read_parquet(PRED / "preds_fireprot.parquet")
    keep = [c for c in ["B3+", "B5", "B6g", "B6gj", "FoldX", "FoldX-clip", "tmol-refit", "KORPM", "DDGun3D"]
            if fp[c].notna().mean() >= 0.95]
    rep = fp[fp[keep].notna().all(axis=1)]
    for lab, sub in (("ALL 347", fp), ("reported 338", rep)):
        for A, B in [("B6gj", "B5"), ("B6gj", "FoldX-clip")]:
            n, o = paired(sub, A, B, "ddg", a.boot)
            show(f"FireProt {lab}  {A} - {B}", n, o)


if __name__ == "__main__":
    main()
