#!/usr/bin/env python
"""AF-mode Q2: Domainome as the low-pLDDT stress test (no treatment applied -- characterisation only).

  PYTHONPATH=. .venv/bin/python scripts/82_domainome_af.py [--v6] [--full]

Outputs in results/af_mode/:
  dom_by_plddt.csv      per-domain Spearman vs fitness, overall and by site pLDDT band / construct mean pLDDT
  dom_models.csv        the same per predictor with a cluster bootstrap
  dom_v4_v6.csv         v4 vs v6 AF models: prediction agreement and accuracy on identical rows
  dom_crop_full.csv     construct crop vs full-length AF model: burial change, agreement, accuracy
  dom_failure_modes.csv per-site flags (physddg.afmode.flags) x accuracy / residual bias
Rows are scored against -score (fitness, higher = more stable), so a positive Spearman means the predictor
ranks destabilising variants correctly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402

PREDS = ["B3+", "B3", "B1", "s1_dF", "s2_dF", "FoldX", "rsa_only"]


def pooled_within(df: pd.DataFrame, pred: str, y: str, group: str = "protein_id") -> float:
    """Spearman after per-domain rank-centring of both variables (keeps power where few variants per domain
    fall in a stratum, at the price of mixing domains of different difficulty)."""
    d = df[[pred, y, group]].dropna()
    if len(d) < 20:
        return np.nan
    def z(s):
        r = rankdata(s)
        return (r - r.mean()) / max(len(r), 1)
    a = d.groupby(group, sort=False)[pred].transform(z)
    b = d.groupby(group, sort=False)[y].transform(z)
    return float(spearmanr(a, b)[0]) if a.nunique() > 1 else np.nan


def add_preds(d: pd.DataFrame) -> pd.DataFrame:
    fx = pd.read_parquet(C.LADDER / "preds_domainome.parquet", columns=C.KEYS + ["FoldX"])
    d = C.attach(d, C._norm(fx), on_structure=False)
    d["rsa_only"] = -d["rsa"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args()
    out = C.ensure_out()
    d = add_preds(C.domainome())
    d["plddt_band"] = C.plddt_band(d.plddt)
    d["construct_band"] = pd.cut(d.construct_plddt_mean, [0, 70, 80, 90, 101],
                                 labels=["<70", "70-80", "80-90", ">90"], right=False)
    print(f"{len(d)} variants, {d.protein_id.nunique()} domains")

    rows = []
    for key, sub in [("all", d)] + [(f"site pLDDT {b}", d[d.plddt_band == b]) for b in C.PLDDT_LABELS] \
            + [(f"construct pLDDT {b}", d[d.construct_band == b]) for b in ["<70", "70-80", "80-90", ">90"]]:
        for p in PREDS:
            if p not in sub:
                continue
            s = C.per_protein_spearman(sub, p, "mscore")
            rows.append(dict(subset=key, pred=p, n=int(sub[[p, "mscore"]].dropna().shape[0]),
                             n_dom=int(s.notna().sum()), median_rho=float(np.nanmedian(s)) if len(s) else np.nan,
                             mean_rho=float(np.nanmean(s)) if len(s) else np.nan,
                             pooled_within=pooled_within(sub, p, "mscore"),
                             sd_mscore=float(sub.mscore.std())))
    t = pd.DataFrame(rows)
    t.to_csv(out / "dom_by_plddt.csv", index=False)
    print(t.pivot_table(index="subset", columns="pred", values="pooled_within").round(3).to_string())
    print(t.pivot_table(index="subset", columns="pred", values="median_rho").round(3).to_string())

    # pLDDT effect controlled for burial (low-pLDDT sites are almost all exposed)
    d["bur_band"] = pd.cut(d.rsa, [0, 0.15, 0.35, 1.01], labels=["buried", "partial", "exposed"], right=False)
    rows = []
    for bb in ["buried", "partial", "exposed"]:
        for pb in C.PLDDT_LABELS:
            sub = d[(d.bur_band == bb) & (d.plddt_band == pb)]
            if len(sub) < 300:
                continue
            for p in PREDS:
                s_ = C.per_protein_spearman(sub, p, "mscore")
                rows.append(dict(burial=bb, plddt=pb, pred=p, n=len(sub), n_dom=int(s_.notna().sum()),
                                 median_rho=float(np.nanmedian(s_)) if len(s_) else np.nan,
                                 pooled_within=pooled_within(sub, p, "mscore"),
                                 sd_mscore=float(sub.mscore.std())))
    pd.DataFrame(rows).to_csv(out / "dom_plddt_vs_burial.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["burial", "plddt"], columns="pred",
                                         values="pooled_within").round(3).to_string())

    # per-predictor bootstrap (clusters = Pfam)
    b = C.boot_median_rho(d, [p for p in PREDS if p in d], "mscore", cluster="cluster", n_boot=a.n_boot)
    b.to_csv(out / "dom_models.csv", index=False)
    print(b.round(3).to_string(index=False))

    # ---- failure-mode flags per site
    from joblib import Parallel, delayed

    from physddg import afmode as A
    from physddg.structure import annotate, load_protein
    sites = d.drop_duplicates(["structure", "chain", "pdb_resnum"])[["structure", "chain", "pdb_resnum"]]

    def per_struct(s, keys):
        p = annotate(load_protein(ROOT / s if not Path(s).is_absolute() else s, keep_het=False))
        key = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
        idx = np.array([key.get((c, r), -1) for c, r in keys])
        ok = idx >= 0
        fl = A.flags(p, idx[ok])
        o = pd.DataFrame({k: v for k, v in fl.items()})
        o["structure"] = s
        o["chain"] = [k[0] for k, m in zip(keys, ok) if m]
        o["pdb_resnum"] = [k[1] for k, m in zip(keys, ok) if m]
        o["contact_density"] = A.contact_density(p)[idx[ok]]
        return o
    groups = {s: list(zip(g.chain, g.pdb_resnum)) for s, g in sites.groupby("structure")}
    fl = pd.concat(Parallel(n_jobs=32)(delayed(per_struct)(s, k) for s, k in groups.items()), ignore_index=True)
    fl.to_parquet(C.FEAT / "af" / "domainome_flags.parquet", index=False)
    d = d.merge(fl, on=["structure", "chain", "pdb_resnum"], how="left")

    rows = []
    for flag in ["low_plddt", "very_low_plddt", "near_low_plddt", "exposed_hydrophobic_patch", "metal_site",
                 "terminal"]:
        for val in (True, False):
            sub = d[d[flag].fillna(False) == val]
            if len(sub) < 200:
                continue
            for p in ["B3+", "s1_dF", "FoldX", "rsa_only"]:
                s = C.per_protein_spearman(sub, p, "mscore")
                rows.append(dict(flag=flag, value=val, pred=p, n=len(sub), n_dom=int(s.notna().sum()),
                                 median_rho=float(np.nanmedian(s)) if len(s) else np.nan,
                                 pooled_within=pooled_within(sub, p, "mscore"),
                                 mean_mscore=float(sub.mscore.mean()), mean_pred=float(sub[p].mean())))
    pd.DataFrame(rows).to_csv(out / "dom_failure_modes.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # zinc fingers: does the metal_site flag find them?
    znf = d.groupby("cluster").agg(metal=("metal_site", "mean"), rho=("B3+", "size")).reset_index()
    znf.to_csv(out / "dom_metal_by_family.csv", index=False)


if __name__ == "__main__":
    main()
