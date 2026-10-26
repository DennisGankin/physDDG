#!/usr/bin/env python
"""Rotamer recovery with the packing energy (structural calibration diagnostic, no ddG data).

Held-out PISCES 'test' chains (not used for the rotamer library). For every well-ordered residue (no altloc, complete
side chain, side-chain B < 40) the WT side chain is re-predicted from library rotamers only (no native candidate):
  stage 1: lowest-U rotamer (U = unit-weight packing energy + rotamer prior) in the fixed native environment
           ("one residue at a time" repacking);
  stage 2: highest mean-field probability with flexible neighbours (neighbours may keep their native side chain);
  prior:   most probable library rotamer at the residue's (phi, psi) (no energy; baseline).
Success: |dchi1| <= 40 deg (chi1) and |dchi1|, |dchi2| <= 40 deg (chi1+2); 2-fold symmetric terminal chis modulo 180.
Reported by aa and burial (rSASA < 0.25 buried). Output: results/mutate/rotamer_recovery.csv (+ _summary.csv).
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/56_rotamer_recovery.py [n_chains=60] [stage2=1]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pisces_io  # noqa: E402
from physddg import topology as T  # noqa: E402
from physddg.mutate import MutateConfig, SiteScan, library  # noqa: E402
from physddg.structure import annotate, load_protein  # noqa: E402


def angdiff(a, b, per):
    return np.abs((a - b + per / 2) % per - per / 2)


def main(n_chains: int = 60, do_s2: int = 1):
    import torch
    rows = [r for r in pisces_io.read_list(pisces_io.download()) if pisces_io.split_of(r["pdb"]) == "test"]
    rows = [r for r in rows if r["length"] <= 300][:n_chains]
    cfg = MutateConfig()
    lib = library()
    recs, times = [], []
    for r in rows:
        p = annotate(load_protein(pisces_io.structure_path(r["pdb"]), chains=[r["chain"]], keep_het=False))
        nat = np.array([len(T.TEMPLATES[a].atoms) for a in p.seq])
        bsc = np.array([p.b14[i, 4:nat[i]].max(initial=0) for i in range(len(p))])
        good = (~p.altloc) & (~p.missing_sc) & (bsc < 40) & np.isfinite(p.ann["phi"]) & np.isfinite(p.ann["psi"]) \
            & np.array([T.N_CHI[a] > 0 for a in p.seq])
        sites = np.flatnonzero(good)
        if len(sites) == 0:
            continue
        sc = SiteScan(p, cfg, "cuda")
        t0 = time.time()
        s1 = sc.stage1(sites)
        t1 = time.time() - t0
        s2 = sc.stage2(sites) if do_s2 else None
        times.append((r["pdb"], len(p), len(sites), t1, time.time() - t0 - t1))
        wt = p.aa[sites]
        ix = np.arange(len(sites))
        for k, i in enumerate(sites):
            a = p.seq[i]
            nchi = T.N_CHI[a]
            per = np.array([180.0 if (a, j + 1) in T.CHI_SYMMETRIC else 360.0 for j in range(nchi)])
            native = p.ann["chi"][i, :nchi]
            q = lib.query(a, [p.ann["phi"][i]], [p.ann["psi"][i]], max_r=1)
            preds = {"prior": q["chis"][0, 0, :nchi], "stage1": s1["chi_best"][k, wt[k], :nchi]}
            if s2 is not None:
                preds["stage2"] = s2["chi_best"][k, wt[k], :nchi]
            rec = {"pdb": r["pdb"], "resnum": p.resnum[i], "aa": a, "rsa": p.ann["rsa"][i],
                   "buried": p.ann["rsa"][i] < 0.25}
            for m, ch in preds.items():
                d = angdiff(ch, native, per)
                rec[f"{m}_chi1"] = bool(d[0] <= 40)
                rec[f"{m}_chi12"] = bool((d[:2] <= 40).all()) if nchi >= 2 else np.nan
            recs.append(rec)
        del ix
    df = pd.DataFrame(recs)
    out = ROOT / "results" / "mutate"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rotamer_recovery.csv", index=False)
    cols = [c for c in df.columns if c.endswith(("_chi1", "_chi12"))]
    summ = []
    for name, g in [("all", df), ("buried", df[df.buried]), ("exposed", df[~df.buried])]:
        summ.append({"group": name, "n": len(g), **{c: g[c].astype(float).mean() for c in cols}})
    for a, g in df.groupby("aa"):
        summ.append({"group": f"aa_{a}", "n": len(g), **{c: g[c].astype(float).mean() for c in cols}})
    for a, g in df[df.buried].groupby("aa"):
        summ.append({"group": f"buried_{a}", "n": len(g), **{c: g[c].astype(float).mean() for c in cols}})
    s = pd.DataFrame(summ)
    s.round(4).to_csv(out / "rotamer_recovery_summary.csv", index=False)
    tm = pd.DataFrame(times, columns=["pdb", "L", "n_sites", "stage1_s", "stage2_s"])
    tm.to_csv(out / "rotamer_recovery_timing.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())
    print(tm.describe().round(3))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
