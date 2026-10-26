#!/usr/bin/env python
"""Trivial / structure-light baselines on a canonical table -> results/baselines/trivial_<bench>.csv + metrics.

All columns follow ddg convention (positive = destabilising) before any fitting:
  kd          KD(wt) - KD(mut)             loss of hydrophobicity (Kyte-Doolittle)
  vol         V(wt) - V(mut)               volume loss (cavity), Zamyatnin volumes (A^3)
  absvol      |V(wt) - V(mut)|             any size change
  blosum      -BLOSUM62(wt, mut)           dissimilarity
  burial      1 - RSA(wt site)             burial alone (RSA via freesasa; falls back to neighbour count)
  burial_kd   burial * kd                  burial-weighted hydrophobicity loss
  burial_absvol, burial_blosum             same weighting for size change / dissimilarity
  nc_kd       ncnorm * kd                  as burial_kd, burial = CB neighbours within 10 A / 20 (clipped to 1)
  scripts/32_baseline_trivial.py --bench megascale [--split test]
"""
import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from physddg import eval as E

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
          P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6,
           M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
B62 = substitution_matrices.load("BLOSUM62")
BASELINES = ["kd", "vol", "absvol", "blosum", "burial", "burial_kd", "burial_absvol", "burial_blosum", "nc_kd"]


def site_burial(path: str) -> pd.DataFrame:
    """Per-residue (chain, pdb_resnum) -> rsa (freesasa, NaN if unavailable) and CB-neighbour count (10 A)."""
    from Bio.PDB import MMCIFParser, PDBParser
    p = Path(path) if Path(path).is_absolute() else E.ROOT / path
    parser = MMCIFParser(QUIET=True) if p.suffix in (".cif", ".mmcif") else PDBParser(QUIET=True)
    res = [r for r in parser.get_structure("s", str(p))[0].get_residues() if r.id[0] == " " and "CA" in r]
    xyz = np.array([(r["CB"] if "CB" in r else r["CA"]).coord for r in res])
    nc = ((np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) < 10).sum(1) - 1)
    rsa = {}
    if p.suffix == ".pdb":
        try:
            import freesasa
            freesasa.setVerbosity(freesasa.silent)
            for ch, d in freesasa.calc(freesasa.Structure(str(p))).residueAreas().items():
                rsa.update({(ch, k): v.relativeTotal for k, v in d.items()})
        except Exception as e:  # noqa: BLE001
            print(f"freesasa failed on {p.name}: {e}", file=sys.stderr)
    rows = [(r.get_parent().id, f"{r.id[1]}{r.id[2].strip()}") for r in res]
    return pd.DataFrame({"structure": path, "chain": [c for c, _ in rows], "pdb_resnum": [n for _, n in rows],
                         "nc": nc, "rsa": [rsa.get(k, np.nan) for k in rows]})


def features(b: pd.DataFrame) -> pd.DataFrame:
    with Pool(min(32, os.cpu_count())) as pool:
        sites = pd.concat(pool.map(site_burial, b["structure"].unique().tolist()))
    b = b.assign(pdb_resnum=E._norm_resnum(b["pdb_resnum"]))
    f = b.merge(sites, on=["structure", "chain", "pdb_resnum"], how="left")
    print(f"  burial matched for {f['nc'].notna().mean():.1%} rows; freesasa RSA for {f['rsa'].notna().mean():.1%}")
    kd = f.wt.map(KD) - f.mut.map(KD)
    dv = f.wt.map(VOL) - f.mut.map(VOL)
    bl = -np.array([B62[w][m] for w, m in zip(f.wt, f.mut)], float)
    bur = (1 - f.rsa.clip(0, 1)).fillna((f.nc / 20).clip(0, 1))  # RSA-based, else neighbour-count proxy
    out = f[list(E.KEYS)].copy()
    out["kd"], out["vol"], out["absvol"], out["blosum"] = kd, dv, dv.abs(), bl
    out["burial"], out["burial_kd"], out["burial_absvol"], out["burial_blosum"] = bur, bur * kd, bur * dv.abs(), bur * bl
    out["nc_kd"] = (f.nc / 20).clip(0, 1) * kd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--split")
    ap.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()
    out_dir = E.ROOT / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in a.bench:
        b = E.load_benchmark(bench)
        print(f"{bench}: {len(b)} rows")
        feats = features(b)
        feats.to_csv(out_dir / f"trivial_{bench}.csv", index=False)
        d = b.assign(pdb_resnum=E._norm_resnum(b["pdb_resnum"])).merge(feats.drop_duplicates(list(E.KEYS)), on=list(E.KEYS))
        if a.split:
            d = d[d["split"] == a.split]
        rows = []
        for c in BASELINES:
            m = E.evaluate(d.dropna(subset=[c]), c, min_n=a.min_n)
            # sign of a raw heuristic is arbitrary in scale, so report rank metrics only
            rows.append({"baseline": c} | {k: m[k] for k in ("n", "spearman", "pearson", "spearman_prot_median", "auroc_destab")})
        tab = pd.DataFrame(rows)
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        tab.to_csv(out_dir / f"trivial_{bench}_metrics{'_' + a.split if a.split else ''}.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
