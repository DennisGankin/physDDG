"""Derive PDB statistics for PhysDDG from the PISCES culled set (pc30, <=1.8 A, R<=0.25; scripts/pisces_io.py).

Stages (all rerunnable; `python scripts/40_build_rotlib.py [extract] [build]`, default both):
  extract: parse every chain from the local PDB mirror in parallel (48 workers), annotate (phi/psi, chis, DSSP,
           rSASA, CB counts, smooth contact number, HSE, B-factors, altlocs) and write per-residue records to
           data/cache/pdbstats_records.parquet (git-ignored).
  build:   from the 'train' split (hash of PDB id; 'test' chains are held out for validation) build
           (a) backbone-dependent rotamer library (physddg/data/rotlib_bbdep.npz),
           (b) Ramachandran potentials, (c) side-chain conformational entropies,
           (d) environment-dependent amino-acid potentials + P(aa | phi>0)  (physddg/data/pdbstats.npz).
Method notes are in physddg/rotlib.py and notes/engine_geometry.md.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pisces_io  # noqa: E402
from physddg import topology as T  # noqa: E402
from physddg.structure import annotate, load_protein  # noqa: E402

CACHE = ROOT / "data" / "cache" / "pdbstats_records.parquet"
DATA = ROOT / "physddg" / "data"


# ----------------------------------------------------------------------------- extraction
def extract_chain(job):
    pdb, chain, split = job
    path = pisces_io.structure_path(pdb)
    if path is None:
        return None
    try:
        p = annotate(load_protein(path, chains=[chain], keep_het=False))
    except Exception as e:  # noqa: BLE001
        print(f"[skip] {pdb}{chain}: {e}", file=sys.stderr)
        return None
    a = p.ann
    L = len(p)
    nat = np.array([len(T.TEMPLATES[x].atoms) for x in p.seq])
    sc_b = np.array([p.b14[i, 4:nat[i]].max(initial=0.0) for i in range(L)], np.float32)
    bb_b = p.b14[:, :4].max(1)
    nxt = np.array(list(p.seq[1:]) + ["-"])
    nxt[:-1][a["chain_break"][1:]] = "-"           # no peptide bond to the next residue
    df = pd.DataFrame({
        "pdb": pdb, "chain": chain, "split": split, "resnum": p.resnum, "aa": list(p.seq), "next_aa": nxt,
        "phi": a["phi"], "psi": a["psi"], "omega": a["omega"],
        "chi1": a["chi"][:, 0], "chi2": a["chi"][:, 1], "chi3": a["chi"][:, 2], "chi4": a["chi"][:, 3],
        "ss8": a["ss8"], "ss3": a["ss3"], "rsa": a["rsa"], "cn": a["cn"],
        "ncb8": a["ncb8"], "ncb10": a["ncb10"], "ncb12": a["ncb12"], "hse_up": a["hse_up"], "hse_down": a["hse_down"],
        "b_sc": sc_b, "b_bb": bb_b, "altloc": p.altloc, "missing_sc": p.missing_sc, "modified": p.modified,
        "occ_min": np.where(p.mask14, p.occ14, 1.0).min(1),
    })
    for c in ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4", "rsa", "cn", "b_sc", "b_bb", "occ_min"):
        df[c] = df[c].astype(np.float32)
    return df


def extract():
    rows = pisces_io.read_list(pisces_io.download())
    jobs = [(r["pdb"], r["chain"], pisces_io.split_of(r["pdb"])) for r in rows]
    t0 = time.time()
    with ProcessPoolExecutor(48) as ex:
        dfs = [d for d in ex.map(extract_chain, jobs, chunksize=4) if d is not None]
    df = pd.concat(dfs, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"extracted {len(dfs)}/{len(jobs)} chains, {len(df)} residues in {time.time() - t0:.0f}s -> {CACHE}")


# ----------------------------------------------------------------------------- build
def quality_backbone(df: pd.DataFrame) -> pd.Series:
    return (~df.altloc) & (df.b_bb < 40) & (df.occ_min >= 0.99) & df.phi.notna() & df.psi.notna() & (~df.modified)


def quality_sidechain(df: pd.DataFrame) -> pd.Series:
    return quality_backbone(df) & (~df.missing_sc) & (df.b_sc < 40)


def build():
    from physddg import rotlib as RL
    df = pd.read_parquet(CACHE)
    tr = df[df.split == "train"]
    print(f"train residues {len(tr)} from {tr[['pdb', 'chain']].drop_duplicates().shape[0]} chains")
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lib = RL.derive_rotlib(tr[quality_sidechain(tr)])
    np.savez_compressed(DATA / "rotlib_bbdep.npz", **lib)
    print(f"rotamer library: {time.time() - t0:.0f}s, {(DATA / 'rotlib_bbdep.npz').stat().st_size / 1e6:.1f} MB")
    t0 = time.time()
    stats = RL.derive_pdbstats(tr[quality_backbone(tr)], tr[quality_sidechain(tr)], lib)
    np.savez_compressed(DATA / "pdbstats.npz", **stats)
    print(f"pdb statistics: {time.time() - t0:.0f}s, {(DATA / 'pdbstats.npz').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["extract", "build"]
    if "extract" in stages:
        extract()
    if "build" in stages:
        build()
