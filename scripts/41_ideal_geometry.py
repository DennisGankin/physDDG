"""Measure side-chain internal coordinates (bond, angle, torsion offset) in the culled high-resolution PDB set.

For every IC_SPEC entry (and CB) collects bond |CD|, angle BCD and torsion ABCD (minus chi_k for chi-dependent
atoms) over well-ordered residues (no altloc, complete side chain, all side-chain B < 30) in up to N train chains,
and prints medians next to the values currently in physddg.topology.IDEAL_IC (Engh & Huber based).
Output: results/geometry/ideal_ic_measured.csv
Usage: python scripts/41_ideal_geometry.py [n_chains=1500]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pisces_io  # noqa: E402
from physddg import topology as T  # noqa: E402
from physddg.structure import dihedral, load_protein  # noqa: E402


def _angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))


def measure(job):
    pdb, chain = job
    path = pisces_io.structure_path(pdb)
    if path is None:
        return []
    try:
        p = load_protein(path, chains=[chain], keep_het=False)
    except Exception:
        return []
    out = []
    for a in T.AA1:
        if a == "G":
            continue
        t = T.TEMPLATES[a]
        nat = len(t.atoms)
        rows = np.flatnonzero((p.aa == T.AA_INDEX[a]) & ~p.altloc & p.mask14[:, :nat].all(1)
                              & (p.b14[:, 4:nat].max(1, initial=0) < 30))
        if len(rows) == 0:
            continue
        X = p.xyz14[rows]
        g = lambda nm: X[:, t.index(nm)]
        chis = [dihedral(*(g(n) for n in q)) for q in t.chis]
        for atom, A, B, C, k in [T.CB_SPEC] + T.IC_SPEC[a]:
            bond = np.linalg.norm(g(atom) - g(C), axis=-1)
            ang = _angle(g(B), g(C), g(atom))
            tor = dihedral(g(A), g(B), g(C), g(atom))
            if k:
                tor = (tor - chis[k - 1] + 180) % 360 - 180
            for b_, a_, t_ in zip(bond, ang, tor):
                out.append((a, atom, b_, a_, t_))
    return out


def main(n_chains: int = 1500):
    chains = [(r["pdb"], r["chain"]) for r in pisces_io.read_list(pisces_io.download())
              if pisces_io.split_of(r["pdb"]) == "train"][:n_chains]
    with ProcessPoolExecutor(48) as ex:
        recs = [x for res in ex.map(measure, chains, chunksize=8) for x in res]
    df = pd.DataFrame(recs, columns=["aa", "atom", "bond", "angle", "torsion"])

    def circ_med(x):
        m = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(x)))))
        return m + np.median((x - m + 180) % 360 - 180)

    agg = df.groupby(["aa", "atom"], sort=False).agg(
        n=("bond", "size"), bond=("bond", "median"), bond_sd=("bond", "std"), angle=("angle", "median"),
        angle_sd=("angle", "std"), torsion=("torsion", circ_med),
        torsion_sd=("torsion", lambda x: np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * np.radians(x)))))))))
    agg = agg.reset_index()
    cur = getattr(T, "IDEAL_IC", {})
    for col in ("bond", "angle", "torsion"):
        agg[f"{col}_table"] = [cur.get(a, {}).get(at, (np.nan,) * 3)[("bond", "angle", "torsion").index(col)]
                               for a, at in zip(agg.aa, agg.atom)]
    out = ROOT / "results" / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    agg.round(4).to_csv(out / "ideal_ic_measured.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
