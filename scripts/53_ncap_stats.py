#!/usr/bin/env python
"""N-cap preference statistical potential from the PISCES culled set -> physddg/data/ncap_pref.json.

Replaces the hand-typed N-cap scale in physddg/features_extra.py (audit of the residual analysis: the literature
table of Doig & Baldwin 1995 could not be retrieved to verify numbers, so no numbers are typed from memory).
Method (Richardson & Richardson 1988, Science 240:1648 — statistical capping preferences in proteins):
  N-cap = residue i that is not DSSP 'H' while i+1 is the first 'H' of a helix segment, peptide-bonded to i+1
  (the same definition as features_extra.site_descriptors 'ncap'); DSSP = physddg.structure.dssp (E1).
  E_ncap(aa) = -ln[ P(aa | N-cap) / P(aa) ]  in kT (nats), with a pseudo-count of 50 x P(aa) (as for the env
  potential, physddg/rotlib.py ENV_ALPHA); negative = favoured at the N-cap.
Data: data/cache/pdbstats_records.parquet (scripts/40_build_rotlib.py extract), split == 'train' only (D3).
Sanity check against the literature ranking (Doig & Baldwin 1995 Protein Sci 4:1325, abstract: Asn, Asp, Ser,
Thr, Cys highest N-cap preference, Gly favoured; Richardson & Richardson 1988: Ser, Thr, Asp, Asn, Gly most
common N-caps) is printed and stored.
  PYTHONPATH=. .venv/bin/python scripts/53_ncap_stats.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg.topology import AA1  # noqa: E402

ALPHA = 50.0
OUT = ROOT / "physddg" / "data" / "ncap_pref.json"


def main():
    d = pd.read_parquet(ROOT / "data" / "cache" / "pdbstats_records.parquet",
                        columns=["pdb", "chain", "split", "aa", "next_aa", "ss8"])
    d = d[(d.split == "train") & d.aa.isin(list(AA1))].reset_index(drop=True)
    h = (d.ss8 == "H").to_numpy()
    same = ((d.pdb == d.pdb.shift(-1)) & (d.chain == d.chain.shift(-1))).to_numpy()
    bonded = same & (d.next_aa != "-").to_numpy()
    starts_next = np.r_[h[1:] & ~h[:-1], False]            # residue i+1 starts a helix segment
    ncap = starts_next & bonded & ~h
    p = d.aa.value_counts(normalize=True).reindex(list(AA1)).fillna(0)
    n = d.aa[ncap].value_counts().reindex(list(AA1)).fillna(0)
    N = float(ncap.sum())
    P = (n + ALPHA * p) / (N + ALPHA)
    E = -np.log(P / p)
    rank = E.sort_values().index.tolist()
    top = set(rank[:5])
    lit_top = {"N", "D", "S", "T"}
    check = {"lit_favoured_SDNT_all_in_top5": lit_top <= top, "rank_best_to_worst": "".join(rank)}
    out = {"energy_kT": {a: round(float(E[a]), 4) for a in AA1}, "counts_ncap": {a: int(n[a]) for a in AA1},
           "p_aa": {a: round(float(p[a]), 6) for a in AA1}, "n_ncap": int(N), "n_residues": int(len(d)),
           "n_chains": int(d[["pdb", "chain"]].drop_duplicates().shape[0]), "pseudocount_alpha": ALPHA,
           "units": "kT (nats); negative = favoured at the N-cap", "definition": __doc__.split("Method")[1].split("Data:")[0].strip(),
           "source": "PISCES pc30 <=1.8 A train split, data/cache/pdbstats_records.parquet", "literature_check": check}
    OUT.write_text(json.dumps(out, indent=1))
    print(E.sort_values().round(2).to_string())
    print(check, "->", OUT)


if __name__ == "__main__":
    main()
