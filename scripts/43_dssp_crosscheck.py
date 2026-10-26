"""Cross-check physddg.structure.dssp (Kabsch-Sander re-implementation, 8-state) against pydssp (3-state H/E/-)
on held-out PISCES test chains with complete, unbroken backbones. Output: results/geometry/dssp_crosscheck.csv
(contingency of our 8-state vs pydssp 3-state). pydssp has no 3-10 helix / isolated-bridge classes, so our G and
B map onto its H and E; agreement is judged on H<->H and E<->E.
Usage: python scripts/43_dssp_crosscheck.py [n_chains=150]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pisces_io  # noqa: E402
import pydssp  # noqa: E402
from physddg.structure import breaks, dssp, load_protein  # noqa: E402


def main(n: int = 150):
    rows = [r for r in pisces_io.read_list(pisces_io.download()) if pisces_io.split_of(r["pdb"]) == "test"][:n]
    c = collections.Counter()
    for r in rows:
        p = load_protein(pisces_io.structure_path(r["pdb"]), chains=[r["chain"]], keep_het=False)
        if (~p.mask14[:, :4].all(1)).any() or breaks(p)[1:].any():
            continue
        ref = pydssp.assign(p.xyz14[:, :4].astype(np.float32), out_type="c3")
        c.update(zip(dssp(p)["ss8"], ref))
    tab = pd.Series(c).unstack(fill_value=0)
    out = ROOT / "results" / "geometry"
    tab.to_csv(out / "dssp_crosscheck.csv")
    print(tab)
    print(f"our H also H in pydssp: {tab.loc['H', 'H'] / tab.loc['H'].sum():.4f}; "
          f"our E also E: {tab.loc['E', 'E'] / tab.loc['E'].sum():.4f}")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
