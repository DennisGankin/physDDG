"""Shared helpers for the classical-competitor runs (KORPM, DDGun): benchmark loading, PDB sanitising."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import eval as E  # noqa: E402

KEYS = list(E.KEYS)
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "MSE": "M"}
BENCHES = ["megascale", "fireprot", "s669", "s461", "ssym", "domainome"]


def load_bench(name, domainome_n=None, seed=0):
    """Benchmark rows to score. megascale/fireprot: split==test. domainome: optional subset of n constructs."""
    df = E.load_benchmark(name, ROOT)
    if name in ("megascale", "fireprot"):
        df = df[df["split"].astype(str) == "test"]
    if name == "domainome" and domainome_n:
        ids = sorted(df.protein_id.unique())
        pick = pd.Series(ids).sample(n=min(domainome_n, len(ids)), random_state=seed).tolist()
        df = df[df.protein_id.isin(pick)]
    df = df.copy()
    df["pdb_resnum"] = df["pdb_resnum"].astype(str)
    return df.reset_index(drop=True)


def abs_structure(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_pdb_residues(path):
    """Ordered list of (chain, resnum_str_with_icode, resname) for ATOM records (first occurrence)."""
    seen, out = set(), []
    for line in open(path):
        if line.startswith("ATOM"):
            key = (line[21], line[22:26].strip() + line[26].strip())
            if key not in seen:
                seen.add(key)
                out.append((key[0], key[1], line[17:20].strip()))
    return out


def sanitise_pdb(src, dst, require=("N", "CA", "C", "O")):
    """Write ATOM-only copy of src to dst. Residues with insertion codes get unique integer numbers (original numbers
    kept where possible, gaps preserved). Residues lacking any backbone atom in `require` are dropped (KORPM aborts
    on missing backbone atoms). Returns map {(chain, 'resnum+icode'): new_int} for kept residues."""
    res, order = {}, []
    for line in open(src):
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:26].strip() + line[26].strip())
        if key not in res:
            res[key] = {"lines": [], "atoms": set(), "orig": int(line[22:26])}
            order.append(key)
        res[key]["lines"].append(line)
        res[key]["atoms"].add(line[12:16].strip())
    mp, last, out = {}, {}, []
    for key in order:
        r = res[key]
        if not set(require) <= r["atoms"]:
            continue
        ch = key[0]
        new = r["orig"] if ch not in last else max(r["orig"], last[ch] + 1)
        mp[key] = new
        last[ch] = new
        out += [l[:22] + f"{new:4d} " + l[27:] for l in r["lines"]]
    Path(dst).write_text("".join(out) + "END\n")
    return mp


def structure_seq_map(path):
    return {(c, r): AA3.get(n, "X") for c, r, n in read_pdb_residues(path)}
