"""Audit (read-only): exact duplicate mutations across fitting and test sets, matched by sequence context
(site +- 7 residues of the structure sequence, wt, mut) so different numbering / PDB entries still match.
Fitting sets: Megascale-train (+val), FireProt-train(+val). Test sets: Megascale-test, FireProt-test, S669, S461,
Ssym (direct). Reports duplicate rows per pair of sets and whether they fall in the reported 'strict' subsets.
Usage: PYTHONPATH=. .venv/bin/python scripts/99_audit_dup_mutations.py
"""
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg.structure import load_protein  # noqa: E402

W = 7


@lru_cache(maxsize=None)
def prot(path, chain):
    p = load_protein(path if Path(path).is_absolute() else ROOT / path, chains=[chain], keep_het=False)
    return p, {r: i for i, r in enumerate(p.resnum)}


def context(df):
    out = []
    for (st, ch), g in df.groupby(["structure", "chain"], sort=False):
        try:
            p, idx = prot(st, ch)
        except Exception:  # noqa: BLE001
            out += [None] * len(g)
            continue
        for r in g.itertuples():
            i = idx.get(str(r.pdb_resnum))
            out.append(None if i is None else p.seq[max(0, i - W):i] + "[" + p.seq[i] + "]" + p.seq[i + 1:i + 1 + W])
    keys = pd.Series(out, index=pd.concat([g for _, g in df.groupby(["structure", "chain"], sort=False)]).index)
    return keys.reindex(df.index)


def load(name, split=None, extra=None):
    p = ROOT / "data/processed"
    d = pd.read_parquet(p / f"{name}.parquet") if (p / f"{name}.parquet").exists() else pd.read_csv(p / f"{name}.csv", low_memory=False)
    d["pdb_resnum"] = d.pdb_resnum.astype(str).str.replace(r"\.0$", "", regex=True)
    if split:
        d = d[d.split.isin(split)]
    if extra:
        d = d.query(extra)
    d = d.drop_duplicates(["structure", "chain", "pdb_resnum", "wt", "mut"]).copy()
    d["ctx"] = context(d)
    d["key"] = d.ctx + ">" + d.mut
    return d


def main():
    fit = {"megascale_train": load("megascale", ["train"]), "megascale_val": load("megascale", ["val"]),
           "fireprot_trainval": load("fireprot", ["train", "val"])}
    test = {"megascale_test": load("megascale", ["test"]), "fireprot_test": load("fireprot", ["test"]),
            "s669": load("s669"), "s461": load("s461"), "ssym_direct": load("ssym", extra="direction=='direct'")}
    for tn, t in test.items():
        print(f"{tn}: {len(t)} rows, context found {t.ctx.notna().mean():.3f}")
        strict = None
        if "homolog_in_megascale_train" in t:
            strict = (t.homolog_in_megascale_train.fillna(0) == 0)
            if tn != "megascale_test" and "homolog_in_fireprot_trainval" in t:
                strict &= (t.homolog_in_fireprot_trainval.fillna(0) == 0)
        for fn, f in fit.items():
            hit = t.key.isin(set(f.key.dropna()))
            ins = int((hit & strict).sum()) if strict is not None else -1
            prots = sorted(t[hit].protein_id.unique())
            print(f"   dup with {fn:18s}: {int(hit.sum()):4d} rows ({hit.mean():.1%}), in strict subset: {ins}; proteins {prots[:8]}")


if __name__ == "__main__":
    main()
