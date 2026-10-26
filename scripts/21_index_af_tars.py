#!/usr/bin/env python
"""Index member offsets of the (uncompressed) AlphaFold DB proteome tars so single
entries can be read by seek without untarring.

Output: data/cache/af_tar_index/<tarname>.tsv  (member, offset_data, size)
Only .pdb.gz members are indexed (cif skipped). Rerunnable; skips existing indices
unless --force.
"""
import argparse, os, sys, tarfile
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/cache/af_tar_index")
DEFAULT_TARS = [
    "/af_data/UP000005640_9606_HUMAN_v4.tar",
    "/alphafold/v6/UP000005640_9606_HUMAN_v6.tar",
    "/alphafold/v6/UP000002311_559292_YEAST_v6.tar",
    "/alphafold/v6/UP000000625_83333_ECOLI_v6.tar",
    "/alphafold/v6/UP000000589_10090_MOUSE_v6.tar",
    "/alphafold/v6/swissprot_pdb_v6.tar",
]


def index_tar(path, force=False):
    out = os.path.join(OUT, os.path.basename(path).replace(".tar", ".tsv"))
    if os.path.exists(out) and not force:
        return out, "exists"
    n = 0
    tmp = out + ".tmp"
    with tarfile.open(path, "r:") as tf, open(tmp, "w") as fh:
        fh.write("member\toffset\tsize\n")
        for m in tf:
            if m.isfile() and m.name.endswith(".pdb.gz"):
                fh.write(f"{m.name}\t{m.offset_data}\t{m.size}\n")
                n += 1
            tf.members = []  # keep memory flat
    os.replace(tmp, out)
    return out, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tars", nargs="*", default=DEFAULT_TARS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with Pool(len(a.tars)) as p:
        for out, n in p.starmap(index_tar, [(t, a.force) for t in a.tars]):
            print(out, n, flush=True)
