#!/usr/bin/env python
"""Overlap between KORPM's training set (Id25c03_1merNCLB.txt: ThermoMutDB+ProTherm, 2371 mutations / 181 PDB entries; the
default korp6Dv1 weights were fitted on it) and each benchmark.

Per benchmark protein: homolog_in_korpm_train = max fraction identity of a qualifying mmseqs hit to a KORPM training
protein (same definition as scripts/14_homology.py: E<=1e-3, fident>=0.25, alignment >= 50% of the shorter sequence).
Per variant: exact_in_korpm_train = same PDB entry + chain + position + WT + mutant as a training mutation.
Writes results/competitors/korpm_leakage_<bench>.csv (protein-level) and data/work/leak/<bench>_ann.csv (row-level
columns used by 73_score_competitors.py via a temporary data root) and prints a summary.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comp_common import AA3, BENCHES, ROOT, abs_structure, load_bench, read_pdb_residues  # noqa: E402

KDIR = ROOT / "data/raw/tools/korpm"
MMSEQS = ROOT / "data/raw/tools/mmseqs/bin/mmseqs"
WORK = ROOT / "data/work/leak"
OUT = ROOT / "results/competitors"


def struct_seq(path):
    return "".join(AA3.get(n, "X") for _, _, n in read_pdb_residues(path))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(KDIR / "Id25c03_1merNCLB.txt", sep=r"\s+", header=None, usecols=[0, 1, 2], names=["pdb", "mut", "ddg"])
    tr["pdb"] = tr.pdb.str.upper()
    m = tr.mut.str.extract(r"^([A-Z])(.)(-?\d+[A-Za-z]?)([A-Z])$")
    tr["key"] = tr.pdb + "|" + m[1] + "|" + m[2] + "|" + m[0] + "|" + m[3]
    # training sequences
    seqs, name = {}, None
    for l in open(KDIR / "Db/all.fasta"):
        l = l.strip()
        if l.startswith(">"):
            name = l[1:].strip().split("|")[-1].split("_")[0].upper()
            seqs[name] = ""
        elif name:
            seqs[name] += l
    train_pdbs = set(tr.pdb)
    with open(WORK / "train.fasta", "w") as f:
        for p in sorted(train_pdbs):
            if p in seqs:
                f.write(f">{p}\n{seqs[p]}\n")
    print(f"training: {len(tr)} mutations, {len(train_pdbs)} PDB entries, {len(set(train_pdbs) & set(seqs))} with sequence")

    for b in BENCHES:
        df = load_bench(b)
        df = df[df.structure.notna()]
        prot = df.drop_duplicates("protein_id")[["protein_id", "structure"]]
        with open(WORK / f"{b}_q.fasta", "w") as f:
            for pid, st in zip(prot.protein_id, prot.structure):
                f.write(f">{pid}\n{struct_seq(abs_structure(st))}\n")
        hits = WORK / f"{b}_hits.tsv"
        subprocess.run([str(MMSEQS), "easy-search", str(WORK / f"{b}_q.fasta"), str(WORK / "train.fasta"), str(hits),
                        str(WORK / f"tmp_{b}"), "-s", "7.5", "-e", "1e-3", "--threads", "32", "-v", "1",
                        "--format-output", "query,target,fident,alnlen,qlen,tlen,evalue"], check=True)
        h = pd.read_csv(hits, sep="\t", names=["q", "t", "fident", "aln", "qlen", "tlen", "e"])
        h = h[(h.fident >= 0.25) & (h.aln >= 0.5 * h[["qlen", "tlen"]].min(axis=1))]
        best = h.sort_values("fident", ascending=False).drop_duplicates("q").set_index("q")
        prot = prot.assign(homolog_in_korpm_train=prot.protein_id.map(best.fident).fillna(0.0),
                           korpm_train_best_hit=prot.protein_id.map(best.t))
        # exact variant overlap (PDB numbering; protein_id begins with the 4-letter PDB id for fireprot/s669/s461/ssym)
        if b in ("fireprot", "s669", "s461", "ssym"):
            key = df.protein_id.str[:4].str.upper() + "|" + df.chain + "|" + df.pdb_resnum + "|" + df.wt + "|" + df.mut
            df["exact_in_korpm_train"] = key.isin(set(tr.key)).astype(int)
            df["pdb_in_korpm_train"] = df.protein_id.str[:4].str.upper().isin(train_pdbs).astype(int)
        else:
            df["exact_in_korpm_train"] = 0
            df["pdb_in_korpm_train"] = 0
        df = df.merge(prot[["protein_id", "homolog_in_korpm_train", "korpm_train_best_hit"]], on="protein_id", how="left")
        prot.to_csv(OUT / f"korpm_leakage_{b}.csv", index=False)
        df.to_csv(WORK / f"{b}_ann.csv", index=False,
                  columns=["protein_id", "chain", "pdb_resnum", "wt", "mut", "homolog_in_korpm_train", "exact_in_korpm_train", "pdb_in_korpm_train"])
        n = len(df)
        print(f"{b:10s} rows {n:7d} proteins {prot.protein_id.nunique():4d} | proteins w/ KORPM-train homolog "
              f"{(prot.homolog_in_korpm_train > 0).mean():.2f}; rows {(df.homolog_in_korpm_train > 0).mean():.2f}; "
              f"rows id>=0.9 {(df.homolog_in_korpm_train >= 0.9).mean():.2f}; same PDB entry {df.pdb_in_korpm_train.mean():.2f}; "
              f"exact variant in train {df.exact_in_korpm_train.sum()} ({df.exact_in_korpm_train.mean():.3f})")


if __name__ == "__main__":
    main()
