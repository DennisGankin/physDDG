#!/usr/bin/env python
"""D23(D) / audit 2 I3: homology screen of the seven abundance_dms proteins against the training sets.

Same rule as scripts/14_homology.py: mmseqs2 easy-search -s 7.5, a hit needs E <= 1e-3, fident >= 0.25 and
>= 50% of the SHORTER sequence aligned. Self-contained (it searches against the existing
data/cache/mmseqs/all.fasta) so it does not rewrite any other table's homology columns.

  PYTHONPATH=. .venv/bin/python scripts/58_abundance_homology.py
Writes data/processed/homology_abundance_dms.csv (protein_id, homolog_in_megascale_train,
homolog_in_fireprot_trainval, best hit id/identity/coverage).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

from ht_common import chain_residues, read_structure  # noqa: E402

MMSEQS = ROOT / "data/raw/tools/mmseqs/bin/mmseqs"
WORK = ROOT / "data/cache/mmseqs"
PROC = ROOT / "data/processed"
COLS = "query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen".split(",")


def main():
    d = pd.read_csv(PROC / "abundance_dms.csv", low_memory=False)
    prot = d.drop_duplicates("protein_id")[["protein_id", "structure", "chain", "uniprot"]]
    rows = []
    for r in prot.itertuples():
        p = r.structure if str(r.structure).startswith("/") else ROOT / r.structure
        res = chain_residues(read_structure(p), r.chain)
        rows.append({"protein_id": r.protein_id, "uniprot": r.uniprot,
                     "sequence": "".join(a for _, a in res)})
    S = pd.DataFrame(rows)
    print(S.assign(length=S.sequence.str.len())[["protein_id", "uniprot", "length"]].to_string(index=False))
    qfa = WORK / "abundance_query.fasta"
    qfa.write_text("".join(f">abundance_dms|{r.protein_id}\n{r.sequence}\n" for r in S.itertuples()))
    m8 = WORK / "abundance_vs_all.m8"
    subprocess.run([str(MMSEQS), "easy-search", str(qfa), str(WORK / "all.fasta"), str(m8), str(WORK / "tmp"),
                    "-s", "7.5", "-e", "10", "--max-seqs", "2000", "--threads", "32",
                    "--format-output", ",".join(COLS)], check=True, capture_output=True)
    h = pd.read_csv(m8, sep="\t", header=None, names=COLS)
    h["cov_short"] = ((h[["qend", "tend"]].values - h[["qstart", "tstart"]].values + 1).min(axis=1)
                      / h[["qlen", "tlen"]].min(axis=1))
    h = h[(h.evalue <= 1e-3) & (h.fident >= 0.25) & (h.cov_short >= 0.5)]
    h["qid"] = h["query"].str.split("|").str[-1]
    h[["tset", "tid"]] = h["target"].str.split("|", n=1, expand=True)
    # which training sets each target belongs to
    ms = pd.read_parquet(PROC / "megascale.parquet", columns=["protein_id", "split"]).drop_duplicates()
    ms_train = set(ms[ms.split == "train"].protein_id)
    fp = pd.read_csv(PROC / "fireprot.csv", usecols=["protein_id", "split"]).drop_duplicates()
    fp_tv = set(fp[fp.split.isin(["train", "val"])].protein_id)
    out = []
    for pid in S.protein_id:
        g = h[h.qid == pid]
        a = g[(g.tset == "megascale") & g.tid.isin(ms_train)]
        b = g[(g.tset == "fireprot") & g.tid.isin(fp_tv)]
        best = lambda x: (x.sort_values("fident", ascending=False).iloc[0] if len(x) else None)
        ba, bb = best(a), best(b)
        out.append({"protein_id": pid,
                    "homolog_in_megascale_train": float(ba.fident) if ba is not None else 0.0,
                    "homolog_in_megascale_train_id": ba.tid if ba is not None else None,
                    "homolog_in_fireprot_trainval": float(bb.fident) if bb is not None else 0.0,
                    "homolog_in_fireprot_trainval_id": bb.tid if bb is not None else None,
                    "n_hits_any_training": int(len(a) + len(b))})
    t = pd.DataFrame(out)
    t.to_csv(PROC / "homology_abundance_dms.csv", index=False)
    print(t.to_string(index=False))
    print(f"\nproteins with a Megascale-train homologue: {(t.homolog_in_megascale_train > 0).sum()}/7; "
          f"FireProt-trainval: {(t.homolog_in_fireprot_trainval > 0).sum()}/7")


if __name__ == "__main__":
    main()
