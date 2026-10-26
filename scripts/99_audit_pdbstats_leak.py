"""Audit (read-only): sequence overlap between the PISCES chains behind physddg/data/{pdbstats,rotlib_bbdep}.npz
and every benchmark protein (data/processed/clusters.csv).

PISCES sequences = observed residues per chain from data/cache/pdbstats_records.parquet (scripts/40 extract cache;
only split=='train' chains enter the statistics, but both are reported). Search: MMseqs2 easy-search -s 7.5,
E <= 1e-3; a 'hit' is fident >= 0.30 with the alignment covering >= 50% of the shorter sequence (same rule as
scripts/14_homology.py, 30% instead of 25%); 'identical' = fident >= 0.95 and >= 90% of the benchmark sequence covered.
Also: exact PDB-ID matches between benchmark structures and the PISCES list.

Usage: .venv/bin/python scripts/99_audit_pdbstats_leak.py OUTDIR      (writes only into OUTDIR)
Writes OUTDIR/{pisces.fasta,bench.fasta,hits.m8,per_protein.csv,summary.csv,exclude_chains.txt}.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MMSEQS = ROOT / "data/raw/tools/mmseqs/bin/mmseqs"
REC = ROOT / "data/cache/pdbstats_records.parquet"
SETS = ["megascale", "fireprot", "s669", "s461", "ssym", "s2648", "domainome"]


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(REC, columns=["pdb", "chain", "split", "aa"])
    ch = r.groupby(["pdb", "chain"], sort=False).agg(seq=("aa", "".join), split=("split", "first")).reset_index()
    ch["uid"] = ch.pdb.str.upper() + "_" + ch.chain
    (out / "pisces.fasta").write_text("".join(f">{u}\n{s}\n" for u, s in zip(ch.uid, ch.seq)))
    c = pd.read_csv(ROOT / "data/processed/clusters.csv")
    c = c[c.dataset.isin(SETS)].copy()
    c["uid"] = c.dataset + "|" + c.protein_id.astype(str)
    (out / "bench.fasta").write_text("".join(f">{u}\n{s}\n" for u, s in zip(c.uid, c.sequence)))
    m8 = out / "hits.m8"
    if not m8.exists():
        subprocess.run([str(MMSEQS), "easy-search", out / "bench.fasta", out / "pisces.fasta", m8, out / "tmp",
                        "-s", "7.5", "-e", "1e-3", "--max-seqs", "5000", "--threads", "32",
                        "--format-output", "query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,qlen,tlen"],
                       check=True, stdout=subprocess.DEVNULL)
    h = pd.read_csv(m8, sep="\t", header=None,
                    names="query target fident alnlen qstart qend tstart tend evalue qlen tlen".split())
    h["qcov"] = (h.qend - h.qstart + 1) / h.qlen
    h["tcov"] = (h.tend - h.tstart + 1) / h.tlen
    h["scov"] = (h.alnlen / h[["qlen", "tlen"]].min(1)).clip(upper=1)
    h = h.merge(ch[["uid", "split"]].rename(columns={"uid": "target", "split": "pisces_split"}), on="target")
    h["hit30"] = (h.fident >= 0.30) & (h.scov >= 0.5)
    h["ident"] = (h.fident >= 0.95) & (h.qcov >= 0.9)
    rows = []
    for u, g in h.groupby("query"):
        gt = g[g.pisces_split == "train"]
        g30 = gt[gt.hit30]
        rows.append(dict(uid=u, n_hit30_train=len(g30), n_ident_train=int(gt.ident.sum()),
                         max_fident_train=g30.fident.max() if len(g30) else 0.0,
                         best_train=g30.sort_values("fident").target.iloc[-1] if len(g30) else "",
                         n_hit30_any=int(g.hit30.sum())))
    pp = c[["uid", "dataset", "protein_id", "split", "length"]].merge(pd.DataFrame(rows), on="uid", how="left").fillna(
        {"n_hit30_train": 0, "n_ident_train": 0, "max_fident_train": 0.0, "best_train": "", "n_hit30_any": 0})
    # exact PDB-id match of benchmark structure with a PISCES entry (any chain)
    pis_ids = set(ch.pdb.str.upper())
    pp["pdbid_in_pisces"] = pp.protein_id.astype(str).str[:4].str.upper().isin(pis_ids) & pp.dataset.ne("domainome")
    pp.to_csv(out / "per_protein.csv", index=False)
    s = (pp.assign(h30=pp.n_hit30_train > 0, idt=pp.n_ident_train > 0, f50=pp.max_fident_train >= 0.5)
         .groupby(["dataset", "split"]).agg(n=("uid", "size"), with_hit30=("h30", "sum"), with_hit50=("f50", "sum"),
                                            with_identical=("idt", "sum"), pdbid_in_pisces=("pdbid_in_pisces", "sum")))
    s.to_csv(out / "summary.csv")
    print(s.to_string())
    # chains to drop for a leakage-free re-derivation: any train chain with a hit30 to any test/external protein
    test_like = pp[~pp.split.isin(["train", "val", "none"])].uid
    excl = sorted(set(h[h.hit30 & h["query"].isin(test_like)].target))
    (out / "exclude_chains.txt").write_text("\n".join(excl) + "\n")
    print(f"exclude list (hit30 to any test/external protein): {len(excl)} chains "
          f"({sum(ch[ch.uid.isin(excl)].split.eq('train'))} in train split of {int(ch.split.eq('train').sum())})")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
