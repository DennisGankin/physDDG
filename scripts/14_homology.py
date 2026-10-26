#!/usr/bin/env python
"""Homology clustering and leakage annotation across all benchmark tables (MMseqs2 18-8cc5c, from scripts/10).

Sequences: the observed residues of the structure chain used for each protein (Megascale: model sequence == assayed
domain; xtal sets: cleaned chain). Ssym reverse rows share the WT protein_id. Hook for other datasets (e.g.
Domainome): Domainome construct sequences are read from data/processed/domainome_structures.csv (construct crops) and
clustered; its annotation goes to data/processed/homology_domainome.csv (the domainome table itself is owned by the
data-af agent and not rewritten). Any data/processed/sequences_*.csv (dataset, protein_id, sequence) is also included.

Outputs
  data/processed/clusters.csv: dataset, protein_id, sequence, length, cluster30, cluster50, family, split,
      + best hits / max identity against Megascale-train, Megascale-train+val, FireProt-train+val.
  In every table: `cluster` := family (see below), plus
      homolog_in_megascale_train      max fraction identity of a qualifying hit to a Megascale-train protein (0 = none)
      homolog_in_megascale_trainval   same vs Megascale train+val
      homolog_in_fireprot_trainval    same vs FireProt train+val
      homolog_in_megascale_train_id / ..._fireprot_trainval_id   best-hit protein
      cluster30, cluster50
Definitions
  cluster30/50: mmseqs cluster --min-seq-id 0.3/0.5 -c 0.8 --cov-mode 0 (both sequences >= 80% covered), -s 7.5,
      cluster-mode 0 (greedy set cover). Standard, but misses domain-in-larger-protein relations.
  homology hit (used for leakage and `family`): all-vs-all mmseqs search -s 7.5, E <= 1e-3, fident >= 0.25,
      alignment covering >= 50% of the SHORTER sequence (catches a Megascale domain inside a full-length protein).
  family: connected components of the homology-hit graph (single linkage; conservative grouping for grouped CV).
  Self hits (same dataset+protein) are ignored; identical sequences in different datasets count (fident 1.0).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ht_common import ROOT, chain_residues, read_structure  # noqa: E402

PROC = ROOT / "data/processed"
MMSEQS = ROOT / "data/raw/tools/mmseqs/bin/mmseqs"
WORK = ROOT / "data/cache/mmseqs"
TABLES = ["megascale", "megascale_censored", "megascale_doubles", "fireprot", "fireprot_dtm", "s669", "s461", "ssym", "s2648"]
SEQ_SOURCE = {"megascale_censored": "megascale", "megascale_doubles": "megascale", "fireprot_dtm": "fireprot"}
E_MAX, ID_MIN, COV_MIN = 1e-3, 0.25, 0.5


def log(*a):
    print(*a, flush=True)


def load_table(name):
    p = PROC / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p), p
    for p in (PROC / f"{name}.csv", PROC / f"{name}.csv.gz"):
        if p.exists():
            return pd.read_csv(p, dtype={"pdb_resnum": str}, low_memory=False), p
    return None, None


def save_table(df, path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        df.to_csv(path.with_suffix(".csv.gz"), index=False)
    else:
        df.to_csv(path, index=False)


def protein_sequences(df, name):
    """One sequence per protein_id from its (direct) structure."""
    rows = []
    sub = df[df.get("direction", pd.Series("direct", index=df.index)).fillna("direct") == "direct"]
    for pid, g in sub.groupby("protein_id"):
        s, c = g.structure.iloc[0], g.chain.iloc[0]
        path = s if s.startswith("/") else ROOT / s
        res = chain_residues(read_structure(path), c)
        rows.append(dict(dataset=name, protein_id=pid, sequence="".join(a for _, a in res),
                         split=g.split.iloc[0]))
    return rows


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in nodes}


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    tables, seqs = {}, []
    for t in TABLES:
        df, path = load_table(t)
        if df is None:
            log(f"skip {t}: not found")
            continue
        tables[t] = (df, path)
        if t in SEQ_SOURCE:
            continue  # proteins already covered by the source table (fireprot_dtm: +25 dTm-only, added below)
        seqs += protein_sequences(df, t)
    df, _ = tables["fireprot_dtm"]
    known = {(r["dataset"], r["protein_id"]) for r in seqs}
    extra = df[~df.protein_id.isin({p for d, p in known if d == "fireprot"})]
    seqs += [dict(r, dataset="fireprot") for r in protein_sequences(extra, "fireprot")]
    dom = PROC / "domainome_structures.csv"  # Domainome (other agent): sequences of the assayed construct crops
    if dom.exists():
        x = pd.read_csv(dom)
        x = x[(x.status == "ok") & x.construct_path.notna()]
        for pid, path in zip(x.domain_ID, x.construct_path):
            st = read_structure(ROOT / path)
            res = chain_residues(st, st[0][0].name)
            seqs.append(dict(dataset="domainome", protein_id=pid, sequence="".join(a for _, a in res), split="external"))
        log(f"hook: domainome constructs: {len(x)}")
    for p in sorted(PROC.glob("sequences_*.csv")):  # generic hook for other datasets
        x = pd.read_csv(p)
        log(f"hook: {p.name}: {len(x)} sequences")
        seqs += x.assign(split=x.get("split", "external"))[["dataset", "protein_id", "sequence", "split"]].to_dict("records")
    S = pd.DataFrame(seqs).drop_duplicates(["dataset", "protein_id"])
    S["uid"] = S.dataset + "|" + S.protein_id
    S["length"] = S.sequence.str.len()
    log(f"sequences: {len(S)} " + str(S.dataset.value_counts().to_dict()))

    WORK.mkdir(parents=True, exist_ok=True)
    fa = WORK / "all.fasta"
    fa.write_text("".join(f">{u}\n{s}\n" for u, s in zip(S.uid, S.sequence)))
    for ident in (0.3, 0.5):
        pre = WORK / f"clu{int(ident * 100)}"
        run([MMSEQS, "easy-cluster", fa, pre, WORK / "tmp", "--min-seq-id", ident, "-c", 0.8, "--cov-mode", 0,
             "-s", 7.5, "--cluster-mode", 0, "--threads", 32])
        c = pd.read_csv(f"{pre}_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
        S[f"cluster{int(ident * 100)}"] = S.uid.map(dict(zip(c.member, c.rep)))
    m8 = WORK / "allvsall.m8"
    run([MMSEQS, "easy-search", fa, fa, m8, WORK / "tmp", "-s", 7.5, "-e", 10, "--max-seqs", 2000, "--threads", 32,
         "--format-output", "query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"])
    h = pd.read_csv(m8, sep="\t", header=None, names="query,target,fident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen".split(","))
    h = h[h["query"] != h["target"]]
    h["cov_short"] = (h[["qend", "tend"]].values - h[["qstart", "tstart"]].values + 1).min(axis=1) / h[["qlen", "tlen"]].min(axis=1)
    h["qcov"] = (h.qend - h.qstart + 1) / h.qlen
    h["tcov"] = (h.tend - h.tstart + 1) / h.tlen
    q = h[(h.evalue <= E_MAX) & (h.fident >= ID_MIN) & (h.cov_short >= COV_MIN)].copy()
    log(f"all-vs-all hits {len(h)}, qualifying homology hits {len(q)}")
    q.to_csv(WORK / "homology_hits.csv", index=False)

    comp = components(list(S.uid), zip(q["query"], q["target"]))
    S["family"] = S.uid.map(comp)
    # readable family ids
    fam_ids = {f: f"fam{i:04d}" for i, f in enumerate(sorted(set(S.family)))}
    S["family"] = S.family.map(fam_ids)
    S["cluster30"] = S.cluster30.map({c: f"c30_{i:04d}" for i, c in enumerate(sorted(set(S.cluster30)))})
    S["cluster50"] = S.cluster50.map({c: f"c50_{i:04d}" for i, c in enumerate(sorted(set(S.cluster50)))})

    split = dict(zip(S.uid, S.split))
    refs = {
        "megascale_train": {u for u in S.uid if u.startswith("megascale|") and split[u] == "train"},
        "megascale_trainval": {u for u in S.uid if u.startswith("megascale|") and split[u] in ("train", "val")},
        "fireprot_trainval": {u for u in S.uid if u.startswith("fireprot|") and split[u] in ("train", "val")},
    }
    for name, ref in refs.items():
        sub = q[q.target.isin(ref)].sort_values(["fident", "bits"], ascending=False).drop_duplicates("query")
        S[f"homolog_in_{name}"] = S.uid.map(dict(zip(sub["query"], sub.fident))).fillna(0.0)
        S[f"homolog_in_{name}_id"] = S.uid.map(dict(zip(sub["query"], sub.target.str.split("|").str[1]))).fillna("")
        S[f"homolog_in_{name}_cov"] = S.uid.map(dict(zip(sub["query"], sub.cov_short))).fillna(0.0)
    S.drop(columns=["uid"]).to_csv(PROC / "clusters.csv", index=False)

    # annotate tables
    cols = ["cluster30", "cluster50", "family"] + [c for c in S.columns if c.startswith("homolog_in_")]
    for t, (df, path) in tables.items():
        ds = SEQ_SOURCE.get(t, t)
        ann = S[S.dataset == ds].set_index("protein_id")[cols]
        df = df.drop(columns=[c for c in cols if c in df.columns])
        df = df.join(ann, on="protein_id")
        df["cluster"] = df.family
        assert df.cluster.notna().all(), t
        save_table(df, path)
    # Domainome table belongs to the data-af agent: write a side table instead of rewriting it
    if (S.dataset == "domainome").any():
        S[S.dataset == "domainome"][["protein_id"] + cols].to_csv(PROC / "homology_domainome.csv", index=False)

    # ---- report ----
    log("\n=== leakage summary (protein level; hit = E<=1e-3, id>=25%, >=50% of shorter seq aligned) ===")
    for ds, g in S.groupby("dataset", sort=False):
        for sp, gg in g.groupby("split"):
            n = len(gg)
            a = (gg.homolog_in_megascale_train > 0).sum()
            b = (gg.homolog_in_fireprot_trainval > 0).sum()
            hi = (gg.homolog_in_megascale_train >= 0.5).sum()
            log(f"{ds:10s} {sp:8s} n={n:4d}  homolog in megascale-train: {a:3d} (>=50% id: {hi:3d})   "
                f"in fireprot-trainval: {b:3d}")
    mt = S[(S.dataset == "megascale") & (S.split == "test")]
    log("megascale-test proteins with megascale-train homolog:\n" +
        mt[mt.homolog_in_megascale_train > 0][["protein_id", "homolog_in_megascale_train", "homolog_in_megascale_train_id",
                                               "homolog_in_megascale_train_cov"]].to_string(index=False))
    log(f"families: {S.family.nunique()}, cluster30: {S.cluster30.nunique()}, cluster50: {S.cluster50.nunique()}")
    # row-level fractions for external tables
    for t in tables:
        df = load_table(t)[0]
        log(f"{t:12s} rows={len(df):7d}  rows w/ megascale-train homolog: {(df.homolog_in_megascale_train > 0).mean():.1%}"
            f"  w/ fireprot-trainval homolog: {(df.homolog_in_fireprot_trainval > 0).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
