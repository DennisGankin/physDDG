#!/usr/bin/env python
"""Extra abundance DMS sets (VAMP-seq / aPCA / abundance) on AlphaFold structures, for cross-assay tests.

Source: ProteinGym v1.3 substitution benchmark (Notin et al. 2023), which redistributes the processed scores
  https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
  reference: https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv
Selected assays (soluble/cytosolic proteins, abundance readout; membrane/surface-expression assays excluded):
  PTEN (Matreyek 2021, VAMP-seq), TPMT (Matreyek 2018, VAMP-seq), CYP2C9 (Amorosi 2021, VAMP-seq abundance),
  NUDT15 (Suiter 2020, VAMP-seq-like abundance), PRKN (Clausen 2023, abundance), GCK/HXK4 (Gersing 2023, abundance),
  KRAS (Weng 2022, aPCA abundance - same assay family as Domainome).
ProteinGym DMS_score is oriented so that higher = fitter = MORE abundant (= more stable): score = DMS_score.
Mutant numbering is 1-based on the ProteinGym target_seq; it is verified against the AF model sequence (UniProt).
Structures: full-length AF models (all assays are on full-length proteins), human v4 from /af_data/af2.
Output: data/processed/abundance_dms.csv (+ .md card), data/structures/abundance_dms/*.pdb
"""
import os, sys, io, zipfile, subprocess, shutil
import numpy as np
import pandas as pd
import gemmi
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afdb_io

ROOT = afdb_io.ROOT
RAW = os.path.join(ROOT, "data/raw/proteingym")
SDIR = os.path.join(ROOT, "data/structures/abundance_dms")
ZIP_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
REF_URL = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
ASSAYS = ["PTEN_HUMAN_Matreyek_2021", "TPMT_HUMAN_Matreyek_2018", "CP2C9_HUMAN_Amorosi_2021_abundance",
          "NUD15_HUMAN_Suiter_2020", "PRKN_HUMAN_Clausen_2023", "HXK4_HUMAN_Gersing_2023_abundance",
          "RASK_HUMAN_Weng_2022_abundance"]


def fetch(url, dest):
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", dest, url], check=True)


def accession(entry_name):
    r = subprocess.run(["curl", "-sSfL", f"https://rest.uniprot.org/uniprotkb/search?query=id:{entry_name}&fields=accession&format=tsv"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()[1].split("\t")[0]


def main():
    os.makedirs(RAW, exist_ok=True); os.makedirs(SDIR, exist_ok=True)
    zp = os.path.join(RAW, "DMS_ProteinGym_substitutions.zip"); fetch(ZIP_URL, zp)
    rp = os.path.join(RAW, "DMS_substitutions.csv"); fetch(REF_URL, rp)
    ref = pd.read_csv(rp).set_index("DMS_id")
    z = zipfile.ZipFile(zp)
    names = {os.path.basename(n): n for n in z.namelist()}
    acc_cache = os.path.join(RAW, "accessions.tsv")
    accs = dict(pd.read_csv(acc_cache, sep="\t").values) if os.path.exists(acc_cache) else {}
    rows, cards = [], []
    for a in ASSAYS:
        r = ref.loc[a]
        en = r.UniProt_ID
        if en not in accs:
            accs[en] = accession(en)
        acc = accs[en]
        d = pd.read_csv(z.open(names[r.DMS_filename]))
        d = d[~d.mutant.str.contains(":")].copy()
        d["wt"] = d.mutant.str[0]; d["mut"] = d.mutant.str[-1]; d["pos"] = d.mutant.str[1:-1].astype(int)
        tseq = r.target_seq
        assert all(tseq[p - 1] == w for p, w in zip(d.pos, d.wt)), a
        c = next(c for c in afdb_io.candidates(acc, sources=("af2_tree", "v6_human", "v6_swissprot")) )
        aseq, anums, apl = afdb_io.structure_seq(c["text"])
        assert c["n_frags"] == 1
        off = aseq.find(tseq)  # target_seq position p -> UniProt p + off
        same = tseq == aseq
        if off >= 0:
            pmap = {p: p + off for p in range(1, len(tseq) + 1)}
        else:
            # e.g. KRAS: assay on isoform 4B (P01116-2), AF model = canonical 4A; map by global alignment and keep
            # only positions whose aligned AF residue is identical
            from Bio import Align
            al = Align.PairwiseAligner(); al.mode = "global"; al.open_gap_score = -10; al.extend_gap_score = -0.5
            aln = al.align(tseq, aseq)[0]
            pmap = {}
            for (t0, t1), (a0, a1) in zip(*aln.aligned):
                for k in range(t1 - t0):
                    if tseq[t0 + k] == aseq[a0 + k]:
                        pmap[t0 + k + 1] = a0 + k + 1
            off = -1
        n_before = len(d)
        d = d[d.pos.isin(pmap)].copy()
        n_dropped = n_before - len(d)
        fname = f"AF-{acc}-F1-model_v{c['version']}.pdb"
        fpath = os.path.join(SDIR, fname)
        if not os.path.exists(fpath):
            open(fpath, "w").write(c["text"])
        pl = dict(zip(anums, apl))
        st = gemmi.read_pdb_string(c["text"])
        ca = {r_.seqid.num: np.array(r_.find_atom("CA", "*").pos.tolist()) for r_ in st[0][0]}
        X = np.array([ca[n] for n in anums]); D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        nc10 = dict(zip(anums, (D < 10).sum(1) - 1))
        d["seq_pos"] = d.pos.map(pmap)
        d["af_aa"] = [aseq[p - 1] for p in d.seq_pos]
        mm = (d.af_aa != d.wt).sum()
        d["plddt"] = d.seq_pos.map(pl); d["ca_nc10"] = d.seq_pos.map(nc10)
        rho_b = spearmanr(d.DMS_score, d.ca_nc10)[0]
        rho_p = spearmanr(d.DMS_score, d.plddt)[0]
        cards.append(dict(dms_id=a, uniprot=acc, entry=en, n_single=len(d), n_dropped_unmappable=n_dropped, n_pos=d.pos.nunique(), wt_mismatch_af=int(mm),
                          target_eq_af_seq=same, offset=off, af_version=c["version"], plddt_median=float(d.plddt.median()),
                          frac_plddt_lt70=float((d.plddt < 70).mean()), rho_score_vs_burial=round(rho_b, 3),
                          rho_score_vs_plddt=round(rho_p, 3), assay=r.selection_assay, year=r.year, author=r.first_author))
        rows.append(pd.DataFrame({
            "dataset": "abundance_dms", "protein_id": a, "structure": os.path.relpath(fpath, ROOT), "chain": "A",
            "pdb_resnum": d.seq_pos.astype(str), "wt": d.wt, "mut": d.mut, "ddg": np.nan, "split": "external",
            "cluster": acc, "uniprot": acc, "seq_pos": d.seq_pos, "assay_pos": d.pos, "score": d.DMS_score, "score_bin": d.DMS_score_bin,
            "plddt": d.plddt, "method": r.selection_assay, "source": f"ProteinGym_v1.3:{a}", "struct_source": "af",
            "af_version": c["version"]}))
    pd.DataFrame(list(accs.items()), columns=["entry", "accession"]).to_csv(acc_cache, sep="\t", index=False)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(ROOT, "data/processed/abundance_dms.csv"), index=False)
    C = pd.DataFrame(cards)
    C.to_csv(os.path.join(ROOT, "data/processed/abundance_dms_assays.csv"), index=False)
    pd.set_option("display.width", 250)
    print(C.to_string())
    print(f"total {len(out)} single variants, {out.protein_id.nunique()} assays")


if __name__ == "__main__":
    main()
