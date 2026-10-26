#!/usr/bin/env python
"""Canonical Domainome variant table (data/processed/domainome.csv + .parquet).

Input (read-only): /workspace/Domainome/Extended_data_Table_2_fitness_scores_normalized_domainranks.txt
  (= Supplementary Table 2 of Beltran et al. 2025 Nature 637:885, Zenodo 10.5281/zenodo.14356805)
Needs: 22_domainome_structures.py outputs; 20_download_domainome_supp.sh (MoCHI table, optional).

Facts established (evidence printed by this script / in notes/data_af_report.md):
  * `position` is UniProt numbering (one exception, Q13472_PF06839_896: +1, see 22_*); domain_ID = <UniProt>_<Pfam>_<UniProt start of construct>;
    position runs start..start+len(aa_seq)-1 for every domain; aa_seq is the full construct sequence
    of the variant (WT row: wt_aa NaN).
  * The file already contains only the 522 retained domains (quality_rank 1..522); no further
    domain filtering is applied. Rows kept: STOP == False, wt_aa/mut_aa present, mut_aa != '*',
    normalized_fitness not NaN  (-> 536,164 missense variants; + 26,848 nonsense + 522 WT = the paper's 563,534).
  * normalized_fitness: WT = 0, median nonsense per domain ~ -1. Lower = less abundant = destabilised.
    score = normalized_fitness (HIGHER = MORE STABLE; opposite sign to ddg).
  * Per-domain measured ddG are NOT released. Supplementary Table 4 holds MoCHI family-level
    ('homologue-averaged') energies, rescaled (WT=0, 2.5th percentile = -1, unitless) for families with
    >=5 human homologues. Its sign is checked here against normalized_fitness and it is stored as
    ddg_mochi_scaled (positive = destabilising) - NOT in `ddg`, which stays NaN (not kcal/mol, not per-domain).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM = "/workspace/Domainome/Extended_data_Table_2_fitness_scores_normalized_domainranks.txt"
MOCHI = os.path.join(ROOT, "data/raw/domainome/Extended_data_Table_4_homolog_predictions.txt")
FX_CON = "/foldx_native_clean/results/variants_domainome_construct.csv"
FX_FULL = "/foldx_native_clean/results/variants_domainome.csv"
OUT = os.path.join(ROOT, "data/processed")


def main():
    d = pd.read_csv(DOM, sep="\t")
    n0 = len(d)
    wtrows = d[d.wt_aa.isna()]
    m = d[(d.STOP == False) & d.wt_aa.notna() & d.mut_aa.notna() & (d.mut_aa != "*")].copy()
    n_miss_all = len(m)
    m = m[m.normalized_fitness.notna()].copy()
    print(f"rows {n0}; WT rows {len(wtrows)}; missense rows {n_miss_all}; with fitness {len(m)}")
    m["position"] = m.position.astype(int)
    m["start"] = m.domain_ID.str.rsplit("_", n=1).str[-1].astype(int)
    m["construct_pos"] = m.position - m.start + 1
    # WT identity vs construct sequence (variant aa_seq carries the mutant -> use WT row seq)
    wseq = wtrows.set_index("domain_ID").aa_seq
    m["construct_seq"] = m.domain_ID.map(wseq)
    m["wt_in_construct"] = [s[p - 1] if 1 <= p <= len(s) else "" for s, p in zip(m.construct_seq, m.construct_pos)]
    mm = (m.wt_in_construct != m.wt_aa).sum()
    # the variant's own aa_seq must differ from WT exactly at construct_pos by mut
    chk = [(a[p - 1] == mu) and (sum(x != y for x, y in zip(a, s)) == 1)
           for a, s, p, mu in zip(m.aa_seq, m.construct_seq, m.construct_pos, m.mut_aa)]
    print(f"wt_aa != construct WT seq at construct_pos: {mm}; variant aa_seq consistent (single subst at pos): {np.mean(chk):.6f}")
    assert mm == 0

    s = pd.read_csv(os.path.join(OUT, "domainome_structures.csv"))
    per = pd.read_csv(os.path.join(OUT, "domainome_residue_plddt.csv"))
    m = m.merge(s[["domain_ID", "status", "construct_path", "af_full_path", "af_version", "af_source", "af_frag_offset",
                   "plddt_mean", "frac_plddt_lt70", "seqpos_shift"]], on="domain_ID", how="left")
    m = m.merge(per.rename(columns={"pdb_resnum": "uniprot_pos", "domainome_position": "position"}),
                on=["domain_ID", "position"], how="left")
    # UniProt position = Domainome position + seqpos_shift (shift != 0 only for Q13472_PF06839_896, +1)
    m["seqpos_shift"] = m.seqpos_shift.fillna(0).astype(int)
    m["uniprot_pos"] = m.position + m.seqpos_shift
    print("domains with shifted numbering:", m.loc[m.seqpos_shift != 0, "domain_ID"].unique())
    has = m.construct_path.notna()
    wt_struct_mm = (m.loc[has, "af_aa"] != m.loc[has, "wt_aa"])
    print(f"variants with AF structure: {has.sum()} / {len(m)}; WT mismatch vs AF residue: {wt_struct_mm.sum()} "
          f"({wt_struct_mm.mean():.4%}); variants missing residue in crop: {m.loc[has,'af_aa'].isna().sum()}")

    # MoCHI family-level scaled ddG
    if os.path.exists(MOCHI):
        mo = pd.read_csv(MOCHI, sep="\t")
        mo = mo.rename(columns={"uniprot_ID": "uniprot_ID", "pos": "position", "wt_aa": "wt_aa", "mut_aa": "mut_aa"})
        dup = mo.duplicated(["uniprot_ID", "position", "wt_aa", "mut_aa"]).sum()
        mo = mo.drop_duplicates(["uniprot_ID", "position", "wt_aa", "mut_aa"])
        mo = mo.rename(columns={"position": "uniprot_pos"})
        m = m.merge(mo[["uniprot_ID", "uniprot_pos", "wt_aa", "mut_aa", "PFAM_entry", "mean_kcalmol_scaled", "std_kcalmol_scaled"]],
                    on=["uniprot_ID", "uniprot_pos", "wt_aa", "mut_aa"], how="left")
        ok = m.mean_kcalmol_scaled.notna()
        rho = spearmanr(m.loc[ok, "mean_kcalmol_scaled"], m.loc[ok, "normalized_fitness"])[0]
        print(f"MoCHI table rows {len(mo)} (dups dropped {dup}); matched {ok.sum()} variants in "
              f"{m.loc[ok,'domain_ID'].nunique()} domains; spearman(mean_kcalmol_scaled, normalized_fitness)={rho:.3f}")
        # positive correlation with fitness => negative values = destabilising => flip
        sign = -1.0 if rho > 0 else 1.0
        m["ddg_mochi_scaled"] = sign * m.mean_kcalmol_scaled
        m["ddg_mochi_scaled_sd"] = m.std_kcalmol_scaled
        m["mochi_pfam_entry"] = m.PFAM_entry
    else:
        print("MoCHI table missing - run scripts/20_download_domainome_supp.sh")
        m["ddg_mochi_scaled"] = np.nan; m["ddg_mochi_scaled_sd"] = np.nan; m["mochi_pfam_entry"] = np.nan

    # FoldX manifests (join key: entity_id == protein_id (construct run) / uniprot (full-length run),
    # position_1based == seq_pos, wt_aa == wt, mut_aa == mut)
    fx = pd.read_csv(FX_CON, usecols=["entity_id", "position_1based", "wt_aa", "mut_aa", "chain", "pdb_resnum"])
    fx = fx.rename(columns={"entity_id": "domain_ID", "position_1based": "position", "chain": "fx_chain", "pdb_resnum": "fx_pdb_resnum"})
    m = m.merge(fx, on=["domain_ID", "position", "wt_aa", "mut_aa"], how="left")
    inf = m.fx_pdb_resnum.notna()
    print(f"FoldX construct manifest: {len(fx)} rows / {fx.domain_ID.nunique()} entities; matched {inf.sum()} of our "
          f"{len(m)} variants ({m.loc[inf,'domain_ID'].nunique()} domains); pdb_resnum equal: "
          f"{(m.loc[inf,'fx_pdb_resnum'].astype(int)==m.loc[inf,'position']).mean():.4f}; chain A: {(m.loc[inf,'fx_chain']=='A').mean():.3f}")
    fxk = set(zip(fx.domain_ID, fx.position, fx.wt_aa, fx.mut_aa))
    ours = set(zip(m.domain_ID, m.position, m.wt_aa, m.mut_aa))
    print(f"FoldX construct variants not in our table: {len(fxk-ours)}")
    ff = pd.read_csv(FX_FULL, usecols=["entity_id", "position_1based", "wt_aa", "mut_aa"])
    ffk = set(zip(ff.entity_id, ff.position_1based, ff.wt_aa, ff.mut_aa))
    m["in_foldx_full"] = [(u, p, w, x) in ffk for u, p, w, x in zip(m.uniprot_ID, m.position, m.wt_aa, m.mut_aa)]
    # FoldX full-length pdb_resnum = Domainome position too (checked on matched rows)
    print(f"FoldX full-length manifest: {len(ff)} rows / {ff.entity_id.nunique()} entities; matched {m.in_foldx_full.sum()} variants")

    pf = m.domain_ID.str.split("_").str[1]
    out = pd.DataFrame({
        "dataset": "domainome",
        "protein_id": m.domain_ID,
        "structure": m.construct_path,
        "chain": np.where(has, "A", None),
        "pdb_resnum": np.where(has, m.uniprot_pos.astype(str), None),
        "wt": m.wt_aa, "mut": m.mut_aa,
        "ddg": np.nan,
        "split": "external",
        "cluster": np.where(m.domain_ID.str.contains("rockdoms"), m.domain_ID, pf),
        "uniprot": np.where(m.domain_ID.str.contains("rockdoms"), None, m.uniprot_ID),
        "seq_pos": m.uniprot_pos,
        "domainome_position": m.position,
        "construct_pos": m.construct_pos,
        "score": m.normalized_fitness,
        "score_sigma": m.normalized_fitness_sigma,
        "fitness": m.fitness, "fitness_sigma": m.fitness_sigma,
        "mean_input_count": m.mean_input_count,
        "quality_rank": m.quality_rank,
        "ddg_mochi_scaled": m.ddg_mochi_scaled, "ddg_mochi_scaled_sd": m.ddg_mochi_scaled_sd,
        "mochi_pfam_entry": m.mochi_pfam_entry,
        "plddt": m.plddt,
        "construct_plddt_mean": m.plddt_mean,
        "construct_frac_plddt_lt70": m.frac_plddt_lt70,
        "af_version": m.af_version,
        "af_full_structure": m.af_full_path,
        "af_full_resnum": np.where(has, (m.uniprot_pos - m.af_frag_offset).astype("Int64").astype(str), None),
        "in_foldx_construct": inf,
        "in_foldx_full": m.in_foldx_full,
        "method": "aPCA (DHFR abundance PCA, yeast)",
        "source": "Beltran2025_Domainome1_SuppTable2",
        "struct_source": np.where(has, "af", None),
    })
    out = out.sort_values(["protein_id", "seq_pos", "mut"]).reset_index(drop=True)
    out.to_csv(os.path.join(OUT, "domainome.csv"), index=False)
    out.to_parquet(os.path.join(OUT, "domainome.parquet"), index=False)
    print(f"wrote {len(out)} variants, {out.protein_id.nunique()} domains, {out.uniprot.nunique()} UniProt; "
          f"with structure {out.structure.notna().sum()}")
    # summaries for the report
    q = out[out.structure.notna()]
    print("pLDDT over measured variants: ", q.plddt.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict())
    pos = q.drop_duplicates(["protein_id", "seq_pos"])
    print("pLDDT over measured positions:", pos.plddt.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict())
    print("frac positions pLDDT<70:", round((pos.plddt < 70).mean(), 4), " <50:", round((pos.plddt < 50).mean(), 4))
    dd = q.drop_duplicates("protein_id")
    print("construct frac pLDDT<70 per domain:", dd.construct_frac_plddt_lt70.describe(percentiles=[.5, .75, .9, .95]).round(3).to_dict())
    print("domains with >20% / >50% residues pLDDT<70:", (dd.construct_frac_plddt_lt70 > .2).sum(), (dd.construct_frac_plddt_lt70 > .5).sum())
    print("spearman(score, plddt) over variants:", round(spearmanr(q.score, q.plddt)[0], 3))


if __name__ == "__main__":
    main()
