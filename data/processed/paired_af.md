# paired_af — the same measured variants on crystal and AlphaFold structures

**Purpose.** Development and evaluation of the AF mode: every row is one measured variant mapped onto both
the experimental structure used by the canonical table and the AlphaFold DB model of the same UniProt entry.
**Script.** `scripts/24_paired_af.py` (uses `scripts/afdb_io.py`, `scripts/21_index_af_tars.py` indices).
Rerun it whenever `data/processed/{fireprot,s669,ssym}.csv` change. `paired_af_entities.csv` holds one row per
(xtal structure, chain, UniProt) with alignment and RMSD stats.

## Inputs
* `fireprot.csv`, `s669.csv`, `ssym.csv` (data-HT agent; xtal `structure`/`chain`/`pdb_resnum`, `ddg` positive = destabilising).
  UniProt from the table (FireProt) or from SIFTS `pdb_chain_uniprot.tsv` (S669/Ssym; for each PDB chain the accession
  covering most residues; 1 S669 row, 1FH5_H antibody chain, has none).
  Fallback when `fireprot.csv` is missing: the raw `/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv`.
* AF model: human proteins **v4** (`/af_data/af2`, same as the Domainome structures), else **v6** (`swissprot_pdb_v6.tar`,
  proteome tars), else EBI AFDB (latest; cached in `data/raw/afdb/`). Merged/secondary accessions are resolved through
  UniProt (Q9H782 → O75400, `af_accession`).

## Mapping
Observed residues of the xtal chain are aligned to the AF sequence (BLOSUM62 local alignment, gap −10/−0.5); the variant's
`xtal_resnum` is mapped to `af_resnum` = UniProt numbering (`seq_pos`). `structure`/`chain`/`pdb_resnum` point to the AF
crop `data/structures/paired_af/af_xtalspan/<acc>__<pdb>_<chain>.pdb` (AF model cut to the span aligned to the xtal chain,
UniProt numbering, chain A, B-factor = pLDDT); `af_full_structure` is the unchanged AF model.
WT identity is checked on both sides (`wt_match_xtal`, `wt_match_af`); **use rows with `usable == True`**.
For Ssym/S669 `direction == reverse` rows the xtal is the mutant structure, so the AF (wild-type) model carries the
reverse's `mut` at the site (`af_expected_aa`; `af_is_mutant_state` = True: AF is the *end* state of that mutation).

## Counts (usable = WT matches xtal and AF)
| dataset | rows | usable | proteins (usable) | no AF model | site not in AF alignment |
|---|---|---|---|---|---|
| fireprot | 3,403 | 3,078 | 98 | 308 (T4 lysozyme P00720) | 17 (1PGA M1: FireProt maps GB1 M1 to UniProt M1 of the full protein G — wrong; not in the AF sequence) |
| s669 | 669 | 629 | 80 | 39 (viral: P69168 M13 g3p 27, TEV 5, DENV 4, P22 2, HIV 1) + 1 no UniProt | 0 |
| ssym | 684 | 305 (152 reverse) | 12 | 378 (T4 lysozyme 368, phage f1 g5p 10) | 1 |

WT mismatch rate among mapped sites: **0 %** (xtal side 0/4,755 rows, AF side 0/4,012). FireProt's own UniProt
`position` disagrees with the alignment for 1 % of rows (6 proteins: 1TEN, 1TTG, 2OCJ/p53, 1UZC, 1HZ6, one 1PGA row) —
FireProt's sequence there is a different isoform/version; the alignment mapping is the one to use.

## Structure agreement (usable rows)
| dataset | pLDDT median | pLDDT<70 | site local CA RMSD (10 Å sphere) median / p90 | > 1 Å |
|---|---|---|---|---|
| fireprot | 97.7 | 0.3 % | 0.24 / 0.78 Å | 6.3 % |
| s669 | 93.9 | 1.0 % | 0.40 / 1.36 Å | 15.7 % |
| ssym | 98.5 | 0 % | 0.26 / 0.49 Å | 4.3 % |
Global aligned CA RMSD per entity: median 0.62 Å; 44/351 entities > 2 Å (multi-domain hinges, NMR models, latent vs active
PAI-1 1C5G 10.3 Å, dystrophin fragment 19.9 Å). Spearman(pLDDT, local RMSD) = −0.44.
Columns: `plddt`, `ca_dev_site` (after global superposition), `local_rmsd_10A`, `ca_rmsd_aligned`, `aln_identity`, `aln_cov_xtal`.

## Known issues
* These benchmark proteins are very well predicted (median pLDDT 94–98): the paired set tests AF-vs-xtal differences in
  side chains/packing, **not** low-confidence regions. Domainome (pLDDT median 90, 6 % of sites < 70) covers the latter.
* No AF DB models exist for phage/viral proteins (AFDB excludes most viral sequences): T4 lysozyme alone removes 308
  FireProt and 368 Ssym rows. Generating models (ColabFold) would be needed to recover them — PI decision.
* AF models are full-length UniProt sequences (signal peptides, pro-regions, other domains); the `af_xtalspan` crops remove
  residues outside the crystallised span but do not model crystal ligands, cofactors, metals or oligomer partners.
* Versions are mixed (v4 human, v6 others, 4 S669 rows on 3G1G (P03322, viral polyprotein) from a new-style EBI segment model `AF-0000000365763032-model_v1`) — recorded in `af_version`/`af_source`.
