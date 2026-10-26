# Data-AF report: Domainome, AlphaFold structures, paired xtal/AF sets, abundance DMS (2026-10-26)

Pipeline (all rerunnable, in order): `20_download_domainome_supp.sh` → `21_index_af_tars.py` → `22_domainome_structures.py`
→ `23_domainome_table.py` → `24_paired_af.py` → `25_abundance_dms.py` → `26_domainome_v4_v6.py`
(shared helper `scripts/afdb_io.py`). Cards: `data/processed/{domainome,paired_af,abundance_dms}.md`.

## Outputs
| file | rows | content |
|---|---|---|
| data/processed/domainome.csv / .parquet (git-ignored, 237 / 22 MB) | 536,164 | Domainome missense; 532,501 on AF construct crops |
| data/processed/domainome_structures.csv | 522 | per-domain AF provenance, construct match, pLDDT summary |
| data/processed/domainome_residue_plddt.csv | 29,916 | per construct residue pLDDT |
| data/processed/domainome_af_v4_v6.csv | 502 | v4 vs v6 AF model agreement per construct |
| data/processed/paired_af.csv (+ _entities) | 4,756 (4,012 usable) | FireProt / S669 / Ssym variants on xtal and AF |
| data/processed/abundance_dms.csv (+ _assays) | 37,947 | 7 abundance DMS (PTEN, TPMT, CYP2C9, NUDT15, PRKN, GCK, KRAS) |
| data/structures/domainome/{af_full,af_construct,af_construct_v6}/ | 430 / 518 / 497 PDB | AF models, construct crops |
| data/structures/paired_af/{af_full,af_xtalspan}/ ; data/structures/abundance_dms/ | | AF models and xtal-span crops |

## A. Domainome table
* Source = Supplementary Table 2 (Beltran et al. 2025 Nature 637:885). The file already holds only the 522 retained domains
  (quality_rank 1–522). Kept STOP==False, non-`*` mut, non-NaN fitness → 536,164 missense (+26,848 nonsense + 522 WT
  = the paper's 563,534, exact).
* **Numbering:** `position` is UniProt numbering; domain_ID suffix = UniProt position of the construct's first residue;
  positions span exactly the construct (522/522). Verified against AF/UniProt sequences: 0 WT mismatches over 532,501
  variants. **One domain is off by one:** Q13472_PF06839_896 (construct starts at UniProt 897; 43/43 residues mismatch at the
  nominal offset, 0 at +1) → `seq_pos = domainome_position + 1` for it only.
* **Sign:** `score = normalized_fitness` (WT = 0, nonsense ≈ −1 per domain, median −1.02); higher = more abundant = more
  stable, i.e. opposite to `ddg`.
* **MoCHI ΔΔG:** per-domain ΔΔG are not public. Supplementary Table 4 (Zenodo 14356805) has MoCHI *family-level,
  homologue-averaged* energies, rescaled to WT = 0 / 2.5th percentile magnitude 1 (unitless), for families with ≥5 human
  homologues. Downloaded and added as `ddg_mochi_scaled` (328,078 variants, 358 domains). Sign check: ρ(Table 4,
  normalized_fitness) = −0.64 → positive = destabilising as given. `ddg` left NaN on purpose (not kcal/mol, not per-domain).
* `cluster` = Pfam ID (129 families); `split` = external.

## B. AF structures for Domainome
* Source: human proteome **v4** from `/af_data/af2` (= the v4 tar, already extracted) for 502 domains; v6 Swiss-Prot tar for
  15 non-human domains; EBI (v6) for A2RRE5 (absent from v4 human). No model: 3 de novo 'rockdoms' designs and
  P03034 (phage λ cI; viral, not in AFDB). The AF version used by the FoldX runs is not recorded — v4 is the plausible match.
* 19 constructs lie in AF-fragmented proteins (> 2,700 aa); the fragment with the construct most central is used
  (UniProt = fragment resnum + 200·(k−1)).
* Construct crops: chain A, renumbered to UniProt (= FoldX `pdb_resnum`), pLDDT in B-factor, no relaxation.
* **pLDDT over measured positions:** median 90.2, IQR 84.6–94.2, 5th pct 67.5; 6.1 % of positions (31,918 variants) < 70,
  1.3 % < 50. Per-construct fraction < 70: median 2.2 %, 90th pct 17 %; 43 constructs > 20 %, 2 > 50 %.
  Low-pLDDT sites are mutation-tolerant (median score −0.08 at pLDDT < 70 vs −0.24 at > 90; ρ(score, pLDDT) = −0.15).
* **FoldX alignment:** construct-run key (`entity_id`, `position_1based`, `wt_aa`, `mut_aa`) = (`protein_id`,
  `domainome_position`, `wt`, `mut`): all 509,388 FoldX rows match our table (493 domains); FoldX `pdb_resnum` =
  our `pdb_resnum` in 100 % of matched rows, chain A. Full-length run key: `entity_id` = `uniprot`. Flags
  `in_foldx_construct`, `in_foldx_full`. FoldX lacks 29 domains (designs, P03034, both Q13472, all 19 constructs in fragmented
  proteins, 4 others).
* **v4 vs v6** (independent re-prediction): over 497 constructs with identical sequence, CA RMSD median 0.13 Å, 95th pct
  1.0 Å; the 6 constructs > 2 Å all have mean pLDDT < 70 (ρ(RMSD, pLDDT) = −0.60). 3 constructs changed sequence in v6.
  v6 crops are kept as a second AF model for robustness checks.

## C. Paired crystal vs AF
* Used the data-HT agent's canonical `fireprot.csv`, `s669.csv`, `ssym.csv` (raw FireProt only as fallback). UniProt from
  table or SIFTS. Xtal chain (observed residues) aligned to the AF sequence; site mapped through the alignment.
* WT mismatch: **0** on the xtal side (4,755 mapped rows) and **0** on the AF side (4,012 mapped rows).
  Usable: FireProt 3,078/3,403 (98 proteins), S669 629/669 (80), Ssym 305/684 (12; 152 are reverse rows where the AF model
  is the mutation's end state).
* Losses are almost all **viral proteins absent from AFDB**: T4 lysozyme P00720 (308 FireProt + 368 Ssym rows), M13 g3p,
  TEV, DENV, P22, HIV, phage f1 g5p. Plus 17 FireProt rows for GB1 M1 (FireProt maps it to protein G residue 1 — wrong) and
  one S669 antibody chain without UniProt.
* FireProt's own UniProt `position` disagrees with the alignment for 1 % of rows (isoform/version differences in 6 proteins);
  WT still matches at the aligned residue — use the alignment.
* Benchmark proteins are well predicted: site pLDDT median 97.7 (FireProt), 93.9 (S669), 98.5 (Ssym); < 1 % of sites < 70.
  Local CA RMSD (10 Å sphere, local superposition) median 0.24 / 0.40 / 0.26 Å; > 1 Å for 6 % / 16 % / 4 % of sites.
  Global RMSD > 2 Å for 44/351 entities (hinges, NMR, latent PAI-1). ρ(pLDDT, local RMSD) = −0.44.
  So this set tests side-chain/packing differences of confident models; low-confidence behaviour must come from Domainome.

## D. Additional abundance DMS
Seven ProteinGym v1.3 abundance assays on full-length human proteins (VAMP-seq PTEN/TPMT/CYP2C9, NUDT15, PRKN, GCK,
KRAS aPCA), 37,947 variants, AF v4 full-length models. `score` = DMS_score (higher = more abundant); WT vs AF mismatch 0;
KRAS assay is isoform 4B → 216 isoform-specific variants dropped after alignment. Score falls with burial in all seven
(ρ −0.20 to −0.50). PTEN and PRKN have ~20 % low-pLDDT sites.

## Surprises / caveats
* Domainome is not "ΔΔG": the only public energies are scaled family-level MoCHI fits. Fit to `score` with rank metrics
  or a saturating link; MoCHI values are a secondary, model-derived target.
* Q13472_PF06839_896 off-by-one; GB1 M1 mis-mapped in FireProt; FireProt UniProt positions differ for 6 proteins.
* Constructs are excised domains assayed alone, but the AF models come from full-length predictions (context-dependent
  pLDDT/packing). Crops remove the context; whether to relax the crop is an engine decision.

## Access issues (flagged)
* `alphafold.ebi.ac.uk/files/...` returns **403 without a browser-like User-Agent** and **429** under parallel bursts; the
  API intermittently 403s. Worked around by setting a UA header, retries and serial downloads (`afdb_io.py`). Only 2 models
  were downloaded (A2RRE5, P03322 segment model); everything else came from the local tars.
* Viral proteins (T4 lysozyme etc.) have no AFDB model. Getting them back means running a structure predictor (ColabFold),
  which this project has not set up — decision for the PI.
* `unzip` is not installed; Python `zipfile` is used instead.
