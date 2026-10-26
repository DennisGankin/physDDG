# domainome — Human Domainome 1 abundance DMS (aPCA) on AlphaFold structures

**Source.** Beltran, Jiang, Shen & Lehner (2025) *Site-saturation mutagenesis of 500 human protein domains*,
Nature 637:885 (doi 10.1038/s41586-024-08370-4). Input: `/workspace/Domainome/Extended_data_Table_2_fitness_scores_normalized_domainranks.txt`
(= Supplementary Table 2, Zenodo 10.5281/zenodo.14356805 v5). MoCHI table: Supplementary Table 4 from the same Zenodo
record (`scripts/20_download_domainome_supp.sh`).
**Scripts.** `20_download_domainome_supp.sh` → `21_index_af_tars.py` → `22_domainome_structures.py` → `23_domainome_table.py`.
**Files.** `domainome.csv` / `domainome.parquet` (git-ignored, regenerable), `domainome_structures.csv` (one row per domain),
`domainome_residue_plddt.csv` (per construct residue).

## Counts
| | n |
|---|---|
| rows in source file | 602,882 |
| missense variants with fitness (kept) | **536,164** in 522 domains / 431 UniProt (519 natural + 3 de novo 'rockdoms') |
| variants with an AF structure | 532,501 (518 domains) |
| variants without structure | 3,663: 3 de novo designs (no AF DB entry), P03034_PF01381_4 (phage lambda cI; not in AF DB, EBI 404) |
| variants with MoCHI family ΔΔG | 328,078 (358 domains) |
| variants in FoldX native construct run | 509,388 (493 domains; all FoldX rows matched) |

## Numbering
* `domain_ID` = `<UniProt>_<Pfam>_<start>`; `<start>` = UniProt position of the construct's first residue.
* `position` (kept as `domainome_position`) is UniProt numbering and runs start..start+len(aa_seq)−1 for every domain.
  Checked against the AF model sequences: 0 WT mismatches in 532,501 variants (29,916 residues).
* **One exception:** `Q13472_PF06839_896` — the construct sequence occurs at UniProt 897, not 896 (all 43 residues
  mismatch at the nominal offset, 0 at +1). For it `seq_pos = domainome_position + 1`. Not in the FoldX runs.
* `seq_pos` = UniProt position (use this). `pdb_resnum` = `seq_pos` (construct crops are renumbered to UniProt numbering).
  `construct_pos` = 1-based index in the assayed construct.
* `af_full_resnum` = residue number in the unchanged AF file `af_full_structure` (= `seq_pos` − 200·(k−1) for fragment Fk).

## Structures
* `structure` = `data/structures/domainome/af_construct/<domain_ID>.pdb`: AF model cropped to the assayed construct,
  chain A, UniProt numbering, B-factor = pLDDT. `af_full_structure` = unchanged copy of the AF model (fragment) used.
* AF source: human proteome **v4** (`/af_data/af2`, 502 domains); AF DB **v6** for the 15 non-human Swiss-Prot entries
  (`swissprot_pdb_v6.tar`) and A2RRE5 (absent from the v4 human set; EBI download). `af_version` column records it.
  **v4 and v6 human coordinates differ** (re-predicted; e.g. max atom deviation 48–77 Å on whole chains, mostly in
  disordered parts), so the version matters; the FoldX runs' AF version is not recorded in /foldx_native_clean.
* 19 domains sit in proteins > 2,700 aa (AF fragments); the fragment in which the construct is most central was used.
* pLDDT over measured positions: median 90.2, 5th pct 67.5; 6.1% of positions < 70, 1.3% < 50.
  Per-domain fraction of construct residues with pLDDT < 70: median 0.022, 90th pct 0.17; 43 domains > 20 %, 2 > 50 %.
  Columns: `plddt` (site), `construct_plddt_mean`, `construct_frac_plddt_lt70`.

## Signal and sign
* `score` = `normalized_fitness` (WT = 0; median nonsense per domain ≈ −1). **Higher score = more abundant = more stable**;
  i.e. opposite sign to `ddg`. `score_sigma` = its error. Raw `fitness`, `fitness_sigma`, `mean_input_count` kept.
* `ddg` = NaN: per-domain ΔΔG are **not** released. The paper's MoCHI two-state models were fit per Pfam family
  (≥5 human homologues, mean count > 29) and give *homologue-averaged* energies, rescaled (WT = 0, 2.5th percentile
  magnitude = 1; unitless, not kcal/mol). They are in `ddg_mochi_scaled` (± `ddg_mochi_scaled_sd`, family `mochi_pfam_entry`),
  sign checked: Spearman(Table-4 value, normalized_fitness) = −0.64 → positive = destabilising, no flip needed.
  These are family-level model outputs, not measurements; do not treat them as per-domain ground truth.

## Filters
None beyond the paper's: the file already holds only the 522 retained domains (quality_rank 1–522; retained if rank ≤ 600,
replicate r > 0.485, > 50 % variants measured, minus 16 non-two-state domains). Dropped: nonsense (`*`), WT rows,
27,624 unobserved and 9,851 missense rows without fitness. Optional quality knobs: `score_sigma` (median 0.088),
`mean_input_count` (13 % ≤ 29, the paper's MoCHI threshold), `quality_rank` (1 = best domain).

## Split / cluster
`split` = external for all (abundance proxy, not ΔΔG). `cluster` = Pfam ID (129 families; design IDs for rockdoms) —
use for grouped CV; cross-dataset homology clustering is item 2.4.

## Joining FoldX (native FoldX 5.1, `/foldx_native_clean/results`)
* Construct run: `foldx_native_domainome_construct.csv` key (`entity_id`, `position_1based`, `wt_aa`, `mut_aa`) =
  (`protein_id`, `domainome_position`, `wt`, `mut`). Its `pdb_resnum` = `domainome_position` = our `pdb_resnum` for all
  matched rows (chain A). `in_foldx_construct` flags coverage. Missing domains: the 3 designs, P03034, Q13472 (both),
  A0A2R8Y422, P57071, Q14863, Q3SY89 and 18 constructs from AF-fragmented proteins.
* Full-length run: `foldx_native_domainome.csv` key (`entity_id`=`uniprot`, `position_1based`=`domainome_position`, wt, mut);
  `in_foldx_full`.

## Known issues
* aPCA is in-cell abundance of a DHFR-fragment fusion in yeast; it saturates for very destabilising variants (floor ≈ −1) and
  for highly stable domains (WT far above threshold); it also captures degradation signals unrelated to ΔG.
* Overlapping constructs from the same UniProt exist (e.g. O14776 domains share residue 955) — same site, different construct.
* Constructs are excised domains; AF full-length context (pLDDT, packing partners) can differ from the isolated construct.
  Crops remove the rest of the protein but keep AF coordinates (no relaxation).
