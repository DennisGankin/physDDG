# Stability ΔΔG datasets: citations, downloads, sign conventions, leakage

Compiled 2026-10-26 for PhysDDG (physics-based, no DL). Project convention (D1): `ddg` in
kcal/mol, **positive = destabilising** (ΔΔG = ΔG_unfold(WT) − ΔG_unfold(mut)), FoldX-like.
Every dataset below is annotated with the sign of its *released* column and the flip needed.

Legend: **[C]** = URL confirmed with `curl -sIL` (HTTP 200) on 2026-10-26; **[U]** = unverified /
inferred. "n" counts are variants unless stated.

---

## 0. Sign-convention cheat sheet (the only table you must not get wrong)

| Source | Released column | Sign of released column | Flip for PhysDDG? | How verified |
|---|---|---|---|---|
| Tsuboyama 2023 / Megascale | `ddG_ML` | **positive = stabilising** (ΔΔG = ΔG_mut − ΔG_wt) | **yes, ×(−1)** | paper: "positive ∆∆G indicates stabilizing substitutions" (ED Fig. 11b); local Dataset-3 table: mean −0.73, 77% negative |
| Tsuboyama `dG_ML`, `deltaG`, `deltaG_t/c` | ΔG of **unfolding** | positive = folded/stable | n/a (absolute) | dG_ML in local file: mean +2.27, range −1…5 |
| ThermoMPNN Megascale CSVs (`mega_splits` data) | `ddG_ML` unchanged | positive = stabilising | **yes** | same file, `/workspace/data/runtime_tables/megascale_runtime.csv` |
| ThermoMPNN FireProt CSV (`4_fireprotDB_bestpH.csv`) | `ddG` | **positive = destabilising** | no | local: Pearson(ddG, dTm) = **−0.82** on the 55 rows with both; buried-hydrophobic→small/polar mean +2.89 (96% > 0) |
| FireProtDB 1.0 / 2.0 (web + API `DDG`) | `DDG` | **positive = destabilising** | no | API query: T4 lysozyme L99A `DDG=+4.1`, M102K `+6.9`, I3L `−0.4…−1.16` (I3L is mildly stabilising) |
| ThermoMutDB | `ddG` | **positive = stabilising** (paper: "negative ΔΔG values indicate that the mutation has destabilized the protein") | **yes** | Xavier 2021 NAR, Methods |
| ProThermDB | `ΔΔG` as published per entry | mixed / per-publication; ProTherm's own convention is ΔΔG = ΔG_mut − ΔG_wt (positive = stabilising) | **check per row** | Nikam 2021; entries copied from literature |
| protddg-bench (S2648, VB1432, Ssym, PTMUL, BROOM, KORPM, S461, myoglobin, p53) | `DDG` | **negative = destabilising** (ΔG_mut − ΔG_wt) | **yes** | all files mean DDG ≈ −0.6…−1.4, 58–84% negative; Ssym DIR `1AMQA C191Y −2.30` vs INV `+2.30`; README: myoglobin "DDG was calculated changing the sign of the DDG reported in literature" |
| Stability Oracle C2878 / T2837 / cDNA117K | `ddG` | **positive = destabilising** | no | `2LZM L99G = +6.5`, `1BPI F4A = +3.0`, `5AZU I7S = +4.24`; Ssym rows carry the opposite sign to protddg-bench |
| ProteinGym `DMS_score` (Tsuboyama assays) | `ddG_ML` passed through, `raw_DMS_directionality = 1` | **positive = stabilising** (higher score = fitter) | **yes** | `reference_files/DMS_substitutions.csv`: `raw_DMS_phenotype_name = ddG_ML_float`, directionality 1 |
| Domainome 1.0 `fitness` / `normalized_fitness` | abundance fitness | **negative = destabilising/low abundance** (WT = 0, 2.5th pct of missense = −1) | not ΔΔG; use as proxy with sign flip relative to ddg | Beltran 2025 Methods; local file header |
| Domainome Supplementary **Table 4** (MoCHI family ΔΔG) | `mean_kcalmol_scaled` | **positive = destabilising** | no, but **rescale** — the units are not true kcal/mol | Pro has the highest mean; paper's "strongly destabilising" cut = +0.3 |
| Lehner ddPCA / MoCHI tables (Faure 2022, Weng 2024, PDZ/Src 2026) | `f_ddg_pred`, `mean_kcal/mol`, MaveDB `00000115-a-7` | **positive = destabilising** | no | WT ΔG_f negative; Pro largest positive; 80–89% of singles > 0 |
| MGnify Stability 2026 | `deltaG` | ΔG of unfolding, positive = stable | n/a (absolute) | header + `train_dG_from_earlier_processing` |
| VAMP-seq style abundance scores | abundance score | higher = more abundant/stable (WT ≈ 1, nonsense ≈ 0) | proxy only | see §7 |
| Classic sets (S2648, Ssym, Q3421, p53, …) | `DDG` | **depends on the copy** — VariBench/DDGun/KORPM/GeoStab/protddg-bench = negative destabilising; PremPS/ThermoNet(Q3214,Q1744,s_sym,myo,p53)/HotMuSiC-S1626 = positive destabilising | per copy | §5.4 table, checked row by row |

**Practical rule:** Megascale, ThermoMutDB, protddg-bench and ProteinGym need a sign flip;
FireProtDB, Stability Oracle sets, Lehner ddPCA and FoldX do not; the classic benchmarks must be
sign-checked **per downloaded file**, because the same set ships both ways (p53 is the cleanest
example: VariBench `42_variations_in_P53.xlsx` Q104H = +0.24 while PremPS `p53.txt` Q104H = −0.24).

---

## 1. Tsuboyama et al. 2023 — "Megascale" cDNA display proteolysis

**Citation.** Tsuboyama K, Dauparas J, Chen J, Laine E, Mohseni Behbahani Y, Weinstein JJ,
Mangan NM, Ovchinnikov S, Rocklin GJ. *Mega-scale experimental analysis of protein folding
stability in biology and design.* **Nature 620:434–444 (2023)**, doi:10.1038/s41586-023-06328-6.
Preprint: doi:10.1101/2022.12.06.519132. Pipeline code:
https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline [C].

**Download.** Zenodo record **10.5281/zenodo.7992926** (v2_230420, published 2023-04-21) [C]
https://zenodo.org/records/7992926 [C]. Files (API URL pattern
`https://zenodo.org/api/records/7992926/files/<KEY>/content` [C]):

| File | Size | Contents |
|---|---|---|
| `Processed_K50_dG_datasets.zip` | 1.01 GB | `Tsuboyama2023_Dataset1_20230416.csv`, **`Tsuboyama2023_Dataset2_Dataset3_20230416.csv`**, `Single_DMS_list.csv`, `Double_DMS_list.csv`, `Triple_DMS_list.csv`, heat-map PDFs |
| `K50_dG_tables.zip` | 450 MB | `K50_dG_lib{1..4}.csv` (per-library K50 + ΔG fits) |
| `Raw_NGS_count_tables.zip` | 243 MB | `NGS_count_lib{1..4}.csv` |
| `AlphaFold_model_PDBs.zip` | 14 MB | AF models of all WT domains (local copy: `/workspace/data/downloads/AlphaFold_model_PDBs.zip`, unpacked `/workspace/data/megascale_pdbs/`, 862 PDBs) |
| `Data_tables_for_figs.zip` | 24 MB | per-figure tables (`dG_site_feature_Fig3.csv`, `dG_for_double_mutants_Fig4.csv`, `dG_non_redundant_natural_Fig5.csv`, …) |
| `Pipeline_K50_dG.zip`, `Pipeline_figure_model.zip`, `Pipeline_qPCR_data.zip`, `Blueprints_for_EEHH.zip` | | pipeline notebooks incl. `Data_quality_filtering_script.ipynb` |

The authors ask users to register at https://forms.gle/wuHv8MKmEu4EEMA99.

**Dataset 1/2/3 definitions** (Extended Data Table 1, read off the published table image — exact):

| Dataset | Description | # total seqs | Breakdown |
|---|---|---|---|
| **#1** | all data, no filtering | **1,841,285** | single a.a. mut. 1,046,752 (983 WTs); double+triple 416,274 (725 pairs, incl. 36 triples); scrambles for unfolded model 68,427; Rocklin-2017 rd1–3 36,707; others 273,125 |
| **#2** | all data **for ΔG** (quality-filtered scans) | **776,298** | singles 566,180 (**478 WTs**); doubles 210,118 (559 site pairs) |
| **#3** | all data **for ΔΔG** (WT ΔG < 4.75 kcal/mol so stabilising mutants stay in range) | **607,839** | singles 448,788 (**412 WTs**); doubles 159,051 (496 pairs) |
| **#4** (Fig. 3) | Dataset 3 minus modified/variant WTs | 325,132 | 365 WTs, 17,093 sites |
| **#5** (Fig. 5) | non-redundant natural domains | 113,572 | 104 WTs |

Datasets 2 and 3 live in the **same file** `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`:
every row has a ΔG estimate; only Dataset-3 rows have a tabulated ΔΔG. Rows with `_ML`
columns = `"-"` are the ones filtered out ("ML" = machine-learning-ready). Computing ΔΔG
yourself for Dataset-2-only rows is possible but biased toward destabilising (WT too stable).

**Column meanings.**
- `log10_K50_t`, `log10_K50_c` — protease concentration giving half-maximal cleavage, trypsin /
  chymotrypsin (the raw observable); `*_95CI` = Bayesian 95% CI width.
- `deltaG_t`, `deltaG_c` — ΔG of unfolding fitted from trypsin-only / chymotrypsin-only data;
  `deltaG` — joint fit to both proteases. Each has a `_95CI`.
- `dG_ML`, `ddG_ML` — the quality-filtered, ML-ready versions of ΔG / ΔΔG. `dG_ML` is the string
  `"<-1"` or `">5"` when the 95% CI lies entirely outside the dynamic range, and `"-"` where the
  data failed Stage-3 mutant-level filtering.
- `mut_type` (e.g. `E16Q`, `wt`, `insA12`, `del12`), `WT_name` (PDB/design id + `.pdb`), `aa_seq`.

**Sign.** ΔΔG = ΔG_mut − ΔG_wt, so **positive `ddG_ML` = stabilising**. Confirmed in the paper
("All ∆∆G measurements are in the original wild-type background; positive ∆∆G indicates
stabilizing substitutions", ED Fig. 11b legend) and numerically on the local curated copy
`/workspace/data/runtime_tables/megascale_runtime.csv` (271,231 singles: mean −0.73, median
−0.48, 76.9% negative, min −5.72, max +3.10). **Flip for PhysDDG.**

**Conditions.** Unfolding at **pH 7.4, 298 K (25 °C, room temperature)**, cell-free; all domains
assayed under identical conditions (unlike ProTherm-derived sets).

**Dynamic range / saturation.** ΔG resolvable only in **−1 to 5 kcal/mol**; all ΔG values are
clipped to that range before analysis, and the assay's own limit is ~5 kcal/mol. Therefore ΔΔG is
compressed/saturated for very stable WTs — hence the Dataset-3 WT ΔG < 4.75 filter. ThermoMPNN
reports the usable ΔΔG span as −3 to +5 kcal/mol vs −9 to +12 for FireProt data.

**Confidence.** For ΔG estimates inside −1…5, the median 95% CI width is **0.14 kcal/mol** and
99.9% are < 0.96 kcal/mol. Low-confidence points (95% CI > 0.5) are marked in heat maps but kept.
Trypsin-vs-chymotrypsin ΔG agreement: **Pearson 0.94 ± 0.04 (median ± s.d.) over 478 Dataset-2
domains**. K50 replicate reproducibility R = 0.97 (trypsin) / 0.99 (chymotrypsin) for ~84% of
806,640 sequences after confidence/dynamic-range filtering. Agreement with purified-protein
literature ΔG: Pearson 0.75–0.96 per domain, with systematic intercept offsets attributed to
temperature differences.

**Quality filters (how the "high-quality subset" is defined).** Domain-level groups G0–G11
(`Data_quality_filtering_script.ipynb`, re-read 2026-10-26):
- **G0** passes everything; **G1** passes everything but WT ΔG > 4.75 kcal/mol (stabilising
  mutants unresolvable). G0+G1 = Dataset 2/3 source.
- G2 poor expression (low NGS counts); G3 WT too unstable; G4 inconsistent WT ΔG; G5 poor
  trypsin–chymotrypsin correlation; G6 poor T–C slope; G7 too many stabilising mutants (suggests
  non-specific intermolecular stabilisation or a mis-measured WT); G8/G9 >1 Cys (G8 = the two
  good ones, G9 removed); G10 poor T–C intercept (ΔG=0 disagreement; ΔΔG still usable);
  G11 probable cleavage from the folded state.
- Metric thresholds used in code: high-confidence subset requires `deltaG_t_95CI ≤ 0.665` **and**
  `deltaG_c_95CI ≤ 0.665` (= 0.5 × 1.33); a scan needs ≥10 such points; K50 correlation computed
  on points with `log10_K50_*_95CI < 0.5`; ODR slope/intercept of ΔG_t vs ΔG_c; cysteine- and
  loop-cleavage heuristics (K/R and F/W/Y site-specific dK50 tests) remove individual mutants
  (Stage 3) even in G0/G1 domains.

**Composition.** 983 WT sequences assayed: **331 natural + 148 de novo designed domains** survive
into the curated 776k set (40–72 aa); designs = Rosetta-based (several topologies) plus trRosetta
hallucination; 121 "destabilised WT backgrounds" (modified WTs, ~12% of assayed proteins) exist to
bring over-stable domains into range. Mutation types include all 19 substitutions **plus single
deletions and Gly/Ala insertions** at every position. **Doubles: 210,118 double mutants at 559
site pairs across 190 domains** (Dataset 2), 159,051 doubles in Dataset 3.

**Known biases.** Only small (<72 aa), monomeric, cooperatively folding, disulfide-free domains;
AlphaFold models (not crystal structures) for all; 5 kcal/mol ceiling; ΔΔG inferred through a
kinetic model rather than measured; de-novo designs are ~1/3 of the data and have never been
structurally validated; GC-rich/aggregation-prone sequences excluded by construction.

---

## 2. ThermoMPNN splits (and other Megascale splits)

**Citation.** Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. *Transfer learning to leverage
larger datasets for improved prediction of protein stability changes.* **PNAS 121(6):e2314853121
(2024)**, doi:10.1073/pnas.2314853121.

**Where the splits live.** https://github.com/Kuhlman-Lab/ThermoMPNN `dataset_splits/` [C]:
- `mega_splits.pkl` (16 kB) [C] — dict of WT names (`*.pdb`), keys: `train` (239), `val` (31),
  `test` (28), `train_s669` (232), and `cv_train_{0..4}` / `cv_val_{0..4}` / `cv_test_{0..4}`.
  (Loaded and counted locally; **239/31/28 confirmed**.)
- `megascale_clusters.tsv` (298 rows → **162 MMseqs2 clusters**), `fireprot_clusters.tsv`
  (100 rows → 83 clusters), `megascale_vs_fireprot_homologues.m8` (41 hits),
  `s669_vs_megascale_homologues.m8` (8 hits). All [C] via
  `https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main/dataset_splits/<file>`.

**Megascale subset used.** Start from Dataset 3 (`ddG_ML != "-"`) = **607,839** rows → remove
insertions, deletions and multi-mutants → **391,090** single substitutions → remove the modified
("destabilised") WT backgrounds → **272,712 mutations over 298 proteins** (181 natural + 90
trRosetta-hallucination + 19 RosettaDesign; all natural WTs have mean pLDDT > 0.75).

**How the splits were made.** MMseqs2 clustering at **25% sequence identity**; Megascale and
FireProt cross-referenced, and any cluster with a cross-dataset homologue forced into the
*training* side of both; remaining clusters randomly assigned ≈80/10/10 by mutation count. FireProt
proteins with >250 measurements forced to train. A 4th FireProt split, **"FireProt-HF"
(homologue-free, 89 proteins / 2,578 mutations)**, keeps only FireProt data with no Megascale
homologue — that is the set to use when scoring a Megascale-trained model.

Row counts of the splits applied to the 272,712-row table (computed locally):
train **215,731**, val **27,328**, test **28,172**, `train_s669` **208,677** (sum of
train/val/test = 271,231 = all non-WT rows; the paper quotes N = 28,312 for the test set, a
~140-row difference, probably a slightly different pre-release curation — noted, unresolved).

**Known homology leakage inside the ThermoMPNN Megascale split.** Our earlier project analysis
found **9/28 test proteins have a training homologue**. Re-checked here with BLOSUM62 local
alignment (identity over the aligned region, no E-value filter):

| test domain | best train hit | aln length | % id |
|---|---|---|---|
| 1TUC | 2CDT | 43 | **98%** |
| 2KXD | 2CDT | 42 | **98%** |
| r18_3_TrROS_Hall | r11_951_TrROS_Hall | — | 57% (crude min-length id) |
| 2K1B | 2N88 | 48 | 44% |
| 2K28 | 2N88 | 47 | 40% |
| 6EWU | 6YSE | 28 | 46% |

So the 25% MMseqs2 clustering did **not** catch at least two near-identical pairs (1TUC/2KXD vs
2CDT, 98% identity over 42–43 residues). Treat the ThermoMPNN Megascale test set as
*not* homology-clean; for PhysDDG report a strict subset excluding these.

**Other Megascale splits in the literature.**
- **RaSP** (Blaabjerg et al. 2023 eLife; https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)
  is trained on *Rosetta* ΔΔG for 35 DMS proteins, not on Megascale; ThermoMPNN retrained it on
  Megascale for comparison.
- **Stability Oracle** uses `cDNA117K` = **116,641** single mutations on **116 PDB-matched natural
  domains** from Tsuboyama Dataset #1, filtered to <30% identity to its test set T2837 (closest
  pair 25.4%) — see §5.
- **ProteinGym** splits Tsuboyama per-domain into 64 separate "Stability" DMS assays (§8), with
  its own random / modulo / contiguous CV folds (`cv_folds_singles_substitutions.zip`).
- **ProtDDG-Bench** (Capriotti, Montanucci, Fariselli 2023) provides blastclust-25%/MMseqs-25%
  cluster-respecting folds for the classic sets (§5).

---

## 3. FireProtDB

**v1.0 citation.** Stourac J, Dubrava J, Musil M, Horackova J, Damborsky J, Mazurenko S, Bednar D.
*FireProtDB: database of manually curated protein stability data.* **NAR 49(D1):D319–D324
(2021)**, doi:10.1093/nar/gkaa981. Web: https://loschmidt.chemi.muni.cz/fireprotdb [C].

- Size (v1.0): **13,274 entries / 237 proteins** at publication (8,189 from ProTherm; rest from
  manual literature curation 18%, ProtaBank 28%, VariBench 53%, own records 1%); later quoted as
  **15,987 mutations**. Mixture of ΔΔG and ΔTm; many entries have only one of the two.
- Composition by effect: 43% destabilising (ΔTm < −1 or ΔΔG > 1), 14% stabilising, 43% neutral.
- Measurement types: chemical/thermal denaturation (CD, DSC, fluorescence), heterogeneous pH and
  temperature per entry; `pH` and `tm` columns carry the condition.
- **Sign: positive ΔΔG = destabilising.** Verified two ways: (i) FireProtDB 2.0 API rows for T4
  lysozyme give `DDG(L99A) = +4.1`, `DDG(M102K) = +6.9`, `DDG(I3L) = −0.4…−1.16` — L99A/M102K are
  the textbook destabilising cavity mutants; (ii) in the ThermoMPNN-curated export
  (`4_fireprotDB_bestpH.csv`, 4,997 rows, 3,438 with ΔΔG) **corr(ddG, dTm) = −0.82**.
  So FireProtDB needs **no flip** for PhysDDG.
- Known biases: 24.4% of the ThermoMPNN-curated set are mutations **to alanine** (vs 7.8% natural
  frequency); two proteins (staphylococcal nuclease, protein G) supply ~1/3 of all rows; 85/100
  proteins have <50 measurements; pH/temperature heterogeneity; duplicated measurements of the
  same mutation across publications; ProTherm-inherited annotation errors.

**v2.0 citation.** Musil M, Borko S, Planas-Iglesias J, Lacko D, …, Bednar D, Damborsky J.
*FireProtDB 2.0: large-scale manually curated database of the protein stability data.*
**NAR 54(D1):D409–D418 (2026)** (online 2025-11), doi:10.1093/nar/gkaf1211; PMC12807726.
- **12,923,886 measurements → 5,465,664 experiments on 2,762 unique proteins** (vs 15,987
  mutations in v1). 42,912 absolute (WT ΔG/Tm) + 12,880,974 relative (mutant) measurements;
  **47,173 insertions, 50,383 deletions, 4,778,176 single substitutions, 186,194 multiples**.
- It is now **an aggregator**: ProTherm (6,691 single + 1,470 multiple + 12,049 WT entries after
  cleaning), **Megascale** (contributing 22,901 deletions, 47,173 insertions, 486,666 singles,
  184,642 doubles + 148 de novo proteins) and **Human Domainome 1** (Supplementary/Extended tables
  2 and 5 from https://zenodo.org/records/13629491), plus VariBench dataset annotations.
  → **FireProtDB 2.0 is no longer independent of Megascale or Domainome**: using it as a test set
  for a Megascale-fitted model is leakage unless you filter on `SOURCE_DATASET`.
- Effect classes (their thresholds): stabilising if ΔΔG < −0.5 kcal/mol or normalized ΔΔG < −0.3
  or ΔTm > 1 °C or fitness > 0.3; destabilising if ΔΔG > 0.5 or normalized ΔΔG > 0.3 or
  ΔTm < −1 °C or fitness < −0.3. Totals: 3,454,310 neutral, 1,389,567 destabilising, 218,044
  stabilising. They cite an average experimental error of ~0.48 kcal/mol and ρ(ΔΔG, ΔTm) ≈ 0.71.
- **API** [C] https://loschmidt.chemi.muni.cz/fireprotdb/api-docs/ — `GET /api/search?query=<JSON>
  &format=csv|tsv|jsonl|json&limit=&offset=&sort=`; csv/tsv/jsonl have **no row limit**, json is
  capped at 500. Working example (tested, HTTP 200, returns a 53-column CSV):
  `https://loschmidt.chemi.muni.cz/fireprotdb/api/search?query=%7B%22fulltext%22%3A%22lysozyme%22%7D&format=csv&limit=4000`
  Columns include `DDG, DG, DTM, TM, DOMAINOME_DDG, DOMAINOME_FITNESS, TRYPSIN_ML, CHYMOTRYPSIN_ML,
  MEGASCALE, SOURCE_DATASET, REFERENCING_DATASET, PH, EXP_TEMPERATURE, METHOD, UNIPROTKB, WWPDB`.
  `SOURCE_DATASET` lets you exclude Megascale/Domainome rows; `REFERENCING_DATASET` says which
  published benchmark (VariBench etc.) a row belongs to.
  Also `/api/sequences/{id}`, `/api/mutants/{id}`, OpenAPI at `/fireprotdb/fireprotdb-openapi.json`.

**ThermoMPNN's FireProt set** (what most papers mean by "FireProt" as a benchmark):
Zenodo **10.5281/zenodo.8169289** [C] (`fireprot_upload.zip`, 5.4 MB; local copy at
`/workspace/data/downloads/fireprot_upload.zip`, unpacked tables at
`/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv`, structures `/workspace/data/fireprot_pdbs/`
= 130 PDBs).
- Built by matching FireProtDB sequence-level ΔΔG to RCSB structures, dropping duplicates and
  rows with missing info, and keeping the measurement **nearest to biological pH** per mutation.
- **3,438 mutations / 100 proteins** (the released CSV has 4,997 rows of which 3,438 have a ΔΔG;
  125 distinct corrected PDB ids in the full file).
- Splits in `fireprot_splits.pkl` (protein-level; row counts with a non-null ddG recomputed here):
  **train 57 proteins / 2,686 ddG rows, val 15 / 402, test 28 / 350**, plus
  **`homologue-free` = 89 proteins / 2,578 ddG rows** (= FireProt-HF, matches the paper's 2,578).
  Note train+val+test ddG rows = 3,438 = the whole set; HF is a *different* (overlapping) slice,
  so never average HF and test numbers.
- `ddG` column = **mutant − WT with positive = destabilising** (Zenodo description says
  "ddG value for mutation (mutant - WT)"; the numbers confirm destabilising-positive).
  `pdb_position` is **0-based** into the PDB file — beware when joining.
- Dynamic range −8.6 … +12.0 kcal/mol in the local copy (mean +1.06, 48.6% positive).

---

## 4. ProTherm, ThermoMutDB, ProThermDB

| DB | Citation | Size | Sign of ΔΔG | Download |
|---|---|---|---|---|
| **ProTherm** (legacy, frozen ~2013) | Kumar MDS et al., NAR 34:D204 (2006), doi:10.1093/nar/gkj103 | 17,113 entries (7,969 single, 1,139 double, 928 multiple, 7,077 WT) | ΔΔG = ΔG_mut − ΔG_wt, **positive = stabilising**, but entries were transcribed with inconsistent signs — this is the origin of most sign bugs downstream | superseded; mirrors via ProThermDB/FireProtDB; original site https://web.iitm.ac.in/bioinfo2/prothermdb/ [C] |
| **ThermoMutDB** | Xavier JS, Nguyen T-B, Karmarkar M, Portelli S, Rezende PM, Velloso JPL, Ascher DB, Pires DEV. *ThermoMutDB: a thermodynamic database for missense mutations.* **NAR 49(D1):D475–D479 (2021)**, doi:10.1093/nar/gkaa925 | **>14,669** experimental data points; +83% unique mutations over ProTherm; 204 new proteins; all 1,902 ProTherm references re-checked (829 kept), 5,654 new points from 393 new papers | **"negative ΔΔG = destabilised, positive = more stable"** (explicit in Methods) → **flip for PhysDDG** | https://biosig.lab.uq.edu.au/thermomutdb/downloads [C] (CSV/JSON + structures; old URL biosig.unimelb.edu.au redirects); REST API |
| **ProThermDB** | Nikam R, Kulandaisamy A, Harini K, Sharma D, Gromiha MM. *ProThermDB: thermodynamic database for proteins and mutants revisited after 15 years.* **NAR 49(D1):D420–D424 (2021)**, doi:10.1093/nar/gkaa1035 | **31,580 entries** (+84% over ProTherm 2006): 12,050 WT, 16,028 single, 2,046 double, 1,456 multiple | per-publication (ΔΔG and ΔTm as reported); **verify per row** | https://web.iitm.ac.in/bioinfo2/prothermdb/index.html [C], downloads page `Downloads.html` [C] |

Caveats common to all three: single measurements, heterogeneous pH (2–11) and temperature
(5–80 °C), denaturant-dependent m-value extrapolation, strong protein bias (lysozyme, barnase,
staphylococcal nuclease, CI2, protein G), alanine bias, and duplicated rows between the databases.
S669's authors had to correct signs or values for **20% of the 669 variants they extracted from
ThermoMutDB** — a direct measure of literature-database error rate.

---

## 5. Classic benchmarks (S2648, Ssym, S669, …)

### 5.1 One-stop download: ProtDDG-Bench
https://github.com/protddg-bench/protddg-bench [C] (Capriotti, Montanucci, Fariselli 2023).
Raw files: `https://raw.githubusercontent.com/protddg-bench/protddg-bench/HEAD/<DIR>/<file>` [C].
Format `#SET,CLID,PDB,MUT,DDG[,pH,T]`; **DDG negative = destabilising**; `CLID` = homology cluster
(blastclust −S 25 −L 0.5 for the 545-chain set; MMseqs 25% for KORPM), so grouped CV is free.

| Set | Rows (direct) | PDB chains | Clusters | mean DDG | % DDG<0 | File |
|---|---|---|---|---|---|---|
| S2648 | 2,648 | 132 | 113 | −1.01 | 77% | `S2648/s2648-10fold-split-{0..9}.csv` |
| VB1432 (VariBench) | 1,432 | 79 | 65 | −1.02 | 73% | `VB1432/vb1432-10fold-split-{0..9}.csv` |
| Ssym (direct half) | 342 (+342 inverse = 684; 634 in their file) | 15 | 13 | −0.98 | 73% | `SSYM/ssym-5fold.csv` (has `DIR/INV`, pH, T) |
| S461 (KORPM test) | 461 | 48 | — | −1.14 | 84% | `KORPM/s461-korpm.csv` |
| KORPM train | 2,371 | 181 | 129 | −0.69 | 58% | `KORPM/korpm-10fold-split-{0..9}.csv` |
| Broom | 605 | 58 | 50 | −1.36 | 76% | `BROOM/broom-5fold.csv` |
| Myoglobin | 134 | 1 (1BZ6) | 1 | −0.63 | 72% | `MYOGLOBIN/myoglobin.csv` |
| p53 | 42 | 1 (2OCJ) | 1 | −1.43 | 74% | `P53/p53.csv` |
| Ptmul (multi-point) | 914 | 90 | 76 | −0.85 | 66% | `PTMUL/ptmul-5fold.csv` (`MUTS` is `;`-separated, `NMUT` 2–…) |

All counts/means computed locally from the downloaded files. The repo also ships matched training
files that already exclude the test homologues (`train-s2648-test-ssym`, `train-korpm-nos461`,
`train-vb1432-test-ptmul`, …) — use these rather than rolling your own.

### 5.2 S669 / S461
S669: Pancotti C, Benevenuta S, Birolo G, Alberini V, Repetto V, Sanavia T, Capriotti E,
Fariselli P. *Predicting protein stability changes upon single-point mutation: a thorough
comparison of the available tools on a new dataset.* **Brief Bioinform 23(2):bbab555 (2022)**,
doi:10.1093/bib/bbab555. Data: Zenodo **10.5281/zenodo.7568094** [C]
(`S669.zip`, 44 MB, includes WT **and** modelled mutant structures) [C].
- 669 variants extracted from ThermoMutDB with **<25% identity to S2648 ∪ VariBench** — i.e. built
  explicitly as a held-out set for anything fitted on those. 94 protein chains; ~24% of the
  candidate 900 were dropped (kinetics, binding, multiple mutations); signs/values corrected for
  20% of the kept ones. 443 variants at physiological T/pH (293–313 K, pH 6–8), 226 outside.
  196 variants from NMR structures (23 proteins) vs 473 from X-ray (71 proteins).
- Also from that paper: **Ssym+** = Ssym extended with 10 more variants and 10 new mutant
  structures = 704 entries, and reverse variants for S669 (1,338 total direct+reverse).
- S461 (Hernández et al. 2023, KORPM) is the curated 461-variant subset — see §5.4.

### 5.2b Ssym
**Citation.** Pucci F, Bernaerts KV, Kwasigroch JM, Rooman M. *Quantification of biases in
predictions of protein stability changes upon mutations.* **Bioinformatics 34(21):3659–3665
(2018)**, doi:10.1093/bioinformatics/bty348.
- **342 direct + 342 inverse** mutations over **15 wild-type proteins / 357 PDB entries** — the
  defining feature is that *both* the WT and the mutant crystal structure exist (≤2.5 Å), so
  inverse ΔΔG can be predicted from a real mutant structure. 13 homology clusters.
- Drawn from ProTherm. Conditions skew acidic (pH 3, 5.4, 2.7) and hot (99 rows at 65 °C) because
  T4 lysozyme thermal-denaturation series dominate it.
- Released sign (original `bty348_dataset.xlsx`, KORPM `Ssym.txt`, protddg-bench):
  **negative = destabilising** (1L63 L99A = −4.5). ThermoNet's `s_sym.txt` copy is **flipped**.
- **Not independent of S2648**: 194/342 direct variants and 13/15 proteins are in S2648 (§10.1).
- Use it for the antisymmetry check (⟨ΔΔG_dir + ΔΔG_inv⟩ and Pearson(dir, −inv)), not as an
  independent accuracy benchmark.

### 5.3 Stability Oracle sets (T2837 / C2878 / cDNA117K) — "Megascale cleaned"
**Citation.** Diaz DJ, Gong C, Ouyang-Zhang J, Loy JM, Wells J, Yang D, Ellington AD, Dimakis AG,
Klivans AR. *Stability Oracle: a structure-based graph-transformer framework for identifying
stabilizing mutations.* **Nat Commun 15:6170 (2024)**, doi:10.1038/s41467-024-49780-2.
**Download** https://github.com/danny305/StabilityOracle `data/datasets/` [C]:
`T2837.csv` (526 kB) [C], `C2878.csv` (469 kB) [C], `cdna117K.csv` (12.6 MB) [C].
Columns: `dataset, uniprot_id, pdb_code, chain_id, position, wtAA, mutAA, from, to, rel_rsa, ddG,
sequence`. **ddG positive = destabilising** (verified: 2LZM L99G +6.5, 1BPI F4A +3.0).

| Set | n | proteins | Built from | Composition (by `dataset` column) |
|---|---|---|---|---|
| **T2837** (test) | 2,837 | 129 | p53 + myoglobin + Ssym + S669 → dedup (keep max |ΔΔG|) = T1187 → remove 30%-identity overlap with C5266 → T2226 → merge+dedup | t2226 1,695; s669 648; ssym 339; myoglobin 113; p53 42 |
| **C2878** (train) | 2,878 | 151 | Q1744 + O2567 + S2648 + FireProtDB → dedup = C5266 → remove 30% identity to T2837 | q1744 1,488; fireprotdb 1,303; s2647 53; o2567 34 |
| **cDNA117K** (train) | 116,641 | 116 | Tsuboyama **Dataset #1** natural domains with a WT PDB id, filtered to <30% identity vs T2837 (closest 25.4%) | — |

Verified locally: **T2837 ∩ C2878 = 0 variants and 0 shared PDB codes**. Augmentations:
"thermodynamic permutations" (TP) expand n measurements at a site into n(n−1) valid ΔΔGs
(cDNA117K+TP = 2.02 M; C2878+TP = 18,912; T2837+TP = 10,557) and are the recommended evaluation
set for stabilising-mutation precision/recall. All C2878 proteins are homologous (>30%) to some
cDNA117K protein, so C2878 adds no new protein space on top of Megascale.

### 5.4 Every other named set, pinned to a primary source

| Set | Origin (primary source) | Size | Derived from | Nature / caveat |
|---|---|---|---|---|
| **S2648** | Dehouck Y et al., *Bioinformatics* 25:2537 (2009), PoPMuSiC-2.0, doi:10.1093/bioinformatics/btp445 | 2,648 singles, 131–132 proteins | ProTherm, hand-checked; heme proteins, pseudo-WT constructs, Pro mutations and |ΔΔG|>5 excluded | T 0–107 °C (median 25; 66% in 20–30 °C), pH 2–11 |
| **S350** | same paper (random subset used for tool comparison) | 350 mutations, 67 PDBs | S2648 (348/350 matched) | not independent of S2648 |
| **"VariBench" / VB1432** | the stability set usually meant is PON-tstab: Yang Y et al., *IJMS* 19:1009 (2018); the VariBench *database* is Sasidharan Nair & Vihinen, *Hum Mutat* 34:42 (2013) | 1,564 variants / 99 proteins (DDGun copy: 1,420, 79 PDBs; protddg-bench VB1432: 1,432) | ProTherm, hand-curated | has T, pH |
| **S1626** | Pucci F, Bourgeas R, Rooman M, *Sci Rep* 6:23257 (2016) (HoTMuSiC) | 1,626 mutations, 90–95 proteins | ProTherm + literature | **ΔTm, not ΔΔG** (the VariBench copy's "ddG" column is only 646/1,626 filled) |
| **S2000** | Usmanova DR et al., *Bioinformatics* 34:3653 (2018) | ~1,000 WT/mutant structure pairs (PremPS copy 993 pairs, 278 UniProts; GeoStab copy 944) | PDB structure pairs | **no experimental ΔΔG** — antisymmetry testing only |
| **Q3421** | Quan L, Lv Q, Zhang Y, *Bioinformatics* 32:2936 (2016) (STRUM) — *not* iStable/DeepDDG | 3,421 mutations, 148–150 proteins; test subset **Q306** (306 mutations, 32 proteins, <60% id to S2648) | ProTherm | pH 2–11; the "T" column mixes Tm into thermal rows |
| **Q3214 / Q1744** | Li B, Yang YT, Capra JA, Gerstein MB, *PLoS Comput Biol* 16:e1008291 (2020) (ThermoNet) | Q3214 = Q3421 − 207 Ssym-shared; **Q1744** = Q3214 − proteins homologous to Ssym/p53/myoglobin (BLAST e<1e-3) = 1,744 mutations, 127 proteins | Q3421 | the standard "leakage-filtered ProTherm" training set |
| **S461** | Hernández IM, Dehouck Y, Bastolla U, López-Blanco JR, Chacón P, *Bioinformatics* 39:btad011 (2023) (KORPM) | 461 mutations, 48 PDBs | curated subset of S669 | see below |
| **Myoglobin** | Kepp KP, *BBA Proteins Proteom* 1854:1239 (2015) | 134 mutations, PDB 1BZ6 | literature compilation | single protein |
| **p53** | Pires DEV, Ascher DB, Blundell TL, *Bioinformatics* 30:335 (2014) (mCSM) | 42 mutations, PDB 2OCJ (DNA-binding domain) | literature | single protein |
| **PTmul** | Montanucci L et al., *BMC Bioinformatics* 20(S14):335 (2019) (DDGun) | 914 multi-point variants (2–10 substitutions), 90 proteins | ProTherm | the only multi-mutant benchmark here |
| **S250** | Fang J, *Brief Bioinform* (2019), doi:10.1093/bib/bbz071 | 125 forward + 125 reverse, 9 proteins (T4 lysozyme 98, human lysozyme 90, …) | structure pairs; **218/250 are in Ssym** | tiny, highly redundant |
| **S921** (a.k.a. "S971" in GeoStab) | Chen Y et al., *PLoS Comput Biol* (2020) (PremPS), doi:10.1371/journal.pcbi.1008543 | 921 mutations, 54 proteins | combined test set kept disjoint from PremPS training set S5296 | naming ambiguity unresolved |
| **S783** | Xu Y, Liu D, Gong H, *Nat Comput Sci* (2024) (GeoStab), doi:10.1038/s43588-024-00716-2 | 783 mutations, 55 proteins | subset of S921 minus S2648 overlap and reverses | 691/783 are **inside S8754** |
| **S8754** | GeoStab (same paper) | 8,754 mutations, 301 proteins | ProThermDB + ThermoMutDB (July 2021), proteins >25% id to S669 removed | largest curated classic set; T 0–111 °C, pH 0–11 |
| **S571** | GeoStab | 571 mutations, 37 proteins | ProThermDB + ThermoMutDB, <25% id to S1626 | **ΔTm**, not ΔΔG (ΔTm train set S4346 ships with it) |
| **S4038** | Zheng et al., *Protein Sci* 33:e4861 (2024), doi:10.1002/pro.4861 | 4,038 singles (>1,000 stabilising), no S2648 overlap | ThermoMutDB + FireProtDB + ProThermDB | **no public download found [U]** |

**Sign of each released copy** (the same benchmark ships with opposite signs depending on source —
always check the file you actually downloaded):

| Copy | Sign | Evidence |
|---|---|---|
| VariBench `PoPMuSiC-2.0_S2648.xlsx` | negative = destabilising | 1A43 C218S −3.7; mean −1.01, 77% negative |
| PremPS `S2648/S350/S921/S250/p53.txt` | **positive = destabilising** (README) | 1E65 I7A +3.11 |
| DDGun Zenodo prediction files (all sets) | negative = destabilising | s2648 1a43A −3.70; myoglobin I28A −2.06 |
| VariBench `STRUM_Q3421.xlsx`, ThermoNet `Q3421.txt` | negative = destabilising | 2LZM L99A −4.53 |
| ThermoNet `Q3214/Q1744/s_sym/myoglobin/p53.txt` | **positive = destabilising** (flipped on purpose) | 1bz6 I28A +2.06; 1amq C191Y +2.3 |
| Ssym original `bty348_dataset.xlsx`, KORPM `Ssym.txt`, VariBench D30 (Ssym+) | negative = destabilising | 1L63 L99A −4.5 |
| VariBench `42_variations_in_P53.xlsx` | negative = destabilising | Q104H +0.24 (**opposite** to the PremPS copy) |
| KORPM `S669.txt` / `S461.txt` | negative = destabilising (README: ≥0 = stabilising) | S461 mean −1.14 |
| GeoStab `S8754/S783/M1261.csv` | negative = destabilising | 1A43 C218S −3.7 |
| GeoStab `S571.csv` | ΔTm, negative = destabilising | mean −3.36 |
| VariBench `HotMuSiC_S1626.xlsx` "ddG" | **positive = destabilising** | 1L63 L99A +3.34 |
| protddg-bench (all) | negative = destabilising | §5.1 |
| Stability Oracle (all) | positive = destabilising | §5.3 |

**How S461 was derived from S669** (KORPM paper + diff of `S461.txt` vs their `S669.txt`): all 461
are S669 direct variants; **208 dropped** = 46 whole proteins (e.g. 1A7V, 1FH5, 1GWY, 1IR3, 3D2A,
4YEE/4YEF, 5VP3) plus 5 of the 31 mutations on 3DV0 (94 → 48 proteins), on the grounds of
incorrect entries and mutations that interfere with protein–protein / protein–ligand binding; plus
**40 values corrected**: sign flips for all 12 3C2I mutations, halved values on 1H0X/1JLV/1SPD/1FRD
(looks like a kJ→kcal fix), re-entered values for 1O6X. Final: 74 stabilising / 387 destabilising,
mean −1.15 kcal/mol. Per-reason counts in Supplementary Appendix SII were not accessible [U].

**DDGun / DDGun3D** (Montanucci et al. 2019 *BMC Bioinformatics*; server paper *NAR* 50:W222
(2022), doi:10.1093/nar/gkac325; server folding.biofold.org/ddgun). Code:
https://github.com/biofold/ddgun (code + AAindex only, **no benchmarks**). Benchmarks are on
Zenodo **10.5281/zenodo.4613881** → `predictions.tar.gz` (388 kB) [C]
`https://zenodo.org/api/records/4613881/files/predictions.tar.gz/content`: direct and `_inv`
versions of s2648 (2,648), varibench (1,420), broom (599), Ssym (684), ptmul (914), 2ocjA/p53 (42),
1bz6A/myoglobin (134), each with experimental and predicted ΔΔG. Caveat: its s2648 file
**renumbers residues sequentially** (1a43A C71S = PDB C218S). The 2019 text defines
ΔΔG = ΔG_wt − ΔG_mut but the released files are negative = destabilising.

**Confirmed mirrors for the classic sets** (all HTTP 200):
- VariBench base `https://structure.bmc.lu.se/VariBench/data/effectspecific/stability/` (index
  `stability.php`): `PoPMuSiC-2.0_S2648.xlsx`, `STRUM_Q3421.xlsx`, `STRUM_Q306.xlsx`,
  `HotMuSiC_S1626.xlsx`, `42_variations_in_P53.xlsx`, `bty348_dataset.xlsx` (Ssym with T/pH),
  `PON-TStab_dataset.xlsx`, `Broom_S605.xlsx`, `D30.csv` (Ssym+ with 21 tools' predictions),
  `D28_Q1744.txt`, `SingleVariants/Dataset26/Protein_stability_iStable2_S630.txt` (+ `_S3568_training`).
- ThermoNet: `https://raw.githubusercontent.com/gersteinlab/ThermoNet/master/data/datasets/{Q3421,Q3214,Q1744,s_sym,myoglobin,p53}.txt`
- PremPS: `https://raw.githubusercontent.com/minghuilab/PremPS/main/Datasets/…`
  (`Eight%20test%20sets/{S250,S350,S2000,p53,S1925,S605,Ssym}.txt`, `S2648/S2648.txt`, `S921/S921.txt`)
- GeoStab: `https://raw.githubusercontent.com/Gonglab-THU/GeoStab/main/data/ddG/{S8754,S783,S2000,M1261}.csv`,
  `…/data/dTm/S571.csv` (S4346/S461/S669/Ssym are listed in the tree but were not fetched [U])
- KORPM: `https://raw.githubusercontent.com/chaconlab/korpm/main/{S461.txt,Ssym.txt,SsymD.txt}`,
  `S461_results/S669.txt`

---

## 6. Human Domainome 1 (aPCA)

**Citation.** Beltran A, Jiang X, Shen Y, Lehner B. *Site-saturation mutagenesis of 500 human
protein domains.* **Nature 637:885–894 (2025)** (online 2024-12), doi:10.1038/s41586-024-08370-4;
PMC11754108.

**Download.** Concept DOI **10.5281/zenodo.11043642** → latest version **record 14356805**
(v5, 2024-12-10) [C], where the files are zipped and renamed `Supplementary_Table_*`:
`Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip` (29 MB),
`Supplementary_Table_4_homolog_predictions.txt.zip` (**20 MB — the ΔΔG table**),
`Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip` (40 MB),
`Supplementary_Table_1/3`, `files_to_reproduce_analyses.tar.gz` (4.1 GB).
Earlier versions: 13629491 (v3, 2024-09-02, same content unzipped under `Extended_data_Table_*`)
[C], 11260616 (v2), 11043643 (v1). Use **14356805** (smaller, current names).
Reads: GEO GSE265942. Code: https://github.com/lehner-lab/domainome.
Main data file (local copy at `/workspace/Domainome/…Table_2…txt`, 602,882 rows; the project's
processed version is `data/processed/domainome.csv`, 536,164 missense variants with fitness).

**Assay and size.** Abundance PCA (aPCA, DHFR-fragment-complementation growth assay in yeast),
3 biological replicates × 9 sub-libraries. Library: 1,230,584 variants in 1,248 domains;
**final filtered dataset: 563,534 variants in 522 domains (503 human)**, 127 Pfam families,
195 all-α, 127 all-β, 48 α/β, 148 zinc fingers.

**Columns / sign.** `fitness` (growth rate based) and **`normalized_fitness`**: per domain, scaled
so **WT = 0** and the 2.5th percentile of the missense distribution = **−1**. So
**more negative = less abundant = destabilising** — i.e. the *opposite* sign to PhysDDG's `ddg`,
and it is a fitness proxy, not kcal/mol. `fitness_sigma` / `normalized_fitness_sigma` give the
per-variant error; `quality_rank` ranks domains; STOP variants are flagged.

**Noise ceiling / quality.** Domains retained if rank ≤600, missense replicate Pearson r > 0.485,
>50% of variants measured; 16 further domains removed as incompatible with 2-state folding →
522. **Median replicate r = 0.85.** Agreement with in-vitro ΔΔG: **median ρ = 0.73 (n = 10
domains)**; with Tsuboyama cDNA-display proteolysis: **median ρ = 0.65 (n = 13 domains)**.
Assay temperature 30 °C (yeast). Our own earlier estimate of the Domainome noise ceiling is
**0.84** (replicate-based), and a fine-tuned ThermoMPNN reaches 0.617 test Spearman = 73% of it.

**fitness ↔ ΔΔG relationship.** Sigmoidal (Boltzmann two-state): fitness saturates for
ΔΔG ≳ 3 kcal/mol (fully unfolded) and for strongly stabilising mutations, so aPCA resolves ΔΔG
only in a window around the WT ΔG; mutations in already-unstable domains are uninformative.
Stability explains a median 30% of ESM1v-predicted evolutionary fitness variance.

**Inferred ΔΔG — yes, it is released.** The paper fits **MoCHI** Boltzmann (two-state) models per
domain *family*, assuming a mutation costs the same ΔΔG in every homologue, and uses them to
predict **4,107,436 variants in 7,271 domains from 26 families**. This is
**Supplementary/Extended Table 4** (`…Table_4_homolog_predictions.txt[.zip]`), columns
`PFAM_entry, uniprot_ID, gene_ID, pos, wt_aa, mut_aa, mean_kcalmol_scaled, std_kcalmol_scaled`
(header read directly from the Zenodo zip, confirmed). Caveats:
- **positive = destabilising** (Pro has the highest mean; the paper's "strongly destabilising"
  cut is scaled ΔΔG > 0.3);
- the values are **rescaled, not true kcal/mol** despite the column name — a sampled family gives
  p2.5 ≈ −0.29, p97.5 ≈ 1.33, which does not match the Methods' "2.5th percentile = −1" text;
  treat the scale as arbitrary and calibrate before comparing with kcal/mol data;
- they are **homologue-averaged** — the same aligned substitution gets nearly the same value in
  every homologue, and no homologue-specific epistasis is modelled;
- model accuracy: median r = 0.80 within family (10-fold CV; 0.78 on homeodomains), r = 0.66 on
  left-out homologues, r = 0.65 against proteolysis data for 38 homologues;
- the families were fitted on Domainome itself → **do not evaluate on Domainome using Table 4**.

FireProtDB 2.0 ingested Domainome tables 2 and 5 and exposes `DOMAINOME_DDG`, `DOMAINOME_DDG_STD`,
`DOMAINOME_FITNESS`, `DOMAINOME_FITNESS_STD` through its API — a convenient join route, with the
same scaling caveat.

**In MaveDB:** 521 per-domain score sets, `urn:mavedb:00000676-a-1` … `urn:mavedb:00001196-a-1`,
602,021 rows — **fitness only (`score, sigma, raw_score, raw_sigma`), no ΔΔG**; WT = 0, 2.5th
percentile of missense = −1, higher = more abundant.

**Domainome 2:** no paper, preprint or Zenodo record found as of Oct 2026 [U = searched, nothing].

MoCHI tool: https://github.com/lehner-lab/MoCHI ; DiMSum pipeline (used for counts→fitness):
https://github.com/lehner-lab/DiMSum.

---

## 7. VAMP-seq / abundance DMS (MaveDB)

**Access.** MaveDB API base https://api.mavedb.org (OpenAPI `/openapi.json`, docs `/docs`) [C].
Score-set URNs look like `urn:mavedb:{8-digit experiment set}-{letter}-{n}`; a `-0-` in the middle
marks a meta-analysis set combining others. Endpoints:
`GET /api/v1/score-sets/{urn}` (metadata, `datasetColumns`, `methodText`),
**`GET /api/v1/score-sets/{urn}/scores` → CSV** (params `start`, `limit`, `drop_na_columns`),
`/counts`, `/variants/data`, and `POST /api/v1/score-sets/search`.
Working example, GET-verified (200, 1.7 MB, 4,409 rows):
`https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000013-a-1/scores` [C].
Web page: `https://www.mavedb.org/score-sets/urn:mavedb:00000013-a-1` [C].
(A bare HEAD on the API returns 405 — use GET.)

**Sign/normalisation.** Every set below was checked in the released CSV: **higher score = more
abundant**. Most use synonymous/WT = 1 and nonsense = 0; the exceptions are flagged. **None of
these papers release an inferred ΔΔG** (ASPA, PRKN, GCK, MLH1 only *compare* with Rosetta ΔΔG;
CYP2C19 with FoldX, r = −0.46).

| Protein | MaveDB URN | Variants (missense) | Paper | Replicates | Normalisation notes |
|---|---|---|---|---|---|
| PTEN | `00000013-a-1` | 4,409 (4,112) | Matreyek et al. 2018, *Nat Genet* 50:874, doi:10.1038/s41588-018-0122-z | mean r 0.63 / ρ 0.62, 8 reps | min–max scaled, WT = 1, median nonsense (pos 51–349) = 0; medians syn 1.00, stop 0.06 |
| PTEN fill-in / combined | `00000101-a-1` / `00000102-0-1` | 4,514 / 5,477 | Matreyek et al. 2021, *Genome Med* 13:165 | combined = mean of 15 reps | overlaps `00000013` — never treat as independent |
| TPMT | `00000013-b-1` | 4,028 (3,689) | Matreyek 2018 | mean r 0.73 / ρ 0.67 | medians syn 1.03, stop 0.15 |
| CYP2C9 | `00000095-b-1` | 6,821 (6,370) | Amorosi et al. 2021, *AJHG* 108:1735, doi:10.1016/j.ajhg.2021.07.001 | mean r 0.789 / ρ 0.754, 3 sorts | nonsense normalisation uses the middle 80%; TM N-terminus (1–20) is abundance-sensitive |
| CYP2C19 | `00001199-a-1` | 8,149 (7,660 single-aa) | Boyle et al. 2024, *Genetics* 228:iyae156 | r 0.82–0.98, 7 sorts | nonsense = 0, WT = 1; r = 0.77 with CYP2C9 |
| CYP2C9/2C19 small | `00000062-a-1` / `-b-1` | 109 / 121 | Zhang et al. 2020, *Clin Transl Sci* | — | low-throughput |
| NUDT15 | `00000055-a-1` | 3,100 (2,922) | Suiter et al. 2020, *PNAS* 117:5394 | n/r | **non-normalised averages** (range −0.05…1.38, median ≈0.75); only one stop variant |
| VKOR | `00000078-b-1` | 2,911 (2,695) | Chiasson et al. 2020, *eLife* 9:e58026 | mean r 0.73, 7 reps | membrane protein; syn = 1, nonsense = 0 |
| ASPA | `00000657-a-1` | 6,152 | Grønbæk-Thygesen et al. 2024, *Nat Commun* 15:4026 | mean r 0.99, 11 reps | missense median 0.415 (low-abundance-heavy); low end poorly resolved |
| PRKN (VAMP-seq) | `00000114-a-1` | 9,220 | Clausen et al. 2024, *Nat Commun* 15:1541 | r 0.96–0.99 | stop = 0, WT = 1 |
| PRKN (antibody stain) | `00001281-a-1` | 9,205 | Aslanzadeh et al. 2026, bioRxiv doi:10.64898/2026.02.09.704817 | — | **WT ≈ 0, stop ≈ −1.2**; after OA depolarisation |
| GCK (yeast DHFR-PCA) | `00000096-b-1` | 9,201 | Gersing et al. 2024, *Genome Biol* 25:98 | — | nonsense = 0, syn = 1; western-blot r = 0.80; activity set is `-a-1` (Gersing 2023) |
| MLH1 C-term (DHFR-PCA) | `00001218-a-1` | 5,056 (4,839) | Larsen-Ledet et al. 2025, *PLoS Genet* 21:e1011681 | — | **syn ≈ 0, stop ≈ −1** |
| SOD1 (TileSeq) | `00001217-a-3` | 2,906 | no publication linked | — | nonsense = 0, syn = 1 |
| G6PD | `00001266-b-1` | 10,674 | Geck et al. 2026, *Genome Biol* 27:249 | 2 reps | WT = 1 |
| LDLR (cell surface) | `00001269-b-1` | 16,796 | Tabet et al. 2025, *Science*, doi:10.1126/science.ady7186 | — | trafficking, not pure folding |
| TYK2 | `00001271-a-1` | 23,698 | Howard et al. 2025, bioRxiv doi:10.1101/2025.10.11.681520 | — | **stop median −0.35**, scale is not 0–1 |
| MAPK-pathway abundance maps (KRAS, GRB2, BRAF, RAF1, MAP2K1/2, SOS1/2, PTPN11, MET, RET, KSR1/2) | `00001302`–`00001318` | — | Simon et al. 2026, bioRxiv doi:10.64898/2026.10.04.756099 | — | WT-relative; includes HSP90-inhibitor conditions |

Also: COMT abundance `00000661-d-1`; surface-expression sets (KCNH2 `00001216`/`00001231`, CCR5,
CXCR4, RHO `00001275`, INSR) measure trafficking, not folding.

**Biases.** Steady-state abundance in HEK293T (VAMP-seq) or yeast (DHFR-PCA), not ΔG: the signal
saturates at the stable end, mixes folding with degradation/quality-control and expression, and is
sensitive to the fusion tag and to termini. Early VAMP-seq replicate r of 0.63–0.79 sets a low
noise ceiling. **Use for AUROC-style validation of "is this destabilising", not for kcal/mol
regression.**

**Project cross-reference:** a sibling agent already built
`data/processed/abundance_dms.csv` (37,947 variants, 7 assays: PTEN, TPMT, CYP2C9, NUDT15, PRKN,
GCK, KRAS-aPCA) from ProteinGym v1.3 with full-length AF models; `score` = ProteinGym `DMS_score`,
higher = more abundant, `ddg` = NaN. Sanity there: score falls with burial in all 7 (ρ −0.20…−0.50).

---

## 8. ProteinGym stability subset

**Citation.** Notin P, Kollasch AW, Ritter D, van Niekerk L, Paul S, Spinner H, Rollins N, Shaw A,
Orenbuch R, Weitzman R, Frazer J, Dias M, Franceschi D, Gal Y, Marks DS. *ProteinGym: Large-scale
benchmarks for protein fitness prediction and design.* **NeurIPS 2023 Datasets & Benchmarks**.
Repo https://github.com/OATML-Markslab/ProteinGym [C], site https://proteingym.org [C].

**Download** (v1.3 is current):
`https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip` [C,
HTTP 200, ~1 GB unzipped]. Reference table (small, no zip needed):
`https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv`
[C] — 217 rows, one per assay. Release Zenodo DOIs: v1.0 13932633, v1.1 13936340, v1.2 14997691,
**v1.3 15293562**.

**Which assays are "Stability"** (`coarse_selection_type == Stability`): **66 of 217 assays,
134,794 variants**, composed of
- **64 Tsuboyama 2023 assays** (`<GENE>_<SPECIES>_Tsuboyama_2023_<PDB>`), 65,887 single +
  51,924 multiple mutants. PDB ids: 1A32, 1AOY, 1E0L, 1F0M, 1GYZ, 1I2T, 1I6C, 1IFY, 1JIC, 1K1V,
  1LP1, 1O6X, 1ORC, 1PSE, 1PV0, 1R69, 1TG0, 1TUD, 1UFM, 1URF, 1UZC, 1V1C, 1W4G, 1WCL, 1YU5, 2AMI,
  2BTT, 2CJJ, 2D1U, 2GP8, 2JVD, 2JVG, 2K28, 2KFV, 2KRU, 2KVT, 2KWH, 2L2D, 2L33, 2L6Q, 2L7M, 2L9R,
  2LCL, 2LHR, 2LO1, 2M0Y, 2MI6, 2MXD, 2MYX, 2N88, 2O2W, 2QFF, 2RRU, 2WNM, 2WQG, 2YSB, 3DKM, 3L1X,
  4G3O, 5JRT, 5OAO, 5UBS, 6ACV, 7JJK.
- `A4_HUMAN_Seuma_2022` (amyloid-beta aggregation, 14,811 variants) and
  `ESTA_BACSU_Nutschel_2020` (T50 thermostability, 2,172) — not ΔΔG.

**DMS_score sign.** For the Tsuboyama assays `raw_DMS_phenotype_name = ddG_ML_float` and
`raw_DMS_directionality = 1`, i.e. the raw `ddG_ML` is kept as-is and higher = fitter. So
**ProteinGym DMS_score positive = stabilising; flip for PhysDDG**. (ProteinGym also z-standardises
nothing: `DMS_score` is the raw phenotype, `DMS_score_bin` is the binarised version.)

**Leakage note.** 53 of the 64 ProteinGym Tsuboyama assays are **ThermoMPNN *training* domains**;
only 6 are in the ThermoMPNN Megascale test set (2K28, 3DKM, 2L33, 1PSE, 2KWH, 1GYZ) and 5 in val.
So "ProteinGym stability" ≈ Megascale train for any ThermoMPNN-derived baseline.

---

## 9. MaveDB and 2024–2026 new large stability data

### 9.1 MGnify Stability dataset (Rocklin lab, 2026) — the big new one
**Citation.** Cho Y, Tsuboyama K, Litberg TJ, Jung MD, Obisesan A, Wang Q, Phoumyvong CM,
Thibeault J, Ovchinnikov S, Rocklin GJ. *Accurate protein stability prediction for small domains
using mega-scale experiments.* **bioRxiv 2026-05-20**, doi:10.64898/2026.05.19.726285.
Data: Zenodo **10.5281/zenodo.19411306** [C] https://zenodo.org/records/19411306 [C], ~8 GB:

| File | Size | Contents |
|---|---|---|
| `230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv` | 2.23 GB | **full MGnify Stability dataset, recalibrated ΔG** (dmsv4+dmsv5+dmsv7) |
| `mgnify_training_index.csv` | 138 MB | sequence index + train/val/test assignment |
| `dmsv4_filtered_train_splits.csv` | 60 MB | DMSv4 training split after unstructured-terminus filtering |
| `220122_dmsv4_count_table.csv`, `220331_dmsv5_count_table.csv` | 486 / 513 MB | raw NGS counts |
| `benchmarks.zip` | 872 MB | incl. **`S1724_thermomutdb_cleaned_withseq.csv`** (new curated ThermoMutDB ΔΔG benchmark, 35 WT proteins) and TED-domain sets |
| `figures.zip` | 716 MB | per-figure prediction CSVs (Megascale ddG test set, MGnify ins/del benchmark, …) |
| `dmsv7_analysis.zip`, `scripts_to_calculate_K50_and_dG.zip` | 2.35 GB / 612 MB | notebooks |

Scope: **1.8 million domains, 60–80 aa, mostly from MGnify metagenomics, >200,000 sequence
families**, same cDNA-display-proteolysis assay (pH 7.4 / 25 °C; the paper quotes a 6 kcal/mol
usable ΔG range, wider than the 2023 −1…5 window thanks to recalibration).

**It is an absolute-ΔG dataset, not a ΔΔG dataset.** Header of the main CSV (range-fetched,
confirmed): `name, dna_seq, log10_K50_t, log10_K50_t_95CI_high/_low/_95CI, fitting_error_t, rmsd_t,
log10_K50unfolded_t, deltaG_t (+CIs), <same for _c>, log10_A0_K50, deltaG, deltaG_95CI_high/_low/
_95CI, lib (dms4/dms5/dms7), aa_seq, aa_seq_full, mgnify (bool), dm_design (bool), train_name,
**split** (train/val/test), train_seq, train_dG_from_earlier_processing`. So: `deltaG` = ΔG of
unfolding, **positive = stable**, with 95% CIs, and the authors' own train/val/test split ships in
the file. ΔΔG benchmarks (Megascale point mutants, Megascale and MGnify insertion/deletion sets,
the new **S1724** ThermoMutDB-cleaned set, TED domains, de novo designs, nanobody Tm) are the
per-figure CSVs inside `figures.zip` / `benchmarks.zip`.
Models: SaProtΔG / ESM3ΔG, RMSE 0.8 kcal/mol on absolute ΔG over a 6 kcal/mol range (ρ = 0.88).
Code https://github.com/yehlincho/absolute-stability-predictor, weights
https://huggingface.co/Yehlin/absolute-stability. This is the dataset our PLM-eval work already
uses (SaProtΔG/ESM3ΔG benchmarks).

### 9.2 MaveDB
- Esposito D et al., *Genome Biol* 20:223 (2019), doi:10.1186/s13059-019-1845-6 (original).
- Rubin AF et al., *MaveDB v2*, bioRxiv 2021, doi:10.1101/2021.11.29.470445 — **preprint only,
  never journal-published**.
- **Current paper:** Rubin AF, Stone J, Bianchi AH et al. *MaveDB 2024: a curated community
  database with over seven million variant effects from multiplexed functional assays.*
  **Genome Biol 26:13 (2025-01-21)**, doi:10.1186/s13059-025-03476-y, PMID 39838450.
  (No NAR MaveDB paper exists for 2025/2026 — searched Crossref and Europe PMC.)
- **Watch out:** the Tsuboyama Megascale data is *also* in MaveDB (≈1,082 "trypsin digestion"
  score sets, e.g. `00000116-0-1` FYN SH3 with combined `dG`, `-a-1`/`-a-2` for trypsin and
  chymotrypsin, columns log10 K50, `dG`, `dG_95CI`), mostly with no publication linked. Easy to
  double-count Megascale if you harvest MaveDB broadly.

### 9.3 Lehner-lab ddPCA / MoCHI — the datasets that release ΔΔG in kcal/mol directly

Sign across all of these (checked in the released tables): **positive ΔΔG = destabilising**
(folding) or weaker binding; WT ΔG_f is negative; proline substitutions have the largest positive
means; 80–89% of single mutants are > 0. "Confident" = Monte-Carlo 95% CI < 1 kcal/mol.

| Dataset | Citation | ΔΔG released | Size | Where |
|---|---|---|---|---|
| **GRB2-SH3, PSD95-PDZ3, GB1** (ddPCA) | Faure AJ, Domingo J, Schmiedel JM, … Lehner B. *Mapping the energetic and allosteric landscapes of protein binding domains.* **Nature 604:175–183 (2022)**, doi:10.1038/s41586-022-04586-4 | folding **and** binding, `f_ddg_pred`/`b_ddg_pred` + `_sd`, `_conf` | singles: GRB2-SH3 1,056 (1,025 confident folding / 591 binding); PSD95-PDZ3 1,587 (1,148 / 1,196); GB1 1,045 (898) | Supplementary Table 7 xlsx [C] `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-022-04586-4/MediaObjects/41586_2022_4586_MOESM10_ESM.xlsx`; code github.com/lehner-lab/doubledeepms; reads GEO GSE184042. PSD95-PDZ3 folding ΔΔG vs in vitro r = 0.79 (n = 30) |
| **KRAS 4B (1–188)** | Weng C, Faure AJ, Escobedo A, Lehner B. *The energetic and allosteric landscape for KRAS inhibition.* **Nature 626:643–652 (2024)**, doi:10.1038/s41586-023-06954-0 | folding ΔΔG `urn:mavedb:00000115-a-7` (`score`, `std`; 89% > 0); binding ΔΔG `-a-6` RAF1, `-a-5` RALGDS, `-a-4` PIK3CG, `-a-3` SOS1, `-a-2` DARPin K27, `-a-1` DARPin K55 | 3,453 non-NA folding ΔΔG over 187 positions, 2,339 confident; **in vitro r = 0.95** | MaveDB set `00000115` (28 score sets); Suppl. Table 5 xlsx (columns `mean_kcal/mol`, `std`, `ci95`, `assay`, `ddG_conf`); code lehner-lab/krasddpcams; reads SRA PRJNA907205 |
| **GRB2-SH3 / SRC combinatorial** | Faure AJ, Martí-Aranda A, … Lehner B. *The genetic architecture of protein stability.* **Nature 634:995–1003 (2024)**, doi:10.1038/s41586-024-07966-0 | ~140 single-mutant terms + pairwise couplings only | 129,320 combinatorial GRB2-SH3 variants (mostly ≥13 mutations) | Suppl. Table 5; GEO GSE246322; code lehner-lab/archstabms, Zenodo 10.5281/zenodo.11671164 |
| **Src kinase** | Beltran A, Naqvi, Faure AJ, Lehner B, *Sci Adv* 12:eaea2726 (2026) | ΔΔG_f and ΔΔG_a in table S2 | >50,000 single + double variants | Zenodo 10.5281/zenodo.10158641 → record 17165108 [C] |
| **PDZ family** | Martí-Aranda A, Lehner B, *Nat Commun* 17:5487 (2026) | 9,064 ΔΔG_f (5 PDZ domains) + 12,738 ΔΔG_b (7 interactions) | — | Zenodo 15978782 [C] (944 MB) |
| **PDZ3 extensions** | Hidalgo-Carcedo C et al., *Nat Commun* 17:2934 (2026) | yes | >190,000 variants; in vitro r = 0.88 | Zenodo 16411462 |
| **KINASE-MAPS** (SRC, FGR, JNK2, ZAK, TSSK2) | bioRxiv doi:10.64898/2026.08.04.742685 | yes | >160,000 variants | Zenodo 23261855 |
| **Randomised cores** | Escobedo A et al., *Science* (2025) | yes | — | Zenodo 16266645 |
| **PYL1** | Stammnitz & Lehner, *Nat Commun* (2026) | [U] not checked | — | MaveDB `00001287`–`00001300` |

Tool: **MoCHI** — Faure AJ & Lehner B, *Genome Biol* 25:303 (2024),
doi:10.1186/s13059-024-03444-y, https://github.com/lehner-lab/MoCHI.

**Why these matter for PhysDDG:** they are the only large sets besides Megascale that give
*folding* ΔΔG in (near-)kcal/mol with per-variant errors, they are on different, larger domains
than Megascale (SH3, PDZ, KRAS, kinases), they include binding ΔΔG so you can separate stability
from function, and the KRAS/PDZ3 sets validate against in vitro at r = 0.88–0.95. Good external
test material; 2–3k single mutants each.

### 9.4 Other 2024–2026 items
- **FireProtDB 2.0** (§3) — the aggregated 5.5 M-experiment release.
- **Human Domainome 1** (§6) — 563,534 human-domain abundance measurements, plus 4.1 M
  family-model ΔΔG predictions.
- **Meltome / Tm datasets** (Jarzab et al. 2020 Nat Methods; FLIP "meltome" split) measure Tm of
  whole proteomes, **not ΔΔG**; ρ(ΔΔG, ΔTm) ≈ 0.71 at best. Keep out of ΔΔG fitting.
- **S1724** — new ThermoMutDB-cleaned ΔΔG benchmark (35 WT proteins) shipped inside
  `benchmarks.zip` of the 2026 MGnify Zenodo record (§9.1).
- No "Domainome 2" exists as of Oct 2026.

---

## 10. Overlap / leakage matrix and recommended usage

### 10.1 Measured overlaps (computed here, variant-level key = PDB + mutation)

| A | B | shared variants | shared PDB chains |
|---|---|---|---|
| S2648 | VB1432 | 719 | 49 |
| S2648 | Ssym(dir) | **194 of 342 (57%)** | **13 of 15** |
| S2648 | Broom | 167 | 33 |
| S2648 | KORPM-train | 566 | 64 |
| S2648 | p53 | 5 | 1 |
| S2648 | S461 | **0** | **0** |
| VB1432 | Ssym(dir) | 98 | 9 |
| VB1432 | S461 | 0 | 0 |
| VB1432 | KORPM-train | 514 | 57 |
| Ssym(dir) | Broom | 14 | 7 |
| Ssym(dir) | S461 | 0 | 0 |
| S461 | KORPM-train | 141 | 25 |
| Broom | KORPM-train | 232 | 46 |
| T2837 | C2878 | **0** | **0** (by construction, 30% id filter) |
| T2837/s669 | ThermoMPNN-FireProt (100 prot) | 52 | 4 |
| T2837/ssym | ThermoMPNN-FireProt | 87 | 8 |
| T2837/t2226 | ThermoMPNN-FireProt | 259 | 17 |
| C2878 | ThermoMPNN-FireProt | 196 | 58 |
| Megascale | FireProt (MMseqs 25%, ThermoMPNN `.m8`) | — | **11 of 100 FireProt proteins hit 32 Megascale domains**: 1CSP, 1FMK, 1MJC, 1MSI, **1PGA**, 1SSO, **1UZC**, 1W3D, **1YU5**, 2A36, 3MBP. Several are near-identical, not just homologous: 1YU5↔1YU5/1VII/1YRF/2RJV/3MYC (97–100% id), 1UZC↔1UZC (100%), 1SSO↔1JIC (100%), 1FMK↔1SRM (98%), 1PGA↔5UCE/5UBS (91%) |
| S669 | Megascale (ThermoMPNN `.m8`) | — | **6 S669 chains** hit Megascale domains, three of them **identical proteins**: 1O6X↔1O6X (100%), 2HBB↔2HBB (100%), 1DIV↔2HBB (100%), 2WQG↔2WQG/4UZW (98–100%), 2RPN↔1Y0M/1GL5 (39–47%), 5VP3↔2AMI (32%). ThermoMPNN reports 420 of 669 variants survive its homology filter, i.e. **249 S669 variants (37%) are homologous to Megascale** |
| ProteinGym "Stability" | Megascale | 64 of 66 assays **are** Megascale domains (53 in ThermoMPNN train) |
| S350 | S2648 | 348 of 350 | — |
| S783 | S8754 | **691 of 783** | — |
| S2648 | S8754 | 2,590 of 2,648 | — |
| Q3421 | S8754 | 2,417 | — |
| S669 | S8754 | 2 | — (S8754 was filtered at 25% id to S669) |
| Ssym(dir) | Q3421 | 205–286 | — |
| S250 | Ssym | 218 of 250 | — |
| S1626 | S2648 | 744 rows | — |
| PON-tstab (VariBench) | S2648 | 739 of 1,564 | — |
| Broom | S2648 | 167 of 605 | 33 |

### 10.1b Overlap with FireProtDB (leakage for anything fitted on FireProt)

Measured against the local FireProt tables (lower bounds — residue-numbering differences hide
some matches). FireProtDB rows even carry source tags: STRUM3421 1,201 rows, PoPMuSiC 822,
Broom 295, STRUM306 95, HotMuSiC 30, PON-tstab 21.

| Benchmark | matched in FireProtDB |
|---|---|
| S2648 | ≥1,749 of 2,648 |
| Q3421 | ≥1,276 |
| S8754 | ≥2,718 |
| S921 | ≥537 |
| Ssym (direct) | ≥196 of 342 |
| S350 | ≥206 |
| S250 | ≥120 |
| p53 | 41 of 42 |
| myoglobin | 74 of 134 (UniProt numbering is +1 vs the PDB) |
| **S669** | **0–12** |
| **S461** | **0** |
| S4038 | includes FireProtDB by construction |

→ S669/S461 are the only classic sets that are genuinely external to FireProtDB.

### 10.2 Qualitative relationships
- **Ssym ⊂ S2648 (largely).** 57% of Ssym's direct variants and 13/15 of its proteins are in
  S2648. Anything fitted on S2648/VariBench is *not* independent of Ssym.
- **S669/S461 were built to exclude S2648 ∪ VariBench** (<25% identity) → zero overlap measured.
  They remain the cleanest classic held-out sets for a non-Megascale-fitted model.
- **S461 ⊂ S669** (461 of 669, those with usable structures for KORPM) — never report both as
  independent.
- **T2837 contains S669 + Ssym + p53 + myoglobin** (deduplicated, 30%-filtered against its own
  training sets). So S669/Ssym results and T2837 results are *the same evidence*.
- **C2878 ⊃ FireProtDB + S2648 + Q1744 + O2567** → C2878 is not independent of FireProt.
- **FireProtDB 2.0 ⊃ Megascale + Domainome + ProTherm** → not an independent benchmark any more
  unless filtered by `SOURCE_DATASET`.
- **cDNA117K ⊂ Megascale Dataset 1**, so Megascale-fitted parameters leak into it.
- **Domainome vs Megascale**: 13 domains overlap (used in the paper to compare aPCA with cDNA
  proteolysis, median ρ = 0.65); our earlier work found Domainome ThermoMPNN performance is
  unchanged (median ρ = 0.57) when restricted to domains with no Megascale homology, so the
  leakage is mild there.

### 10.3 Recommended usage for PhysDDG (~10–50 fitted parameters)

With so few parameters the risk is less classic overfitting than **tuning to one dataset's
systematics** (one assay, one pH, one size range). Suggested protocol:

1. **Fit** on a Megascale subsample: take Dataset 3, singles only, drop modified WT backgrounds,
   **exclude any domain whose MMseqs2 25% cluster appears in your evaluation sets**, and subsample
   to ~50–200 variants per domain so a few huge scans don't dominate. Flip the sign. Clip targets
   to |ΔΔG| ≤ 4 kcal/mol to avoid fitting the assay's saturation.
   Hold out at least the ThermoMPNN `test` + `val` clusters.
2. **Model selection / early stopping** on the ThermoMPNN `val` split (31 domains) or on grouped
   CV by cluster (`megascale_clusters.tsv`, `cv_train/val/test_{0..4}`). Never on S669/Ssym.
3. **Primary held-out evaluation** (never touched during fitting):
   - **S669** (669 variants, 94 chains) — the only set built to be homology-free w.r.t. the classic
     training corpora; report on the **420-variant Megascale-homology-free subset** too, since you
     fit on Megascale.
   - **Ssym direct + inverse** for antisymmetry and bias (α-index, ⟨ΔΔG_dir + ΔΔG_inv⟩). A
     physics-based model should be near-perfectly antisymmetric; this is the one place where
     PhysDDG should beat DL by construction — report it.
   - **FireProt-HF** (2,578 variants, 89 proteins) for larger proteins and a wider ΔΔG range
     (−9…+12), and as a sanity check against the −1…5 kcal/mol Megascale window.
   - **T2837** only *in addition*, clearly labelled as overlapping S669+Ssym+p53+myoglobin.
   - **S461** — 0 overlap with FireProtDB and with S2648/VariBench; the single cleanest
     small benchmark, but 48 proteins only, so report cluster-bootstrapped CIs.
4. **External, orthogonal checks**:
   - **Lehner ddPCA folding ΔΔG in kcal/mol** — KRAS (3,453 variants, in-vitro r = 0.95),
     PSD95-PDZ3 + GRB2-SH3 (≈2.6k), PDZ family 2026 (9,064). Different domains and a different
     assay from Megascale, same sign as ours, with per-variant errors. This is the best *external*
     kcal/mol test available.
   - **Domainome 1.0** (522 domains, per-domain Spearman, noise ceiling ≈ 0.84) — ranking only.
   - **MGnify Stability 2026** for absolute ΔG and for insertions/deletions.
   - **Abundance DMS** (`data/processed/abundance_dms.csv`, 7 assays) — AUROC only.
5. **Always report**: pooled Spearman, **per-protein median Spearman**, RMSE *without*
   recalibration and after a single global linear rescale, AUROC for ΔΔG > 1 kcal/mol, antisymmetry
   on Ssym, and bootstrap CIs over clusters (not over variants).
6. **Never** fit on ProThermDB/ThermoMutDB raw rows without sign-checking each source dataset, and
   never report S2648 cross-validation as evidence of generalisation (homology + Ssym contamination).
7. **Sign-check every file at ingestion** with a fixed assertion, not by reading the README: the
   mean ΔΔG of buried-hydrophobic → Ala/Gly/Pro substitutions must come out **positive** in our
   convention (it is ≈ +2.1 to +2.9 in Megascale and FireProt). Record the assertion and the
   original column name in the processing script, per CONVENTIONS.md.

---

## 11. Already on this machine (read-only, do not re-download)

| Path | What it is |
|---|---|
| `/workspace/data/runtime_tables/megascale_runtime.csv` | Tsuboyama Dataset-3 curated à la ThermoMPNN: 272,712 rows (271,231 mutants + 1,481 WT rows), columns `ddG_ML, mut_type, WT_name, aa_seq, dG_ML`. **ddG_ML positive = stabilising (unflipped).** |
| `/workspace/data/megascale_pdbs/` | 862 AlphaFold models of Megascale WT domains |
| `/workspace/data/downloads/AlphaFold_model_PDBs.zip`, `fireprot_upload.zip` | original Zenodo archives (7992926, 8169289) |
| `/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv` | ThermoMPNN FireProt table, 4,997 rows / 3,438 with ddG / 125 PDBs. **ddG positive = destabilising.** `position` is 1-based in the FireProtDB sequence, `pdb_position` is **0-based** in the shipped PDB (row 1: position 1 ↔ pdb_position 0). Conditions: pH median 6.78 (range 2–11; 189 rows missing), methods thermal 1,980 / GdnHCl 817 / urea 645 / plate-based chemical 645. Separate project caveat from memory: `fireprot_feat.csv` positions are 0-based vs 1-based prediction files — +1 before merging. |
| `/workspace/data/fireprot_pdbs/` | 130 PDB structures for the FireProt set |
| `/workspace/Domainome/Extended_data_Table_2_fitness_scores_normalized_domainranks.txt` | Domainome 1.0 main fitness table, 602,882 rows |
| scratch copies made today | ThermoMPNN `dataset_splits/*`, StabilityOracle `T2837.csv`/`C2878.csv`, protddg-bench CSVs, ProteinGym `reference_files/DMS_substitutions.csv` — all under `/tmp/claude-603270/-workspace/46f46ddf-a5d1-4b38-9381-67791a3f7e64/scratchpad/` (ephemeral; re-fetch with the URLs above into `data/raw/` via a script in `scripts/`) |

---

## 12. Open items / explicitly unverified

- `S4038` (Zheng 2024, *Protein Sci*): no public download located.
- KORPM Supplementary Appendix SII per-reason drop counts for S669 → S461.
- Whether PremPS's test set is correctly called S921 or S971 (GeoStab uses both names).
- GeoStab's `S4346`, `S461`, `S669`, `Ssym` copies: listed in the repo tree, fetch not completed.
- Temperature/pH metadata for PTmul, myoglobin and p53 files.
- Whether DeepDDG (Cao 2019) used Q3421 — Q3421 is STRUM (Quan 2016) and Q1744 is ThermoNet
  (Li 2020); the DeepDDG attribution commonly seen in the literature is wrong or at least unchecked.
- ProteinGym's exact per-assay handling of the Tsuboyama multi-mutants (the reference file says
  `includes_multiple_mutants = True` for most, but the ΔΔG of doubles is the measured double, not
  the sum).
- MGnify Stability (2026): whether any released table contains per-mutant ΔΔG with the Tsuboyama
  sign convention, beyond the per-figure prediction CSVs.
- `PYL1` (Stammnitz & Lehner 2026): whether ΔΔG is released.
- The 140-row difference between the ThermoMPNN paper's test-set size (28,312) and the split file
  applied to the released table (28,172).
