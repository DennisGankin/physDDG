# megascale (Tsuboyama et al. 2023, cDNA-display proteolysis)

**Files** (made by `scripts/10_download_ht.py`, then `11_build_megascale.py`, then `14_homology.py`; checked by `15_validate_tables.py`)
- `megascale.parquet` / `.csv.gz` (git-ignored, 45 MB): 271,231 single substitutions, 298 domains. This is the ThermoMPNN "mega" set, i.e. Tsuboyama **Dataset 3** singles.
- `megascale_censored.csv`: 14,656 single substitutions of the same 298 domains that Tsuboyama left **out** of Dataset 3. `ddg` is NaN for these; the interval is in `ddg_lo`/`ddg_hi`.
- `megascale_doubles.csv.gz` (git-ignored): 162,472 double substitutions on 153 domains (138,136 of them in Dataset 3). Each row also carries the two component single values, `ddg1` and `ddg2`.

**Sources.** On-disk `/workspace/data/runtime_tables/megascale_runtime.csv`. Zenodo 10.5281/zenodo.7992926 v2_230420, `Processed_K50_dG_datasets.zip` → `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` and `Single_DMS_list.csv`. Split: ThermoMPNN `dataset_splits/mega_splits.pkl`.

**What the runtime table is (verified).** It is exactly the Dataset-3 rows (`ddG_ML != '-'`) that are single substitutions or `wt`, restricted to the 298 WT_names in ThermoMPNN's train+val+test lists: 272,712 rows matched 1:1 on WT_name|mut_type|aa_seq|ddG_ML, of which 1,481 are `wt` rows. Its `dG_ML` column is NaN for 216,919 rows, so every dG value is taken from the Zenodo file instead.

**Sign.** Tsuboyama defines dG as the unfolding free energy, and ddG_ML = dG_ML(mut) − dG_wt, so negative means destabilising. We set `ddg = −ddG_ML`, which gives positive = destabilising; the original is kept in `ddG_ML_orig`. Evidence:
- Tsuboyama's `Stabilizing_mut == True` occurs only with ddG_ML > 0 (1,397/1,397).
- Buried hydrophobic → A/G/P/D/K: mean ddg +2.07, 98% > 0. Exposed: +0.33. Mutations to Pro: +1.65, 92% > 0.

**WT reference.** One `dG_wt` per domain: dG_mut − ddG_ML is constant (sd < 1e-9) and equals `Single_DMS_list.wt_dg_med` exactly. It can differ from the median of the WT replicate rows in Dataset 3 by up to 0.15 kcal/mol. `dG_wt_ci95` is the median CI width over the WT replicates; `dG_wt_rep_sd` is their spread.

**Dynamic range and censoring.** Tsuboyama reports dG in [−1, 5] kcal/mol; values outside are given as '<-1' / '>5' with ddG '-'. Inside Dataset 3, dG_ML equals the unclipped `deltaG` exactly, so nothing in Dataset 3 is clipped. Censoring therefore shows up in two places:
1. **Rows excluded from Dataset 3** (`megascale_censored.csv`):
   - 12,270 with dG '<-1' (mutant fully unfolded): the true ddg is at least dG_wt + 1, so `ddg_lo = dG_wt + 1` and `ddg_hi = inf`.
   - 916 with dG '>5': `ddg_hi = dG_wt − 5`.
   - 1,470 '-' rows (unreliable).
   - 78% of the '<-1' sites are buried (≥16 Cβ neighbours), against 30% in Dataset 3. **Dataset 3 under-represents the most destabilising core mutations.** Any fit or evaluation on Dataset 3 alone inherits this selection.
2. **Values near the limits, as per-row flags:**
   - `flag_mut_floor`: dG_mut < −0.5, so ddg is close to a lower bound.
   - `flag_mut_ceiling`: dG_mut > 4.5.
   - `flag_wt_high`: dG_wt > 4, so stabilising effects are compressed.
   - `flag_wt_low`: dG_wt < 1.
   - `flag_lowconf`: 95% CI of dG_mut or dG_wt wider than 0.5 kcal/mol.
   - `flag_protease_disagree`: |dG_trypsin − dG_chymotrypsin| > 1.

   Raw columns: `dG_mut`, `dG_mut_label` (raw string), `deltaG_unclipped`, `dG_mut_ci95`, `dG_mut_ci_low/high`, `dG_t`, `dG_c`, the per-protease CIs, `log10_K50_t/c` and `log10_K50U_t/c` (unfolded-state K50).

**Duplicates.** There are no duplicate (domain, position, mutant) singles, so nothing needed averaging. Among the doubles, 18,316 rows had a synonymous component (e.g. E24E:S41R) and were dropped. Replicate measurements of the same double were averaged over their Dataset-3 replicates (4,695 doubles had more than one measurement). For doubles, Pearson(ddg, ddg1+ddg2) = 0.81.

**Structure mapping.** `/workspace/data/megascale_pdbs/<WT_name>` contains Tsuboyama's AF2/design models. All 298 are single chain A, numbered 1..L without gaps, and their observed sequence is identical to the WT aa_seq. `pdb_resnum` = sequence position. WT identity is 100%; 0 rows were dropped.

**Columns.**
- `domain_origin`: natural (189 domains, named by PDB ID) or designed (109: EEHEE/EHEE/HEEH/HHH/TrROS).
- `cb_nb`: number of Cβ atoms within 10 Å.
- `in_train_s669`: membership of ThermoMPNN's S669-filtered training list.
- `WT_cluster_tsuboyama`.

**Splits** (ThermoMPNN, by protein):

| split | rows | domains (natural/designed) |
|---|---|---|
| train | 215,731 | 239 (148/91) |
| val | 27,328 | 31 (20/11) |
| test | 28,172 | 28 (21/7) |

ddg across all rows: mean 0.73, sd 1.00, 5–95% range −0.45 to 2.73, 31% > 1 kcal/mol.

**Homology** (`14_homology.py`; a hit means E ≤ 1e-3, identity ≥ 25%, and ≥ 50% of the shorter sequence aligned):
- 9 of the 28 test domains have a homologue in train: 1TUC and 2KXD are 100% identical to 1PWT; the others are 2K1B, 2K28, 1W4H, 2BTH, 2WXC, HEEH_KT_rd6_0793 and HHH_rd1_0142.
- 11 of the 31 val domains have a train homologue.
- Use `homolog_in_megascale_train == 0` for the strict test subset.
- `cluster` = single-linkage homology family. The 298 domains fall into 122 families.
