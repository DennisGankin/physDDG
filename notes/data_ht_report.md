# Data-HT report: high-quality ddG tables (2026-10-26, agent: data-ht)

**Pipeline**, run in this order:
1. `scripts/10_download_ht.py`: downloads into data/raw/ (about 1 GB from Zenodo).
2. `11_build_megascale.py`
3. `12_build_fireprot.py`
4. `13_build_external.py`
5. `14_homology.py`: annotates the tables in place, so rerun it after any of steps 2–4.
6. `15_validate_tables.py`: asserts 100% WT identity, schema, signs and Ssym pairing. Currently **ALL OK**.

Shared helpers are in `scripts/ht_common.py`. Every table follows CONVENTIONS.md: `ddg` in kcal/mol, positive = destabilising. Each has a card at `data/processed/<set>.md`.

## Tables

| table | rows | proteins | split | ddg mean / sd | frac ddg>1 |
|---|---|---|---|---|---|
| megascale (Dataset 3 singles) | 271,231 | 298 | train 215,731/239 · val 27,328/31 · test 28,172/28 | 0.73 / 1.00 | 0.31 |
| megascale_censored (excluded from Dataset 3) | 14,656 | 294 | train 12,421 · val 890 · test 1,345 | interval only | – |
| megascale_doubles | 162,472 (138,136 in Dataset 3) | 153 | same as singles | – | – |
| fireprot | 3,403 | 99 | train 2,669/57 · val 387/15 · test 347/27 | 1.07 / 1.81 | 0.42 |
| fireprot_dtm (dTm only) | 1,559 | 75 | + 25 proteins with split "none" | score = dTm | – |
| s669 | 669 | 94 | external | 0.96 / 1.64 | 0.45 |
| s461 (subset of s669) | 461 | 48 | external | 1.15 / 1.25 | 0.48 |
| ssym (342 direct + 342 reverse) | 684 | 15 WT (357 structures) | external | ±1.02 / 1.55 | – |
| s2648 | 2,633 | 132 chains | external | 1.02 / 1.47 | 0.45 |

Other outputs:
- `clusters.csv`: one row per protein, 1,228 proteins including 517 Domainome constructs.
- `homology_domainome.csv`: homology columns for the Domainome table, written as a side table so the data-af agent's table is left untouched.
- `fireprot_structures.csv`, `external_structures.csv`: per-structure provenance.

## Sign decisions (evidence)

**Megascale: flipped.** Tsuboyama's ddG_ML = dG_mut − dG_wt, with dG the unfolding free energy.
- dG_mut − ddG_ML is constant per domain and equals Tsuboyama's `wt_dg_med`.
- `Stabilizing_mut == True` occurs only with ddG_ML > 0.
- After flipping, buried hydrophobic → A/G/P/D/K has mean +2.07 (98% > 0) and X→Pro has mean +1.65.

**FireProt: not flipped.** FireProtDB's help page defines stabilising as ddG < −0.5 or dTm > +1. Pearson(ddG, dTm) = −0.82, and buried hydrophobic losses give mean +2.89 (96% > 0). 1QLP (serpin) has a negative mean, which is real.

**S669, S461, S2648: flipped.** In the files, negative = destabilising (means −0.96, −1.14, −1.01). After flipping, buried hydrophobic losses are +2.2 to +2.5 with 94–99% > 0.

**Ssym.** The ThermoNet copy (positive = destabilising) is used directly. Reverse rows are −direct.

## Mapping and failures

The rule throughout: 100% WT identity, with no silent offsets.

**Megascale.** All 298 models are single chain, numbered 1..L, and their sequence equals the WT aa_seq. 0 failures.

**FireProt.** `pdb_position` is a 0-based index into `pdb_sequence`, which uses 'X' for unresolved residues. That sequence was aligned to the observed chain.
- 1/4,997 rows failed: 1AYE, where FireProt's sequence does not match the file. It was dropped, so FireProt-test has 27 proteins, not 28.
- The previous `/foldx_native_clean` mapping ignored the X gaps and lost 227 rows (5.5%), which this mapping recovers. On its runnable rows the two mappings agree 99.8%; the remaining 5 disagreements were wrong in the old mapping.
- 4 chain IDs were corrected where the file has a single chain.

**External sets.** Exact author numbering. Failures:
- s669 0/669
- s461 0/461
- ssym 0/684
- s2648 15/2,648: all of 1LVE (Kabat numbering), dropped; `--allow-offset` would recover them but is off by PI decision.
- 2A01 is obsolete and was taken from the mirror's `obsolete/` directory.

**Structures.** No structures were downloaded from RCSB: all come from /rsyncPDB.
- Cleaned copies are in `data/structures/xtal/{fireprot,pdb}/<PDB>_<chain>.pdb`: model 1, one chain, no water/H/ligands, first altloc, MSE→MET, numbering and insertion codes kept.
- Removed hetero groups are listed per structure.
- `near_hetatm` / `near_other_chain` (5 Å) come from the original entry.

## Key findings / known issues

1. **Dataset 3 selection bias (Megascale).** For the 298 domains, Tsuboyama excluded 12,270 singles because the mutant was fully unfolded (dG < −1). 78% of those sites are buried, against 30% in Dataset 3, so the strongest core destabilisations are missing from the standard set. They are in `megascale_censored.csv` with `ddg_lo = dG_wt + 1` and `ddg_hi = inf`, ready for the interval-censored fit.
   - Within Dataset 3 nothing is clipped (dG_ML == deltaG).
   - Near-limit values are flagged: `flag_mut_floor` 1.7%, `flag_mut_ceiling` 2.2%, `flag_wt_high`/`flag_wt_low`, `flag_lowconf` (CI > 0.5) 0.6%, `flag_protease_disagree` 5.7%.
2. **The runtime Megascale table is exactly ThermoMPNN's mega set**: Dataset-3 singles plus wt for 298 proteins, 1:1. Its `dG_ML` column is 80% NaN, so dG values were taken from Zenodo instead. The Zenodo release adds dG_wt and dG_mut with CIs, per-protease dG, K50 and K50,U, the censored rows, the doubles (Pearson additivity 0.81), and the natural/designed origin.
3. **The Ssym copies disagree.** KORPM's copy differs on 24/342 pairs. ThermoNet and ACDC-NN agree with each other, so their values are used and KORPM's is kept as `ddg_korpm`.
4. **FireProt 1PGA** has 70 values at the 4.0 kcal/mol ceiling (`flag_censored`). Duplicates were averaged; 34 groups, almost all identical double imports.
5. **Missing assay metadata.** S669 and S461 have no pH or temperature in the copies used (the original S669 supplement was not fetched). S2648 has pH and T.
6. **Duplicates.** Megascale singles have none. FireProt duplicates were averaged by mean. Megascale doubles with a synonymous component were dropped and Dataset-3 replicates averaged.

## Homology / leakage (`14_homology.py`, MMseqs2 18-8cc5c)

Parameters:
- `cluster30` / `cluster50`: `mmseqs easy-cluster`, minimum identity 0.3 / 0.5, `-c 0.8 --cov-mode 0`, `-s 7.5`.
- **Leakage hit**: all-vs-all search at `-s 7.5`, keeping hits with E ≤ 1e-3, identity ≥ 25%, and ≥ 50% of the **shorter** sequence aligned. This catches a short Megascale domain inside a full-length protein.
- `cluster` (all tables) = `family` = connected components of the hit graph, i.e. single linkage, the conservative choice for grouped CV. 1,228 proteins form 436 families; the 298 Megascale domains form 122.

Protein-level overlap with Megascale-train:

| set | proteins with a Megascale-train homologue | at ≥50% identity | share of rows |
|---|---|---|---|
| Megascale-test | **9/28** (1TUC and 2KXD 100% = 1PWT; 2K1B/2K28, 1W4H/2BTH/2WXC, two designs) | 3 | – |
| Megascale-val | 11/31 | – | – |
| FireProt-test | 2/27 | – | – |
| FireProt-train | 12/57 | – | – |
| FireProt overall | – | – | 25.6% |
| S669 | 7/94 (1O6X, 2HBB, 1DIV, 2WQG at 100%; 3DV0 98%; 2RPN; 5VP3) | – | 27.1% |
| S461 | 5/48 (all ≥ 98%) | – | 37.1% |
| Ssym | 0/15 | – | – |
| S2648 | 16/132 | – | 6.4% |
| Domainome | 151/517 | 75 | – |

Overlap with FireProt-trainval:
- **Ssym: 11/15 proteins, 96.5% of rows.**
- **S2648: 65/132 chains, 68.6% of rows.**
- S669: 7/94.

**Recommendation.** Report every external set on all rows and on a strict subset with `homolog_in_megascale_train == 0 & homolog_in_fireprot_trainval == 0`. Use Megascale-test with `homolog_in_megascale_train == 0` (19 domains). Treat S2648 and Ssym as ProTherm-family data, not independent tests, whenever FireProt is used for fitting.

## Access

No downloads were blocked. Sources: GitHub raw (ThermoMPNN, KORPM, PROSTATA, ThermoNet), Zenodo 7992926, and the MMseqs2 GitHub release. The original S669 supplementary (OUP) was not attempted; the copy used is from the S461 authors.
