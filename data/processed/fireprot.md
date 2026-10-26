# fireprot (FireProtDB, ThermoMPNN-curated "best pH" export)

**Files** (`scripts/12_build_fireprot.py`, then `14_homology.py`)
- `fireprot.csv`: 3,403 unique single mutations with ddG, on 99 proteins.
- `fireprot_dtm.csv`: 1,559 rows that have only a dTm, on 75 proteins, 25 of which have no ddG data. Here `ddg` is NaN and `score` = dTm, where positive = stabilising.
- `fireprot_structures.csv`: per-protein mapping and chain information.

**Source.** `/workspace/data/runtime_tables/4_fireprotDB_bestpH.csv`: 4,997 rows, 125 proteins. It is a FireProtDB export curated by the ThermoMPNN authors, with `pdb_id_corrected`, a 0-based `pdb_position` into `pdb_sequence`, and the best pH per mutation. Structures are in `/workspace/data/fireprot_pdbs/`. Those files are **pre-cleaned**: one chain, no HETATM, one model.

**Sign: no flip.** FireProtDB's help page defines stabilising as ddG < −0.5 kcal/mol or dTm > +1 °C, so positive ddG = destabilising, which is our convention. Data checks:
- Pearson(ddG, dTm) = −0.82 on the 55 rows that have both.
- Buried hydrophobic → small/polar: mean +2.89, 96% > 0.
- Mean ddG over all rows: +1.06.
- One protein runs negative: 1QLP, the metastable serpin α1-antitrypsin, with mean −0.69. That is expected for this protein (many stabilising mutations of the metastable native state are known), so it was not flipped.

**Mapping.**
- `pdb_sequence[pdb_position] == wild_type` for 100% of rows.
- `pdb_sequence` contains 'X' for unresolved residues. It was globally aligned to the observed residues of the cleaned chain, and the WT residue had to match at every row.
- 4,996/4,997 rows mapped. The one failure is 1AYE: FireProt's sequence is a different procarboxypeptidase sequence than the file, and the row was dropped. As a result the test split has **27** proteins, not 28.
- Chain IDs that differ from the file were substituted where the file has a single chain: 1HK0 A→X, 1RN1 A→C, 1RTP A→1, 1YU5 A→X.
- Cross-check against the earlier `/foldx_native_clean` mapping:
  - 99.8% agreement on the rows that run marked runnable.
  - It had marked 227 rows "wt_letter_mismatch" because it ignored the gaps in pdb_sequence. All 227 are mapped here.
  - The 5 remaining disagreements were coincidental letter matches in the old mapping.

**Structure columns.**
- `structure` = `data/structures/xtal/fireprot/<PDB>_<chain>.pdb`: the on-disk file, minimally cleaned (model 1, the chain only, no H, first altloc, MSE→MET), with numbering and insertion codes kept.
- `assembly` = `asym_unit_model1_chain_only`.
- Because the on-disk files carry no ligands, `near_hetatm`, `hetatm_near` and `near_other_chain` (any heavy atom within 5 Å) come from the **original wwPDB entry** in the local mirror. Every mirror chain matched with 100% identity and the same numbering.
- 2LZM: the on-disk coordinates sit in a different frame from the entry (median Cα displacement 20 Å). Context mapping is by residue, so this does not affect the flags.
- 163 rows (4.8%) are near a ligand. 19 proteins have additional chains in the entry, listed in `other_protein_chains`.

**Duplicates.**
- 3,437 ddG rows → 3,403 unique mutations: 34 were averaged, and nearly all were identical values imported twice.
- `n_meas`, `ddg_sd` and `ddg_range` are kept. `flag_inconsistent` (range > 2, or opposite signs with both |ddg| > 1) applies to 1 row.
- `flag_censored`: 70 rows of 1PGA sit at exactly 4.00 kcal/mol, the assay ceiling for unfolded variants in Nisthal 2019. Treat them as lower bounds.

**Split.** ThermoMPNN `fireprot_splits.pkl` (identical to the 28/72 test/trainval split in `/foldx_native_clean`):

| split | rows | proteins | mean ddg |
|---|---|---|---|
| train | 2,669 | 57 | 1.01 |
| val | 387 | 15 | 1.50 |
| test | 347 | 27 | 1.03 |

`homologue_free_thermompnn` marks ThermoMPNN's 89-protein "homologue-free" list.

**Homology.**
- 2/27 test proteins and 12/57 train proteins have a Megascale-train homologue.
- Rows with a Megascale-train homologue: 25.6% (column `homolog_in_megascale_train`).
- Other columns: `pH`, `method` (GdnHCl 815, Thermal 714, plate-based chemical denaturation 645, Urea 638, ...), `source` (DOI) and `dTm`.
