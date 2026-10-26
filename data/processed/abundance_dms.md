# abundance_dms — seven full-length abundance DMS on AlphaFold models (cross-assay test set)

**Source.** ProteinGym v1.3 substitution benchmark (processed, directionality-normalised scores):
`DMS_ProteinGym_substitutions.zip` from marks.hms.harvard.edu/proteingym/ProteinGym_v1.3 and
`reference_files/DMS_substitutions.csv` (GitHub OATML-Markslab/ProteinGym). Script: `scripts/25_abundance_dms.py`
(downloads into `data/raw/proteingym/`, resolves entry names to accessions through UniProt REST).

## Assays (single substitutions only)
| protein_id | UniProt | assay | variants | positions | AF pLDDT median | sites pLDDT<70 | ρ(score, CA count 10 Å) |
|---|---|---|---|---|---|---|---|
| PTEN_HUMAN_Matreyek_2021 | P60484 | VAMP-seq (GFP fusion, FACS) | 5,083 | 383 | 95.7 | 21.6 % | −0.39 |
| TPMT_HUMAN_Matreyek_2018 | P51580 | VAMP-seq | 3,648 | 241 | 98.3 | 5.7 % | −0.42 |
| CP2C9_HUMAN_Amorosi_2021_abundance | P11712 | VAMP-seq abundance | 6,370 | 486 | 95.4 | 3.0 % | −0.29 |
| NUD15_HUMAN_Suiter_2020 | Q9NV35 | abundance (VAMP-seq style) | 2,844 | 163 | 97.6 | 7.3 % | −0.45 |
| PRKN_HUMAN_Clausen_2023 | O60260 | abundance (GFP fusion, FACS) | 8,756 | 465 | 88.5 | 20.5 % | −0.50 |
| HXK4_HUMAN_Gersing_2023_abundance | P35557 | abundance (yeast) | 8,396 | 463 | 96.1 | 2.2 % | −0.23 |
| RASK_HUMAN_Weng_2022_abundance | P01116 | aPCA (yeast DHFR PCA; Domainome's assay) | 2,850 | 167 | 97.4 | 1.9 % | −0.20 |
Total 37,947 variants. Excluded on purpose: membrane / surface-expression assays (VKOR1, S22A1, KCNE1, KCNJ2, OPSD,
Elazar insertion sets), SPIKE (binding), OXDA, Q53Z42.

## Sign, numbering, structures
* `score` = ProteinGym `DMS_score`: **higher = more abundant = more stable** (opposite to `ddg`; `ddg` = NaN).
  Sanity: score falls with burial in every assay (ρ −0.20 … −0.50) and with site pLDDT.
* Assay positions are 1-based on ProteinGym `target_seq` (`assay_pos`); `seq_pos` = `pdb_resnum` = UniProt numbering of
  the AF model. For six assays target_seq is identical to the AF (UniProt canonical) sequence. **KRAS**: the assay used
  isoform 4B (P01116-2) while the AF model is canonical 4A; mapped by global alignment, keeping identical aligned residues —
  216 variants at isoform-specific C-terminal positions dropped. WT mismatch vs AF: 0 in all assays.
* `structure` = full-length AF model (human v4, `data/structures/abundance_dms/AF-<acc>-F1-model_v4.pdb`, B = pLDDT);
  the assays measured full-length proteins, so no cropping. `plddt` per site.
* `split` = external; `cluster` = UniProt accession.

## Known issues
* VAMP-seq / abundance readouts saturate (floor = degraded, ceiling = WT-like) and also capture degrons, trafficking and
  expression effects; ProteinGym scores are not calibrated to ΔΔG. Use rank metrics per protein.
* PTEN and PRKN have ~20 % low-pLDDT sites (disordered linkers/tails): useful for testing low-confidence handling.
* Score errors are not carried by ProteinGym; replicate-level data would need the original MaveDB/GEO sources.
