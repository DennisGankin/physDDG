# PhysDDG

> **How this project was made — please read first.**
> This predictor was built as an activity to test whether AI agents can run a bioinformatics project.
> An AI agent (Claude, by Anthropic; models Opus 5.5 and Sonnet 5.5) acted as principal investigator. It
> directed other AI agents, and those agents wrote the code, curated the data, and ran the benchmarks and the
> audits. The human owner supplied the brief and the data access. The owner has not independently re-checked
> every result. Two independent AI audits are included ([notes/audit_1.md](notes/audit_1.md), [notes/audit_2.md](notes/audit_2.md)), and they corrected
> several claims that were too strong. Please use the results with this in mind. Details:
> [docs/PROVENANCE.md](docs/PROVENANCE.md).

**New here? Start with the [interactive explainer](https://dennisgankin.github.io/physDDG/explainer/) ([Markdown versions: method](explainer/METHOD.md), [research journey](explainer/JOURNEY.md), [insights](explainer/INSIGHTS.md)): the method in six steps, a worked example, how the agents arrived at it, and the key insights. Developer details (every script, speed table, run order) are in [README_DEV.md](README_DEV.md).**

PhysDDG predicts how a single amino-acid substitution changes the folding stability of a protein (ΔΔG, in
kcal/mol; **positive = destabilising**). It needs a protein structure, either a crystal structure or an
AlphaFold model. It uses no deep learning. The method has four parts:

- energy terms in the style of a force field (CHARMM19/EEF1-type van der Waals and solvation, geometric hydrogen
  bonds, screened electrostatics), computed in batches on a GPU;
- a backbone-dependent rotamer library derived from the PDB, with Boltzmann averaging over rotamers;
- optional repacking of the neighbouring side chains (mean field, stage 2);
- statistical potentials from the PDB and a small linear model fitted to ΔΔG.

The default model has 41 numbers fitted to ΔΔG (40 effective). It also uses tables estimated from PDB structures
and about 15 fixed CHARMM19/EEF1 constants taken from the literature.

## Results

All numbers come from the model card ([notes/ladder_physics.md](notes/ladder_physics.md), top section). The metric tables behind them are in
`results/ladder_physics/` (see [Reproducing the tables](#reproducing-the-tables)).

**Main result.** On the natural proteins of the Megascale test set (21 proteins, 22,159 variants, same rows and
same structures for both methods), the default model `final_stage2` ranks variants better than FoldX with its
output clipped to [−3, 8] kcal/mol. The difference in per-protein Spearman correlation is **+0.063, 95% CI
[+0.006, +0.120]** (cluster bootstrap). Pearson correlation is 0.701 against 0.596, and RMSE is 0.78 against
1.85 kcal/mol. The model is better on 19 of the 21 proteins.

**What we do not claim.**
- We do not claim that the method generalises across assays. On FireProt, S669, S461 and Domainome the model is
  at the level of FoldX, or the difference is not resolved. On crystal-structure benchmarks it does not beat a
  simple baseline with 10 parameters (B3+).
- The method is not close to deep-learning predictors. Published deep-learning numbers on the same Megascale test
  set are about 0.77 per-protein Spearman (ThermoMPNN), against 0.69–0.70 here. We did not run those models.
- The model selection was not fully blind to the test sets: at least 12 evaluations used test data, and at least 5
  of them changed which model ships (listed in the model card). Small differences between models carry some
  selection optimism.

Accuracy table, copied verbatim from the model card. Megascale, Domainome and abundance columns are the median
of per-protein Spearman correlations; FireProt, S669 and S461 columns are Pearson correlations. "str" (strict)
columns remove proteins with a homologue in the training data of that model.

| model | MS nat | MS nat-str | MS all | FireProt (347) | S669-str | S461-str | Domainome | Dom strict | Dom flag-kept | abundance |
|---|---|---|---|---|---|---|---|---|---|---|
| B3+ (no engine) | 0.648 | 0.649 | 0.649 | 0.432 | 0.339 | 0.570 | 0.389 | 0.369 | 0.407 | 0.446 |
| B5 (previous rung) | 0.687 | 0.697 | 0.685 | 0.456 | 0.334 | 0.559 | 0.412 | 0.386 | 0.440 | 0.469 |
| **final_stage2** | 0.692 | 0.702 | 0.692 | 0.451 | 0.336 | 0.552 | 0.418 | 0.391 | 0.449 | 0.469 |
| **final_stage1** | 0.678 | 0.690 | 0.684 | 0.444 | 0.346 | 0.572 | 0.414 | 0.389 | 0.450 | 0.468 |
| **final_af** | 0.699 | 0.704 | 0.699 | 0.451 | 0.336 | 0.552 | 0.425 | 0.397 | 0.454 | 0.478 |
| alt_multiassay_stage2 | 0.690 | 0.698 | 0.684 | **0.478** | 0.319 | 0.531 | 0.450 | 0.403 | 0.480 | 0.484 |
| alt_multiassay_stage1 | 0.678 | 0.692 | 0.675 | 0.471 | 0.330 | 0.550 | 0.448 | 0.405 | 0.475 | 0.480 |
| alt_multiassay_af | 0.703 | 0.706 | 0.692 | 0.478 | 0.319 | 0.531 | 0.457 | 0.410 | 0.482 | 0.492 |
| FoldX raw / clipped | 0.629 | 0.640 | 0.633 | 0.411 / 0.417 *(subset, 341)* | – | – | 0.430 | 0.385 | 0.458 | – |
| tmol-refit | 0.622 | 0.637 | 0.626 | 0.477 *(subset, 344)* | – | – | – | – | – | – |
| KORPM | 0.500 | 0.513 | 0.533 | 0.397 | 0.194 | 0.543 | 0.302 | 0.274 | 0.315 | – |
| DDGun3D | 0.516 | 0.475 | 0.502 | **0.488** | 0.347 | 0.552 | *(subset, 28 dom)* | *(subset)* | *(subset)* | – |

Numbers of variants and proteins per column are in the model card. The abundance column is a median over only 7
proteins and cannot be interpreted.

Known weaknesses (from the model card): predictions are not antisymmetric (Ssym r(direct, −reverse) 0.24–0.27
for the shipped models; KORPM 0.88); stabilising mutations are ranked poorly at the top of the list; correlation
is close to 0 at sites with pLDDT below 70; buried small-to-large substitutions are worse than FoldX (0.49 against
0.55); and Megascale structures are AlphaFold models, so structure source and assay are confounded.

## Install

Python 3.10 or newer (tested with 3.12). A CUDA GPU is strongly recommended; the CPU works but is very slow (minutes for a few sites).

```bash
git clone https://github.com/DennisGankin/physDDG.git physddg && cd physddg
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # versions used for the results
export PYTHONPATH=$PWD
```

The rotamer library, the PDB statistics (`physddg/data/`, 6 MB) and the fitted models
(`results/ladder_physics/models/`) are part of the repository. No download is needed to make predictions.

## Quick start

Command line (`scripts/physddg`):

```bash
scripts/physddg predict --structure examples/1PSE.pdb --mutations "I2A,V8A,Y17A,G20A,F49A" --terms --out m.csv
scripts/physddg predict --structure model.cif --scan-all --sites 10-80 --gpus 0,1 --out scan.csv
scripts/physddg predict --structure AF-P12345-F1.pdb --mode af --crop A:30-140 --scan-all --out af_scan.csv
scripts/physddg predict --table variants.csv --gpus 0,1,2,3 --out preds.csv   # columns: structure,chain,pdb_resnum,wt,mut
```

Python:

```python
from physddg.predict import PhysDDG
m = PhysDDG()                                   # default model final_stage2.json
df = m.predict("1abc.pdb", ["A12G", "L45P", "B:K7E"], terms=True)
fast = PhysDDG(stage=1)                         # final_stage1.json, about 2x faster
af = PhysDDG(mode="af")                         # AlphaFold mode, final_af.json
alt = PhysDDG(weights="alt")                    # multi-assay variants
```

Mutations are written `<wt><resnum>[insertion code]<mut>`, with an optional chain prefix (`B:L45P`). Residue
numbers are the author numbering in the file. The output has the columns `ddg`, `site_found`, `wt_match` and
`flags`. With `--terms` it also has the contribution of each term (`c_<feature>`, `c_ref`, `c_intercept`); these
add up to `ddg`. The example output in `examples/` was produced by the first command above, and a fresh install
reproduces it to about 1e-5 kcal/mol on our GPU.

Flags (separated by `;`): `wt_mismatch`, `site_not_found`, `site_missing_atoms`, `missing_atoms_8A`, `hetatm_5A`
(a ligand or metal is near the site and is not modelled), `chain_break_10A`, `terminal`. AlphaFold mode adds
`low_plddt`, `very_low_plddt`, `near_low_plddt`, `exposed_hydrophobic_patch`, `metal_site`, `site_trimmed` and a
column `withheld`. We recommend that you do not use predictions with `withheld = True`.

## Models

| file (`results/ladder_physics/models/`) | use | what is computed | fitted on |
|---|---|---|---|
| `final_stage2.json` | **default** | site rotamers plus mean-field repacking of up to 16 side chains within 6 Å | Megascale-train |
| `final_stage1.json` | `--stage 1`: large scans | Boltzmann average over site rotamers in the fixed wild-type environment | Megascale-train |
| `final_af.json` | `--mode af`: AlphaFold models only | stage 2 plus attenuation by pLDDT | Megascale-train |
| `alt_multiassay_{stage1,stage2,af}.json` | `--weights alt`: FireProt-like, abundance or AlphaFold assays | same forms | Megascale-train + FireProt-train |

Which model when: crystal structure → `final_stage2`; saturation scans at proteome scale → `final_stage1`;
AlphaFold model → `final_af` with construct cropping and the `withheld` rule; if your target data resembles
FireProt or abundance assays → `alt_multiassay_stage2`. The schema of the JSON files is described in
[results/ladder_physics/models/README.md](results/ladder_physics/models/README.md).

## AlphaFold mode

`--mode af` is for AlphaFold models (pLDDT in the B-factor column). It does the following, in order: optional
cropping to the assayed construct (`--crop A:30-140`); trimming of low-confidence segments (10-residue windows
with pLDDT < 50 and termini with pLDDT < 70; well-packed segments are kept); stage 2 prediction with the AlphaFold
side chains kept outside the 6 Å shell; pLDDT attenuation; and confidence flags. The most useful step we found is
flagging, not correction: withholding variants at low pLDDT, at termini, in exposed hydrophobic patches and at
metal sites gains about +0.03 to +0.04 per-protein Spearman on Domainome. Crystal input is never attenuated.
Details: [notes/af_mode.md](notes/af_mode.md).

## Limits

- Ligands, metals, cofactors and nucleic acids are ignored (sites near them get the `hetatm_5A` flag).
- No binding or interface ΔΔG; only the chains in the file are used as environment.
- Single substitutions only. Summing single ΔΔGs for multiple mutants ignores coupling.
- No insertions or deletions. The backbone is fixed, so large backbone changes are not modelled.
- Predictions in disordered regions or low-confidence AlphaFold segments are unreliable (flagged).
- Weights were fitted on Megascale (small domains, 25 °C, proteolysis assay). The absolute scale differs for other
  assays (FireProt ΔΔGs are about 1.7 times larger). pH, temperature and salt are not inputs.

## Speed

On one RTX 4090, a full saturation scan of a 269-residue protein (5,111 variants) takes 3.6 s with stage 1 and
8.2 s with stage 2. All 532,501 Domainome variants take 71 s with stage 1 on four GPUs. FoldX needs CPU-hours for
the same scan. Per mutation, stage 1 is at the same speed as KORPM on one CPU core, so speed alone is not the
contribution. Full table: [README_DEV.md](README_DEV.md#speed-rtx-4090-shipped-models-end-to-end-incl-loading-annotation-all-features-and-scoring) and `scripts/59_speed_benchmark.py`.

## Reproducing the tables

The accuracy tables are in `results/ladder_physics/` (`metrics.csv`, `multiassay_summary.csv`, `matched_rows.csv`,
`multiassay_deltas.csv`), and the model card quotes them. Per-variant prediction files are **not** part of this
repository, because they contain FoldX values (see [THIRD_PARTY.md](THIRD_PARTY.md)). To regenerate the predictions
and tables, follow the run order in [README_DEV.md](README_DEV.md#reproducing-results); this needs the benchmark data
described in [data/README.md](data/README.md). A quick check that needs no data: the example in `examples/` and
`pytest tests/test_geometry.py tests/test_features_extra.py`.

## Retraining and refitting

To refit the weights you need the benchmark data and features, which are not in the repository (about 120 GB of
downloads). [data/README.md](data/README.md) explains where each dataset comes from, and
[README_DEV.md](README_DEV.md#reproducing-results) lists the scripts in the order to run them. In short: download and build the
tables (`scripts/10`–`26`), build the rotamer library and statistics (`40`, `53`), compute features (`50`, `55`),
fit (`51`, `52`), and export the shipped models (`57_export_final.py`). The competitor comparisons need FoldX,
KORPM and DDGun3D, which we do not distribute.

## Documentation

- [Explainer website](https://dennisgankin.github.io/physDDG/explainer/) and [explainer/](explainer/) (Markdown): the method, the research journey and the key insights (start here)
- [notes/ladder_physics.md](notes/ladder_physics.md): model card (terms, training data, results, known weaknesses)
- [docs/PROVENANCE.md](docs/PROVENANCE.md): how the AI agents built this, and the decisions D1–D23
- [REPORT.md](REPORT.md) and [RESEARCH_TREE.md](RESEARCH_TREE.md): project report and research log
- [notes/audit_1.md](notes/audit_1.md), [notes/audit_2.md](notes/audit_2.md): the two independent audits
- [docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md): what did not work
- [notes/](notes/): engine, AlphaFold-mode and experiment notes; [LITERATURE.md](LITERATURE.md): literature review
- [manuscript/](manuscript/): the Bioinformatics-format paper and supplement
- [README_DEV.md](README_DEV.md): technical README (every script, run order, speed table)

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the datasets and the tools listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Licence

MIT ([LICENSE](LICENSE)) for everything in this repository that is ours: code, documentation and result tables.
Third-party data and parameters keep their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
