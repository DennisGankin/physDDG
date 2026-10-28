# PhysDDG

A structure-based, physics-inspired predictor of the effect of single missense mutations on protein folding
stability (ΔΔG, kcal/mol, **positive = destabilising**). No deep learning: a GPU-batched force-field-style energy
(CHARMM19/EEF1-type terms, geometric H-bonds, screened electrostatics), a PDB-derived backbone-dependent rotamer
library with Boltzmann averaging over rotamers, optional mean-field repacking of the neighbouring side chains, PDB
statistical potentials, and a linear read-out fitted to ΔΔG (Megascale-train + FireProt-train).

Numbers fitted to ΔΔG, counted from the shipped JSONs by `scripts/57_export_final.py`
(`results/ladder_physics/models/SHIPPED.json`): **41** for `final_stage2.json` (20 weights + 20 reference
energies + intercept; one weight pinned at 0 → 40 effective), **43** for `final_stage1.json` (22 weights) and
**83** for `final_af.json` (21 + 20 + 1 plus its complete nested base model `scale_base`, 41 more, evaluated at
inference). The multi-assay variants are 40 / 42 / 81 and additionally estimated 22 nuisance numbers from ΔΔG
(FireProt scale and offset, 20 Megascale δ_ref) that are fitted but never applied at inference. In addition the
model uses tables estimated from structures,
never from ΔΔG (rotamer library, Ramachandran maps, environment and N-cap potentials), and ~15 frozen
CHARMM19/EEF1 constants with literature provenance (`physddg/params.py`). Torch is used only as a GPU array library.

Project write-up: `REPORT.md`. Full model card (terms, training data, results, weaknesses):
`notes/ladder_physics.md`. Engine internals:
`notes/engine_{geometry,energy,mutate}.md`; AlphaFold handling: `notes/af_mode.md`; research log: `RESEARCH_TREE.md`.

## Install
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # Python 3.12, CUDA GPU recommended
export PYTHONPATH=$PWD
```
The rotamer library and statistics (`physddg/data/*.npz`) and the fitted weights
(`results/ladder_physics/models/*.json`) are in the repository. A CUDA GPU is strongly recommended (CPU works, slowly).

## Usage
Command line:
```bash
scripts/physddg predict --structure 1abc.pdb --mutations "A12G,L45P,B:K7E" --out preds.csv
scripts/physddg predict --structure model.cif --scan-all --sites 10-80 --gpus 0,1 --terms --out scan.csv
scripts/physddg predict --structure AF-P12345-F1.pdb --mode af --crop A:30-140 --scan-all --out af_scan.csv
scripts/physddg predict --table variants.csv --gpus 0,1,2,3 --out preds.csv   # columns structure,chain,pdb_resnum,wt,mut
```
Python:
```python
from physddg.predict import PhysDDG
m = PhysDDG()                       # default: results/ladder_physics/models/final_stage2.json (stage 2)
df = m.predict("1abc.pdb", ["A12G", "L45P"], terms=True)
m2 = PhysDDG(stage=1)               # fast: final_stage1.json
m3 = PhysDDG(mode="af")             # AF mode: final_af.json + the D18 treatments; m3.predict(f, ..., crop="A:30-140")
m4 = PhysDDG(weights="alt")         # multi-assay joint fits (alt_multiassay_*.json); or weights="model.json"
```
Mutation syntax: `<wt><resnum>[icode]<mut>`, optionally prefixed by the chain (`B:L45P`); residue numbers are the
author numbering of the file. Output columns: `ddg`, `site_found`, `wt_match`, `flags`, and with `--terms` the
per-term contributions `c_<feature>` (weight × feature), `c_ref` (reference energies) and `c_intercept`; their sum
is `ddg`. WT→WT is exactly 0.

Flags (`;`-separated, empty = none): `wt_mismatch`, `site_not_found`, `site_missing_atoms`, `missing_atoms_8A`,
`hetatm_5A` (ligand/metal near the site: not modelled), `chain_break_10A`, `terminal` (≤ 2 residues from a chain end).
AF mode adds `low_plddt` (< 70), `very_low_plddt` (< 50), `near_low_plddt`, `terminal` (< 3 from an end),
`exposed_hydrophobic_patch` (missing partner), `metal_site` (Cys/His cluster), `site_trimmed` (the site lies in a
trimmed low-confidence segment; scored on the untrimmed model) and a boolean column `withheld`
(pLDDT < 70 ∪ terminal ∪ exposed hydrophobic patch ∪ metal site; recommended: do not use these numbers).
Weights are chosen by `physddg.predict.default_weights`; `--weights alt` selects the multi-assay joint fits and
`--weights path.json` any other model. The JSON schema and feature contract are in
`results/ladder_physics/models/README.md`. The `nuisance` block of a model (per-assay scale/offset, Megascale
δ_ref) is **never** applied at inference.

## Models and modes
Shipped weights follow the model card (`notes/ladder_physics.md`, D23); all files are written by
`scripts/57_export_final.py` (`--check` reports staleness).

| file (`results/ladder_physics/models/`) | = | fit | what is computed | use |
|---|---|---|---|---|
| `final_stage2.json` | B6g | Megascale-only | stage 2: stage 1 + mean-field repacking of the ≤ 16 side chains within 6 Å (identical candidate sets for WT and mutant) | **default** (`--stage 2`, the default) |
| `final_stage1.json` | B6gs1 | Megascale-only | stage 1: Boltzmann over the site's library rotamers in the fixed WT environment, + PDB statistics, N-cap, helix dipole | `--stage 1`: large scans, ~2× faster |
| `final_af.json` | B6gaf | Megascale-only | stage 2 + pLDDT attenuation, with the D18 treatments below | `--mode af`: AlphaFold models only (B-factor = pLDDT) |
| `alt_multiassay_stage{1,2}.json`, `alt_multiassay_af.json` | B6gj* | joint Megascale+FireProt | same forms | `--weights alt`: FireProt-like, abundance or AF assays |

`--mode af` (D18) additionally: optional construct crop (`--crop A:30-140`, equivalent to running on a construct
file of that range) → MAVISp-style trim of low-pLDDT 10-residue windows (< 50) and termini (< 70), packed segments
exempt → stage 2 with the AF side chains kept outside the 6 Å shell → pLDDT attenuation → AF flags and `withheld`.

Structure type decides whether any pLDDT term is used: `--structure-type xtal|af|auto` (default: `af` with
`--mode af`, else `xtal`; `auto` reads the header, an `AF-*-model_v*` file name, then per-residue-constant
B-factors in [0, 100]). Crystal input is never attenuated, even with AF weights; an AF model whose B-factor column
carries no pLDDT is scored without pLDDT terms and flagged `plddt_missing`.

The site is always modelled from library rotamers (the WT side chain at the mutated site is never reused), so WT and
mutant go through the same code path. The backbone is fixed.

### Accuracy (table copied verbatim from the model card, `notes/ladder_physics.md`; read the card for the CIs,
the known weaknesses and the selection history, and `REPORT.md` for the project-level write-up)
Per-model strictness (a model trained on FireProt also excludes FireProt-trainval homologues), so n differs
between families in the S669/S461/Domainome-strict columns; the card also gives a matched-row table.
Megascale/Domainome/abundance = per-protein median ρ; FireProt/S669/S461 = Pearson.

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

n per column: MS nat 22,159/21 proteins · MS nat-strict 15,733/14 · MS all 28,172/28 · FireProt 347/27
(FoldX 341/25, tmol 344/27) · S669-strict 488/87 Megascale-only, 470/82 joint · S461-strict 290/43 vs 285/42 ·
Domainome 509,388/493 · Dom strict 358,755/347 vs 323,643/320 joint · Dom flag-kept 441,283/493 ·
abundance 37,947/**7 proteins — a median over 7 is not interpretable**, quoted for completeness only.

**The headline comparison.** Megascale-test natural, 21 proteins, 22,159 variants, identical rows and
identical structures: **final_stage2 − FoldX-clipped = +0.063 [+0.006, +0.120]** per-protein ρ; Pearson 0.701
vs 0.596; RMSE 0.78 vs 1.85 kcal/mol; better on **19 of 21** proteins. Mean per-protein ρ 0.702 vs median
0.692. Independently reproduced by audit 2 at 2,000 draws.

Known weaknesses, all from the card: not antisymmetric (Ssym r 0.23–0.25 against KORPM 0.88); stabilising
mutations precision@5 0.15–0.23 against FoldX 0.219; ρ ≈ 0 below pLDDT 70 — flag and withhold rather than predict;
buried small→large 0.49 against FoldX 0.55; S669-strict parity with DDGun3D, not a win; Megascale is AF2 models, so
structure source is confounded with assay; the shipped default is not distinguishable from the previous rung
(+0.005 [−0.004, +0.019]) and the B6 term set was chosen with repeated test-set evaluations (audit 2 counts ≥ 12),
so those test deltas carry selection optimism; the abundance column is a median over 7 proteins and is not
interpretable.

## Limits (what PhysDDG does not do)
- **Ligands, metals, cofactors, nucleic acids** are ignored (HETATM groups are dropped; `hetatm_5A` flags affected sites).
- **Interfaces / binding** ΔΔG: only the chains present in the file are used as environment; no binding free energy.
- **Multiple mutants**: not supported as such (single substitutions only; summing single ΔΔGs is an additive
  approximation that ignores coupling).
- **Insertions/deletions, non-standard target residues** are not supported (modified residues in the input are
  mapped to their parent amino acid).
- **Disordered regions / low-confidence AF segments**: predictions there are unreliable (flagged); crop AF models to
  the assayed domain.
- **Backbone changes** (Pro/Gly effects beyond Ramachandran statistics, large cavities, register shifts) are not
  modelled; X→Pro relies on the Pro-specific terms and reference energies.
- Weights were fitted on Megascale (small domains, 25 °C, proteolysis-derived ΔG); absolute scale for other assays
  differs (FireProt ΔΔGs are ~1.7× larger). pH, temperature and salt are not inputs.

## Speed (RTX 4090, shipped models; end to end incl. loading, annotation, all features and scoring)
`scripts/59_speed_benchmark.py` → `results/mutate/speed_benchmark.csv`. Full saturation scans (19 × L variants);
1 GPU = in-process with the model loaded (a cold CLI start adds ≈ 4–6 s); 4 GPUs = sites sharded over one process
per GPU and includes ≈ 5–8 s of process spawn + CUDA init, so it only pays off for large proteins or stage 2.
The machine was shared with other jobs during this run; single-GPU numbers vary by ~20% between runs.

| protein (residues) | variants | stage 1, 1 GPU | stage 1, 4 GPUs | stage 2, 1 GPU | stage 2, 4 GPUs | FoldX 5.1 (5–10 CPU-s/mut) | KORPM (0.44 ms/mut, 1 core) |
|---|---|---|---|---|---|---|---|
| 7A38 A (59) | 1,121 | **0.9 s** | 8.3 s | 1.8 s | 9.0 s | 1.6–3.1 CPU-h | 0.5 s |
| 1ZD0 A (141) | 2,679 | **1.9 s** | 9.8 s | 4.0 s | 10.7 s | 3.7–7.4 CPU-h | 1.2 s |
| 6Q62 A (269) | 5,111 | **3.6 s** | 10.3 s | 8.2 s | 13.0 s | 7–14 CPU-h | 2.2 s |
| 3EDF A (597) | 11,343 | **10.0 s** | 13.2 s | 23.7 s | 19.0 s | 16–32 CPU-h | 5.0 s |

How to read this table: against FoldX the comparison is GPU-seconds versus CPU-hours (a 269-residue scan takes
3.6 s on one GPU against 7–14 CPU-hours for FoldX BuildModel). Per mutation, stage 1 is **at parity with KORPM**
(≈ 0.7 ms per mutation on one GPU vs 0.44 ms on one CPU core); PhysDDG is not the fastest method per mutation.

AF mode (crop/trim + stage 2 + attenuation + flags), 1 GPU, AlphaFold DB yeast models: 150 res 5.3 s, 298 res
12.0 s, 603 res 22.6 s. Stage 1 is ≈ 80% GPU and 20% CPU (0.8 s of 3.6 s at 269 residues: site descriptors, SASA,
structure-statistics tables). Whole tables on 4 GPUs (one process per GPU, CPU preparation overlapped):
Megascale-test 28,172 variants: stage 1 14.7 s, stage 2 23.5 s; Domainome 532,501 variants: stage 1 **71 s**
(7,500/s), stage 2 173 s, AF mode 178 s. Target (300-residue saturation scan < 1 min on one 4090, stage 1) met.

Example outputs (`--scan-all --terms`): `results/examples/1PSE_megascale_scan_stage1.csv` (crystal/Megascale domain,
stage 1), `results/examples/AF-A0A2R8Y422_PF00240_crop2-71_scan_af.csv` (AF model cropped to the Domainome
construct) and `results/examples/AF-P20486_yeast_scan_af.csv` (full AF model with trimming, flags, `withheld`).

## Reproducing results
Results come from numbered scripts; run them as `PYTHONPATH=. .venv/bin/python scripts/<script>` (the `*.sh`
ones with bash). The shipped model files (`final_*.json`, `alt_multiassay_*.json`, `SHIPPED.json`) are exported
from the fitted `B6g*.json` by `scripts/57_export_final.py`; `--check` reports staleness (audit 2, I8, fixed). **Only the code, the fitted models,
the PDB-derived tables (`physddg/data/*.npz`, 6 MB) and `examples/` are in the repository.** Everything else is
regenerated: `data/raw` (≈ 117 GB of downloads), `data/processed` (0.4 GB), `data/structures` (0.5 GB),
`data/features` (11 GB) and the large prediction tables are git-ignored. External requirements that are *not*
installed by `requirements.txt`: internet access (dataset downloads, UniProt/PDB/AFDB), the read-only local mirrors
on this machine (`/rsyncPDB`, `/af_data`, `/alphafold`, `/workspace/data`, `/workspace/Domainome`,
`/foldx_native_clean`), `mmseqs2` (homology clustering, script 14), and — for the competitor comparisons only —
FoldX 5.1, KORPM and DDGun3D (installed by scripts `70`–`72b`). A CUDA GPU is needed for anything using the engine.

Order and approximate wall-clock times on this machine (64 CPUs, 4 × RTX 4090):

| order | scripts | what | approx. runtime |
|---|---|---|---|
| 1 | `10_download_ht.py`, `11`–`13` | download + build Megascale, FireProt, S669/S461/Ssym/S2648 tables | ~1 h (mostly downloads) |
| 2 | `14_homology.py`, `15_validate_tables.py` | mmseqs2 clusters, leakage columns, table validation | ~15 min |
| 3 | `20_download_domainome_supp.sh`, `21`–`26` | Domainome, AF construct crops, paired xtal/AF, abundance DMS | ~2 h (AF tar indexing dominates) |
| 4 | `40_build_rotlib.py` | PISCES download + rotamer library, Ramachandran, entropy, environment potentials → `physddg/data/*.npz` | ~3 min (48 CPUs) + 2 GB download |
| 5 | `41`–`43`, `44`, `45a`–`45d` | geometry and energy validation (ideal ICs, builder RMSD/coverage, DSSP cross-check, gradients, native sanity, chi discrimination, kernel speed) | ~20 min |
| 6 | `50_features_simple.py` | structure-statistics features for all tables | ~10 min (48 CPUs) |
| 6b | `53_ncap_stats.py` | PDB-derived N-cap potential → `physddg/data/ncap_pref.json` (needed by the B6 terms) | ~2 min |
| 7 | `55_phys_features.py --stage 1` then `--stage 2` (optionally `--variants capA capB capC`) | engine features for all 9 tables → `data/features/phys_s{1,2}_*.parquet` | ~5 min + ~15 min per variant (2 GPUs) |
| 9 | `51_baseline_ladder.py`, then `52_ladder_physics.py --stage all`, `--stage fit`, `--stage final`, then `57_export_final.py` | baselines B0–B3+, then selection, fits, ablations, gates and the final models → `results/ladder_physics/models/*.json` + the shipped exports | ~1–2 h |
| 9b | `55_multiassay_report.py` | model-card headline tables (`multiassay_summary.csv`, `multiassay_deltas.csv`); note the shared `55_` prefix with `55_phys_features.py` (step 7) | ~10 min |
| 9c | `54_doubles_additivity.py` | additivity probe on Megascale double mutants | ~10 min |
| 10 | `30`–`32`, `60`–`61`, `70`–`74` | evaluation harness, FoldX/trivial baselines, tmol, KORPM, DDGun3D | hours (FoldX and DDGun dominate; external binaries) |
| 11 | `80`–`86` | AlphaFold-mode study and gates | ~1 h |
| 12 | `56`–`59`, `90`–`94`, `10a`–`10d`, `11x`, `12x` | rotamer recovery, ΔF sanity, backbone ensemble, speed benchmark, residual analysis, optional evolution/ENM/differentiable-parameter extensions | minutes to hours each |

Audit re-computations (`notes/audit_1.md`, `notes/audit_2.md`): `99_audit_eval.py`, `99_audit_dup_mutations.py`,
`99_audit_pdbstats_leak.py`, `99_audit_pdbstats_rederive.py` (minutes each, need the step-1–9 outputs).
`59_speed_benchmark.py` needs the AlphaFold DB yeast models under `/workspace/data/yeast_af/pdbs/` and the
local PDB mirror; the four crystal benchmark structures it extracts are listed in its docstring.

Smoke test of a fresh clone (code only, no data):
```bash
git clone <repo> physddg && cd physddg
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt     # ~35 s
scripts/physddg predict --structure examples/1PSE.pdb --mutations "I2A,V8A,Y17A,G20A,F49A" --out m.csv
# compare with examples/1PSE_mutations_expected.csv (and examples/1PSE_scan_expected.csv for --scan-all)
pytest tests/test_geometry.py tests/test_features_extra.py            # pass without any downloaded data
```
Verified on 2026-10-27: a fresh clone reproduces both expected files bit-for-bit. Tests that need the benchmark
tables or the local PDB mirror skip automatically.

Tests: `pytest tests/` (geometry, energy, mutate/predict, fit, eval, AF mode, features_extra regression).
