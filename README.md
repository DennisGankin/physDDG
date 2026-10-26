# PhysDDG

A structure-based, physics-inspired predictor of the effect of single missense mutations on protein folding
stability (ΔΔG, kcal/mol, **positive = destabilising**). No deep learning: a GPU-batched force-field-style energy
(CHARMM19/EEF1-type terms, geometric H-bonds, screened electrostatics), a PDB-derived backbone-dependent rotamer library
with Boltzmann averaging over rotamers, optional mean-field repacking of the neighbouring side chains, PDB statistical
potentials, and ~20 interpretable weights plus 20 reference energies fitted to Megascale (Tsuboyama 2023).
Torch is used only as a GPU array library.

Status and numbers: `RESEARCH_TREE.md`, `notes/ladder_physics.md` (accuracy, gates), `notes/engine_*.md` (engine),
`notes/af_mode.md` (AlphaFold models).

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
scripts/physddg predict --structure AF-P12345-F1.pdb --mode af --scan-all --out af_scan.csv     # stage 2 (B5)
scripts/physddg predict --table variants.csv --gpus 0,1,2,3 --out preds.csv   # columns structure,chain,pdb_resnum,wt,mut
```
Python:
```python
from physddg.predict import PhysDDG
m = PhysDDG()                       # stage 1, weights results/ladder_physics/models/B4.json
df = m.predict("1abc.pdb", ["A12G", "L45P"], terms=True)
m2 = PhysDDG(stage=2)               # stage 2, B5; or PhysDDG(weights="path/to/model.json")
```
Mutation syntax: `<wt><resnum>[icode]<mut>`, optionally prefixed by the chain (`B:L45P`); residue numbers are the
author numbering of the file. Output columns: `ddg`, `site_found`, `wt_match`, `flags`, and with `--terms` the
per-term contributions `c_<feature>` (weight × feature), `c_ref` (reference energies) and `c_intercept`; their sum
is `ddg`. WT→WT is exactly 0.

Flags (`;`-separated, empty = none): `wt_mismatch`, `site_not_found`, `site_missing_atoms`, `missing_atoms_8A`,
`hetatm_5A` (ligand/metal near the site: not modelled), `chain_break_10A`, `terminal` (≤ 2 residues from a chain end),
`low_plddt` (< 70, `--mode af` only) plus the AF-mode flags of `physddg/afmode.py` when available.

## Modes
| mode | what is computed | weights | use |
|---|---|---|---|
| stage 1 (default) | every rotamer of the 20 residue types at the site in the fixed native environment; Boltzmann free energy and averaged terms | B4 | crystal structures, fastest |
| stage 2 | stage 1 + mean-field repacking of the side chains within 6 Å (identical sets for WT and mutant) | B5 | AlphaFold models, buried small→large substitutions |
| `--mode af` | stage 2 + flags for pLDDT < 70 / low-confidence neighbourhoods (see `notes/af_mode.md`; withhold flagged sites) | B5 | predicted structures |

The site is always modelled from library rotamers (the WT side chain at the mutated site is never reused), so WT and
mutant go through the same code path. The backbone is fixed.

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

## Speed (one RTX 4090 unless noted; end to end incl. loading/annotation; `scripts/59_speed_benchmark.py`)
Full saturation scans (19 × L variants), wall clock incl. loading, annotation, features and scoring; 1 GPU =
in-process with the model loaded (cold CLI start adds ≈ 4 s); 4 GPUs = sites sharded over one process per GPU
(includes ≈ 5 s process spawn + CUDA init, so it only pays off for large proteins / stage 2).

| protein (observed residues) | variants | stage 1, 1 GPU | stage 1, 4 GPUs | stage 2, 1 GPU | stage 2, 4 GPUs | FoldX 5.1 (5–10 CPU-s/mut) | KORPM (0.44 ms/mut, 1 core) |
|---|---|---|---|---|---|---|---|
| 7A38 A (59) | 1,121 | **0.6 s** (1,985/s) | 4.9 s | 1.7 s (664/s) | 6.2 s | 1.6–3.1 CPU-h | 0.5 s |
| 1ZD0 A (141) | 2,679 | **1.4 s** (1,903/s) | 5.7 s | 3.9 s (695/s) | 7.4 s | 3.7–7.4 CPU-h | 1.2 s |
| 6Q62 A (269) | 5,111 | **2.9 s** (1,775/s) | 6.4 s | 7.2 s (715/s) | 8.1 s | 7–14 CPU-h | 2.2 s |
| 3EDF A (597) | 11,343 | **7.8 s** (1,460/s) | 8.8 s | 21.0 s (541/s) | 12.8 s | 16–32 CPU-h | 5.0 s |

Whole benchmark tables on 4 GPUs (run_jobs: one process per GPU, CPU preparation overlapped):
Megascale-test 28,172 variants / 28 structures: stage 1 **9.5 s** (2,957/s), stage 2 17.1 s;
Domainome 532,501 variants / 518 structures: stage 1 **55 s** (9,719/s), stage 2 172 s (3,105/s).
Target (300-residue saturation scan < 1 min on one 4090, stage 1) met with ~20× margin.

## Reproducibility (every result comes from a script; run with `PYTHONPATH=. .venv/bin/python scripts/<script>`)
| step | scripts | outputs |
|---|---|---|
| datasets (Megascale, FireProt, S669/S461/Ssym/S2648, homology) | `10`–`15` | `data/processed/*.csv/parquet` + cards `*.md` |
| Domainome, AlphaFold crops, paired xtal/AF, abundance DMS | `20`–`26` | `data/processed/domainome*`, `paired_af*`, `abundance_dms*` |
| evaluation harness, FoldX and trivial baselines | `30`–`32` (`physddg/eval.py`) | `results/baselines/` |
| rotamer library, PDB statistics, ideal geometry, geometry validation | `40`–`43` | `physddg/data/*.npz`, `results/geometry/` |
| energy-function validation | `44`, `45a`–`45d` | `results/engine_energy/` |
| structure-statistics features, baseline ladder B0–B3+ | `50`, `51` | `data/features/simple_*`, `results/baselines_ladder/` |
| physics ladder B4/B5 (fit, ablations, gates, antisymmetry) | `52` (`physddg/fit.py`) | `results/ladder_physics/` (models `*.json`) |
| engine features for all tables (stage 1/2, cap variants) | `55` (`physddg/mutate.py`) | `data/features/phys_s{1,2}_*.parquet` |
| rotamer recovery, raw ΔF sanity, backbone ensemble, speed | `56`–`59` | `results/mutate/` |
| tmol comparison; KORPM, DDGun competitors | `60`–`61`; `70`–`74` | `results/tmol/`, `results/competitors/` |
| AlphaFold mode | `80`–`86` (`physddg/afmode.py`) | `results/af_mode/` |
| residual analysis, audit | `90`–`94`, `99_*` | `results/residuals/`, `notes/audit_1.md` |
Tests: `pytest tests/` (geometry, energy, mutate/predict, fit, eval, AF mode).
