# PhysDDG — final report (2026-10-27)

PhysDDG is a structure-based, GPU-fast, physics-inspired predictor of single-point ΔΔG (kcal/mol, positive = destabilising).
It uses force-field-style energy terms, rotamer ensembles (Boltzmann-averaged), mean-field neighbour repacking and a small
linear read-out. There is no neural network and no learned embedding. Every number below traces to `notes/ladder_physics.md`
(the model card, top section), `notes/audit_1.md`, `notes/audit_2.md` and `RESEARCH_TREE.md`. The claims here are limited to
what the two independent audits allow (decisions D17, D20, D23).

## What was built
| piece | where |
|---|---|
| Literature review, target numbers, dataset/sign conventions | `LITERATURE.md`, `notes/lit_*.md` |
| Curated datasets, homology clusters, leakage columns (Megascale, FireProt, S669/S461/Ssym/S2648, Domainome, ProteinGym abundance sets, paired xtal/AF) | `data/processed/`, `notes/data_*_report.md` |
| Evaluation harness (per-protein / pooled metrics, cluster bootstrap, paired deltas, offset detection) | `physddg/eval.py`, `scripts/30_*` |
| Geometry: PDB-derived rotamer library and statistics, GPU side-chain builder | `physddg/{topology,structure,rotlib,builder}.py` |
| Energy terms on GPU (11 separate terms, verified CHARMM19/EEF1 parameters) | `physddg/{energy,params}.py` |
| Mutation engine: Stage 1 (site Boltzmann ensemble) and Stage 2 (mean-field flexible neighbours) | `physddg/mutate.py` |
| Fitting framework (censored/Huber losses, shrunk reference energies, grouped CV, bootstrap) | `physddg/fit.py` |
| AlphaFold mode (trim, crop, shell repack, pLDDT flags) | `physddg/afmode.py` |
| Predictor CLI and API | `scripts/physddg predict`, `physddg/predict.py`, `README.md` |
| Two design critiques, two audits | `notes/critique_design_v0.md`, `notes/critique_2.md`, `notes/audit_1.md`, `notes/audit_2.md` |

## Results (model card numbers; D17 protocol)
Shipped models: `final_stage2` (default), `final_stage1` (fast), `final_af` (AlphaFold inputs). The joint
Megascale+FireProt fits are the optional `alt_multiassay_*` variant. The Megascale-only family ships because it wins on
validation and cross-validation (the model-selection rule is in the card).

**Headline, Megascale-test natural (21 proteins, 22,159 variants, identical rows and structures):**
`final_stage2` − clipped FoldX = **+0.063 per-protein Spearman, 95% CI [+0.006, +0.120]**. Pearson is 0.701 vs 0.596 (clipped
FoldX) and RMSE is 0.78 vs 1.85 kcal/mol. The model is better on 19 of the 21 proteins. Audit 2 reproduced this independently.

| per-protein median ρ (Pearson for FireProt/S669/S461) | Megascale nat | Megascale all 28 | FireProt (347) | S669-strict | S461-strict | Domainome | Domainome strict |
|---|---|---|---|---|---|---|---|
| final_stage2 | 0.692 | 0.692 | 0.451 | 0.336 | 0.552 | 0.418 | 0.391 |
| final_stage1 | 0.678 | 0.684 | 0.444 | 0.346 | 0.572 | 0.414 | 0.389 |
| final_af (AF inputs) | 0.699 | 0.699 | 0.451 | 0.336 | 0.552 | 0.425 | 0.397 |
| alt_multiassay_stage2 | 0.690 | 0.684 | 0.478 | 0.319 | 0.531 | 0.450 | 0.403 |
| FoldX (raw / clipped) | 0.629 | 0.633 | 0.411 / 0.417 (341 rows) | – | – | 0.430 | 0.385 |
| tmol, refit | 0.622 | 0.626 | 0.477 (344 rows) | – | – | – | – |
| KORPM | 0.500 | 0.533 | 0.397 | 0.194 | 0.543 | 0.302 | 0.274 |
| DDGun3D | 0.516 | 0.502 | 0.488 | 0.347 | 0.552 | (28 domains) | – |
| 10-parameter simple baseline B3+ | 0.648 | 0.649 | 0.432 | 0.339 | 0.570 | 0.389 | 0.369 |

Deep-learning reference values from the literature (not run here): ThermoMPNN about 0.77 per-protein / 0.725 pooled on the
Megascale test set, SPURS 0.74 pooled, ESM-MSR 0.573 on Domainome (`LITERATURE.md` §2.1).

## What we can claim (D20/D23)
1. On natural Megascale-test proteins, PhysDDG ranks about 0.06 per-protein Spearman better than FoldX
   (CI 0.006–0.120) and is clearly better in Pearson and error size. Both models were scored on the same AlphaFold models;
   FoldX was run on Tsuboyama's AF2 structures (`notes/foldx_protocol.md`).
2. Speed: a full Stage-1 saturation scan of a 269-residue protein takes about 3.6 s on one RTX 4090 (about 1,500–2,000
   mutations/s per GPU). A scan of Domainome (532,000 variants) takes about 71 s on four GPUs. FoldX needs CPU-hours per
   scan, and KORPM is at parity per mutation (0.44 ms on a CPU core), so speed alone is not the contribution.
3. AlphaFold: predictions on crystal vs AlphaFold structures agree closely (median per-entity r 0.95 for the fitted
   models), and accuracy is never worse on the AlphaFold structure. The most useful AlphaFold handling found is flagging
   rather than correcting: withholding pLDDT<70, terminal, exposed-hydrophobic-patch and metal-site variants gains about
   +0.03–0.04 on Domainome. The AF-only pLDDT attenuation has a confidence interval that touches zero.
4. Parameters: 41 / 43 / 83 numbers fitted to ΔΔG (default / fast / AF, the AF count includes its nested base model),
   plus PDB-derived statistical tables and frozen literature constants. All free weights have the expected sign.
5. Negative results (kept in the research tree): FoldX-term reweighting, backbone ensembles, elastic-network flexibility
   as dynamics, end-to-end fitting of 32 physical parameters (small calibration gain only), pairwise/double-mutant terms,
   floor-aware loss, per-protein pLDDT scaling as an AF mode.

## What we do not claim
- That the physics generalises across assays. Evidence is per assay: on FireProt, S669, S461 and Domainome the model is at
  FoldX level or the difference is not significant, and on crystal benchmarks it does not beat the 10-parameter B3+ baseline.
- Closing the gap to deep learning. The structure-only model is below the published DL numbers (≈0.69–0.70 vs ≈0.77
  per-protein on Megascale). The optional evolution mode (`notes/evo_extension.md`) adds one sequence-profile weight and gains
  +0.040 on Megascale (CI [−0.005, +0.071]; +0.014 on the strict natural subset) and +0.074 on Domainome, but it is a different
  information source and is not part of the headline.
- Antisymmetry, stabilising-mutation ranking and buried small-to-large accuracy: these are weaker than competitors or FoldX
  (see the model card's known weaknesses).
- That selection never touched test data: at least 12 test evaluations occurred and at least 5 changed what ships (listed in
  the model card). Read marginal test deltas with that in mind.

## Open issues for the user
- **Rotamer library licence:** we derived our own library from PISCES-culled PDB structures. The Dunbrack 2010 library
  needs a manually approved application; tmol bundles a copy whose licence we have not resolved.
- Viral proteins have no AlphaFold model (T4 lysozyme etc.), so they are missing from the paired xtal/AF set.
- Megascale structures are themselves AF2 models, so structure source and assay are confounded in training.

## Reproducing
See `README.md` ("Reproducing results") for the script order, runtimes and what is regenerated. A fresh clone plus
`scripts/physddg predict` on `examples/1PSE.pdb` reproduces the bundled expected output (tested).
