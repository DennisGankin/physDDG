# Key insights

Sorted into three categories and ranked within each by how general and how surprising they are. Each item says
what the evidence is and links to where it arose ([TP] = turning point in [JOURNEY.md](JOURNEY.md)).

## The half page: top scientific and methodological insights

**Scientific**
1. **Most of the predictable ΔΔG signal is *which position*, not *which substitution*.** Knowing only the average
   ΔΔG of each position in a held-out Megascale protein ranks its mutations at per-protein ρ ≈ 0.72, above every
   physics method tested (FoldX 0.63, PhysDDG 0.69). In the final model, burial and cavity terms carry more of a
   typical buried prediction than all force-field energies together (worked example V24A: 1.43 of 1.78 kcal/mol).
   [TP1](JOURNEY.md#tp1), [TP3](JOURNEY.md#tp3)
2. **A 10-parameter statistical model with no side chains already matches FoldX** on Megascale ranking (0.648 vs
   0.629) and on FireProt (0.43 vs 0.41), and on crystal benchmarks nothing in this project beat it clearly. Detailed
   atomistic energies add a real but modest +0.04. [TP3](JOURNEY.md#tp3)
3. **Published force-field constants are already close to optimal for ΔΔG.** Refitting 32 physical parameters
   (radii, well depths, solvation values) end to end moved each by less than ×1.6 and gave no ranking gain; fits on
   two independent assays agreed with each other (r = 0.80 over 32 parameters). The remaining error is not in the
   constants. [TP8 appendix](JOURNEY.md#appendix)
4. **On high-confidence regions, AlphaFold models are at least as good a structure for ΔΔG as crystal structures.**
   In 3,858 paired variants the AlphaFold-based prediction was never worse, and where the two structures differ it
   was better. Below pLDDT 70 every method, FoldX included, is at ρ ≈ 0, and the apparent pLDDT effect is largely a
   burial effect (low-confidence residues are mostly exposed). [TP6](JOURNEY.md#tp6)
5. **Abundance in cells (Domainome) is not folding stability at the surface.** Megascale-fitted amino-acid
   preferences transfer to Domainome in the core (ρ 0.84) but not at the surface (ρ −0.09): exposed hydrophobic
   substitutions lower abundance in cells without destabilising the fold, which is consistent with degradation or
   aggregation signals. The simplest two-weight model (burial + environment) was the best of its family on Domainome.
   [TP3](JOURNEY.md#tp3)

**Methodological**
1. **Model the wild type with exactly the same procedure as the mutant.** Re-using the experimental wild-type side
   chain makes every mutation look destabilising (forward/reverse pairs were biased by +8 kcal/mol before fitting).
   This is the main design rule behind FoldX-beating physics. [TP2](JOURNEY.md#tp2), [TP4](JOURNEY.md#tp4)
2. **Build the trivial baselines before the sophisticated model, and move the bar to them.** The project's original
   target ("beat FoldX by 0.08") would have been met by a lookup-style model; re-basing the gate on the 10-parameter
   model changed what counted as progress. [TP3](JOURNEY.md#tp3)
3. **For unreliable inputs, flag rather than correct.** Withholding ~14% of AlphaFold-model predictions improved
   the rest more than every correction that was fitted, and the one "AlphaFold correction" that seemed to work was
   matched by a burial-only control. This is selective prediction: it improves trusted outputs, not the flagged ones.
   [TP6](JOURNEY.md#tp6)
4. **Independent audits of the agent's own claims are worth their cost.** Two audits found a benchmark computed on
   338 instead of 347 rows (which hid the decisive rows), a rule derived from 2 test rows, and at least 12 test-set
   evaluations that shaped the model. They reversed the shipped model family. [TP8](JOURNEY.md#tp8)

## 1. Scientific / domain insights (full list)

| # | insight | evidence | where |
|---|---|---|---|
| S1 | Position sensitivity dominates (site-mean oracle ρ ≈ 0.72 per-protein, 0.76 pooled; site-level ρ 0.68 vs within-site 0.44 for the 10-parameter model) | `notes/baseline_ladder.md`, `scripts/lit_site_vs_substitution.py` | TP1, TP3 |
| S2 | 10-parameter statistics ≈ FoldX; physics adds +0.044 [0.018, 0.071] | recomputed here from `preds_megascale.parquet` | TP3 |
| S3 | Literature force-field constants are near a local optimum for ΔΔG; corrections are consistent across assays but tiny | `notes/diff_params.md` | appendix |
| S4 | AlphaFold models are not worse than crystals for ΔΔG at high confidence; the crystal is the fragile input where they differ | `notes/af_mode.md` §1 | TP6 |
| S5 | Low pLDDT failure is mostly a burial confound; construct termini (ρ 0.11–0.15) and metal sites are the real failures | `notes/af_mode.md` | TP6 |
| S6 | Domainome surface signal ≠ folding stability (core transfer ρ 0.84, surface −0.09) | `notes/baseline_ladder.md` | TP3 |
| S7 | Residual error splits ~43% between-site / 57% within-site, and only ~17% of the between-site part is predictable from 99 structural descriptors | `notes/residual_analysis.md` | TP7 |
| S8 | Elastic-network "flexibility" adds nothing beyond contact number (correlation −0.85 to −0.91): in these small domains, dynamics descriptors are burial in disguise | `notes/enm_flexibility.md` | appendix |
| S9 | Megascale double mutants show little consistent pairwise coupling once the assay floor is removed (mean ε −0.23, no distance trend) | `notes/ladder_physics.md` (doubles) | appendix |
| S10 | A fixed backbone cannot be made forward/reverse consistent: the defect is almost entirely the repulsion term | `results/ladder_physics/antisymmetry*.csv` | TP4 |
| S11 | Megascale reference energies are partly assay-specific (zeroing them helps Domainome +0.02 but hurts all ΔΔG sets) | model card, A1 diagnostic | TP8 |
| S12 | Evolutionary profiles carry information physics lacks: one profile weight adds +0.07 on Domainome and +0.04 on Megascale (not part of PhysDDG proper) | `notes/evo_extension.md` | appendix |

## 2. Methodological insights (full list)

| # | insight | evidence | where |
|---|---|---|---|
| M1 | Same protocol for wild type and mutant; never reuse the native side chain at the site | Ssym bias +8 → fixed by design + fitting; WT→WT = 0 test | TP2, TP4 |
| M2 | Climb a baseline ladder (lookup → burial → statistics → physics) and gate each rung on the previous | `notes/baseline_ladder.md`, D16 | TP3 |
| M3 | Flag unreliable inputs instead of correcting them; test any "correction" against a matched burial control | `notes/af_mode.md` 5.2 | TP6 |
| M4 | Have the design critiqued before building; pre-register kill gates for expensive components | `notes/critique_design_v0.md`, gates G1–G7 | TP2 |
| M5 | Independent audits recompute every headline from raw predictions; report on identical rows; count test-set looks | `notes/audit_1.md`, `notes/audit_2.md` | TP8 |
| M6 | Benchmarks are leaky for any fitted classical method: KORPM's training set covers 90% of FireProt rows; S669 ρ falls 0.34 → 0.15 on the homology-strict subset | `notes/competitors.md` | TP3 |
| M7 | Compare to FoldX both raw and clipped; most of FoldX's Pearson gap is its uncapped tail | audit 1 | TP8 |
| M8 | Treat assay censoring as intervals and weight by homology cluster (a safeguard; point metrics were insensitive to the loss) | `notes/baseline_ladder.md` | TP2 |
| M9 | Boltzmann averaging over a library ensemble beats single best-model energies for robustness and gives entropy for free | design critique, D5 | TP2 |
| M10 | Prefer few interpretable terms with sign constraints; check transfer by fitting on two assays separately (G6) and drop terms that flip sign | `transfer_*.csv` | TP3 |

## 3. Engineering details needed only for reproduction

- Sign convention: positive ΔΔG = destabilising; Megascale `ddG_ML` is flipped at ingestion (D1).
- Rotamer library derived by the project from PISCES (≤ 1.8 Å, 30% identity, 9,266 chains) because the Dunbrack 2010
  library requires a licence application (D3).
- Repulsion radii scaled ×0.9 (D14), repulsion signed-log transformed before fitting, per-pair cap 3 kcal/mol.
- Stage-2 parameters: ≤ 16 neighbours within 6 Å, ≤ 12 neighbour rotamers + native, 30 damped mean-field sweeps
  (damping 0.5), ≤ 20 site rotamers per amino acid.
- PyTorch is used only as a GPU array library; float64 in the fitter must not leak into the engine (a bug fixed in
  commit `860d136`).
- The released FoldX numbers were produced on Tsuboyama's AlphaFold2 models after RepairPDB with default options
  (options were never recorded; `notes/foldx_protocol.md`).
- On CPU the predictor is roughly 50–100× slower (worked example: 4 min for 4 sites).
- `scripts/57_export_final.py --check` detects stale shipped model files (the audit found hand-made aliases once).
- Structure type (`xtal|af|auto`) must be explicit: pLDDT terms are only used for AlphaFold inputs.
