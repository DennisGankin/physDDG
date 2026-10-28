# Independent critique #2 — project state vs. the brief, remaining ideas, evaluation risks

Reviewer: external (stability prediction / physics methods / benchmarking). 2026-10-26. Scope: PROMPT.md,
RESEARCH_TREE.md (D1–D19), notes/{ladder_physics, residual_analysis, af_mode, evo_extension, audit_1,
competitors, tmol_eval}.md, README.md, LITERATURE.md §7. Review #1 (`critique_design_v0.md`) is assumed read;
I do not repeat it except where the project's own results now settle a point.

**Overall.** The engineering is excellent and the record is unusually honest (audit_1 + D17 + the D19.4
selection caveat are better practice than most published work). The *scientific* position is narrower than the
tree's tone suggests: **PhysDDG beats FoldX decisively on the one assay it was fitted on, and is at
FoldX/DDGun3D parity on every assay and structure source it was not.**

---

## 1. Brief requirement by requirement

| requirement | verdict | evidence |
|---|---|---|
| Beat FoldX **clearly** | **Met on Megascale only** | MS-test nat ρ_prot +0.063 [+0.008, +0.120]; all three MS subsets exclude 0; Pearson +0.105 vs *clipped* FoldX; RMSE 0.78 vs 1.84. But FireProt (0.464 vs 0.417, n.s.; FoldX wins ρ_prot 0.634 vs 0.578), Domainome (0.418 vs 0.430, **below**), S669-strict (0.316 vs DDGun3D 0.347) are parity-or-worse. |
| Close the gap to DL | **Partly, on one assay** | MS per-protein: (0.692−0.633)/(0.77−0.633) ≈ **43%** of the FoldX→ThermoMPNN gap; pooled ≈ 45–55%. Domainome: **0%** (we are below FoldX; DL 0.50–0.57). FireProt: ~20%, n.s. With `+evo`: MS 68%, Domainome ~85% — but that is not physics, and the MS gain vanishes on natural-strict. |
| **Generalise across assays** | **Not met** | On the two crystal benchmarks the full engine does **not** beat the 10-parameter descriptor baseline B3+: FireProt r 0.464 vs 0.456, S669-strict 0.316 vs **0.330**, S461-strict 0.549 vs **0.569**. The engine's entire measurable value (+0.044 MS, +0.03 Domainome) sits on AF2-model inputs. "The physics transfers" is currently unsupported for the *energy function*; G6 (weight stability) shows the weights are not absurd, which is weaker. |
| AF mode | **Met as characterisation, weak as a correction** | 5.1a/5.1b/5.1c are genuine results (AF ≥ crystal where they differ; low-pLDDT failure is a burial confound; crop > full-length; flagging > correcting). But every *treatment* is ≤ +0.017, the MAVISp-trim gain on Megascale is a test-set observation, and B6gaf − B5 is +0.012 [0.000, +0.027]. Also note the model is **fitted on AF2 structures**, so "AF mode" is close to tautological (§3.5). |
| Fast | **Exceeded** | 0.5–0.7 ms/mut (1 GPU), 300-res scan < 5 s, Domainome 532k in 63 s. Caveat the tree already makes: KORPM is 0.44 ms/mut on **one CPU core**, so speed is a deployment argument, not a scientific one. Stage 1 is ~50% pure-Python CPU — fix before quoting the number. |
| No deep learning | **Met** | Torch as array/autograd only; statistical potentials and rotamers are PDB counting; hhblits is HMM–HMM. |
| Interpretable, few parameters | **Met with a caveat to state** | 20 weights + 20 Δref are few, signed, and physical. But the model also carries PDB-estimated tables (env potential ~180 entries, N-cap 20, rotamer library, Rama) and ~15 frozen force-field constants. Say "40 parameters fitted to ΔΔG, plus N tables estimated from structures (not from ΔΔG)" — do not say "40 parameters". |

### What the report may claim
- Beats FoldX 5.1 on Megascale-test on rank, Pearson and RMSE simultaneously, on identical rows and the same
  structures, with cluster-bootstrap CIs excluding 0 on all three subsets.
- Closes roughly half of the FoldX→ThermoMPNN gap on Megascale with physics alone (two thirds with the optional
  evolutionary term, stated as a different information source).
- ~10³–10⁴× faster than FoldX; proteome-scale saturation scanning is routine.
- No deep learning; every fitted weight has the physically expected sign, in both an AF2-structure high-throughput
  assay and a crystal-structure low-throughput assay, without sign flips and within 3× (G6).
- On AlphaFold models: not degraded relative to crystals; where the two disagree the AF model is the better input;
  construct cropping matters; **flagging beats correcting** (14% withheld → +0.038 ρ_dom).

### What the report may **not** claim
- "Generalises across assays" / "the physics transfers". State the parity result instead, and state that the
  engine adds ≈0 over B3+ on crystal sets.
- "Closes the gap to DL" without "on Megascale".
- Any significance for B6g − B5 (+0.005 [−0.004, +0.019]), B6gaf − B5 (CI touches 0), Domainome B6g − B5, or
  any FireProt comparison (every FireProt pair is n.s.; ρ_prot there is a median over **9** proteins).
- Best-in-class on reverse mutations / antisymmetry (r 0.23–0.25, bias +1.4; KORPM 0.88, DDGun3D 0.99).
- An AF-*specific* correction that improves accuracy (§3.5).
- A stabilising-mutation advantage: precision@5 0.190 (B6g) vs FoldX 0.219; only B6gaf (0.229) edges it, on AF input.

---

## 2. The eight highest-value remaining ideas (non-DL, interpretable)

Ordered by (expected gain on the brief) / cost. "Gain" = Megascale-test per-protein ρ unless stated.

**I1. Make the headline model multi-assay, and decompose Δref.** The project already has the answer and is not
using it: B5j/B4j (joint Megascale-train + FireProt-train, Megascale δ_ref nuisance dropped at inference) scores
**Domainome 0.444 vs B5 0.412 and FoldX 0.430** — the only structure-only configuration that beats FoldX there —
and FireProt r 0.486 vs 0.467, at a cost of −0.006 on Megascale. This is exactly the mechanism critique #1
predicted (§2.1): the Megascale-only Δref is partly the assay's PSSM unfolded-state model, and removing it
improves transfer. It is the single cheapest way to fix the weakest brief requirement.
*Gain:* +0.03 Domainome, +0.02 FireProt, −0.005 Megascale. *Cost:* ~3 h (code exists; refit B6g jointly → B6gj).
*Risk:* Ssym and the S461/S669 "strict" columns must be redefined (§3.4); FireProt-test stays valid (0 exact
duplicates, 1/27 homologous).
*First experiment:* two fits — (a) B6gj joint; (b) **B6g scored with prior+shared Δref only, Megascale nuisance
zeroed, no FireProt data at all**. If (b) recovers most of the Domainome gain, the finding is "the Megascale
reference energies are an assay artefact", which is a result in its own right and needs no FireProt.

**I2. Site tolerance from an elastic-network model.** 43% of residual variance is site-level and a 99-descriptor
GBM recovers only R² 0.17 — but that descriptor set contains **no dynamics**: burial, packing, charge, H-bond,
SS, termini, pLDDT only. The flagship failure (six homologous SH3 buried positions where all 14–17 substitutions
measure ≈0) is precisely "statically buried, dynamically tolerant". A GNM/ANM mean-square fluctuation per site
(seconds per domain, one fitted weight) is the obvious untested physics.
*Gain:* +0.01–0.02, concentrated in the site-level half. *Cost:* ~1 day. *Risk:* moderate — D9's contact-number
"rigidity" cap variants were within noise, so flexibility may already be captured by contact number; use GNM
fluctuation as a *site-sensitivity scale*, not as a repulsion cap.
*First experiment:* add `log⟨Δr²⟩_GNM` and its interaction with burial to the B6g residual regression on
Megascale-train OOF; proceed only if it beats contact number in the drop-one probe.

**I3. Local relaxation under the engine energy (buried small→large).** The only stratum where FoldX still wins
(0.495 vs 0.548) and the one with a documented mechanism: raw repulsion 50–330 kcal/mol compressed to 0.6–0.8 by
`slog`. G3 (rigid backbone ensembles) failed — but that test used **unrefitted B4 weights** and random Cartesian
perturbation, so it did not test relaxation at all. A few steps of gradient descent on φ/ψ of the site ±2 and the
shell side chains, under the same differentiable energy, is the physically correct fix and is GPU-friendly.
*Gain:* +0.03–0.06 on the stratum, +0.005–0.015 overall. *Cost:* high (3–5 days). *Risk:* high — cost 5–10×,
and the stratum is ~10% of rows. *Gate:* must refit weights on the relaxed features; require ≥ +0.03 on the
stratum OOF in ≥ 4/5 folds **and** ≤ 2× stage-2 runtime, else drop and report the limitation.
*First experiment:* relax 2,000 buried small→large training variants only; check whether the repulsion
distribution loses its tail and whether the fitted `slog` weight rises.

**I4. E3 end-to-end fit of physical parameters — continue, but gate hard.** Fitting radii/well depths/solvation
ΔG_free/kT through Stage 1 is the principled version of "the fit compresses the terms". Prior gain is modest
because the ablation shows no group dominates (largest single group −0.047).
*Gain:* +0.01–0.03. *Cost:* already running. *Risk:* it converts frozen, citable constants into fitted ones —
the interpretability claim and the "transfer" claim both weaken. *Gate:* report the new parameter count
explicitly; require G6 transfer (no sign flips, ≤3×) on the *physical parameters*, not only the weights; require
the fitted radii/depths to stay within ±20% of CHARMM19/EEF1, else treat as a regression and say so.

**I5. A real unfolded-state model (tree item 4.5, still open).** Replace 20 free Δref with
r(aa) = EEF1 prior + f(aa) + g(aa | neighbour class i±1), fitted jointly, plus a per-assay nuisance. This both
addresses the Megascale-PSSM confound and the documented failure that Δref does not transfer to Domainome's
surface (core ρ 0.84, surface −0.09).
*Gain:* small on Megascale, +0.01–0.03 on Domainome/FireProt; conceptually it is what makes "the physics
transfers" defensible. *Cost:* 1–2 days. *Risk:* weak identifiability; shrink hard.
*First experiment:* check whether the fitted Megascale nuisance vector is large (> 0.4 kcal/mol for several
residues) — critique #1 predicted this and I1(b) tests it. Publish that vector either way; it is a useful
warning to anyone training on Megascale ΔG.

**I6. Use the 138k Megascale doubles.** Free, in-split, never touched. Three uses: (a) an **additivity residual
test** of the current model — a direct, label-backed probe of whether the engine has any genuine pairwise
content beyond additive main effects; (b) an additivity regulariser during fitting; (c) a small number of
explicit coupling terms (contacting-pair substitution), validated on doubles and then checked not to hurt
singles.
*Gain:* on singles probably +0.00–0.01; the real value is (a) as evidence and (b) multi-mutant capability, which
the README currently lists as unsupported and which DL ΔΔG models also handle badly. *Cost:* (a) 2 h, (b)+(c) 2 days.
*Risk:* low for (a); (c) risks fitting assay epistasis (both mutations censored together).
*First experiment:* (a) alone — plot measured ΔΔG(double) − ΔΔG(m1) − ΔΔG(m2) against predicted coupling and
against Cβ–Cβ distance. If the engine predicts the sign of coupling at contacting pairs, that is a strong,
cheap, DL-free result.

**I7. Assay-aware loss, not assay-aware output.** Two separate points. (i) *State clearly in the report that a
per-domain monotone link (the Domainome sigmoid) cannot change per-domain Spearman* — it is monotone in the
prediction, so ρ_dom is invariant. It can only help pooled ρ, RMSE, AUC and cross-domain comparability. The brief
asks for it; the honest answer is "it cannot move our primary metric". (ii) Where a link *does* matter is the
**training loss**: 1.8% of rows sit at the per-domain floor (dG_wt + 1) and carry 6.6% of the squared error, and
86% of the 11.4k excluded "dG < −1" rows are predicted below their bound. A per-domain saturating link in the
loss de-biases the weights.
*Gain:* +0.005–0.01 ranking (indirect), clear gain in RMSE/calibration. *Cost:* ~1 day (dG_wt and the interval
machinery exist). *Risk:* low. *First experiment:* refit B6g with a soft floor at dG_wt + 1 and compare grouped-CV
ρ and the fitted weight vector; if weights move > 20%, the current fit was floor-biased.

**I8. Leakage-safe additional ΔΔG data for the crystal regime.** We are weakest exactly where training data is
thinnest (FireProt-train = 2,669 rows / 57 proteins). FireProtDB 2.0 / S2648 / ThermoMutDB / ProThermDB could
multiply that. But KORPM is the cautionary tale this project already produced: 90% of FireProt, 69% of S669 and
97% of Ssym rows have a training homologue, and KORPM's S669 ρ falls 0.337 → 0.148 on the strict subset.
*Gain:* +0.02–0.04 on FireProt/S669-strict if done right. *Cost:* 2–3 days (curation dominates). *Risk:* **high**
— this is the single easiest way to produce a number that collapses under external scrutiny.
*Rule:* admit a new row only if its protein has < 25% identity (mmseqs2 **and** foldseek) to every protein in
FireProt-test, S669, S461, Ssym and Megascale-test; de-duplicate exact (protein, position, wt, mut) across sources;
keep pH/T as covariates or restrict to pH 6–8, 20–30 °C. *First experiment:* count how many rows survive that
filter before building anything. If it is < 1,000, drop the idea.

**Explicitly deprioritised** (asked about in the brief; reasons):
- *Ligand/metal/cofactor.* Megascale excludes such structures, so it cannot be fitted there; on Domainome metal
  sites (7% of rows) we already degrade **less** than FoldX (0.22–0.31 vs 0.14). Keep as flags + a scope limit.
- *Interface/oligomer burial.* Not measurable on any current benchmark (Domainome = isolated domains by design;
  Megascale = monomeric minidomains). Ship as a flag and say so; do not claim an untested capability.
- *Hydration / water-mediated H-bonds.* Low prior, no benchmark isolates it, high parameter cost.
- *Antisymmetry by construction.* B4-odd buys r 0.59 at −0.044 ρ. Report the defect, its mechanism (lj_rep,
  native-adaptation bias) and the 2-structure symmetrised column as the diagnostic; do not spend the accuracy.
- *Tuning on difficult strata.* This is how the G2 verdict already got into trouble; train-CV/val selection only.
- *Ensembling with tmol* (+0.03 indicative): real, but it averages two physics models at 13–50× runtime and
  destroys the single-engine story. A robustness note, not a product.

---

## 3. Evaluation risks that could still embarrass us, and reporting fixes

**3.2 Selection optimism, and a shipped model that is not better than the previous rung.** D19.4 already
discloses that the B6 term set was chosen after a test look. The consequence is sharper than the note says:
B6g − B5 = +0.005 [−0.004, +0.019] and **B6gs1 − B5 = −0.008**. The shipped default is statistically
indistinguishable from B5, and the fast default is worse.
*Fix:* present the model as "B5 + two standard terms (N-cap, helix dipole) kept because they transfer, at no
measurable cost"; put val/CV deltas next to every test delta; never bold a B6−B5 cell.

**3.3 Small-n headline statistics.** The primary metric is a **median over 21 proteins** (14 strict; FireProt
ρ_prot is a median over 9). A median of 21 has very wide sampling error, and the FoldX margin CI on the primary
subset is [+0.008, +0.120] — the point estimate is 15× the lower bound.
*Fix:* report mean **and** median per-protein ρ, n_proteins in every cell, the per-protein win/loss count
(e.g. "B6g > FoldX in k/21"), and a paired per-protein scatter figure. Write "+0.06 (95% CI 0.01–0.12)", never
"clearly ahead".

**3.4 Strict subsets are defined against the wrong training set for joint models.** `homolog_in_megascale_train
== 0` defines the strict columns, but B4j/B5j are also fitted on FireProt-trainval. Their S669-strict (0.350) and
S461-strict (0.585) cells are therefore **not strict** for those models. If I1 makes a joint model the headline,
every strict column must be re-derived as "no homologue in *any* training set used by that model".
*Fix:* make strictness a per-model property in the harness, not a fixed column.

**3.5 Structure source is confounded with assay — including in G6.** Training is on Tsuboyama **AF2 models**;
Domainome is AF crops; FireProt/S669/S461/Ssym are crystals. The pattern "engine helps on AF-based assays, adds
≈0 over B3+ on crystal assays" is consistent with *either* assay effects *or* a structure-source domain shift.
G6 ("Megascale vs FireProt weights") cannot separate the two, because it changes both at once — this weakens the
headline transfer evidence more than the note admits. It also means "AF mode" partly means "the mode the model
was fitted for".
*Fix:* (a) say so explicitly; (b) run the one available disentangling experiment — the 4,012-row paired xtal/AF
set: refit (or just rescore) on crystal vs AF members of the same pairs and compare term weights/accuracy;
(c) reframe the AF claim as "AF-native by construction, and we show the crystal is the fragile input", which is
both defensible and more interesting than "we correct AF models".

**3.6 FoldX baseline conditions.** Protocol is RepairPDB → BuildModel, FoldX 5.1, on the same structures
(documented in `/foldx_native_clean/README.md`, **outside this repo**). Unrecorded: `numberOfRuns`, temperature /
pH / ionic strength, and whether RepairPDB was applied to the AF2 models (it was run on "the benchmark
structures"; RepairPDB on AF2 models is standard but worth stating).
*Fix:* copy the exact command line, version string and a checksum of the prediction files into the repo, and
state numberOfRuns. Keep reporting FoldX raw **and** clipped, and state symmetrically that our own `slog`
squashing was selected on validation — both models benefit from a monotone transform.

**3.7 Multiplicity.** ~25 model variants × ~8 benchmarks × ~6 metrics have been computed. The FoldX margins
survive it; the marginal ones (B6gaf − B5, Domainome B6g − B5, G2's stratum) do not. *Fix:* designate one primary
metric and one primary comparison per claim before the final table; mark everything else exploratory.

**3.8 Smaller items.** (a) Megascale-test headline is not homology-clean (9/28 train homologues; strict is not
lower — disclose). (b) The evo mode's Megascale gain **does not survive natural-strict** (+0.014, 9/14 up): that
sentence belongs directly under the evo headline, with the UniClust30 statement. (c) PDB-statistics overlap with
test proteins: measured, negligible (≤0.0005 ρ) — one sentence. (d) Keep CXXC out of the headline. (e) Add the
missing CIs on FireProt/S669/S461/Domainome cells. (f) Close the `fit.py` float64 side effect before freezing.

---

## 4. Prioritised plan with decision gates

Pre-register this before any further test look. One final test look at the end; everything else on train-CV + val.

**Phase A — one day, cheap and decisive (do all four):**
- A1 `I1(b)`: score B6g with the Megascale Δref nuisance zeroed, on Domainome + FireProt.
  **Gate:** recovers ≥ half of the B5→B5j Domainome gain → the "Megascale Δref is an assay artefact" result goes
  in the report and I5 is justified. Otherwise the gain is FireProt data, and I1(a) is the route.
- A2 `I1(a)`: fit B6gj. **Gate:** Domainome ≥ FoldX + 0.01 with CI excluding 0 **and** Megascale loss ≤ 0.01 →
  B6gj becomes the shipped default and the generalisation claim is rebuilt around it.
- A3 `I6(a)`: additivity residuals on the 138k doubles. **Gate:** none — it is a reported diagnostic either way.
- A4 Reporting fixes 3.2, 3.3, 3.4, 3.6, 3.8e. No gate; do it.

**Phase B — two to three days, model work (parallel):**
- B1 `I2` GNM site tolerance + `I7(ii)` floor-aware loss, selected in **one** pre-registered round on train CV +
  val. **Gate:** ΔCV ≥ 0.004 in ≥ 4/5 folds and Δval ≥ 0.004, G6 ratio within 3× with no sign flip. Terms that
  pass go in; the test set is not looked at until Phase D.
- B2 `I4` E3 verdict. **Gate:** ≥ +0.01 CV, physical parameters within ±20% of literature values, G6 passes on
  the parameters. Otherwise report E3 as a negative result (valuable: "the frozen force field was not the
  bottleneck").
- B3 `I8` feasibility count only (how many rows survive the strict filter). **Gate:** ≥ 1,000 usable rows → build;
  else drop and say why, citing the KORPM leakage analysis.

**Phase C — only if Phase B leaves the gap, 3–5 days:**
- C1 `I3` local relaxation, with the gate stated in §2 (stratum ≥ +0.03 OOF in ≥4/5 folds, ≤ 2× runtime).
- C2 `I5` structured unfolded state, if A1 said the nuisance is large.
- **Stop rule:** if by the end of Phase B no change clears its gate, freeze the model. The remaining 43%
  site-level / 57% within-site residual is largely not recoverable from a fixed-backbone WT model, and the
  project's most defensible contributions (speed, honest benchmarking, the AF characterisation, the Δref/assay
  diagnostic) do not depend on another +0.01.

**Phase D — freeze, single test look, write.** Final table on identical rows, per-model strict subsets, mean and
median with n, CIs everywhere, FoldX raw + clipped, evo reported separately with its strict caveat, and an
explicit "where we are not better" table (audit_1 already has it — keep it in the paper, not in an appendix).
