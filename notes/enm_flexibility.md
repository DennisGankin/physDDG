# ENM site flexibility as a 1-weight term on B6g / B6gs1 (2026-10-26)

Code: `physddg/enm.py` (descriptors), `tests/test_enm.py` (8 analytic tests), `scripts/120_enm_sites.py`
(descriptors for all benchmarks), `scripts/121_enm_select.py` (probe, screen, controls, SH3 check; train CV + val only),
`scripts/122_enm_test.py` (single test look, G6 transfer, Ssym). Outputs: `results/enm_flexibility/`.
Origin: critique 2, idea I2 (43% of the residual is site-level; the descriptor set had no dynamics).

## Pre-registered before any run
- Adoption bar: Megascale-val natural ρ_prot ≥ +0.004 **and** grouped-CV natural ρ_prot ≥ +0.004 with ≥ 4/5 folds up
  (vs the refitted base, identical config); at the single test look, no loss on FireProt-test or S669-strict worse than −0.01.
- Primary candidate: GNM MSF, per-structure z of log, multiplicative on the engine energy, ΔΔG = core + w·f·ΔE_engine.
- Screen: 11 descriptors × 3 normalisations × 8 forms = 264 configurations per stage (stage 2 on B6g, stage 1 on
  B6gs1). The best by Δval + ΔCV goes to the test look as `best`. **Multiplicity is disclosed: it was chosen from 264.**

## Descriptors (`enm_descriptors(protein or CA coords, plddt=None) -> dict`)
GNM MSF (10 Å, all modes; slowest 10 modes), ANM MSF (15 Å), CA+side-chain-centroid ANM MSF of the centroid node
(`cen_msf`, 12 Å), Eyal–Bahar effective spring constant (mean over j within 10 Å / all j), PRS effectiveness /
sensitivity (analytic isotropic force average), DFI, slowest-mode hinge score, contact degree, and a pLDDT-scaled GNM
(γ_ij = √(p_i p_j); AF models only). Raw values are kept, plus per-structure log / z / rank. Tests: path-graph and cycle
spectra (exact), cycle MSF = (n²−1)/12n, k zero modes for k components, the 6 ANM rigid-body zero modes, rotation/
translation invariance, symmetric PRS / positive stiffness, pLDDT=100 → unweighted. Cost: all 1,807 structures of the
nine benchmarks (196k residues) in 131 s on 48 cores (≤ 0.2 s for a 450-residue protein, single thread).

**Structure sources.** Per-residue log GNM MSF has the same distribution on Megascale AF2 models (median −2.52),
Megascale designs (B = 0, −2.57), FireProt crystals (−2.64), Domainome AF crops (−2.51). ANM and centroid MSF grow with
protein size (log ANM −1.08 Megascale vs −1.47 FireProt), so the per-structure z / rank versions are the usable ones.
**All descriptors are mostly burial:** Spearman(GNM MSF z, contact number) −0.88 (AF2), −0.87 (crystal), −0.85
(Domainome); cen_msf vs heavy atoms within 8 Å −0.91. On crystals the GNM MSF follows the B-factor (+0.27 pooled z).
Caveat: the pLDDT variant treats B-factors as pLDDT when they lie in [0, 100] with mean ≥ 30; that fires on a few
FireProt crystals, so the scripts use it only where `struct_source` is AF (else plain GNM).

## Antisymmetry (D15)
- Additive `w·f_i` is a WT-site main effect: it adds +w·f_i to both direct and reverse, so it cannot cancel. Ablation only.
- `(1 + w·f_i)·ΔE` keeps whatever antisymmetry ΔE has, provided f_i is similar in the two structures. But the B6g core is
  not antisymmetric (Ssym bias +1.34 kcal/mol), so amplifying it at stiff sites amplifies the defect:
  Ssym r(dir, −rev) B6g 0.265 → **0.162** with `best` (direct Pearson 0.701 → 0.719); additive 0.263; static control 0.323.
- `s_i·|ΔV|` is even under reversal (same problem as the additive form); `s_i·(−ΔV)` is odd.

## Results on selection data (Megascale-train grouped CV + val; `screen.csv`, `control.csv`, `incremental.csv`)
Base: B6g val 0.7411 / CV 0.6813; B6gs1 0.7262 / 0.6758 (reproduces `notes/ladder_physics.md`).

| term (1 weight) | stage 2 Δval / ΔCV (folds) | stage 1 Δval / ΔCV (folds) | w |
|---|---|---|---|
| **primary**: GNM MSF z × ΔE_engine | −0.005 / +0.003 (1/5) | −0.000 / +0.003 (4/5) | −0.16 / −0.19 |
| additive GNM MSF z (ablation) | +0.000 / +0.001 | −0.000 / +0.001 | −0.02 |
| **best**: cen_msf rank × (core − w0) | **+0.005 / +0.012 (4/5)** | **+0.011 / +0.014 (4/5)** | −0.56 / −0.59 |
| static control: contact-number rank × (core − w0) | +0.003 / +0.009 (5/5) | +0.010 / +0.007 (5/5) | −0.49 / −0.53 |
| cen_msf on top of the contact-number control (2 weights) | +0.002 / +0.003 (4/5) | +0.001 / +0.007 (4/5) | cn → −0.03 |
| cen_msf on top of heavy-atom-count control | +0.003 / +0.003 (5/5) | +0.002 / +0.003 (5/5) | |

- By form (median over descriptors): only multiplicative-on-the-whole-prediction (mulC) helps consistently (ΔCV ≈ +0.004);
  ×ΔE_engine helps CV but loses val at stage 2; term-specific modulation (×repulsion, ×vdW/solvation, 1–3 weights) ≤ +0.006 CV
  and loses val; additive offset ≈ 0; **stiffness × volume change (|ΔV|, growth only, signed) adds nothing** (ΔCV ≤ +0.003,
  and the growth-only weight takes the unphysical sign).
- Passing the bar: 3 configurations at stage 2 (cen_msf × mulC, all three normalisations), 11 at stage 1 (cen_msf, GNM, slow
  GNM, ANM, pLDDT-GNM, all mulC). Every fitted weight is negative: flexible sites are attenuated, stiff sites amplified.
- **Probe** (critique I2's gate, `probe_site.csv`): on B6g OOF site-mean residuals (per-protein offsets removed, 12,037
  sites) ENM descriptors add ΔR² ≤ 0.004 over the static set (RSA, contact number, heavy atoms, Cβ14, side-chain contacts,
  degree). On buried sites contact number correlates more with the residual (ρ +0.15, Cβ14 +0.19) than any ENM descriptor
  (cen_msf −0.12). ENM does not beat contact number here.
- Reading: the gain is a burial-dependent **scale on the whole prediction** (the fit is too flat at packed sites, too steep at
  loose ones). cen_msf is the best per-site proxy for that scale (it takes the weight from contact number in the joint fit),
  but its increment over a static control is +0.002–0.003 val / +0.003–0.007 CV, below the bar. This is not dynamics.

## SH3 buried-but-tolerant sites (train OOF; `sh3_sites.csv`)
ENM calls them **rigid**: cen_msf percentile in the domain 0.14–0.26 (2LJ3 V44, 6SCW V46, 2CDT V40, 1QKX V41, 1PWT V42,
1QLY/6NMW W36–37), GNM MSF z −0.15 to −0.79. So the term amplifies them: site residual (y − pred, protein offset removed)
2LJ3 V44 −1.95 → −2.04, 1PWT V42 −1.80 → −1.89, W37 −1.40 → −1.48. **Flexibility does not fix them**; an ENM of the AF2
model does not see their tolerance either. SH3 family per-protein ρ still rises (0.653 → 0.671) through the other sites.
Residual-analysis strata (OOF): buried small→large 0.530 → 0.561, buried 0.676 → 0.686, exposed 0.585 → 0.594.

## Single test look (`test_metrics.csv`, `test_paired.csv`; Δ vs refitted B6g / B6gs1, 95% cluster-paired bootstrap)
| set | B6g | +best | Δ [CI] | B6gs1 | +best | Δ [CI] | static ctrl (s2 / s1) |
|---|---|---|---|---|---|---|---|
| MS natural ρ_prot (21) | 0.692 | 0.697 | +0.005 [−0.013, +0.026] | 0.678 | 0.697 | +0.018 [−0.006, +0.024] | 0.694 / 0.689 |
| MS natural-strict ρ_prot (14) | 0.702 | 0.716 | +0.015 [−0.000, +0.038] | 0.690 | 0.708 | +0.018 [+0.005, +0.029] | 0.711 / 0.706 |
| MS all ρ_prot (28) | 0.692 | 0.696 | +0.005 [−0.009, +0.024] | 0.684 | 0.697 | +0.013 [−0.004, +0.024] | 0.691 / 0.694 |
| MS all pooled ρ / Pearson | 0.644 / 0.687 | 0.648 / 0.691 | +0.004 / +0.005 (CIs incl. 0) | 0.637 / 0.681 | 0.642 / 0.687 | +0.006 [+0.000, +0.013] / +0.006 | |
| MS natural site / within-site ρ | 0.740 / 0.486 | 0.742 / 0.489 | | 0.732 / 0.478 | 0.736 / 0.483 | | |
| FireProt-test (347) Spearman / Pearson | 0.474 / 0.451 | 0.478 / 0.464 | Pearson +0.013 [−0.016, +0.040] | 0.466 / 0.444 | 0.472 / 0.461 | +0.017 [−0.014, +0.048] | 0.481 / 0.473 (ρ) |
| S669-strict (470) Spearman / Pearson | 0.324 / 0.316 | 0.322 / 0.325 | ρ −0.002 [−0.019, +0.013] | 0.340 / 0.328 | 0.336 / 0.333 | ρ −0.004 [−0.024, +0.012] | 0.338 / 0.353 (ρ) |
| S461-strict (285) Spearman / Pearson | 0.559 / 0.549 | 0.556 / 0.561 | ρ −0.003 | 0.578 / 0.569 | 0.573 / 0.580 | ρ −0.006 | |
| Domainome all ρ_dom (518) | 0.418 | 0.430 | **+0.012 [+0.009, +0.023]** | 0.413 | 0.428 | **+0.015 [+0.010, +0.023]** | 0.426 / 0.424 |
| Domainome flag-kept | 0.449 | 0.467 | **+0.018 [+0.008, +0.023]** | 0.448 | 0.464 | +0.016 [+0.009, +0.025] | 0.461 / 0.454 |
| abundance DMS ρ_prot | 0.469 | 0.482 | +0.013 [−0.012, +0.020] | 0.468 | 0.476 | +0.009 | 0.482 / 0.477 |

- Primary (GNM × ΔE_engine) on test: MS natural 0.691 / 0.684, Domainome 0.416 / 0.411, S669-strict ρ 0.326 / 0.340
  (Pearson unchanged; the per-protein S669 median drops, but it is a median over tiny groups). Additive: pooled metrics change by ≤ 0.003 everywhere.
- Megascale-test strata (B6g → +best): small→large (all) 0.575 → 0.601, to-Pro 0.657 → 0.672, pLDDT < 50 0.380 → 0.436;
  **buried small→large 0.487 → 0.469 and from-Gly 0.645 → 0.615 get worse** (the OOF gain on buried small→large does not hold).
- Non-inferiority: FireProt-test and S669-strict changes are within −0.004 on every metric → the test part of the bar passes.

## G6 transfer (`transfer.csv`; Megascale-train-only vs FireProt-train-only fit, 100 cluster bootstraps of the second stage)
| term | w Megascale | w FireProt | raw ratio | sign |
|---|---|---|---|---|
| best (cen_msf rank × core), stage 2 / 1 | −0.56 ± 0.10 / −0.59 ± 0.11 | −0.60 ± 0.08 / −0.63 ± 0.09 | 1.08 / 1.06 | same |
| static control (contact-number rank × core) | −0.49 / −0.53 | −0.38 / −0.40 | 0.78 / 0.75 | same |
| primary (GNM z × ΔE_engine) | −0.16 / −0.19 | −0.28 / −0.32 | 1.75 / 1.67 | same |
| additive GNM z | −0.017 / −0.022 | −0.075 / −0.054 | 4.4 / 2.5 | same |
The multiplicative weights are dimensionless (they scale a prediction already in each assay's units), so the raw ratio is the
G6 quantity; the `ratio_over_1.7` column in the CSV applies only to the additive term. The cen_msf scale transfers cleanly
from AF2 minidomains to crystal structures.

## Verdict
- **Primary (pre-registered GNM MSF × ΔE_engine): rejected** (stage 2 Δval −0.005, folds 1/5). Additive site offset and
  stiffness × |ΔV|: rejected (≈ 0).
- **`best` = (1 + w·f_i)·(core − w0) with f_i = per-structure rank of the CA+centroid ANM MSF − 0.5, w ≈ −0.56: passes the
  pre-set bar** at both stages (val +0.005 / +0.011, CV +0.012 / +0.014, 4/5 folds; test: Domainome +0.012–0.018 with CIs
  excluding 0, Megascale natural-strict +0.015–0.018, FireProt/S669 non-inferior; G6 ratio 1.08, no sign flip).
- What it is, honestly: a packing-dependent scale on the whole prediction. A static contact-number rank in the same form
  recovers ~70–80% of the gain, and the ENM increment over it is below the bar. It is chosen from 264 configurations.
  It does **not** fix the SH3 buried-but-tolerant sites (ENM calls them rigid), it hurts buried small→large and from-Gly
  on the test set, and it lowers Ssym antisymmetry (r 0.265 → 0.162).
- Recommendation: if adopted, ship it as `B6g + site scale` (1 weight, `results/enm_flexibility/models/B6g+enm_best.json`,
  two-stage like `scale_plddt`: `flex_base` holds the core) and describe it as a site-sensitivity scale, not as dynamics.
  The static-contact version (`B6g+enm_ctrl.json`) is the simpler near-equivalent if an ENM dependency is unwanted.
  The site-level residual from "statically buried, dynamically tolerant" positions is not recovered by a CA/centroid ENM
  of the AF2 model.

## Caveats
- `scripts/52_ladder_physics.py` and the B6g JSONs were being modified concurrently by another agent; these runs refit every
  base from the current code and reproduce B6g val 0.7411 / CV 0.6813. Nothing in 52 or `physddg/fit.py` was edited.
- Two-stage fits: the second-stage bootstrap SE holds the base fixed (slightly optimistic SEs).
- Test numbers are for refitted bases (identical to B6g/B6gs1 weights). FireProt per-protein ρ (9 proteins) is not used for decisions.
- Large outputs (`preds_*.parquet`, `oof_s2.parquet`) are not committed; rerun 122 (~10 min) to regenerate.
