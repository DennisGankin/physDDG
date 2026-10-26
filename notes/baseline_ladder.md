# Baseline ladder B0–B3+, fitter, G6 transfer (2026-10-26, fitting/statistics lead)

Code: `physddg/features_simple.py` (descriptors), `physddg/fit.py` (generic fitter), `scripts/50_features_simple.py`,
`scripts/51_baseline_ladder.py --stage select|fit|eval|transfer|report`. Outputs: `results/baselines_ladder/`
(models/*.json with weights, cluster-bootstrap SEs and provenance; metrics.csv with 95% cluster-bootstrap CIs;
paired_vs_foldx.csv; site_decomposition.csv; ssym_antisymmetry.csv; paired_af_agreement.csv; transfer_*.csv; main_table.md).
Feature caches `data/features/simple_<ds>.parquet` and `preds_*.parquet` are git-ignored and regenerable (features take ~30 s for all sets).

## Protocol
- **Fit set:** Megascale-train (Dataset 3 + 12.4k censored train rows as intervals). **Selection:** Megascale-val, natural
  domains, per-protein median Spearman; grouped 5-fold CV on train (by `cluster`) reported alongside (`selection.csv`).
  FireProt-train is used only for G6 and the joint variant B3+j. No test number was used for any choice.
- **Selected:** env potential `de_env_bb` (CN × SS3 × φ/ψ basin) beat CN×SS and RSA×SS by 0.01 on val and 0.02 in CV.
  Objective: Huber + censoring, row weighting, ref_l2 = 1e-2. All objective variants (Gauss/Huber, censored or not,
  per-domain balancing, rank surrogate 0.3) lie within ±0.007 on val and within CV SEM, so **the objective choice does
  not matter for the point metrics**. Censoring cannot show up here because test and val are scored on Dataset-3 rows only.
- **Models:** B0 = (wt,mut) pair means (380). Bref = intercept + 20 reference energies. B1 = burial (1−RSA).
  B2 = bur, bur×Δhyd, bur×ΔV−, bur×ΔV+ (PoPMuSiC split) + Δref. B2occ = the same with Occ burial
  (Σ V_j exp(−d²/2·3.5²) around Cβ). B3 = de_env_bb + bur (2 weights). B3r = B3 + Δref.
  B3+ = B3r + ΔE_rama, pre-Pro(i−1), X→P × WT-NH-H-bonded, G→X at φ>0, disulfide broken, cis-Pro lost, Occ×ΔV+, Occ×ΔV−
  (10 weights). B3+j = B3+ fitted on Megascale-train + FireProt-train, with dataset weight ∝ number of clusters,
  a FireProt scale/offset nuisance and a Megascale δ_ref nuisance that is dropped at inference (D7).

## Main table (identical rows per column; per-protein median Spearman = ρ_prot; r = Pearson)
| model | params | MS nat ρ_prot | MS nat-strict ρ_prot | MS all ρ | MS all r | FP-test r | FP strict r | S669 strict r | S461 r | Ssym dir r | Dom ρ_dom | Dom pLDDT≥70 | Abund ρ_prot | MS site/within | Dom site/within |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 380 pair means | 0.491 | 0.504 | 0.460 | 0.496 | 0.259 | 0.274 | 0.288 | 0.468 | 0.296 | 0.174 | 0.189 | 0.279 | 0.42/0.47 | 0.16/0.10 |
| Bref | 0+1+20 ref | 0.379 | 0.423 | 0.369 | 0.401 | 0.213 | 0.226 | 0.250 | 0.443 | 0.290 | 0.069 | 0.078 | 0.184 | 0.40/0.33 | 0.17/−0.10 |
| B1 | 1+1 | 0.528 | 0.516 | 0.465 | 0.478 | 0.231 | 0.230 | 0.148 | 0.439 | 0.337 | 0.389 | 0.390 | 0.457 | 0.61/– | 0.51/– |
| B2 | 4+1+20 ref | 0.581 | 0.576 | 0.538 | 0.598 | 0.320 | 0.322 | 0.289 | 0.578 | 0.546 | 0.296 | 0.296 | 0.396 | 0.63/0.37 | 0.44/−0.02 |
| B2occ | 4+1+20 ref | 0.567 | 0.576 | 0.531 | 0.612 | 0.313 | 0.315 | 0.283 | 0.560 | 0.550 | 0.282 | 0.279 | 0.422 | 0.63/0.36 | 0.42/−0.02 |
| B3 | 2+1 | 0.564 | 0.556 | 0.483 | 0.498 | 0.287 | 0.288 | 0.177 | 0.473 | 0.382 | **0.442** | **0.449** | **0.481** | 0.63/0.08 | **0.54/0.24** |
| B3r | 2+1+20 ref | 0.623 | 0.624 | 0.577 | 0.613 | 0.375 | 0.379 | 0.295 | 0.588 | 0.550 | 0.369 | 0.373 | 0.429 | 0.67/0.43 | 0.51/−0.01 |
| **B3+** | 10+1+20 ref | **0.648** | **0.649** | **0.603** | **0.631** | **0.432** | **0.434** | 0.330 | 0.604 | 0.586 | 0.389 | 0.400 | 0.446 | **0.68/0.44** | 0.51/0.04 |
| B3+j | 10+1+20 ref | 0.616 | 0.621 | 0.561 | 0.594 | 0.315* | 0.316* | **0.364** | **0.618** | **0.638** | 0.418 | 0.426 | 0.445 | 0.66/0.36 | 0.51/0.19 |
| FoldX 5.1 | – | 0.629 | 0.640 | 0.569 | 0.447 | 0.411 | 0.409 | – | – | – | 0.430 | 0.447 | – | 0.65/0.35 | 0.52/0.18 |
| site-mean LOO oracle | oracle | 0.719 | 0.714 | 0.757 | 0.770 | (54% rows) | | | | | 0.617 | 0.617 | 0.636 | | |

CIs: `metrics.csv`; e.g. B3+ MS-all ρ_prot 0.649 [0.61, 0.68], FoldX 0.633 [0.54, 0.68]; FP-test r B3+ 0.43 [0.17, 0.60].
Site/within: Spearman of site means (sites with ≥5 variants) / mean within-site Spearman (sites with ≥10 variants).
*B3+j on FireProt-test: see suspicious item 2. Coverage is 100% everywhere except S669 (2 rows of 3DV0_I residue 129
missing from E1's loader); FoldX covers 98.3% of FireProt-test and 95.7% of Domainome.

**Paired deltas vs FoldX** (identical rows, cluster-paired bootstrap; `paired_vs_foldx.csv`):
- Megascale-test, B3+: ρ_prot +0.016 [−0.03, +0.10] (n.s.); pooled ρ +0.034; Pearson +0.18 [+0.13, +0.23].
  FoldX's Pearson is poor (0.45) and its RMSE is 3.2 kcal/mol (scale), but its ranking is close to B3+.
- FireProt-test, B3+: Pearson +0.03 [−0.11, +0.09] (n.s.).
- Domainome, B3: ρ_dom +0.015 [−0.03, +0.06] (n.s.); Pearson +0.045 [+0.03, +0.06].

## G6 transfer: Megascale-train-only vs FireProt-train-only fits (`transfer_*.csv`, 200 cluster bootstraps each)
FireProt ΔΔG are ~1.7× Megascale's (the joint fit's FireProt scale is 1.70, offset −0.75). Ratios below are therefore
also given divided by 1.70.
| B3+ term | w_MS ± se | w_FP ± se | FP/MS | /1.70 | verdict |
|---|---|---|---|---|---|
| de_env_bb (per kT) | 0.157 ± 0.013 | 0.385 ± 0.089 | 2.45 | 1.44 | pass |
| bur | 1.56 ± 0.13 | 1.94 ± 0.38 | 1.24 | 0.73 | pass |
| ΔE_rama | 0.132 ± 0.025 | 0.153 ± 0.078 | 1.16 | 0.68 | pass |
| pre-Pro(i−1) | 0.133 ± 0.022 | 0.452 ± 0.172 | 3.39 | 2.0 | borderline (>3× raw, 2× scaled) |
| X→P × NH H-bonded | 0.85 ± 0.07 | 1.13 ± 1.21 | 1.33 | 0.78 | pass (FP undetermined) |
| G→X at φ>0 | −0.41 ± 0.08 | −1.40 ± 0.33 | 3.47 | 2.0 | borderline; negative = correction to ΔE_rama |
| disulfide broken | 0 (no rows in MS) | 5.5 ± 2.0 | – | – | **undetermined on MS** |
| cis-Pro lost | +0.75 ± 0.44 | −1.93 ± 0.22 | −2.6 | – | **sign flip (z = −5.4): delete** |
| Occ×ΔV+ (overpacking) | −0.03 ± 0.13 | 0.19 ± 0.39 | – | – | **zero in both: delete** |
| Occ×ΔV− (cavity) | 1.27 ± 0.06 | 2.33 ± 0.31 | 1.83 | 1.08 | pass |

The Δref vectors from the two fits correlate r = 0.79 (B3+), 0.85 (B3r) and 0.65 (B2). B2 and B3r pass G6 with no flip
and no >3× change. In B3+j the Megascale nuisance δ_ref exceeds 0.4 kcal/mol only for Cys (−0.63; next largest R +0.23).
By the critique's criterion, the Megascale reference energies are therefore mostly shared, except Cys.

## Interpretation
1. **What carries signal on Megascale.** Burial alone (B1) gives 0.53 ρ_prot on natural domains. The substitution-type
   term adds ~0.06–0.09 (B3→B3r +0.06, B2 vs B1 +0.05). Rama, Pro-NH, cavity and the rest add another ~0.025 (B3r→B3+).
   The environment potential on its own adds little to burial (B3 vs B1: +0.04). Its weight is 0.11–0.26 kcal/mol per kT unit (B3, B3r,
   B3+ 0.16), against kT = 0.59, so the raw PDB log-odds are over-confident by ~2–5×. Site-level ranking is the strong part: B3+ site ρ 0.68 vs FoldX 0.65.
   The weak part is within-site ranking (0.44; B0's pair lookup is best at 0.47), against a LOO site-mean oracle of
   0.72 ρ_prot / 0.757 pooled. **Most of the remaining headroom is in the per-site mean,** i.e. how sensitive each site is;
   the ordering of substitutions at a site has less.
2. **The B3+ bar for the engine.** A 10-weight, structure-statistics model with no side-chain modelling already matches
   or beats FoldX everywhere (n.s. on ρ, significant on Pearson). Read literally, G1's "B4 − B3 ≥ 0.03" with B3 =
   2 weights (0.556 ρ_prot) is a low bar. **I recommend judging B4 against B3+: B4 ≥ 0.68 ρ_prot (MS-test natural),
   ≥ 0.46 Pearson FP-test.** G1's "B4 ≥ FoldX + 0.08" means ≥ 0.71, which is at the site-mean-oracle level. That is ambitious.
3. **Overpacking (small→large) carries no signal in simple descriptors.** Occ×ΔV+ is 0 ± 0.13 once ΔE_env is present.
   Cavity creation (ΔV−) is strong and transfers. A clash/repulsion term (D9) therefore has to earn its weight from real
   geometry; there is no free signal to inherit from burial × volume.
4. **Domainome is a different target.** B3, with no Δref, is best (0.442 ρ_dom; on par with FoldX). Every model with a
   Megascale-fitted Δref or pair lookup loses its within-site ranking on Domainome (B3r −0.01, B0 0.10, Bref −0.10,
   against B3's 0.24). Checked directly: per-mutant-aa mean effects agree between Domainome and Megascale at buried sites
   (ρ = 0.84) but not at exposed sites (ρ = −0.09). The abundance assay penalises exposed →F/W/Y/I/L/V (consistent with
   degron/aggregation), while Megascale penalises →P/G/D. Consequence for D7/D8: **Megascale Δref transfers to cores,
   not to abundance-assay surfaces.** Domainome needs its own surface treatment (validation only, as planned), and the
   joint fit's partial recovery (B3+j 0.19 within-site) suggests shrinking Δref helps there.
5. **AF vs crystal (G5).** Simple descriptors and the model predictions agree at r = 0.96–0.985 between AF and xtal
   structures, with identical correlation to ΔΔG (B3+: 0.546 AF vs 0.543 xtal). G5's r ≥ 0.9 is therefore met trivially
   by burial-level features. The engine must do at least this well, and this agreement is not evidence that it handles AF.
6. **Antisymmetry.** Every model with a WT-structure burial main effect is strongly biased on Ssym: mean
   pred(dir)+pred(rev) = +1.8–2.1 kcal/mol, r(dir, −rev) ≈ −0.2 (B1 −0.95). The intercept alone contributes 2·w0.
   Antisymmetry needs ΔE terms evaluated on modelled mutant states, not WT-site descriptors. S2648 and Ssym overlap
   FireProt-trainval, so B3+j numbers there are not independent.

## Suspicious / caveats
1. **Megascale-test strict vs all:** no gap (0.648 vs 0.649), so G7 is not triggered for these baselines.
2. **B3+j drops on FireProt-test (0.315 vs 0.432) because of two rows.** Thioredoxin 2TRX C32A/C35A (redox-active CGPC
   disulfide) is measured at −4 kcal/mol (breaking it stabilises). `ss_broken`, fitted on 8 FireProt-train rows, predicts
   +4.4. Without these rows the scores are 0.416 vs 0.450. Catalytic/strained disulfides (CXXC) need an exception or a
   shrunk weight. Disulfide terms cannot be fitted from Megascale.
3. **`gly_posphi` is negative** because ΔE_rama already charges non-Gly residues at φ>0. The indicator is a correction;
   interpret the two terms jointly (the critique's collinearity warning applies).
4. The FireProt/S669 site decomposition is meaningless (too few multi-variant sites). The LOO oracle covers only 54% of
   FireProt-test rows and is undefined for Ssym (direct/reverse at the same site).
5. Megascale's FoldX RMSE (3.2) reflects scale: FoldX overestimates magnitudes on these small domains.
6. Recorded data issues: 3DV0_I residue 129 is missing from `structure.load_protein` (2 S669 rows).
   Megascale has no disulfide-broken rows, so `ss_broken` = 0 there.

## fit.py interface (for the physics-engine ΔE terms)
`FT.prepare_targets(df, "megascale"|"fireprot")` → y/lo/hi/sigma_meas; `FT.fit(table, FitConfig(features=[...], ref=True,
ref_nuisance=("megascale",), loss="huber"|"gauss", censor=True, within="rows"|"clusters"|"proteins", weighting="clusters",
rank_weight=..., signs={"term": +1}), n_boot=200)` → JSON-able model; `FT.predict(model, df)`; `FT.cross_validate` /
`FT.cv_metrics` for grouped CV. The table can be any frame with dataset, cluster, protein_id, wt, mut, y (+ lo/hi) and
numeric feature columns. Tests: `tests/test_fit.py`.
