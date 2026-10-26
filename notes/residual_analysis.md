# Residual analysis of B5: where the remaining error is, and which missing terms fix part of it (2026-10-26)

Scripts `scripts/90_residual_base.py` … `94_confirm_test.py`; new descriptor code `physddg/features_extra.py` (site
descriptors from the structure only + candidate terms). Outputs in `results/residual_analysis/` (CSV committed; `base.parquet`
and the GBM OOF table are git-ignored and regenerable in about 1 min). Logs: `logs/9x_*.log` (git-ignored).

**Protocol.** All diagnosis and selection used Megascale-**train** (out-of-fold B5 predictions, grouped 5-fold CV over
homology clusters, the same folds as `fit.cross_validate`) and Megascale-**val** (B5 fitted on train). 243,059 point rows,
13,536 sites with ≥ 5 variants. Residual `r_raw = y − pred`; `r_cal` = residual after a per-protein linear recalibration
(`y ~ a_p + b_p·pred`), which removes per-domain offset and scale. Per-protein Spearman ignores both, so `r_cal` is the part
that matters for ranking. Test rows were used once, in `94_confirm_test.py`, after both selection outcomes had been fixed.
Baseline = B5 as in `notes/ladder_physics.md` (stage 2, slog repulsion, DH electrostatics, 18 weights + 20 reference energies).

## 1. Decomposition (train OOF + val, `strata.csv`, `pairs.csv`, `within_site_classes.csv`)
- **Burial is U-shaped in the error.** Mean r_cal is +0.13 at RSA < 0.05 and +0.33 at ≥ 25 Cβ within 10 Å, but negative at
  intermediate burial. Buried rows (RSA < 0.25, 30% of rows) carry 47% of the squared error. Within-protein ρ is lowest for
  exposed sites (0.53 at RSA > 0.6).
- **Secondary structure / position.** The **N-cap is the clearest single defect.** Mutating an N-cap residue to a
  hydrophobic is under-predicted by +0.36 to +0.46 kcal/mol (Ncap→M/V/Y/W/F/I/L). Ncap→S/D/T/N is over-predicted by
  0.35–0.54 within the site (these are good N-caps). Within-protein ρ at N-caps is 0.32, against 0.77 at helix-interior
  sites. Strand-centre rows have an SSE ratio of 1.40 and helix-interior rows 0.86. Residues 1–2 from a terminus are
  over-predicted (−0.14), and the terminal residue itself has ρ 0.33.
- **pLDDT.** Mean r_cal is −0.12 to −0.13 for pLDDT 50–80, and within-protein ρ is 0.19 at pLDDT 50–70. The model
  over-predicts effects in low-confidence regions of the AF2 models.
- **Domain level.** Designed domains: mean r_raw −0.23 and calibration slope b_p 0.74 (natural 1.00). Within-protein ρ is
  identical (0.677 for both). Per-protein ρ rises with dG_wt (Spearman over proteins +0.30) and with the spread of y (+0.34):
  low-stability domains have a compressed range and are ranked worse.
- **Substitution level** (site effect removed, `r_within`):
  - Pro: G→P −0.63; →P at φ>0 (αL) −0.96. →P in helices +0.24 and at N1–3 +0.37.
  - Buried apolar→apolar is over-predicted (−0.26) and buried apolar→polar under-predicted (+0.22). Buried →D/H/N are
    under-predicted (+0.20 to +0.31).
  - →Cys is over-predicted everywhere (−0.2 to −0.3, the known Megascale Cys artefact). V→I −0.46. →D in strands +0.32.
  - These are class-mean biases of 0.2–0.5 kcal/mol, small next to rms r_within (0.5–0.8). Most within-site error is not a
    simple class offset.
- **Outliers** (top/bottom 1% of r_cal, cut-offs +1.81 / −1.67 kcal/mol; 50 of each in `outliers_pos.csv` /
  `outliers_neg.csv`). They are concentrated: 60% / 53% of the rows sit in sites with ≥ 3 outliers. The SH3 family
  (fam0409, 27 domains) supplies about 80 of the 2% most extreme sites in each tail. Hypotheses below come from the
  descriptor context, not from visual inspection of each model:
  - *Under-predicted (+):*
    (a) Buried small→large mutations in β-strands of very stable natural domains (1OPS, 2KCM, 6NMW, 1QLY, 2CDT, 1E6H, 4HCK;
    A/V→F/W/L/I, dG_mut ≈ −1, i.e. at the floor). Raw repulsion is 50–330 kcal/mol, but slog(rep)·0.14 turns that into
    0.6–0.8 kcal/mol. The fixed backbone cannot tell a real clash from a relaxable one, so the fit compresses all of them.
    (b) N-cap Thr/Asn/Ser → hydrophobic (2KRU T40, v2K43S_2KVV N52): the capping side-chain–backbone H-bond is lost.
    (c) Buried polar residues with 3–4 side-chain polar partners in turns (SH3 T43/T44 of 3CQT/6NMW/3I35, 3I35 D21/D23,
    2JTV T55): the H-bond network is under-weighted.
    (d) Super-core sites with very low void fraction (0.06–0.13) and ≥ 40 Cβ within 14 Å, where every substitution unfolds
    the domain (2O2W G44, 2KCM A60, 1QLY A6, 3CQT A5): y ≈ 3–4 against a predicted 0.5–1.3.
  - *Over-predicted (−):*
    (a) G→P at αL Gly in exposed loops (2KVT G15, 1TUD G18, 1F0M G44, 3CQT G33, 1SRM G32, 2MA4 G26; predicted +2.4 to +3.9,
    measured ≈ 0). The Rama penalty for Pro at φ > 0 is applied in full where the loop evidently tolerates Pro: disordered or
    mis-modelled loop, or the measured protein relaxes it. pLDDT is lower here (70–90).
    (b) One homologous buried strand position in six SH3 domains (2LJ3 V44, 6SCW V46, 2CDT V40, 1QKX V41, 1PWT V42; also
    W36/37 of 1QLY/6NMW). All 14–17 substitutions are measured ≈ 0 although the site is buried (RSA 0.12, 37 Cβ in 14 Å).
    This is family-consistent, so it is most likely real tolerance (a peripheral/loop-adjacent position in the solution
    ensemble) that the AF2 model does not show. No descriptor here captures it.
    (c) Designed EHEE_rd4 G8 (Gly at φ > 0, N-cap position, no pLDDT): every substitution is tolerated, predicted ≈ 1.9 from
    d_rama + gly_posphi. The design model's backbone at this Gly is probably not the real one.
    (d) Aromatic→Pro in buried strands (2JT1 W64P, 2L2P F25P, 1QLY W37P): the truth saturates at the floor (dG_wt + 1)
    while the prediction keeps rising.
    (e) X→Cys (assay).
- **Censoring** (`censoring.csv`):
  - `flag_mut_floor` rows (dG_mut < −0.5, 1.8% of rows) have mean r_cal +1.02 and carry 6.6% of the SSE.
  - 86% of the 11.4k excluded "dG < −1" rows are predicted below their lower bound, by a mean of 1.2 kcal/mol.
  - Rows predicted above dG_wt + 1 (0.9%) are over-predicted by 1.5 kcal/mol raw.
  - The model is too flat at the destabilising end and does not saturate at the per-domain floor. A per-domain saturating
    link is monotone within a protein, so it would change Pearson/RMSE, not per-protein ρ.
- Measurement noise from the reported CIs is tiny (mean σ²_meas 0.003 against var r_cal 0.42). It is not a usable noise
  ceiling here.

## 2. Site-level vs within-site
**Variance components** (method-of-moments random-effects ICC on r_cal, sites with ≥ 5 variants, `variance.csv`):
**43% of residual variance is site-level** (σ²_site 0.184, σ²_within 0.239; train 0.44, val 0.42). On r_raw the site share is
56%, of which 18 points are per-protein offsets that per-protein ρ ignores.

**Headroom** (`headroom.csv`, per-protein median ρ; "model" = calibrated B5):

| scenario | train OOF natural | val natural | val all |
|---|---|---|---|
| B5 | 0.673 | 0.733 | 0.706 |
| + site-mean residual, leave-one-out (site part perfectly known) | 0.835 | 0.846 | 0.838 |
| + site-mean residual, in-sample | 0.854 | 0.863 | 0.854 |
| model site means + perfect within-site ordering | 0.842 | 0.874 | 0.872 |
| LOO site-mean of y only (literature oracle) | 0.718 | 0.749 | 0.714 |
| LOO site-mean of y + model within-site deviations | 0.835 | 0.847 | 0.837 |

Both halves are worth about +0.11 to +0.14 per-protein ρ. The across-site ρ looks good (0.74) and the within-site ρ bad (0.48),
but per-protein ranking loses about as much to site-sensitivity errors as to substitution ordering. Our within-site ordering
is already worth +0.12 on top of the site-mean oracle.

**GBM probe** (diagnostic only, `probe_site.csv`; HistGradientBoosting, grouped CV): site descriptors (99 features in 12
families) explain only **R² 0.17 (CV) / 0.25 (val)** of the site-mean residual. Most site-level error is not recoverable
from structure-derived descriptors of the WT model.
- Drop-family losses (CV / val): pLDDT −0.037/−0.013, the model's own site mean −0.026/−0.023 (calibration non-linearity),
  burial −0.024/−0.018, packing −0.024/−0.013, domain −0.013/−0.019, charge −0.009/−0.002, H-bond −0.002/−0.014,
  WT identity −0.003/−0.033.
- SS position, termini and aromatic families add ≤ 0.007.
- Marginals: residual rises at the extremes of packing (natom8 > 89 → +0.13; void_frac < 0.15 → +0.14, > 0.39 → −0.12;
  hb_sc_any ≥ 2 → +0.13; pLDDT > 98 → +0.10; pLDDT < 88 → −0.07; ≤ 3 residues from a terminus → −0.09).

The within-site probe (`probe_within.csv`) reaches R² 0.20 / 0.26, almost all from amino-acid property vectors × B5 term
values (site families add ≤ 0.005). The within-site defects are interactions between substitution type and existing terms,
not missing site context.

## 3. Candidate terms (`candidates_*_round*.csv`, `forward_*.csv`)
51 single candidates plus 2 scale groups, each added to B5 with its physical sign constraint and refitted (Huber + censoring,
same config). Base: CV natural 0.6727, val natural 0.7329.

| candidate (1 weight each) | Δval nat | Δval all | ΔCV nat | folds ↑ | w | verdict |
|---|---|---|---|---|---|---|
| `site_scsc` = (side-chain–side-chain contacts of the WT side chain within 4.5 Å)/30, for every substitution | +0.011 | +0.003 | +0.012 | 4/5 | 1.04 kcal/mol | **accepted (strict)** |
| `ncap_pref` = ΔN-cap preference (Doig–Baldwin-type scale) at N-cap sites | +0.010 | −0.002 | +0.010 | 4/5 | 0.55 | passes alone; after site_scsc Δval +0.0004 |
| `site_hbsc` / `site_hbsc_bur` = WT side-chain polar partners (×burial) | +0.001 | +0.006 | +0.012 | 5/5 | 0.09–0.13 | CV only |
| `scale:flag_lowplddt` = B5 core × (1 − pLDDT/100) | +0.003 | +0.001 | +0.007 | 5/5 | −0.74 | CV only (attenuation at low pLDDT) |
| `site_ncap`, `ncap_any` | +0.004/+0.008 | ≈0 | +0.005 | 4–5/5 | | subsumed by ncap_pref |
| helix propensity (Pace–Scholtz; interior / all / exposed) | ≤ +0.001 | | ≤ 0 | | 0.07–0.15 | no |
| β-propensity (Minor–Kim; all / centre / exposed) | +0.003 | −0.002 | 0.000 | | 0.05–0.19 | no |
| helix dipole (Δq × N1–3 / C1–3) | 0.000 | 0.000 | 0.000 | | 0.08 | no alone (accepted late, relaxed) |
| cation–π loss, aromatic stacking loss (angle-weighted) | ≤ +0.004 | −0.009 | ≤ 0 | | 0.2–0.26 | no |
| salt-bridge loss (geometric), like-charge environment | ≤ +0.001 | | ≤ +0.001 | | | no |
| overpacking: ΔV+ × burial / × strand / × density | 0 | | 0 | | 0 (at bound) | no; signal already in e_lj_rep |
| repulsion re-forms: log(rep) × burial, linear-20 × burial | +0.004 | −0.003 | +0.003 | 4/5 | 0.17 | marginal |
| cavity: ΔV− × burial × (1 − void) | +0.004 | −0.005 | +0.001 | 2/5 | | no |
| hydrophobic transfer × burial (Fauchère–Pliska), polar burial | ≤ +0.001 | | ≤ +0.002 | | | no; already in solvation + env |
| Gly/Pro backbone entropy, Pro at φ>0 exposed (w −2.3), Pro in strand/helix | ≤ +0.002 | | ≤ +0.002 | | | no |
| terminal flags / exp(−d_term/3) / core × term3 | −0.004 | | +0.002 | 4/5 | negative | CV-only, val worse |
| per-SS scaling of the B5 core (helix, strand, both; buried, loop) | ≤ +0.003 | | ≤ +0.003 | ≤ 3/5 | | no |
| other site main effects (Cβ count 14 Å, HSE-up, atom density, void, strand centre) | ≤ +0.002 | | ≤ +0.004 | ≤ 3/5 | | no |

**Forward selection.**
- *Strict rule* (pre-set: Δval-natural ≥ 0.004 and CV up in ≥ 4/5 folds): **B5 + site_scsc** only (1 weight).
  CV 0.6727 → 0.6844, val 0.7329 → 0.7438.
- The val criterion is a median over only 20 natural proteins and is noisy: after site_scsc, ncap_pref has Δval-nat +0.0004
  but Δval-all +0.005, Δval-mean +0.004, ΔCV +0.007.
- *Relaxed rule* (secondary, also fixed before the test look: any of Δval-nat / Δval-all / Δval-mean ≥ 0.004, ΔCV ≥ 0.004,
  ≥ 4/5 folds) adds, in order: scale:flag_lowplddt, site_hbsc_bur, ncap_pref, helix_dipole.
  **B5 + 5 weights: CV 0.6727 → 0.7104 (+0.038), val natural 0.7329 → 0.7487, val all 0.7059 → 0.7278.**
  Weights: site_scsc 0.93, lowpLDDT scale −0.67, hbsc_bur 0.09/partner, ncap_pref 0.49, helix_dipole 0.087.
- Functional-form tests mostly fail: per-SS scaling, angular aromatic/cation–π terms, geometric salt bridges,
  helix/β-propensity and cavity/void all add nothing measurable. B5's environment potential (aa | CN × SS × φ/ψ basin) and
  the engine terms already carry the SS-propensity and pairwise signal at this resolution. The terms that work are
  **site-sensitivity descriptors that the RSA-based `bur` misses** (side-chain packing contacts, buried side-chain H-bond
  partners), **N-cap chemistry** and **pLDDT attenuation**.

## 4. Single-look confirmation (test sets; `confirm_test.csv`, `confirm_paired.csv`, 95% cluster-paired bootstrap)
| set (identical rows) | B5 | B5+strict (site_scsc) | B5+relaxed (5 terms) | FoldX |
|---|---|---|---|---|
| **Megascale-test all 28 proteins, per-protein median ρ** | 0.685 | 0.690 (+0.005 [−0.019, +0.025]) | **0.704 (+0.019 [−0.010, +0.044])** | 0.633 |
| Megascale-test all 28, pooled ρ / Pearson | 0.642 / 0.686 | 0.650 / 0.692 (+0.009 [+0.001, +0.016]) | 0.658 / 0.695 (+0.017 [+0.005, +0.029]) | 0.569 / 0.447 |
| Megascale-test natural 21, per-protein ρ | 0.687 | 0.690 (+0.004 [−0.009, +0.028]) | 0.712 (+0.026 [+0.004, +0.047]) | 0.629 |
| Megascale-test strict (19 non-homologous) | 0.684 | 0.680 (−0.004) | 0.692 (+0.009 [−0.002, +0.024]) | 0.629 |
| FireProt-test (347 rows) Spearman / Pearson | 0.479 / 0.456 | 0.463 / 0.435 (−0.016 [−0.042, +0.017]) | 0.474 / 0.447 (−0.005 [−0.030, +0.023]) | 0.449 / 0.411 |
| Domainome per-domain ρ (518 domains) | 0.411 | 0.431 (+0.020 [+0.009, +0.026]) | **0.438 (+0.026 [+0.015, +0.036])** | 0.430 (493) |

- Single terms on test (all 28, per-protein ρ): site_scsc +0.005, site_hbsc_bur +0.005, ncap_pref +0.004, helix_dipole
  +0.003, lowpLDDT scale +0.001.
- On FireProt, site_hbsc_bur alone helps (Spearman +0.018, per-protein +0.06) and site_scsc alone hurts (−0.016).
- The literature DL numbers (ThermoMPNN/SPURS per-protein 0.77–0.83 on the full Megascale test set) remain well above
  0.70 on the same 28 proteins.

**G6-style transfer** (`transfer_new_terms.csv`; refit on FireProt-train only vs Megascale-train only, 100 cluster
bootstraps; ratio divided by the 1.7 assay scale):
- **No sign flips**, including in sign-free refits.
- `ncap_pref` 0.50 ± 0.06 vs 0.93 ± 0.29 (ratio 1.09) and `helix_dipole` 0.091 vs 0.133 (0.86) transfer cleanly.
- `site_scsc` 0.95 ± 0.16 vs 0.58 ± 0.22: ratio 0.36, at the edge of the 3× rule.
- `site_hbsc_bur` 0.091 vs 0.036 ± 0.071: ratio 0.24, undetermined on FireProt.
- The pLDDT scale cannot be tested on crystal structures.

## Recommendation
1. **Add `ncap_pref` and `helix_dipole`** (2 weights, physically standard, transfer passes, CV +0.006 / +0.005 in the
   relaxed path, test non-negative).
2. **Add `site_scsc`** as the strict-rule winner (+0.012 CV, +0.011 val, Domainome +0.020 significant, Megascale pooled
   +0.009 significant). Flag that it does not transfer to FireProt (−0.016, n.s.; FireProt weight 0.6× scaled). It is a
   site-sensitivity correction for the compressed engine terms on small AF2/designed domains. Re-check after E1's D9 cap
   rerun, which may absorb it.
3. **Add the pLDDT attenuation scale** in AF mode only (CV +0.007 in 5/5 folds on Megascale's AF2 models; AF-mode G5
   already asks for it). Keep `site_hbsc_bur` optional: it helps FireProt and CV, but its Megascale val gain is
   within noise.
4. **Not worth adding:** explicit helix/β propensity, aromatic/cation–π angular terms, geometric salt bridges, per-SS
   scaling, cavity-void and Gly/Pro entropy terms.
5. The largest remaining error is site-level and is not predictable from WT-model descriptors (GBM R² ≤ 0.25):
   - SH3-type tolerant "core" positions;
   - Pro at φ > 0 in loops;
   - floor-saturated super-core sites, where a per-domain saturating link (dG_wt + 1) would help calibration, not ranking;
   - buried small→large in stable β-sheets, where fixed-backbone repulsion cannot tell relaxable clashes from real ones
     (backbone ensembles, G3).

## Caveats
- `scale:` features use the fold-specific B5 core (no leakage across folds).
- Literature scales (Pace–Scholtz, Minor–Kim, the N-cap scale in `features_extra.py`) are typed from memory of the papers
  and are approximate. Their weights are fitted, so only their shape matters, but verify them before citing.
- Design-model sites carry no pLDDT and are treated as pLDDT 100. FireProt crystal B-factors are not used as pLDDT.
- Megascale val natural has 20 proteins: per-protein-median gains below ~0.005 are not resolvable there, which is why the
  strict rule admits only one term.
- `scripts/52_ladder_physics.py` was being modified concurrently by the fitting lead. Its `feature_set` now uses
  `ss_broken`, which has zero variance on Megascale, so the B5 base numbers here are unchanged
  (CV 0.6727 / val 0.7329 reproduced).
