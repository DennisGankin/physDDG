# Physics rungs B4 / B5: fitting, ablations, gates G1 / G2 / G6, antisymmetry (2026-10-26, fitting lead)

Code `scripts/52_ladder_physics.py` (`--stage select|ablate|fit|eval|antisym|transfer|report`), fitter `physddg/fit.py`
(+ `sym_pair` option added for D15). Inputs: E1's `data/features/phys_s{1,2}_<ds>.parquet` and my
`data/features/simple_<ds>.parquet`. Outputs in `results/ladder_physics/` (models/*.json with cluster-bootstrap SEs,
selection.csv, ablation.csv, metrics.csv, paired.csv, strata.csv, stabilising.csv, antisymmetry{,_terms}.csv,
paired_af_agreement.csv, transfer_B{4,5}.csv, weights.csv, main_table.md). Predictions `preds_*.parquet` git-ignored.

**Protocol.** Fitted on Megascale-train only (Dataset 3 + 12.4k censored rows as intervals), Huber + interval loss,
20 shrunk reference energies (D7). Everything selected on Megascale-val (natural, per-protein median ρ) plus grouped
5-fold CV on train; no test number entered any choice. B4 = stage-1 terms + pruned B3+ features (burial, Occ×ΔV−,
X→P×NH-H-bonded, G→X at φ>0, disulfide-broken with a CXXC exception; cis-Pro and ΔV+ dropped per G6). B5 = same with
stage-2 (flexible-neighbour) energies. `j` = joint fit with FireProt-train (FireProt scale 1.75, offset −0.61,
Megascale δ_ref nuisance dropped at inference).

## Main table (identical rows per column; MS = Megascale-test, ρ_prot = per-protein median Spearman)
| model | MS nat ρ_prot | MS nat-strict | MS ρ | MS r | FP r | FP ρ | S669 strict r | S461 r | Ssym dir r | Dom ρ_dom | Abund ρ | MS site/within |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3+ (no engine) | 0.648 | 0.649 | 0.603 | 0.631 | 0.432 | 0.455 | 0.330 | 0.604 | 0.586 | 0.389 | 0.446 | 0.71/0.42 |
| **B4** (stage 1) | **0.677** | 0.687 | 0.635 | 0.681 | 0.450 | 0.472 | 0.328 | 0.618 | 0.697 | 0.405 | 0.468 | 0.73/0.47 |
| B4-noenv | 0.662 | 0.672 | 0.625 | 0.669 | 0.439 | 0.458 | 0.318 | 0.602 | 0.686 | 0.380 | 0.468 | 0.72/0.45 |
| B4-phys (energies only) | 0.587 | 0.596 | 0.545 | 0.605 | 0.388 | 0.435 | 0.260 | 0.498 | 0.591 | 0.288 | 0.381 | 0.65/0.36 |
| B4-dF (ΔF + Δref only) | 0.599 | 0.604 | 0.539 | 0.566 | 0.318 | 0.357 | 0.184 | 0.375 | 0.583 | 0.282 | 0.409 | 0.65/0.31 |
| B4j (joint FireProt) | 0.672 | 0.688 | 0.618 | 0.669 | 0.479 | 0.494 | 0.350 | 0.624 | 0.714 | 0.437 | 0.479 | 0.74/0.43 |
| **B5** (stage 2) | **0.687** | **0.697** | **0.642** | **0.686** | 0.456 | 0.479 | 0.315 | 0.605 | 0.699 | 0.411 | 0.469 | 0.74/0.48 |
| B5-noenv | 0.678 | 0.686 | 0.634 | 0.676 | 0.445 | 0.468 | 0.304 | 0.591 | 0.690 | 0.390 | 0.474 | 0.73/0.45 |
| B5-phys | 0.607 | 0.614 | 0.561 | 0.619 | 0.398 | 0.451 | 0.244 | 0.490 | 0.594 | 0.291 | 0.396 | 0.66/0.37 |
| B5j | 0.681 | 0.695 | 0.626 | 0.674 | **0.484** | **0.502** | 0.342 | 0.619 | **0.716** | **0.440** | **0.481** | 0.75/0.43 |
| FoldX 5.1 | 0.629 | 0.640 | 0.569 | 0.447 | 0.411 | 0.449 | – | – | – | 0.430 | – | 0.68/0.33 |
| tmol-refit | 0.622 | 0.637 | 0.596 | 0.629 | 0.477 | 0.495 | – | – | – | – | – | 0.67/0.43 |
| KORPM | 0.500 | 0.513 | 0.482 | 0.524 | 0.397 | 0.408 | 0.194 | 0.581 | 0.566 | 0.302 | – | 0.53/0.37 |
| DDGun3D | 0.516 | 0.475 | 0.458 | 0.537 | 0.488 | 0.431 | 0.347 | 0.604 | 0.556 | 0.278 | – | 0.57/0.32 |

95% cluster-bootstrap CIs (MS natural ρ_prot): B5 0.687 [0.659, 0.713], B4 0.677 [0.654, 0.703], B3+ 0.648 [0.632, 0.670],
FoldX 0.629 [0.551, 0.697], tmol-refit 0.622 [0.590, 0.678]. Coverage 100% everywhere except S669 (2 rows), FoldX on
FireProt-test (98.3%) and Domainome (95.7%), DDGun3D on Domainome (9%, its own subset).

**Paired deltas on identical rows** (Megascale-test natural, per-protein median ρ / Pearson):
B4 − FoldX +0.048 [−0.008, +0.105] / +0.252; B4 − B3+ +0.029 [−0.002, +0.051] / +0.051; B4 − tmol-refit +0.055 [+0.016, +0.085];
B5 − B3+ +0.038 [+0.005, +0.062]; B5 − B4 +0.010 [−0.011, +0.015]. On FireProt-test no pair is significant
(B4 − FoldX Pearson +0.038 [−0.076, +0.105]; B4 − tmol-refit −0.017). On Domainome B4 − FoldX ρ_dom −0.024 [−0.048, +0.013].

## Gate verdicts
**G1 (does the physics earn its keep?) — partly.**
- B4 − B3 (the pre-registered 2-weight B3 = 0.564 natural) = **+0.113**, far above the 0.03 bar. B4 − B3+ (my
  10-weight simple model) = **+0.029 [−0.002, +0.051]**, just under the 0.03 bar I proposed and not significant;
  B5 − B3+ = **+0.038 [+0.005, +0.062]**, above it and significant.
- B4 ≥ FoldX + 0.08: **fails** (+0.048). ≥ +0.05 Pearson on FireProt-test: **fails** (+0.038, n.s.).
- Reading: the engine is a real but modest gain in ranking over my structure-statistics model, and a large gain in
  calibration (Megascale Pearson 0.681 vs 0.631 for B3+ and 0.447 for FoldX). It is clearly ahead of FoldX, KORPM,
  DDGun3D and tmol-refit on Megascale, and roughly tied with tmol-refit on FireProt.
- The force field is **not** mere decoration on a statistical potential, and the converse also holds: removing the
  env potential costs only 0.015 (B4 0.677 → B4-noenv 0.662), while removing all energy terms costs 0.037 (val) /
  0.041 (CV) and dropping to energies alone (B4-phys, no burial/env/rama/B3+ terms) costs 0.090. Both halves carry
  signal and they overlap heavily.

**G2 (neighbour repacking) — fails overall, passes on its target stratum.**
- B5 − B4 = +0.010 [−0.011, +0.015] per-protein ρ (needs ≥ 0.02); CV folds: stage 2 ahead in **2/5** (needs ≥ 70%).
- Buried small→large (2,528 rows): pooled ρ 0.447 → 0.495 (**+0.048**, needs ≥ 0.03); per-protein 0.437 → 0.495.
  Small→large overall 0.500 → 0.524. So the mechanism works exactly where it was predicted to, but the stratum is
  9% of rows and the overall effect is within noise.
- Runtime is within target (550 mutations/s/GPU, 1.9 s for a 63-residue scan). **Recommendation: ship stage 1 as the
  default, keep stage 2 as an option for buried small→large and for AF uniform repacking (G5).**

**G6 (transfer Megascale-only vs FireProt-only) — passes.** FireProt ΔΔG are ~1.7× Megascale's, so ratios are also
reported divided by 1.7. **No sign flips** in either B4 or B5. Scaled ratios are 0.6–1.8 for every energy term,
the env potential, Rama, burial and Occ×ΔV−. Exceptions: `dTS` is 0 in the FireProt fit (undetermined there, not a
flip), `ss_broken_nc` is 0 in Megascale (no disulfide rows exist there, as before), and `dref_apolar/polar` shrink to
~0 on FireProt. Reference energies correlate r = 0.70 (B4) / 0.75 (B5) between the two fits. Megascale δ_ref in the
joint fit exceeds 0.4 kcal/mol only for **Cys (−0.55)**, the same single assay artefact found for B3+.

## Term weights (kcal/mol per unit; ± cluster-bootstrap SE; sign-constrained where physics demands)
| term | B4 | B5 | B4-phys | reading |
|---|---|---|---|---|
| e_lj_atr | 0.075 ± 0.004 | 0.089 ± 0.005 | 0.145 | attraction heavily down-weighted vs unit |
| e_lj_rep (signed log1p) | 0.140 ± 0.026 | 0.143 ± 0.027 | 0.311 | per log-unit; see heavy tails |
| e_solv_apolar | 0.149 ± 0.016 | 0.104 ± 0.015 | 0.344 | |
| e_solv_polar | 0.112 ± 0.006 | 0.129 ± 0.005 | 0.159 | |
| e_hb_sc_bb | 0.249 ± 0.041 | 0.250 ± 0.039 | 0.259 | |
| e_hb_sc_sc | 0.098 ± 0.039 | 0.122 ± 0.037 | 0.108 | weakest H-bond channel |
| e_hb_bb_bb | 0.350 ± 0.016 | 0.340 ± 0.016 | 0.411 | |
| e_elec_dh | 0.284 ± 0.042 | 0.332 ± 0.048 | 0.307 | D10 screened form survives |
| dTS | −0.063 ± 0.016 | −0.075 ± 0.014 | −0.078 | correct sign (entropy stabilises) |
| dprior_avg | 0.102 ± 0.023 | 0.115 ± 0.020 | 0.110 | |
| d_env_cn_ss_bb | 0.262 ± 0.021 | 0.231 ± 0.019 | – | vs kT = 0.59: still ~2.5× over-confident |
| d_rama / d_rama_prev | 0.204 / 0.131 | 0.212 / 0.121 | – | |
| bur | 0.904 ± 0.066 | 0.932 ± 0.075 | – | |
| occ_dvol_loss | 1.205 ± 0.119 | 1.093 ± 0.128 | – | cavity term, unchanged from B3+ |
| gly_posphi | −0.533 ± 0.075 | −0.441 ± 0.077 | – | correction to d_rama (collinear; read jointly) |
| pro_nhhb | 0.000 (at bound) | 0.000 | – | **absorbed by e_hb_bb_bb + Δref(P)**: the engine already has the N–H loss |
| ss_broken_nc | 0.000 (no MS rows) | 0.000 | – | 3.3 ± 1.0 in the joint fit |

Every energy weight is positive and every bound is slack except `pro_nhhb` (unconstrained it would be −0.065, a
small wrong-sign value) and the two features with no support. Removing all sign constraints moves every other weight
by ≤ 0.022 kcal/mol and changes val/CV metrics by ≤ 0.0001, so **the constraints are essentially not binding** — the
data already prefers physical signs, which is the strongest evidence here that the terms mean what they claim. All
weights are well below 1, i.e. the raw unit-weight sum is systematically too steep; the entropy term is the only
negative one, as it should be.

## Heavy tails (point 2)
Unit-weight X→Pro reaches +136 kcal/mol in ΔE_lj_rep (median 26.6 for mut = P). Transforms compared by val + grouped CV:
signed log1p on the repulsion term only (**slog_rep**) won for both stages (val natural 0.7265 vs none 0.7192,
slog_all 0.7229, clip at 5 kcal/mol 0.7195, clip at 20 0.7241, per-term 0.5/99.5-percentile clip 0.7190). A separate
X↔Pro repulsion weight did not help (0.7231). Electrostatics variant: `dh` ≈ `bur` ≈ `dd` (within 0.001; dh kept per
D10). Boltzmann vs min-U at stage 1: min is +0.002 on val, −0.002 on CV, i.e. indistinguishable — D5's Boltzmann
averaging is kept on principle (it supplies dTS) but it is not what makes the engine work. Huber beat Gaussian on CV
(0.671 vs 0.667).

## Drop-one ablations (val natural / CV natural; full B4 = 0.7265 / 0.6709)
Largest losses: burial −0.058/−0.039, hb_sc_bb −0.021/−0.017, env potential −0.020/−0.008, Occ×ΔV− −0.017/−0.011,
lj_rep −0.015/−0.013, d_rama −0.013/−0.018, lj_atr −0.011/−0.007, elec −0.010/−0.008, solv_polar −0.009/−0.003,
reference energies −0.026/−0.023. Near-zero or slightly positive: hb_bb_bb, hb_sc_sc, solv_apolar, d_rama_prev,
dTS, dref_apolar/polar, and the three indicators with no Megascale support. **Candidates for deletion on this
evidence: solv_apolar and hb_bb_bb as free weights** (both are absorbed by burial and Δref), though they are
individually well determined and physically motivated — I would keep them and re-test after the D9 cap rerun.

## Antisymmetry (D15, point 3; Ssym, diagnostic only — 96.5% of its rows have a FireProt-trainval homologue)
| predictor | r(dir, −rev) | bias (kcal/mol) | Pearson direct | Pearson reverse | Pearson direct after ½[p_dir − p_rev] |
|---|---|---|---|---|---|
| raw stage-1 ΔF | 0.13 | +7.95 | 0.26 | −0.04 | 0.16 |
| raw stage-2 ΔF | 0.04 | +6.25 | 0.39 | 0.07 | 0.36 |
| B4 | 0.23 | +1.39 | 0.70 | 0.21 | 0.60 |
| B5 | 0.25 | +1.37 | 0.70 | 0.24 | 0.61 |
| B4-odd (no intercept, no WT-site-only features) | 0.59 | +0.44 | 0.62 | 0.41 | 0.57 |
| B5-odd | 0.57 | +0.38 | 0.62 | 0.45 | 0.60 |
| B4-odd + symmetric pair offset S(a,b) | 0.59 | +0.50 | 0.62 | 0.41 | 0.57 |
| B3+ | −0.22 | +1.82 | 0.59 | 0.00 | 0.48 |
| KORPM / DDGun3D | 0.88 / 0.99 | 0.15 / 0.04 | 0.57 / 0.56 | 0.49 / 0.53 | 0.55 / 0.55 |

**Where the asymmetry lives.** Per-term direct+reverse sums (`antisymmetry_terms.csv`) show the engine terms are
already nearly antisymmetric across the two structures: r(dir, −rev) is 0.97 for lj_atr, 0.98 for solv_apolar,
0.93 for solv_polar, 0.90 for elec, with mean sums of ±0.1–0.3 kcal/mol. The exception is **lj_rep: mean sum +7.5
(stage 1) / +5.9 (stage 2), r = 0.29**. The non-antisymmetry is essentially *one* term: each crystal structure is
relaxed around its resident side chain, so placing the other residue clashes in both directions. The fitted weights
shrink this (bias 8.0 → 1.4), and stage 2 removes about 20% of it by letting neighbours move.

**What I recommend.** Three options were tested:
(a) *Symmetric pair offsets S(a,b) = S(b,a)* fitted as part of Δref — **no help** (r 0.587 → 0.592). The bias is not a
function of the residue pair; it is a function of how tightly that particular structure is packed around the site.
(b) *Dropping WT-structure-only main effects* (intercept, `bur`, the indicators) — the "odd" models: antisymmetry
r jumps to 0.57–0.59 and bias falls to 0.4, at a cost of 0.04 per-protein ρ on Megascale (0.677 → 0.633). This is a
real trade and it is available at inference on a single structure.
(c) *Explicit symmetrisation* ½[ΔF(wt→mut) − ΔF(mut→wt)] — exactly antisymmetric by construction, but it **lowers**
direct Pearson for B4 (0.70 → 0.60) and needs the mutant structure, which does not exist at inference.
**Proposal:** keep B4/B5 as the headline, and treat antisymmetry as a property to earn through the repulsion term,
not through the loss. Concretely, the fix I would try first is the D9 site-dependent cap (below): the whole defect is
a fixed-backbone clash artefact in one term. The principled single-structure version of (c) is to score mut→wt on
the *modelled* mutant environment, which is a difference of the same state function; **this is a request to E1**
(see below). Until then I would report B4-odd as the antisymmetric variant and say plainly that our headline model
is not antisymmetric.

## Strata (point 5; `strata.csv`, `stabilising.csv`)
- Megascale-test pooled ρ, B4 / B5 / FoldX / B3+: natural 0.657/0.665/0.592/0.624, designed 0.568/0.571/0.483/0.538;
  buried 0.603/0.615/0.526/0.522, exposed 0.487/0.493/0.408/0.447; WT apolar 0.711/0.722/0.658/0.668, polar
  0.567/0.583/0.451/0.528, charged 0.464/0.458/0.430/0.427; to-Pro 0.602/0.598/0.530/0.566; from-Gly 0.463/0.480/0.404/0.454;
  large→small 0.734/0.735/0.659/0.709; small→large 0.500/0.524/0.511/0.474; **buried small→large 0.447/0.495/0.548/0.381**
  (FoldX is still the best there — its backbone relaxation shows).
- Domainome by pLDDT (per-domain ρ): ≥90 0.463/0.466/0.488, 70–90 0.304/0.306/0.321, 50–70 −0.025/−0.018/0.049,
  <50 −0.045/−0.036/−0.020. **Every method is at zero below pLDDT 70**, which supports the design's abstention flag
  rather than an attenuation factor.
- Stabilising mutations (ddg < −0.5, base rate 3.6% natural): AUROC B4 0.739, B5 0.738, tmol-refit 0.759, FoldX 0.720,
  B3+ 0.695. Precision@5 per protein: FoldX 0.219, tmol-refit 0.181, B5 0.152, B4 0.133. **We rank stabilising
  mutations worse than FoldX at the top of the list** even though our AUROC is higher — a calibration problem in the
  stabilising tail, worth a look before any design application.
- AF vs crystal (G5): B4/B5 predictions agree between AF and xtal structures at Pearson 0.956/0.955 (mean |Δ| 0.14
  kcal/mol) and have the same accuracy on both (ρ vs ΔΔG 0.591 AF vs 0.576 xtal). Raw ΔF agrees much less
  (0.82/0.73, mean |Δ| 3.2/2.7 kcal/mol) because the repulsion term is structure-sensitive — same root cause as the
  antisymmetry defect.

## Requests to E1 (feature reruns)
1. **D9 site-dependent cap** `cap = c0 + c1·(1 − burial)`, say three settings (c0 = 1, 3, 10 with c1 = 0 and
   c0 = 1, c1 = 4). The repulsion term is both the heavy-tail problem (needs a log transform to be usable) and the
   entire antisymmetry defect; a softer, flexibility-aware cap is the principled fix and would let me drop the
   transform. Highest value of anything on this list.
2. **Reverse scoring on the modelled mutant environment**: for a variant wt→mut, also return ΔF(mut→wt) computed on
   the mutant state built by the same pipeline (i.e. re-run the site scan with the mutant as the resident residue).
   That makes the antisymmetrised score available from one input structure.
3. Minor: `disulf` has no variance on Megascale and effectively none elsewhere; it is dropped from the model
   (the CXXC-aware indicator covers it). No rerun needed.

## Suspicious / caveats
1. `pro_nhhb` sits at its bound (0) in every engine model but was 0.85 kcal/mol in B3+ — the engine's backbone H-bond
   term already accounts for the lost N–H, so this is a consistency check passing, not a lost feature.
2. The joint models (B4j/B5j) are better on every low-throughput set and on Domainome but slightly worse on
   Megascale-test; the thioredoxin CXXC exception (added here) removed the pathology seen for B3+j, and `ss_broken_nc`
   is now 3.3 ± 1.0 kcal/mol, a sane disulfide penalty.
3. S669-strict is the one set where DDGun3D (0.347) still beats us (B4 0.328). S669 is also where our per-protein
   numbers are noisiest (many proteins with few variants).
4. Megascale-test shows no strict/full gap (0.677 vs 0.687 natural/natural-strict), so G7 is not triggered.
5. Stage 2's `nb_native_frac_wt` is only 0.49, i.e. the mean-field packer often does not put neighbours back in their
   native rotamer. That, not the idea of repacking, may be why G2 misses.
