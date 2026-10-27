# PhysDDG — final model card (D22, 2026-10-27)

**What it is.** A fixed-backbone, GPU physics engine for single-point ΔΔG with a small linear read-out. No neural
network, no learned embeddings. `ddg = intercept + Σ_k w_k·x_k + r[mut] − r[wt]`, positive = destabilising (D1).

| ship | file | form | when to use |
|---|---|---|---|
| **default** | `models/final_stage2.json` (B6gj) | Stage 2: site + ≤16 flexible neighbours, batched mean field | anything, crystal or AF |
| **fast** | `models/final_stage1.json` (B6gjs1) | Stage 1: Boltzmann over site rotamers in the fixed WT environment | large scans; ~3.5× faster, −0.016 on Megascale |
| **AF mode** | `models/final_af.json` (B6gjaf) | Stage 2 + pLDDT attenuation | only where B-factors are pLDDT |
| reference | `models/ref_stage{1,2}.json`, `ref_af.json` (B6g*) | same form, **Megascale-only** fit | ablation / comparison |

**Parameters.** 20 fitted weights + 20 reference energies = **40 numbers fitted to ΔΔG** (21 weights for the AF
model, which adds the attenuation coefficient). On top of that the model uses tables *estimated from structures,
never from ΔΔG*: the backbone-dependent rotamer library, the Ramachandran maps, the environment potential
(~180 entries), the N-cap potential (20) and ~15 frozen CHARMM19/EEF1 constants with literature provenance
(`physddg/params.py`). Quote it as "40 parameters fitted to ΔΔG plus PDB-derived statistical tables" — not
"40 parameters".

**Terms** (B6gj; kcal/mol per unit of the feature, ± cluster-bootstrap SE; all signs are the physically expected
ones and the sign constraints are not binding — removing them moves weights by ≤0.02):
| term | w | meaning |
|---|---|---|
| `e_lj_atr` | +0.083 ± 0.008 | LJ attraction lost on mutation (per kcal/mol of raw ΔE) |
| `e_lj_rep` | +0.178 ± 0.025 | steric clash, signed-log transformed (raw ΔE reaches +136 kcal/mol for X→Pro) |
| `e_solv_apolar` | +0.141 ± 0.035 | EEF1 apolar desolvation (hydrophobic burial) |
| `e_solv_polar` | +0.114 ± 0.009 | EEF1 polar desolvation (cost of burying N/O) |
| `e_hb_sc_bb` / `e_hb_sc_sc` / `e_hb_bb_bb` | +0.219 / +0.158 / +0.272 | H-bonds lost, by channel |
| `e_elec_dh` | +0.365 ± 0.066 | Debye–Hückel screened electrostatics (0.15 M) |
| `dTS` | −0.097 ± 0.019 | side-chain conformational entropy (Boltzmann, negative = stabilises) |
| `dprior_avg` | +0.084 ± 0.021 | rotamer prior −kT ln p(rot\|φ,ψ) |
| `d_env_cn_ss_bb` | +0.296 ± 0.030 | PDB environment potential, aa \| contact number × SS × φ/ψ basin (per kT) |
| `d_rama` / `d_rama_prev` | +0.150 / +0.150 | Ramachandran at the site / pre-Pro at i−1 |
| `bur` | +0.825 ± 0.093 | burial main effect, 1 − RSA |
| `occ_dvol_loss` | +1.052 ± 0.133 | cavity creation: volume lost × occupancy |
| `gly_posphi` | −0.590 ± 0.090 | G→X at φ>0 — a *correction* to `d_rama`, read the two jointly |
| `ss_broken` | +3.338 ± 1.043 | disulfide broken (fitted on 8 FireProt-train rows; undetermined on Megascale) |
| `ncap_pref` | +0.273 ± 0.038 | N-cap preference, PDB-derived potential (per kT) |
| `helix_dipole` | +0.080 ± 0.019 | charge change at a helix terminus (per elementary charge) |
| `pro_nhhb` | 0.000 (at bound) | X→Pro with an H-bonded backbone N–H: already covered by `e_hb_bb_bb` + Δref(P) |
| intercept | −0.212 | |
Plus 20 reference energies r(aa) = EEF1 prior + shared δ (D7).

**Training data.** Megascale-train (215,731 point rows + 11,363 censored intervals, 239 domains, 96 clusters) and
FireProt-train (2,669 rows, 57 proteins, 45 clusters), weighted by homology clusters, not rows (68%/32%). Huber
loss with interval (Tobit) terms for censored rows; per-assay scale/offset and a Megascale δ_ref nuisance, all
**dropped at inference**. Every hyper-parameter was chosen on Megascale-val + grouped 5-fold CV on train.

**Held out entirely:** Megascale-test, FireProt-test, S669, S461, Ssym, S2648, Domainome, abundance_dms,
paired xtal/AF. Domainome is validation for the AF/abundance regime and is **never trained on**.

**Results** (identical rows per column; strict = no homologue of any training set that model saw; 95%
cluster-bootstrap CIs in `multiassay_summary.csv`; Megascale/Domainome/abundance = per-protein median ρ,
FireProt/S669/S461 = Pearson):
| model | MS nat | MS nat-strict | MS all | FireProt | S669-str | S461-str | Domainome | Dom flag-kept | abundance |
|---|---|---|---|---|---|---|---|---|---|
| **B6gj** (default) | 0.692 | 0.695 | 0.682 | 0.482 | 0.344 | 0.568 | 0.451 | 0.478 | 0.482 |
| **B6gjs1** (fast) | 0.676 | 0.690 | 0.674 | 0.475 | **0.351** | **0.579** | 0.450 | 0.477 | 0.478 |
| **B6gjaf** (AF) | **0.700** | 0.703 | 0.692 | 0.481 | 0.343 | 0.568 | **0.459** | **0.481** | **0.491** |
| B6g (Megascale-only) | 0.692 | 0.702 | 0.692 | 0.464 | 0.336 | 0.552 | 0.418 | 0.449 | 0.469 |
| B5 (previous rung) | 0.687 | 0.697 | 0.685 | 0.467 | 0.334 | 0.559 | 0.412 | 0.440 | 0.469 |
| FoldX 5.1 raw / clipped | 0.629 | 0.640 | 0.633 | 0.417 / 0.423 | – | – | 0.430 | 0.458 | – |
| tmol-refit / KORPM / DDGun3D | 0.622 / 0.500 / 0.516 | | | 0.486 / 0.411 / 0.529 | – / 0.194 / 0.347 | – / 0.543 / 0.552 | – / 0.302 / 0.282 | | |

Headline comparison, Megascale-test natural (21 proteins, 22,159 variants): **B6gj − FoldX-clipped = +0.063
[+0.008, +0.095]** per-protein ρ; Pearson 0.694 vs 0.596 and RMSE 0.79 vs 1.85 kcal/mol. B6gj beats FoldX on 19/21
proteins. Mean per-protein ρ 0.688 vs median 0.692. **Against the previous rung the gains are assay-specific:**
Domainome +0.047 [+0.025, +0.058] and abundance +0.021 [+0.008, +0.027] (both significant), Megascale +0.005
[−0.018, +0.017] and FireProt +0.014 [−0.017, +0.033] (both not).

**Known weaknesses** — none of these should be glossed in a report:
1. **Not antisymmetric.** Ssym r(dir, −rev) 0.23–0.25, bias +1.4 kcal/mol, against KORPM 0.88 and DDGun3D 0.99.
   The defect is almost entirely the repulsion term on a fixed backbone; no cap or transform we tried fixes it.
2. **Stabilising mutations:** precision@5 0.15–0.23 against FoldX 0.219 and tmol-refit 0.181; only the AF variant
   edges FoldX, and only on AF input. AUROC is better (0.74 vs 0.72), the top of the ranked list is not.
3. **Below pLDDT 70 every method is at ρ ≈ 0** on Domainome (50–70 band: −0.01). Flag and withhold rather than
   predict; flagging 14% of variants is worth more (+0.03) than any correction we fitted.
4. **Buried small→large:** 0.49 against FoldX 0.55. Fixed backbone cannot distinguish a relaxable clash from a
   real one; this is the one stratum where FoldX still wins.
5. **S669-strict:** 0.344 against DDGun3D 0.347 — parity, not a win, on the hardest crystal set.
6. **Megascale is AF2 models.** We are fitted on Tsuboyama's AF2 structures, and so is the FoldX baseline
   (`notes/foldx_protocol.md`). Structure source is confounded with assay; the crystal sets are where the energy
   function is tested on crystals, and there we are at parity with a 10-parameter descriptor baseline (B3+).
7. **Selection optimism:** the B6 term set was chosen after one test look by the residual analysis (D19.4), so
   B5 → B6g/B6gj test deltas on Megascale carry mild optimism. The val/CV deltas are the cleaner evidence.
8. Megascale-test natural is **21 proteins** and is not homology-clean (1TUC/2KXD share mutations with train;
   the strict subset is not lower). FireProt per-protein ρ is a median over **9** proteins.

**When to use which.** Crystal structure, accuracy first → `final_stage2`. Saturation scans / proteome scale →
`final_stage1`. AlphaFold model → `final_af`, with the pLDDT<70-or-terminal flag rule, and crop to the assayed
construct. Multi-point mutants, ligands/metals, interfaces and pH dependence are **out of scope** (see §Doubles
and the flags in `physddg/predict.py`).

---

## Multi-assay headline: B6gj, and what the Megascale Δref is doing (critique #2 I1/A1/A2, fifth issue)

`scripts/55_multiassay_report.py` → `multiassay_{summary,deltas}.csv`. **Strict subsets are now per model**
(critique #2 3.4): a model fitted with FireProt-trainval must also exclude FireProt-trainval homologues, so
S669-strict is n = 488 for Megascale-only models and n = 470 for joint ones, S461-strict 290 vs 285. On
Megascale-test the two rules coincide (the 5 test proteins with FireProt-trainval homologues are a subset of the
9 with Megascale-train homologues), and Domainome strict drops 347 → 339 domains for joint models.

### All models, all assays (identical rows within each column, per-model strict)
| model | MS nat | MS nat-strict | MS all | FireProt | S669-str | S461-str | Domainome | Dom flag-kept | abundance |
|---|---|---|---|---|---|---|---|---|---|
| B3+ | 0.648 | 0.649 | 0.649 | 0.456 | 0.339 | 0.570 | 0.389 | 0.407 | 0.446 |
| B5 | 0.687 | 0.697 | 0.685 | 0.467 | 0.334 | 0.559 | 0.412 | 0.440 | 0.469 |
| B6g | 0.692 | 0.702 | 0.692 | 0.464 | 0.336 | 0.552 | 0.418 | 0.449 | 0.469 |
| B6gs1 | 0.678 | 0.690 | 0.684 | 0.455 | 0.346 | 0.572 | 0.414 | 0.450 | 0.468 |
| B6gaf | 0.699 | 0.704 | 0.699 | 0.464 | 0.336 | 0.552 | 0.425 | 0.454 | 0.478 |
| **B6gj** (joint) | 0.692 | 0.695 | 0.682 | **0.482** | 0.344 | 0.568 | **0.451** | **0.478** | **0.482** |
| **B6gjs1** (joint, stage 1) | 0.676 | 0.690 | 0.674 | 0.475 | **0.351** | **0.579** | 0.450 | 0.477 | 0.478 |
| B6g-refzero (Δref := 0) | 0.687 | 0.671 | 0.660 | 0.443 | 0.308 | 0.519 | 0.439 | 0.466 | 0.467 |
| B6g-noref (refit, no Δref) | 0.684 | 0.675 | 0.670 | 0.440 | 0.313 | 0.517 | 0.412 | 0.442 | 0.471 |
| B6x (+site_scsc) | 0.701 | 0.705 | 0.695 | 0.439 | 0.313 | 0.548 | 0.437 | 0.471 | 0.472 |
| FoldX raw / clipped | 0.629 | 0.640 | 0.633 | 0.417 / 0.423 | – | – | 0.430 | 0.458 | – |
| tmol-refit | 0.622 | 0.637 | 0.626 | 0.486 | – | – | – | – | – |
| KORPM | 0.500 | 0.513 | 0.533 | 0.411 | 0.194 | 0.543 | 0.302 | 0.315 | – |
| DDGun3D | 0.516 | 0.475 | 0.502 | 0.529 | 0.347 | 0.552 | 0.282 | 0.296 | – |
(Megascale/Domainome/abundance = per-protein median ρ; FireProt/S669/S461 = Pearson.)

### A1 — is the Megascale Δref an assay artefact? **Partly: gate passes, but the story is Domainome-specific.**
Scoring B6g with its reference energies **zeroed at inference** (same weights, no FireProt data anywhere):
- Domainome **0.418 → 0.439**, paired +0.020 [+0.009, +0.032]; flag-kept 0.449 → 0.466, +0.017 [+0.001, +0.024].
  The B5 → B6gj Domainome gain is +0.039, so zeroing Δref recovers **51%** of it — the pre-registered gate
  ("recovers ≥ half") **passes**, and it passes without using FireProt at all.
- But it **costs everywhere else**: FireProt 0.464 → 0.443, S669-strict 0.336 → 0.308 (−0.028 [−0.056, −0.002]),
  S461-strict 0.552 → 0.519, Megascale natural-strict 0.702 → 0.671, Megascale-all 0.692 → 0.660.
- Refitting without Δref (`B6g-noref`) recovers nothing on Domainome (0.412): other weights absorb the reference
  energies, so the diagnostic only works by dropping them *at inference*.

**Reading.** The Megascale reference energies are not globally wrong — they help on Megascale and on both crystal
sets. What they contain is a component that is *actively wrong for Domainome*, consistent with the earlier finding
that Megascale Δref transfers to buried sites (per-aa ρ 0.84) but not to abundance-assay surfaces (ρ −0.09).
That is an assay-specific unfolded-state/degron mismatch, not "the PSSM artefact" in general. Critique #1's
prediction is therefore half-confirmed: the vector is contaminated, but removing it wholesale is not the fix —
re-estimating it jointly is (below). The Megascale δ_ref nuisance in the joint fits stays large only for **Cys**
(−0.55 kcal/mol; next largest R +0.19), so the contamination is concentrated, not diffuse.

### A2 — the joint model B6gj. **Gate narrowly fails on its letter, passes on its intent.**
B6gj = B6g fitted on Megascale-train + FireProt-train, cluster-balanced, per-assay scale/offset, with the
Megascale δ_ref nuisance dropped at inference (D7).
| comparison | Megascale nat | Megascale all | FireProt | S669-str | S461-str | Domainome | abundance |
|---|---|---|---|---|---|---|---|
| B6gj − B5 | +0.005 [−0.018, +0.017] | −0.003 [−0.043, +0.012] | +0.015 [−0.016, +0.033] | +0.029 [−0.017, +0.100] | +0.012 [−0.040, +0.111] | **+0.039 [+0.019, +0.052]** | **+0.013 [+0.000, +0.013]** |
| B6gj − B6g | — | — | +0.018 | +0.027 | +0.019 | **+0.032 [+0.015, +0.044]** | +0.013 |
| B6gj − FoldX-clip | +0.063 | +0.049 | +0.059 | – | – | +0.021 [−0.002, +0.047] | – |
- **Megascale cost ≈ 0**: +0.005 on the primary subset, −0.003 pooled, −0.007 natural-strict. Well inside the
  "≤ 0.01 loss" condition.
- **Domainome vs FoldX: +0.021 [−0.002, +0.047]** — the CI touches zero, so the literal gate
  ("≥ FoldX + 0.01 with CI excluding 0") **fails**. Against B5 and B6g the Domainome gain *is* significant.
- B6gj is the only configuration that is ≥ B5 on **every** assay simultaneously, and it is the best structure-only
  model on FireProt (0.482), Domainome (0.451), flag-kept Domainome (0.478) and abundance_dms (0.482).
- Stage-1 joint (`B6gjs1`) is the best model on both crystal sets (S669-strict 0.351, S461-strict 0.579) —
  better than DDGun3D (0.347/0.552) and B3+ (0.339/0.570) — at a Megascale cost of 0.016.

**Recommendation.** Make **B6gj the reported multi-assay model** and keep B6g as the Megascale-only reference fit,
stating plainly that (a) the Domainome advantage over FoldX is +0.021 with a CI that touches zero, and (b) the
joint model's S669/S461/Ssym numbers are *not* independent tests, because FireProt-trainval overlaps those sets
(96.5% of Ssym rows, and the strict columns above already remove the overlap for S669/S461). Nothing here changes
the Megascale-vs-FoldX claim, which is the same for B6g and B6gj.

**Not done, deliberately:** no Domainome-derived term was added to the joint fit. Domainome is the validation set
for the AF/abundance regime; training on it would make every Domainome number self-referential. The critique asked
for it "only if leak-free", and it is not.

### Reporting upgrades (critique #2 3.3)
`multiassay_summary.csv` and `main_table.md` now carry, for every cell: n variants, n proteins, 95%
cluster-bootstrap CI, **mean next to median** per-protein ρ, and the **per-protein win count against FoldX** on
Megascale-test natural (21 proteins): B6gaf 20/21, B6g 19/21, B6gj 19/21, B5 18/21, B6x 18/21, B3+ 16/21,
tmol-refit 15/21, KORPM 3/21, DDGun3D 0/21. Mean and median agree for the Megascale-only models (B6g 0.692
median / 0.702 mean) but diverge for the joint ones (B6gj 0.692 / 0.688) and for the Δref diagnostics
(B6g-refzero 0.687 / 0.672), i.e. zeroing Δref hurts the weaker proteins most. The exact FoldX protocol,
its unrecorded options and the AF2-structure fairness caveat are in **`notes/foldx_protocol.md`**.

---

## Doubles: how much is there for a pairwise term? (critique #2 I6a; `scripts/54_doubles_additivity.py`)

127,476 Megascale Dataset-3 doubles carry both single values, so ε = ΔΔG(AB) − ΔΔG(A) − ΔΔG(B) is measurable
without any model. **Raw, it looks enormous and distance-independent**: mean ε = −0.88 kcal/mol, |ε| > 0.5 for
64% of doubles, and the mean is just as negative at CB–CB > 15 Å (−0.76) as at < 5 Å (−0.78). A distance-blind
negative offset is not coupling — it is the assay floor: the double is more destabilising, so it saturates and
the measured ΔΔG falls short of the additive sum.

Restricting to doubles that cannot be floor-limited (sum of singles < 1.5 kcal/mol, dG_mut > 0.5, not
`flag_mut_floor`; 28,123 rows) removes most of it:
| CB–CB | n | mean ε | sd ε | \|ε\| > 0.5 |
|---|---|---|---|---|
| all (floor-safe) | 28,123 | −0.225 | 0.498 | 31% |
| < 5 Å | 6,193 | −0.093 | 0.495 | 26% |
| 5–6.5 Å | 12,601 | −0.284 | 0.512 | 35% |
| 6.5–8 Å | 2,946 | −0.321 | 0.529 | 41% |
| 8–10 Å | 2,696 | −0.234 | 0.414 | 26% |
| 10–12 Å | 1,715 | −0.108 | 0.398 | 23% |
| > 15 Å | 921 | −0.263 | 0.461 | 19% |

**Reading.** After the floor control, genuine non-additivity is small (sd 0.50 kcal/mol, part of which is
measurement noise) and only weakly distance-dependent — and not monotonically so: the contact bin (< 5 Å) has the
*smallest* mean ε, the 5–8 Å shell the largest. There is no clean "close pairs are coupled" signal to exploit.
**Upper bound on what a pairwise term could buy: ≲0.5 kcal/mol rms on doubles, and nothing identifiable on
singles.** Combined with the fact that singles are what every benchmark scores, I do not recommend spending
parameters on explicit coupling terms.

**What is missing to test our own coupling.** This probe uses measured ε only. Scoring the engine's predicted
coupling needs a double-mutant scan from E1's pipeline: both sites mutated in the same Stage-2 mean-field
problem, with the two sites in each other's flexible set, returning ΔF(AB) alongside ΔF(A) and ΔF(B) on the same
structure. Stage 2 already couples a site to its neighbours, so the machinery exists; what is needed is a
`mutate.pair_scan(site_i, site_j)` entry point and a feature file with `s2_dF_AB`. Given the bound above I would
only build it if multi-mutant prediction becomes a product requirement — which is a capability argument, not an
accuracy one.

## Floor-aware loss (critique #2 I7ii)

4,010 Megascale-train rows (1.8%) carry `flag_mut_floor` (dG_mut < −0.5, i.e. the mutant is at the per-domain
measurement floor) and the residual analysis found they hold 6.6% of the squared error. Refitting B6g with those
rows as **lower bounds** (Tobit, the same interval machinery already used for the excluded dG < −1 rows):
val natural ρ_prot 0.7411 → 0.7412, grouped-CV natural 0.6813 → **0.6718**, and no weight moves by more than
**3.8%**. The critique's own diagnostic was "if weights move > 20%, the current fit was floor-biased" — they move
4%, so **the fit is not floor-biased and the floor-aware loss does not help ranking** (it slightly hurts CV,
presumably by discarding real information in those rows). Reported as a negative result; `B6g-floor.json` is kept
for the record. The separate point stands and is already implemented: the 11,363 rows Tsuboyama *excluded*
(dG < −1) are used as intervals in every fit.

---

## D19 final models: B6g / B6gs1 / B6gaf (fourth issue)

**Decision D19.** The shipped default must generalise across assays, so it drops both WT-site main effects:
`site_hbsc_bur` (fails G6, scaled ratio 0.25) and `site_scsc` (hurts FireProt/S669/S461, G6 edge at 0.35).
**B6g = B5 + `ncap_pref` + `helix_dipole`** (stage 2), **B6gs1** the stage-1 analogue (default fast mode),
**B6gaf** = B6g + AF-only pLDDT attenuation. `B6x` = B6g + `site_scsc` is kept in the tables for transparency.
Both B6g terms are state-function differences (D15-compliant).

> **Selection-optimism caveat (D19.4).** The B6 term set was chosen *after* the residual analysis had taken one
> look at the test sets (`notes/residual_analysis.md` §4). Selection itself used only train CV and val, but the
> decision to pursue these terms at all was informed by that look. **Read B5 → B6g/B6x test deltas as mildly
> optimistic.** The val/CV deltas below are the cleaner evidence, and they are smaller than the test deltas for
> `site_scsc` — which is part of why D19 drops it.

### Selection (`final_selection_ablation.csv`; Megascale-val natural ρ_prot / grouped-CV natural ρ_prot)
| model | val | CV | test natural (reported, not used for any choice) |
|---|---|---|---|
| B4 (stage-1 base) | 0.7265 | 0.6709 | 0.6768 |
| **B6gs1** | 0.7262 | 0.6758 | 0.6784 |
| B6xs1 (+site_scsc) | 0.7326 | 0.6896 | 0.6903 |
| B5 (stage-2 base) | 0.7329 | 0.6727 | 0.6866 |
| **B6g** | 0.7411 | 0.6813 | 0.6919 |
| B6x (+site_scsc) | 0.7434 | 0.6944 | 0.7005 |
| 4-term set − site_scsc (i.e. +hbsc, ncap, dipole) | 0.7391 | 0.6909 | 0.6934 |
| **B6gaf** (B6g + pLDDT attenuation) | **0.7439** | **0.6900** | 0.6988 |
| B6g + burial attenuation (control, constrained) | 0.7411 | 0.6813 | 0.6919 |

The PI's quoted "val 0.739 / CV 0.691" is the **4-term-minus-scsc** row (it still contains `site_hbsc_bur`).
B6g proper — N-cap and dipole only — is **val 0.7411 / CV 0.6813**: better on val, lower on CV, because
`site_hbsc_bur` is a CV-only term (that is exactly how it was described in the residual analysis, and why G6
could reject it without much loss). Confirmed.

**Attenuation control survives without site_scsc.** Sign-constrained, the burial version is pinned at 0 and
changes nothing; refitted sign-free it takes **+0.82** — it wants to amplify buried predictions, not attenuate —
while the pLDDT weight stays at −0.97. So B6gaf is kept: it is not the burial proxy that `notes/af_mode.md` §4
rejected. Its gain is small and its weight CI includes 0.

### Final results (D17 protocol: identical rows, strict-only S461/S669, FoldX raw and clipped)
| model | MS nat ρ_prot | MS nat-strict | MS all ρ_prot | MS all r | MS RMSE | FP r (338) | S669-str | S461-str | Dom ρ_dom | Dom flag-kept | Abund ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.687 | 0.697 | 0.685 | 0.686 | 0.776 | 0.467 | 0.315 | 0.556 | 0.412 | 0.440 | 0.469 |
| **B6gs1** (fast) | 0.678 | 0.690 | 0.684 | 0.681 | 0.780 | 0.455 | 0.328 | 0.569 | 0.414 | — | 0.468 |
| **B6g** (default) | 0.692 | 0.702 | 0.692 | 0.687 | 0.776 | 0.464 | 0.316 | 0.549 | 0.418 | 0.449 | 0.469 |
| **B6gaf** (AF only) | 0.699 | 0.704 | 0.699 | 0.690 | 0.768 | 0.464 | 0.316 | 0.549 | 0.425 | 0.454 | 0.478 |
| B6x (+site_scsc) | 0.701 | 0.705 | 0.695 | 0.692 | 0.770 | 0.439 | 0.291 | 0.545 | 0.437 | 0.471 | 0.472 |
| FoldX raw / clipped | 0.629 | 0.640 | 0.633 | 0.447 / 0.590 | 3.17 / 1.84 | 0.417 / 0.423 | – | – | 0.430 | 0.458 | – |
| tmol-refit | 0.622 | 0.637 | 0.626 | 0.629 | 0.822 | 0.486 | – | – | – | – | – |
FireProt on all 347 rows: B5 0.456, B6g 0.451, B6gs1 0.444, B6x 0.430, FoldX 0.411, tmol-refit 0.477.

**Paired bootstrap** (identical rows, 95% cluster CI):
| comparison | MS natural ρ_prot | MS nat-strict | MS all | Pearson (natural) |
|---|---|---|---|---|
| B6g − B5 | +0.005 [−0.004, +0.019] | +0.004 [−0.007, +0.009] | +0.006 [−0.005, +0.019] | +0.001 [−0.002, +0.003] |
| B6gaf − B5 | +0.012 [+0.000, +0.027] | +0.006 [−0.001, +0.012] | +0.013 [+0.001, +0.026] | +0.003 [−0.006, +0.013] |
| B6gs1 − B5 | −0.008 [−0.011, +0.006] | −0.007 [−0.027, +0.006] | −0.001 [−0.020, +0.010] | −0.004 [−0.009, +0.002] |
| B6g − FoldX | +0.063 [+0.008, +0.120] | +0.062 [+0.009, +0.125] | +0.058 [+0.017, +0.121] | +0.256 [+0.200, +0.300] |
| B6g − FoldX-clip | +0.063 [+0.008, +0.120] | +0.062 [+0.009, +0.125] | +0.058 [+0.016, +0.121] | **+0.105 [+0.068, +0.135]** |
| Domainome B6g − B5 | +0.007 [−0.000, +0.014] (flag-kept +0.011 [+0.000, +0.017]) | | | |
| Domainome B6g − FoldX | −0.012 [−0.038, +0.019] (flag-kept −0.009 [−0.028, +0.015]) | | | |
| FireProt B6g − B5 | Pearson −0.005 [−0.011, −0.000] | | | |

**Honest reading.** Removing `site_scsc` removes most of the Megascale/Domainome test gain: B6g − B5 is
+0.005 [−0.004, +0.019] on the primary subset, i.e. **not distinguishable from B5**. What D19 buys is that B6g
no longer *loses* on the crystal sets (FireProt 0.464 vs B6x 0.439; S669-strict 0.316 vs 0.291) while keeping a
small, well-transferring improvement. B6gaf is the only variant with a CI excluding 0 against B5 (+0.012), and
only on AF-type inputs. **Against FoldX the margin is unchanged and significant on all three Megascale subsets**
(+0.058 to +0.063 rank, +0.105 Pearson over clipped FoldX, RMSE 0.776 vs 1.84), which is the claim that matters.
Note B6gs1 is *below* B5 on Megascale natural (−0.008): at stage 1 the two new terms do not pay for themselves,
so the fast mode is "B4 + N-cap + dipole", not a free upgrade.

### What contributes (drop-one-group from B6g, stage 2; val / CV / test natural ρ_prot)
| dropped | Δval | ΔCV | Δtest |
|---|---|---|---|
| B3+ remainder (burial, cavity, Pro/Gly, SS-broken) | −0.080 | −0.047 | −0.036 |
| all engine energies (8 terms) | −0.040 | −0.032 | −0.038 |
| reference energies (Δref, 20) | −0.027 | −0.022 | −0.008 |
| vdW (lj_atr + lj_rep) | −0.024 | −0.011 | −0.015 |
| solvation (apolar + polar) | −0.024 | +0.003 | −0.006 |
| H-bonds (sc-bb, sc-sc, bb-bb) | −0.022 | −0.009 | −0.008 |
| env statistical potential | −0.018 | −0.006 | −0.006 |
| Ramachandran (site + pre-Pro) | −0.015 | −0.010 | −0.009 |
| electrostatics | −0.011 | −0.007 | +0.003 |
| side-chain entropy + rotamer prior | −0.006 | −0.005 | −0.007 |
| N-cap | −0.008 | −0.009 | −0.004 |
| helix dipole | +0.001 | −0.001 | −0.000 |
| stage 2 → stage 1 (B6g → B6gs1) | −0.015 | −0.006 | −0.014 |

Reading: the structure-derived scalar terms (burial, cavity, Pro/Gly indicators) and the engine energies are each
worth 3–8 points of per-protein ρ, the reference energies 2–3, and every individual energy group 1–2. The two new
D19 terms are worth ~1 point (N-cap) and ~0 (dipole, kept because it is physically standard, free and transfers).
No single group dominates: this is a model where ~6 groups each contribute a little.

### G6 transfer of the final term set (`transfer_B6g.csv`, `transfer_B6gs1.csv`)
**All 20 B6g weights pass**: no sign flips (in constrained or sign-free refits), every scaled ratio in 0.67–1.87,
reference energies correlate r = 0.75. `ncap_pref` 0.294 ± 0.037 vs 0.465 ± 0.120 (scaled ratio 0.93) and
`helix_dipole` 0.086 ± 0.021 vs 0.144 ± 0.070 (0.98) are among the cleanest terms in the model. Undetermined, as
before: `ss_broken` (no Megascale rows) and `pro_nhhb` (0 in both). In the **stage-1** fit two terms fail the 3×
rule, both because they are undetermined on FireProt rather than contradicted: `dTS` (−0.066 ± 0.018 vs
0.000 ± 0.046; sign-free FireProt +0.052) and `dref_apolar/polar` (0.070 ± 0.006 vs 0.008 ± 0.061). They are kept
in B6gs1 because they are well determined on Megascale and physically required, and the failure is a FireProt
power problem (2,669 rows); this is disclosed rather than acted on.

### Strata (Megascale-test per-protein ρ) and flags
B6g vs B5 gains on exposed sites (0.554 → 0.565), charged WT (0.516 → 0.533), from-Gly (0.639 → 0.645) and
large→small (0.795 → 0.802); unchanged on buried and buried small→large (0.495 → 0.487, still the fixed-backbone
problem). B6gaf additionally gains on to-Pro (0.659 → 0.679) and at pLDDT < 50 (0.356 → 0.416). Stabilising
mutations: AUROC 0.738 → 0.746 (B6g) and precision@5 0.152 → 0.190 (B6g) / 0.229 (B6gaf), the first variant to
beat FoldX (0.219) on that metric. Domainome by pLDDT is unchanged in shape: ≥90 0.474, 70–90 0.311, 50–70
−0.013, <50 −0.050 — still useless below 70, so AF reports should use the flag-kept subset.

### Shipped files
`results/ladder_physics/models/final_stage2.json` (B6g, default), `final_stage1.json` (B6gs1, fast mode),
`final_af.json` (B6gaf, AF inputs only); `README.md` documents the schema, the feature contract and the
two-stage attenuation rule (`scale_kind` / `scale_base` / `scale_plddt`) that `physddg/predict.py` must branch on
(also posted to `notes/engine_mutate.md` as R-fit3). The superseded `B6*.json` files were removed; `B6x.json`
remains for transparency.

---

## B6: residual-analysis terms (2026-10-26, third issue)

> **Superseded by D19** (section above): this section describes the 4-term B6 set, of which the shipped models
> keep only `ncap_pref` and `helix_dipole`. The feature audit, the N-cap replacement and the G6 analysis below
> still stand; the headline numbers are those of `B6x`.

Rung B6 adds the terms that survived the residual analysis (`notes/residual_analysis.md`) to B5 (stage 2) and B4
(stage 1 → `B6s1`). Everything was re-selected on Megascale-val + grouped CV on train after the feature audit below;
no test number entered any choice. `B6af` adds the AF-mode pLDDT attenuation and is **only meaningful where
B-factors are pLDDT** (Megascale's AF2 models, Domainome, AF mode) — on crystal inputs the term is identically 0.

### Feature audit (the analyst's scales were typed from memory)
- **N-cap preference — replaced.** The hand-typed "Doig–Baldwin-type" table could not be verified: PMC/Europe PMC
  would not serve the paper's tables, and its ranking contradicted the abstract (it put Gln mid-scale; Doig &
  Baldwin 1995, Protein Sci 4:1325, report Gln **worst**). I replaced it with a statistical potential derived from
  our own PISCES train split (`scripts/53_ncap_stats.py` → `physddg/data/ncap_pref.json`, method of Richardson &
  Richardson 1988, Science 240:1648): E(aa) = −ln P(aa | N-cap)/P(aa) in kT, over 58,800 N-caps. Ranking
  S < D < N < T < P < G < H < C < … < L < V < I, which matches the published ordering (Asn/Asp/Ser/Thr/Cys best,
  Gly favoured). The B6 gain is unchanged by the swap (`+ncap_pref` alone: val natural 0.7329 → 0.7422, CV +0.008).
- **Helix propensity (Pace & Scholtz 1998, Biophys J 75:422)** — the 19 non-Pro values were verified against the
  paper's abstract; Pro = 3.16 is the commonly quoted value and is *not* verified. Not used by any shipped model
  (helix propensity failed selection), so this is documentation only.
- **β-propensity (Minor & Kim 1994)** — unverified, Pro hand-set; also unused. Both are now flagged in the
  `physddg/features_extra.py` docstring with "do not cite".
- **Helix dipole** needs no scale: it is Δ(formal charge) × (+1 at N1–N3, −1 at C1–C3), with the weight fitted
  (Hol et al. 1978, Nature 273:443; Shoemaker et al. 1987, Nature 326:563).

### Selection (Megascale-val natural ρ_prot / grouped-CV natural ρ_prot; stage 2)
| model | val | CV |
|---|---|---|
| B5 (base) | 0.7329 | 0.6727 |
| + site_scsc | 0.7438 | 0.6844 |
| + ncap_pref | 0.7422 | 0.6802 |
| + helix_dipole | 0.7330 | 0.6728 |
| + site_hbsc_bur | 0.7330 | 0.6845 |
| **B6 = +3 terms (scsc, ncap, dipole)** | 0.7474 | 0.7011 |
| B6 + site_hbsc_bur (= B6hb, 4 terms) | 0.7474 | 0.7011 |
| B6 **+ pLDDT attenuation** (= B6af) | **0.7494** | **0.7096** |
| B6 + burial attenuation (control) | 0.7474 | 0.7011 |
| B6-D15 (state-function variant) | 0.7360 | 0.6839 |

**The burial-matched control matters.** `notes/af_mode.md` §4 found that pLDDT shrinkage is a proxy for a
burial-dependent calibration and recommended against shipping it. Re-run here *on top of a model that already
contains burial and burial-weighted terms*, the control behaves differently: the burial version is pinned at 0 by
its sign constraint and, refitted sign-free, takes **+0.75** — it wants to *amplify* buried predictions, not
attenuate them — while pLDDT attenuation keeps its negative weight (−0.90 ± 0.66) and adds +0.0085 CV. So in B6
the pLDDT term is not the burial confound the AF study identified. Its CI still includes 0, and it buys +0.006 to
+0.018 on test, so it is a small effect that I ship only in AF mode.

### Results (D17 protocol: identical rows, strict-only S461/S669, FoldX raw and clipped; full table in `main_table.md`)
| model | MS nat ρ_prot | MS nat-strict | MS all ρ_prot | MS all r | FP r (338) | S669-strict r | S461-strict r | Dom ρ_dom | Dom flag-kept | Abund ρ |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 0.687 | 0.697 | 0.685 | 0.686 | 0.467 | 0.315 | 0.556 | 0.412 | 0.440 | 0.469 |
| **B6s1** (stage 1) | 0.690 | 0.699 | 0.690 | 0.688 | 0.427 | 0.302 | 0.567 | 0.433 | — | 0.471 |
| **B6** (stage 2) | 0.701 | 0.705 | 0.695 | 0.692 | 0.439 | 0.291 | 0.545 | 0.437 | 0.471 | 0.472 |
| **B6af** (AF mode) | **0.705** | 0.707 | **0.701** | **0.695** | 0.439 | 0.291 | 0.545 | **0.444** | **0.472** | 0.480 |
| B6hb (+site_hbsc_bur) | 0.711 | 0.703 | 0.702 | 0.693 | 0.452 | 0.286 | 0.536 | 0.436 | — | 0.481 |
| B6d15 (state-function only) | 0.691 | 0.703 | 0.689 | 0.686 | 0.459 | 0.319 | 0.551 | 0.417 | — | 0.467 |
| B6j (joint, FireProt-train) | 0.702 | 0.700 | 0.682 | 0.681 | 0.469 | 0.329 | 0.567 | 0.463 | — | 0.484 |
| FoldX raw / clipped | 0.629 | 0.640 | 0.633 | 0.447 / 0.590 | 0.417 / 0.423 | – | – | 0.430 | 0.458 | – |
| tmol-refit | 0.622 | 0.637 | 0.626 | 0.629 | 0.486 | – | – | – | – | – |

Paired bootstrap, Megascale-test natural (identical rows):
- **B6 − B5 = +0.014 [−0.001, +0.036]** ρ_prot, Pearson +0.009 [+0.003, +0.015]. B6af − B5 = +0.018 [+0.006, +0.038].
  B6s1 − B5 = +0.004 [−0.010, +0.028]: at stage 1 the new terms roughly recover what stage 2 buys.
- **B6 − FoldX = +0.072 [+0.015, +0.133]**, B6 − FoldX-clip the same on rank and +0.113 [+0.074, +0.143] on Pearson,
  B6 − tmol-refit +0.078 [+0.034, +0.108], B6 − B3+ +0.052 [+0.031, +0.076]. All exclude 0.
- Domainome: **B6 − B5 = +0.025 [+0.013, +0.033]** per-domain ρ (and +0.032 on the flag-kept subset), but
  B6 − FoldX is only +0.007 [−0.020, +0.039] — we finally match FoldX there, we do not beat it.
- **FireProt-test: B6 is worse than B5** (Pearson −0.026 [−0.049, +0.015], Spearman 0.490 → 0.466) and B6s1 worse
  still (−0.036). `site_scsc` is the cause (the residual analysis saw the same: −0.016 alone). On crystal structures
  of larger proteins the side-chain-contact term mis-scales. **B6's gains are Megascale/Domainome gains; on
  FireProt it costs ~0.03 Pearson, all of it within noise but consistently negative.**
- S669-strict and S461-strict both *drop* (0.315 → 0.291, 0.556 → 0.545), same cause, same non-significance.

### G6 transfer of the new terms (Megascale-only vs FireProt-only fits, 200 cluster bootstraps, ratio ÷1.7)
| term | w(Megascale) | w(FireProt) | scaled ratio | verdict |
|---|---|---|---|---|
| `ncap_pref` | 0.287 ± 0.035 | 0.473 ± 0.122 | 0.97 | passes cleanly |
| `helix_dipole` | 0.090 ± 0.021 | 0.132 ± 0.065 | 0.87 | passes cleanly |
| `site_scsc` | 1.034 ± 0.162 | 0.616 ± 0.243 | 0.35 | passes, but at the edge of the 3× rule |
| ~~`site_hbsc_bur`~~ | 0.093 ± 0.019 | 0.039 ± 0.047 | 0.25 | **fails → dropped from the headline** (kept as `B6hb`) |

No sign flips, in constrained or sign-free refits. Reference energies correlate r = 0.73 between the two fits.
Dropping `site_hbsc_bur` costs 0.004 val / 0.007 CV at stage 2 and nothing at stage 1, so the G6 rule is cheap to
honour here. `site_scsc` at 0.35 is the term to watch: it is a WT-site main effect, it does not transfer well, and
it is the term that hurts FireProt/S669/S461.

### Strata (Megascale-test, per-protein median ρ; `strata.csv`)
B6/B6af vs B5 gain where the terms were designed to act — exposed sites (0.554 → 0.578/0.580), charged WT
(0.516 → 0.525), pLDDT 70–90 (0.565 → 0.589) — and lose on from-Gly (0.639 → 0.594) and designed domains
(0.684 → 0.671, no pLDDT there). Buried small→large is unchanged (0.495 → 0.477/0.491), so B6 does not touch the
fixed-backbone clash problem. Stabilising mutations: AUROC 0.738 → 0.742, precision@5 0.152 → 0.190 for B6af,
still behind FoldX (0.219) and tmol-refit (0.181).
Domainome by pLDDT band: ≥90 0.489, 70–90 0.326, 50–70 −0.002, <50 −0.037 — **still zero below pLDDT 70**, which is
why the flag-kept subset (pLDDT ≥ 70 and ≥ 3 residues from a terminus, `notes/af_mode.md` §5) is the right report
for AF inputs: 0.471 for B6 against 0.440 for B5 and 0.458 for FoldX.

### Term weights (kcal/mol per unit; `weights.csv`)
`site_scsc` 1.034 ± 0.162 (per unit of contacts/30, so ~1 kcal/mol between an isolated and a fully packed side
chain), `ncap_pref` 0.287 ± 0.035 (per kT of N-cap log-odds; kT = 0.59, so the potential is ~2× over-confident, the
same factor as the environment potential), `helix_dipole` 0.090 ± 0.021 (per elementary charge at a helix end:
0.09 kcal/mol, the right order for a charge–dipole term), `scale_plddt` −0.902 ± 0.659 (the prediction is scaled by
1 − 0.90·(1 − pLDDT/100), i.e. ×0.73 at pLDDT 70, ×0.55 at pLDDT 50). Adding these three terms pulls `bur` down
from 0.932 to 0.536: `site_scsc` takes over part of the burial main effect, which is consistent with it being a
better site-sensitivity descriptor than RSA — and with its weaker transfer.

### Verdict and recommendation
- Ship **B6** (stage 2) as the default and **B6s1** as the fast mode; both are xtal-compatible. Use **B6af** only
  when B-factors are pLDDT. Models are in `results/ladder_physics/models/` with `README.md` documenting the
  schema, the feature contract and the two-stage attenuation rule for E1's `predict.py`.
- B6 is the first rung that beats FoldX on Megascale rank with a CI excluding 0 on **all three** Megascale subsets
  (natural +0.072 [+0.015, +0.133]), so **G1 now passes against FoldX on rank**, not only on Pearson. Against the
  D16 bar (B3+ + 0.03) it passes at +0.052 [+0.031, +0.076].
- It does not help, and slightly hurts, on the low-throughput crystal sets. If the final product is judged on
  FireProt/S669, `site_scsc` should be dropped (B6 minus site_scsc: val 0.7391, CV 0.6909 — still above B5).

---

## Gate verdicts under D16/D17 (paired-bootstrap CIs, identical rows)

**G1 — fails on rank against FoldX, passes on calibration.**
| comparison (Megascale-test) | natural ρ_prot | natural-strict ρ_prot | all ρ_prot | natural Pearson |
|---|---|---|---|---|
| B4 − FoldX | +0.048 [−0.008, +0.105] | +0.048 [+0.002, +0.119] | +0.050 [+0.002, +0.110] | +0.252 [+0.193, +0.297] |
| B4 − FoldX-clip | +0.048 [−0.008, +0.105] | +0.048 [+0.001, +0.119] | +0.050 [+0.001, +0.110] | **+0.100 [+0.062, +0.132]** |
| B5 − FoldX | +0.058 [−0.003, +0.113] | +0.058 [+0.011, +0.118] | +0.052 [+0.010, +0.118] | +0.256 [+0.200, +0.299] |
| B5 − FoldX-clip | +0.058 [−0.003, +0.113] | +0.058 [+0.011, +0.118] | +0.052 [+0.010, +0.118] | +0.104 [+0.068, +0.134] |
| B4 − B3+ | +0.029 [−0.002, +0.051] | +0.038 [+0.016, +0.052] | +0.034 [+0.005, +0.049] | +0.051 [+0.037, +0.062] |
| B5 − B3+ | +0.038 [+0.005, +0.062] | +0.049 [+0.016, +0.066] | +0.036 [+0.008, +0.060] | +0.055 [+0.039, +0.069] |
| B4 − tmol-refit | +0.055 [+0.016, +0.085] | +0.050 [−0.001, +0.107] | +0.057 [+0.003, +0.076] | +0.067 [+0.054, +0.080] |

- D16 requires B4 ≥ B3+ + 0.03 **and** a paired CI over FoldX excluding 0 on the primary subset (natural).
  **B4 fails both** (+0.029, and the FoldX CI includes 0). **B5 clears the B3+ bar** (+0.038, CI excludes 0) but its
  FoldX CI also includes 0 on natural. On natural-strict and on all 28 proteins the FoldX CI does exclude 0.
  The honest statement: *ahead of FoldX in point estimate on every Megascale subset, significantly so on two of the
  three, but not on the pre-registered one.*
- **The calibration claim survives clipping but shrinks.** On the natural subset raw FoldX has Pearson 0.445 and
  RMSE 3.22 kcal/mol against B4's 0.696 / 0.78, but most of that gap is FoldX's uncapped tail (predictions up to
  71 kcal/mol). Clipping FoldX to [−3, 8], with nothing fitted, lifts it to Pearson 0.596 and RMSE 1.85. The
  remaining advantage, +0.100 [+0.062, +0.132] Pearson and ~1.07 kcal/mol RMSE, is real and significant — but it is
  +0.10, not the +0.25 the first issue quoted.
- Against the other classical methods the margin is larger and significant: B4 − tmol-refit +0.055 [+0.016, +0.085],
  B4 − KORPM +0.177, B4 − DDGun3D +0.160 (natural ρ_prot).
- **Not decoration either way.** Removing the environment statistical potential costs 0.015 (B4 0.677 → 0.662);
  removing all engine energy terms costs 0.041 (CV); keeping only the energies (no burial/env/Rama/indicators) costs
  0.090. Both halves carry signal and they overlap heavily.
- **On FireProt-test nothing is significant**: B4 − FoldX Pearson +0.038 [−0.076, +0.105], B4 − tmol-refit
  −0.017 [−0.075, +0.036], B4 − DDGun3D −0.038. CIs there are ±0.2 wide.

**G2 — fails overall, passes on buried small→large, and that now rests on held-out folds, not on test.**
The first issue justified the stratum claim with test data (audit item 9). Re-done with grouped-CV out-of-fold
predictions on Megascale-train and on Megascale-val (`g2_stratum_check.csv`):
| stratum | train CV OOF ΔρB5−B4 (pooled / per-prot) | folds B5 better | Megascale-val (pooled / per-prot) |
|---|---|---|---|
| all | +0.004 / +0.003 | 5/5 | +0.005 / +0.005 |
| buried (RSA < 0.25) | +0.008 / +0.016 | 5/5 | +0.013 / −0.005 |
| small→large (ΔV < −30 Å³) | +0.017 / +0.026 | 5/5 | +0.017 / +0.007 |
| **buried small→large** | **+0.034 / +0.058** | **5/5** | **+0.068 / +0.078** |
- Overall gain is +0.004 (CV) / +0.005 (val) / +0.010 (test, n.s.), far below the 0.02 bar → **G2 fails overall**.
- On buried small→large the gain is ≥ 0.03 and present in 5/5 folds, i.e. it meets both the magnitude and the
  ≥70%-of-folds criterion on data never used for fitting. The stratum is 8–9% of rows.
- **Recommendation unchanged**: ship stage 1 as the default; keep stage 2 for buried small→large and for AF uniform
  repacking (G5). Runtime is within target either way.

**G6 — passes.** No sign flips in B4 or B5 between the Megascale-only and FireProt-only fits; scaled ratios (÷1.7 for
the assay scale) are 0.6–1.8 for every energy term, the env potential, Rama, burial and the cavity term. `dTS` is
undetermined on FireProt (0) and the disulfide indicator is undetermined on Megascale (no such rows). Reference
energies correlate r = 0.70 (B4) / 0.75 (B5). The Megascale δ_ref nuisance exceeds 0.4 kcal/mol only for Cys (−0.55).
Full tables in `transfer_B4.csv` / `transfer_B5.csv`.

## Repulsion cap selection (D9) — chosen on train/val only, and it does not matter
E1 supplied three stage-1 cap settings (capA c0=1, capB c0=3 = the original, capC c0=1 c1=2 rising with burial), each
with an extra log-softened repulsion column. Selected by Megascale-val natural ρ_prot + grouped-CV natural ρ_prot:
| variant | val natural ρ_prot | CV natural ρ_prot | p99 of the repulsion column |
|---|---|---|---|
| capA hard / soft | 0.7276 / 0.7190 | 0.6691 / 0.6672 | 65.9 / 53.3 |
| capB hard / soft | 0.7265 / 0.7195 | 0.6709 / 0.6678 | 132.9 / 52.8 |
| **capC hard** (selected) | **0.7291** | **0.6715** | 122.4 |
| capC soft | 0.7201 | 0.6680 | 52.8 |
The spread across all six is 0.010 on val and 0.004 on CV, i.e. **within the CV standard error (0.0064)**. Refitting
the headline with the winner changes nothing measurable on test either (B4 0.677 → B4-capC 0.678 natural ρ_prot).
The log-softened column halves the raw heavy tail (p99 123 → 53 kcal/mol) and removes the need for my signed-log
transform, at a cost of ~0.007 val ρ; it is the better-behaved feature but not the more accurate one.
**Conclusion: the cap setting is not where the remaining error is.** I would keep capC (physically motivated, D9)
and state that the choice is immaterial. Stage-2 cap variants arrived after this selection; given that all six
stage-1 settings are within noise I did not rerun the stage-2 selection, and `B5` uses the original stage-2 features.

## Antisymmetry (D15) — unchanged conclusion, now with the two-structure column labelled
Single-structure scores (what is available at inference) and, separately, the two-structure symmetrisation:
| predictor | r(dir, −rev) | bias (kcal/mol) | Pearson direct | Pearson reverse | Pearson direct, 2-structure sym |
|---|---|---|---|---|---|
| raw stage-1 ΔF | 0.13 | +7.95 | 0.260 | −0.037 | 0.163 |
| raw stage-2 ΔF | 0.04 | +6.25 | 0.386 | 0.073 | 0.364 |
| B4 | 0.23 | +1.39 | 0.697 | 0.213 | 0.598 |
| B5 | 0.25 | +1.37 | 0.699 | 0.240 | 0.613 |
| B4-odd (no WT-site-only terms) | 0.59 | +0.44 | 0.618 | 0.410 | 0.574 |
| B5-odd | 0.57 | +0.38 | 0.618 | 0.451 | 0.604 |
| B3+ | −0.22 | +1.82 | 0.586 | −0.001 | 0.480 |
| KORPM / DDGun3D | 0.87 / 0.99 | +0.15 / +0.04 | 0.566 / 0.556 | 0.494 / 0.534 | 0.549 / 0.546 |

The last column is ½[pred(direct on the WT structure) − pred(reverse on the mutant structure)]. It needs **both**
experimental structures, so it is a two-structure diagnostic and not an inference-time score; it is antisymmetric by
construction and it *lowers* direct accuracy for B4 (0.697 → 0.598). E1's single-structure symmetrisation
(`sym_s2_dF`, scoring the reverse on the modelled mutant environment) does not fix the defect either (r 0.15,
bias +3.5 unfitted), so option (b) from the first issue is closed.

**Where the defect lives** (`antisymmetry_terms.csv`): every engine term is already nearly antisymmetric across the
two structures (r(dir, −rev) 0.97 lj_atr, 0.98 solv_apolar, 0.93 solv_polar, 0.90 elec, mean sums ±0.1–0.3 kcal/mol)
**except lj_rep** (mean dir+rev +7.5 stage 1 / +5.9 stage 2, r 0.29). The softened repulsion halves that bias
(+3.2) without improving r (0.35), and the capped variants only move it from +7.5 to +4.2 (capA). So the fixed
backbone, not the cap, is the cause: each crystal structure is relaxed around its resident residue. Honest summary:
**our headline models are not antisymmetric, and no cap or transform available to us fixes it.** Dropping the
WT-site-only main effects (B4-odd) buys r = 0.59 at a cost of 0.044 natural ρ_prot — a real trade, available on a
single structure, and the variant I would report whenever antisymmetry matters.

## Where we are not better (carried over from the audit, re-checked on identical rows)
| set / metric | ours | better |
|---|---|---|
| Domainome ρ_dom | B4 0.406, B5 0.412 | FoldX 0.430 (B4 − FoldX −0.024 [−0.048, +0.013], n.s.) |
| FireProt-test Pearson (347 rows) | B4 0.450, B5 0.456 | DDGun3D 0.488, tmol-refit 0.477 (all n.s.) |
| FireProt-test ρ_prot (9 proteins) | B4 0.578 | tmol-refit 0.678, FoldX 0.634 |
| S669-strict Pearson | B4 0.328 (B3+ 0.330 — no engine gain at all) | DDGun3D 0.347 |
| Megascale buried small→large | B5 0.495 | FoldX 0.548 |
| Stabilising precision@5 | B4 0.133, B5 0.152 | FoldX 0.219, tmol-refit 0.181 |
| Ssym antisymmetry | B4 r 0.23 | KORPM 0.87, DDGun3D 0.99 |

## Disclosure: PDB-statistics overlap
The PISCES set behind the rotamer library, Ramachandran maps and environment potential contains homologues of 10/28
Megascale-test domains, 24/27 FireProt-test proteins, 70/94 S669, 33/48 S461 and 262/517 Domainome constructs. The
audit re-derived the tables without the 300 overlapping chains: the env potential moves by 0.017 kT on average,
Rama by 0.010 kT, and B4/B5 predictions by 0.0014 kcal/mol, changing per-protein ρ by ≤ 0.0005. Disclosed, not
corrected.

## Term weights, ablations, strata, heavy tails
Unchanged from the first issue except that the headline disulfide feature is now the plain indicator; the numbers in
the appendix sections "Term weights", "Heavy tails", "Drop-one ablations" and "Strata" still stand (the weights are
identical to 3 decimals, because Megascale-train contains no disulfide-breaking rows and the feature is inactive
there). The one correction: in `B4j`/`B5j` the disulfide weight (3.25 / 3.32 kcal/mol, fitted on 8 FireProt-train
rows) now also applies to the two catalytic-CXXC test rows, which is why their FireProt Pearson falls to 0.389/0.396.

## Requests to E1
- **R-fit1** (posted in `notes/engine_mutate.md`): the WT→WT test in `tests/test_mutate.py:43-45` is tautological —
  ΔF is defined as F − F[wt], so the WT entry is 0 by construction and the test cannot fail. Suggested replacements
  that would actually test D6 are in that note.
- Stage-2 cap variants and reverse/frozen columns have landed; given that all stage-1 cap settings are within CV
  noise I have not rerun the stage-2 selection. If the backbone-ensemble and single-structure symmetrisation are
  both dropped (as E1's pre-tests indicate), the remaining lever on the repulsion term is backbone give, not the cap.

---

# APPENDIX — superseded first issue (2026-10-26, before the audit)

Everything below is the original write-up, kept for traceability. Its headline table used S461/S669-all,
ranked joint models on Ssym, reported FoldX only raw, evaluated each model on its own rows, and contained the
test-informed CXXC exception in the headline joint models. **Do not quote these tables.**

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
