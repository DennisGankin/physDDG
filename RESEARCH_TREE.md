# PhysDDG research tree

Status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

Decisions are logged as D<n> (see bottom). Dates are 2026.

## 0. Setup
- [x] 0.1 Project skeleton, PROMPT.md, CONVENTIONS.md, git, venv (10-26)

## 1. Literature & landscape
- [x] 1.1 LITERATURE.md (1,003 lines + notes/lit_*.md): targets table (Megascale-test ρ: FoldX 0.569, PhysDDG v1.2 0.640, ThermoMPNN 0.725, SPURS 0.74; S461 DDGun3D 0.635; Domainome FoldX 0.391, ESM-MSR 0.573)
  - [!] 1.1a Site-only oracle (LOO site mean) = 0.757 pooled ρ on Megascale-test; site+aa additive 0.800 → per-site sensitivity is where the accuracy lives (lit agent, scripts/lit_*.py)
  - [-] 1.1b Refitting FoldX's 22 terms under grouped CV is worse than stock FoldX (0.541 vs 0.569): reweighting FoldX-style terms is a dead end
  - [x] 1.1c Competitors KORPM / DDGun / DDGun3D on all our tables (reproduce literature: KORPM Ssym r 0.695, DDGun3D S669 r 0.421) — notes/competitors.md
    - Megascale-test pooled ρ: KORPM 0.482, DDGun3D 0.458 (FoldX 0.569); FireProt-test 0.408 / 0.431 (FoldX 0.449); S461 0.557 / 0.573; Ssym 0.718 / 0.534
    - [!] KORPM is trained on ThermoMutDB+ProTherm: 90% of FireProt, 69% S669, 97% Ssym rows have training homologues; S669 ρ 0.337 → 0.148 on strict subset. Classical sets are not clean tests for any fitted classical method; Megascale and Domainome are (no overlap)
    - KORPM 0.44 ms/mutation on one core: our speed target must be compared with this, not FoldX
  - [x] 1.1d tmol 0.2.0 (beta_nov2016, GPU) as reference engine — notes/tmol_eval.md
    - Megascale-test: raw 0.492 (rep. blow-ups, Pearson 0.18); 12-term refit + 20 ref energies 0.596 (per-prot 0.626), ≈ FoldX / B3+; 3.1 min per 349-res scan (ours stage 2: 18.7 s per 490 res, 13-50x faster)
    - [!] tmol-refit correlates only 0.76 with B3+ and 0.66 with FoldX; equal-weight sums gain ~+0.03 over B3+ (indicative, test z-scores)
    - [-] Not adopted as engine (too slow, Rosetta-style clashes); kept as baseline. Open: Dunbrack-in-tmol licence unresolved
- D12 (10-26) Primary fit on Megascale-train only (val for selection); FireProt-train only for G6 transfer check / optional joint variant; all other sets held out.
- D13 (10-26) AF side chains: repack only site + shell (lit evidence that AF side chains beat packers on AF backbones) — global repack only as an ablation.

## 2. Data
- [x] 2.1 Megascale: ThermoMPNN set (train 215,731/239 · val 27,328/31 · test 28,172/28), dG/CI/censor flags, 14,656 censored singles as intervals, 162k doubles; structures are Tsuboyama's AF2 models (!)
- [x] 2.2 FireProt (train 2,669/57 · val 387/15 · test 347/27, no sign flip), S669/S461/Ssym(pairs)/S2648; 100% WT identity — notes/data_ht_report.md
- [x] 2.3 Domainome (536k variants / 522 domains; AF v4 construct crops, 0 WT mismatches; FoldX join 100%) + paired xtal/AF (4,012 usable rows) + ProteinGym abundance DMS (7 assays, 38k) — notes/data_af_report.md
  - [-] MoCHI per-variant ΔΔG not released (only family-level scaled energies → `ddg_mochi_scaled`, validation only)
  - D11 (10-26) No ColabFold models for AF-less viral proteins (T4 lysozyme etc.) for now: paired set is large enough; revisit if AF-mode needs more low-pLDDT paired data.
- [x] 2.4 MMseqs2 homology families + leakage columns: Megascale-test 9/28 with train homologs; Ssym 96.5% rows homologous to FireProt-trainval (→ antisymmetry diagnostic only)

## 3. Evaluation harness & baselines
- [x] 3.1 physddg/eval.py + scripts/30_evaluate.py (metrics, bootstrap, paired comparisons, offset detection)
- [x] 3.2 Baselines: FoldX 5.1 Megascale-test ρ 0.569 (per-prot 0.633), FireProt-test ρ 0.449; burial-only 0.468 / 0.314

## 4. PhysDDG engine
- [x] 4.1 Design doc v0 + independent critique (notes/critique_design_v0.md) → DESIGN v1 (D4–D10, gates G1–G7)
- [x] 4.1b Fitting framework physddg/fit.py (censored/Huber/rank losses, Δref prior+shared+nuisance, cluster weights, grouped CV, bootstrap SE) + baseline ladder — notes/baseline_ladder.md
  - [!] B3+ (env potential + burial + Rama + Pro/Gly/SS indicators + Δref; 10+20 params, no side-chain modelling) = FoldX-level: MS-test natural per-protein ρ 0.648 vs FoldX 0.629 (n.s.), Pearson +0.18; FP-test r 0.43 vs 0.41; Domainome 0.39 vs 0.43
  - [!] Signal sits across sites (site ρ 0.68) more than within sites (0.44); LOO site oracle 0.719 per-protein
  - [!] Overpacking (ΔV+) has no signal beyond env potential; cavity (ΔV−) strong and transfers
  - [!] Domainome penalises exposed →F/W/Y/I/L/V (in-cell abundance, not folding): Megascale Δref transfers in the core (ρ 0.84) not at the surface (ρ −0.09)
  - [!] Ssym: every model with a WT-burial term is non-antisymmetric (bias +1.8–2.1) → D15
  - [-] G6 on B3+: cis-Pro term flips sign, ΔV+ zero → removed; disulfide only fittable on FireProt; catalytic CXXC disulfides need an exception
- [x] 4.2 Structure prep / topology / PDB-derived rotamer library (PISCES pc30 ≤1.8 Å, 9,266 chains; coverage top-20 96.6%, all 98.4% within 30°) / GPU NeRF builder (0.10 Å median rebuild RMSD, 24.5M res-rot/s) / env + Rama + entropy statistics — notes/engine_geometry.md
- [ ] 4.3 GPU batched mutant packing
- [x] 4.4 Energy terms on GPU: 11 separate terms (LJ atr/rep, EEF1 apolar/polar, H-bond sc-bb/sc-sc/bb-bb, Debye–Hückel / buried / distance-dep elec, disulfide); DSSP H-bond P/R 0.98/0.99; 10–80k variants/s — notes/engine_energy.md
  - [x] 4.4b Parameters verified vs param19.inp + OSPREY eef1parm.dat (fixed dG_free != dG_ref for 11 types, amide N dG_ref); StatTerms wired; clash-free native recognition 0.70/0.71
- [x] 4.3 Mutation pipeline physddg/mutate.py: Stage 1 (Boltzmann site model, fixed env, ~1,900 mut/s/GPU) + Stage 2 (mean-field, site + ≤16 flexible neighbours, ~550 mut/s/GPU); features for 983k variants / 1,713 structures — notes/engine_mutate.md
  - Unfitted unit-weight ΔF: MS-test ρ 0.507 (S1) / 0.519 (S2), FP-test 0.276 / 0.308 (FoldX 0.569 / 0.449); χ recovery buried χ1 0.96 / χ1+2 0.94
  - [!] Ssym unfitted: direct vs reverse r = 0.13 / 0.04, bias +8 / +6 kcal/mol — each structure is adapted to its resident residue (a native-adaptation bias); fix must come from fitting + state-function design (D15)
- [x] 4.6 Fit physics features B4/B5 — notes/ladder_physics.md (fit on Megascale-train only; selection on val + grouped CV)
  - [!] B4 0.677 / B5 0.687 per-protein ρ (MS-test natural; FoldX 0.629, tmol-refit 0.622, DDGun3D 0.516, KORPM 0.500); Pearson 0.68 vs FoldX 0.45; FP-test r 0.45–0.48 (≈ FoldX 0.41, n.s.); Domainome ρ_dom 0.41–0.44 (FoldX 0.43)
  - G1 partial: B4−B3 +0.113 ✓; B4−B3+ +0.029 [−0.002,+0.051]; B5−B3+ +0.038 sig.; B4 ≥ FoldX+0.08 ✗ (+0.048); not decoration: energies-only loses 0.09, no-env loses 0.015
  - G2: B5−B4 +0.010 overall ✗ (2/5 folds) but buried small→large +0.048 ✓ → ship Stage 1 default, Stage 2 for buried small→large / AF repacking
  - [!] G6 passes: no sign flips, ratios 0.6–1.8; all energy weights positive (physical) even unconstrained
  - [!] Antisymmetry defect is essentially lj_rep (r 0.29, dir+rev +7.5 kcal/mol); fitting cuts bias 8.0→1.4 but headline models aren't antisymmetric (r 0.59 only if WT-only main effects dropped)
  - [-] Watch: stabilising precision@5 worse than FoldX (0.13–0.15 vs 0.22); all methods ≈ 0 at pLDDT<70 on Domainome; FoldX still better on buried small→large
- [x] 4.7 D9 cap variants (capA 1/0, capB 3/0, capC 1+2·rigidity; Stage 1+2 all tables), soft log repulsion column, reverse scoring, backbone ensemble pre-test — notes/engine_mutate.md "v2 additions"
  - Unit-weight Ssym lj_rep bias: capA +4.2 (r 0.35), capB +7.5 (r 0.29), soft column +3.2 (r 0.35); Megascale-val raw ρ capA 0.562 / capB 0.552 / capC 0.559
  - [-] Single-structure symmetrisation sym_s2_dF = ½(fz − rev): r(dir,−rev) 0.15, bias +3.5 — the modelled mutant environment stays too close to the WT crystal environment; antisymmetry cannot be obtained from one structure with a fixed-backbone model
  - [-] G3 backbone ensemble: no gain (B4 weights unrefitted: Megascale-val 0.7265 → 0.7188–0.7257; buried small→large ≤ +0.009), cost 9× Stage 1 → dropped
- [x] 4.6b D17 re-issue (notes/ladder_physics.md; old tables in superseded appendix): CXXC out of headline; S461/S669 strict; FoldX raw+clipped; G1 fails on rank (B4−FoldX +0.048 [−0.008,+0.105]; B5 +0.058 [−0.003,+0.113]), passes Pearson (+0.100 [+0.062,+0.132] vs clipped FoldX) and RMSE (0.78 vs 1.85)
  - [!] G2 stratum now on held-out data: buried small→large +0.058 per-protein ρ OOF (5/5 folds), +0.078 val; overall +0.004–0.005
  - [-] Cap selection (capA/B/C × hard/soft) within noise (0.010 val, 0.004 CV): the cap is not where the error is; soft repulsion halves tail (p99 123→53) but −0.007 ρ; B4-odd (antisymmetric by dropping WT-only main effects) r 0.59 at −0.044 ρ
- [x] 4.8 Residual analysis (notes/residual_analysis.md, physddg/features_extra.py)
  - [!] B5 residual variance: 43% site-level / 57% within-site; each half worth equal headroom (val natural 0.733 → 0.846 if site means known, → 0.874 if within-site ordering known); GBM probe on 99 descriptors explains only 17% (CV) of site-mean residuals → mostly unpredictable from WT structure
  - Outlier families: under-predicted buried small→large in stable β-sheets (log-compressed repulsion caps ~0.7 kcal/mol), N-cap Ser/Thr/Asn→hydrophobic, buried polar H-bond networks; over-predicted Gly→Pro at positive φ in exposed loops, mutations to Cys (assay artefact); SH3 supplies ~80 of extreme sites per tail
  - 51 single terms + 2 scale groups tested: strict rule passes only site_scsc (CV +0.012, val +0.011); relaxed pre-fixed rule adds pLDDT attenuation, buried sc H-bond partners, N-cap, helix dipole (CV 0.673→0.710, val 0.733→0.749)
  - [-] No gain: helix/β propensity, angular aromatic/cation–π, geometric salt bridges, overpack/cavity forms, hydrophobicity×burial, Gly/Pro entropy, terminal flags, per-SS scaling
  - Single held-out look: MS-test all-28 per-protein ρ 0.685 → 0.690 (strict) → 0.704 (relaxed); natural-21 0.712 (+0.026 [+0.004,+0.047]); Domainome 0.411 → 0.431 / 0.438; FireProt n.s. (≈ lit. DL 0.77–0.83 still far)
  - Caveat: propensity / N-cap / dipole scales typed from memory → being replaced with literature values (agent: fit)
- [x] 4.10 B6 (notes/ladder_physics.md §B6): B5 + site_scsc + N-cap (PISCES-derived, 58,800 caps) + helix dipole; B6af adds pLDDT attenuation (AF only)
  - [!] B6 MS-test natural/all per-protein ρ 0.701 / 0.695 (B6af 0.705 / 0.701); B6 − FoldX +0.072 [+0.015,+0.133] — G1 now passes on rank, CI excludes 0 on all 3 Megascale subsets; B6 − B3+ +0.052 [+0.031,+0.076]; B6 − B5 +0.014 [−0.001,+0.036]; Domainome B6 − B5 +0.025 [+0.013,+0.033]
  - [-] site_scsc hurts crystal sets (FireProt −0.026 Pearson, S669/S461-strict) and sits at the G6 3× edge (0.35); site_hbsc_bur fails G6 (0.25, undetermined on FireProt) → out of headline
  - Hand-typed N-cap table contradicted its source (Gln) → replaced by PISCES-derived potential; Minor–Kim/Pace–Scholtz scales unused and marked 'do not cite'
  - pLDDT attenuation passes the burial control (sign-free burial version amplifies buried predictions, +0.75) but weight CI includes 0, test gain only +0.006…+0.018 → AF only
  - [~] 4.10b Final headline B6g = B6 without site_scsc/hbsc (D19), Stage-1 analogue B6gs1 (default fast), B6gaf (AF); B6x (with site_scsc) as transparent variant; final ablation + G6 tables; JSONs final_{stage1,stage2,af}.json (agent: fit)
- [~] 6.2 Wire final models + AF-mode defaults into predict() (agent: E1)
- [x] 4.11 Optional 'PhysDDG+evo' (notes/evo_extension.md): B5 + ONE weight on a hhblits/UniClust30 profile log-ratio (−0.113 ± 0.007 kcal/mol per nat); 1,492 chains profiled (natural 89–98% deep; 109 designed Megascale domains have no homologue → term reduces to BLOSUM62)
  - Megascale-test per-protein ρ B5 0.685 → 0.726 (+0.040 [−0.005,+0.071]); natural-21 +0.052 but natural-strict-14 only +0.014 (−0.006 on the ρ column); 7 natural proteins with a train homologue +0.062 (7/7 up)
  - FireProt +0.022 (n.s.); S669-strict +0.037 (sig.); Domainome per-domain 0.411 → 0.486 (+0.074 [+0.059,+0.084]), pLDDT<50 −0.036 → 0.061
  - [!] Closes ~48% of the per-protein gap to ThermoMPNN on Megascale but only ~20% of pooled ρ/Pearson; ~85% on Domainome (0.486 vs 0.50)
  - Caveat: different information source (evolution, not physics), UniClust30 contains homologues of every natural benchmark protein; offered only as optional mode
- [x] 4.9 Independent audit (notes/audit_1.md): headline numbers recomputed from raw preds match to 4 decimals; no test labels enter fitting; no critical issues
  - [!] Over-claims corrected: (a) CXXC disulfide exception was derived from 2 FireProt-test rows (2TRX) — joint-model FP-test r falls 0.48→0.39 without it → treat as test-informed, report without; (b) S461 24.5% rows are exact Megascale-train mutations → report strict only; (c) Ssym is NOT a test for joint (FireProt-trainval) models (85.7% exact overlap); (d) G1 FAILS under D16 (B4−FoldX +0.048 [−0.008,+0.105], B5 +0.058 [−0.003,+0.113]) — only Pearson is clearly better; (e) FoldX Pearson gap is mostly its uncapped tail (clip to [−3,8], no fitting: 0.445→0.596 → gap +0.10 not +0.25) → always report FoldX-clipped too; (f) 'natural' Megascale subset not homology-clean (1TUC, 2KXD) but strict 0.687 ≈ 0.677; headline rests on 21 proteins
  - PDB-statistics overlap with test proteins (10/28 MS-test, 24/27 FP-test, 70/94 S669…) exists but harmless: re-derived without 300 overlapping chains → ρ change ≤ 0.0005; disclose only
  - Where we are NOT better than a competitor: Domainome ρ_dom (FoldX 0.430 vs 0.405), FireProt per-protein ρ (tmol 0.678 vs ours 0.578, 9 proteins), S669-strict r (DDGun3D), S461-strict per-protein (KORPM), buried small→large (FoldX), stabilising precision@5 (FoldX), Ssym antisymmetry (KORPM, DDGun3D)
- [ ] 4.5 Unfolded-state / reference model
- [ ] 4.6 Parameter fitting (few interpretable weights, grouped CV)

## 5. AlphaFold mode
- [x] 5.1 Xtal vs AF paired evaluation + Domainome by pLDDT band — notes/af_mode.md, results/af_mode/
  - [!] 5.1a On the paired set the AF model is never worse than the crystal and is **better where the two differ**
    (local RMSD 0.6–1 Å: raw ΔF ρ 0.463 AF vs 0.173 xtal; B4-class model +0.066). The fragile input is the crystal.
  - [!] 5.1b Domainome's low-pLDDT failure is mostly a **burial confound**: at buried sites pLDDT 50–70 costs only
    ~0.1 ρ, at exposed sites every physics model is ≈0 at *all* pLDDT. Biggest real failure modes: construct termini
    (10% of rows, ρ 0.11–0.15) and metal sites (7%, FoldX 0.44→0.14).
  - [x] 5.1c AF v4 vs v6 irrelevant (r 0.98, same ρ); **construct crop beats the full-length model** in every band.
- [x] 5.2 AF treatments + gate G5 — physddg/afmode.py (trim, cap, repack, shrink, burial swap, flags)
  - [x] G5 part 1 **PASS** for every fitted model (per-entity median AF↔xtal r: B3+ 0.982, P1 0.950, B4-class ≈0.95);
    fails only for the unit-weight energy (0.837). Global repacking does not help (0.950 → 0.940) — keep AF side chains.
  - [-] 5.2a pLDDT attenuation **does not ship**: +0.006…+0.022 ρ_dom, and the burial-matched control of the same
    2-parameter logistic gives +0.014…+0.073. It is a burial calibration, not an AF correction (→ D8).
  - [-] 5.2b pLDDT-modulated repulsion cap: 0.000; softer cap −0.004…−0.028. Backbone-only burial swap: ≤ +0.004.
  - [x] 5.2c Default AF mode = MAVISp-style trim (<50 windows + <70 termini, packed-exempt) + construct crop +
    stage-2 shell repack + flags. Trimming: 0.000…+0.009 ρ_dom, **+0.013…+0.017 on Megascale-test** (identical rows).
  - [!] 5.2d **Flagging beats correcting**: withholding pLDDT<70 ∪ terminal ∪ exposed-hydrophobic-patch sites
    (14% of rows) gains +0.038 ρ_dom for B4, +0.020 for B3+ — more than all treatments combined.

## 6. Speed & packaging
- [x] 6.1 Predictor CLI/API (`scripts/physddg predict`, physddg/predict.py; per-term contributions c_* sum to ddg; flags), README.md, real tests (no native rotamer at site to 1e-6; WT→WT exactly 0; bitwise deterministic across runs/GPUs) — predict() reproduces B4 stored preds exactly
  - [!] Speed (1× RTX 4090, end to end): Stage 1 ~1,500–2,000 mut/s (0.5–0.7 ms/mut): 59 res 0.6 s, 141 res 1.4 s, 269 res 2.9 s, 597 res 7.8 s; Stage 2 540–715 mut/s (269 res 7.2 s). Whole Megascale-test (28k) 9.5 s S1 / 17 s S2 on 4 GPUs; Domainome (532k) 55 s S1 / 172 s S2. vs FoldX 5.1 hours of CPU per scan; ≈ KORPM (0.44 ms/mut/core) per mutation — speed alone is not the contribution
  - [-] torch.compile (no gain S1, 15% S2, 60 s compile), multi-site batching (2× slower), fp16/bf16, CUDA graphs: not applied
  - Open: fit.py sets torch float64 globally on import (breaks test_energy::test_site_dependent_cap in full pytest) → fix in fit.py; AF-mode trimming not yet wired into predict()

## Decisions
- D19 (10-26) Headline model must generalise across assays: default = B6g (B5 + PISCES N-cap + helix dipole; no site_scsc, no buried-HB-partner term), Stage 1 by default (fast), Stage 2 for buried small→large/AF; B6x (with site_scsc) kept as a Megascale/Domainome-tuned variant; evo mode optional and separate. B6 selection followed a test-set look of the residual analysis → mild optimism in B5→B6 deltas, disclosed.
- D18 (10-26) AF mode default: MAVISp-style trim of low-pLDDT windows/termini (packed-exempt) + construct crop + Stage-2 shell repack with AF side chains kept elsewhere + per-variant confidence flags (low pLDDT, terminal, exposed hydrophobic patch, metal site); pLDDT shrinkage and pLDDT-modulated cap are not shipped. Note: the trim benefit on Megascale-test is a test-set observation (treatments were fitted on abundance_dms + Domainome half), so it is reported as supporting evidence only.
- D17 (10-26) Reporting protocol after audit: every table on identical rows; S461/S669 on strict subsets (no Megascale-train overlap); Ssym only as antisymmetry diagnostic (and never for models that touch FireProt-trainval); FoldX reported both raw and clipped to [−3,8]; Megascale headline on natural AND natural-strict AND all; no test-informed rules in headline models (CXXC exception removed from headline, shown as a separate variant); gate verdicts from paired-bootstrap CIs.
- D15 (10-26) Antisymmetry by construction: every term must be a difference of a state function S(aa, structure) between mutant and WT (burial-only / site-only terms are not allowed in the primary model; if used, only as a reported ablation). Reverse mutations are scored on the mutant structure.
- D16 (10-26) Gate G1 re-based on the measured ladder: primary metric = Megascale-test natural per-protein median ρ; physics engine B4 must reach ≥ B3+ + 0.03 (≈ 0.68) and beat FoldX with paired-bootstrap CI excluding 0; B3 (2 weights) was too low a bar.
- D14 (10-26) Repulsion radii scaled ×0.9 by default (raises native-χ recognition among non-clashing perturbations 28%→62%; calibrated on structures, not ΔΔG).
- D4–D10 (10-26) see DESIGN.md v1 revisions (env statistical potential, Boltzmann averaging, symmetric site modelling, shrunk/nuisance reference energies, censoring-aware fitting, site-dependent repulsion cap, screened electrostatics).
- D1 (10-26) Sign: ddg positive = destabilising, kcal/mol (FoldX convention); Megascale ddG_ML flipped at ingestion.
- D3 (10-26) Rotamer library + Ramachandran/entropy statistics derived from a PISCES-culled set of the local PDB mirror (licence-free, reproducible) unless a clearly better redistributable library exists.
- D2 (10-26) PyTorch allowed only as a GPU array/autograd library (no NN, no pretrained models) — satisfies "no deep learning" while enabling GPU speed.
