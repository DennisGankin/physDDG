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
- [~] 4.3' Mutation pipeline: Stage 1 Boltzmann site model in fixed env, Stage 2 mean-field neighbour repacking, features for all datasets (agent: E1)
- [ ] 4.5 Unfolded-state / reference model
- [ ] 4.6 Parameter fitting (few interpretable weights, grouped CV)

## 5. AlphaFold mode
- [ ] 5.1 Xtal vs AF paired evaluation
- [ ] 5.2 pLDDT-aware handling (trimming, confidence weighting, relaxation)

## 6. Speed
- [ ] 6.1 Profiling, GPU kernels, throughput benchmark vs FoldX

## Decisions
- D15 (10-26) Antisymmetry by construction: every term must be a difference of a state function S(aa, structure) between mutant and WT (burial-only / site-only terms are not allowed in the primary model; if used, only as a reported ablation). Reverse mutations are scored on the mutant structure.
- D16 (10-26) Gate G1 re-based on the measured ladder: primary metric = Megascale-test natural per-protein median ρ; physics engine B4 must reach ≥ B3+ + 0.03 (≈ 0.68) and beat FoldX with paired-bootstrap CI excluding 0; B3 (2 weights) was too low a bar.
- D14 (10-26) Repulsion radii scaled ×0.9 by default (raises native-χ recognition among non-clashing perturbations 28%→62%; calibrated on structures, not ΔΔG).
- D4–D10 (10-26) see DESIGN.md v1 revisions (env statistical potential, Boltzmann averaging, symmetric site modelling, shrunk/nuisance reference energies, censoring-aware fitting, site-dependent repulsion cap, screened electrostatics).
- D1 (10-26) Sign: ddg positive = destabilising, kcal/mol (FoldX convention); Megascale ddG_ML flipped at ingestion.
- D3 (10-26) Rotamer library + Ramachandran/entropy statistics derived from a PISCES-culled set of the local PDB mirror (licence-free, reproducible) unless a clearly better redistributable library exists.
- D2 (10-26) PyTorch allowed only as a GPU array/autograd library (no NN, no pretrained models) — satisfies "no deep learning" while enabling GPU speed.
