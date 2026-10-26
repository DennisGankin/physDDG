# PhysDDG research tree

Status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

Decisions are logged as D<n> (see bottom). Dates are 2026.

## 0. Setup
- [x] 0.1 Project skeleton, PROMPT.md, CONVENTIONS.md, git, venv (10-26)

## 1. Literature & landscape
- [x] 1.1 LITERATURE.md (1,003 lines + notes/lit_*.md): targets table (Megascale-test ρ: FoldX 0.569, PhysDDG v1.2 0.640, ThermoMPNN 0.725, SPURS 0.74; S461 DDGun3D 0.635; Domainome FoldX 0.391, ESM-MSR 0.573)
  - [!] 1.1a Site-only oracle (LOO site mean) = 0.757 pooled ρ on Megascale-test; site+aa additive 0.800 → per-site sensitivity is where the accuracy lives (lit agent, scripts/lit_*.py)
  - [-] 1.1b Refitting FoldX's 22 terms under grouped CV is worse than stock FoldX (0.541 vs 0.569): reweighting FoldX-style terms is a dead end
  - [ ] 1.1c Competitor baselines KORPM, DDGun3D on our tables (agent: competitors)
  - [ ] 1.1d tmol (GPU Rosetta REF2015, Apache-2.0) evaluation as engine/baseline (agent: tmol)
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
- [~] 4.1b Fitting framework physddg/fit.py + baseline ladder B0–B3+ + G6 tooling (agent: fit)
- [x] 4.2 Structure prep / topology / PDB-derived rotamer library (PISCES pc30 ≤1.8 Å, 9,266 chains; coverage top-20 96.6%, all 98.4% within 30°) / GPU NeRF builder (0.10 Å median rebuild RMSD, 24.5M res-rot/s) / env + Rama + entropy statistics — notes/engine_geometry.md
- [ ] 4.3 GPU batched mutant packing
- [~] 4.4 Energy terms on GPU (agent: E2)
- [ ] 4.5 Unfolded-state / reference model
- [ ] 4.6 Parameter fitting (few interpretable weights, grouped CV)

## 5. AlphaFold mode
- [ ] 5.1 Xtal vs AF paired evaluation
- [ ] 5.2 pLDDT-aware handling (trimming, confidence weighting, relaxation)

## 6. Speed
- [ ] 6.1 Profiling, GPU kernels, throughput benchmark vs FoldX

## Decisions
- D4–D10 (10-26) see DESIGN.md v1 revisions (env statistical potential, Boltzmann averaging, symmetric site modelling, shrunk/nuisance reference energies, censoring-aware fitting, site-dependent repulsion cap, screened electrostatics).
- D1 (10-26) Sign: ddg positive = destabilising, kcal/mol (FoldX convention); Megascale ddG_ML flipped at ingestion.
- D3 (10-26) Rotamer library + Ramachandran/entropy statistics derived from a PISCES-culled set of the local PDB mirror (licence-free, reproducible) unless a clearly better redistributable library exists.
- D2 (10-26) PyTorch allowed only as a GPU array/autograd library (no NN, no pretrained models) — satisfies "no deep learning" while enabling GPU speed.
