# PhysDDG research tree

Status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

Decisions are logged as D<n> (see bottom). Dates are 2026.

## 0. Setup
- [x] 0.1 Project skeleton, PROMPT.md, CONVENTIONS.md, git, venv (10-26)

## 1. Literature & landscape
- [~] 1.1 LITERATURE.md: methods (physics, statistical, DL), datasets, benchmarks, AF-structure issues, fast packing / GPU (agent: lit)

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
- [ ] 4.1b' Baseline ladder B0–B3 (pair lookup, RSA, hydrophobicity×burial, env statistical potential)
- [~] 4.2 Structure prep / topology / PDB-derived rotamer library / GPU NeRF builder (agent: E1)
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
