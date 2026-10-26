# PhysDDG research tree

Status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

Decisions are logged as D<n> (see bottom). Dates are 2026.

## 0. Setup
- [x] 0.1 Project skeleton, PROMPT.md, CONVENTIONS.md, git, venv (10-26)

## 1. Literature & landscape
- [~] 1.1 LITERATURE.md: methods (physics, statistical, DL), datasets, benchmarks, AF-structure issues, fast packing / GPU (agent: lit)

## 2. Data
- [~] 2.1 Megascale (Tsuboyama) canonical table + ThermoMPNN split + structures (agent: data-ht)
- [~] 2.2 FireProt + external low-throughput sets (S669, Ssym, ...) with homology vs fitting data (agent: data-ht)
- [~] 2.3 Domainome + AlphaFold structures (construct crops, pLDDT) + paired xtal/AF sets (agent: data-af)
- [ ] 2.4 Homology clustering across all sets; strict test subsets

## 3. Evaluation harness & baselines
- [~] 3.1 physddg/eval.py (metrics, bootstrap, paired comparisons) (agent: eval)
- [ ] 3.2 Baselines: FoldX 5.1 native, simple burial×hydrophobicity, substitution-matrix, RSA-only

## 4. PhysDDG engine
- [ ] 4.1 Design doc + independent critique
- [ ] 4.2 Structure prep / topology / rotamer library
- [ ] 4.3 GPU batched mutant packing
- [ ] 4.4 Energy terms
- [ ] 4.5 Unfolded-state / reference model
- [ ] 4.6 Parameter fitting (few interpretable weights, grouped CV)

## 5. AlphaFold mode
- [ ] 5.1 Xtal vs AF paired evaluation
- [ ] 5.2 pLDDT-aware handling (trimming, confidence weighting, relaxation)

## 6. Speed
- [ ] 6.1 Profiling, GPU kernels, throughput benchmark vs FoldX

## Decisions
- D1 (10-26) Sign: ddg positive = destabilising, kcal/mol (FoldX convention); Megascale ddG_ML flipped at ingestion.
- D2 (10-26) PyTorch allowed only as a GPU array/autograd library (no NN, no pretrained models) — satisfies "no deep learning" while enabling GPU speed.
