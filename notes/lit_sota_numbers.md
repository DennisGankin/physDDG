# Reported ΔΔG-predictor performance on the standard benchmarks

Compiled 2026-10-26 for PhysDDG (physics-based, no deep learning). Every number below was read from a
primary source (journal full text, preprint PDF, supplementary file, or released prediction file) unless
it carries one of the marks in the next section.

## 0. How to read this file

| mark | meaning |
|---|---|
| (no mark) | read directly from the cited table/text of the cited paper |
| **[U]** | UNVERIFIED — could not reach the primary source; value comes from a secondary citation or an abstract |
| **[D]** | DERIVED — computed by us/our agents from released per-variant prediction files, not published as such |
| **[P]** | preprint, not peer reviewed |

**Sign conventions differ between papers.** Pancotti/KORPM/Rossi use ΔΔG < 0 = destabilising; FoldX,
ThermoMPNN and this project use ΔΔG > 0 = destabilising. Correlation magnitudes are comparable; the bias
term ⟨δ⟩ and the sign of r are not. ⟨δ⟩ is always the mean of (direct + reverse) prediction, which is 0 for a
perfectly antisymmetric predictor. The antisymmetry correlation is written r_{d-r} (Pearson between direct
and reverse predictions); −1 is perfect, 0 is none. KORPM reports |r_{d-r}| (always positive).

**Pooled vs per-protein correlation.** Almost every number here is *pooled* over all variants in the
benchmark. The exceptions are flagged. Pooled correlation on Megascale is inflated relative to per-protein
ranking because between-protein variance in mean ΔΔG contributes; conversely on FireProt a handful of
proteins dominate. Where a paper reports both, both are given.

**RMSE is not comparable across methods** unless the predictor is on a kcal/mol scale. Rosetta
(REU), FoldX (overpredicts, unbounded) and ProteinMPNN (log-odds) produce large RMSE that says more about
scale than about ranking quality. Judge physics methods on Spearman/Pearson and on RMSE only after
calibration.

---

## 1. The benchmarks

| name | content | structures | notes |
|---|---|---|---|
| **Megascale / Tsuboyama** | cDNA-display proteolysis ΔΔG; 776k measurements, ~851k variants total | AF2 models + designed | Tsuboyama et al. 2023 *Nature* 620:434. ΔΔG inferred from a kinetic model; agreement with literature ΔΔG PCC 0.75–0.96; trypsin-vs-chymotrypsin agreement PCC 0.91 (the practical noise ceiling) |
| **Megascale test (ThermoMPNN split)** | 28,312 single mutations, 28 proteins | AF2 / designed | from 272,712 curated single substitutions over 298 proteins; MMseqs2 25% id clustering → 239/31/28 proteins (80/10/10 by mutation). THE default modern split |
| **S669** | 669 single variants, 94 experimental structures | X-ray + NMR | Pancotti 2022; from ThermoMutDB v1.3, <25% id to S2648 and VariBench; reverse set modelled with Robetta → 1,338 balanced |
| **S461** | 461 variants, 44 families | X-ray | KORPM (Hernández 2023) curation of S669: removed wrong entries and binding/metal-site mutations. **Not** a leakage-removal subset |
| **S612** | 612 of S669 that have an AlphaFold DB model with pLDDT > 70 | **AF2** | Degn 2025 [P]; the only published head-to-head of FoldX/Rosetta/RaSP/DDGun/ThermoMPNN *on AF structures* |
| **Ssym** | 342 direct + 342 reverse = 684 | X-ray WT + 342 mutant X-ray | Pucci 2018. Built for antisymmetry testing. All direct mutations are in the training sets of the classical tools, so only the inverse half is independent |
| **Ssym+** | 704 (352 + 352) | X-ray | Pancotti 2022 extension (+10 variants/structures) |
| **S4038** | 4,038 variants, non-overlapping with S2648 | X-ray | Zheng 2024 *Protein Sci* 33:e4861; 27 tools benchmarked |
| **FireProt (ThermoMPNN curation)** | 3,438 mutations, 100 proteins | X-ray | FireProtDB, pH nearest 7.4; splits 57/15/28 proteins |
| **FireProt-HF** | 2,578 mutations, 89 proteins | X-ray | homologue-free vs Megascale; the standard out-of-distribution test for Megascale-trained models |
| **Human Domainome 1** | 536,164 missense variants, 522 domains (aPCA abundance, **not** ΔΔG) | AF2 crops | Beltran, Jiang, Shen, Lehner 2025 *Nature* 637:885 |
| **ProteinGym (stability category)** | DMS assays, mostly Tsuboyama-derived | — | per-assay Spearman, then averaged |
| **S2648 / VariBench / VB1432 / Broom / PTMUL / KORPM-2371** | legacy training and test sets | X-ray | definitions at protddg-bench.github.io (Capriotti group) |

---

## 2. Master table

Megascale test = ThermoMPNN split, pooled, N=28,312, unless noted. S669/Ssym numbers are Pearson unless the cell
says ρ. Blank = not reported by anyone I could reach. Sources in §3–§6.

| method | type | Megascale test ρ / r / RMSE | S669 r dir (ρ) | S669 r rev | S461 r (ρ) | Ssym r dir / inv | r_{d−r} | ⟨δ⟩ | FireProt-HF ρ / r | Domainome ρ | PGym stability ρ | speed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **FoldX** | physics, empirical FF | 0.57 / 0.40 / 3.87 | 0.22 (0.283 **[D]**) | 0.22 | 0.30 · 0.221 | 0.63 / 0.39 | −0.20 S669 · −0.38 Ssym | −0.34 · −0.47 | 0.57 / 0.43 | — | — | **395–807 s/residue**, 64 cores; 0.0025 res/s |
| **Rosetta ddg_monomer** | physics, hybrid | — | 0.39 | 0.40 | — | 0.69 / 0.43 | −0.72 · −0.41 | −0.61 · −0.69 | — | — | — | — |
| **Rosetta cartesian_ddg** | physics, hybrid | 0.56 / 0.53 / 5.18 | — | — | 0.60 · 0.590 | 0.68 / 0.45 | \|0.41\| | −3.12 | 0.52 / 0.45 | not run (months) | — | **192–414 s/residue**, 64 cores |
| **EvoEF2 ("EvoFF")** | physics, empirical FF | — | — | — | 0.463 | 0.57 / 0.31 | \|0.58\| | −0.36 | — | — | — | — |
| **KORPM** | statistical, 6D KB potential, **12 fitted weights** | — | — | — | 0.569 | 0.57 / 0.49 | \|0.88\| | −0.15 | — | — | — | **684 muts / 3.5 s on ONE core** |
| **PoPMuSiC / -sym** | statistical + NN | — | 0.41 | 0.24 | 0.609 | 0.48 / 0.48 sym · 0.63 / 0.25 v2.1 | −0.77 | +0.03 | — | — | — | — |
| **DDGun** (seq) | statistical, **untrained** | — | 0.41 | 0.38 | 0.583 **[D]** | 0.48 / 0.48 | −0.99 | −0.007 | — | — | — | — |
| **DDGun3D** | statistical, **untrained** | — | 0.43 | 0.41 | **0.635** | 0.56 / 0.53 | −0.99 | −0.02 | — | — | — | — |
| **MAESTRO** | ML-classical | 0.47 / 0.55 / 1.04 | 0.50 | 0.20 | 0.630 | 0.52 / 0.32 | −0.22 | −0.57 | 0.60* / 0.62* | — | — | — |
| **mCSM** | ML-classical, graph sig. | 0.41 / 0.49 / 0.97 | 0.36 | 0.22 | 0.538 | 0.61 / 0.14 | −0.05 | −0.85 | 0.57* / 0.59* | — | — | — |
| **SDM** | statistical, env. propensity | — | 0.41 | 0.13 | 0.555 | 0.51 / 0.32 | −0.40 | −0.40 | — | — | — | — |
| **DUET** | ML-classical, meta | — | 0.41 | 0.23 | 0.593 | 0.63 / 0.13 | −0.12 | −0.67 | — | — | — | — |
| **CUPSAT** | statistical | — | — | — | — | 0.39 / 0.05 | −0.54 | −0.72 | — | — | — | — |
| **I-Mutant3.0** | ML-classical, SVM | — | 0.36 | 0.15 | 0.490 | 0.62 / −0.04 | +0.02 | −0.68 | — | — | — | — |
| **MUpro** | ML-classical, SVM | 0.29 / 0.31 / 1.08 | 0.25 | 0.20 | 0.396 | **0.79** / 0.07 | −0.02 | −0.97 | 0.57* / 0.58* | — | — | — |
| **DynaMut** | ML-classical, NMA | — | 0.41 | 0.34 | 0.504 | — | −0.58 | −0.06 | — | — | — | — |
| **DynaMut2** | ML-classical, meta | — | 0.34 | 0.17 | 0.504 ⚠ | 0.63 / 0.05 | +0.03 · \|0.11\| | −0.64 | — | — | — | — |
| **INPS-Seq** | ML-classical, SVR | — | 0.43 | 0.43 | 0.555 | 0.51 / 0.50 | −1.00 | 0.00 | — | — | — | — |
| **INPS3D** | ML-classical | — | 0.43 | 0.33 | 0.615 | 0.59 / 0.44 | −0.50 · −0.86 | −0.38 | — | — | — | — |
| **PremPS** | ML-classical, RF | — | 0.41 | 0.42 | 0.632 | 0.75 (fit on Ssym) | −0.85 | **+0.09** | — | — | — | — |
| **SAAFEC-SEQ** ⚠ | ML-classical | — | 0.36 ⚠ | −0.01 | 0.490 ⚠ | 0.71 / −0.39 | +0.58 | −1.84 | — | — | — | — |
| **DeepDDG** | DL, NN on struct. features | — | **[U]** 0.48–0.56 self-reported | — | — | — | — | — | — | — | — | — |
| **ThermoNet** | DL, 3D CNN | 0.27 / 0.33 / 1.20 | 0.39 | 0.38 | 0.554 | 0.47 / 0.47 | −0.96 | −0.01 | 0.36* / 0.37* | — | — | 8–13 s/res + huge preprocessing |
| **ACDC-NN** | DL, antisymmetric CNN | 0.45 / 0.52 / 1.05 | 0.46 | 0.45 | 0.604 | 0.58 / 0.55 | −0.98 | −0.02 | 0.51* / 0.57* | — | — | 1.4–1.6 s/residue |
| **ACDC-NN-Seq** | DL, seq | 0.44 / 0.48 / 1.08 | 0.42 | 0.42 | 0.570 | 0.55 / 0.55 | −0.99 | −0.01 | 0.48* / 0.54* | — | — | — |
| **RaSP** | DL 3D CNN, **distils Rosetta** | 0.67 / 0.71 / 1.08 retrained; **zero-shot r 0.62** on 164k | 0.39 | — | — | 0.58 / 0.18 (Ssym+) | — | — | 0.44 / 0.47 | — | — | **0.4 s/residue (V100)**, 480–1036× Rosetta |
| **PROSTATA** | DL, ESM2 fine-tune | 0.59 / 0.64 / 0.83 retrained | 0.49 | 0.49 | — | 0.51 / 0.50 | ≈−1 | ≈0 | 0.55 / 0.59 | — | — | 306 GPU-h for one 317-aa SSM+doubles |
| **DDMut** | DL, Siamese + transformer | — | **[U]** (S1304 pooled r 0.62, ρ 0.61) | — | — | **[U]** | −0.93 (own CV) | — | — | **[U]** | — | — |
| **ThermoMPNN** | DL GNN, ProteinMPNN transfer | **0.725 ± 0.003 / 0.754 ± 0.004 / 0.708** | 0.43 (HF-trained) · ρ 0.44 | — | ρ 0.61 | 0.72 / 0.60 | — | — | **0.657 / 0.650** | **0.50** (0.57 w/o Zn) | — | SSM 2 s (105 aa) / 8 s (693 aa), 1 GPU |
| **ThermoMPNN-D** | DL, Siamese doubles | Megascale-D r 0.54 / ρ 0.55 | — | — | — | — | — | — | — | — | — | — |
| **Stability Oracle** | DL graph transformer | — (T2837 r 0.59 / ρ 0.62) | 0.52 (ρ 0.53) | 0.52 | 0.618 (ρ 0.63) | **0.72 / 0.72** | −0.57 | −0.50 | — | — | — | **~50 ms per residue** |
| **Mutate Everything** | DL, AlphaFold/ESM2 decoder | cDNA split; T2837 ρ 0.65 | **ρ 0.56** | ρ 0.49 | — | — | — | — | — | — | **0.52** (its own 7-protein subset) | all singles+doubles of a 317-aa protein in **0.6 s** (ESM2) / 12.1 s (AF) |
| **ABYSSAL** [P] | DL, ESM2 + light attn. | 0.73 / 0.77 own 35%-id split | 0.50 own (420) · 0.37 per ThermoMPNN | — | — | 0.46 / 0.44 | **−0.98** | +0.02 | — | — | — | — |
| **ESM-therm** | DL, ESM2 fine-tune | ρ 0.65 own split (100,794) | — | — | — | — | — | — | — | — | — | — |
| **Pythia** | DL GNN, **zero-shot, 1.3 M params** | **ρ 0.602 pooled / 0.620 per-domain / r 0.633** (no split; 177,315 muts, 181 natural domains) | **ρ 0.66** | — | — | **[U]** | — | — | — | — | — | **25.2 M mutations in 3 min on one RTX 4090**; 10⁵× force fields |
| **HERMES** (PR 0.50 + FT) [P] | DL SO(3)-equivariant, fine-tuned on **Rosetta** labels | — | 0.39 pooled / 0.50 per-structure | — | — | **0.64 / 0.46** | — | — | — | — | — | — |
| **JanusDDG** | DL, seq-only, antisymmetric by construction | not evaluated | **0.55** (total 0.69) | 0.55 | **0.69** (ρ 0.66) | 0.84 / 0.84 (non-blind) | **−1 exactly** | **0.00** | — | — | — | — |
| **SPURS** | hybrid, ESM rewired into ProteinMPNN | **ρ 0.74 / r 0.78** (per-protein median ρ **0.83**) | ρ 0.50 / r 0.46 | — | ρ 0.66 / r 0.64 | ρ 0.76 / 0.68 | — | — | **ρ 0.67 / r 0.66** | **0.54** | excluded by design | all singles of 118 proteins (mean 492 aa) in **<20 s** on one A40 |
| **Constraint-aware opt.** [P] | SPURS backbone + 3 losses | ρ 0.749 baseline → **0.713** full | ρ **0.540** | — | ρ **0.711** | ρ 0.726 / 0.608 | still biased +0.40 kcal/mol | — | ρ 0.650 | — | — | — |
| **UniStab** | DL, ESM2-3B + LoRA folding trunk | ρ 0.67 / r 0.75 (**own 25%-id split**, 54,600 test) | — | — | — | — | — | — | — | — | — | — |
| **ESM-MSR** [P] | DL, ESM3 + LoRA dual-view | own split, 43,700 muts | — | — | uses S461 not S669 | best antisymmetry −0.87, bias −0.160 | −0.87 | −0.160 | — | **0.573** (best published) | — | Domainome in ~1 h (RTX 5090); WT-only ≈1000× faster |
| **ProteinMPNN** zero-shot | DL inverse folding, unsupervised | **0.487 ± 0.006 / 0.43 / 1.295**; 0.61 pooled (SPURS); 0.41 (SPURS own run) | 0.26–0.34 | — | ρ 0.50 | 0.26 / 0.56 (ThermoMPNN) · 0.57 / 0.40 (JanusDDG) | −0.58 | −1.40 | 0.49 / 0.41 · 0.508 | — | 0.565 | 9,800 res/s w/ tied decoding |
| **ProteinMPNN-ddG** | DL inverse folding + unfolded-state reference | Success@10 **77%**, AUROC 0.82 (164k set) | AUROC 0.71 | — | — | — | — | — | — | — | — | **9,800 res/s**; human proteome in 30 min on 1 V100 |
| **ESM-IF1** zero-shot | DL inverse folding | Success@10 64%, AUROC 0.77 | AUROC 0.63 | — | ρ 0.41 | 0.54 / 0.15 | −0.17 | −0.03 | — | — | **0.624** | — |
| **ESM-1v** zero-shot | PLM | — | — | — | ρ 0.43 | 0.10 / 0.10 | −1.00 (seq-only) | 0.00 | — | **0.48** | 0.477 | — |
| **ESM2 650M / 3B / 15B** | PLM | **0.46 pooled (SPURS)**; 0.33 (SPURS own run); 0.36–0.43 (Chu split) | — | — | ρ 0.44 | 0.27 / 0.27 | −1.00 | 0.00 | 0.43 | — | 0.523 / 0.509 / 0.488 | — |
| **SaProt 650M** | PLM + 3Di tokens | **[U]** | — | — | — | — | — | — | — | — | 0.592 | — |
| **ProSST-2048** | PLM + structure tokens | **[U]** | — | — | — | — | — | — | 0.399 → 0.590 w/ AF trunk | — | **0.653** | — |
| **ESM3 (1.4B)** | PLM multimodal | **[U]** | — | — | — | — | — | — | — | — | 0.641 | — |
| **MIF / MIF-ST** | DL structure-aware LM | — | — | — | ρ 0.46 / 0.38 | 0.56 / 0.35 | −0.44 | −3.46 | — | — | 0.522 / 0.485 | — |
| **GEMME / EVE / TranceptEVE** | MSA / generative | — | — | — | — | — | — | — | — | EVE 0.48 | 0.519 / 0.491 / 0.500 | — |
| **Kermut** (supervised, per-assay CV) | DL GP, struct+seq | — | — | — | — | — | — | — | — | — | **0.817** | — |
| **"Foldetta"** = mean(FoldX, Rosetta) | physics consensus | — | 0.48 on S612/AF | — | — | — | — | — | — | — | — | — |
| **RSA of WT residue** | trivial baseline | **0.468 / 0.481** (this project) | 0.235 **[D]** | — | **0.456** **[D]** | — | — | — | 0.314 (this project) | — | — | ms |
| **hydrophobicity × burial** | trivial baseline | 0.299 (this project) | 0.239 **[D]** | — | 0.395 **[D]** | — | — | — | 0.209 (this project) | — | — | ms |
| **−\|Δvolume\| × burial** | trivial baseline | 0.343 (this project) | 0.232 **[D]** | — | 0.420 **[D]** | — | — | — | 0.157 (this project) | — | — | ms |
| **BLOSUM62** | trivial baseline | 0.159 (this project) | 0.030 **[D]** | — | 0.054 **[D]** | — | — | — | 0.118 (this project) | — | — | ms |
| **substitution-type lookup (ddMBC)** | trivial baseline | — | — | — | 0.39 | — | — | — | — | — | 0.269 (one-hot, supervised) | ms |

⚠ data-quality problem in the published value, see §7. `*` inflated by FireProt homologues in that tool's training set.
"this project" rows are our own runs (`results/baselines/trivial_*_metrics_test.csv`, RESEARCH_TREE 3.2), not literature.

### 2b. AlphaFold-structure benchmark (S612) — the most relevant table for PhysDDG's AF mode

Degn, Utichi, Sánchez-Izquierdo Besora, Tiberti, Papaleo 2025 [P], bioRxiv 10.1101/2025.03.28.645695, Table 1.
S612 = the 612 S669 variants with an AF-DB model and pLDDT > 70. **All predictions run on AlphaFold2 structures.**
Second block = the 332 variants with <25% identity to ThermoMPNN's training set.

| tool | S612 PCC / SCC / MAE | 332-variant subset PCC / SCC / MAE |
|---|---|---|
| MutateX (FoldX) | 0.44 / 0.50 / 1.20 | 0.33 / 0.37 / 1.50 |
| Rosetta cartesian | 0.44 / 0.47 / 1.17 | 0.31 / 0.34 / 1.46 |
| RaSP | 0.42 / 0.45 / 1.14 | 0.30 / 0.34 / 1.40 |
| DDGun seq | 0.29 / 0.31 / 1.31 | 0.13 / 0.16 / — ⚠ |
| DDGun3D | 0.38 / 0.37 / 1.16 | 0.16 / 0.20 / 1.38 |
| **ThermoMPNN** | **0.51 / 0.58 / 0.96** | **0.36 / 0.44 / 1.15** |
| "foldetta" (mean FoldX+Rosetta) | 0.48 | 0.35 |
| "foldaSP" (mean FoldX+RaSP) | 0.47 | 0.34 |

Two findings to design against: (i) an MD ensemble as input beats a single static AF model, and **FoldX gains most**
(+0.07 PCC on their CPA2 case study, ThermoMPNN +0.06); (ii) coarse-grained (CABS-flex) ensembles *degrade* results.
RaSP independently shows crystal-vs-AF2 ΔΔG agreement of r 0.89–0.97 overall but only **0.66–0.96 at pLDDT < 0.9**
(eLife Table 3) — i.e. the AF penalty is concentrated in low-confidence regions, which is where a pLDDT-aware mode pays.

---

## 3. The three reference tables, reproduced in full

### 3a. S669 — Pancotti et al. 2022, Brief Bioinform 23(2):bbab555, Table 1 (21 tools)

Pearson r / RMSE / MAE. "Total" = direct + reverse pooled (n=1,338). Reverse structures were modelled with Robetta.
Rosetta here is **ddg_monomer** (v2021.16), FoldX is BuildModel with defaults.

| method | input | Total r / RMSE / MAE | Direct r / RMSE / MAE | Reverse r / RMSE / MAE | r_{d−r} | ⟨δ⟩ |
|---|---|---|---|---|---|---|
| ACDC-NN | 3D | 0.61 / 1.50 / 1.05 | 0.46 / 1.49 / 1.05 | 0.45 / 1.50 / 1.06 | −0.98 | −0.02 |
| DDGun3D | 3D | 0.57 / 1.61 / 1.13 | 0.43 / 1.60 / 1.11 | 0.41 / 1.62 / 1.14 | −0.97 | −0.05 |
| PremPS | 3D | 0.62 / 1.49 / 1.07 | 0.41 / 1.50 / 1.08 | 0.42 / 1.49 / 1.05 | −0.85 | +0.09 |
| ThermoNet | 3D | 0.51 / 1.64 / 1.20 | 0.39 / 1.62 / 1.17 | 0.38 / 1.66 / 1.23 | −0.85 | −0.05 |
| Rosetta (ddg_monomer) | 3D | 0.47 / 2.69 / 2.05 | 0.39 / 2.70 / 2.08 | 0.40 / 2.68 / 2.02 | −0.72 | −0.61 |
| DynaMut | 3D | 0.50 / 1.65 / 1.21 | 0.41 / 1.60 / 1.19 | 0.34 / 1.69 / 1.24 | −0.58 | −0.06 |
| INPS3D | 3D | 0.55 / 1.64 / 1.19 | 0.43 / 1.50 / 1.07 | 0.33 / 1.77 / 1.31 | −0.50 | −0.38 |
| SDM | 3D | 0.32 / 1.93 / 1.45 | 0.41 / 1.67 / 1.26 | 0.13 / 2.16 / 1.64 | −0.40 | −0.40 |
| PoPMuSiC v3.1 | 3D | 0.46 / 1.82 / 1.37 | 0.41 / 1.51 / 1.09 | 0.24 / 2.09 / 1.64 | −0.32 | −0.69 |
| MAESTRO | 3D | 0.44 / 1.80 / 1.30 | **0.50** / 1.44 / 1.06 | 0.20 / 2.10 / 1.655 | −0.22 | −0.57 |
| FoldX | 3D | 0.31 / 2.39 / 1.53 | 0.22 / 2.30 / 1.56 | 0.22 / 2.48 / 1.50 | −0.20 | −0.34 |
| DUET | 3D | 0.41 / 1.86 / 1.39 | 0.41 / 1.52 / 1.10 | 0.23 / 2.14 / 1.68 | −0.12 | −0.67 |
| I-Mutant3.0 | 3D | 0.32 / 1.96 / 1.49 | 0.36 / 1.52 / 1.12 | 0.15 / 2.32 / 1.87 | −0.06 | −0.81 |
| mCSM | 3D | 0.37 / 1.96 / 1.49 | 0.36 / 1.54 / 1.13 | 0.22 / 2.30 / 1.86 | −0.05 | −0.85 |
| DynaMut2 | 3D | 0.36 / 1.90 / 1.42 | 0.34 / 1.58 / 1.15 | 0.17 / 2.16 / 1.69 | **+0.03** | −0.64 |
| INPS-Seq | seq | 0.61 / 1.52 / 1.10 | 0.43 / 1.52 / 1.09 | 0.43 / 1.53 / 1.10 | −1.00 | 0.00 |
| ACDC-NN-Seq | seq | 0.59 / 1.53 / 1.08 | 0.42 / 1.53 / 1.08 | 0.42 / 1.53 / 1.08 | −1.00 | 0.00 |
| DDGun | seq | 0.57 / 1.74 / 1.25 | 0.41 / 1.72 / 1.25 | 0.38 / 1.75 / 1.25 | −0.96 | −0.05 |
| I-Mutant3.0-Seq | seq | 0.37 / 1.91 / 1.47 | 0.34 / 1.54 / 1.15 | 0.22 / 2.22 / 1.79 | −0.48 | −0.76 |
| MUpro | seq | 0.32 / 2.03 / 1.58 | 0.25 / 1.61 / 1.21 | 0.20 / 2.38 / 1.96 | −0.32 | −0.95 |
| SAAFEC-SEQ ⚠ | seq | 0.26 / 2.02 / 1.54 | 0.36 / 1.54 / 1.13 | −0.01 / 2.40 / 1.94 | −0.03 | −0.83 |

Pancotti reports **Pearson only**. Spearman **[D]** from the released prediction CSV: MAESTRO 0.463, ACDC-NN 0.454,
INPS-Seq 0.440, INPS3D 0.439, ACDC-NN-Seq 0.431, DDGun3D 0.427, DDGun 0.427, PremPS 0.416, DUET 0.416,
PoPMuSiC 0.415, SDM 0.392, DynaMut 0.374, ThermoNet 0.373, mCSM 0.363, I-Mutant3.0 0.354, I-Mutant3.0-Seq 0.346,
**FoldX 0.283**, MUpro 0.255.

Later DL entrants on S669 direct, for context: PROSTATA r 0.49 · Stability Oracle r 0.52 / ρ 0.53 ·
Mutate Everything **ρ 0.56** · JanusDDG r 0.55 · DDGemb r 0.53 · SPURS ρ 0.50 · constraint-aware ρ 0.540 ·
**Pythia (zero-shot) ρ 0.66** · ThermoMPNN r 0.43 / ρ 0.44.

### 3b. Ssym (684 = 342 direct + 342 reverse) — Pucci, Bernaerts, Kwasigroch, Rooman 2018, Bioinformatics 34:3659, Table 1

σ = RMSE (kcal/mol). Read from the bioRxiv version 10.1101/308239 (journal version paywalled).

| method | σ_dir | r_dir | σ_inv | r_inv | r_{d−inv} | ⟨δ⟩ |
|---|---|---|---|---|---|---|
| PoPMuSiC-sym | 1.58 | 0.48 | 1.62 | 0.48 | −0.77 | +0.03 |
| MAESTRO | 1.36 | 0.52 | 2.09 | 0.32 | −0.34 | −0.58 |
| FoldX v3.0 | 1.56 | 0.63 | 2.13 | 0.39 | −0.38 | −0.47 |
| PoPMuSiC v2.1 | 1.21 | 0.63 | 2.18 | 0.25 | −0.29 | −0.71 |
| SDM | 1.74 | 0.51 | 2.28 | 0.32 | −0.75 | −0.32 |
| iSTABLE | 1.10 | 0.72 | 2.28 | −0.08 | −0.05 | −0.60 |
| I-Mutant 3.0 | 1.23 | 0.62 | 2.32 | −0.04 | +0.02 | −0.68 |
| NeEMO | 1.08 | 0.72 | 2.35 | 0.02 | +0.09 | −0.60 |
| DUET | 1.20 | 0.63 | 2.38 | 0.13 | −0.21 | −0.84 |
| mCSM | 1.23 | 0.61 | 2.43 | 0.14 | −0.26 | −0.91 |
| MUpro | 0.94 | **0.79** | 2.51 | 0.07 | −0.02 | −0.97 |
| STRUM | 1.05 | 0.75 | 2.51 | −0.15 | +0.34 | −0.87 |
| Rosetta (Kellogg protocol) | 2.31 | 0.69 | 2.61 | 0.43 | −0.41 | −0.69 |
| AUTO-MUTE | 1.07 | 0.73 | 2.61 | −0.01 | −0.06 | −0.99 |
| CUPSAT | 1.71 | 0.39 | 2.88 | 0.05 | −0.54 | −0.72 |

Pucci's own caveat: **every direct Ssym mutation is in the training set of these tools**, so only the inverse half is
an independent test. This is the single most important reason the direct column looks respectable for methods with
⟨δ⟩ near −1 (they simply predict "destabilising" for everything, which is right 80% of the time).

### 3c. Ssym — KORPM (Hernández, Dehouck, Bastolla, López-Blanco, Chacón 2023, Bioinformatics 39:btad011), Table 1 + SI S4/S5

~25 Ssym ΔΔG values corrected; cartesian_ddg and FoldX predictions capped at −8 kcal/mol. **rSym here is |r_{d−r}|,
so +1 is perfect antisymmetry** — opposite sign convention to §3b.

| method | Total RMSE | Total PCC | rSym | δ | dir RMSE / PCC | rev RMSE / PCC |
|---|---|---|---|---|---|---|
| **KORPM** | **1.33** | **0.69** | **0.88** | −0.15 | 1.28 / 0.57 | 1.38 / 0.49 |
| Rosetta cartesian_ddg | 3.44 | 0.63 | 0.41 | −3.12 | 3.32 / 0.68 | 3.55 / 0.45 |
| FoldX v3.0 | 1.86 | 0.54 | 0.27 | −1.25 | 1.48 / 0.65 | 2.18 / 0.38 |
| EvoEF2 ("EvoFF") | 1.56 | 0.54 | 0.58 | −0.36 | 1.36 / 0.57 | 1.75 / 0.31 |
| PoPMuSiC-sym | 1.58 | 0.52 | 0.77 | −0.06 | 1.56 / 0.47 | 1.60 / 0.47 |
| DynaMut2 | 1.88 | 0.38 | 0.11 | −1.56 | 1.20 / 0.63 | 2.38 / 0.05 |
| DDGun3D | 1.43 | 0.63 | **0.99** | −0.04 | 1.42 / 0.55 | 1.45 / 0.53 |
| ThermoNet | 1.53 | 0.55 | 0.96 | −0.01 | 1.52 / 0.47 | 1.53 / 0.47 |
| ACDC-NN | 1.38 | **0.69** | 0.98 | −0.05 | 1.35 / 0.61 | 1.41 / 0.58 |

### 3d. S461 — direct Pearson / Spearman, assembled from KORPM SI S7, Rossi 2025 (Protein Sci 34:e70134) Table S4 and JanusDDG Suppl. Table 5

| method | Pearson | Spearman | RMSE |
|---|---|---|---|
| **JanusDDG** | 0.69 | **0.66** | 0.97 |
| **constraint-aware opt.** [P] | — | **0.711** | — |
| SPURS | 0.64 | 0.66 | — |
| DDGun3D (untrained!) | **0.635** | 0.58 | 1.11 |
| PremPS | 0.632 | 0.60 | 1.028 |
| MAESTRO | 0.630 | 0.60 | 1.043 |
| Stability Oracle | 0.618 | 0.63 | 1.188 |
| INPS3D | 0.615 | 0.59 | 1.012 |
| ThermoMPNN | 0.58 | 0.61 | — |
| PoPMuSiC-sym | 0.609 | 0.60 | 1.023 |
| ACDC-NN | 0.604 | 0.56 | 1.065 |
| Rosetta cartesian_ddg (KORPM run, capped) | 0.597 | 0.61 | 3.59 |
| cartesian_ddg (Reeves run, uncapped) | 0.590 | — | 4.651 |
| DUET | 0.593 | 0.57 | 1.061 |
| DDGun (seq) | 0.583 **[D]** | — | — |
| KORPM | 0.569 | 0.54 | 1.21 |
| ACDC-NN-Seq | 0.570 | — | 1.101 |
| INPS-Seq | 0.555 | — | 1.103 |
| SDM | 0.555 | 0.51 | 1.333 |
| ThermoNet | 0.554 | 0.48 | 1.24 |
| mCSM | 0.538 | 0.51 | 1.069 |
| ProteinMPNN (zero-shot) | 0.40 | 0.50 | 2.36 |
| DynaMut ⚠ | 0.504 | 0.43 | 1.27 |
| SAAFEC-SEQ / I-Mutant3.0 ⚠ | 0.490 | 0.47 | 1.121 |
| EvoEF2 | 0.463 | — | 1.28 |
| MIF | 0.45 | 0.46 | 4.37 |
| ESM2-650M | 0.43 | 0.44 | 4.41 |
| ESM-1v mean | 0.39 | 0.43 | 4.29 |
| **RSA alone [D]** | 0.422 | **0.456** | — |
| ESM-IF multimer | 0.37 | 0.41 | 1.64 |
| MUpro | 0.396 | — | 1.168 |
| ddMBC (per-substitution-type ridge) | 0.39 | — | 1.38 |
| FoldX + mass-balance correction | 0.38–0.399 | — | 1.34–1.44 |
| FoldX | 0.30 (KORPM run) / 0.221 (Pancotti run) | 0.39 | 1.91 / 2.229 |

**S461 is systematically easier than S669** — the same model typically gains 0.15–0.21 Spearman (SPURS 0.66 vs 0.50;
constraint-aware 0.711 vs 0.540). Do not compare an S461 number to an S669 number.

---

## 4. Method notes (one paragraph each)

### 4.1 Physics and statistical

**FoldX** (Schymkowitz et al. 2005 NAR 33:W382; v5 current). Empirical force field with ~10 weighted terms fitted to
experimental ΔΔG; BuildModel/PositionScan repack side chains on a fixed backbone. The reference method to beat.
Characteristic failure modes, all documented: strong destabilising bias (Ssym ⟨δ⟩ −0.47 to −1.25, S669 ⟨δ⟩ −0.34),
poor antisymmetry (r_{d−r} only −0.20 on S669), unbounded overprediction (RMSE 2.3–3.9 kcal/mol uncalibrated),
sensitivity to input structure (prefers X-ray over NMR; Pancotti Fig. 3), and slowness (395–807 s per saturated
residue on 64 cores, RaSP eLife Table 4; ~1 mutation/min/core per Pythia; ~210 s/mutation on SKEMPI per StaB-ddG).
It is nevertheless the **best** classical method on two tasks: ranking in pooled Megascale (SCC 0.57, better than
cartesian_ddg's 0.56 despite a worse Pearson), and SKEMPI binding ΔΔG, where it still beats every DL method except
an ensemble (StaB-ddG Table 7: per-interface Pearson 0.49).

**Rosetta cartesian_ddg** (Park et al. 2016 JCTC 12:6201, PMC5515585; `ddg_monomer` is the older Kellogg 2011
protocol). Cartesian-space minimisation with the beta_nov16 / ref2015 energy function. Park reports Pearson 0.743
on the Kellogg 1,211-mutation set; Frenz et al. 2020 (Front Bioeng Biotechnol 8:558247) report r 0.49 and 59.1%
three-class accuracy on a 767-mutation ProTherm set, vs 0.25 / 51.7% for ddg_monomer "Protocol 3". On blind modern
sets it lands at S669 r 0.39, S461 r 0.59–0.60, Ssym 0.68 direct / 0.45 inverse. Its two systematic problems are a
huge RMSE (REU scale, ~10% of cases blow up on clashes — KORPM had to cap at −8 kcal/mol) and cost
(192–414 s per saturated residue on 64 cores).

**EvoEF2** (Huang et al. 2020). Empirical force field in the FoldX family. Ssym total PCC 0.54 with much better
antisymmetry than FoldX (|r_{d−r}| 0.58 vs 0.27) and a far smaller bias (−0.36 vs −1.25); S461 r 0.463.

**KORPM** — Hernández IM, Dehouck Y, Bastolla U, López-Blanco JR, Chacón P. *Bioinformatics* 39(1):btad011 (2023),
PMC9850275, DOI 10.1093/bioinformatics/btad011. **The closest published precedent for PhysDDG and the model to
study hardest.** A 6D orientational knowledge-based potential (KORP) over backbone-only residue pairs
(Cα–Cα distance + two polar angles + one torsion), with **only 12 amino-acid-group weights** fitted by 10-fold CV
with 25% homology exclusion. Almost antisymmetric by construction (|r_{d−r}| 0.88), smallest bias of any physics
method (⟨δ⟩ −0.15), best Ssym total PCC 0.69 and lowest RMSE 1.33 of the nine methods they compared, and
**684 mutations in 3.5 s on a single i7 core**. On S461 it is mid-pack (0.569), behind the untrained DDGun3D.
Its stated failures are the ones a packing-free backbone potential must have: disulfides (C3T in 1ITM predicted
0.08 vs −5 kcal/mol) and salt bridges (D303A in 2M5S, −0.18 vs −4.0). Sign convention: positive = **stabilising**.

**PoPMuSiC / PoPMuSiC-sym** (Dehouck et al. 2009 Bioinformatics 25:2537; Pucci et al. 2018 for the symmetrised
version). Statistical potentials combined by a small neural network with solvent-accessibility-dependent weights.
PoPMuSiC-sym is explicitly symmetrised and pays for it: Ssym direct drops 0.63 → 0.48 but inverse rises 0.25 → 0.48,
and ⟨δ⟩ goes from −0.71 to +0.03. The cleanest demonstration in the literature that the direct-only score of an
asymmetric predictor is partly a bias artefact. S461 Pearson 0.609 / RMSE 1.02 (lowest RMSE in the KORPM table).

**DDGun / DDGun3D** — Montanucci, Capriotti, Birolo, Benevenuta, Pancotti, Lal, Fariselli. *BMC Bioinformatics*
20:335 (2019), PMC6606456; web server *NAR* 50:W222 (2022), PMC9252764. An **untrained** linear combination of a
profile-weighted BLOSUM62 score, Kyte-Doolittle hydrophobicity difference and a Skolnick statistical potential;
DDGun3D adds the Bastolla-Vendruscolo contact potential modulated by accessibility. The authors propose it
explicitly as a baseline that new methods must beat. It is **still the best or near-best classical method on
S461 (Pearson 0.635)** and has essentially perfect antisymmetry (−0.97 to −0.99) with near-zero bias. Single-score
components, |Pearson| on S2648 / VariBench / Broom / p53 / myoglobin (their Table 1): BLOSUM62 0.284 / 0.269 /
0.354 / 0.636 / 0.148; Kyte-Doolittle 0.298 / 0.248 / 0.263 / 0.143 / 0.500; Skolnick 0.387 / 0.398 / 0.423 /
0.328 / 0.454; Bastolla-Vendruscolo 0.497 / 0.452 / 0.581 / 0.423 / 0.548. **A single contact potential reaches
Pearson ~0.5 on S2648.** That is the bar a physics model must clear before any of its machinery pays for itself.

**MAESTRO, mCSM, SDM, DUET, DynaMut, DynaMut2, CUPSAT, I-Mutant, MUpro, PremPS, INPS, SAAFEC-SEQ.** Classical ML on
hand-made features. Their defining property is a destabilising bias: ⟨δ⟩ from −0.57 to −0.99 on Ssym, inverse-mutation
Pearson collapsing to 0.07 (MUpro), −0.04 (I-Mutant3.0) or −0.15 (STRUM) while direct Pearson looks like 0.79, 0.62,
0.75. MUpro's Ssym direct 0.79 is the highest number in §3b and is almost entirely bias. PremPS (Chen et al. 2020
*PLoS Comput Biol* 16:e1008543, PMC7802934) is the exception: a random forest trained on S2648 **plus all modelled
reverse mutations**, giving r_{d−r} −0.85 to −0.92 and the only positive ⟨δ⟩ (+0.09) in the S669 table; it is also
known to have been fit on all of Ssym, so its Ssym+ 0.81/0.73 is not blind.

### 4.2 Deep learning on structure

**ThermoNet** — Li B, Yang YT, Capra JA, Gerstein MB. *PLoS Comput Biol* 16(11):e1008291 (2020), PMC7728386.
Ensemble of ten 3D CNNs on 16×16×16 Å, 7-channel voxel grids; mutant structures built with Rosetta FastRelax.
Trained on **Q3488 = Q1744 + its reverse mutations**, after removing Ssym members and BLAST homologues. The
ablation is the key result: training on direct mutations only gives Ssym direct r 0.47 but **reverse r −0.06**.
Adding reverses costs nothing on direct and fixes the reverse (0.47 / 0.47, r_{d−r} −0.96). They also document the
leakage that motivates all later splits: S2648 shares 14 identical proteins and 8 homologues with Ssym. Slowest
method in the RaSP timing table by preprocessing (80,442 s for a 107-residue protein).

**ACDC-NN / ACDC-NN-Seq** — Benevenuta S, Pancotti C, Fariselli P, Birolo G, Sanavia T. *J Phys D* 54:245403 (2021),
DOI 10.1088/1361-6463/abedfb; sequence version Pancotti et al. *Genes* 12:911 (2021), PMC8231498. A differential
Siamese CNN whose final layer outputs (ΔΔG_dir − ΔΔG_inv)/2, so antisymmetry is **architectural, not learned**.
Pretrained on ~600k DDGun3D-labelled synthetic variants, then transfer-learned on S2648+VariBench with <25% identity
to Ssym. Best combination of accuracy and antisymmetry among pre-2023 methods (S669 direct 0.46, reverse 0.45,
r_{d−r} −0.98). The paper also states a useful number: the expected Pearson **upper bound on ProTherm-derived data
is 0.70–0.85** given experimental uncertainty.

**RaSP** — Blaabjerg LM et al. *eLife* 12:e82593 (2023), PMC10266766, DOI 10.7554/eLife.82593. Two stages: a
self-supervised 3D CNN on masked local atomic environments (2,099 PISCES structures), then a small MLP trained to
reproduce **Rosetta cartesian_ddg**, not experiment. It therefore inherits Rosetta's accuracy and loses Rosetta's
cost: **0.4 s per saturated residue, independent of protein length, 480–1,036× faster than Rosetta**. Zero-shot on
Tsuboyama (164,524 variants, 164 natural domains) it reaches Pearson 0.62 / MAE 0.94 — for a model that never saw
an experimental ΔΔG, this is the strongest single argument that distilling a physics function is viable. Its
antisymmetry is terrible (Ssym+ direct 0.58, reverse 0.18) and the authors attribute this to one-sided training.
Crystal-vs-AF2 ΔΔG agreement r 0.89–0.97, dropping to 0.66–0.96 for pLDDT < 0.9 (Table 3).

**ThermoMPNN** — Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. *PNAS* 121(6):e2314853121 (2024),
PMC10861915, DOI 10.1073/pnas.2314853121. A 2.7M-parameter light-attention MLP head on **frozen ProteinMPNN**
embeddings (v_48_020), trained on 272,712 Megascale single substitutions. Defines the split everyone now uses.
Megascale test 0.725 SCC / 0.754 PCC / 0.708 RMSE; FireProt-HF 0.657 / 0.650 / 1.51. Two numbers from its ablation
matter: pretrained ProteinMPNN embeddings alone give SCC 0.487, and training from naive weights drops to 0.642 —
so most of the 0.725 comes from the Megascale labels, and a sizeable chunk from the PDB-scale pretraining.
Trained in 18 h on one V100; inference 2–8 s per site-saturation scan.

**ThermoMPNN-D** — Dieckhaus H, Kuhlman B. "Protein stability models fail to capture epistatic interactions of
double point mutations." *Protein Science* 34(1):e70003 (2025), PMC11659742, DOI 10.1002/pro.70003; preprint
10.1101/2024.08.20.608844. Siamese double-mutant extension with "over-and-back" augmentation. Its headline is a
**negative result**: on Megascale-D, simply summing two predicted single ΔΔG scores (SCC 0.59) beats the
epistasis-aware model (0.55). Only Rosetta and ESM-1v improved when run in epistatic rather than additive mode.
Error propagation from singles to doubles follows the random-additive-error trajectory for every method **except
FoldX and DDGun**, which track the identity line. Final ensemble: Megascale-D r 0.54 / ρ 0.55 / RMSE 1.07;
PTMUL-D r 0.57 / ρ 0.59 / RMSE 1.95.

**Stability Oracle** — Diaz DJ et al. *Nature Communications* **15:6170** (2024), PMC11266546,
DOI 10.1038/s41467-024-49780-2. (Not 15:7407.) Graph transformer over atoms with a pairwise-distance attention
bias, plus a Siamese head on two "structural amino-acid embeddings"; one masked microenvironment yields all 380
mutation types without a mutant structure. ~1.2M parameters, ~50 ms per residue. Its lasting contributions are
(i) **thermodynamic permutations** — from n measured ΔΔG at one position, generate n(n−1) valid mutations that are
never *to* wild type, avoiding the leakage that thermodynamic reversal introduces; and (ii) a direct leakage audit:
retraining PROSTATA on its own 75%-filtered sets leaves 93/93 S669 proteins and 15/15 Ssym proteins leaked, giving
an absurd Pearson of **0.98**; train on the clean Q1744 instead and it is 0.49 / 0.51. Ssym 0.72 direct **and**
0.72 inverse; S669 0.52; T2837 Pearson 0.59 / Spearman 0.62.

**HERMES** [P] — Visani GM et al., arXiv 2407.06703 / bioRxiv 10.1101/2024.07.09.602403 (2024); NeurIPS 2024 MLSB
*workshop*, no journal version as of 2026-10-26. SO(3)-equivariant network over 10 Å atomic neighbourhoods,
pretrained on masked residue identity, then **fine-tuned on Rosetta ΔΔG** (so its S669/Ssym numbers are a transfer
result, not supervision on experiment). Best config (PyRosetta preprocessing, 0.50 Å noise, fine-tuned): S669
pooled 0.39 / per-structure 0.50 ± 0.07; Ssym direct 0.64, reverse 0.46 — a much smaller direct/reverse gap than
RaSP+FT (0.58 / 0.18). It states the per-structure vs pooled inversion explicitly: per-structure > pooled on S669,
the reverse on Ssym-direct.

**Pythia** — Sun J, Zhu T, Cui Y, Wu B. "Structure-based self-supervised learning enables ultrafast protein
stability prediction upon mutation." *The Innovation* 6(1):100750 (2025), PMC11763918, DOI 10.1016/j.xinn.2024.100750;
preprint bioRxiv 10.1101/2023.08.09.552725. **The single most relevant DL comparator for a fast physics method.**
A 1.3M-parameter masked-residue GNN, **zero-shot** (no ΔΔG labels at all, hence no label leakage), reaching
S669 Spearman **0.66** (best in its comparison, beating 9 supervised ML and 4 statistical methods on every metric),
S2648 ρ 0.616 / r 0.598, and Megascale ρ 0.602 pooled / 0.620 per-domain / r 0.633 over 177,315 mutations in 181
natural domains. Speed: **25,189,782 mutations across 4,214 E. coli structures in 3 minutes on one RTX 4090**,
~10⁵× force fields. If a zero-shot 1.3M-parameter network gets S669 ρ 0.66 from structure alone, the ceiling for a
fitted physics model on that benchmark is higher than the classical table suggests.

**Mutate Everything** — Ouyang-Zhang J, Diaz DJ, Klivans A, Krähenbühl P. "Predicting a Protein's Stability under a
Million Mutations." *NeurIPS 2023*, arXiv 2310.12979. A light decoder on a frozen/fine-tuned AlphaFold (OpenFold) or
ESM2 trunk predicting **all single and double mutations in one forward pass**; the double head learns the residual
to the additive sum. S669 forward Spearman **0.56** (SOTA at publication), reverse 0.49; T2837 forward ρ 0.65.
No reverse-mutation augmentation in training. Runtime: all singles + doubles of a 317-residue protein in **0.6 s**
(ESM2 trunk) vs 306 GPU-hours for pairwise PROSTATA. Note the AlphaFold trunk beats ESM2 (S669 ρ 0.56 vs 0.47)
and an AlphaFold trunk trained from scratch collapses to 0.36 — the pretraining, not the architecture, carries it.

### 4.3 Sequence / PLM

**PROSTATA** — Umerenkov D et al. *Bioinformatics* 39(11):btad671 (2023), PMC10651431. ESM-2 650M fine-tuned
end-to-end on WT and mutant sequences, ensemble of five regression heads; **every training mutation is mirrored**.
5-fold CV on its own set r 0.69; S669 direct and reverse both 0.49. Its symmetry experiment is instructive: trained
on the non-symmetric Q3421, the outer-product head gives Ssym direct 0.55 but reverse **−0.29**; trained on the
symmetrised Q3488 it gives 0.42 / 0.44. Linear-merge heads learn antisymmetry even from biased data; bilinear ones
do not. See the Stability Oracle leakage audit above before quoting any PROSTATA number.

**JanusDDG** — Barducci G, Rossi I, Codicè F, Rollo C, Repetto V, Pancotti C, Iannibelli V, Sanavia T, Fariselli P.
*Communications Biology* (2026), DOI 10.1038/s42003-026-09632-9, title "JanusDDG: a physics-informed neural network
for sequence-based protein stability via two-fronts attention"; preprint arXiv 2504.03278. Sequence-only PLM
embeddings into a bidirectional cross-interleaved attention transformer, with a Siamese construction that makes
r_{d−r} = −1 and ⟨δ⟩ = 0 **exact**. Trained on S2450 (a re-cleaned S2648) doubled by antisymmetry. Best published
S669 direct Pearson for a blind method (0.55, total 0.69) and S461 Pearson 0.69 / Spearman 0.66. Its Ssym 0.84/0.84
is self-declared **not blind**. Its supplementary tables 4, 5 and 8 are the most complete modern cross-method
compilations and were the source for several §3d rows.

**ABYSSAL** [P] — Pak MA, Dovidchenko NV, Sharma SM, Ivankov DN. bioRxiv 10.1101/2022.12.31.522396 (2023). ESM2 +
light attention on 367,858 Megascale mutations split at 35% identity, **with reverse mutations in training**.
MegaTest symmetric PCC 0.85 (forward-only 0.77), antisymmetry −0.98, ⟨δ⟩ 0.01. Two numbers worth keeping: (i) the
MegaValidation correlation **plateaus at 0.85 from ~87,773 training mutations** (24% of the training set) and a
model trained on only 2,441 still reaches 0.80 — i.e. Megascale's size is not what buys the performance; (ii) the
same model gets only 0.50 on S669. Their explanation is that proteolysis-derived ΔΔG is a proxy with its own
systematics. **Conflict to flag:** ThermoMPNN Table 3 reports ABYSSAL S669 0.37 where the ABYSSAL preprint reports
0.50 on the same 420-variant subset. Do not average them.

**ESM-therm** — Chu SKS, Narang K, Siegel JB. *PLoS Comput Biol* 20(7):e1012248 (2024), PMC11293664. ESM2 fine-tuned
on Megascale (own split, 100,794 test mutants). Spearman 0.65, vs RaSP 0.64, ELASPIC-2 0.64, Rosetta 0.61, ESM-2 3B
zero-shot 0.43, MUpro 0.31. The generalisation result is the interesting one: on six single-protein stability DMS
sets of 177–510 residues, ESM-therm collapses to 0.03–0.06 while **Rosetta holds 0.33–0.49 and RaSP 0.30–0.44**.
A Megascale-trained sequence model does not transfer off 40–72-residue domains; a physics function does.

**ProteinMPNN zero-shot and ProteinMPNN-ddG** — Dutton O et al. (Peptone Ltd.), "Improving Inverse Folding models at
Protein Stability Prediction without additional Training or Data", bioRxiv 10.1101/2024.06.15.599145 v5, NeurIPS
2024 MLSB workshop; no journal version found. **Directly useful to PhysDDG because the fix is a reference-state
correction, not a model.** Two changes, no training: (i) decode each residue *last* so it sees full sequence
context (sequence recovery 51.2% → 55.0%); (ii) subtract the model's prediction given only that residue's own
backbone atoms, which acts as an explicit **unfolded-state reference** and removes both the amino-acid frequency
prior (which alone correlates with the single-residue term at Spearman 0.72) and the backbone-geometry term.
Tsuboyama Success@10 rises 66% → **77%**, S669 AUROC 0.64 → 0.71. Tied decoding makes it O(N log N):
**9,800 residues/s**, the whole human proteome (23,391 AF2 structures) in 30 minutes on one V100 — vs 2.5 res/s for
RaSP, 0.0052 for Rosetta, 0.0025 for FoldX. It reports no Spearman/Pearson anywhere, only Success@10/AUROC/AUPRC.

**ESM-1v / ESM2 / ESM-IF1 / SaProt / ProSST / ESM3 zero-shot.** None of these papers reports Megascale ΔΔG
Spearman; use the ProteinGym Stability column (§5) as the proxy, remembering it is 64/66 Tsuboyama assays scored
per-domain. Verified anchors: ProteinMPNN pooled Megascale 0.61 and ESM 0.46 (SPURS Results text);
ProteinMPNN 0.487 ± 0.006 on the ThermoMPNN split; ESM2-3B 0.43 on the Chu split. The consistent finding across
Reeves & Kalyaanamoorthy (*Nat Mach Intell*, 2024, DOI 10.1038/s42256-024-00887-7) is that **inverse-folding
likelihoods are competitive with Rosetta per-protein** and that weighted-ρ cleanly separates structure-based from
sequence-based scores. ESM2 does **not** scale on stability: 650M 0.523 > 3B 0.509 > 15B 0.488 on ProteinGym.

### 4.4 2025–2026 hybrids

**SPURS** — Li Z, Luo Y. "Generalizable and scalable protein stability prediction with rewired protein generative
models." *Nature Communications* (2025), DOI 10.1038/s41467-025-67609-4; preprint 10.1101/2025.02.13.638154.
**Current SOTA on the ThermoMPNN Megascale split.** ESM layers injected as adapters *into* the ProteinMPNN encoder,
with an all-at-once decoder (O(1) forward passes instead of O(L×20)). Trained on the ThermoMPNN split with extra
filtering (>25% identity to the test split *or to any benchmark* removed from train). Megascale test
**ρ 0.74 / r 0.78 pooled, per-protein median ρ 0.83** (vs ThermoMPNN 0.72 / 0.75 / 0.77), better on 24 of 28 test
proteins. FireProt-HF ρ 0.67, Ssym 0.76 / 0.68, S669 ρ 0.50, S461 ρ 0.66, Domainome 0.54. All singles of 118
proteins (mean 492 aa) in under 20 s on one A40.

**Constraint-Aware Optimization** [P] — Shivram A, Chivukula AS, Gupta M, Chowdhury S, arXiv 2606.08100 (2026).
No architecture change to SPURS, just balanced MSE + a Siamese antisymmetry regulariser + an OOD-margin consistency
loss. S669 ρ 0.486 → **0.540**, S461 0.653 → **0.711**, but **Megascale-test ρ falls 0.749 → 0.713**: an explicit
in-distribution/OOD trade-off. The paper's most useful finding for us is negative: the Siamese antisymmetry penalty
does **not** remove directional bias — the forward over-estimate goes from +0.335 to +0.396 kcal/mol, because
(f→ + f←)² is minimised by shrinking both magnitudes, not by making them opposite.

**UniStab** — Tan H, Lin S, Xiong Y. *Chemical Science* 17:18220 (2026), DOI 10.1039/d6sc02103d. ESM2-3B + a
LoRA-adapted folding trunk ("implicit structure learning"); handles substitutions, multi-point mutations and indels
in one model. **Its own 25%-identity Megascale split** (373,325/43,070/54,600) — not comparable to 0.725. Mega
Single ρ 0.67 / r 0.75; Mega Indel 0.78; Mega Double 0.60. Two honest negatives: core–core double mutants are much
worse (PCC 0.426) than surface–surface (0.662), and in the strongly stabilising regime (ΔΔG < −2) **every** method
has negative Spearman, because Megascale's training range is roughly [−2, +5].

**ESM-MSR** [P] — Reeves S, Kalyaanamoorthy S, bioRxiv 10.64898/2026.06.04.730231 (2026). ESM3 + LoRA with separate
WT and MT forward passes. **Best published Domainome number: ρ_avg 0.573** on 536,145 mutations / 522 domains,
ahead of ESM3 and ProteinMPNN. Best Ssym antisymmetry of any method they tested (−0.87) and least bias (−0.160)
**with no antisymmetry loss or augmentation at all**. Two warnings for anyone building a Domainome comparison:
they excluded SPURS for train–test overlap, and ThermoMPNN retrained on their Megascale subset "performed
significantly worse than all others, including its own zero-shot baseline" — so the 0.49–0.50 Domainome figure is
not robust to the training subset.

**METL** — Gelman S et al. *Nature Methods* 22(9):1868 (2025), DOI 10.1038/s41592-025-02776-2. Biophysics-pretrained
(Rosetta attributes on 20M simulated variants) transformer, then fine-tuned on DMS data. **It reports no ΔΔG
benchmark at all** — no Megascale, S669, Ssym, S461, FireProt or Domainome — and belongs in a DMS-fitness
discussion, not this table. Worth citing for one thing: pretraining on Rosetta attributes recovers Rosetta total
score at mean ρ 0.91, confirming that a transformer can absorb a physics function cheaply.

**StaB-ddG** — Deng A, Householder KD, Wu F, Garcia KC, Trippe BL, *ICML 2025*, PMLR 267:13129, arXiv 2507.05502
(Stanford, not Meta). Reparameterises ProteinMPNN log-likelihood as a folding energy and composes it into a
**binding** ΔΔG that is provably antisymmetric and path-independent. Its Megascale number is a by-product
(Pearson 0.71 / Spearman 0.69, below ThermoMPNN's 0.75 / 0.73). The headline is that it is the first DL method to
match FoldX on SKEMPI (per-interface Pearson 0.49 vs 0.49) with >1,000× speed-up — and that **FoldX + StaB-ddG
ensembled beats either (0.56)**.

**Names that do not exist**, checked against arXiv/bioRxiv/PubMed/PMC/GitHub on 2026-10-26: **ThermoAGT**,
**TopoDDG**, **Stab-Former**, **ProtiGeno**, **MutRank**, **Prostata-IFML**, **DeepTherm**. "ProtDDG" is a
benchmark resource (protddg-bench.github.io), not a predictor. "ThermoFinder" is a thermophilic-protein classifier,
not a ΔΔG predictor. "ESM-Effect" (Glaser & Braegelmann, bioRxiv 10.1101/2025.02.03.635741) is real but predicts
functional effect, not ΔΔG, and reports none of these benchmarks.

---

## 5. ProteinGym stability, Domainome, and the noise ceilings

### 5a. ProteinGym Stability category — what the number actually means

Source: ProteinGym v1.3 repo, `benchmarks/DMS_zero_shot/substitutions/Spearman/Summary_performance_...csv` and
`reference_files/DMS_substitutions.csv` (Notin et al., NeurIPS 2023 D&B 36:64331; Zenodo 10.5281/zenodo.15293562).

- 217 assays in 5 coarse categories; **Stability = 66 assays, of which 64 are Tsuboyama 2023** cDNA-display
  proteolysis. Median target length **57.5 aa**. It is effectively a per-domain Megascale ranking benchmark.
- Spearman is computed per assay → aggregated by UniProt → by category → the headline `Average_Spearman` is the
  **unweighted mean of the five category means** (verified arithmetically for all 97 zero-shot and 11 supervised
  models). So Stability is 1/5 of the headline despite being 30% of the assays.

Zero-shot Stability Spearman (selected): ProSST-2048 **0.653** · VenusREM 0.650 · ESM3-open-1.4B 0.641 ·
**ESM-IF1 0.624** · SaProt-650M 0.592 · ProtSSN 0.568 · **ProteinMPNN 0.565** · GEMME 0.519 · ESM2-650M 0.523 ·
MIF 0.522 · ESM2-3B 0.509 · TranceptEVE-L 0.500 · VESPA 0.500 · EVE 0.491 · ESM2-15B 0.488 · MIF-ST 0.485 ·
ESM-1v ensemble 0.477 · Progen2-XL 0.445 · CARP-640M 0.412 · ProtGPT2 0.257.
Supervised (per-assay CV): **Kermut 0.817** · ProteinNPT 0.776 · MSA-Transformer embeddings 0.746 ·
ESM-1v embeddings 0.710 · one-hot 0.269.

Two things to carry into our own evaluation. **ProteinMPNN is stability-specialised**: Stability 0.565 but overall
average 0.257 (rank 91/97) — the headline badly understates inverse-folding models for our task. And the one-hot
supervised baseline goes from 0.582 under a random split to **0.022 under a modulo split**, which is the cleanest
available demonstration that random splits on DMS data measure nothing.

### 5b. Human Domainome 1

Beltran A, Jiang X, Shen Y, Lehner B. "Site-saturation mutagenesis of 500 human protein domains." *Nature*
637:885–894 (2025), PMC11754108, DOI 10.1038/s41586-024-08370-4. 563,534 variants / 522 domains by aPCA abundance
(our processed table keeps 536,164 missense with fitness). Reproducibility: **median replicate Pearson r = 0.85**;
median Spearman vs in-vitro ΔΔG **0.73** (n=10 domains); vs Tsuboyama proteolysis stabilities **0.65** (n=13).

Predictor medians across the 522 domains (Fig. 3b,c): **ThermoMPNN 0.50** (best overall), ESM1v 0.48, EVE 0.48;
ThermoMPNN rises to **0.57** with zinc-fingers excluded and stays at 0.57 on the Megascale-homologue-free subset
(Extended Data Fig. 2a,b). Medians for FoldX, RaSP, DDMut, AlphaMissense, popEVE, Tranception and rSASA appear only
in the Fig. 3c bitmap — **[U]**; source data is on Zenodo 10.5281/zenodo.11043642. Later re-evaluations:
SPURS **0.54** (reproducing ThermoMPNN at 0.49), ESM-MSR **0.573**. Homology control in the paper is **hmmscan vs
Pfam**, matching our Pfam-grouped split. Note: FoldX values in that paper come from the ProtVar FTP precomputed
set, and DDMut/ThermoMPNN were run on AlphaFold2 structures.

> **Memory correction** for `domainome-finetune-results.md`: the paper's published ceiling is **0.85**
> (median replicate *Pearson*), not 0.84. If our 0.84 is a recomputation, label it as ours.

### 5c. Megascale noise ceiling

Tsuboyama K et al. *Nature* 620:434–444 (2023), PMC10412457, DOI 10.1038/s41586-023-06328-6. 1.8M measurements,
**~776,000 curated folding stabilities**, 331 natural + 148 designed domains, 40–72 aa; 210,118 double mutants
over 559 site pairs in 190 domains. Replicate K50 R = 0.97 (trypsin) / 0.99 (chymotrypsin);
**ΔG from the two independent proteases agrees at R = 0.94**; agreement with purified-protein literature ΔΔG is
**Pearson > 0.75 for all of 1,188 variants across 10 proteins**. Median 95% CI width 0.14 kcal/mol — but this
reflects only sequencing-count uncertainty, so it is a floor, not the real error. Dynamic range is clipped to
**−1 to 5 kcal/mol**. ABYSSAL quotes the trypsin/chymotrypsin PCC 0.91 as the practical ceiling; at
ThermoMPNN's 0.754 and SPURS' 0.78 Pearson, models are at roughly 83–86% of it.

---

## 6. Speed (the column that matters most for PhysDDG)

Per *saturated residue* = all 19 substitutions at one position. RaSP eLife Table 4 is the only head-to-head.

| method | hardware | per residue | per mutation (derived) | notes |
|---|---|---|---|---|
| FoldX | 64 CPU cores, 3 runs | 395–807 s | ~21–42 s wall, ~22–45 core-min | preprocessing 78–1,306 s; Pythia quotes ~1 mutation/min/core |
| Rosetta cartesian_ddg | 64 CPU cores, 3 runs | 192–414 s | ~10–22 s wall, ~11–23 core-min | preprocessing 677–17,211 s |
| ThermoNet | 1 V100 | 8.3–12.6 s | — | preprocessing 80,442–11,627,433 s (Rosetta relax) |
| ACDC-NN | 1 V100 | 1.4–1.6 s | — | needs an MSA |
| **KORPM** | **1 CPU core** | — | **~5 ms** (684 muts in 3.5 s) | the physics-method speed target |
| RaSP | 1 V100 | **0.4 s** | ~21 ms | 480–1,036× Rosetta; 230M human-proteome ΔΔG |
| ThermoMPNN | 1 GPU | — | SSM of 105 aa in 2 s, 693 aa in 8 s | |
| Stability Oracle | 1 A100 | ~50 ms | ~2.6 ms | |
| Mutate Everything | 1 A100 | — | all singles+doubles of 317 aa in 0.6 s (ESM2) | 12.1 s with the AlphaFold trunk |
| SPURS | 1 A40 | — | 118 proteins (mean 492 aa), all singles, < 20 s | |
| ProteinMPNN-ddG | 1 V100 | — | **9,800 residues/s** | human proteome (23,391 AF2 models) in 30 min |
| Pythia | 1 RTX 4090 | — | 25.2M mutations in 3 min (~140k/s) | zero-shot, 1.3M params |

Reference points for a target: **KORPM shows ~5 ms/mutation on one CPU core is achievable for a knowledge-based
pairwise potential with no packing.** Adding rotamer packing and minimisation is what makes FoldX 4,000× slower.
With 64 CPUs and 4 GPUs, matching RaSP (0.4 s/residue on one GPU) while doing real packing would already be a
10³× speed-up over FoldX.

---

## 7. Metric choices, leakage, and known data problems

**Per-protein vs pooled changes the headline by ~0.06–0.09.** SPURS: pooled Megascale ρ 0.74 but per-protein median
0.83; ThermoMPNN 0.72 pooled, 0.77 per-protein. HERMES shows the two metrics rank methods *differently*, and in
opposite directions on S669 (per-structure > pooled) vs Ssym-direct (pooled > per-structure). Our own FoldX run
shows the same gap: Megascale-test ρ 0.569 pooled, 0.633 per-protein. Always state which.

**Reverse mutations in training** — yes: ThermoNet (Q3488), ACDC-NN/-Seq (architectural), PROSTATA, DDMut, PremPS,
ABYSSAL, JanusDDG (architectural), StaB-ddG (architectural). No: RaSP, Mutate Everything, ThermoMPNN,
ProteinMPNN-ddG, Pythia, ESM-MSR. Stability Oracle deliberately uses thermodynamic *permutations* instead of
reversal, arguing that reversal leaks because all its stabilising mutations are *to* wild type.

**Antisymmetry is cheap architecturally and expensive to learn.** JanusDDG and StaB-ddG get r_{d−r} = −1 exactly by
construction. Training a Siamese penalty does *not* work: the constraint-aware paper shows the forward
over-estimate going from +0.335 to +0.396 kcal/mol because (f→ + f←)² is minimised by shrinking both. ESM-MSR
reaches −0.87 with no antisymmetry machinery at all. ThermoMPNN's own experiment: a Siamese variant given both
structures is perfectly antisymmetric and reaches Ssym-inverse 0.68, but only 0.62 from the WT structure alone,
and FireProt-HF then falls 0.65 → 0.56. **A backbone-based physics potential gets antisymmetry nearly for free
(KORPM 0.88, DDGun3D 0.99) — this is a real advantage to claim, not an afterthought.**

**Leakage, documented:**
- S669 controls only for overlap with **S2648 and VariBench**, not with PDB-scale structural corpora. ProteinMPNN
  was trained on 19,700 clusters covering the whole PDB, and every Ssym protein is in the PDB by construction
  (Ssym requires WT *and* mutant crystal structures). No published source states this overlap explicitly —
  present it as inference, citing Pancotti 2022 and Dauparas 2022.
- S669 is not clean even for post-2022 models: ABYSSAL could only use 420 of 669 variants.
- ThermoMPNN PNAS Table 2 asterisks ACDC-NN, ACDC-NN-Seq, ThermoNet, MAESTRO, mCSM, MUpro on FireProt-HF for
  >25%-identity homologues in their training data — the most citable per-model leakage flag available.
- Stability Oracle: retraining PROSTATA on its own 75%-filtered sets leaves 93/93 S669 and 15/15 Ssym proteins
  leaked → Pearson 0.98. Also: *all* C2878 proteins are >30% similar to something in cDNA117K.
- PremPS was fit on all of Ssym. JanusDDG declares Ssym, K2369, Q3421 non-blind. MutateEverything had 13/20 of
  SPURS' double-mutant test proteins in training. ESM-MSR excluded SPURS from external benchmarks for overlap.
- ProteinGym Stability is 64/66 Tsuboyama assays, so any Megascale-trained model scored there is scoring on its own
  training distribution. SPURS excludes the stability category from its ProteinGym evaluation for exactly this reason.

**Known data errors to avoid propagating:**
1. **SAAFEC-SEQ** ⚠ — in Pancotti's released `Data_s669_with_predictions.csv`, the `SAAFEC-Seq_dir` column is an
   exact copy of `I-Mutant3.0_dir` (667/667 identical). The error propagated into Benevenuta 2022 Table 2 and
   Reeves/Rossi Table S4 (both 0.490 / 1.121). Treat all SAAFEC-SEQ direct numbers as unreliable.
2. **DynaMut vs DynaMut2** ⚠ — KORPM's S461 "DynaMut2" column equals Pancotti's "DynaMut" column; one label is wrong.
3. **Sign of ⟨δ⟩** — DDGemb (Savojardo 2025) reproduces Pancotti Table 1 with ⟨δ⟩ flipped positive.
4. **ABYSSAL S669** — 0.50 in its own preprint vs 0.37 in ThermoMPNN Table 3, same 420-variant subset.
5. **S461 composition** — KORPM's text says "19.1% destabilizing / 80.8% stabilizing"; the released XLSX has
   83.7% with ΔΔG < 0, i.e. the labels are swapped in the text. 40 of 461 ΔΔG values differ from S669.
6. **Ssym size** — 684 (342+342) in Pucci 2018 and Pancotti; 704 for Ssym+; protddg-bench lists 634 after its own
   corrections; ThermoMPNN says "19 experimental structures for the direct variants" where Pucci says 15 WT proteins.
7. **ThermoMPNN mislabelled "SEQ"** in JanusDDG Suppl. Table 4, with a third-hand S669 value and no antisymmetry
   entries. Prefer SPURS' independent measurement (S669 ρ 0.44 / r 0.39).
8. **Zheng et al. 2024** (*Protein Sci* 33:e4861, S4038, 27 tools) could not be retrieved; only the figures quoted
   by Degn et al. are verified — ThermoMPNN best on balanced subsets (mean PCC 0.43), then PremPS and DDMut 0.40;
   after removing training-overlapping variants (568 left) ThermoMPNN 0.34, INPS3D 0.33, DDMut 0.31; on balanced +
   unbiased data AUTO-MUTE(RF) 0.31 and ThermoMPNN 0.30. **[U]** for the full table.

**A benchmark's difficulty is mostly its stabilising fraction.** Stabilising mutations are 12.7% of S669, 5.0% of
S461 and 11.4% of Ssym-direct; only 4.2% of Megascale is stabilising; Megascale-D doubles are >95% destabilising.
Every method's advantage shrinks when the set is balanced (Degn et al.: all tools drop on balanced subsamples of
S4038, significantly so for all but ThermoNet). Gerasimavicius, Livesey & Marsh (*Protein Sci* 32:e4688, 2023,
PMC10273344) measured 9 stability tools against 49 DMS datasets (170,940 variants): mean |Spearman| only
**0.26** (ΔΔG) / 0.28 (|ΔΔG|), with FoldX and Rosetta best and their "Foldetta" mean beating both; VEPs average
0.36. Høie et al. 2022 put Rosetta-vs-DMS at 0.25.

---

## 8. Target numbers for PhysDDG

What a physics-based, DL-free predictor has to clear. "Best classical non-DL" excludes anything with a neural
network; DDGun is included because it is an untrained linear score, KORPM because it fits 12 weights.

| benchmark | metric | best FoldX / Rosetta | best classical non-DL | best DL / hybrid | PhysDDG target |
|---|---|---|---|---|---|
| **Megascale test** (ThermoMPNN split, pooled) | Spearman | FoldX **0.57** · Rosetta 0.56 | — (no classical entry published) | **SPURS 0.74** · ThermoMPNN 0.725 · RaSP-retrained 0.67 | **beat 0.57 clearly → aim 0.62–0.68** |
| same | Pearson | Rosetta 0.53 · FoldX 0.40 | — | SPURS **0.78** · ThermoMPNN 0.754 | > 0.60 |
| same | RMSE (kcal/mol) | FoldX 3.87 · Rosetta 5.18 | — | ThermoMPNN **0.71** | < 1.2 uncalibrated |
| **Megascale, zero-shot / untrained on it** | Pearson | — | — | **RaSP 0.62** (distils Rosetta), Pythia r 0.633 / ρ 0.602 | match RaSP: r ≥ 0.62 |
| **S669 direct** | Pearson | FoldX 0.22 · Rosetta 0.39 | **MAESTRO 0.50**; DDGun3D 0.43; PoPMuSiC 0.41 | JanusDDG 0.55 · Stability Oracle 0.52 | **> 0.45**, ideally ≥ 0.50 |
| **S669 direct** | Spearman | FoldX 0.283 **[D]** | MAESTRO 0.463 **[D]**; DDGun3D 0.427 | **Pythia 0.66** (zero-shot) · Mutate Everything 0.56 · constraint-aware 0.540 | > 0.45 |
| **S669 reverse** | Pearson | FoldX 0.22 · Rosetta 0.40 | **DDGun3D 0.41**; INPS-Seq 0.43 | JanusDDG 0.55 | ≥ 0.40 with ⟨δ⟩ near 0 |
| **S461** | Pearson | Rosetta 0.60 · FoldX 0.30 | **DDGun3D 0.635**; PoPMuSiC 0.609; KORPM 0.569 | JanusDDG 0.69 · SPURS 0.64 | **> 0.60** |
| **Ssym** | direct / inverse Pearson | FoldX 0.63/0.39 · Rosetta 0.69/0.43 | **KORPM 0.57/0.49**; DDGun3D 0.56/0.53; PoPMuSiC-sym 0.48/0.48 | **Stability Oracle 0.72/0.72** · SPURS 0.76/0.68 | ≥ 0.60 / ≥ 0.50 |
| **Ssym antisymmetry** | r_{d−r} (−1 ideal) | FoldX −0.20…−0.38 · Rosetta −0.41 | **DDGun3D −0.99**; KORPM −0.88; PoPMuSiC-sym −0.77 | JanusDDG −1 (by construction) · ACDC-NN −0.98 · ESM-MSR −0.87 | **≤ −0.90** (nearly free for a physics model) |
| **Ssym bias** | ⟨δ⟩ (0 ideal) | FoldX −0.47…−1.25 · Rosetta −0.69, cart. −3.12 | **DDGun3D −0.04**; PoPMuSiC-sym +0.03; KORPM −0.15 | ACDC-NN −0.05 · JanusDDG 0.00 | **\|⟨δ⟩\| < 0.15** |
| **FireProt-HF** | Spearman / Pearson | FoldX **0.57** / 0.43 · Rosetta 0.52 / 0.45 | — | **SPURS 0.67** · ThermoMPNN 0.657 / 0.650 | > 0.57 |
| **S612 (AlphaFold structures)** | Pearson / Spearman | FoldX(MutateX) **0.44 / 0.50** · Rosetta 0.44 / 0.47 | DDGun3D 0.38 / 0.37 | ThermoMPNN 0.51 / 0.58 | > 0.45 / > 0.50, and **no worse than on crystal structures** |
| **Domainome** (aPCA, 522 domains) | median per-domain ρ | FoldX **[U]** (ProtVar run, not published) | — | **ESM-MSR 0.573** · SPURS 0.54 · ThermoMPNN 0.50 (0.57 w/o Zn) | > 0.45 would be strong for physics |
| **ProteinGym stability** | mean per-assay ρ | not evaluated | not evaluated | **Kermut 0.817** (supervised) · ProSST 0.653 · ESM-IF1 0.624 · ProteinMPNN 0.565 | ≥ 0.55 (ProteinMPNN parity) |
| **speed** | per mutation | FoldX ~21–42 s wall on 64 cores | **KORPM ~5 ms, 1 core** | RaSP 21 ms / GPU · ProteinMPNN-ddG 9,800 res/s | **< 10 ms/mutation**, i.e. ≥ 10³× FoldX |
| **trivial-baseline floor** | Megascale ρ / S461 ρ | — | RSA alone **0.468** / **0.456** | — | must beat RSA by a wide margin on both |

Three calibration points worth repeating. (1) The **noise ceiling** on Megascale is Pearson ~0.91–0.94
(protease-to-protease) and >0.75 against purified-protein ΔΔG; on Domainome it is replicate r 0.85; on
ProTherm-style data ACDC-NN's authors put it at 0.70–0.85. (2) **RSA alone gets Spearman 0.468 on Megascale test
and 0.456 on S461** — more than half of what FoldX achieves, so any new method must be shown against burial, not
just against FoldX. (3) **A single contact potential (Bastolla-Vendruscolo) reaches Pearson ~0.50 on S2648** in
DDGun's ablation, and untrained DDGun3D is still the best classical method on S461. The marginal value of
packing, minimisation and an explicit unfolded state has to be demonstrated against these, not assumed.

---

## 9. Primary sources

Tsuboyama 2023 *Nature* 620:434 (10.1038/s41586-023-06328-6) · Beltran 2025 *Nature* 637:885
(10.1038/s41586-024-08370-4) · Stourac 2021 *NAR* 49:D319 (10.1093/nar/gkaa981) · Notin 2023 NeurIPS D&B
(ProteinGym v1.3, Zenodo 10.5281/zenodo.15293562) · Pancotti 2022 *Brief Bioinform* 23:bbab555 (10.1093/bib/bbab555)
· Pucci 2018 *Bioinformatics* 34:3659 (10.1093/bioinformatics/bty348; preprint 10.1101/308239) · Hernández 2023
*Bioinformatics* 39:btad011 (10.1093/bioinformatics/btad011) · Montanucci 2019 *BMC Bioinformatics* 20:335 ·
Montanucci 2022 *NAR* 50:W222 · Park 2016 *JCTC* 12:6201 · Frenz 2020 *Front Bioeng Biotechnol* 8:558247 ·
Li 2020 *PLoS Comput Biol* 16:e1008291 · Benevenuta 2021 *J Phys D* 54:245403 · Pancotti 2021 *Genes* 12:911 ·
Chen 2020 *PLoS Comput Biol* 16:e1008543 · Blaabjerg 2023 *eLife* 12:e82593 (10.7554/eLife.82593) ·
Umerenkov 2023 *Bioinformatics* 39:btad671 · Zhou 2023 *NAR* 51:W122 (DDMut) · Zhou 2024 *NAR* 52:W207 (DDMut-PPI) ·
Cao 2019 *JCIM* 59:1508 (DeepDDG) · Ouyang-Zhang 2023 NeurIPS (arXiv 2310.12979) · Dieckhaus 2024 *PNAS*
121:e2314853121 (10.1073/pnas.2314853121) · Dieckhaus 2025 *Protein Sci* 34:e70003 (10.1002/pro.70003) ·
Diaz 2024 *Nat Commun* 15:6170 (10.1038/s41467-024-49780-2) · Dutton 2024 bioRxiv 10.1101/2024.06.15.599145 ·
Reeves 2024 *Nat Mach Intell* (10.1038/s42256-024-00887-7) · Chu 2024 *PLoS Comput Biol* 20:e1012248 ·
Pak 2023 bioRxiv 10.1101/2022.12.31.522396 (ABYSSAL) · Sun 2025 *The Innovation* 6:100750 (Pythia) ·
Visani 2024 bioRxiv 10.1101/2024.07.09.602403 (HERMES) · Barducci 2026 *Commun Biol* 10.1038/s42003-026-09632-9
(JanusDDG) · Li & Luo 2025 *Nat Commun* 10.1038/s41467-025-67609-4 (SPURS) · Deng 2025 ICML PMLR 267:13129
(StaB-ddG) · Tan 2026 *Chem Sci* 17:18220 (UniStab) · Reeves 2026 bioRxiv 10.64898/2026.06.04.730231 (ESM-MSR) ·
Shivram 2026 arXiv 2606.08100 · Gelman 2025 *Nat Methods* 22:1868 (METL) · Gerasimavicius 2023 *Protein Sci*
32:e4688 · Zheng 2024 *Protein Sci* 33:e4861 · Degn 2025 bioRxiv 10.1101/2025.03.28.645695 ·
Vithayapalert 2026 bioRxiv 10.64898/2026.10.06.756934 · Rossi 2025 *Protein Sci* 34:e70134 ·
Marabotti 2021 *BMC Bioinformatics* 22:345 · Iqbal 2021 *Brief Bioinform* 22:bbab184 **[U]** ·
protddg-bench.github.io (dataset definitions).

**Not retrieved / open gaps:** Zheng 2024 S4038 per-tool table; Reeves & Kalyaanamoorthy Table 1 numerics
(bitmap + paywall; raw predictions at github.com/skalyaanamoorthy/thermostability-transfer);
Domainome Fig. 3c per-predictor medians for FoldX/RaSP/DDMut/rSASA (Zenodo 10.5281/zenodo.11043642);
DDMut S669-forward-only, Ssym and S461; Iqbal 2021 per-tool numbers; Pythia's Ssym Table S3;
ESM-MSR per-dataset Spearman (figure-only); Megascale ΔΔG Spearman for SaProt / ProSST / ESM3 (never published —
use ProteinGym Stability as a stated proxy). The MGnify Stability dataset (Rocklin lab, 1.8M domains, announced
2026-05-26) is press-release only so far.
