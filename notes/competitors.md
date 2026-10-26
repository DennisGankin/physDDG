# Classical competitor baselines: KORPM, DDGun, DDGun3D

Scripts: `70_install_korpm.sh`, `71_run_korpm.py`, `72a_download_uniclust30.sh`, `72b_install_ddgun.sh`, `72_run_ddgun.py` (+ worker
`72w_ddgun_worker.py`), `73_score_competitors.py` (scores with `30_evaluate.py` / `physddg/eval.py`), `74_korpm_leakage.py`, shared `comp_common.py`.
Predictions: `results/competitors/<method>_<bench>.csv` (KEYS + `ddg_pred`, positive = destabilising, plus the raw tool value). Metrics:
`results/competitors/metrics/*.json`, `summary.csv`, `summary.md` (all tables below). Large prediction files (megascale, domainome) are git-ignored.
Tools live in `data/raw/tools/{korpm,ddgun,uniclust30_2018_08}` (git-ignored); the DDGun venv is `.venv_ddgun` (python 3.10, biopython 1.79).

## Setup and checks

* Benchmarks: Megascale test (28 proteins, 28,172 variants), FireProt test (347 / 27), S669 (669 / 94), S461 (461 / 48), Ssym (684 = 342 direct + 342 reverse, 15 WT proteins; reverse rows are scored on the mutant structure), Domainome (see below). The same structure files as in the canonical tables were used for every method.
* **Sign.** Both tools report positive = stabilising (KORPM README; DDGun "Increase" = positive). `ddg_pred = -raw`. Verified: after the flip KORPM correlates +0.695 with `ddg` on Ssym (published PCC 0.69/0.695), and the raw outputs reproduce the README example (1BNI IA76A = -1.396).
* **Reproduction of published numbers.** KORPM: Ssym Pearson 0.695 (paper 0.69), antisymmetry r(dir,rev) 0.875 (paper 0.88), bias +0.149 (paper -0.15 in the opposite sign convention), S461 Pearson 0.581 (paper 0.57); our outputs correlate r = 0.965 with the authors' own S461 and S669 prediction files (`korpm/S461_results`; they used slightly different weights/structure preparation, so not identical). DDGun3D: S669 Pearson 0.421 (literature 0.42), S461 0.604 (0.58-0.62), Ssym r(dir,-rev) = -0.991 (literature 0.99); r = 0.98 with the authors' S669/S461 DDGun3D outputs; the `ddgun/test/1aar` example reproduces the README values.
* **KORPM** (`korpm_gcc`, default weights, `korp6Dv1.bin` from the repo's git-LFS; git-lfs is not installed, so `70_install_korpm.sh` fetches the pointer files via `media.githubusercontent.com`). KORPM aborts on any residue missing N/CA/C/O: such residues are dropped from a temporary copy of the structure (S669: 2 variants lost because their site lacks a backbone atom). KORPM cannot parse insertion codes, so residues are renumbered in the copy where needed (none of the test sets has insertion codes). Domainome: 3,663 of 536,164 rows have no structure and are uncovered.
* **DDGun / DDGun3D.** Original code, original database (UniClust30 2018_08, 26.7 GB download, 92 GB unpacked, hhblits v3.3.0 built from source, `-n 2 -e 1e-6` as in the tool). No approximation was necessary. Predictions are recomputed from the unrounded features (the CLI rounds to 1 decimal, which creates ties); formulas as in `ddgun_3d.py`. hhblits results are cached by sequence and the per-site environment is cached (speed only). Ssym reverse rows get their own profile from the mutant sequence, so 3D antisymmetry is 0.991 rather than exactly 1 (the sequence-only version: 1.000).
* **Domainome.** KORPM on all 518 constructs (532,501 variants; `fitness` proxy, scored against -score as in the harness: Spearman 0.294 [0.26, 0.32], per-protein median 0.302, Pearson 0.346; `metrics/korpm_domainome_fullset.json`). DDGun/DDGun3D on a random subset of 50 constructs (47,794 variants, seed 0); the tables below compare all methods on that same subset. Domainome has no kcal/mol target in the canonical table, so no RMSE/AUROC.
* Metrics are uncalibrated. RMSE is raw. CIs: bootstrap over homology clusters (300 replicates).

## Results

**All rows**

| bench | method | n | prot | Spearman [95% CI] | per-protein median Sp | Pearson | RMSE | AUROC(ddg>1) | Ssym r(dir,-rev) | Ssym bias |
|---|---|---|---|---|---|---|---|---|---|---|
| megascale | KORPM | 28172 | 28 | 0.482 [0.44, 0.52] | 0.533 | 0.524 | 0.97 | 0.761 |  |  |
| megascale | DDGun3D | 28172 | 28 | 0.458 [0.38, 0.54] | 0.502 | 0.537 | 1.17 | 0.768 |  |  |
| megascale | DDGun (seq) | 28172 | 28 | 0.404 [0.29, 0.51] | 0.482 | 0.432 | 1.71 | 0.745 |  |  |
| fireprot | KORPM | 347 | 27 | 0.408 [0.24, 0.55] | 0.385 | 0.397 | 2.11 | 0.715 |  |  |
| fireprot | DDGun3D | 347 | 27 | 0.431 [0.24, 0.56] | 0.478 | 0.488 | 2.00 | 0.755 |  |  |
| fireprot | DDGun (seq) | 347 | 27 | 0.437 [0.24, 0.58] | 0.473 | 0.465 | 2.07 | 0.768 |  |  |
| s669 | KORPM | 667 | 94 | 0.337 [0.14, 0.52] | 0.506 | 0.275 | 1.71 | 0.684 |  |  |
| s669 | DDGun3D | 667 | 94 | 0.425 [0.23, 0.58] | 0.529 | 0.421 | 1.59 | 0.718 |  |  |
| s669 | DDGun (seq) | 667 | 94 | 0.413 [0.21, 0.56] | 0.461 | 0.388 | 1.72 | 0.717 |  |  |
| s461 | KORPM | 461 | 48 | 0.557 [0.40, 0.65] | 0.610 | 0.581 | 1.20 | 0.783 |  |  |
| s461 | DDGun3D | 461 | 48 | 0.573 [0.32, 0.67] | 0.588 | 0.604 | 1.13 | 0.791 |  |  |
| s461 | DDGun (seq) | 461 | 48 | 0.539 [0.27, 0.65] | 0.551 | 0.543 | 1.30 | 0.778 |  |  |
| ssym | KORPM | 684 | 15 | 0.718 [0.42, 0.78] | 0.768 | 0.695 | 1.35 | 0.885 | 0.875 | 0.149 |
| ssym | DDGun3D | 684 | 15 | 0.534 [0.33, 0.70] | 0.649 | 0.637 | 1.43 | 0.792 | 0.991 | 0.036 |
| ssym | DDGun (seq) | 684 | 15 | 0.511 [0.24, 0.67] | 0.642 | 0.611 | 1.48 | 0.789 | 1.000 | 0.011 |
| domainome | KORPM | 47794 | 50 | 0.293 [0.24, 0.34] | 0.290 | 0.348 |  |  |  |  |
| domainome | DDGun3D | 47794 | 50 | 0.265 [0.23, 0.30] | 0.278 | 0.384 |  |  |  |  |
| domainome | DDGun (seq) | 47794 | 50 | 0.285 [0.25, 0.32] | 0.288 | 0.333 |  |  |  |  |

**Strict: no Megascale-train homologue (homolog_in_megascale_train==0)**

| bench | method | n | prot | Spearman [95% CI] | per-protein median Sp | Pearson | RMSE | AUROC(ddg>1) | Ssym r(dir,-rev) | Ssym bias |
|---|---|---|---|---|---|---|---|---|---|---|
| megascale | KORPM | 20156 | 19 | 0.483 [0.44, 0.53] | 0.542 | 0.519 | 0.97 | 0.771 |  |  |
| megascale | DDGun3D | 20156 | 19 | 0.461 [0.38, 0.56] | 0.478 | 0.545 | 1.20 | 0.781 |  |  |
| megascale | DDGun (seq) | 20156 | 19 | 0.395 [0.28, 0.52] | 0.433 | 0.427 | 1.81 | 0.755 |  |  |
| fireprot | KORPM | 341 | 25 | 0.408 [0.27, 0.53] | 0.385 | 0.395 | 2.13 | 0.711 |  |  |
| fireprot | DDGun3D | 341 | 25 | 0.436 [0.22, 0.56] | 0.478 | 0.489 | 2.02 | 0.751 |  |  |
| fireprot | DDGun (seq) | 341 | 25 | 0.439 [0.23, 0.58] | 0.473 | 0.465 | 2.08 | 0.763 |  |  |
| s669 | KORPM | 488 | 87 | 0.259 [0.06, 0.45] | 0.417 | 0.214 | 1.88 | 0.630 |  |  |
| s669 | DDGun3D | 488 | 87 | 0.344 [0.10, 0.50] | 0.386 | 0.355 | 1.76 | 0.669 |  |  |
| s669 | DDGun (seq) | 488 | 87 | 0.329 [0.09, 0.48] | 0.415 | 0.322 | 1.89 | 0.664 |  |  |
| s461 | KORPM | 290 | 43 | 0.514 [0.36, 0.65] | 0.689 | 0.545 | 1.26 | 0.743 |  |  |
| s461 | DDGun3D | 290 | 43 | 0.500 [0.20, 0.61] | 0.621 | 0.554 | 1.22 | 0.749 |  |  |
| s461 | DDGun (seq) | 290 | 43 | 0.452 [0.18, 0.56] | 0.571 | 0.477 | 1.39 | 0.729 |  |  |
| ssym | KORPM | 684 | 15 | 0.718 [0.42, 0.78] | 0.768 | 0.695 | 1.35 | 0.885 | 0.875 | 0.149 |
| ssym | DDGun3D | 684 | 15 | 0.534 [0.33, 0.70] | 0.649 | 0.637 | 1.43 | 0.792 | 0.991 | 0.036 |
| ssym | DDGun (seq) | 684 | 15 | 0.511 [0.24, 0.67] | 0.642 | 0.611 | 1.48 | 0.789 | 1.000 | 0.011 |
| domainome | KORPM | 29004 | 32 | 0.281 [0.21, 0.33] | 0.277 | 0.334 |  |  |  |  |
| domainome | DDGun3D | 29004 | 32 | 0.263 [0.22, 0.31] | 0.282 | 0.384 |  |  |  |  |
| domainome | DDGun (seq) | 29004 | 32 | 0.275 [0.23, 0.32] | 0.288 | 0.326 |  |  |  |  |

**Strict: no Megascale-train AND no KORPM-training homologue**

| bench | method | n | prot | Spearman [95% CI] | per-protein median Sp | Pearson | RMSE | AUROC(ddg>1) | Ssym r(dir,-rev) | Ssym bias |
|---|---|---|---|---|---|---|---|---|---|---|
| megascale | KORPM | 20156 | 19 | 0.483 [0.44, 0.53] | 0.542 | 0.519 | 0.97 | 0.771 |  |  |
| megascale | DDGun3D | 20156 | 19 | 0.461 [0.38, 0.56] | 0.478 | 0.545 | 1.20 | 0.781 |  |  |
| megascale | DDGun (seq) | 20156 | 19 | 0.395 [0.28, 0.52] | 0.433 | 0.427 | 1.81 | 0.755 |  |  |
| fireprot | KORPM | 36 | 8 | 0.268 [-0.15, 0.60] | 0.385 | 0.332 | 3.67 | 0.646 |  |  |
| fireprot | DDGun3D | 36 | 8 | 0.388 [-0.10, 0.63] | 0.259 | 0.417 | 3.20 | 0.718 |  |  |
| fireprot | DDGun (seq) | 36 | 8 | 0.337 [-0.11, 0.59] | 0.084 | 0.388 | 3.71 | 0.688 |  |  |
| s669 | KORPM | 206 | 41 | 0.148 [-0.07, 0.47] | 0.392 | 0.065 | 2.25 | 0.596 |  |  |
| s669 | DDGun3D | 206 | 41 | 0.226 [-0.01, 0.41] | 0.437 | 0.245 | 2.05 | 0.611 |  |  |
| s669 | DDGun (seq) | 206 | 41 | 0.216 [-0.01, 0.38] | 0.396 | 0.230 | 2.19 | 0.604 |  |  |
| s461 | KORPM | 93 | 19 | 0.601 [0.10, 0.71] | 0.744 | 0.637 | 1.12 | 0.820 |  |  |
| s461 | DDGun3D | 93 | 19 | 0.370 [-0.32, 0.58] | 0.683 | 0.512 | 1.26 | 0.681 |  |  |
| s461 | DDGun (seq) | 93 | 19 | 0.286 [-0.38, 0.49] | 0.581 | 0.344 | 1.54 | 0.633 |  |  |
| ssym | KORPM | 22 | 3 | 0.841 [-0.80, 0.85] | 0.745 | 0.793 | 1.29 |  | 0.855 | 0.484 |
| ssym | DDGun3D | 22 | 3 | 0.002 [-0.50, 0.75] | 0.745 | 0.008 | 2.11 |  | 0.986 | 0.063 |
| ssym | DDGun (seq) | 22 | 3 | 0.065 [-0.79, 0.72] | 0.709 | 0.067 | 2.06 |  | 0.984 | 0.048 |
| domainome | KORPM | 27565 | 31 | 0.267 [0.19, 0.32] | 0.276 | 0.326 |  |  |  |  |
| domainome | DDGun3D | 27565 | 31 | 0.260 [0.22, 0.30] | 0.281 | 0.385 |  |  |  |  |
| domainome | DDGun (seq) | 27565 | 31 | 0.270 [0.23, 0.32] | 0.287 | 0.324 |  |  |  |  |

**No exact variant (same PDB, chain, position, WT, mutant) in KORPM's training set**

| bench | method | n | prot | Spearman [95% CI] | per-protein median Sp | Pearson | RMSE | AUROC(ddg>1) | Ssym r(dir,-rev) | Ssym bias |
|---|---|---|---|---|---|---|---|---|---|---|
| megascale | KORPM | 28172 | 28 | 0.482 [0.44, 0.52] | 0.533 | 0.524 | 0.97 | 0.761 |  |  |
| megascale | DDGun3D | 28172 | 28 | 0.458 [0.38, 0.54] | 0.502 | 0.537 | 1.17 | 0.768 |  |  |
| megascale | DDGun (seq) | 28172 | 28 | 0.404 [0.29, 0.51] | 0.482 | 0.432 | 1.71 | 0.745 |  |  |
| fireprot | KORPM | 292 | 25 | 0.387 [0.22, 0.54] | 0.541 | 0.381 | 2.16 | 0.700 |  |  |
| fireprot | DDGun3D | 292 | 25 | 0.447 [0.19, 0.57] | 0.517 | 0.535 | 1.96 | 0.747 |  |  |
| fireprot | DDGun (seq) | 292 | 25 | 0.447 [0.19, 0.59] | 0.510 | 0.491 | 2.05 | 0.762 |  |  |
| s669 | KORPM | 476 | 77 | 0.272 [0.04, 0.49] | 0.534 | 0.179 | 1.84 | 0.655 |  |  |
| s669 | DDGun3D | 476 | 77 | 0.394 [0.18, 0.57] | 0.480 | 0.374 | 1.68 | 0.715 |  |  |
| s669 | DDGun (seq) | 476 | 77 | 0.375 [0.16, 0.54] | 0.412 | 0.338 | 1.80 | 0.712 |  |  |
| s461 | KORPM | 321 | 38 | 0.558 [0.39, 0.66] | 0.546 | 0.578 | 1.21 | 0.786 |  |  |
| s461 | DDGun3D | 321 | 38 | 0.569 [0.28, 0.69] | 0.571 | 0.607 | 1.12 | 0.797 |  |  |
| s461 | DDGun (seq) | 321 | 38 | 0.514 [0.23, 0.66] | 0.454 | 0.521 | 1.28 | 0.776 |  |  |
| ssym | KORPM | 684 | 15 | 0.718 [0.42, 0.78] | 0.768 | 0.695 | 1.35 | 0.885 | 0.875 | 0.149 |
| ssym | DDGun3D | 684 | 15 | 0.534 [0.33, 0.70] | 0.649 | 0.637 | 1.43 | 0.792 | 0.991 | 0.036 |
| ssym | DDGun (seq) | 684 | 15 | 0.511 [0.24, 0.67] | 0.642 | 0.611 | 1.48 | 0.789 | 1.000 | 0.011 |
| domainome | KORPM | 47794 | 50 | 0.293 [0.24, 0.34] | 0.290 | 0.348 |  |  |  |  |
| domainome | DDGun3D | 47794 | 50 | 0.265 [0.23, 0.30] | 0.278 | 0.384 |  |  |  |  |
| domainome | DDGun (seq) | 47794 | 50 | 0.285 [0.25, 0.32] | 0.288 | 0.333 |  |  |  |  |

FoldX 5.1 for reference (`notes/eval_harness.md`): Megascale test Spearman 0.569, per-protein median 0.633, Pearson 0.447, RMSE 3.17 (uncalibrated; the competitor RMSEs are small because both tools have a compressed output range); FireProt test Spearman 0.449, per-protein median 0.634.

Takeaways:
* On Megascale (not used to build either tool; KORPM's training set has no Megascale overlap) KORPM 0.482 > DDGun3D 0.458 > DDGun 0.404, all below FoldX (0.569). The strict Megascale subset (no Megascale-train homologue) changes nothing, as expected for non-fitted tools.
* FireProt: all three about 0.41-0.44 (CIs overlap fully); FoldX 0.449 is in the same range.
* S669: DDGun3D (0.425) > DDGun (0.413) > KORPM (0.337); S461: all about 0.54-0.57. Ssym: KORPM 0.718 vs DDGun3D 0.534, but see the leakage section (97% of Ssym rows are homologous to KORPM training proteins).
* KORPM is not antisymmetric (Ssym r(dir,-rev) 0.875, bias +0.15 kcal/mol); DDGun is by construction (0.99-1.00).
* Domainome: all three about 0.27-0.29 pooled Spearman on the 50-construct subset.

## Run times

| tool | cost |
|---|---|
| KORPM | reads the 332 MB potential once per process (about 1.2 s), then **0.44 ms per mutation** on one core (1,121 Megascale variants of 1GYZ: 1.69 s total, 1.19 s of it loading). All 6 benchmarks (about 567k variants, 1,100+ structures) took 48 s wall on 40 workers. |
| DDGun / DDGun3D | dominated by hhblits against UniClust30: **31-58 s per unique sequence** with 4 threads (machine load about 100; first search with a cold cache 93 s), independent of the number of mutations. After the profile exists: 0.7-0.8 ms per mutation on dense sets (Megascale, Domainome; site-environment cache), 20-90 ms per mutation when each structure carries only 1-10 mutations (DSSP and parsing dominate). Wall times with 15 concurrent structures: FireProt 177 s, S461 251 s, S669 232 s, Megascale 73 s, Ssym 831 s (357 searches), Domainome-50 149 s. Without the cache the original CLI recomputes the environment for every mutation, which is much slower for dense scans. |

## Leakage

`scripts/74_korpm_leakage.py` compares the benchmarks with KORPM's training set `Id25c03_1merNCLB.txt` (the set behind the released `korp6Dv1` weights: ThermoMutDB + ProTherm, 10-40 C, pH 5-9, monomers, no ligand within 5 A, at most 15 mutations per amino-acid pair: 2,371 mutations, 181 PDB entries / 129 families; the repo says only the Ssym *mutations* were excluded, not Ssym homologues). Homologue = same definition as `scripts/14_homology.py` (mmseqs, E <= 1e-3, id >= 25%, >= 50% of the shorter sequence). Per-protein tables: `results/competitors/korpm_leakage_<bench>.csv`.

| bench | proteins with a KORPM-train homologue | rows with homologue | rows with id >= 90% | rows on a PDB entry present in training | variants identical to a training mutation (same PDB, chain, position, WT, mut) |
|---|---|---|---|---|---|
| megascale test | 5/28 (18%) | 16% | 16% | 0% | 0 |
| fireprot test | 70% | 90% | 87% | 59% | 55 (15.9%) |
| s669 | 56% | 69% | 66% | 65% | 191 (28.6%) |
| s461 | 60% | 80% | 79% | 78% | 140 (30.4%) |
| ssym | 80% | 97% | 97% | 87% | 0 (excluded by the authors) |
| domainome (518 constructs) | 24% | 26% | 4% | 0% | 0 |

* **KORPM is fitted on ProTherm-type data, so FireProt, S669, S461 and Ssym are in-sample to a large degree** (S669 and S461 were meant to be independent of other tools' training sets, not of KORPM's). Only 12 weights are fitted, and the authors report that retraining without S461 homologues changes S461 PCC 0.575 to 0.570, so the practical inflation is probably small, but it is not zero and the strict rows are the clean ones: S669 Spearman 0.337 -> 0.148 (n = 206, 41 proteins) and FireProt 0.408 -> 0.268 (n = 36) when KORPM-train homologues are removed (both noisy; CIs include 0). Ssym has only 22 rows (3 proteins) left, so it cannot be used for a clean KORPM comparison. We did not retrain the weights (`korpm_opt` and the strict training subsets `Id25c03_1merNCLB_{S461,Ssym}.txt` are in the repo if a leakage-free KORPM is wanted).
* **Megascale and Domainome are clean for KORPM** (no PDB or mutation overlap; some domains have homologous training proteins but the weights are 12 numbers).
* **DDGun is untrained** (weights set proportional to feature correlations with ΔΔG on VariBench-type data, so Ssym/FireProt/ProTherm overlap with those sets is a mild, hard-to-quantify leak); its profile database (UniClust30 2018) contains homologues of all benchmark proteins, which is by design.
* Megascale test has 9/28 proteins with a Megascale-train homologue; irrelevant for these tools, but the strict_ms table is shown for use in side-by-side comparisons with trained models.
* Ssym `ddg` uses the ThermoNet/ACDC copy; KORPM's own copy differs on 24/342 pairs (see `data/processed/ssym.md`). Using ours, KORPM Pearson is 0.695 vs 0.695 with theirs.

## Other notes
* `scripts/30_evaluate.py` crashed on S669/S461 (`pair_id`/`direction` present but only direct rows): fixed to compute antisymmetry only when both directions exist.
* `results/competitors/korpm_timing.jsonl`, `ddgun_timing.jsonl`: raw timing logs of the final runs. (`korpm_ms_per_mut_cpu` in the jsonl includes the 1.2 s potential load per structure; the 0.44 ms figure above excludes it.)
