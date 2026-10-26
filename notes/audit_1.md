# Audit 1: leakage, evaluation correctness, over-claims (2026-10-26, agent: audit)

Scope: PDB statistics / rotamer library, fitting (`physddg/fit.py`, `scripts/51`, `scripts/52`), evaluation
(`physddg/eval.py`, `results/ladder_physics/`), feature hygiene, and the claims in `notes/ladder_physics.md` and
`RESEARCH_TREE.md`. I wrote these scripts (they only write into a directory you pass on the command line, or to stdout):
- `scripts/99_audit_pdbstats_leak.py OUTDIR`: MMseqs2 search of the PISCES chains against every benchmark protein.
- `scripts/99_audit_pdbstats_rederive.py OUTDIR`: re-derives `rotlib_bbdep` / `pdbstats` without the overlapping chains and estimates the effect. It never overwrites `physddg/data`.
- `scripts/99_audit_eval.py`: independent re-computation of the metrics from `preds_*.parquet`, joined back to `data/processed`.
- `scripts/99_audit_dup_mutations.py`: exact duplicate mutations across sets, matched on the ±7-residue sequence window plus wt and mut.

## Verdict summary (by severity)

### Important
1. **The FireProt-test numbers of the joint models depend on a rule built from FireProt-test rows.** The CXXC
   disulfide exception (`52:cxxc_flags`, `ss_broken_nc`) was added after looking at thioredoxin 2TRX C32A/C35A
   (`notes/baseline_ladder.md`, suspicious item 2). Both rows are `split == test` in `data/processed/fireprot.csv`.
   - With the exception: B4j r = 0.479 and B5j r = 0.484. These are bolded as the best FireProt r in the main table.
   - Without it (adding the fitted `ss_broken_nc` weight of 3.25 / 3.32 kcal/mol to those 2 rows): r ≈ 0.389 / 0.396. That is below FoldX (0.411) and below B4 (0.450).
   - B4 and B5 are unaffected (the weight is 0 because Megascale has no disulfide rows).
   - Fix: report B4j/B5j FireProt-test as test-informed, or justify the rule from FireProt-train/val or literature and say so.
2. **The S461 column is contaminated by exact training duplicates.**
   - 113/461 S461 rows (24.5%) are mutations that also appear in Megascale-train: 1DIV, 1O6X, 2HBB, 2WQG, 3DV0. A further 11 rows duplicate FireProt-trainval.
   - The main table reports S461 on all rows. Strict-subset values: B4 r 0.618 → 0.575, B5 0.605 → 0.556, B3+ 0.604 → 0.569, DDGun3D 0.604 → 0.552.
   - S669 uses the strict subset, which is correct. S669-all has 16.9% duplicate rows.
   - Use S461-strict, or drop the column.
3. **The joint models' Ssym numbers are not test numbers.** 293/342 Ssym direct rows (85.7%) are exact duplicate
   mutations of FireProt-trainval. B4j/B5j are fitted on FireProt-train, yet their Ssym dir r (0.714/0.716) is bolded
   as the best. It is only a valid diagnostic for the Megascale-only fits (0 duplicates with Megascale-train).
4. **G1 under D16 fails for both B4 and B5.** D16 is the operative gate: B4 ≥ B3+ + 0.03, and beat FoldX with a paired CI that excludes 0, on Megascale-test natural ρ_prot.
   - B4 − B3+ = +0.029 [−0.002, +0.051]. B4 − FoldX = +0.048 [−0.008, +0.105].
   - B5 − FoldX = +0.058 [−0.003, +0.113]. B5 clears the B3+ bar, but neither model beats FoldX with a CI excluding 0 on the primary subset.
   - On "all" (+0.050 [+0.002, +0.110]) and "natural_strict" (+0.048 [+0.002, +0.119]) the CI just excludes 0. So the honest statement is "ahead of FoldX in point estimate; significant on 2 of 3 Megascale subsets, not on the pre-registered one".
   - `notes/ladder_physics.md` says "clearly ahead of FoldX … on Megascale". That holds for Pearson, not for ranking.
5. **The Pearson advantage over FoldX is mostly FoldX's uncapped tail.** FoldX predictions on Megascale-test reach 71 kcal/mol.
   - Fixed monotone transforms of FoldX, with no fitting: clipping to [−3, 8] raises its Megascale-natural Pearson from 0.445 to 0.596; signed log1p gives 0.583.
   - So the +0.25 Pearson gap shrinks to about +0.10 against a capped FoldX.
   - The same trick applied to our model was selected on validation (`slog_rep`). Report the capped-FoldX Pearson next to the raw one, or soften "large gain in calibration".
6. **The headline Megascale "natural" subset contains exact training duplicates.** 1TUC and 2KXD (test) share 1,437 identical mutations (5.1% of test rows) with a Megascale-train domain at 100% identity.
   - The strict subset is not lower (B4 0.687 vs 0.677), so there is no inflation. But the text should say the headline subset is not homology-clean.
   - The natural headline has only 21 proteins (7 with train homologs). The strict natural subset has 14.

### Minor
7. **PDB-statistics overlap: real but negligible.**
   - The PISCES pc30 'train' chains contain homologues (≥ 30% id, ≥ 50% of the shorter sequence) of:
     - 10/28 Megascale-test domains, all natural. 3DKM is itself in the list. Others: 2KWH 92%, 1TUC/2KXD 88%, 2K28 74%, 2K1B 69%, 1W4H/2BTH 51%, 2WXC 43%, 6EWT 38%.
     - 24/27 FireProt-test proteins, 7 of them identical: 1KF2, 2AFG, 1HZ6 (in list), 1CAH, 1A23, 1JNX, 1FTG.
     - 70/94 S669 (17 identical), 33/48 S461, 13/15 Ssym (5 identical), and 262/517 Domainome (26 identical).
   - Re-deriving the library and statistics without the 300 overlapping train chains (8,362 → 8,062) changes:
     - env potential: mean |Δ| 0.017 kT;
     - Rama: mean 0.010 kT;
     - rotamer P: mean |Δ| 0.0007.
   - With the fitted weights, B4/B5 predictions move by 0.0014 kcal/mol on average. Per-protein ρ changes by ≤ 0.0005 on Megascale natural, FireProt and Domainome.
   - Not re-run: the engine energies with the clean rotamer library. The library change is tiny, but E1 can verify it with `99_audit_pdbstats_rederive.py`, which writes `rotlib_bbdep_clean.npz`.
   - Conclusion: disclose it; no re-derivation is needed for the headline.
8. **Main-table cells are not on identical rows.** `52:445-446` evaluates each model on its own non-missing rows, while the table header says "identical rows".
   - FoldX covers 341/347 FireProt-test rows and 493/518 Domainome domains. DDGun3D covers 50 Domainome domains.
   - On common rows the conclusions do not change: Domainome B4 0.406 vs FoldX 0.430; FireProt B4 r 0.459 vs FoldX 0.417. The DDGun3D Domainome cell should be greyed out.
   - The paired deltas (`paired.csv`) are on identical rows and are correct.
9. **Test-informed choices beyond item 1.**
   - D16 re-based G1 on measured ladder numbers, which included test numbers (B3+ test 0.648 → bar 0.68).
   - G2's "passes on its target stratum" (buried small→large +0.048) comes from the test set (`strata.csv` uses `split==test`). No CV or val evidence is reported for the stratum.
   - Selection otherwise touched only Megascale-val and train CV. 11/31 val proteins have train homologs, which makes selection slightly optimistic but does not leak test data.
10. **FireProt per-protein ρ is a median over 9 proteins.** Only 9 have ≥ 10 variants, although `n_proteins` says 27, and the CIs are about ±0.2. There FoldX (0.634) and tmol-refit (0.678) beat B4/B5 (0.578/0.584).
11. **The WT→WT test is tautological.** In `tests/test_mutate.py:43-45`, dF is defined as F − F[wt], so the WT entry is 0 by construction.
    - The real D6 property (no native rotamer at the site) holds by code (`mutate.py:4`, library-only candidates), but no test checks it.
    - Known WT-favouring asymmetries remain by design and are documented (the adaptation bias in D15): neighbours keep their exact native rotamer, and mutants keep the native CB.

## Checks that passed (no action)
- **Fitting.** Standardisation uses training-table SDs with no centring (`fit.py:120-124`). Clip caps come from train percentiles only (`52:165`). Prediction is raw w·x + Δref, with no per-protein or dataset-wide normalisation and no test labels. dG_wt is not a feature; it only sets censoring intervals on train. Censored rows enter training only, and test is Dataset-3 rows. CV groups are by `family` (single-linkage).
- **Megascale strict subset** = `homolog_in_megascale_train==0`. It covers 19 domains (14 natural) and excludes all 9 homologous test domains. FireProt-test has 0 exact duplicates with FireProt-trainval or Megascale; 1/27 (1MGR) is homologous to FireProt-trainval.
- **Metric re-computation.** My independent per-protein median, pooled Spearman and Pearson reproduce every headline to 4 decimals: B4 0.6768, B5 0.6866, FoldX 0.6288, B3+ 0.6481, tmol-refit 0.6221 (Megascale natural, 21 proteins). FireProt r 0.450/0.456/0.411. Domainome ρ_dom 0.405/0.411/0.430.
- **Row alignment.** Prediction ddg equals canonical ddg exactly; there are no duplicate keys; the natural flag is identical for all predictors. FoldX covers 100% of Megascale-test.
- **Domainome.** The target is −score (higher score = more abundant). All 518 domains have ≥ 50 variants, so min-n has no effect.
- **Signs.** 518 buried (RSA < 0.1) I/L/V/F/M → D/E/K/R Megascale-test variants: ddg mean +2.64 (99.6% > 0), B4 mean +2.49 (100% > 0), FoldX +5.0. I spot-checked 20 by eye and all are positive for every predictor.
- **Structures.** Ssym reverse rows use the mutant crystal structure, and 'wt' is that structure's resident residue (`wt_match` 1.0). FireProt uses one crystal structure per protein, its PDB id matches `protein_id`, and the WT residue matches at 100% of sites (no mutant structure for WT-labelled rows). B4/B5 do not use B-factor or pLDDT (`ctx_bfactor` is not a feature). Megascale AF2 pLDDT does not come from the labels anyway.

## Metrics where we are NOT better (for the final report)
| set / metric | ours | better competitor |
|---|---|---|
| Domainome ρ_dom (all; strict) | B4 0.405, B5 0.411; 0.379 | FoldX 0.430 (CI of B4 − FoldX −0.024 [−0.048, +0.013]); 0.385 |
| Domainome pooled ρ | B4 0.402 | FoldX 0.410 |
| FireProt-test pooled ρ / r | B4 0.472 / 0.450 | tmol-refit 0.495 / 0.477; DDGun3D r 0.488 |
| FireProt-test ρ_prot (9 proteins) | B4 0.578 | FoldX 0.634, tmol-refit 0.678 |
| S669-strict r | B4 0.328 (B3+ 0.330: no engine gain) | DDGun3D 0.347 |
| S461-strict ρ_prot | B4 0.665 | KORPM 0.689 |
| Megascale buried small→large ρ | B5 0.495 | FoldX 0.548 |
| Stabilising precision@5 | B4 0.133 | FoldX 0.219, tmol-refit 0.181 |
| Ssym antisymmetry r / bias | B4 0.23 / +1.39 | KORPM 0.88 / 0.15, DDGun3D 0.99 / 0.04 |

## Missing uncertainty
Only the Megascale-natural column of the main table has CIs. Please add 95% cluster-bootstrap CIs for the FireProt, S669, S461 and Domainome cells (they are already in `metrics.csv`; FireProt r CI ≈ ±0.2). Every FireProt pair is non-significant.
