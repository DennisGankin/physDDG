# Audit 2 — joint models, leakage, inference, claims (2026-10-27, agent: audit, independent of audit 1)

Scope: everything added since `notes/audit_1.md` — the joint B6gj family, B6g/B6x term sets, the evo
extension, AF mode, ENM and differentiable-parameter work, and the D22 model card at the top of
`notes/ladder_physics.md`. All metrics below were **recomputed from the raw prediction files**
(`results/ladder_physics/preds_*.parquet`) and `data/processed/*` with my own metric code, never from
`physddg/eval.py`, `scripts/52` or `scripts/55`. Audit-1 findings are not repeated.

Scripts I added (read-only w.r.t. the project; they print to stdout and write nothing):
- `scripts/99b_audit2_metrics.py` — independent re-computation of every model-card cell (per-protein
  median/mean ρ, pooled ρ, Pearson, RMSE), own subset and strictness logic.
- `scripts/99b_audit2_paired.py` — independent paired cluster-bootstrap (2,000 draws) for the headline deltas.

---

## Verdict summary

### CRITICAL

**C1. The FireProt headline of the joint models is produced by dropping the two rows where they fail.**
The FireProt column of the model card is computed on **338 of 347 rows** — the intersection of the
non-missing predictions of all predictors with ≥95% coverage (`scripts/55_multiassay_report.py:85`,
`keep = [c for c in have if b[c].notna().mean() >= 0.95]`). `tmol-refit` is missing on exactly
**2TRX C32A and C35A**, the catalytic CXXC disulfide pair that audit 1 flagged (item 1). Those are the two
rows where the joint models are catastrophically wrong: measured ΔΔG −4.17 / −4.00 kcal/mol (stabilising),
B6gj predicts **+4.31 / +3.81** because the `ss_broken` weight fires. That weight is **+3.338 ± 1.043**,
exists only in the joint models (it is exactly 0.000 in B6g/B5 — undetermined on Megascale) and is fitted
on a handful of FireProt-train disulfide rows (the card says 8; FireProt-trainval has 29 Cys→X rows in
all). So the entire FireProt difference between the joint and Megascale-only fits is one term estimated
from ~8 rows, whose sign is right for structural disulfides and badly wrong for catalytic ones.

| row set | n | B6gj | B6gjs1 | B6gjaf | B6g | B5 | FoldX | FoldX-clip |
|---|---|---|---|---|---|---|---|---|
| **reported (338)** | 338 | **0.482** | 0.475 | 0.481 | 0.464 | 0.467 | 0.417 | 0.423 |
| our models + FoldX (341) | 341 | **0.388** | 0.378 | 0.388 | 0.453 | 0.458 | 0.411 | 0.417 |
| all FireProt-test rows | 347 | **0.388** | 0.380 | 0.388 | 0.451 | 0.456 | 0.411 | 0.417 |
| 341 minus all 13 2TRX rows | 328 | 0.462 | 0.453 | 0.461 | 0.442 | 0.445 | 0.395 | 0.406 |

Dropping those two rows moves B6gj by **+0.094** and B6g/B5 by only +0.011. On the honest row set the
joint models are the **worst** PhysDDG variants on FireProt and are **below FoldX**. D17 removed the CXXC
exception from the model, but the identical-rows rule silently re-removed the rows the exception existed
for, so the audit-1 fix did not take effect. Every FireProt statement in the model card, RESEARCH_TREE 4.14
and 4.14b ("best structure-only model on FireProt (0.482)") is an artefact of this.
*Fix:* report FireProt on all 347 rows (or on a row set defined by our models + FoldX only), and if 2TRX is
excluded, exclude the whole protein and say so. The `ss_broken` term, fitted on 8 training rows, should
carry a scope flag (catalytic disulfides) derived from train/literature, not from a row drop.

**C2. The shipped default model was selected on test-set numbers, and the card says the opposite.**
The card states "Every hyper-parameter was chosen on Megascale-val + grouped 5-fold CV on train" and lists
one test look (weakness 7). The hyper-parameter tables (`selection.csv`, `b6_selection.csv`,
`cap_selection.csv`, `ablation.csv`) are indeed val/CV-only — but the *model-choice* decisions are not:
- D19 dropped `site_scsc` (i.e. chose B6g over B6x) because "site_scsc hurts crystal sets (FireProt −0.026
  Pearson, S669/S461-strict)" — FireProt-test, S669 and S461 are test sets.
- D22/A2 shipped B6gj because of a gate stated directly on **Domainome, Megascale-test, FireProt-test**
  (`notes/critique_2.md:227`), and shipped B6gjaf "for AF inputs only" on the Domainome test delta.
- `results/ladder_physics/final_selection_ablation.csv` carries a column literally named
  `test_natural_prot (reported, not used)`.
Counting the distinct test evaluations that fed a subsequent decision, from the git history and notes:
B4/B5 gates (`5942d30`), D17 re-issue (`794bb2c`), audit 1 (`58ccce9`), AF treatments / D18 trim
(`6b87e15`), residual analysis (`9d5d501`), B6 (`fecda54`), evo (`eb9ff44`), D19 final models (`7e8989e`),
ENM (`21b6848`), diff-params (`7e4dceb`), critique-2 Phase A + joint fits (`6b347b2`), D22 ship
(`b278ea3`) — **≥ 12 test looks**, at least 5 of which changed what is shipped. With ~25 variants × 8
benchmarks the implied selection optimism on the *marginal* deltas (B6g−B5 +0.005, B6gjaf−B6gj +0.013,
Domainome B6gj−FoldX +0.021) is of the same order as the deltas themselves. The card must say "≥12 test
evaluations, several of which informed the shipped configuration", not "one test look".

**C3. The shipped prediction files in the repository are stale: `final_*` ≠ the shipped models.**
`results/ladder_physics/models/final_stage2.json` is `alias_of: B6gj` (verified: identical weights,
intercept, ref). But the `final_stage2` / `final_stage1` / `final_af` **columns** in every
`preds_*.parquet` are bit-identical to **B6g / B6gs1 / B6gaf** (the superseded D19 Megascale-only models;
mean |Δ| to B6gj 0.137 on Megascale, up to 3.3 kcal/mol on Domainome), and
`results/ladder_physics/metrics.csv` carries `final_*` rows computed from them. The published example
scans in `results/examples/` were also generated with the old weights (their `c_intercept` is −0.2759 =
B6gs1, not −0.2319 = B6gjs1). The headline tables use the `B6gj`/`B6gjs1`/`B6gjaf` columns and are
unaffected, but anything that cites `final_*` from `metrics.csv` or the example CSVs reports the wrong
model. *Fix:* re-score the `final_*` columns and regenerate `results/examples/`, or delete them.

### IMPORTANT

**I1. "Identical rows per column" is false for three of the nine columns.** `scripts/55` applies
per-model strictness (`strict_for`, line 42) but then reports all models in one column:
S669-strict is n = 488 for Megascale-only models and n = 470 for joint ones; S461-strict 290 vs 285;
Domainome DDGun3D is on 45,351 rows / 47 domains against 509,388 / 493 for everything else. The card's
parenthetical "(identical rows per column)" is wrong there. On **matched** rows (BOTH-strict, n = 470 /
285) the picture changes in the joint models' favour, so this is a reporting error, not inflation:

| matched rows | B3+ | B5 | B6g | B6gs1 | B6gj | B6gjs1 | DDGun3D | KORPM |
|---|---|---|---|---|---|---|---|---|
| S669 both-strict (470) | 0.330 | 0.315 | 0.316 | 0.328 | 0.344 | **0.351** | 0.347 | 0.194 |
| S461 both-strict (285) | 0.569 | 0.556 | 0.549 | 0.569 | 0.568 | **0.579** | 0.552 | 0.543 |

Note also that `main_table.md` puts *all* models on the both-strict rows and therefore reports B6g
S669-strict **0.316**, while `multiassay_summary.csv` and the model card report **0.336** for the same
model. Two tables in the repo disagree; critique_2 §1 quotes the first, the card the second.

**I2. The Domainome column — the basis of the generalisation claim — has no homology filter at all.**
151/517 Domainome domains have a Megascale-train homologue and 90/517 a FireProt-trainval homologue
(`data/processed/homology_domainome.csv`), yet the reported Domainome and Dom-flag-kept columns use
`strict_rule = all`, while Megascale/S669/S461 are reported strict. Recomputed:

| Domainome subset | domains | B5 | B6g | B6gj | B6gjaf | FoldX | B3+ |
|---|---|---|---|---|---|---|---|
| all (as reported) | 493 | 0.412 | 0.418 | 0.451 | 0.459 | 0.430 | 0.389 |
| no Megascale-train homologue | 347 | 0.386 | 0.391 | 0.415 | 0.419 | 0.385 | 0.369 |
| no MS-train **and** no FP-trainval homologue | 320 | 0.381 | 0.385 | 0.404 | 0.405 | 0.378 | 0.366 |

The B6gj − FoldX margin survives (+0.021 all → +0.027 strict) but **every absolute Domainome number falls
by ~0.045** once homologues are removed. The card should quote the strict column next to the headline, as
it does for Megascale. (`ladder_physics.md` says Domainome strict is "347 → 339 domains"; the correct
pre-intersection counts are 366 → 339.)

**I3. `abundance_dms` has never been homology-screened.** `data/processed/abundance_dms.csv` has no
`homolog_in_*` columns and the assay does not appear in `clusters.csv`, so the abundance column
(B6gj 0.482, quoted as a win and as a significant B6gj−B5 gain) has no leakage check against either
training set. Seven human proteins (P60484 PTEN, P51580 TPMT, P11712 CYP2C9, Q9NV35 NUDT15, O60260 PRKN,
P35557 GCK, P01116 KRAS). Run the same mmseqs2 screen as `scripts/14_homology.py` before quoting it.

**I4. The "crystal input is never attenuated" guarantee holds, but only through a fragile indirection.**
The property **is true on every shipped path** — verified end to end: `PhysDDG(final_af.json, mode="xtal")`
on `1JNX_X` gives `c_scale_plddt == 0` for every variant, while the same weights in `mode="af"` reach
1.10 kcal/mol. But the guard is not in `PhysDDG.score` (`physddg/predict.py:427-429` applies the
attenuation whenever the model has `scale_kind`, with no reference to `self.mode`). It lives in
`physddg/features_extra.py:614-617`, as a string match on a `struct_source` column that `_run` happens to
set (`predict.py:490`), with a **B-factor fallback when `struct_source` is absent or NaN**:
`is_af = x_mean_plddt.between(30,100) & (x_plddt.max() <= 100)`. That fallback misclassifies any crystal
with mean B-factor 30–100 Å² and max ≤ 100 as an AF model: **21/94 S669, 9/48 S461, 5/99 FireProt and
8/357 Ssym structures**. Calling `PhysDDG.features()` directly (the documented building block) without
`struct_source` reproduces the bug — on 1JNX it yields `flag_lowplddt` 0.553–0.794 instead of 0. The batch
tables are safe because all of them carry `struct_source` (`af2_model_tsuboyama` / `xtal` / `af`).
*Fix:* gate on `self.mode` in `score()`, make `struct_source` required, and drop the B-factor heuristic.
The test that is supposed to cover this (`tests/test_mutate.py:238`) uses a single Megascale AF2 model as
its "crystal", so it would not catch a regression in either direction.

**I5. Parameter count is understated.** The card says "20 fitted weights + 20 reference energies = 40
numbers fitted to ΔΔG (21 weights for the AF model)". Counting from the JSONs: B6gj = 20 weights + 20 ref
+ **1 intercept** = 41. The *shipped fast default* `final_stage1.json` (B6gjs1) has **22** weights
(`dref_apolar`, `dref_polar` additionally) → 43. `final_af.json` has 21 weights + 20 ref + intercept = 42
**plus a complete nested base model** (`scale_base`: another 20 weights + 20 ref + intercept) that must be
evaluated at inference → **83 numbers fitted to ΔΔG**. Also unstated: the joint fit additionally estimated
2 FireProt nuisance parameters and 20 Megascale δ_ref values from ΔΔG; dropping them at inference does not
make them unfitted. The disclosure of the PDB-derived tables and frozen CHARMM19/EEF1 constants is good and
should be kept verbatim.

**I6. Confidence intervals are quoted to three decimals from 200 bootstrap draws.** `scripts/55` uses
`NBOOT = 200` and `scripts/52` defaults to `--n-boot 200`; a 2.5% percentile from 200 replicates is the
5th order statistic. For the headline this turns out not to matter — my 2,000-draw rerun gives
+0.0627 [+0.0081, +0.0955] against the card's +0.063 [+0.008, +0.095] — but the bounds of the *marginal*
cells (B6gjaf − B6gj, Domainome B6gj − FoldX) are exactly the ones a reviewer will probe. Re-run those at
≥2,000 draws, or round every bound to two decimals.

**I7. The paired xtal/AF set is 64% joint-model training data.** `data/processed/paired_af.csv` is 3,403
FireProt rows of 4,756, of which 2,669 are FireProt-**train** and 387 val. The 5.1a result ("the AF model
is never worse than the crystal") was obtained with Megascale-only models (B4/B5/P1) and is fine as such,
but it must not be restated for the shipped joint models without removing FireProt-trainval. Separately,
"never worse" is a point-estimate claim: of the seven rows in `af_mode.md` §1, the two for fitted models
on the full set are +0.003 [−0.006, +0.015] and +0.016 [−0.002, +0.030] — CIs touching 0. Only the
**unfitted** energy shows significant AF-over-crystal gains.

**I8. The shipped model files are not produced by any script, and the committed example outputs are
stale.** `results/ladder_physics/models/final_{stage1,stage2,af}.json` and `ref_*.json` carry
`alias_of`/`release`/`inference_note` keys that `physddg.fit.save_model` does not write, and no script in
`scripts/` creates them — they were copied by hand. README:130 says "Everything in `results/` comes from a
numbered script", which is therefore false for exactly the files the predictor loads by default. Also,
`scripts/52_ladder_physics.py --stage all` (README step 9) does **not** run `caps`, `b6` or `final`;
`--stage fit` does produce `models/B6gj*.json`, so the shipped weights are recoverable, but the aliases
and `final_selection_ablation.csv` are not. Finally, the committed example scans in `results/examples/`
(README:125-127) were generated with the **pre-D22** weights — their `c_intercept` is −0.2759 (B6gs1) /
−0.2982 (B6gaf), not −0.2319 (B6gjs1) / −0.2191 (B6gjaf). (The newer fresh-clone fixtures in `examples/`
*are* current: `c_intercept` −0.2319.)

*Speed (checked, now correct).* Three different speed tables existed during this audit; commit `68506c0`
re-ran the benchmark and README.md:112-123 now agrees with `results/mutate/speed_benchmark.csv`. I
re-verified a slice on a shared RTX 4090: stage 1 7a38A 0.89 s / 6q62A 4.02 s, stage 2 2.34 s / 8.94 s vs
the committed 0.86 / 3.61 / 1.84 / 8.18 — reproduced. RESEARCH_TREE 6.2 still carries the older,
inconsistent numbers (0.8 / 2.5 / 4.8 / 13.1 s; Domainome 63 s / 203 s) and should be updated or dropped.

### MINOR

- **M1. README (fixed during this audit, one residue left).** Commits `68506c0`/`0e8507c`/`ffbe07f`
  landed while I was working and corrected the pre-D22 provenance line and the Modes table. What the new
  README inherits from the model card: the FireProt 0.482 cell (C1), "identical rows per column" (I1),
  "chosen after one test look" (C2) and "20 weights + 20 reference energies" (I5 — the *default* shipped
  model, `final_stage1`, has 22 weights). Those four must be fixed in both files together.
- **M2. FireProt-test is not family-clean for the joint models.** 1MGR has a 37.2% identity /
  58.8% coverage hit to FireProt-**val** 1A2P (barnase/binase), and 1RRO/1RTP share a single-linkage
  family with train 1IGV — 14/347 rows (4%). Small, but FireProt-test is reported with no strict column at
  all while S669/S461 are.
- **M3. Ssym.** Independently confirmed: 9/15 Ssym proteins are exact or substring sequence matches to
  FireProt-trainval, 660/684 rows carry a FireProt-trainval homologue. The decision to use Ssym only as an
  antisymmetry diagnostic is correct and should be restated in the report.
- **M4. The gap-to-DL percentages are cross-protocol.** "Closes roughly half of the FoldX→ThermoMPNN gap",
  the evo note's 48% / 85%, and the 0.77 / 0.50 reference points are literature numbers from other papers'
  evaluations (SPURS paper; ThermoMPNN paper) compared against our own per-protein protocol. Neither model
  was run in this repo, and LITERATURE.md:458-460 itself says the Domainome DL figure is not robust. Quote
  them as "literature reference points, different evaluation", never as a measured fraction.
- **M5. Remaining reproducibility gaps** (the rewritten README fixed most of the ordering and now lists
  `53`, `10a`–`10d`, `11x`, `12x`). Still missing: `54_doubles_additivity.py` and
  `55_multiassay_report.py` — the latter *is* the script that produces the model card's headline table and
  it collides on the "55" prefix with `55_phys_features.py`, which is the one the README names.
  `scripts/59_speed_benchmark.py` depends on `/workspace/data/yeast_af/pdbs/*` and a PISCES mirror outside
  the repo (the four crystal bench PDBs are committed); its docstring still says "stage 1 (B4) and
  stage 2 (B5)".
- **M6. `_atten_col` docstring vs code** (`scripts/52_ladder_physics.py:850-853`): the docstring says
  "1 - pLDDT/100" and the code returns `d["flag_lowplddt"]`. They agree numerically
  (`features_extra.py:639`) but only via the `is_af` gate described in I4 — worth a comment.
- **M7. "The sign constraints are not binding"** (card, Terms section) is contradicted two lines later by
  `pro_nhhb | 0.000 (at bound)`. One constraint *is* binding; say "binding only for `pro_nhhb`, which the
  fit drives to zero". All 20 B6gj weights and SEs otherwise reproduce the card exactly from
  `final_stage2.json`, as do the δ_ref values (Cys −0.546, next largest Arg +0.188).
- **M8. Megascale pLDDT coverage.** 90 of 298 Megascale structures have an all-zero B-factor column, so
  the AF attenuation term is effectively fitted on the 208 structures that carry pLDDT. "Megascale
  structures are AF2 models" is right, but the pLDDT feature is not available for 30% of them.

---

## Q1 — independent re-computation of the model card

`scripts/99b_audit2_metrics.py`. Identical rows within each column = non-null for every listed predictor
with ≥95% coverage; strictness per model (joint models also exclude FireProt-trainval homologues).
**Every cell of the model-card table reproduces to 3 decimals except the FireProt column (C1).**

| view | n | n_prot | model | med ρ | mean ρ | pooled ρ | Pearson | RMSE |
|---|---|---|---|---|---|---|---|---|
| MS nat | 22,159 | 21 | B6gj | **0.6916** | 0.6885 | 0.6585 | 0.6935 | 0.788 |
| | | | B6gjs1 | 0.6760 | 0.6788 | 0.6490 | 0.6886 | 0.793 |
| | | | B6gjaf | 0.6995 | 0.6923 | 0.6614 | 0.6959 | 0.785 |
| | | | B6g | 0.6919 | 0.7016 | 0.6676 | 0.7012 | 0.776 |
| | | | B5 | 0.6866 | 0.6958 | 0.6653 | 0.7006 | 0.777 |
| | | | FoldX | 0.6288 | 0.6288 | 0.5922 | 0.4447 | 3.218 |
| | | | FoldX-clip | 0.6288 | 0.6288 | 0.5922 | 0.5962 | 1.845 |
| MS nat-strict | 15,733 | 14 | B6gj / B6g / B5 | 0.6954 / 0.7016 / 0.6974 | 0.6922 / 0.7047 / 0.7013 | | 0.7007 / 0.7073 / 0.7069 | 0.793 / 0.780 / 0.781 |
| | | | FoldX-clip | 0.6397 | 0.6258 | 0.5908 | 0.5965 | 1.868 |
| MS all | 28,172 | 28 | B6gj / B6g / B5 | 0.6819 / 0.6916 / 0.6852 | 0.6740 / 0.6927 / 0.6872 | 0.6294 / 0.6441 / 0.6416 | 0.6752 / 0.6865 / 0.6862 | 0.792 / 0.776 / 0.776 |
| | | | FoldX-clip | 0.6331 | 0.6207 | 0.5686 | 0.5898 | 1.843 |
| FireProt | **341** | 25 (9 eval) | B6gj / B6g / B5 / FoldX-clip | 0.502 / 0.577 / 0.584 / 0.634 | | 0.473 / 0.478 / 0.483 / 0.449 | **0.388** / 0.453 / 0.458 / 0.417 | 2.11 / 2.06 / 2.06 / 2.29 |
| S669-str | 470 / 488 | 82 / 87 | B6gj / B6gjs1 / B6g / B5 | | | | 0.3436 / 0.3513 / 0.3365 / 0.3337 | |
| S461-str | 285 / 290 | 42 / 43 | B6gj / B6gjs1 / B6g / B5 | | | | 0.5677 / 0.5789 / 0.5519 / 0.5589 | |
| Domainome | 509,388 | 493 | B6gj / B6gjaf / B6g / B5 / FoldX | 0.4507 / 0.4588 / 0.4183 / 0.4121 / 0.4302 | 0.4244 / 0.4293 / 0.4085 / 0.4036 / 0.3941 | | | |
| Dom flag-kept | 441,283 | 493 | B6gj / B6gjaf / B6g / FoldX | 0.4778 / 0.4813 / 0.4491 / 0.4584 | | | | |
| abundance | 37,947 | 7 | B6gj / B6gjaf / B6g / B5 | 0.4819 / 0.4906 / 0.4687 / 0.4691 | | | | |

Row alignment: prediction keys are unique, `domain_origin`, `plddt` and `ddg` agree between the prediction
files and `data/processed`, and the per-protein win counts against FoldX on the 21 natural Megascale
proteins reproduce exactly (B6gaf 20, B6g 19, B6gj 19, B6gjaf 19, B5 18, B6gjs1 18, B3+ 16, tmol 15,
KORPM 3, DDGun3D 0).

### Paired cluster-bootstrap, 2,000 draws (`scripts/99b_audit2_paired.py`)

Own code, 2,000 draws, clusters resampled with replacement (`scripts/99b_audit2_paired.py`).

**The headline reproduces exactly.** Megascale-test natural, B6gj − FoldX-clipped, n = 22,159:

| metric | Δ | 95% CI | card |
|---|---|---|---|
| per-protein median ρ | **+0.0627** | [+0.0081, +0.0955] | +0.063 [+0.008, +0.095] ✓ |
| per-protein mean ρ | +0.0596 | [+0.0362, +0.0783] | – |
| pooled ρ | +0.0663 | [+0.0456, +0.0808] | – |
| Pearson | +0.0973 | [+0.0577, +0.1267] | 0.694 vs 0.596 ✓ |
| RMSE | −1.0564 | [−1.2682, −0.8624] | 0.79 vs 1.85 ✓ |

Against *raw* FoldX the Pearson margin is +0.2488 [+0.1895, +0.2927] and RMSE −2.43 — the clipped column
is the one to quote, as `foldx_protocol.md` already says.

**FireProt, the same comparison on three row sets** (Pearson, 2,000 draws) — this is C1 in one table:

| row set | B6gj − B5 | B6gj − B6g | B6gj − FoldX-clip |
|---|---|---|---|
| all 347 | **−0.068 [−0.238, +0.032]** | −0.063 [−0.229, +0.033] | −0.029 [−0.233, +0.109] |
| our models + FoldX (341) | −0.070 [−0.242, +0.030] | −0.065 [−0.231, +0.033] | −0.029 [−0.233, +0.109] |
| **reported (338)** | **+0.015 [−0.014, +0.034]** | +0.018 [−0.008, +0.036] | +0.059 [−0.049, +0.125] |

The sign of every FireProt delta flips with the two dropped rows. None of them is significant on any row
set, which is the one thing audit 1 already said and which the card still does not say in its FireProt
column.

**Cross-check of `multiassay_deltas.csv`** (their 200 draws vs my point estimates): all point estimates
agree to 4 decimals. The ones the report will lean on:
B6gj − FoldX-clip Domainome **+0.0207 [−0.0024, +0.0468]** (CI touches 0, as the card says);
flag-kept +0.0194 [−0.0019, +0.0335]; B6gj − B5 Domainome +0.0386 [+0.0193, +0.0524];
B6gjaf − B6gj Megascale natural +0.0077 [−0.0224, +0.0344]; B6g − FoldX Domainome **−0.0118** (FoldX ahead);
B6gjs1 − B6g Megascale-all −0.0178 [−0.0606, −0.0029].

---

## Q2 — leakage in the joint fits

- **FireProt-train/val vs FireProt-test:** 0 shared proteins, 0 exact `(protein, resnum, wt, mut)`
  duplicates, 0 shared `cluster30`/`cluster50`. 3 test proteins share a single-linkage `family` with a
  trainval protein (1MGR↔1A2P at 37% identity, 1RRO/1RTP↔1IGV): 14/347 rows (M2).
- **FireProt-trainval vs the external sets** (homology columns, row counts): S669 51/669 rows, S461
  31/461, **Ssym 660/684 (96.5%)**, Megascale-test 4,582/28,172 rows in 5 of 28 domains. Sequence-level
  check: 8 of 15 Ssym proteins are *exact* sequence matches to FireProt-trainval and a 9th is a substring;
  S669/S461 have one (3DV0_I); Megascale has three (1JIC, 1UZC, 1YU5 — all in **train**, not test).
- **Cells of the model-card table that are NOT clean for the joint models:**
  | cell | status |
  |---|---|
  | MS nat | not homology-clean (9/28 domains, disclosed); same for all models |
  | MS nat-strict | clean (both strictness rules coincide, n = 15,733) |
  | MS all | not homology-clean |
  | **FireProt** | row set manipulated (C1); 4% family overlap with FireProt-val (M2); *not* an out-of-assay test for joint models at all |
  | S669-str / S461-str | clean for the joint models, but **not on the same rows** as the B6g/B5 cells (I1) |
  | **Domainome / Dom flag-kept** | **not clean**: 29% of domains have a Megascale-train homologue, 17% a FireProt-trainval homologue (I2) |
  | **abundance** | **unscreened** (I3) |
  | Ssym (reported elsewhere) | not a test for joint models (96.5% overlap) — the note already says so |
- **Domainome / abundance never enter fitting:** confirmed. `stage_fit`
  (`scripts/52_ladder_physics.py:369-404`) builds its training frame only from `train_tables()` =
  `megascale_train_table('train'|'val')` + `fireprot_train_table()` (lines 208-210); `domainome` and
  `abundance` appear in script 52 only inside `SUBSETS`/evaluation (lines 507-510, 607, 986-987) and
  nowhere in `physddg/fit.py`. `fit_attenuated` (line 857) also fits on the training frame only, and its
  grouped CV refits the base per fold.
- **Hyper-parameter selection** (`selection.csv`, `b6_selection.csv`, `cap_selection.csv`, `ablation.csv`)
  contains no test columns — val + grouped CV only. The *model-level* decisions do not (C2).

---

## Q3 — inference-time correctness

- **Nuisance is not applied at inference.** `physddg.fit.predict` (`physddg/fit.py:353-364`) uses only
  `intercept`, `weights`, `ref` and `sym_pair`; the `nuisance` block is never read. `final_stage2.json`
  carries `inference_note: "nuisance … is NEVER applied at inference"`, and
  `results/ladder_physics/models/README.md` says the same. Empirical confirmation: the FireProt nuisance
  scale is 1.753, and the regression slope of measured ΔΔG on B6gj prediction over FireProt-test is 1.26
  (B6g 1.48, B5 1.50) — i.e. the predictions are on the un-rescaled Megascale scale.
- **CLI round-trip (3 proteins, 167 variants, `1RN1`/`1KF2`/`1FTG`):**
  `scripts/physddg predict --table … --stage 2` reproduces the stored **B6gj** column to max 7.3e-6
  (stage 1 vs B6gjs1: max 2.4e-6 — float32 engine noise, not a mismatch). It does **not** reproduce the
  stored `final_stage2` column (max 0.40, mean 0.084) because that column is stale (C3). Per-term `c_*`
  contributions sum to `ddg` to 9e-16.
- **WT→WT = 0:** exact for all 18 tested self-substitutions (`PhysDDG.score` forces it,
  `physddg/predict.py:432`). Note this is enforced, not emergent — `fit.predict` alone would return the
  intercept (−0.212).
- **Sign convention:** 8 buried (RSA < 0.10) I/L/V/F/M sites of 1RN1 × {D,E,K,R} = 32 variants, mean
  **+2.76 kcal/mol, min +1.53, 100% positive**; the same sites → Ala +1.94; exposed polar → Ala +0.21.
  Correct and well separated. Predictions are bitwise reproducible across repeated calls.
- **Crystal path and pLDDT attenuation:** holds on the public path. `PhysDDG(final_af.json, mode="xtal")`
  on the crystal `1JNX_X` returns `c_scale_plddt == 0` for every variant; the same weights with
  `mode="af"` reach 1.10 kcal/mol. The guard is indirect and has a fragile fallback — see I4.
- **AF mode on a Domainome construct** (`A0A2R8Y422_PF00240_2`, 1,300 variants):
  `scripts/physddg predict --mode af` reproduces the stored **B6gjaf** column exactly (max 0.0000) and
  differs from the stale `final_af` column by mean 0.13 / max 0.57. `withheld` = 7.8% of variants.
- **CLI flags:** sensible and consistent with the docs (`--mode`, `--crop`, `--no-trim`, `--stage`,
  `--weights final|ref`, `--gpus`, `--terms`). `--mode af` defaults to stage 2; `--mode xtal` to stage 1.

---

## Q4 — claims audit

Statements in `README.md`, the model card and `RESEARCH_TREE.md` that the numbers do not support, beyond
the findings above:

1. **"best structure-only model on FireProt (0.482)"** (card; tree 4.14, 4.14b) — reverses on the full row
   set (C1).
1b. **"B6gj is the only configuration that is ≥ B5 on *every* assay simultaneously"** (card §A2; tree
   4.14) — **false by the project's own `multiassay_deltas.csv`**: B6gj − B5 is −0.0033 on Megascale-all
   and −0.0020 on Megascale natural-strict. The card contradicts itself three lines later ("−0.003
   pooled, −0.007 natural-strict"; the natural-strict delta is −0.002, not −0.007). Note also that B6gj
   is **not** better than the Megascale-only reference B6g on any Megascale subset (−0.0003 nat,
   −0.0062 nat-strict, −0.0096 all) and that B6gjs1 − B6g on Megascale-all is −0.0178 [−0.0606, −0.0029],
   i.e. the *fast shipped default* is significantly worse than the superseded reference fit there.
1c. **abundance_dms significance.** The card reports B6gj − B5 as "+0.013 [+0.000, +0.013]" and calls it
   significant. The upper bound equals the point estimate to four decimals (+0.0127 / +0.0127) — the
   cluster bootstrap of a median over **7 proteins** is degenerate. Do not call any abundance_dms delta
   significant.
2. **"identical rows per column"** (card) — false for S669-str, S461-str and the Domainome DDGun3D cell (I1).
3. **"Every hyper-parameter was chosen on Megascale-val + grouped 5-fold CV"** and **"one test look"**
   (card; weakness 7) — ≥12 test evaluations, ≥5 of which changed the shipped configuration (C2).
4. **"40 numbers fitted to ΔΔG"** — 41 / 43 / 83 for the three shipped models (I5).
5. **"Everything in `results/` comes from a numbered script"** (README:130) — not true for
   `models/final_*.json` / `ref_*.json` (I8).
6. **AF-mode claims.** B6gjaf − B6gj is +0.013 [−0.012, +0.033] on Megascale (tree 4.14b) and the
   `scale_plddt` weight is −0.853 ± 0.621 (CI includes 0). The card bolds B6gjaf in five of nine columns.
   Nothing here supports an AF-*specific* correction; the AF-mode value is the crop + trim + **flagging**
   rule, and "flagging beats correcting" is the defensible claim. The structure-source confound (fitted on
   Tsuboyama AF2 models) is properly disclosed in `foldx_protocol.md` and card weakness 6 — keep it.
7. **Speed.** The README table is now correct and I reproduced a slice of it. Caveat for the report: per
   *mutation* we are at parity with KORPM (0.44 ms on one CPU core vs ~0.8 ms on one GPU for stage 1) —
   the README table shows this honestly — so any "fastest" or "orders of magnitude faster" framing must be
   scoped to FoldX and must state that it compares GPU-seconds with CPU-hours. RESEARCH_TREE 6.2 still
   carries superseded numbers.
8. **"Closes ~half the gap to DL"** — cross-protocol (M4).
9. **"The AF model is never worse than the crystal"** — point estimates with CIs touching 0 for fitted
   models (I7).
10. **Physical-parameter framing.** The card's wording ("tables *estimated from structures, never from
    ΔΔG*", "~15 frozen CHARMM19/EEF1 constants with literature provenance") is accurate and well judged.
    Two riders: the repulsion radii are scaled ×0.9 (D14, calibrated on structures), and E3 showed the
    literature parameters move by ×/÷1.6 when fitted — report E3 as the negative result it is.

### What the final report must **not** claim

- That PhysDDG is the best structure-only model on FireProt, or that the joint fit improves FireProt at
  all (on all 347 rows it is 0.388 vs B5 0.456 and FoldX-clipped 0.417).
- That any table is on identical rows across models, unless the strictness rule is fixed across models.
- That the Domainome or abundance results are homology-clean, or that "the physics generalises across
  assays", until the strict Domainome column and an abundance homology screen are reported.
- That hyper-parameters and the model were selected without looking at test data, or that there was one
  test look.
- That the model has 40 ΔΔG-fitted parameters.
- Any significance for B6g − B5, B6gaf/B6gjaf − B6gj, Domainome B6g − B5, the ENM term, the E3
  differentiable fit, or any FireProt pairwise comparison.
- An AF-*specific* correction that improves accuracy.
- Best-in-class antisymmetry or stabilising-mutation ranking (unchanged from audit 1).
- A measured fraction of the FoldX→ThermoMPNN gap, or any comparison to ThermoMPNN/SPURS/ESM-MSR as if
  they had been run here.
- "Better than PhysDDG v1.2" — still unmeasured (critique_2 §3.1 is unresolved; no v1.2 predictions exist
  in this workspace).
- "Fastest" / "orders of magnitude faster" without naming FoldX as the comparator (KORPM is at parity per
   mutation, on one CPU core), and without stating that GPU-seconds are being compared with CPU-hours.

### What the report **may** claim (per D20, after the fixes above)

- Beats FoldX 5.1 on Megascale-test on rank (+0.063 [+0.008, +0.095], independently reproduced at 2,000
  draws), Pearson (0.694 vs 0.596 clipped) and RMSE (0.79 vs 1.85) simultaneously, on identical rows and
  identical structures, on all three Megascale subsets, winning on 19 of 21 natural proteins — with the
  caveat that the structures are Tsuboyama AF2 models and both methods were run on them.
- The joint fit buys Domainome and abundance_dms at a Megascale cost within ±0.01 (+0.005 natural,
  −0.002 natural-strict, −0.003 all — i.e. "no measurable cost", not "no cost"), and the margin over
  FoldX on Domainome survives homology filtering (+0.027 on 320 strict domains) although its CI touches 0.
- On crystal benchmarks the full engine does not beat the 10-parameter B3+ descriptor baseline.
- No deep learning; 41–83 ΔΔG-fitted numbers depending on the shipped variant, plus PDB-derived tables;
  all weights carry the physically expected sign.
- GPU throughput: ~1,300 variants/s stage 1, ~500/s stage 2 end-to-end on one RTX 4090 (measured), vs
  FoldX CPU-hours per scan and KORPM 0.44 ms/mut/core.
- The AF characterisation: construct cropping matters, the crystal is the fragile input for the *unfitted*
  energy, and **flagging beats correcting** (14% withheld → +0.038 ρ_dom).

---

## Q5 — reproducibility of three key results

1. **B6gj Megascale-test row.** Chain exists and is correct: `11_build_megascale.py` → `14_homology.py` →
   `55_phys_features.py` (cached `data/features/phys_s2_megascale.parquet`) → `52_ladder_physics.py
   --stage fit` → `55_multiassay_report.py`. All inputs present. I did not refit (hours), but I
   re-derived every reported metric from `preds_megascale.parquet` independently and they match (Q1).
   **Gap:** `--stage all` omits `caps`, `b6` and `final`, so the documented invocation does not rebuild
   the shipped models; the README does not mention sub-stages at all (M5).
2. **Domainome B6gjaf.** Chain exists: `22_domainome_structures.py` → `23_domainome_table.py` →
   `55_phys_features.py` → `52 --stage fit` → `55_multiassay_report.py`, with the flag-kept subset needing
   `x_dist_term` from `features_extra.site_table` (cached in `data/features/extra_sites_domainome.parquet`
   — present). Verified end-to-end on one construct: `scripts/physddg predict --mode af` reproduces the
   stored B6gjaf prediction exactly for all 1,300 variants of `A0A2R8Y422_PF00240_2`. Reported metric
   reproduced (0.4588 all / 0.4813 flag-kept).
3. **Speed benchmark.** `scripts/59_speed_benchmark.py` runs, but `prepare()` needs a PISCES mirror and
   `AF_PROTS` points at `/workspace/data/yeast_af/` outside the repo (the four `data/structures/bench/*.pdb`
   are committed, so the crystal part is reproducible). Re-ran the slice: stage 1 7a38A 0.89 s /
   6q62A 4.02 s, stage 2 2.34 s / 8.94 s — matches the committed CSV (0.86 / 3.61 / 1.84 / 8.18) and the
   current README.

`pytest tests/`: **108 passed** in 322 s (one NumPy-writability warning from `physddg/fit.py:125`). Two
tests give less assurance than their names suggest: `test_crystal_input_never_attenuated`
(`tests/test_mutate.py:238`) uses a Megascale **AF2 model** as its crystal input, and the shipped-model
reproduction tests compare against the current `final_*.json`, not against the stored `final_*` prediction
columns — which is why C3 went unnoticed.
