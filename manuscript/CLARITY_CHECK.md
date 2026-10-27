# Clarity edit and number check

Two things are recorded here: (1) the clarity pass over all prose in `main.tex` and `supplementary.md`,
following `manuscript/edit-for-clarity-SKILL.md`, and (2) a check that every number in `main.tex` matches the
model card (`notes/ladder_physics.md`) and the result tables it is derived from.

## 1. Changes (by pattern)

The drafts were written with the six patterns in mind, so the pass found few instances. Those found were
fixed; nothing else was rewritten.

- **Pattern 1 (rare word combinations):** 4 fixes.
  "antisymmetry is the free win a physics model should get" → "a physics model should obtain antisymmetry for
  free"; "a burial-dependent calibration wearing a pLDDT label" → "...that carries a pLDDT label"; "the engine
  is not decoration on a statistical potential" → "...not redundant given the statistical potential" (and the
  converse clause); "both models benefit from squashing" → "compressing the extreme values helps both models".
- **Pattern 2 (inanimate subject, human verb):** 3 fixes.
  "the data prefers those signs on its own" → "those signs come out of the fit even without the constraints";
  "the sign-free weight wants to amplify buried predictions" → "refitting it without the sign constraint gives
  +0.82, which amplifies buried predictions instead of attenuating them"; "the abundance assay penalises
  exposed substitutions ... while Megascale penalises ..." → "exposed substitutions ... score worst in the
  abundance assay ... whereas in Megascale the worst substitutions are those to P/G/D".
  Left alone deliberately: "a fixed backbone over-penalises small-to-large substitutions", "the model ranks",
  "the engine returns", "the fit drives the weight to zero" — these verbs do not require a living subject in
  this field's usage.
- **Pattern 3 (nouns used as verbs):** 0 instances found.
- **Pattern 4 (dropped "that"):** 7 insertions.
  "the quantity we need", "a confound we cannot remove", "the contribution we can support", "the residue it
  contains", "the sites it was meant to fix", "the code they audited", "a different assay than the one FoldX
  was tuned on".
- **Pattern 5 (three or more stacked nouns):** 0 new fixes. Standing noun stacks are established technical
  names and were kept: "per-protein median Spearman", "cluster-bootstrap confidence interval",
  "unfolded-state reference energies", "backbone-dependent rotamer library", "site-saturation mutagenesis".
  Where a stack was avoidable it was written with a preposition from the start ("the logic of how backup paths
  are handled" style), for example "the environment statistical potential" rather than "PDB environment
  potential table entries".
- **Pattern 6 (long subject-verb gap):** 2 fixes.
  The abstract's results sentence was split so that the subject and verb are adjacent ("the per-protein median
  Spearman rho is 0.692 ... on identical rows and identical structures (22,159 variants)"); one ablation
  sentence was restructured so the list of groups follows the verb rather than separating it from its subject.

No numbers, citations, code, file names or claims were changed by the clarity pass.

## 2. Numbers in `main.tex` against the model card

Every value in the abstract, the results, Table 1 and Table 2 was re-read from
`results/ladder_physics/multiassay_summary.csv`, `multiassay_deltas.csv`, `paired.csv`, `metrics.csv`,
`final_selection_ablation.csv`, `models/final_*.json` and `models/SHIPPED.json`, not copied from prose.

Checked and matching (selection):

| claim in main.tex | source | status |
|---|---|---|
| per-assay Table 1 (all 63 cells) | `multiassay_summary.csv` | matches to 3 decimals |
| headline +0.063 [+0.006, +0.120]; strict +0.062 [+0.009, +0.119]; all 28 +0.058 [+0.017, +0.121] | `multiassay_deltas.csv` | matches |
| Pearson 0.701 vs 0.596; paired +0.105 [+0.068, +0.135]; RMSE 0.776 vs 1.845 | `metrics.csv`, `paired.csv` | matches |
| 19 of 21 proteins better than FoldX | recomputed per protein from `preds_megascale.parquet` by `figures/make_figures.py` | matches (19/21) |
| default − B3+ +0.044 [+0.013, +0.072]; − tmol +0.070 [+0.032, +0.097] | `paired.csv` | matches |
| 41 / 43 / 83 fitted numbers | `models/SHIPPED.json` | matches |
| all 20 weights and their SEs | `models/final_stage2.json` | matches |
| drop-one-group values | `final_selection_ablation.csv` | matches |
| Domainome pLDDT bands 0.474 / 0.311 / −0.013 / −0.050 | recomputed from `preds_domainome.parquet` (Fig. 4) | matches the card |
| speed 0.9 / 3.6 / 10.0 s; Domainome 71 s on 4 GPUs | `results/mutate/speed_benchmark.csv`, README | matches |
| training set 215,731 rows + 11,363 intervals, 239 domains, 96 clusters | `models/final_stage2.json` train block | matches |

### Discrepancies found in the notes themselves

1. **The headline confidence interval is quoted with two different lower bounds inside the project.**
   `multiassay_deltas.csv` and `REPORT.md` give $+0.063\ [+0.006, +0.120]$; the older D19 paired-bootstrap table
   inside `notes/ladder_physics.md` gives $+0.063\ [+0.008, +0.120]$; audit 2's independent 2,000-draw rerun of
   the *joint* model gives $+0.0627\ [+0.0081, +0.0955]$. These are different bootstrap runs of the same
   comparison at 200 or 2,000 draws. The manuscript quotes the result table, $[+0.006, +0.120]$, and the
   supplement notes audit 2's rerun. The project's own audit recommends rounding such bounds to two decimals,
   which would make all three read $[+0.01, +0.12]$ or $[+0.01, +0.10]$.
2. **Two S669-strict numbers for the same model exist in the repository.** `multiassay_summary.csv` and the
   model card report 0.336 on 488 rows (the strictness rule of the Megascale-only family); `main_table.md`
   reports 0.316 for the same model on the 470 rows that are strict for both families. Audit 2 flagged this
   (item I1). The manuscript uses 0.336 with the row count stated, and gives the matched-row table (0.316) in
   Supplementary S21.
3. **The pLDDT attenuation weight is quoted as −0.90, −0.902 ± 0.659, −0.853 ± 0.621 and −0.97 in different
   sections of the card**, which correspond to different fits (B6af, B6gaf, B6gjaf). The shipped
   `final_af.json` has −0.966 with a bootstrap standard error of 0.651 and a 2.5–97.5 percentile range of
   −2.52 to −0.23 over 200 draws; the manuscript uses the value from the shipped file.
4. **The evolution-mode gain on the strict natural Megascale subset is summarised as "+0.014" in the card**,
   which is the *mean* per-protein change; the median change, which is the metric used everywhere else, is
   −0.006 [−0.037, +0.080], with 9 of 14 proteins improving. The manuscript reports both and does not call the
   strict-subset gain established.
5. **`REPORT.md` and the card describe the AF-mode gain as "about +0.03–0.04 on Domainome" for the flagging
   rule**, which is the gain of the flag-and-withhold rule measured on the AF-mode study's own predictors
   (+0.038 for the engine model); the model-level difference between the shipped AF model and the default on
   Domainome is +0.006 [−0.001, +0.012]. The manuscript keeps the two separate.
6. **Minor rounding differences between audit 2 and the card** on the homology-filtered Domainome column
   (audit: 0.451 → 0.404; card: 0.450 → 0.403) and on the pre-intersection strict domain count (339 vs 347
   vs 366 depending on which intersection is taken). The manuscript quotes the card's numbers with the domain
   count from `multiassay_summary.csv` (347 domains, 358,755 variants).
7. **One number that no table in the repository supports and that was therefore dropped from the manuscript:**
   the claim that FoldX needs "5–34 CPU-s per mutation" (in `notes/engine_mutate.md`) against "5–10 CPU-s per
   mutation" (README). The manuscript uses the README range, 5 to 10 CPU-seconds per mutation, which is the one
   the speed table is built on.

Numbers that could not be sourced and are therefore absent from the manuscript: any paired comparison against
a deep-learning method (none was rerun here), and any statement about the earlier internal version of this
predictor.
