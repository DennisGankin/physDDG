# Discrepancies, open questions, and judgement calls

## Where the record and the code or results disagree
1. **`RESEARCH_TREE.md` line 119** (and decision D22, item 4.14b) still describe the joint Megascale+FireProt models as
   the predictor defaults. Commit `47a1b89` reversed this: the shipped `final_*` models are the Megascale-only family
   (B6g / B6gs1 / B6gaf); the joint fits are `alt_multiassay_*` (`--weights alt`). Only tree item 4.17 and the model
   card record the reversal. I followed the code and model card. The same stale line is in the release copy.
2. **`physddg/predict.py` docstring** still says the default is B4/B5; the live code path loads `final_stage2.json`.
3. **Headline confidence interval** appears as [+0.006, +0.120] (model card, REPORT) and [+0.0081, +0.0955]
   (audit 2, joint model). A percentile bootstrap I ran over the 21 per-protein values gave [−0.003, +0.120]; the
   lower bound depends on the bootstrap scheme. I quote the
   project's interval and flag that it sits close to zero.
4. **Parameter count:** 41 / 43 / 83 (counted from JSONs). Earlier text said 40; one weight is pinned at 0, so 40 are
   effective for the default model.
5. **S669-strict for the default model** is 0.336 (488 rows) or 0.316 (470 rows shared with the joint family).
6. **pLDDT attenuation weight** is quoted four ways in the notes (−0.90, −0.853, −0.966, −0.97).
7. **Evolution mode** is documented in `notes/evo_extension.md` but has no code path in `predict()`.
8. **Number of subagents:** the brief said 44 logs; there are 29 distinct agents (58 files).
9. **Model attribution:** the PI session switched from Opus to Sonnet after a safety stop; commit trailers differ.

## Things I could not verify or did not do
- **Literature citations** were taken from the project's manuscript; I did not re-check them.
- **Deep-learning numbers** (ThermoMPNN 0.77, SPURS) are published values, not rerun.
- **The interactive page was checked with a DOM emulator (no errors, all controls respond) but not viewed in a real
  browser**, because no browser could start in this sandbox. Figures were rendered and inspected as images.
- **Worked example on CPU:** the stored GPU predictions and my CPU rerun agree to display precision for the four
  mutations; I did not do a full numeric diff of all 1,213 variants.
- The `1PSE` example uses the AlphaFold2 model that Megascale ships, not a crystal structure.
- I did not rerun the benchmark; ρ values come from stored prediction files (per-protein values recomputed from
  `preds_megascale.parquet`; the 21 medians match the model card exactly).

## Where I was unsure what mattered
- **Which 8 turning points.** I merged "FoldX re-weighting fails" into TP1 and "the three negative engineering
  results" into an appendix. TP4 (forward/reverse bias) is a failure; I kept it because it explains why the wild type
  is rebuilt, but a reader could argue the cost to the final method was small.
- **How much of the audit story to put in the method.** It decides which claims survive, so it is in Results and TP8,
  not in the steps.
- **Credit for ideas.** The manuscript lists them; I kept that list, but some attributions (e.g. Rossi 2025 for the
  "mass balance" reference energies) are the agents' own and I did not check the papers.
- **Selection optimism.** The headline test differences are probably optimistic by an unknown amount (at least 12
  test looks). Validation and cross-validation numbers are cleaner but less comparable to other papers.
