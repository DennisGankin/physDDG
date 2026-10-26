# FoldX 5.1 baseline: exact protocol, what is recorded and what is not (2026-10-26, fitting lead)

Written for critique #2 §3.6. The FoldX runs were produced outside this repository, in `/foldx_native_clean`
(read-only). This note records everything that *is* documented there, states plainly what is **not**, and
explains the one fairness issue that matters (the Megascale structures).

## Binary and provenance
- **FoldX 5.1**, academic licence; binary `/foldx_native_clean/bin/foldx` (86,717,072 bytes,
  sha256 `faed5e54e47744ab…`), banner `*** FoldX 5.1 (c) ***`, with `molecules/` beside it.
- Protocol, verbatim from `/foldx_native_clean/README.md`: *"All values here were computed with FoldX 5.1
  (RepairPDB -> BuildModel) on the benchmark structures."*
- Sign convention `destabilising_positive` (column `foldx_ddg`), i.e. already our D1 convention — no flip.
- Join key `(entity_id, position_1based, wt_aa, mut_aa)`; each run also ships its input manifest
  (`variants_<set>.csv`: entity_id, pos, wt_aa, mut_aa, chain, pdb_resnum, runnable, reason, foldx_mutation).
- Output carries `total_energy` plus the 22 FoldX energy terms, so the runs are re-usable as features.

## Prediction files used in our tables (sha256, first 16 hex)
| file | variants | proteins | sha256 |
|---|---|---|---|
| `foldx_native_megascale.csv` | 28,172 | 28 | `c84c2d2adbfcc19b…` |
| `foldx_native_megascale_train.csv` | 215,731 | 239 | `da50b8358a728a80…` |
| `foldx_native_fireprot.csv` | 341 | 25 | `f3ff58340e0956b4…` |
| `foldx_native_fireprot_trainval.csv` | 2,849 | 67 | `b5190c5b1c1178d4…` |
| `foldx_native_domainome_construct.csv` | 509,388 | 493 | `0f04ffee22482e9b…` |
(There is also `foldx_native_domainome.csv`, the full-length-AF run, which we do not use: the assay measures
isolated constructs, so the construct run is the right comparison.)

## What is NOT recorded
`numberOfRuns`, `--temperature`, `--pH`, `--ionStrength` and `--vdwDesign` were not written down by whoever ran
FoldX, and no FoldX run configuration files were kept. The FoldX 5.1 defaults, from `foldx -h` on this binary, are
**numberOfRuns 1, temperature 298 K, pH 7, ionStrength 0.05 M, vdwDesign 2**; unless the runner passed flags, those
are the conditions. We cannot verify it, so every statement about the FoldX baseline in our reports should say
"FoldX 5.1, RepairPDB → BuildModel, run options not recorded (defaults assumed)".
Practical consequences: with `numberOfRuns 1` FoldX's BuildModel is a single stochastic run, so some of its noise
is run-to-run variance that averaging would remove — this makes our FoldX comparison mildly pessimistic for FoldX
on rank, and we should not claim otherwise. FireProt rows span pH 1.5–11 (`pH` column in `fireprot.csv`), so a
single pH 7 FoldX run is wrong for part of that set — for FoldX and for us equally, since our model has no pH term.

## The Megascale structures (fairness — state this in the paper)
Megascale has **no crystal structures**: all 298 domains are Tsuboyama's own AF2 models
(`struct_source == af2_model_tsuboyama`, `/workspace/data/megascale_pdbs/*.pdb`). So on our primary benchmark:
- FoldX ran **RepairPDB on AF2 models**. That is standard practice and is what any user would do, but it means the
  FoldX baseline is not being run on its home turf (crystal structures) either.
- **Our model is fitted on those same AF2 models.** This is the structure-source confound of critique #2 §3.5:
  "physics helps on AF-based assays and adds ≈0 over B3+ on crystal assays" is equally consistent with an assay
  effect and with a structure-source domain shift, and the comparison on Megascale is between a model trained on
  AF2 inputs and a method parameterised on crystals. The honest framing is that our Megascale margin over FoldX is
  measured on AF2 inputs, and our crystal-set results (FireProt/S669/S461, where we are at parity with FoldX and
  with the 10-parameter B3+) are the ones that test the energy function on crystals.
- Both models see exactly the same files, so the comparison is fair in the "identical rows, identical structures"
  sense; it is the *generalisation* reading that needs the caveat.

## Transform symmetry (critique #2 §3.6, second half)
We report FoldX raw **and** clipped to [−3, 8], a fixed monotone transform with nothing fitted, because FoldX's raw
predictions reach 71 kcal/mol on Megascale and that alone costs it 0.15 Pearson (0.445 → 0.596 on natural). Our own
repulsion term carries a signed-log transform (`slog_rep`) that was **selected on validation**. Both models benefit
from squashing; the clipped-FoldX column is the like-for-like comparison and is the one to quote.

## Reproducing
`scripts/31_baseline_foldx.py` joins the FoldX CSVs onto our canonical tables (coverage printed per split) and
writes `results/baselines/foldx_<bench>.csv`. It does not run FoldX. Re-running FoldX would need the binary, its
`molecules/` directory and the manifests in `/foldx_native_clean/results/variants_*.csv`.
