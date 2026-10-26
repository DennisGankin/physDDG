# Evaluation harness (`physddg/eval.py`, `scripts/30-32_*.py`)

Run: `PYTHONPATH=. .venv/bin/python scripts/30_evaluate.py --bench megascale --split test --pred results/<exp>/preds.csv --pred-col ddg_pred`
Tests: `.venv/bin/python -m pytest tests`.

## Sign / target
- Predictions must use ddg convention (positive = destabilising); `--flip` negates the main prediction.
- Benchmarks with `ddg` are scored against `ddg`. If `ddg` is all NaN (or `--target score`), against **-score**
  (proxy such as Domainome fitness: lower = destabilising). Only Spearman/Pearson (pooled, per protein) are reported
  for proxies; RMSE/MAE/AUROC/sign are NaN.

## API
- `load_benchmark(name, root)`: `data/processed/<name>.parquet|csv`; `pdb_resnum` normalised to string ('4.0' -> '4').
- `merge_predictions(bench, pred, pred_col="ddg_pred", keys=KEYS, out_col="pred", min_coverage=0.99)`: inner join on
  `(protein_id, chain, pdb_resnum, wt, mut)`; duplicates in `pred` dropped with a warning. Coverage = fraction of
  benchmark rows with a non-NaN prediction, stored in `result.attrs["merge_info"]`. Below 99% it prints a `!!!!!!!!`
  line to stderr and warns. It also tries integer shifts -3..3 of the prediction's `pdb_resnum`; if one gives clearly
  higher coverage, `merge_info["offset"] = {shift, coverage_shifted}` and the message says "LIKELY POSITION OFFSET"
  (shift = add this to the prediction resnum). Never applied automatically. Per-protein offsets are not detected.
- `apply_subset(df, "split==test,strict==true,plddt>=70")`: AND of terms (`== != >= <= > <`; true/false, numbers, strings).
  `add_bins(df, "plddt", [50,70,90])` adds `plddt_bin` with labels `[a,b)`.
- `evaluate(df, pred_col, target=None, min_n=10)` -> dict of the metrics below.
- `bootstrap_ci(df, pred_col, n_boot=1000)` -> `{metric: (lo, hi)}`, 95% percentile; resamples `cluster` (rows with
  missing/blank cluster count as their own cluster) else `protein_id`; per-protein Spearman is computed once per protein.
- `paired_bootstrap(df, a, b)` -> `{metric: {delta, lo, hi, p}}`, delta = metric(a) - metric(b) on rows where both exist,
  the same resampled groups for both. `p` = 2*min(P(delta<=0), P(delta>=0)) over replicates: a bootstrap fraction, not a
  calibrated test. For rmse/mae, lower is better (negative delta favours a).
- `antisymmetry(df, pred_col, pair_col="pair_id", dir_col="direction")`: needs one row per direction per pair
  (`direction` in direct/reverse, dir/rev). Reports `r_pearson`, `r_spearman` of pred(direct) vs -pred(reverse),
  `bias` = mean(pred_dir + pred_rev) (0 = antisymmetric; positive = both directions predicted destabilising), the same
  quantities for the experimental ddg (`exp_*`) and Pearson of each direction against experiment.
  The CLI calls it automatically if the table has `pair_id` and `direction` columns (not yet in the canonical schema).

## Metrics (`evaluate`)
| key | definition |
|---|---|
| n, n_proteins, n_prot_eval | rows, proteins, proteins with >= min_n variants and a defined Spearman |
| spearman, pearson | pooled over all rows (pred vs ddg) |
| spearman_prot_median / _mean | per-protein Spearman, proteins with >= min_n (10) variants |
| rmse, mae | raw, no calibration |
| rmse_fit, mae_fit | after in-sample linear fit ddg ~ a*pred + b, pooled (optimistic; for calibration-free comparison only) |
| auroc_destab | rank AUROC of pred for ddg > 1 (needs >= 10 positives) |
| auroc_stab | rank AUROC of -pred for ddg < -0.5 (needs >= 10 positives) |
| sign_acc | fraction with sign(pred) == sign(ddg), rows with ddg != 0, raw predictions |

## CLI (`30_evaluate.py`)
`--bench --split --subset --pred --pred-col [--compare path:col]... [--by col] [--bins col:e1,e2] [--target] [--flip]
[--exp] [--n-boot]`. Prints a table, paired deltas for each `--compare` (main - other) and writes
`results/<exp>/metrics_<bench>.json` (+ `.csv`); `<exp>` defaults to the parent folder of `--pred`.

## Baselines
- `31_baseline_foldx.py`: native FoldX 5.1 (`/foldx_native_clean/results`; already destabilising-positive) ->
  `results/baselines/foldx_<bench>.csv` (KEYS, `ddg_pred`) for megascale (test+train), fireprot (test+trainval),
  domainome (construct-level run). FoldX has no Megascale val split (coverage 0 there), nothing for s669/ssym.
  Gitignored, regenerable.
- `32_baseline_trivial.py`: kd, vol, absvol, blosum, burial (1-RSA, freesasa), burial_kd, burial_absvol,
  burial_blosum, nc_kd (neighbour count). Writes `results/baselines/trivial_<bench>.csv` and a metrics CSV.

## Numbers so far (Megascale test, 28 proteins, 28,172 variants)
FoldX 5.1: Spearman 0.569 [0.53, 0.61], per-protein median 0.633, Pearson 0.447, AUROC(ddg>1) 0.829, RMSE 3.17 (0.94 after linear fit).
Trivial baselines (Spearman pooled): burial 0.468, burial_absvol 0.343, burial_blosum 0.337, nc_kd 0.310, kd 0.304, blosum 0.159, vol 0.135.
