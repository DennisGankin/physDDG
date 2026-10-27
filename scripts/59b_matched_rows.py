#!/usr/bin/env python
"""D23(D): matched-row table — every model scored on the SAME rows, for the columns where per-model
strictness or competitor coverage makes the main table's rows differ (audit 2, I1).

Rows are the intersection over all compared models of: non-missing prediction AND the strictest rule any of
them requires (i.e. both-strict wherever a joint/ProTherm-trained model is in the comparison). This is the
apples-to-apples view; the main table keeps per-model strictness, which is the correct-per-model view.

  PYTHONPATH=. .venv/bin/python scripts/59b_matched_rows.py
Writes results/ladder_physics/matched_rows.csv.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import eval as E  # noqa: E402

_spec = importlib.util.spec_from_file_location("m55", ROOT / "scripts" / "55_multiassay_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
OUT = ROOT / "results" / "ladder_physics"
MODELS = ["B3+", "B5", "final_stage1", "final_stage2", "final_af",
          "alt_multiassay_stage1", "alt_multiassay_stage2", "alt_multiassay_af",
          "FoldX", "FoldX-clip", "tmol-refit", "KORPM", "DDGun3D"]
# (bench, metric, label, models to exclude from the intersection because their coverage would collapse it)
VIEWS = [("s669", "pearson", "S669 both-strict", ()), ("s461", "pearson", "S461 both-strict", ()),
         ("domainome", "spearman_prot_median", "Domainome both-strict", ("DDGun3D",)),
         ("fireprot", "pearson", "FireProt matched-to-FoldX (NOT the headline: drops the 2 CXXC rows)", ()),
         ("megascale", "spearman_prot_median", "Megascale natural", ())]


def main():
    rows = []
    for bench, metric, label, drop in VIEWS:
        b = R.load_bench(bench)
        target = "ddg" if b.ddg.notna().any() else "score"
        have = [m for m in MODELS if m in b and b[m].notna().any() and m not in drop]
        # strictest rule needed by anyone in the comparison; FireProt/Megascale keep their own definition
        if bench in ("s669", "s461", "domainome"):
            b = E.apply_subset(b, R.BOTH)
        elif bench == "megascale":
            b = E.apply_subset(b, "domain_origin==natural")
        mask = b[have].notna().all(axis=1)
        d = b[mask]
        for m in have:
            r = E.evaluate(d, m, target=target)
            rows.append({"view": label, "model": m, "n": r["n"], "n_prot": r["n_proteins"],
                         "value": r[metric], "median_prot": r["spearman_prot_median"],
                         "mean_prot": r["spearman_prot_mean"], "metric": metric})
        print(f"{label}: {len(d)} rows / {d.protein_id.nunique()} proteins, all {len(have)} models scored "
              f"on the same rows", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "matched_rows.csv", index=False)
    print(t.pivot_table(index="model", columns="view", values="value", sort=False).reindex(
        [m for m in MODELS if m in set(t.model)]).round(3).to_string())
    print("\nn per view:", {v: int(t[t.view == v].n.iloc[0]) for v in t.view.unique()})


if __name__ == "__main__":
    main()
