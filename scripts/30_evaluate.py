#!/usr/bin/env python
"""Score a predictor against a canonical benchmark.

  scripts/30_evaluate.py --bench megascale --split test --pred results/foo/preds.csv --pred-col ddg_pred \
      [--compare results/bar/preds.csv:col] [--subset strict==true] [--by plddt_bin] [--bins plddt:50,70,90]

Writes results/<exp>/metrics_<bench>.json (exp = parent folder of --pred unless --exp) and a CSV next to it.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from physddg import eval as E


def read_pred(path, col, keys, flip):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if col not in df:
        sys.exit(f"{path}: no column {col!r} (have {list(df.columns)[:15]})")
    if flip:
        df[col] = -df[col]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-col", default="ddg_pred")
    ap.add_argument("--compare", action="append", default=[], help="path:col (repeatable), paired bootstrap vs --pred")
    ap.add_argument("--split", help="keep rows with split==value (e.g. test)")
    ap.add_argument("--subset", help="'col==val[,col2>=x]' filters, e.g. strict==true")
    ap.add_argument("--by", help="also report per-group metrics for this column")
    ap.add_argument("--bins", help="col:e1,e2,.. adds <col>_bin column (use with --by)")
    ap.add_argument("--target", choices=["ddg", "score"], help="default: ddg if present else -score (lower score = destabilising)")
    ap.add_argument("--flip", action="store_true", help="negate --pred values (predictor has opposite sign)")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--exp", help="results/<exp> output folder (default: parent folder of --pred)")
    ap.add_argument("--out")
    ap.add_argument("--root", default=str(E.ROOT), help="project root holding data/processed (testing)")
    a = ap.parse_args()

    bench = E.load_benchmark(a.bench, Path(a.root))
    if a.split:
        bench = bench[bench["split"].astype(str) == a.split]
    bench = E.apply_subset(bench, a.subset).reset_index(drop=True)
    if a.bins:
        c, e = a.bins.split(":")
        bench = E.add_bins(bench, c, [float(x) for x in e.split(",")])
    print(f"bench {a.bench}: {len(bench)} rows, {bench.protein_id.nunique()} proteins (split={a.split}, subset={a.subset})")

    preds = {"main": (a.pred, a.pred_col)} | {f"cmp{i}": tuple(c.rsplit(":", 1)) for i, c in enumerate(a.compare)}
    df, infos = bench, {}
    for name, (path, col) in preds.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = E.merge_predictions(bench, read_pred(path, col, E.KEYS, a.flip and name == "main"), col, out_col=name)
        infos[name] = m.attrs["merge_info"]
        print(f"  {name}: {path}:{col} coverage {infos[name]['coverage']:.1%} ({len(m)} rows)"
              + (f"  OFFSET? {infos[name]['offset']}" if infos[name]["offset"] else ""))
        df = m if name == "main" else df.merge(m[[*E.KEYS, name]].drop_duplicates(list(E.KEYS)), on=list(E.KEYS))
    kw = dict(target=a.target)
    res = {"bench": a.bench, "split": a.split, "subset": a.subset, "predictors": {}, "merge": infos}
    tgt = None
    for name, (path, col) in preds.items():
        met = E.evaluate(df, name, min_n=a.min_n, **kw)
        tgt = met["target"]
        ci = E.bootstrap_ci(df, name, n_boot=a.n_boot, min_n=a.min_n, **kw)
        entry = {"pred": f"{path}:{col}", "metrics": met, "ci95": ci}
        if name != "main":
            entry["paired_vs_main"] = E.paired_bootstrap(df, "main", name, n_boot=a.n_boot, min_n=a.min_n, **kw)  # main - this
        if {"pair_id", "direction"} <= set(df.columns):
            entry["antisymmetry"] = E.antisymmetry(df, name)
        if a.by:
            entry["by_group"] = {str(g): E.evaluate(d, name, min_n=a.min_n, **kw) for g, d in df.groupby(a.by)}
        res["predictors"][name] = entry

    show = ["n", "n_proteins", "spearman", "pearson", "spearman_prot_median", "rmse", "rmse_fit", "auroc_destab", "auroc_stab", "sign_acc"]
    rows = []
    for name, e in res["predictors"].items():
        r = {"pred": name, **{k: e["metrics"][k] for k in show}}
        r |= {f"{k}_ci": f"[{e['ci95'][k][0]:.3f},{e['ci95'][k][1]:.3f}]" for k in ("spearman", "spearman_prot_median") if k in e["ci95"]}
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(f"\ntarget = {tgt} ({'-score; lower score = destabilising' if tgt == 'score' else 'ddg'}), bootstrap over "
          f"{'cluster' if 'cluster' in df else 'protein_id'} (n={a.n_boot})")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for name, e in res["predictors"].items():
        if "paired_vs_main" in e:
            for k in ("spearman", "spearman_prot_median", "auroc_destab"):
                d = e["paired_vs_main"].get(k)
                if d:
                    print(f"  main - {name} {k}: delta {d['delta']:+.3f} CI [{d['lo']:+.3f},{d['hi']:+.3f}] p~{d['p']:.3f}")
        if "antisymmetry" in e:
            print(f"  {name} antisymmetry: {e['antisymmetry']}")
        for g, m in e.get("by_group", {}).items():
            print(f"  {name} [{a.by}={g}] n={m['n']} spearman={m['spearman']:.3f} prot_median={m['spearman_prot_median']:.3f}")

    out = Path(a.out) if a.out else E.ROOT / "results" / (a.exp or Path(a.pred).resolve().parent.name) / f"metrics_{a.bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    tab.to_csv(out.with_suffix(".csv"), index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
