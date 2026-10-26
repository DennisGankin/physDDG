#!/usr/bin/env python
"""Native FoldX 5.1 baseline -> results/baselines/foldx_<bench>.csv (columns: schema KEYS, seq_pos, ddg_pred).

Source: /foldx_native_clean/results (join key there: entity_id, position_1based, wt_aa, mut_aa; foldx_ddg is
destabilising-positive = our convention, no flip). If data/processed/<bench>.* exists the FoldX rows are joined
onto it (by KEYS, falling back to (protein_id, seq_pos, wt, mut)) and the coverage is printed; otherwise the
raw FoldX keys (protein_id=entity_id, chain, pdb_resnum, wt, mut) are written.
  scripts/31_baseline_foldx.py [--bench megascale fireprot domainome]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from physddg import eval as E

FX = Path("/foldx_native_clean/results")
FILES = {"megascale": ["foldx_native_megascale.csv", "foldx_native_megascale_train.csv"],
         "fireprot": ["foldx_native_fireprot.csv", "foldx_native_fireprot_trainval.csv"],
         "domainome": ["foldx_native_domainome_construct.csv"]}


def load_foldx(bench: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(FX / f, dtype={"pdb_resnum": str}) for f in FILES[bench]], ignore_index=True)
    out = pd.DataFrame({"protein_id": df.entity_id, "chain": df.chain, "pdb_resnum": E._norm_resnum(df.pdb_resnum),
                        "wt": df.wt_aa, "mut": df.mut_aa, "seq_pos": df.position_1based, "ddg_pred": df.foldx_ddg})
    return out.drop_duplicates(list(E.KEYS))


def attach(bench: str, fx: pd.DataFrame) -> pd.DataFrame:
    try:
        b = E.load_benchmark(bench)
    except FileNotFoundError:
        print(f"  data/processed/{bench} not found: writing raw FoldX keys")
        return fx
    keys = list(E.KEYS)
    for name, on in (("KEYS", keys), ("seq_pos", ["protein_id", "seq_pos", "wt", "mut"])):
        if not set(on) <= set(b):
            continue
        m = E.merge_predictions(b, fx, "ddg_pred", keys=on, out_col="ddg_pred", quiet=True)  # FoldX lacks some splits: per-split below
        hit = E._norm(b, keys).merge(m[keys].assign(hit=True), on=keys, how="left")["hit"].notna()
        print(f"  join on {name}: coverage {hit.mean():.1%}" + ("; per split " + str(hit.groupby(b["split"].to_numpy()).mean().round(3).to_dict()) if "split" in b else ""))
        if hit.mean() > 0.5:  # otherwise try the next key set (position offset / naming)
            break
    return m[[*keys, "ddg_pred"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=list(FILES))
    a = ap.parse_args()
    out_dir = E.ROOT / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in a.bench:
        print(bench)
        fx = load_foldx(bench)
        out = attach(bench, fx)
        out.to_csv(out_dir / f"foldx_{bench}.csv", index=False)
        print(f"  wrote {len(out)} rows -> results/baselines/foldx_{bench}.csv")


if __name__ == "__main__":
    main()
