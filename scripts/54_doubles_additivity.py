#!/usr/bin/env python
"""Measured non-additivity of the 138k Megascale double mutants vs CB-CB distance (critique #2, I6a).

eps = ddg(AB) - ddg(A) - ddg(B), using the per-row single values that data-ht stored with each double.
This bounds the gain available to any pairwise term: it is a property of the DATA, not of our model.
Scoring the engine's own coupling would need a double-mutant scan from E1's pipeline (see notes).
  PYTHONPATH=. .venv/bin/python scripts/54_doubles_additivity.py
Outputs results/ladder_physics/doubles_additivity.csv (+ by-split and by-burial breakdowns).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import structure as S  # noqa: E402

OUT = ROOT / "results" / "ladder_physics"
EDGES = [0, 5, 6.5, 8, 10, 12, 15, np.inf]


def cb_distance(df: pd.DataFrame) -> np.ndarray:
    """CB-CB distance (ideal CB for Gly) between the two mutated sites of each double."""
    d = np.full(len(df), np.nan)
    for st, g in df.groupby("structure"):
        p = S.load_protein(str(ROOT / st) if not Path(st).is_absolute() else st, keep_het=False)
        cb = S.cb_coords(p)
        idx = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
        for row in g.itertuples():
            i, j = idx.get((row.chain, str(row.pos1))), idx.get((row.chain, str(row.pos2)))
            if i is not None and j is not None:
                d[row.Index] = float(np.linalg.norm(cb[i] - cb[j]))
    return d


def main():
    p = ROOT / "data" / "processed" / "megascale_doubles.csv.gz"
    df = pd.read_csv(p, low_memory=False)
    print("columns:", list(df.columns)[:25])
    need = {"ddg", "ddg1", "ddg2"}
    assert need <= set(df.columns), sorted(need - set(df.columns))
    d = df[df[["ddg", "ddg1", "ddg2"]].notna().all(1)].reset_index(drop=True)
    d["eps"] = d["ddg"] - d["ddg1"] - d["ddg2"]
    d = d[d.in_dataset3.fillna(False).astype(bool)].reset_index(drop=True) if "in_dataset3" in d else d
    # positions of the two substitutions
    d["pos1"] = E._norm_resnum(d["pdb_resnum1"]) 
    d["pos2"] = E._norm_resnum(d["pdb_resnum2"])
    d["dist"] = cb_distance(d)
    print(f"{len(d)} doubles with both singles, CB distance for {d.dist.notna().mean():.1%}")
    d["bin"] = pd.cut(d.dist, EDGES, right=False)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        rows.append({"cb_bin": str(b), "n": len(g), "mean_eps": g.eps.mean(), "sd_eps": g.eps.std(),
                     "frac_abs_gt_0.5": float((g.eps.abs() > 0.5).mean()),
                     "frac_abs_gt_1.0": float((g.eps.abs() > 1.0).mean()),
                     "mean_abs_eps": g.eps.abs().mean(),
                     "r_sum_vs_obs": float(np.corrcoef(g.ddg1 + g.ddg2, g.ddg)[0, 1]) if len(g) > 2 else np.nan,
                     "sd_sum_of_singles": float((g.ddg1 + g.ddg2).std())})
    t = pd.DataFrame(rows)
    overall = {"cb_bin": "all", "n": len(d), "mean_eps": d.eps.mean(), "sd_eps": d.eps.std(),
               "frac_abs_gt_0.5": float((d.eps.abs() > 0.5).mean()), "frac_abs_gt_1.0": float((d.eps.abs() > 1.0).mean()),
               "mean_abs_eps": d.eps.abs().mean(), "r_sum_vs_obs": float(np.corrcoef(d.ddg1 + d.ddg2, d.ddg)[0, 1]),
               "sd_sum_of_singles": float((d.ddg1 + d.ddg2).std())}
    t = pd.concat([pd.DataFrame([overall]), t], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "doubles_additivity.csv", index=False)
    print(t.round(3).to_string(index=False))
    # how much of the variance of the double is NOT explained by the sum of singles, per bin
    var = d.groupby("bin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "var_ddg": g.ddg.var(), "var_eps": g.eps.var(),
                             "eps_share_of_var": g.eps.var() / g.ddg.var()}), include_groups=False)
    print(var.round(3).to_string())
    var.to_csv(OUT / "doubles_variance.csv")
    # Floor control: the mean eps is strongly negative at EVERY distance, including > 15 A, which is the
    # signature of assay saturation (the double cannot be measured below dG_wt - ... ), not of coupling.
    # Repeat on rows that cannot be floor-limited: the double is well inside the dynamic range.
    safe = d[(~d.flag_mut_floor.fillna(False).astype(bool)) & (d.dG_mut > d.dG_wt * 0 - 0.5)
             & ((d.ddg1 + d.ddg2) < 1.5) & (d.dG_mut > 0.5)]
    rows = []
    for b, g in safe.groupby("bin", observed=True):
        rows.append({"cb_bin": str(b), "n": len(g), "mean_eps": g.eps.mean(), "sd_eps": g.eps.std(),
                     "frac_abs_gt_0.5": float((g.eps.abs() > 0.5).mean()), "mean_abs_eps": g.eps.abs().mean()})
    sf = pd.DataFrame([{"cb_bin": "all (floor-safe)", "n": len(safe), "mean_eps": safe.eps.mean(),
                        "sd_eps": safe.eps.std(), "frac_abs_gt_0.5": float((safe.eps.abs() > 0.5).mean()),
                        "mean_abs_eps": safe.eps.abs().mean()}] + rows)
    sf.to_csv(OUT / "doubles_additivity_floorsafe.csv", index=False)
    print("\nFloor-safe subset (sum of singles < 1.5 kcal/mol, dG_mut > 0.5, not flag_mut_floor):")
    print(sf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
