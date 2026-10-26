#!/usr/bin/env python
"""Residual analysis, step 0: analysis table for Megascale train + val with B5 residuals (no test rows).

- train rows: out-of-fold B5 predictions from grouped 5-fold CV over homology clusters (same folds as fit.cross_validate,
  seed 0, B5 configuration from results/ladder_physics/selection.json); val rows: the saved B5 model (fitted on train).
- residual  r_raw = y - pred (kcal/mol, positive = model under-predicts destabilisation)
            r_cal = y - (a_p + b_p * pred), per-protein linear recalibration (removes per-domain offset and scale,
                    which per-protein Spearman ignores); the per-protein a_p, b_p are stored too.
- site descriptors from physddg/features_extra.py, merged by (structure, chain, pdb_resnum).
Outputs: results/residual_analysis/base.parquet (git-ignored), site descriptor cache data/features/extra_sites_megascale.parquet.

  PYTHONPATH=. .venv/bin/python scripts/90_residual_base.py
"""
import importlib.util
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import features_extra as X  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = ROOT / "results" / "residual_analysis"


def load52():
    spec = importlib.util.spec_from_file_location("lp52", ROOT / "scripts" / "52_ladder_physics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def recal(d, pred="pred", y="y"):
    """Per-protein OLS y ~ a + b pred on point rows; returns r_cal, a_p, b_p."""
    out = pd.DataFrame(index=d.index, columns=["r_cal", "a_p", "b_p"], dtype=float)
    for _, g in d[d[y].notna()].groupby("protein_id"):
        b, a = np.polyfit(g[pred], g[y], 1)
        out.loc[g.index, "a_p"], out.loc[g.index, "b_p"] = a, b
        out.loc[g.index, "r_cal"] = g[y] - (a + b * g[pred])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L52 = load52()
    tr, va, _ = L52.train_tables()
    sel = L52.load_sel()
    o = sel["opts_s2"]
    feats, extra = L52.feature_set("full", o)
    cfg = L52.make_cfg(feats, sel, extra)
    trd, vad = L52.derive(tr, 2, o), L52.derive(va, 2, o)
    oof = FT.cross_validate(trd, cfg, k=5, seed=0)
    trd["pred"], trd["fold"] = oof["oof"].to_numpy(), oof["fold"].to_numpy()
    m = FT.load_model(ROOT / "results" / "ladder_physics" / "models" / "B5.json")
    vad["pred"], vad["fold"] = FT.predict(m, vad), -1
    trd["part"], vad["part"] = "train", "val"
    d = pd.concat([trd, vad], ignore_index=True)
    # B5 term contributions (fold-agnostic: final-model weights) for decomposition plots
    for f, w in m["weights"].items():
        d[f"c_{f}"] = w * d[f]
    d["c_ref"] = d["mut"].map(m["ref"]) - d["wt"].map(m["ref"])
    # extra columns from the canonical table
    ms = pd.read_parquet(ROOT / "data" / "processed" / "megascale.parquet")
    cen = pd.read_csv(ROOT / "data" / "processed" / "megascale_censored.csv", low_memory=False)
    keep = ["protein_id", "chain", "pdb_resnum", "wt", "mut", "dG_wt", "dG_mut", "seq_pos", "flag_mut_floor",
            "flag_mut_ceiling", "flag_wt_high", "flag_wt_low", "flag_lowconf", "flag_protease_disagree", "censor",
            "homolog_in_megascale_train", "family"]
    ex = pd.concat([ms[keep], cen[[c for c in keep if c in cen]]], ignore_index=True)
    ex["pdb_resnum"] = ex["pdb_resnum"].astype(str)
    d["pdb_resnum"] = d["pdb_resnum"].astype(str)
    ex = ex.drop_duplicates(["protein_id", "chain", "pdb_resnum", "wt", "mut"])
    d = d.merge(ex, on=["protein_id", "chain", "pdb_resnum", "wt", "mut"], how="left", suffixes=("", "_x"))
    # dG_wt for censored rows (constant per protein)
    dgw = ms.groupby("protein_id").dG_wt.first()
    d["dG_wt"] = d["protein_id"].map(dgw)
    # simple descriptors (ss3, phi, psi, ...)
    simp = pd.read_parquet(ROOT / "data" / "features" / "simple_megascale.parquet")
    simp["pdb_resnum"] = simp["pdb_resnum"].astype(str)
    sc = ["ss3", "ss8", "phi", "psi", "cn", "ncb10", "occn", "nh_hb", "bfac", "dhyd", "dvol", "de_env_bb", "to_pro",
          "from_pro", "to_gly"]
    d = d.merge(simp[["protein_id", "chain", "pdb_resnum", "wt", "mut", "structure"] + sc].drop_duplicates(
        ["protein_id", "chain", "pdb_resnum", "wt", "mut", "structure"]),
        on=["protein_id", "chain", "pdb_resnum", "wt", "mut", "structure"], how="left", suffixes=("", "_s"))
    # site descriptors
    st = X.site_table(d.structure.unique(), "megascale")
    st = st.drop(columns=["structure_abs"]).add_prefix("x_").rename(
        columns={"x_structure": "structure", "x_chain": "chain", "x_pdb_resnum": "pdb_resnum"})
    n0 = len(d)
    d = d.merge(st, on=["structure", "chain", "pdb_resnum"], how="left")
    assert len(d) == n0
    print("site descriptor coverage", d["x_ncb10"].notna().mean(), "aa match", (d.x_aa_struct == d.wt).mean())
    d["site"] = d.protein_id.astype(str) + "|" + d.chain.astype(str) + "|" + d.pdb_resnum.astype(str)
    d["r_raw"] = d["y"] - d["pred"]
    rc = recal(d)
    d[["r_cal", "a_p", "b_p"]] = rc.to_numpy(float)
    d.to_parquet(OUT / "base.parquet", index=False)
    from physddg import eval as E
    for part in ("train", "val"):
        g = d[(d.part == part) & d.y.notna()]
        for tag, gg in (("all", g), ("natural", g[g.domain_origin == "natural"])):
            r = E.evaluate(gg.assign(ddg=gg.y), "pred")
            print(part, tag, "n", r["n"], "prot", r["n_prot_eval"], "rho_prot", round(r["spearman_prot_median"], 4),
                  "rho", round(r["spearman"], 4), "r", round(r["pearson"], 4))


if __name__ == "__main__":
    main()
