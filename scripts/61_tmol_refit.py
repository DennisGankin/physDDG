#!/usr/bin/env python
"""Linear refit of tmol per-term dE (scripts/60_tmol_ddg.py) on a Megascale-train subset -> test predictions.

  PYTHONPATH=. .venv/bin/python scripts/61_tmol_refit.py --mode min   (or pack)
Inputs : results/tmol/megascale_train40_<mode>.parquet (fit), results/tmol/{megascale,fireprot}_test_<mode>.parquet
Model  : physddg.fit (Gaussian/Tobit loss on Megascale incl. censored rows, proteins weighted equally),
         features = grouped beta2016 terms (REU, already weighted by beta2016):
           atr, rep_cap (fa_ljrep clipped at a cap, or signed log1p, chosen by 5-fold family-grouped CV on the train subset), sol (fa_lk),
           lkball (lk_ball_iso + lk_ball + lk_bridge + lk_bridge_uncpl), elec, hbond, dun (rot + rotdev + semirot),
           rama, omega, cart (all cart_*), ref (tmol reference energies), dslf
         variants: 'total' (1 weight on raw ddg_reu, calibration only), 'terms' (12 weights), 'terms_ref'
         (+ 20 fitted reference energies, shrunk).
Outputs: results/tmol/refit_<mode>_<variant>_<bench>.csv (KEYS, ddg_pred), results/tmol/refit_<mode>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physddg import eval as E
from physddg import fit as F

KEYS = list(E.KEYS)
GROUPS = {"atr": ["d_fa_ljatr"], "rep": ["d_fa_ljrep"], "sol": ["d_fa_lk"],
          "lkball": ["d_lk_ball_iso", "d_lk_ball", "d_lk_bridge", "d_lk_bridge_uncpl"], "elec": ["d_fa_elec"],
          "hbond": ["d_hbond"], "dun": ["d_dunbrack_rot", "d_dunbrack_rotdev", "d_dunbrack_semirot"],
          "rama": ["d_rama"], "omega": ["d_omega"],
          "cart": ["d_cart_lengths", "d_cart_angles", "d_cart_torsions", "d_cart_impropers", "d_cart_hxltorsions"],
          "ref": ["d_ref"], "dslf": ["d_disulfide"]}
TERMS = ["atr", "rep_cap", "sol", "lkball", "elec", "hbond", "dun", "rama", "omega", "cart", "ref", "dslf"]


def features(t: pd.DataFrame, cap: float) -> pd.DataFrame:
    t = t[t.status == "ok"].copy()
    t["pdb_resnum"] = E._norm_resnum(t["pdb_resnum"])
    for g, cols in GROUPS.items():
        t[g] = t[cols].sum(axis=1)
    r = t["rep"]
    t["rep_cap"] = np.log1p(r.clip(lower=0)) - np.log1p((-r).clip(lower=0)) if cap == "log" else r.clip(upper=float(cap))
    t["total"] = t["ddg_reu"]
    return t


def attach(bench: str, split: str, t: pd.DataFrame) -> pd.DataFrame:
    b = E.load_benchmark(bench)
    b = b[b.split == split]
    return b.merge(t.drop(columns=[c for c in t.columns if c in b.columns and c not in KEYS]), on=KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="min")
    ap.add_argument("--caps", default="1,2,5,10,20,1e9,log", help="fa_ljrep clip values; 'log' = signed log1p")
    a = ap.parse_args()
    out = E.ROOT / "results" / "tmol"
    raw_tr = pd.read_parquet(out / f"megascale_train40_{a.mode}.parquet")
    res = {"mode": a.mode, "n_train_proteins": int(raw_tr.protein_id.nunique())}

    def cfg_for(variant):
        feats = ["total"] if variant == "total" else TERMS
        return F.FitConfig(features=feats, ref=(variant == "terms_ref"), ref_l2=1e-2, within="proteins")

    # cap selection by grouped CV (families) on the train subset, 'terms' variant
    cv = {}
    for cap in [c if c == "log" else float(c) for c in a.caps.split(",")]:
        tr = F.prepare_targets(attach("megascale", "train", features(raw_tr, cap)), "megascale")
        oof = F.cross_validate(tr, cfg_for("terms"), k=5, group_col="family")
        cv[cap] = F.cv_metrics(oof)
        print(f"cap {cap}: CV " + " ".join(f"{k}={v:.3f}" for k, v in cv[cap].items()))
    best = max(cv, key=lambda c: cv[c]["spearman_prot_median"])
    res["cv_cap"] = {str(k): v for k, v in cv.items()}
    res["cap"] = best
    tr = F.prepare_targets(attach("megascale", "train", features(raw_tr, best)), "megascale")
    res["n_train_rows"] = len(tr)
    for variant in ("total", "terms", "terms_ref"):
        cfg = cfg_for(variant)
        oof = F.cross_validate(tr, cfg, k=5, group_col="family")
        m = F.fit(tr, cfg, name=f"tmol_{a.mode}_{variant}")
        res[variant] = {"cv": F.cv_metrics(oof), "intercept": m["intercept"], "weights": m["weights"],
                        "ref": m.get("ref", {})}
        print(variant, " ".join(f"{k}={v:.3f}" for k, v in res[variant]["cv"].items()), {k: round(v, 3) for k, v in m["weights"].items()})
        for bench in ("megascale", "fireprot"):
            p = out / f"{bench}_test_{a.mode}.parquet"
            if not p.exists():
                continue
            te = features(pd.read_parquet(p), best)
            te["ddg_pred"] = F.predict(m, te)
            te[KEYS + ["ddg_pred"]].to_csv(out / f"refit_{a.mode}_{variant}_{bench}.csv", index=False)
    (out / f"refit_{a.mode}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
