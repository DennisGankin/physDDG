#!/usr/bin/env python
"""D23(A)+(B): decide (a) how the disulfide-broken term enters the joint fit and (b) which family ships,
using ONLY training-set evidence: Megascale-val, grouped CV on each model's own training set, and G6 transfer.
No test set is read anywhere in this script.

  PYTHONPATH=. .venv/bin/python scripts/56_ss_decision.py
Writes results/ladder_physics/ss_decision.csv and family_decision.csv.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

_spec = importlib.util.spec_from_file_location("l52", ROOT / "scripts" / "52_ladder_physics.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
OUT = ROOT / "results" / "ladder_physics"
SS = "ss_broken"
# Literature prior for breaking a structural disulfide: 2-5 kcal/mol (notes/critique_design_v0.md R12 and refs
# therein); the midpoint 3.0 and a conservative 2.0 are both tried as FIXED (not fitted) values.
PRIORS = {"fixed2.0": 2.0, "fixed3.0": 3.0}


def cv_by_dataset(df, cfg, k=5):
    """Grouped-CV OOF predictions, then per-dataset metrics (so the FireProt folds are visible separately)."""
    oof = FT.cross_validate(df, cfg, k=k)
    out = {}
    for ds, g in oof.groupby("dataset"):
        p = g[g.y.notna() & g.oof.notna()].assign(ddg=lambda d: d.y)
        if len(p) < 20:
            continue
        r = E.evaluate(p, "oof")
        out[f"cv_{ds}_prot"] = r["spearman_prot_median"]
        out[f"cv_{ds}_pearson"] = r["pearson"]
    nat = oof[(oof.domain_origin == "natural") & oof.y.notna() & oof.oof.notna()]
    out["cv_megascale_natural_prot"] = E.evaluate(nat.assign(ddg=nat.y), "oof")["spearman_prot_median"]
    return out


def val_scores(m, vad, atten=None):
    v = vad.assign(ddg=vad.y)
    v = v.assign(pred=M.predict_attenuated(m, v) if atten else FT.predict(m, v))
    vn = v[(v.domain_origin == "natural") & v.ddg.notna()]
    return {"val_natural_prot": E.evaluate(vn, "pred")["spearman_prot_median"],
            "val_all_prot": E.evaluate(v[v.ddg.notna()], "pred")["spearman_prot_median"]}


def main():
    tr, va, fp = M.train_tables(M.EVAL_CAP)
    sel = M.load_sel()
    rows = []
    for stage in (2, 1):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        trd, fpd, vad = M.derive(tr, stage, o), M.derive(fp, stage, o), M.derive(va, stage, o)
        joint = pd.concat([trd, fpd], ignore_index=True)
        feats, extra = M.feature_set("b6g", o)
        for label, variant in (("free (current)", {}), ("dropped", "drop"),
                               *[(f"{k} (literature prior, not fitted)", {SS: v}) for k, v in PRIORS.items()]):
            f = [x for x in feats if x != SS] if variant == "drop" else list(feats)
            cfg = M.make_cfg(f, sel, extra)
            cfg.ref_nuisance = ("megascale",)
            if isinstance(variant, dict) and variant:
                cfg.fixed = dict(variant)
            m = FT.fit(joint, cfg)
            r = {"stage": stage, "ss_term": label, "n_feat": len(f),
                 "w_ss": m["weights"].get(SS, np.nan), **val_scores(m, vad), **cv_by_dataset(joint, cfg)}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ss_decision.csv", index=False)
    print("\n== (A) disulfide term, joint fit — val/CV only\n", t.round(4).to_string(index=False))

    # ---- (B) family decision: Megascale-only vs joint, on val + CV on their own training sets + G6
    fam = []
    for stage in (2, 1):
        o = dict(sel[f"opts_s{stage}"], cxxc=False)
        trd, fpd, vad = M.derive(tr, stage, o), M.derive(fp, stage, o), M.derive(va, stage, o)
        feats, extra = M.feature_set("b6g", o)
        fnoss = [x for x in feats if x != SS]
        for label, data, fl, nuis in (("Megascale-only (B6g)", trd, feats, ()),
                                      ("joint, ss free (B6gj)", pd.concat([trd, fpd], ignore_index=True), feats, ("megascale",)),
                                      ("joint, ss dropped", pd.concat([trd, fpd], ignore_index=True), fnoss, ("megascale",))):
            cfg = M.make_cfg(fl, sel, extra)
            cfg.ref_nuisance = nuis
            m = FT.fit(data, cfg)
            r = {"stage": stage, "family": label, **val_scores(m, vad), **cv_by_dataset(data, cfg)}
            # G6: refit each source alone, count sign flips and >3x (scaled by the 1.7 assay factor)
            a = FT.fit(M.derive(tr, stage, o), M.make_cfg(fl, sel, extra))
            b = FT.fit(M.derive(fp, stage, o), M.make_cfg(fl, sel, extra))
            flips = sum(1 for x in fl if a["weights"][x] * b["weights"][x] < 0)
            und = sum(1 for x in fl if (a["weights"][x] == 0) != (b["weights"][x] == 0))
            big = sum(1 for x in fl if a["weights"][x] != 0 and b["weights"][x] != 0
                      and not (1 / 3 <= abs(b["weights"][x] / a["weights"][x] / 1.7) <= 3))
            r.update(g6_sign_flips=flips, g6_undetermined_in_one=und, g6_over_3x_scaled=big)
            fam.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    f = pd.DataFrame(fam)
    f.to_csv(OUT / "family_decision.csv", index=False)
    print("\n== (B) family decision — val/CV/G6 only\n", f.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
