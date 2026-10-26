#!/usr/bin/env python
"""Residual analysis, step 2: diagnostic gradient-boosting probes (NOT a deliverable) on B5 residuals.

(a) site-level probe: target = per-site mean of r_cal (sites with >= 10 point variants), sample weight = n variants;
    features = site descriptor families (physddg/features_extra.py) + WT identity + domain-level + the model's own
    site-mean prediction. Grouped 5-fold CV over homology clusters on train sites (OOF R^2), plus fit-on-train ->
    val-site R^2. Family importance = drop-family loss and single-family R^2.
(b) within-site probe: target = r_cal - site mean; features = site families + wt/mut amino-acid property vectors +
    B5 per-term feature values; same CV. Family importance by drop-family.
Outputs results/residual_analysis/probe_site.csv, probe_within.csv, probe_site_oof.parquet.
  PYTHONPATH=. .venv/bin/python scripts/92_site_probe.py
"""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "3")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from physddg import features_extra as X  # noqa: E402
from physddg import features_simple as FS  # noqa: E402
from physddg.fit import group_folds  # noqa: E402

OUT = ROOT / "results" / "residual_analysis"
AA = "ACDEFGHIKLMNPQRSTVWY"

FAM = {
    "burial": ["x_ncb6", "x_ncb8", "x_ncb10", "x_ncb12", "x_ncb14", "x_ncen8", "x_ncen10", "x_ncen12", "x_hse_up",
               "x_hse_down", "x_cn", "x_rsa", "x_sasa_sc"],
    "packing": ["x_natom6", "x_natom8", "x_void_frac", "x_scsc_contacts", "x_scsc_res"],
    "hbond": ["x_hb_sc_any", "x_hb_sc_bb", "x_hb_sc_sc", "x_hb_bb_nh"],
    "ss_position": ["x_ncap", "x_n1", "x_n2", "x_n3", "x_c1", "x_c2", "x_c3", "x_ccap", "x_helix_int", "x_helix_len",
                    "x_helix_pos", "x_strand_edge", "x_strand_centre", "x_strand_len", "x_loop_len"] + [f"ss8_{s}" for s in "HGIEBTS-"],
    "backbone": ["x_phi", "x_psi", "x_cis_prev"] + [f"basin_{b}" for b in ("alpha", "beta", "ppII", "alphaL", "other")],
    "termini": ["x_dist_nterm", "x_dist_cterm", "x_dist_term", "x_rel_pos"],
    "plddt": ["x_plddt", "x_plddt_win", "x_plddt_min_win", "x_plddt_z"],
    "charge": ["x_q_site", "x_q_env8", "x_q_env12", "x_n_pos6", "x_n_neg6", "x_sb_partners", "x_same_q6"],
    "aromatic": ["x_n_aro6", "x_n_aro8", "x_aro_stack", "x_catpi"],
    "wt_identity": [f"wt_{a}" for a in AA],
    "domain": ["x_L", "x_frac_helix", "x_frac_strand", "x_mean_plddt", "dG_wt", "designed"],
    "model_pred": ["site_pred"],
}
PROPS = {"hyd": FS.HYD, "vol": FS.VOL, "helix": X.HELIX_PS, "beta": X.BETA_MK,
         "chg": {a: X.CHARGE.get(a, 0) for a in AA}, "aro": {a: float(a in "FWY") for a in AA},
         "gly": {a: float(a == "G") for a in AA}, "pro": {a: float(a == "P") for a in AA},
         "polar": {a: float(a in "STNQHDEKRYW") for a in AA}, "bbranch": {a: float(a in "VIT") for a in AA}}


def add_onehots(d):
    for s in "HGIEBTS-":
        d[f"ss8_{s}"] = (d.ss8 == s).astype(float)
    for b in ("alpha", "beta", "ppII", "alphaL", "other"):
        d[f"basin_{b}"] = (d.x_basin == b).astype(float)
    for a in AA:
        d[f"wt_{a}"] = (d.wt == a).astype(float)
    d["designed"] = (d.domain_origin == "designed").astype(float)
    return d


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def r2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def cv_r2(df, feats, target, weight, groups, k=5):
    folds = group_folds(groups, k, 0)
    oof = np.zeros(len(df))
    Xm, y, w = df[feats].to_numpy(float), df[target].to_numpy(float), df[weight].to_numpy(float)
    for j in range(k):
        tr, te = folds != j, folds == j
        m = gbm().fit(Xm[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict(Xm[te])
    return r2(y, oof, w), oof


def val_r2(tr, va, feats, target, weight):
    m = gbm().fit(tr[feats].to_numpy(float), tr[target].to_numpy(float), sample_weight=tr[weight].to_numpy(float))
    return r2(va[target].to_numpy(float), m.predict(va[feats].to_numpy(float)), va[weight].to_numpy(float))


def family_scan(tr, va, fams, target, weight, tag):
    allf = [f for v in fams.values() for f in v]
    jobs = [("ALL", allf)] + [(f"-{k}", [f for f in allf if f not in v]) for k, v in fams.items()] + \
           [(f"only {k}", v) for k, v in fams.items()]

    def run(lab, feats):
        r_cv, _ = cv_r2(tr, feats, target, weight, tr.cluster)
        return {"probe": tag, "set": lab, "n_feat": len(feats), "r2_cv_train": r_cv, "r2_val": val_r2(tr, va, feats, target, weight)}
    res = pd.DataFrame(Parallel(n_jobs=16)(delayed(run)(lab, f) for lab, f in jobs))
    base = res[res.set == "ALL"].iloc[0]
    res["d_cv_vs_all"] = res.r2_cv_train - base.r2_cv_train
    res["d_val_vs_all"] = res.r2_val - base.r2_val
    return res


def main():
    d = pd.read_parquet(OUT / "base.parquet")
    d = add_onehots(d[d.y.notna()].copy())
    # ---------------------------------------------------------------- (a) site level
    g = d.groupby("site")
    first = g.first()
    s = pd.DataFrame({"r_site": g.r_cal.mean(), "n": g.size(), "site_pred": g.pred.mean(), "site_y": g.y.mean()})
    cols = sorted({f for v in FAM.values() for f in v if f != "site_pred"})
    s = s.join(first[cols + ["part", "cluster", "protein_id", "wt", "domain_origin"]])
    s = s[s.n >= 10].reset_index()
    tr, va = s[s.part == "train"].reset_index(drop=True), s[s.part == "val"].reset_index(drop=True)
    print("sites train", len(tr), "val", len(va), "var(r_site) train", np.average((tr.r_site - tr.r_site.mean()) ** 2, weights=tr.n))
    res = family_scan(tr, va, FAM, "r_site", "n", "site")
    res.to_csv(OUT / "probe_site.csv", index=False)
    print(res.round(4).to_string(index=False))
    _, oof = cv_r2(tr, [f for v in FAM.values() for f in v], "r_site", "n", tr.cluster)
    tr.assign(oof=oof).to_parquet(OUT / "probe_site_oof.parquet", index=False)
    # marginal views of the strongest families (binned mean residual) for the notes
    for c in ("x_ncb14", "x_hse_up", "x_void_frac", "x_plddt", "x_dist_term", "x_hb_sc_any", "x_natom8", "x_ncb6", "site_pred"):
        q = pd.qcut(s[c].rank(method="first"), 8, labels=False)
        t = s.groupby(q).apply(lambda x: pd.Series({"lo": x[c].min(), "hi": x[c].max(), "mean_r_site": np.average(x.r_site, weights=x.n)}))
        print(c, " | ".join(f"[{a:.2f},{b:.2f}] {m:+.3f}" for a, b, m in t.to_numpy()))
    # ---------------------------------------------------------------- (b) within site
    d["r_within"] = d.r_cal - d.groupby("site").r_cal.transform("mean")
    for nm, tab in PROPS.items():
        d[f"wt_{nm}"], d[f"mut_{nm}"] = d.wt.map(tab).astype(float), d.mut.map(tab).astype(float)
        d[f"d_{nm}"] = d[f"mut_{nm}"] - d[f"wt_{nm}"]
    for a in AA:
        d[f"mut_{a}"] = (d.mut == a).astype(float)
    terms = ["e_lj_atr", "e_lj_rep", "e_solv_apolar", "e_solv_polar", "e_hb_sc_bb", "e_hb_sc_sc", "e_hb_bb_bb", "e_elec_dh",
             "dTS", "dprior_avg", "d_env_cn_ss_bb", "d_rama", "d_rama_prev", "occ_dvol_loss", "gly_posphi"]
    WF = {k: v for k, v in FAM.items() if k not in ("model_pred",)}
    WF["aa_props"] = [f"{p}_{nm}" for nm in PROPS for p in ("wt", "mut", "d")]
    WF["mut_identity"] = [f"mut_{a}" for a in AA]
    WF["b5_terms"] = terms
    WF["model_pred"] = ["pred"]
    dd = d[d.groupby("site").y.transform("size") >= 10]
    rng = np.random.default_rng(0)
    trw = dd[dd.part == "train"]
    trw = trw[trw.site.isin(rng.choice(trw.site.unique(), 5000, replace=False))].reset_index(drop=True)   # speed
    vaw = dd[dd.part == "val"].reset_index(drop=True)
    trw["w1"], vaw["w1"] = 1.0, 1.0
    print("within-site rows train", len(trw), "val", len(vaw), "var(r_within)", trw.r_within.var())
    res2 = family_scan(trw, vaw, WF, "r_within", "w1", "within")
    res2.to_csv(OUT / "probe_within.csv", index=False)
    print(res2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
