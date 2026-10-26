#!/usr/bin/env python
"""Residual analysis, step 1: decomposition of B5 residuals on Megascale train (OOF) + val (no test rows).

Reads results/residual_analysis/base.parquet (scripts/90). Writes results/residual_analysis/:
  strata_<factor>.csv   per-group n, mean r_raw, mean r_cal, rms r_cal, share of SSE, within-group per-protein rho
  pairs.csv             wt->mut pair table; outliers_pos.csv / outliers_neg.csv (top 1% r_cal, ~50 listed each)
  variance.csv          site-level vs within-site variance of r_cal (method of moments random-effects ICC)
  headroom.csv          rho_prot if site-level / within-site parts were perfect
  censoring.csv         residual vs dynamic-range position; interval rows violated by the model
  PYTHONPATH=. .venv/bin/python scripts/91_residual_decompose.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from physddg import eval as E  # noqa: E402

OUT = ROOT / "results" / "residual_analysis"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def rho_prot(d, col, y="y"):
    r = E.evaluate(d.assign(ddg=d[y]), col)
    return r["spearman_prot_median"]


def group_table(d, key, min_n=100):
    tot = (d.r_cal ** 2).sum()
    rows = []
    for k, g in d.groupby(key, observed=True):
        if len(g) < min_n:
            continue
        ws = []
        for _, gp in g.groupby("protein_id"):
            if len(gp) >= 10 and gp.pred.nunique() > 1 and gp.y.nunique() > 1:
                ws.append(spearmanr(gp.pred, gp.y)[0])
        rows.append({"factor": key if isinstance(key, str) else "+".join(key), "group": k, "n": len(g),
                     "frac_rows": len(g) / len(d), "mean_y": g.y.mean(), "mean_pred": g.pred.mean(),
                     "mean_r_raw": g.r_raw.mean(), "mean_r_cal": g.r_cal.mean(), "rms_r_cal": np.sqrt((g.r_cal ** 2).mean()),
                     "sse_share": (g.r_cal ** 2).sum() / tot, "rho_pooled": spearmanr(g.pred, g.y)[0],
                     "rho_within_prot_median": float(np.median(ws)) if ws else np.nan})
    t = pd.DataFrame(rows)
    t["sse_ratio"] = t.sse_share / t.frac_rows
    return t


def variance_components(d, col="r_cal", unit="site"):
    g = d.groupby(unit)[col]
    n = g.transform("size")
    dd = d[n >= 5]
    g = dd.groupby(unit)[col]
    means, ns, vw = g.mean(), g.size(), g.var(ddof=1)
    sw2 = float(np.average(vw, weights=ns - 1))
    n0 = (ns.sum() - (ns ** 2).sum() / ns.sum()) / (len(ns) - 1)
    msb = float((ns * (means - dd[col].mean()) ** 2).sum() / (len(ns) - 1))
    sb2 = max(0.0, (msb - sw2) / n0)
    return {"unit": unit, "col": col, "n_units": len(ns), "n_rows": int(ns.sum()), "var_total": float(dd[col].var()),
            "var_between": sb2, "var_within": sw2, "icc": sb2 / (sb2 + sw2)}


def headroom(d):
    """Per-protein median rho on point rows for oracle combinations (sites with >= 2 variants for LOO)."""
    d = d.copy()
    g = d.groupby("site")
    n, sy, sr, sp = g.y.transform("size"), g.y.transform("sum"), g.r_cal.transform("sum"), g.pred.transform("sum")
    d["site_n"] = n
    pc = d.a_p + d.b_p * d.pred            # calibrated prediction
    d["pc"] = pc
    out = {"model (calibrated pred; = B5 ranking)": "pc"}
    d["o_site_insample"] = pc + sr / n
    d["o_site_loo"] = pc + np.where(n > 1, (sr - d.r_cal) / (n - 1).clip(lower=1), 0.0)
    psite = g.pc.transform("mean") if "pc" in d else None
    d["o_within_perfect"] = d.groupby("site").pc.transform("mean") + (d.y - sy / n)
    d["o_sitemean_loo_only"] = np.where(n > 1, (sy - d.y) / (n - 1).clip(lower=1), np.nan)
    d["o_sitemean_loo_plus_model_within"] = d.o_sitemean_loo_only + (pc - d.groupby("site").pc.transform("mean"))
    out.update({"+ site-mean residual (in-sample oracle)": "o_site_insample",
                "+ site-mean residual (leave-one-out)": "o_site_loo",
                "model site means + perfect within-site": "o_within_perfect",
                "LOO site-mean of y only (literature oracle)": "o_sitemean_loo_only",
                "LOO site-mean of y + model within-site": "o_sitemean_loo_plus_model_within"})
    rows = []
    for part in ("train", "val"):
        for tag in ("all", "natural"):
            g2 = d[(d.part == part) & ((d.domain_origin == "natural") | (tag == "all"))]
            for lab, col in out.items():
                gg = g2[g2[col].notna()]
                rows.append({"part": part, "subset": tag, "scenario": lab, "rho_prot": rho_prot(gg, col),
                             "rho_pooled": spearmanr(gg[col], gg.y)[0], "n": len(gg)})
    return pd.DataFrame(rows)


def main():
    d0 = pd.read_parquet(OUT / "base.parquet")
    iv = d0[d0.y.isna()]                      # censored interval rows
    d = d0[d0.y.notna()].copy()
    print("rows", len(d), "interval rows", len(iv))
    # ---------------------------------------------------------------- factors
    d["rsa_bin"] = pd.cut(d.x_rsa, [-0.01, 0.05, 0.15, 0.25, 0.4, 0.6, 2], labels=["<.05", ".05-.15", ".15-.25", ".25-.4", ".4-.6", ">.6"])
    d["ncb10_bin"] = pd.cut(d.x_ncb10, [-1, 8, 12, 16, 20, 24, 99])
    d["term_bin"] = pd.cut(d.x_dist_term, [-1, 0, 2, 5, 10, 999], labels=["0", "1-2", "3-5", "6-10", ">10"])
    d["plddt_bin"] = pd.cut(d.x_plddt, [0, 50, 70, 80, 90, 101])
    d["dgwt_bin"] = pd.cut(d.dG_wt, [-5, 1, 2, 3, 4, 10])
    d["helixpos"] = np.select([d.x_ncap == 1, d.x_n1 + d.x_n2 + d.x_n3 > 0, d.x_ccap == 1, d.x_c1 + d.x_c2 + d.x_c3 > 0,
                               d.x_helix_int == 1, d.x_strand_edge == 1, d.x_strand_centre == 1],
                              ["Ncap", "N1-3", "Ccap", "C1-3", "helix_int", "strand_edge", "strand_centre"], "other/loop")
    d["dvol_bin"] = pd.cut(d.dvol, [-2, -0.6, -0.3, -0.05, 0.05, 0.3, 0.6, 2])
    cls = {**{a: "apolar" for a in "AVILMC"}, **{a: "aromatic" for a in "FWY"}, **{a: "polar" for a in "STNQH"},
           **{a: "charged" for a in "DEKR"}, "G": "G", "P": "P"}
    d["wt_cls"], d["mut_cls"] = d.wt.map(cls), d.mut.map(cls)
    d["bur_cls"] = np.where(d.x_rsa < 0.25, "buried", "exposed")
    factors = ["wt", "mut", "rsa_bin", "ss8", "x_basin", "term_bin", "ncb10_bin", "domain_origin", "dgwt_bin",
               "plddt_bin", "helixpos", "dvol_bin", ["wt_cls", "mut_cls"], ["bur_cls", "wt_cls", "mut_cls"],
               ["helixpos", "mut"], ["bur_cls", "mut"]]
    allt = []
    for f in factors:
        t = group_table(d, f, min_n=150 if isinstance(f, list) else 100)
        allt.append(t)
    T = pd.concat(allt, ignore_index=True)
    T.to_csv(OUT / "strata.csv", index=False)
    for f in ["wt", "mut", "rsa_bin", "ss8", "x_basin", "term_bin", "ncb10_bin", "domain_origin", "dgwt_bin", "plddt_bin",
              "helixpos", "dvol_bin"]:
        print(f"\n== {f}")
        print(T[T.factor == f].drop(columns=["factor"]).round(3).to_string(index=False))
    for f in ["wt_cls+mut_cls", "bur_cls+wt_cls+mut_cls", "helixpos+mut", "bur_cls+mut"]:
        t = T[T.factor == f].sort_values("mean_r_cal")
        print(f"\n== {f} (most negative / positive mean r_cal)")
        print(pd.concat([t.head(8), t.tail(8)]).drop(columns=["factor"]).round(3).to_string(index=False))
    # ---------------------------------------------------------------- substitution level (site effect removed)
    d["r_within"] = d.r_cal - d.groupby("site").r_cal.transform("mean")
    W = []
    for f in (["bur_cls", "wt_cls", "mut_cls"], ["helixpos", "mut"], ["x_basin", "mut"], ["wt", "mut"], ["bur_cls", "mut"],
              ["ss3", "mut"]):
        for k, g in d.groupby(f, observed=True):
            if len(g) >= 150:
                W.append({"factor": "+".join(f), "group": k, "n": len(g), "mean_r_within": g.r_within.mean(),
                          "rms_r_within": np.sqrt((g.r_within ** 2).mean()), "within_sse_share": (g.r_within ** 2).sum() / (d.r_within ** 2).sum(),
                          "frac_rows": len(g) / len(d)})
    W = pd.DataFrame(W)
    W.to_csv(OUT / "within_site_classes.csv", index=False)
    for f in W.factor.unique():
        t = W[W.factor == f].sort_values("mean_r_within")
        print(f"\n== within-site residual (r_cal - site mean) by {f}: most over- / under-predicted classes")
        print(pd.concat([t.head(6), t.tail(6)]).drop(columns=["factor"]).round(3).to_string(index=False))
    # ---------------------------------------------------------------- pairs
    P = group_table(d, ["wt", "mut"], min_n=100)
    P.to_csv(OUT / "pairs.csv", index=False)
    print("\n== wt->mut pairs: largest |mean r_cal| (n>=100)")
    print(P.reindex(P.mean_r_cal.abs().sort_values(ascending=False).index).head(25).drop(columns=["factor"]).round(3).to_string(index=False))
    # ---------------------------------------------------------------- variance components
    V = [variance_components(d, c, u) for c in ("r_raw", "r_cal") for u in ("site", "protein_id")]
    for part in ("train", "val"):
        V.append({**variance_components(d[d.part == part], "r_cal", "site"), "part": part})
    # how much of site-level variance is explained by protein offsets (raw only)
    V = pd.DataFrame(V)
    V.to_csv(OUT / "variance.csv", index=False)
    print("\n== variance components\n", V.round(4).to_string(index=False))
    # measurement-noise floor (CI-based) vs residual variance
    print("mean sigma_meas^2", float((d.sigma_meas ** 2).mean()), "var r_cal", float(d.r_cal.var()))
    # ---------------------------------------------------------------- headroom
    H = headroom(d)
    H.to_csv(OUT / "headroom.csv", index=False)
    print("\n== headroom\n", H.round(4).to_string(index=False))
    # ---------------------------------------------------------------- censoring
    d["ddg_cap"] = d.dG_wt + 1.0                          # mutant dG floor -1 => ddg <= dG_wt + 1
    d["head_cap"] = d.ddg_cap - d.y
    d["pred_over_cap"] = d.pred - d.ddg_cap
    C = []
    for lab, m in (("all", np.ones(len(d), bool)), ("flag_mut_floor", d.flag_mut_floor.fillna(False).astype(bool)),
                   ("flag_mut_ceiling", d.flag_mut_ceiling.fillna(False).astype(bool)),
                   ("flag_wt_high", d.flag_wt_high.fillna(False).astype(bool)),
                   ("flag_wt_low", d.flag_wt_low.fillna(False).astype(bool)),
                   ("flag_lowconf", d.flag_lowconf.fillna(False).astype(bool)),
                   ("flag_protease_disagree", d.flag_protease_disagree.fillna(False).astype(bool)),
                   ("pred > dG_wt+1", (d.pred > d.ddg_cap).to_numpy()), ("pred > dG_wt+1.5", (d.pred > d.ddg_cap + 0.5).to_numpy()),
                   ("pred < dG_wt-5 (stab. ceiling)", (d.pred < d.dG_wt - 5).to_numpy())):
        g = d[m]
        C.append({"subset": lab, "n": len(g), "frac": len(g) / len(d), "mean_r_raw": g.r_raw.mean(), "mean_r_cal": g.r_cal.mean(),
                  "rms_r_cal": np.sqrt((g.r_cal ** 2).mean()), "sse_share": (g.r_cal ** 2).sum() / (d.r_cal ** 2).sum()})
    # interval rows: lower-bound rows (mutant unfolded) with the model below the bound
    lo = iv[np.isfinite(iv.lo) & ~np.isfinite(iv.hi)]
    C.append({"subset": "interval rows dG<-1 (ddg>=lo): pred<lo", "n": len(lo), "frac": float((lo.pred < lo.lo).mean()),
              "mean_r_raw": float((lo.lo - lo.pred).clip(lower=0).mean())})
    hi = iv[np.isfinite(iv.hi) & ~np.isfinite(iv.lo)]
    C.append({"subset": "interval rows dG>5 (ddg<=hi): pred>hi", "n": len(hi), "frac": float((hi.pred > hi.hi).mean()),
              "mean_r_raw": float((hi.hi - hi.pred).clip(upper=0).mean())})
    C = pd.DataFrame(C)
    C.to_csv(OUT / "censoring.csv", index=False)
    print("\n== censoring\n", C.round(3).to_string(index=False))
    # per-protein: does the model's ranking suffer more in domains with low dG_wt (compressed range)?
    pp = []
    for pid, g in d.groupby("protein_id"):
        pp.append({"protein_id": pid, "part": g.part.iloc[0], "origin": g.domain_origin.iloc[0], "dG_wt": g.dG_wt.iloc[0],
                   "n": len(g), "rho": spearmanr(g.pred, g.y)[0], "b_p": g.b_p.iloc[0], "a_p": g.a_p.iloc[0],
                   "sd_y": g.y.std(), "mean_y": g.y.mean(), "frac_floor": g.flag_mut_floor.fillna(False).mean(),
                   "mean_plddt": g.x_mean_plddt.iloc[0], "L": g.x_L.iloc[0], "frac_helix": g.x_frac_helix.iloc[0],
                   "noise": float(np.sqrt((g.sigma_meas ** 2).mean()))})
    pp = pd.DataFrame(pp)
    pp.to_csv(OUT / "per_protein.csv", index=False)
    cc = {c: spearmanr(pp[c], pp.rho, nan_policy="omit")[0] for c in ("dG_wt", "sd_y", "mean_y", "frac_floor", "mean_plddt", "L", "frac_helix", "noise", "b_p")}
    print("\n== per-protein rho vs protein descriptors (Spearman over proteins)\n", {k: round(v, 3) for k, v in cc.items()})
    print(pp.groupby("origin")[["rho", "b_p", "dG_wt", "sd_y"]].median().round(3))
    # ---------------------------------------------------------------- outliers
    ctx = ["protein_id", "part", "domain_origin", "pdb_resnum", "wt", "mut", "y", "pred", "r_raw", "r_cal", "dG_wt", "dG_mut",
           "x_rsa", "x_ncb10", "ss8", "helixpos", "x_basin", "phi", "psi", "x_plddt", "x_dist_term", "x_hb_sc_any",
           "x_sb_partners", "x_catpi", "x_aro_stack", "x_void_frac", "x_q_env8", "s2_dE_lj_rep", "c_e_lj_rep", "c_bur",
           "c_occ_dvol_loss", "c_d_rama", "flag_mut_floor", "flag_lowconf", "sigma_meas"]
    q_hi, q_lo = d.r_cal.quantile(0.99), d.r_cal.quantile(0.01)
    pos, neg = d[d.r_cal >= q_hi], d[d.r_cal <= q_lo]
    for lab, g in (("pos", pos), ("neg", neg)):
        g = g.sort_values("r_cal", ascending=(lab == "neg"))
        g[ctx].head(50).to_csv(OUT / f"outliers_{lab}.csv", index=False)   # ~50 listed; full set = top 1%
        print(f"\n== top-1% {lab} residuals: n={len(g)}, cut {q_hi if lab == 'pos' else q_lo:.2f}; enrichment vs all rows")
        for f in ("wt", "mut", "helixpos", "bur_cls", "domain_origin", "x_basin", "term_bin"):
            e = (g[f].value_counts(normalize=True) / d[f].value_counts(normalize=True)).dropna()
            cnt = g[f].value_counts()
            e = e[cnt.reindex(e.index).fillna(0) >= 15].sort_values(ascending=False)
            print(f"  {f}: " + ", ".join(f"{k} {v:.2f} (n={cnt[k]})" for k, v in e.head(5).items()))
        top_prot = g.protein_id.value_counts().head(8)
        print("  proteins:", dict(top_prot), "sites:", g.site.nunique(), "of rows", len(g))
        print("  per-site concentration: rows in sites with >=3 outliers:",
              int(g.site.map(g.site.value_counts()).ge(3).sum()))
        print(g[ctx].head(50).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
