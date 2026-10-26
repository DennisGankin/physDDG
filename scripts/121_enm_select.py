#!/usr/bin/env python
"""ENM site flexibility as 1-3-weight terms on top of the fitted B6g (stage 2) / B6gs1 (stage 1).
Selection data ONLY: Megascale-train grouped 5-fold CV (same folds as fit.cross_validate) + Megascale-val.

  PYTHONPATH=. .venv/bin/python scripts/121_enm_select.py --stage probe|screen|sh3|all
Outputs (results/enm_flexibility/): probe_site.csv (B6g OOF site-residual vs ENM / static descriptors),
screen.csv (every descriptor x normalisation x functional form, both stages), sh3_sites.csv, strata_oof.csv.

Flexibility f_i (per WT site, positive = flexible; stiffness s_i = -f_i). Normalisations: `z` (per-structure z of
log descriptor), `rank` (per-structure percentile - 0.5), `glog` (log descriptor minus the Megascale-train mean).
Functional forms (new weights sign-free; engine weights keep their physical sign constraints):
  add    w * f_i * [wt != mut]                    site-level offset (NOT a mutant-minus-WT difference; ablation only)
  mulE   ddG = core + w * f_i * dE_engine          dE_engine = sum of the fitted engine-term contributions (8 energies
                                                  + dTS + dprior) of the base model -> (1 + w f) dE (two-stage fit)
  mulC   ddG = core + w * f_i * (core - w0)        whole mutant-minus-WT prediction (incl. statistics, B3+, dref)
  rep    w * f_i * e_lj_rep                       1 weight
  c3     f_i * (e_lj_rep, e_lj_atr, e_solv_apolar + e_solv_polar)   3 weights
  dabs   w * s_i * |dV|                           stiff sites penalise any size change (even under reversal)
  dgrow  w * s_i * max(-dV, 0)                    stiff sites penalise growth only (dV = (V_wt - V_mut)/100 A^3)
  dodd   w * s_i * (-dV)                          antisymmetric variant
Two-stage forms follow scripts/52 fit_attenuated: base fit, then the product column with the base model's
contributions, then a full refit with the extra weight; the base is refitted inside each CV fold.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from physddg import enm  # noqa: E402
from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

_spec = importlib.util.spec_from_file_location("lp52", Path(__file__).with_name("52_ladder_physics.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

ROOT = E.ROOT
OUT = ROOT / "results" / "enm_flexibility"
DESC = ["gnm_msf", "gnm_msf_slow", "anm_msf", "cen_msf", "gnm_msf_plddt", "stiff_local", "stiff_all", "prs_eff",
        "prs_sens", "dfi", "gnm_hinge"]
NORMS = ["z", "rank", "glog"]
FORMS = ["add", "mulE", "mulC", "rep", "c3", "dabs", "dgrow", "dodd"]
TWO_STAGE = {"mulE", "mulC"}
PRIMARY = ("gnm_msf", "z", "mulE")          # pre-registered primary candidate (critique I2: GNM MSF as site scale)
SH3 = [("2LJ3", "44"), ("6SCW", "46"), ("2CDT", "40"), ("1QKX", "41"), ("1PWT", "42"), ("1QLY", "37"), ("6NMW", "37"),
       ("1QLY", "36"), ("6NMW", "36")]
BAR = {"dval": 0.004, "dcv": 0.004, "folds": 4}
_GLOG = {}


# ----------------------------------------------------------------------------- features
def add_enm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge ENM site descriptors (prefix n_) on structure/chain/pdb_resnum."""
    t = enm.enm_site_table(df.structure.unique(), name)
    cols = [c for c in t.columns if c not in ("structure", "chain", "pdb_resnum")]
    t = t.rename(columns={c: f"n_{c}" for c in cols})
    d = df.assign(pdb_resnum=df.pdb_resnum.astype(str), chain=df.chain.astype(str))
    n0 = len(d)
    d = d.merge(t, on=["structure", "chain", "pdb_resnum"], how="left")
    assert len(d) == n0
    # pLDDT-scaled variant only where B-factors are pLDDT (AF models); elsewhere identical to the plain GNM
    af = d["n_bf_as_plddt"].fillna(False).astype(bool) & d["x_mean_plddt"].between(30, 100) if "x_mean_plddt" in d else d["n_bf_as_plddt"].fillna(False).astype(bool)
    if "struct_source" in d:
        af &= d.struct_source.astype(str).str.contains("af", case=False) | d.struct_source.isna()
    for sfx in ("", "_log", "_z", "_rank"):
        d[f"n_gnm_msf_plddt{sfx}"] = d[f"n_gnm_msf_plddt{sfx}"].where(af, d[f"n_gnm_msf{sfx}"])
    return d


def glog_means() -> dict:
    """Megascale-train site means of the log descriptors (fixed reference for the `glog` normalisation)."""
    if not _GLOG:
        f = OUT / "glog_means.json"
        if not f.exists():
            tr = P.train_tables(P.EVAL_CAP)[0]
            site = add_enm(tr[["structure", "chain", "pdb_resnum", "x_mean_plddt"]].drop_duplicates(), "megascale")
            f.write_text(json.dumps({c: float(site[f"n_{c}_log"].mean()) for c in DESC}, indent=1))
        _GLOG.update(json.loads(f.read_text()))
    return _GLOG


def add_glog(frames):
    g = glog_means()
    for d in frames:
        for c in DESC:
            d[f"n_{c}_glog"] = d[f"n_{c}_log"] - g[c]


CTRL = {"ctrl_rsa": ("x_rsa", 1.0), "ctrl_cn": ("x_cn", -1.0), "ctrl_natom8": ("x_natom8", -1.0),
        "ctrl_ncb14": ("x_ncb14", -1.0)}       # static burial/packing controls in the same functional forms


def flex(d: pd.DataFrame, desc: str, norm: str) -> np.ndarray:
    """Site flexibility f (positive = flexible); NaN (isolated residue) -> 0 (neutral)."""
    if desc in CTRL:
        col, sign = CTRL[desc]
        v = d[col].astype(float)
        g = v.groupby(d["structure"])
        v = (g.rank(pct=True) - 0.5) if norm == "rank" else (v - g.transform("mean")) / g.transform("std")
        return np.nan_to_num(sign * v.to_numpy(float))
    if norm == "z":
        v = d[f"n_{desc}_z"]
    elif norm == "rank":
        v = d[f"n_{desc}_rank"] - 0.5
    else:
        v = d[f"n_{desc}_glog"]
    sign = -1.0 if desc in enm.STIFF else 1.0
    return np.nan_to_num(sign * v.to_numpy(float))


def form_columns(d: pd.DataFrame, f: np.ndarray, form: str) -> dict:
    """Linear forms: new feature columns (name -> array). Two-stage forms return {}."""
    s = -f
    dv = d["dvol"].to_numpy(float)
    nonc = (d.wt != d.mut).to_numpy(float)
    if form == "add":
        return {"flx_add": f * nonc}
    if form == "rep":
        return {"flx_rep": f * d.e_lj_rep.to_numpy(float)}
    if form == "c3":
        return {"flx_rep": f * d.e_lj_rep.to_numpy(float), "flx_atr": f * d.e_lj_atr.to_numpy(float),
                "flx_solv": f * (d.e_solv_apolar + d.e_solv_polar).to_numpy(float)}
    if form == "dabs":
        return {"stf_dabs": s * np.abs(dv)}
    if form == "dgrow":
        return {"stf_dgrow": s * np.clip(-dv, 0, None)}
    if form == "dodd":
        return {"stf_dodd": s * (-dv)}
    return {}


def engine_part(model: dict, d: pd.DataFrame, cols: list) -> np.ndarray:
    return sum(model["weights"][c] * d[c].to_numpy(float) for c in cols if c in model["weights"])


def two_stage_col(base: dict, d: pd.DataFrame, f: np.ndarray, form: str, ecols: list) -> np.ndarray:
    if form == "mulE":
        return f * engine_part(base, d, ecols)
    return f * (FT.predict(base, d) - base["intercept"])


def fit_term(train, cfg, f, form, ecols, n_boot=0, name="model"):
    """Fit the base + flexibility term. Returns model dict (two-stage: with `flex_base`, `flex_form`)."""
    if form in TWO_STAGE:
        base = FT.fit(train, cfg)
        col = f"flx_{form}"
        d = train.assign(**{col: two_stage_col(base, train, f, form, ecols)})
        cfg2 = FT.FitConfig(**{**cfg.__dict__, "features": list(cfg.features) + [col]})
        m = FT.fit(d, cfg2, n_boot=n_boot, n_jobs=16, name=name)
        m["flex_base"], m["flex_form"] = base, form
        return m
    cols = form_columns(train, f, form)
    d = train.assign(**cols)
    cfg2 = FT.FitConfig(**{**cfg.__dict__, "features": list(cfg.features) + list(cols)})
    m = FT.fit(d, cfg2, n_boot=n_boot, n_jobs=16, name=name)
    m["flex_form"] = form
    return m


def predict_term(m, d, f, ecols):
    form = m.get("flex_form")
    if form in TWO_STAGE:
        return FT.predict(m, d.assign(**{f"flx_{form}": two_stage_col(m["flex_base"], d, f, form, ecols)}))
    if form:
        return FT.predict(m, d.assign(**form_columns(d, f, form)))
    return FT.predict(m, d)


def new_weights(m):
    return {k: v for k, v in m["weights"].items() if k.startswith(("flx_", "stf_"))}


# ----------------------------------------------------------------------------- data
_DATA = {}
CACHE = ROOT / "data" / "features" / "cache"


def load(stage: int):
    """(train, val, fireprot-train, cfg, ecols, opts) for the stage; built once and cached as a slim pickle so that
    joblib workers read it once instead of receiving a copy per task."""
    if stage in _DATA:
        return _DATA[stage]
    f = CACHE / f"enm_select_s{stage}.pkl"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(_build(stage), f)
    _DATA[stage] = pd.read_pickle(f)
    return _DATA[stage]


def _build(stage: int):
    """Derived Megascale-train / val tables for the stage, slimmed to model columns + ENM flexibility inputs."""
    tr, va, fp = P.train_tables(P.EVAL_CAP)
    sel = P.load_sel()
    o = dict(sel[f"opts_s{stage}"], cxxc=False)
    feats, extra = P.feature_set("b6g", o)
    ecols = P.energy_cols(o)
    out = []
    for d, name in ((tr, "megascale"), (va, "megascale"), (fp, "fireprot")):
        dd = add_enm(P.derive(d, stage, o), name)
        keep = feats + ["y", "lo", "hi", "sigma_meas", "dataset", "cluster", "protein_id", "wt", "mut", "domain_origin",
                        "structure", "chain", "pdb_resnum", "dvol", "x_rsa", "x_cn", "x_ncb14", "x_natom8", "x_mean_plddt",
                        "x_scsc_contacts", "e_solv_apolar", "e_solv_polar", "e_lj_rep", "e_lj_atr"]
        keep += [f"n_{c}_{k}" for c in DESC for k in ("z", "rank", "log")] + ["n_degree"]
        out.append(dd[list(dict.fromkeys(c for c in keep if c in dd))].reset_index(drop=True))
    add_glog(out)
    cfg = P.make_cfg(feats, sel, extra)
    return out[0], out[1], out[2], cfg, ecols, o


def val_scores(pred, va):
    v = va.assign(ddg=va.y, pred=pred)
    v = v[v.ddg.notna()]
    return {"val_natural_prot": E.evaluate(v[v.domain_origin == "natural"], "pred")["spearman_prot_median"],
            "val_all_prot": E.evaluate(v, "pred")["spearman_prot_median"]}


def cv_oof(train, cfg, desc, norm, form, ecols, k=5):
    folds = FT.group_folds(train["cluster"], k, 0)
    oof = np.full(len(train), np.nan)
    for j in range(k):
        tr_, te_ = train[folds != j], train[folds == j]
        if desc is None:
            m = FT.fit(tr_, cfg)
            oof[folds == j] = FT.predict(m, te_)
        else:
            m = fit_term(tr_, cfg, flex(tr_, desc, norm), form, ecols)
            oof[folds == j] = predict_term(m, te_, flex(te_, desc, norm), ecols)
    return oof, folds


def cv_scores(train, oof, folds):
    d = train.assign(ddg=train.y, oof=oof, fold=folds)
    d = d[d.ddg.notna()]
    nat = d[d.domain_origin == "natural"]
    per = [E.evaluate(g, "oof")["spearman_prot_median"] for _, g in nat.groupby("fold")]
    return {"cv_natural_prot": E.evaluate(nat, "oof")["spearman_prot_median"], "cv_all_prot": E.evaluate(d, "oof")["spearman_prot_median"],
            "cv_pearson": E.evaluate(d, "oof")["pearson"], "cv_fold_prot": per}


def run_one(stage, desc, norm, form):
    import torch
    torch.set_num_threads(1)
    train, va, _, cfg, ecols, _ = load(stage)
    if desc is None:
        m = FT.fit(train, cfg)
        pv = FT.predict(m, va)
        w = {}
    else:
        m = fit_term(train, cfg, flex(train, desc, norm), form, ecols)
        pv = predict_term(m, va, flex(va, desc, norm), ecols)
        w = new_weights(m)
    oof, folds = cv_oof(train, cfg, desc, norm, form, ecols)
    r = {"stage": stage, "desc": desc or "BASE", "norm": norm or "", "form": form or "", **val_scores(pv, va),
         **cv_scores(train, oof, folds), "weights": json.dumps({k: round(v, 4) for k, v in w.items()})}
    return r, (oof if (desc, norm, form) == PRIMARY or desc is None else None)


# ----------------------------------------------------------------------------- stages
def stage_probe():
    """Critique I2 first experiment: does ENM flexibility explain the B6g OOF site-level residual beyond contact number?"""
    tr, va, _, cfg, ecols, _ = load(2)
    oof, _ = cv_oof(tr, cfg, None, None, None, ecols)
    d = tr.assign(r=tr.y - oof)
    d = d[d.r.notna()]
    d["r"] = d.r - d.groupby("protein_id").r.transform("mean")          # per-protein offset removed
    d["site"] = d.protein_id + "|" + d.chain + "|" + d.pdb_resnum
    g = d.groupby("site")
    s = g.agg(r=("r", "mean"), n=("r", "size"), rsa=("x_rsa", "first"), cn=("x_cn", "first"), natom8=("x_natom8", "first"),
              ncb14=("x_ncb14", "first"), scsc=("x_scsc_contacts", "first"), protein_id=("protein_id", "first"),
              **{f"{c}_z": (f"n_{c}_z", "first") for c in DESC}, deg=("n_degree", "first"))
    s = s[s.n >= 5].dropna()
    static = ["rsa", "cn", "natom8", "ncb14", "scsc", "deg"]
    Z = lambda X: np.c_[np.ones(len(X)), (X - X.mean(0)) / X.std(0)]
    A = Z(s[static].to_numpy(float))
    res_r = s.r - A @ np.linalg.lstsq(A, s.r, rcond=None)[0]
    rows = []
    for c in [f"{x}_z" for x in DESC] + static:
        x = s[c].to_numpy(float)
        res_x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0] if c not in static else np.full(len(x), np.nan)
        B = Z(np.c_[s[static].to_numpy(float), x]) if c not in static else A
        r2_static = 1 - np.var(res_r) / np.var(s.r)
        r2_with = 1 - np.var(s.r - B @ np.linalg.lstsq(B, s.r, rcond=None)[0]) / np.var(s.r)
        bur = s.rsa < 0.25
        rows.append({"descriptor": c, "n_sites": len(s), "spearman_site_resid": spearmanr(x, s.r)[0],
                     "spearman_buried_sites": spearmanr(x[bur], s.r[bur])[0],
                     "partial_r_given_static": float(np.corrcoef(res_x, res_r)[0, 1]) if c not in static else np.nan,
                     "R2_static": r2_static, "R2_static_plus": r2_with, "dR2": r2_with - r2_static,
                     "spearman_with_cn": spearmanr(x, s.cn)[0]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "probe_site.csv", index=False)
    print(t.round(4).to_string(index=False))


def stage_screen(n_jobs=40):
    rows, oofs = [], {}
    for stage in (2, 1):
        load(stage)                                   # build the cache in the parent
        tasks = [(None, None, None)] + [(d, n, f) for d in DESC for n in NORMS for f in FORMS]
        res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_one)(stage, d, n, f) for d, n, f in tasks)
        base = dict(res[0][0])
        base_folds = list(base["cv_fold_prot"])
        for r, oof in res:
            r["d_val_natural"] = r["val_natural_prot"] - base["val_natural_prot"]
            r["d_val_all"] = r["val_all_prot"] - base["val_all_prot"]
            r["d_cv_natural"] = r["cv_natural_prot"] - base["cv_natural_prot"]
            r["d_cv_all"] = r["cv_all_prot"] - base["cv_all_prot"]
            r["folds_up"] = int(sum(a > b for a, b in zip(r["cv_fold_prot"], base_folds)))
            r["passes_bar"] = bool(r["d_val_natural"] >= BAR["dval"] and r["d_cv_natural"] >= BAR["dcv"]
                                   and r["folds_up"] >= BAR["folds"])
            r["cv_fold_prot"] = json.dumps([round(x, 4) for x in r["cv_fold_prot"]])
            rows.append(r)
            if oof is not None:
                oofs[f"s{stage}_{r['desc']}_{r['norm']}_{r['form']}"] = oof
        print(f"stage {stage} done", flush=True)
        pd.DataFrame(oofs).to_parquet(OUT / f"oof_s{stage}.parquet") if stage == 2 else None
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "screen.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 600):
        for stage in (2, 1):
            s = t[t.stage == stage].sort_values("d_cv_natural", ascending=False)
            print(f"== stage {stage}: base val {s[s.desc == 'BASE'].val_natural_prot.iloc[0]:.4f} "
                  f"cv {s[s.desc == 'BASE'].cv_natural_prot.iloc[0]:.4f}")
            print(s.head(30)[["desc", "norm", "form", "d_val_natural", "d_val_all", "d_cv_natural", "d_cv_all", "folds_up",
                              "passes_bar", "weights"]].round(4).to_string(index=False))
            print("passing:", int(s.passes_bar.sum()))


def stage_control(n_jobs=40):
    """Static burial/packing descriptors in the same multiplicative forms (is the ENM gain just burial?)."""
    rows = []
    for stage in (2, 1):
        load(stage)
        tasks = [(None, None, None)] + [(d, n, f) for d in CTRL for n in ("rank", "z") for f in ("mulC", "mulE", "add")] + \
                [("cen_msf", "rank", "mulC")]
        res = Parallel(n_jobs=n_jobs)(delayed(run_one)(stage, d, n, f) for d, n, f in tasks)
        base = dict(res[0][0])
        for r, _ in res:
            r.update(d_val_natural=r["val_natural_prot"] - base["val_natural_prot"],
                     d_cv_natural=r["cv_natural_prot"] - base["cv_natural_prot"],
                     d_val_all=r["val_all_prot"] - base["val_all_prot"], d_cv_all=r["cv_all_prot"] - base["cv_all_prot"],
                     folds_up=int(sum(a > b for a, b in zip(r["cv_fold_prot"], base["cv_fold_prot"]))))
            rows.append({k: v for k, v in r.items() if k != "cv_fold_prot"})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "control.csv", index=False)
    print(t[["stage", "desc", "norm", "form", "d_val_natural", "d_val_all", "d_cv_natural", "d_cv_all", "folds_up", "weights"]]
          .round(4).to_string(index=False))


def _fit_mulC_multi(train, cfg, fs):
    base = FT.fit(train, cfg)
    core = FT.predict(base, train) - base["intercept"]
    cols = {f"flx_mulC_{k}": f * core for k, f in fs.items()}
    m = FT.fit(train.assign(**cols), FT.FitConfig(**{**cfg.__dict__, "features": list(cfg.features) + list(cols)}))
    m["flex_base"] = base
    return m


def _pred_mulC_multi(m, d, fs):
    core = FT.predict(m["flex_base"], d) - m["flex_base"]["intercept"]
    return FT.predict(m, d.assign(**{f"flx_mulC_{k}": f * core for k, f in fs.items()}))


def _incr_one(stage, specs):
    import torch
    torch.set_num_threads(1)
    tr, va, _, cfg, _, _ = load(stage)
    F = lambda d: {f"{de}_{no}": flex(d, de, no) for de, no in specs}
    m = _fit_mulC_multi(tr, cfg, F(tr))
    pv = _pred_mulC_multi(m, va, F(va))
    folds = FT.group_folds(tr["cluster"], 5, 0)
    oof = np.full(len(tr), np.nan)
    for j in range(5):
        mj = _fit_mulC_multi(tr[folds != j], cfg, F(tr[folds != j]))
        oof[folds == j] = _pred_mulC_multi(mj, tr[folds == j], F(tr[folds == j]))
    return {"stage": stage, "specs": "+".join(f"{a}:{b}" for a, b in specs), **val_scores(pv, va), **cv_scores(tr, oof, folds),
            "weights": json.dumps({k: round(v, 4) for k, v in new_weights(m).items()})}


def stage_incr():
    """Incremental value of ENM over the best static control in the same multiplicative (mulC) form."""
    rows = []
    for stage in (2, 1):
        load(stage)
        specs = [[("ctrl_cn", "rank")], [("ctrl_natom8", "z")], [("cen_msf", "rank")],
                 [("ctrl_cn", "rank"), ("cen_msf", "rank")], [("ctrl_natom8", "z"), ("cen_msf", "rank")]]
        rows += Parallel(n_jobs=10)(delayed(_incr_one)(stage, sp) for sp in specs)
    t = pd.DataFrame(rows)
    for stage in (2, 1):
        for ref in ("ctrl_cn:rank", "ctrl_natom8:z"):
            b = t[(t.stage == stage) & (t.specs == ref)].iloc[0]
            m = (t.stage == stage) & (t.specs == f"{ref}+cen_msf:rank")
            t.loc[m, "d_val_natural_vs_ctrl"] = t.loc[m, "val_natural_prot"] - b.val_natural_prot
            t.loc[m, "d_cv_natural_vs_ctrl"] = t.loc[m, "cv_natural_prot"] - b.cv_natural_prot
            t.loc[m, "folds_up_vs_ctrl"] = [int(sum(x > y for x, y in zip(f, b.cv_fold_prot))) for f in t.loc[m, "cv_fold_prot"]]
    t["cv_fold_prot"] = t.cv_fold_prot.map(lambda v: json.dumps([round(x, 4) for x in v]))
    t.to_csv(OUT / "incremental.csv", index=False)
    print(t.round(4).to_string(index=False))


def stage_sh3():
    """SH3 buried-but-tolerant sites and residual-analysis strata, on train OOF: B6g vs B6g + primary / best term."""
    t = pd.read_csv(OUT / "screen.csv")
    s2 = t[(t.stage == 2) & (t.desc != "BASE")].copy()
    s2["score"] = s2.d_val_natural + s2.d_cv_natural
    best = s2.sort_values("score", ascending=False).iloc[0]
    cands = {"primary": PRIMARY, "best_screen": (best.desc, best.norm, best.form)}
    tr, va, _, cfg, ecols, _ = load(2)
    d = tr.copy()
    d["oof_base"], folds = cv_oof(tr, cfg, None, None, None, ecols)
    for k, (de, no, fo) in cands.items():
        d[f"oof_{k}"], _ = cv_oof(tr, cfg, de, no, fo, ecols)
    d = d[d.y.notna()]
    rows = []
    for pid, rn in SH3:
        g = d[(d.protein_id == pid) & (d.pdb_resnum == rn)]
        if not len(g):
            continue
        prot = d[d.protein_id == pid]
        r = {"protein_id": pid, "pdb_resnum": rn, "wt": g.wt.iloc[0], "n": len(g), "y_mean": g.y.mean(),
             "rsa": g.x_rsa.iloc[0], "gnm_msf_z": g.n_gnm_msf_z.iloc[0], "gnm_msf_rank": g.n_gnm_msf_rank.iloc[0],
             "cen_msf_rank": g.n_cen_msf_rank.iloc[0], "stiff_local_rank": g.n_stiff_local_rank.iloc[0]}
        for k in ["base"] + list(cands):
            # site residual relative to the protein (per-protein offset removed)
            r[f"pred_mean_{k}"] = g[f"oof_{k}"].mean()
            r[f"resid_site_{k}"] = (g.y - g[f"oof_{k}"]).mean() - (prot.y - prot[f"oof_{k}"]).mean()
            r[f"site_pct_pred_{k}"] = float((prot.groupby("pdb_resnum")[f"oof_{k}"].mean() < g[f"oof_{k}"].mean()).mean())
        rows.append(r)
    sh = pd.DataFrame(rows)
    sh.to_csv(OUT / "sh3_sites.csv", index=False)
    print(json.dumps({k: list(v) for k, v in cands.items()}))
    print(sh.round(3).to_string(index=False))
    # strata (train OOF, per-protein median rho)
    bur = d.x_rsa < 0.25
    st = {"all": np.ones(len(d), bool), "natural": d.domain_origin == "natural", "buried": bur, "exposed": ~bur,
          "buried small->large": bur & (d.dvol < -0.3), "buried large->small": bur & (d.dvol > 0.3),
          "SH3 family (fam0409)": d.cluster == "fam0409", "SH3 buried": (d.cluster == "fam0409") & bur}
    rr = []
    for nm, msk in st.items():
        g = d[np.asarray(msk)]
        for k in ["base"] + list(cands):
            e = E.evaluate(g.assign(ddg=g.y), f"oof_{k}")
            rr.append({"stratum": nm, "model": k, "n": e["n"], "spearman_prot_median": e["spearman_prot_median"],
                       "spearman": e["spearman"]})
    rr = pd.DataFrame(rr)
    rr.to_csv(OUT / "strata_oof.csv", index=False)
    print(rr.pivot(index="stratum", columns="model", values="spearman_prot_median").round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (["probe", "screen", "sh3"] if a.stage == "all" else [a.stage]):
        {"probe": stage_probe, "screen": stage_screen, "sh3": stage_sh3, "control": stage_control, "incr": stage_incr}[s]()


if __name__ == "__main__":
    main()
