#!/usr/bin/env python
"""Evaluate fitted physical parameters theta (scripts/111) with the standard B6gs1 fit pipeline (scripts/52).

  PYTHONPATH=. .venv/bin/python scripts/113_diffparams_eval.py --stage select    # CV traces -> lambda, iteration
  ... --stage confirm --cand NAME=fitjson[:iter] ...    # full-protocol: refit on full train, val + grouped CV
  ... --stage transfer --cand ...                       # FireProt-train (never fitted on) + FireProt-only theta fits
  ... --stage params                                    # parameter tables (literature vs fitted +- bootstrap SE)
  ... --stage test --cand ...                           # THE single test look (Megascale-test, FireProt-test,
                                                        #  S669/S461 strict, Domainome), paired bootstrap
Selection uses Megascale-train (grouped CV, folds of physddg.fit.cross_validate) and Megascale-val only.
Outputs: results/diff_params/*.csv, *.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _imp("l52", "52_ladder_physics.py")
F111 = _imp("f111", "111_diffparams_fit.py")
F112 = _imp("f112", "112_diffparams_features.py")
from physddg import diffparams as D  # noqa: E402
from physddg import eval as EV  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = ROOT / "results" / "diff_params"
FITS = OUT / "fits"
KS = M.KS
S1COLS = [f"s1_dE_{t}" for t in D.TERMS] + ["s1_dTS", "s1_dprior_avg", "s1_dref_apolar", "s1_dref_polar", "s1_dF"]
GPUS = [0, 1]


def load_fit(name: str) -> dict:
    return json.loads((FITS / f"{name}.json").read_text())


def theta_at(fit: dict, it: int | None) -> np.ndarray:
    if it is None:
        return np.asarray(fit["theta"], float)
    tr = {t["iter"]: t for t in fit["trajectory"]}
    return np.asarray(tr[min(it, max(tr))]["theta"], float)


def swap_s1(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Replace the stage-1 engine columns of a 52-style table by the theta features (merge on KEYS+structure)."""
    f = feat[KS + S1COLS].drop_duplicates(KS).copy()
    f["pdb_resnum"] = EV._norm_resnum(f["pdb_resnum"])
    d = df.drop(columns=[c for c in S1COLS if c in df]).copy()
    d["pdb_resnum"] = EV._norm_resnum(d["pdb_resnum"].astype(str))
    out = d.merge(f, on=KS, how="left")
    assert len(out) == len(df)
    return out


def feats_at(theta, datasets, rows=None) -> dict:
    paths = F112.features(datasets, theta, GPUS, 8, rows=rows)
    return {n: pd.read_parquet(p) for n, p in paths.items()}


def model_setup():
    return F111.model_setup()


def fit_b6gs1(tr: pd.DataFrame, name="B6gs1-theta", n_boot=0) -> dict:
    sel, o, feats, extra = model_setup()
    cfg = M.make_cfg(feats, sel, extra)
    return FT.fit(M.derive(tr, 1, o), cfg, n_boot=n_boot, n_jobs=32, name=name)


def cv_b6gs1(tr: pd.DataFrame) -> dict:
    """Grouped 5-fold CV exactly as scripts/52 (_run_config): natural per-protein rho overall and per fold."""
    sel, o, feats, extra = model_setup()
    cfg = M.make_cfg(feats, sel, extra)
    trd = M.derive(tr, 1, o)
    oof = FT.cross_validate(trd, cfg, k=5)
    nat = oof[(oof.domain_origin == "natural") & oof.y.notna()]
    per = [EV.evaluate(g.assign(ddg=g.y), "oof")["spearman_prot_median"] for _, g in nat.groupby("fold")]
    allp = oof[oof.y.notna()]
    per_all = [EV.evaluate(g.assign(ddg=g.y), "oof")["spearman_prot_median"] for _, g in allp.groupby("fold")]
    return {"cv_natural_prot": FT.cv_metrics(nat, by_fold=False)["spearman_prot_median"],
            "cv_all_prot": FT.cv_metrics(allp, by_fold=False)["spearman_prot_median"],
            "cv_pearson": FT.cv_metrics(allp, by_fold=False)["pearson"],
            "cv_fold_nat": per, "cv_fold_all": per_all}


def cv_fold_theta(tr: pd.DataFrame, feats_by_fold: dict) -> dict:
    """CV where fold j's features come from theta_j (fitted without fold j): weights fitted on the other folds'
    full rows at theta_j, predictions for fold j at theta_j."""
    sel, o, feats, extra = model_setup()
    cfg = M.make_cfg(feats, sel, extra)
    folds = FT.group_folds(tr["cluster"], 5, 0)
    oof = []
    for j, fe in feats_by_fold.items():
        trd = M.derive(swap_s1(tr, fe), 1, o)
        m = FT.fit(trd[folds != j], cfg)
        te = trd[folds == j].copy()
        te["oof"], te["fold"] = FT.predict(m, te), j
        oof.append(te)
    oof = pd.concat(oof)
    nat = oof[(oof.domain_origin == "natural") & oof.y.notna()]
    per = [EV.evaluate(g.assign(ddg=g.y), "oof")["spearman_prot_median"] for _, g in nat.groupby("fold")]
    allp = oof[oof.y.notna()]
    per_all = [EV.evaluate(g.assign(ddg=g.y), "oof")["spearman_prot_median"] for _, g in allp.groupby("fold")]
    return {"cv_natural_prot": FT.cv_metrics(nat, by_fold=False)["spearman_prot_median"],
            "cv_all_prot": FT.cv_metrics(allp, by_fold=False)["spearman_prot_median"],
            "cv_pearson": FT.cv_metrics(allp, by_fold=False)["pearson"], "cv_fold_nat": per, "cv_fold_all": per_all}


def val_eval(m, va) -> dict:
    sel, o, feats, extra = model_setup()
    v = M.derive(va, 1, o)
    r = M.val_metrics(m, v)
    vv = v[v.y.notna()].assign(ddg=lambda d: d.y)
    r["val_all_pearson"] = EV.evaluate(vv.assign(pred=FT.predict(m, vv)), "pred")["pearson"]
    return r


# ----------------------------------------------------------------------------- stage: select (CV traces only)
def stage_select():
    """Held-out-subsample traces of the fold fits -> mean over folds per (lambda, iteration)."""
    rows = []
    for p in sorted(FITS.glob("megascale_lam*_fold*.json")):
        r = json.loads(p.read_text())
        base = r["trajectory"][0]
        for t in r["trajectory"]:
            rows.append({"lam": r["lam"], "fold": r["fold"], "iter": t["iter"], "obj": t["obj"],
                         "d_ho_nat_prot": t.get("ho_nat_prot", np.nan) - base.get("ho_nat_prot", np.nan),
                         "d_ho_prot": t.get("ho_prot", np.nan) - base.get("ho_prot", np.nan),
                         "d_ho_pearson": t.get("ho_pearson", np.nan) - base.get("ho_pearson", np.nan),
                         "max_abs_theta": float(np.abs(t["theta"]).max())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cv_traces.csv", index=False)
    # carry the last iterate forward for folds that converged earlier
    out = []
    for lam, g in t.groupby("lam"):
        mx = int(g.iter.max())
        for it in range(mx + 1):
            vals = []
            for f, gf in g.groupby("fold"):
                gg = gf[gf.iter <= it]
                vals.append(gg.iloc[-1])
            v = pd.DataFrame(vals)
            out.append({"lam": lam, "iter": it, "n_folds": len(v), "mean_d_ho_nat_prot": v.d_ho_nat_prot.mean(),
                        "mean_d_ho_prot": v.d_ho_prot.mean(), "mean_d_ho_pearson": v.d_ho_pearson.mean(),
                        "folds_up_nat": int((v.d_ho_nat_prot > 0).sum()), "folds_up_all": int((v.d_ho_prot > 0).sum())})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "cv_select.csv", index=False)
    print(s.round(4).to_string(index=False))
    best = s.loc[(s.mean_d_ho_nat_prot + s.mean_d_ho_prot).idxmax()]
    print("BEST (mean held-out natural + all per-protein rho):", best.to_dict())
    return best


def parse_cands(cands):
    out = {}
    for c in cands:
        name, spec = c.split("=", 1)
        if spec == "zero":
            out[name] = (None, None)
            continue
        f, _, it = spec.partition(":")
        out[name] = (f, int(it) if it else None)
    return out


def cand_theta(spec, fold=None):
    f, it = spec
    if f is None:
        return np.zeros(D.NP)
    if fold is not None:
        f = f.replace("_full", f"_fold{fold}")
    return theta_at(load_fit(f), it)


# ----------------------------------------------------------------------------- stage: confirm (train/val only)
def stage_confirm(cands: dict, do_cv=True):
    tr, va, fp = M.train_tables()
    rows = []
    for name, spec in cands.items():
        th = cand_theta(spec)
        fe = feats_at(th, ["megascale"])["megascale"]
        trs, vas = swap_s1(tr, fe), swap_s1(va, fe)
        m = fit_b6gs1(trs, name)
        FT.save_model(m, OUT / "models" / f"{name}.json")
        r = {"cand": name, "fit": spec[0], "iter": spec[1], **val_eval(m, vas)}
        if do_cv:
            if spec[0] is None:
                cv = cv_b6gs1(trs)
            else:
                fb = {j: feats_at(cand_theta(spec, j), ["megascale"])["megascale"] for j in range(5)}
                cv = cv_fold_theta(tr, fb)
            r.update({k: (json.dumps([round(x, 4) for x in v]) if isinstance(v, list) else v) for k, v in cv.items()})
        # assay transfer without any test data: Megascale-fitted model on FireProt-train (never used for fitting)
        fpe = feats_at(th, ["fireprot"])["fireprot"]
        sel, o, feats, extra = model_setup()
        fd = M.derive(swap_s1(fp, fpe), 1, o)
        fd = fd[fd.y.notna()].assign(ddg=lambda d: d.y)
        e = EV.evaluate(fd.assign(pred=FT.predict(m, fd)), "pred")
        r.update({"fp_train_spearman": e["spearman"], "fp_train_pearson": e["pearson"],
                  "fp_train_prot": e["spearman_prot_median"]})
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    t = pd.DataFrame(rows)
    p = OUT / "confirm.csv"
    if p.exists():
        old = pd.read_csv(p)
        t = pd.concat([old[~old.cand.isin(t.cand)], t], ignore_index=True)
    t.to_csv(p, index=False)
    print(t.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- stage: params
def stage_params(cands: dict):
    """Parameter table: literature vs fitted (scale and natural units) +- cluster-bootstrap SE of the scale."""
    lit = D.literature_values()
    rows = []
    for name, spec in cands.items():
        th = cand_theta(spec)
        boots = sorted(FITS.glob(f"{spec[0].replace('_full', '')}_boot*.json")) if spec[0] else []
        bt = np.array([json.loads(p.read_text())["theta"] for p in boots]) if boots else None
        se = np.exp(bt).std(0, ddof=1) if bt is not None and len(bt) > 1 else None
        for k, r in enumerate(D.describe(th, se)):
            r["cand"] = name
            r["n_boot"] = 0 if bt is None else len(bt)
            if bt is not None and len(bt) > 1:
                r["scale_boot_lo"], r["scale_boot_hi"] = np.exp(np.percentile(bt[:, k], [2.5, 97.5]))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "params.csv", index=False)
    print(t.round(3).to_string(index=False))
    del lit


# ----------------------------------------------------------------------------- stage: test (single look)
TEST_SETS = {"megascale": ["natural", "natural_strict", "all"], "fireprot": ["all", "strict"], "s669": ["strict"],
             "s461": ["strict"], "domainome": ["all", "flag_kept"]}


def stage_test(cands: dict, n_boot=1000):
    tr, _, _ = M.train_tables()
    models, feats = {}, {}
    shipped = FT.load_model(ROOT / "results" / "ladder_physics" / "models" / "B6gs1.json")
    for name, spec in cands.items():
        th = cand_theta(spec)
        feats[name] = feats_at(th, ["megascale"] + [b for b in TEST_SETS if b != "megascale"])
        mp = OUT / "models" / f"{name}.json"
        models[name] = FT.load_model(mp) if mp.exists() else fit_b6gs1(swap_s1(tr, feats[name]["megascale"]), name)
    rows, pairs = [], []
    for bench, subs in TEST_SETS.items():
        b = M.add_extra(M.add_phys(M.L.bench_table(bench), bench), bench)
        b["cxxc"] = M.cxxc_flags(b)
        o = model_setup()[1]
        b["B6gs1"] = FT.predict(shipped, M.derive(b, 1, o))
        for name in cands:
            b[name] = FT.predict(models[name], M.derive(swap_s1(b, feats[name][bench]), 1, o))
        b = M.attach_competitors(b, bench)
        cols = ["B6gs1"] + list(cands) + [c for c in ("FoldX", "FoldX-clip") if c in b]
        b[[c for c in KS + ["ddg", "score", "cluster", "domain_origin"] + cols if c in b]].to_parquet(
            OUT / f"preds_{bench}.parquet", index=False)
        for sub in subs:
            d = EV.apply_subset(b, M.SUBSETS[bench][sub])
            common = d[["B6gs1"] + list(cands)].notna().all(1)
            d = d[common]
            target = "ddg" if d.ddg.notna().any() else "score"
            for c in cols:
                dd = d[d[c].notna()]
                r = EV.evaluate(dd, c, target=target)
                rows.append({"bench": bench, "subset": sub, "model": c, **{k: r[k] for k in M.METRICS}})
            nb = 200 if bench == "domainome" else n_boot
            for c in cands:
                for vs in ["B6gs1"] + [x for x in ("FoldX-clip",) if x in d] + [x for x in cands if x != c]:
                    both = d[d[c].notna() & d[vs].notna()]
                    pb = EV.paired_bootstrap(both, c, vs, target=target, n_boot=nb)
                    pr = {"bench": bench, "subset": sub, "model": c, "vs": vs, "n": len(both)}
                    for k in ("spearman", "spearman_prot_median", "pearson"):
                        if k in pb:
                            pr.update({f"d_{k}": pb[k]["delta"], f"d_{k}_lo": pb[k]["lo"], f"d_{k}_hi": pb[k]["hi"]})
                    pairs.append(pr)
    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "test_paired.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(pairs).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "confirm", "params", "test"])
    ap.add_argument("--cand", nargs="*", default=[])
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    global GPUS
    GPUS = [int(g) for g in a.gpus.split(",")]
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    cands = parse_cands(a.cand)
    if a.stage == "select":
        stage_select()
    elif a.stage == "confirm":
        stage_confirm(cands, not a.no_cv)
    elif a.stage == "params":
        stage_params(cands)
    elif a.stage == "test":
        stage_test(cands)


if __name__ == "__main__":
    main()
