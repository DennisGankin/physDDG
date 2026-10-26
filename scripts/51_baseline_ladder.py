#!/usr/bin/env python
"""Baseline ladder B0-B3+ (fitted on Megascale-train; selection on Megascale-val, grouped CV secondary),
evaluated on every benchmark with physddg/eval.py, FoldX 5.1 native on identical rows, G6 transfer check.

  PYTHONPATH=. .venv/bin/python scripts/51_baseline_ladder.py --stage all   (select | fit | eval | transfer)
Outputs results/baselines_ladder/ (models/*.json, selection_*.csv, metrics*.csv, paired_vs_foldx.csv,
site_decomposition.csv, ssym_antisymmetry.csv, paired_af_agreement.csv, transfer_*.csv; preds_*.parquet git-ignored).
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import features_simple as F  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = E.ROOT / "results" / "baselines_ladder"
KEYS = list(E.KEYS)
PROC = E.ROOT / "data" / "processed"
ENV_OPTIONS = ["de_env", "de_env_bb", "de_env_rsa"]


def specs(env: str) -> dict:
    """name -> (features, use_ref). B3+ <= 10 weights."""
    return {
        "Bref": ([], True),
        "B1": (["bur"], False),
        "B2": (["bur", "bur_dhyd", "bur_dvol_loss", "bur_dvol_gain"], True),
        "B2occ": (["occn", "occ_dhyd", "occ_dvol_loss", "occ_dvol_gain"], True),
        "B3": ([env, "bur"], False),
        "B3r": ([env, "bur"], True),
        "B3+": ([env, "bur", "de_rama", "de_prepro", "pro_nhhb", "gly_posphi", "ss_broken", "cis_pro_lost",
                 "occ_dvol_gain", "occ_dvol_loss"], True),
    }


# ----------------------------------------------------------------------------- data
def with_feats(b: pd.DataFrame, feat_name: str) -> pd.DataFrame:
    f = F.load(feat_name).drop(columns=["site_found", "wt_match"]).drop_duplicates(KEYS + ["structure"])
    b = b.assign(pdb_resnum=E._norm_resnum(b["pdb_resnum"]))
    m = b.merge(f, on=KEYS + ["structure"], how="left")
    assert len(m) == len(b)
    return m


def megascale_train_table(split: str) -> pd.DataFrame:
    a = pd.read_parquet(PROC / "megascale.parquet")
    c = pd.read_csv(PROC / "megascale_censored.csv", low_memory=False)
    d = pd.concat([a, c[c.split == split]], ignore_index=True)
    d = d[d.split == split]
    d = FT.prepare_targets(with_feats(d, "megascale"), "megascale")
    return d.assign(dataset="megascale")


def fireprot_train_table() -> pd.DataFrame:
    d = E.load_benchmark("fireprot")
    return FT.prepare_targets(with_feats(d[d.split == "train"], "fireprot"), "fireprot").assign(dataset="fireprot")


def bench_table(name: str) -> pd.DataFrame:
    b = E.load_benchmark(name)
    if name == "megascale":
        b = b[b.split == "test"]
    if name == "fireprot":
        b = b[b.split == "test"]
    if name == "domainome":
        h = pd.read_csv(PROC / "homology_domainome.csv")[["protein_id", "homolog_in_megascale_train"]]
        b = b.merge(h, on="protein_id", how="left")
    b = b[b.structure.notna()] if name == "domainome" else b
    return with_feats(b.reset_index(drop=True), name)


def foldx(name: str) -> pd.DataFrame | None:
    p = E.ROOT / "results" / "baselines" / f"foldx_{name}.csv"
    return E._norm(pd.read_csv(p, dtype={"pdb_resnum": str}), KEYS) if p.exists() else None


# ----------------------------------------------------------------------------- models
def b0_lookup(train: pd.DataFrame) -> dict:
    pt = train[train.y.notna()]
    return {f"{w}{m}": float(v) for (w, m), v in pt.groupby(["wt", "mut"]).y.mean().items()}


def predict_all(models: dict, lookup: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["B0"] = (df.wt + df.mut).map(lookup).astype(float)
    for name, m in models.items():
        out[name] = FT.predict(m, df)
    return out


def base_cfg(sel: dict, feats, ref) -> FT.FitConfig:
    return FT.FitConfig(features=list(feats), ref=ref, loss=sel["loss"], censor=sel["censor"], within=sel["within"],
                        rank_weight=sel["rank_weight"], ref_l2=sel["ref_l2"], ref_dataset="megascale")


# ----------------------------------------------------------------------------- stage: select
def val_metrics(m: dict, val: pd.DataFrame) -> dict:
    v = val[val.y.notna()].assign(ddg=lambda d: d.y, pred=lambda d: FT.predict(m, d))
    out = {}
    for tag, sub in (("all", v), ("natural", v[v.domain_origin == "natural"])):
        r = E.evaluate(sub, "pred")
        out.update({f"val_{tag}_prot": r["spearman_prot_median"], f"val_{tag}_pooled": r["spearman"]})
    return out


def stage_select(train, val):
    rows = []
    default = dict(loss="gauss", censor=True, within="rows", rank_weight=0.0, ref_l2=1e-2)

    def run(label, feats, ref, opts, cv=True):
        cfg = base_cfg(opts, feats, ref)
        m = FT.fit(train, cfg)
        r = {"label": label, "features": "+".join(feats), "ref": ref, **opts, **val_metrics(m, val)}
        if cv:
            oof = FT.cross_validate(train, cfg, k=5)
            r.update({f"cv_{k}": v for k, v in FT.cv_metrics(oof).items()})
            nat = oof[oof.domain_origin == "natural"]
            r["cv_natural_prot"] = FT.cv_metrics(nat, by_fold=False)["spearman_prot_median"]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        rows.append(r)
        return r
    # 1) environment-potential variant (B3 form)
    env_res = {e: run(f"B3[{e}]", [e, "bur"], False, default) for e in ENV_OPTIONS}
    env = max(env_res, key=lambda e: env_res[e]["val_natural_prot"])
    feats = specs(env)["B3+"][0]
    # 2) objective options on B3+
    cands = {"gauss_censor": default, "gauss_nocensor": {**default, "censor": False},
             "gauss_censor_perdomain": {**default, "within": "proteins"}, "huber_censor": {**default, "loss": "huber"},
             "gauss_censor_rank0.3": {**default, "rank_weight": 0.3}}
    res = {k: run(f"B3+[{k}]", feats, True, o) for k, o in cands.items()}
    best = max(res, key=lambda k: res[k]["val_natural_prot"])
    sel = dict(cands[best])
    # 3) reference-energy shrinkage
    l2res = {l2: (res[best] if l2 == sel["ref_l2"] else run(f"B3+[{best},ref_l2={l2}]", feats, True, {**sel, "ref_l2": l2}))
             for l2 in (1e-3, 1e-2, 1e-1)}
    sel["ref_l2"] = max(l2res, key=lambda k: (l2res[k]["val_natural_prot"], k))
    sel["env"] = env
    pd.DataFrame(rows).to_csv(OUT / "selection.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print("SELECTED", sel)
    return sel


# ----------------------------------------------------------------------------- stage: fit
def stage_fit(train, fp_train, sel, n_boot):
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    prov = {"train": "megascale split==train (Dataset 3 + censored intervals)", "selection": "megascale val (natural, per-protein median Spearman)",
            "features": "data/features/simple_megascale.parquet (scripts/50_features_simple.py)", "selection_options": sel}
    models = {}
    for name, (feats, ref) in specs(sel["env"]).items():
        m = FT.fit(train, base_cfg(sel, feats, ref), n_boot=n_boot, n_jobs=32, name=name, provenance=prov)
        FT.save_model(m, OUT / "models" / f"{name}.json")
        models[name] = m
        print(name, {k: f"{v:.3f}±{m['se']['weights'][k]:.3f}" for k, v in m["weights"].items()}, "conv", m["converged"], flush=True)
    # joint Megascale + FireProt-train (per-dataset scale/offset, Megascale delta_ref nuisance, cluster weighting)
    feats, ref = specs(sel["env"])["B3+"]
    cfg = base_cfg(sel, feats, ref)
    cfg.ref_nuisance = ("megascale",)
    cfg.within = "rows" if sel["within"] == "rows" else sel["within"]
    m = FT.fit(pd.concat([train, fp_train], ignore_index=True), cfg, n_boot=n_boot, n_jobs=32, name="B3+j",
               provenance={**prov, "train": prov["train"] + " + fireprot split==train; dataset weight = n clusters"})
    FT.save_model(m, OUT / "models" / "B3+j.json")
    print("B3+j", {k: round(v, 3) for k, v in m["weights"].items()}, m["nuisance"]["scale"], m["nuisance"]["offset"])
    (OUT / "models" / "B0_lookup.json").write_text(json.dumps({"name": "B0", "pairs": b0_lookup(train),
                                                               "train": "megascale train Dataset-3 point rows"}, indent=1))


def load_models():
    d = OUT / "models"
    models = {p.stem: FT.load_model(p) for p in sorted(d.glob("*.json")) if p.stem != "B0_lookup"}
    return models, json.loads((d / "B0_lookup.json").read_text())["pairs"]


# ----------------------------------------------------------------------------- stage: eval
SUBSETS = {
    "megascale": {"all": "", "natural": "domain_origin==natural", "strict": "homolog_in_megascale_train==0",
                  "natural_strict": "domain_origin==natural,homolog_in_megascale_train==0"},
    "fireprot": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0",
                 "no_hetatm": "near_hetatm==false"},
    "s669": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
    "s461": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
    "ssym": {"direct": "direction==direct", "all": ""},
    "s2648": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
    "domainome": {"all": "", "plddt70": "plddt>=70", "strict": "homolog_in_megascale_train==0"},
    "abundance_dms": {"all": "", "plddt70": "plddt>=70"},
}
METRICS = ["n", "n_proteins", "spearman", "spearman_prot_median", "pearson", "rmse", "auroc_destab"]


def site_decomp(df, col, target):
    """Site-level Spearman (site means, sites with >= 5 variants) vs within-site ranking
    (mean per-site Spearman over sites with >= 10 variants; pooled Spearman of site-centred values)."""
    y = df.ddg if target == "ddg" else -df.score
    d = pd.DataFrame({"s": df.protein_id.astype(str) + "|" + df.chain.astype(str) + "|" + df.pdb_resnum.astype(str),
                      "y": y.to_numpy(), "p": df[col].to_numpy()}).dropna()
    g = d.groupby("s")
    n = g.y.transform("size")
    sm = g[["y", "p"]].mean()[g.size() >= 5]
    per = [spearmanr(x.p, x.y)[0] for _, x in d[n >= 10].groupby("s") if x.p.nunique() > 1 and x.y.nunique() > 1]
    c = d.assign(yc=d.y - g.y.transform("mean"), pc=d.p - g.p.transform("mean"))
    return {"site_spearman": spearmanr(sm.p, sm.y)[0] if len(sm) > 2 else np.nan, "n_sites": len(sm),
            "within_site_mean_spearman": float(np.nanmean(per)) if per else np.nan,
            "within_site_centred_spearman": spearmanr(c.pc, c.yc)[0]}


def _eval_task(bench, sub, model, df, n_boot):
    target = "ddg" if df.ddg.notna().any() else "score"
    r = E.evaluate(df, model, target=target)
    ci = E.bootstrap_ci(df, model, target=target, n_boot=n_boot)
    row = {"bench": bench, "subset": sub, "model": model, **{k: r[k] for k in METRICS}}
    for k in METRICS[2:]:
        if k in ci:
            row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    pair = None
    if "FoldX" in df and model != "FoldX" and df.FoldX.notna().sum() > 30:
        both = df[df.FoldX.notna() & df[model].notna()]
        pb = E.paired_bootstrap(both, model, "FoldX", target=target, n_boot=n_boot)
        a, b = E.evaluate(both, model, target=target), E.evaluate(both, "FoldX", target=target)
        pair = {"bench": bench, "subset": sub, "model": model, "n_both": len(both), "coverage_foldx": df.FoldX.notna().mean()}
        for k in ("spearman", "spearman_prot_median", "pearson"):
            if k in pb:
                pair.update({f"{k}_model": a[k], f"{k}_foldx": b[k], f"d_{k}": pb[k]["delta"], f"d_{k}_lo": pb[k]["lo"],
                             f"d_{k}_hi": pb[k]["hi"], f"d_{k}_p": pb[k]["p"]})
    site = {"bench": bench, "subset": sub, "model": model, **site_decomp(df, model, target)}
    return row, pair, site


def stage_eval(n_boot):
    models, lookup = load_models()
    names = ["B0"] + list(models)
    tasks, preds_cov = [], []
    for bench, subs in SUBSETS.items():
        b = bench_table(bench)
        P = predict_all(models, lookup, b)
        b = pd.concat([b, P], axis=1)
        fx = foldx(bench)
        cols = list(names)
        if fx is not None:
            b = b.merge(fx[KEYS + ["ddg_pred"]].rename(columns={"ddg_pred": "FoldX"}), on=KEYS, how="left")
            cols.append("FoldX")
        # leave-one-out site-mean oracle (uses targets of the other variants at the site; reference only)
        y = b.ddg if b.ddg.notna().any() else -b.score
        s = b.protein_id.astype(str) + "|" + b.chain.astype(str) + "|" + b.pdb_resnum.astype(str)
        g = y.groupby(s)
        n = g.transform("count")
        b["oracle_site_loo"] = np.where(n > 1, (g.transform("sum") - y) / (n - 1), np.nan)
        cols.append("oracle_site_loo")
        b[KEYS + ["structure"] + cols].to_parquet(OUT / f"preds_{bench}.parquet", index=False)
        preds_cov.append({"bench": bench, "rows": len(b), **{c: float(b[c].notna().mean()) for c in cols}})
        nb = 200 if bench == "domainome" else n_boot
        for sub, expr in subs.items():
            d = E.apply_subset(b, expr)
            for c in cols:
                dd = d[d[c].notna()]
                if len(dd) >= 20:
                    keep = [c] + (["FoldX"] if "FoldX" in d and c != "FoldX" else [])
                    tasks.append((bench, sub, c, dd[KEYS + ["ddg", "score", "cluster"] + keep] if "score" in dd
                                  else dd[KEYS + ["ddg", "cluster"] + keep], nb))
        if bench == "ssym":
            rows = [{"model": c, **E.antisymmetry(b, c)} for c in names]
            pd.DataFrame(rows).to_csv(OUT / "ssym_antisymmetry.csv", index=False)
    pd.DataFrame(preds_cov).to_csv(OUT / "prediction_coverage.csv", index=False)
    print(pd.DataFrame(preds_cov).to_string())
    res = Parallel(n_jobs=40, verbose=0)(delayed(_eval_task)(*t) for t in tasks)
    met = pd.DataFrame([r[0] for r in res])
    met.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame([r[1] for r in res if r[1]]).to_csv(OUT / "paired_vs_foldx.csv", index=False)
    pd.DataFrame([r[2] for r in res]).to_csv(OUT / "site_decomposition.csv", index=False)
    paired_af(models, lookup)
    summary(met)


def paired_af(models, lookup):
    p = pd.read_csv(PROC / "paired_af.csv", low_memory=False)
    p = p[p.usable.astype(bool) & ~p.af_is_mutant_state.fillna(False).astype(bool)].reset_index(drop=True)
    af = with_feats(p[KEYS + ["structure"]].copy(), "paired_af")
    xt = with_feats(p.assign(structure=p.xtal_structure, chain=p.xtal_chain, pdb_resnum=p.xtal_resnum)[KEYS + ["structure"]],
                    "paired_af_xtal")
    pa, px = predict_all(models, lookup, af), predict_all(models, lookup, xt)
    rows = []
    for ds in ["all"] + sorted(p.dataset.unique()):
        m = np.ones(len(p), bool) if ds == "all" else (p.dataset == ds).to_numpy()
        for c in list(pa.columns) + ["bur", "de_env", "occn"]:
            a = (af[c] if c in af and c not in pa else pa[c])[m].to_numpy(float)
            x = (xt[c] if c in xt and c not in px else px[c])[m].to_numpy(float)
            y = p.ddg[m].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(x)
            rows.append({"dataset": ds, "quantity": c, "n": int(ok.sum()),
                         "pearson_af_vs_xtal": np.corrcoef(a[ok], x[ok])[0, 1], "spearman_af_vs_xtal": spearmanr(a[ok], x[ok])[0],
                         "mean_abs_diff": float(np.mean(np.abs(a[ok] - x[ok]))),
                         "spearman_af_vs_ddg": spearmanr(a[ok], y[ok])[0], "spearman_xtal_vs_ddg": spearmanr(x[ok], y[ok])[0]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "paired_af_agreement.csv", index=False)
    print(t[t.dataset == "all"].round(3).to_string(index=False))


def summary(met):
    keym = {"megascale": "spearman_prot_median", "domainome": "spearman_prot_median", "abundance_dms": "spearman_prot_median"}
    rows = []
    for (bench, sub), g in met.groupby(["bench", "subset"], sort=False):
        k = keym.get(bench, "pearson")
        for _, r in g.iterrows():
            rows.append({"bench": bench, "subset": sub, "metric": k, "model": r.model,
                         "value": f"{r[k]:.3f} [{r.get(k + '_lo', np.nan):.2f},{r.get(k + '_hi', np.nan):.2f}]", "n": r.n})
    w = pd.DataFrame(rows).pivot_table(index=["bench", "subset", "metric"], columns="model", values="value", aggfunc="first", sort=False)
    w.to_csv(OUT / "summary_wide.csv")
    print(w.to_string())


# ----------------------------------------------------------------------------- stage: transfer (G6)
def stage_transfer(train, fp_train, sel, n_boot):
    for name in ("B3+", "B3r", "B2"):
        feats, ref = specs(sel["env"])[name]
        fits = {src: FT.fit(d, base_cfg(sel, feats, ref), n_boot=n_boot, n_jobs=32, name=f"{name}_{src}")
                for src, d in (("megascale", train), ("fireprot", fp_train))}
        for src, m in fits.items():
            FT.save_model(m, OUT / "transfer" / f"{name}_{src}.json")
        a, b = fits["megascale"], fits["fireprot"]
        rows = []
        for f in feats:
            wa, wb = a["weights"][f], b["weights"][f]
            sa, sb = a["se"]["weights"][f], b["se"]["weights"][f]
            ratio = wb / wa if wa != 0 else np.nan
            rows.append({"feature": f, "w_megascale": wa, "se_megascale": sa, "w_fireprot": wb, "se_fireprot": sb,
                         "ratio_fp_over_ms": ratio, "sign_flip": bool(np.sign(wa) != np.sign(wb) and wa != 0 and wb != 0),
                         "sign_flip_significant": bool(np.sign(wa) != np.sign(wb) and abs(wa) > 2 * sa and abs(wb) > 2 * sb),
                         "gt3x": bool(np.isfinite(ratio) and (abs(ratio) > 3 or abs(ratio) < 1 / 3)),
                         "z_diff": (wb - wa) / np.sqrt(sa ** 2 + sb ** 2) if sa + sb > 0 else np.nan})
        t = pd.DataFrame(rows)
        if ref:
            ra, rb = np.array(list(a["ref"].values())), np.array(list(b["ref"].values()))
            t.attrs["ref_corr"] = float(np.corrcoef(ra - ra.mean(), rb - rb.mean())[0, 1])
            extra = pd.DataFrame([{"feature": "ref_energy_pearson", "w_megascale": np.nan, "w_fireprot": t.attrs["ref_corr"]}])
            t = pd.concat([t, extra], ignore_index=True)
        t.to_csv(OUT / f"transfer_{name}.csv", index=False)
        print(f"== transfer {name}\n", t.round(3).to_string(index=False))


def stage_report():
    """Compact main table (markdown) from metrics.csv / site_decomposition.csv / models."""
    met = pd.read_csv(OUT / "metrics.csv")
    site = pd.read_csv(OUT / "site_decomposition.csv")
    cols = [("megascale", "natural", "spearman_prot_median", "MS nat ρ_prot"),
            ("megascale", "natural_strict", "spearman_prot_median", "MS nat-strict ρ_prot"),
            ("megascale", "all", "spearman", "MS all ρ"), ("megascale", "all", "pearson", "MS all r"),
            ("fireprot", "all", "pearson", "FP-test r"), ("fireprot", "strict", "pearson", "FP strict r"),
            ("s669", "strict", "pearson", "S669 strict r"), ("s461", "all", "pearson", "S461 r"),
            ("ssym", "direct", "pearson", "Ssym dir r"),
            ("domainome", "all", "spearman_prot_median", "Dom ρ_dom"), ("domainome", "plddt70", "spearman_prot_median", "Dom pLDDT≥70"),
            ("abundance_dms", "all", "spearman_prot_median", "Abund ρ_prot")]
    order = ["B0", "Bref", "B1", "B2", "B2occ", "B3", "B3r", "B3+", "B3+j", "FoldX", "oracle_site_loo"]
    nw = {"B0": "380 pair means", "FoldX": "-", "oracle_site_loo": "oracle"}
    for p in (OUT / "models").glob("B*.json"):
        if p.stem != "B0_lookup":
            m = FT.load_model(p)
            nw[p.stem] = f"{len(m['weights'])}+1{'+20 ref' if m['ref'] else ''}"
    rows = []
    for mod in order:
        r = {"model": mod, "params": nw.get(mod, "")}
        for b, s, k, lab in cols:
            x = met[(met.bench == b) & (met.subset == s) & (met.model == mod)]
            r[lab] = f"{x[k].iloc[0]:.3f}" if len(x) and np.isfinite(x[k].iloc[0]) else "–"
        for b, lab in (("megascale", "MS"), ("domainome", "Dom")):
            x = site[(site.bench == b) & (site.subset == "all") & (site.model == mod)]
            if mod != "oracle_site_loo" and len(x):
                r[f"{lab} site/within"] = f"{x.site_spearman.iloc[0]:.2f}/{x.within_site_mean_spearman.iloc[0]:.2f}"
        rows.append(r)
    t = pd.DataFrame(rows).fillna("–")
    md = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    md += "\n".join("| " + " | ".join(map(str, v)) + " |" for v in t.to_numpy())
    (OUT / "main_table.md").write_text(md + "\n")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select", "fit", "eval", "transfer", "report", "all"])
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer").mkdir(exist_ok=True)
    if a.stage == "report":
        return stage_report()
    need_train = a.stage in ("select", "fit", "transfer", "all")
    train = megascale_train_table("train") if need_train else None
    if a.stage in ("select", "all"):
        sel = stage_select(train, megascale_train_table("val"))
    else:
        sel = json.loads((OUT / "selection.json").read_text())
    fp = fireprot_train_table() if a.stage in ("fit", "transfer", "all") else None
    if a.stage in ("fit", "all"):
        stage_fit(train, fp, sel, a.n_boot)
    if a.stage in ("eval", "all"):
        stage_eval(1000)
    if a.stage in ("transfer", "all"):
        stage_transfer(train, fp, sel, a.n_boot)
    if a.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
