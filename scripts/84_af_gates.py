#!/usr/bin/env python
"""AF-mode Q4/Q5: evaluate every treatment, fit the AF-specific parameters on held-in data only, and give the
G5 verdicts.

  PYTHONPATH=. .venv/bin/python scripts/84_af_gates.py

Fitting discipline (as instructed): any AF parameter (band down-weight w, shrinkage logistic, burial-swap band,
cap preset) is chosen on abundance_dms + a random HALF of the Domainome domains (split by Pfam cluster, seed 0)
and reported on the other half. No AF parameter is fitted on Megascale-test or FireProt-test; the Megascale-test
column is only ever read as the "does it hurt" side of gate G5.

Outputs in results/af_mode/:
  treat_domainome.csv    per-treatment per-domain Spearman (fit / report half, by pLDDT band) + paired bootstrap
  treat_megascale.csv    the same treatments on Megascale-test (per-protein median Spearman, natural domains)
  shrink_fit.csv         fitted shrinkage parameters and their effect
  abstain.csv            flag-based abstention: coverage vs accuracy
  gates.csv              G5 verdicts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402
from physddg import afmode as A  # noqa: E402
from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

PREDS = ["B3+", "B3", "B4", "P1", "s1dF"]


def treatments(d: pd.DataFrame) -> dict:
    out = {}
    for c in d.columns:
        if "__" in c:
            p, t = c.split("__", 1)
            out.setdefault(p, []).append(t)
    return out


def table_domainome(d: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    tr = treatments(d)
    for half in ("fit", "report", "all"):
        sub = d if half == "all" else d[d.half == half]
        for p, ts in tr.items():
            if p not in PREDS:
                continue
            cols = [f"{p}__{t}" for t in ts if f"{p}__{t}" in sub]
            if f"{p}__off" not in cols:
                continue
            for band, s2 in [("all", sub)] + [(b, sub[sub.plddt_band == b]) for b in C.PLDDT_LABELS]:
                if len(s2) < 500:
                    continue
                base = C.per_protein_spearman(s2, f"{p}__off", "mscore")
                for c in cols:
                    s = C.per_protein_spearman(s2, c, "mscore")
                    common = base.dropna().index.intersection(s.dropna().index)
                    rows.append(dict(half=half, band=band, pred=p, treatment=c.split("__", 1)[1],
                                     n=int(s2[c].notna().sum()), n_dom=int(s.notna().sum()),
                                     median_rho=float(np.nanmedian(s)) if len(s) else np.nan,
                                     delta_vs_off=float(np.nanmedian(s[common]) - np.nanmedian(base[common]))
                                     if len(common) else np.nan,
                                     mean_delta_per_domain=float(np.nanmean(s[common] - base[common]))
                                     if len(common) else np.nan,
                                     coverage=float(s2[c].notna().mean())))
    t = pd.DataFrame(rows)
    # paired cluster bootstrap for the headline (report half, all bands)
    boots = []
    for p, ts in tr.items():
        if p not in PREDS:
            continue
        cols = [f"{p}__{x}" for x in ["off"] + [t_ for t_ in ts if t_ != "off"] if f"{p}__{x}" in d]
        sub = d[d.half == "report"]
        b = C.boot_median_rho(sub, cols, "mscore", cluster="cluster", n_boot=n_boot)
        b["treatment"] = [c.split("__", 1)[1] for c in b.pred]
        b["pred"] = p
        boots.append(b)
    bt = pd.concat(boots, ignore_index=True) if boots else pd.DataFrame()
    return t, bt


def megascale_treatments(n_boot: int = 200) -> pd.DataFrame:
    """Same treatments on Megascale-test (natural domains): the 'must not degrade' side of G5."""
    b = E.load_benchmark("megascale")
    b = b[b.split == "test"].reset_index(drop=True)
    rows = []
    mdl = C.models(("B3+", "B3"))
    p1 = FT.load_model(C.OUT / "models" / "P1.json")
    import importlib.util
    spec = importlib.util.spec_from_file_location("s83", ROOT / "scripts" / "83_af_treatments.py")
    s83 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s83)
    prod = s83.production_models()
    afdir = C.FEAT / "af"
    base = C._norm(b)
    out = {}
    for tag in ["off", "trim50", "trim70", "mavisp", "cap_plddt", "cap_soft"]:
        sf = C.FEAT / "simple_megascale.parquet" if tag == "off" else afdir / f"simple_megascale_{tag}.parquet"
        pf = C.FEAT / "phys_s1_megascale.parquet" if tag == "off" else afdir / f"phys_s1_megascale_{tag}.parquet"
        if sf.exists():
            t = C.attach(base[C.KEYS], C.load_feat(sf).drop(columns=["structure"]), on_structure=False)
            for name, m in mdl.items():
                out[f"{name}__{tag}"] = FT.predict(m, t)
        if pf.exists():
            t = C.attach(base[C.KEYS], C.load_feat(pf).drop(columns=["structure"]), on_structure=False)
            out[f"P1__{tag}"] = FT.predict(p1, t)
            out[f"s1dF__{tag}"] = t["s1_dF"].to_numpy()
            if "B4" in prod:   # the cap treatments do not change descriptors -> fall back to the untreated ones
                sfile = sf if sf.exists() else C.FEAT / "simple_megascale.parquet"
                sfeat = C.attach(base[C.KEYS], C.load_feat(sfile).drop(columns=["structure"]), on_structure=False)
                out[f"B4__{tag}"] = s83.predict_production(prod["B4"], sfeat, t, 1)
    for k, v in out.items():
        base[k] = v
    s2 = C.phys("megascale", 2)
    t2 = C.attach(base[C.KEYS], s2.drop(columns=["structure"]), on_structure=False)
    base["s1dF__s2"] = t2["s2_dF"].to_numpy()
    base["P1__s2"] = FT.predict(p1, t2.rename(columns={c: c.replace("s2_", "s1_") for c in t2.columns}))
    if "B5" in prod:
        sfeat = C.attach(base[C.KEYS], C.simple("megascale").drop(columns=["structure"]), on_structure=False)
        base["B4__s2"] = s83.predict_production(prod["B5"], sfeat, t2, 2)
    base.to_parquet(C.FEAT / "af" / "treated_megascale.parquet", index=False)
    natural = base[base.domain_origin == "natural"] if "domain_origin" in base else base
    for name, sub in (("all", base), ("natural", natural)):
        for col in [c for c in base.columns if "__" in c]:
            p, t = col.split("__", 1)
            s = C.per_protein_spearman(sub, col, "ddg")
            rows.append(dict(subset=name, pred=p, treatment=t, n=int(sub[col].notna().sum()),
                             n_prot=int(s.notna().sum()), median_rho=float(np.nanmedian(s)),
                             pooled_rho=float(E.evaluate(sub.assign(pred=sub[col]), "pred", "ddg")["spearman"])))
    return pd.DataFrame(rows)


def fit_shrinkage(d: pd.DataFrame, ab: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    fit = d[d.half == "fit"]
    rep = d[d.half == "report"]
    for p in PREDS:
        col = f"{p}__off"
        if col not in d:
            continue
        pool_pred = np.r_[fit[col].to_numpy(float), ab[col].to_numpy(float)] if ab is not None and col in ab \
            else fit[col].to_numpy(float)
        pool_pl = np.r_[fit.plddt_site.to_numpy(float), ab.plddt_site.to_numpy(float)] \
            if ab is not None and col in ab else fit.plddt_site.to_numpy(float)
        pool_y = np.r_[fit.mscore.to_numpy(float), ab.mscore.to_numpy(float)] if ab is not None and col in ab \
            else fit.mscore.to_numpy(float)
        pool_g = np.r_[fit.protein_id.to_numpy(str), ab.protein_id.to_numpy(str)] \
            if ab is not None and col in ab else fit.protein_id.to_numpy(str)
        cmean = float(np.nanmean(pool_pred))
        best = A.fit_shrink(pool_pred, pool_pl, pool_y, pool_g, c_grid=(0.0, cmean))
        for half, sub in (("fit", fit), ("report", rep)):
            base = C.per_protein_spearman(sub, col, "mscore")
            q = sub.assign(sh=A.shrink(sub[col], sub.plddt_site, best["x0"], best["s"], best["c"]))
            s = C.per_protein_spearman(q, "sh", "mscore")
            common = base.dropna().index.intersection(s.dropna().index)
            rows.append(dict(pred=p, half=half, x0=best["x0"], s=best["s"], c=round(best["c"], 3),
                             fit_score=best["score"], median_rho=float(np.nanmedian(s)),
                             base_rho=float(np.nanmedian(base)),
                             delta=float(np.nanmedian(s[common]) - np.nanmedian(base[common]))))
    return pd.DataFrame(rows)


def abstain_table(d: pd.DataFrame) -> pd.DataFrame:
    fl = pd.read_parquet(C.FEAT / "af" / "domainome_flags.parquet")
    d = d.merge(fl, on=["structure", "chain", "pdb_resnum"], how="left", suffixes=("", "_fl"))
    rules = {
        "none": np.zeros(len(d), bool),
        "plddt<70": d.plddt_site.fillna(100) < 70,
        "plddt<50": d.plddt_site.fillna(100) < 50,
        "terminal": d.terminal.fillna(False).astype(bool),
        "exposed_hydrophobic_patch": d.exposed_hydrophobic_patch.fillna(False).astype(bool),
        "metal_site": d.metal_site.fillna(False).astype(bool),
        "plddt<70 | terminal": (d.plddt_site.fillna(100) < 70) | d.terminal.fillna(False).astype(bool),
        "plddt<70 | terminal | patch": (d.plddt_site.fillna(100) < 70) | d.terminal.fillna(False).astype(bool)
        | d.exposed_hydrophobic_patch.fillna(False).astype(bool),
    }
    rows = []
    for name, drop in rules.items():
        sub = d[~np.asarray(drop)]
        for p in PREDS:
            col = f"{p}__off"
            if col not in d:
                continue
            s = C.per_protein_spearman(sub, col, "mscore")
            rows.append(dict(rule=name, pred=p, kept_frac=float((~np.asarray(drop)).mean()),
                             n=int(sub[col].notna().sum()), n_dom=int(s.notna().sum()),
                             median_rho=float(np.nanmedian(s))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--parts", nargs="+", default=["tables", "shrink", "abstain", "megascale"],
                    choices=["tables", "shrink", "abstain", "megascale"])
    a = ap.parse_args()
    out = C.ensure_out()
    d = pd.read_parquet(C.FEAT / "af" / "treated_domainome.parquet")
    d["half"] = C.split_halves(d)
    d["plddt_band"] = C.plddt_band(d.plddt_site.fillna(d.plddt))
    print(f"{len(d)} rows, {d.protein_id.nunique()} domains; fit half {d.half.eq('fit').mean():.2f}")
    ab = None
    p_ab = C.FEAT / "af" / "treated_abundance_dms.parquet"
    if p_ab.exists():
        ab = pd.read_parquet(p_ab)

    if "tables" in a.parts:
        t, bt = table_domainome(d, a.n_boot)
        t.to_csv(out / "treat_domainome.csv", index=False)
        bt.to_csv(out / "treat_domainome_boot.csv", index=False)
        print(t[t.half == "report"].pivot_table(index="treatment", columns=["pred", "band"],
                                                values="median_rho").round(3).to_string())
    if "shrink" in a.parts:
        sh = fit_shrinkage(d, ab)
        sh.to_csv(out / "shrink_fit.csv", index=False)
        print(sh.round(3).to_string(index=False))
    if "abstain" in a.parts:
        ab_t = abstain_table(d)
        ab_t.to_csv(out / "abstain.csv", index=False)
        print(ab_t.round(3).to_string(index=False))
    if "megascale" in a.parts:
        ms = megascale_treatments()
        ms.to_csv(out / "treat_megascale.csv", index=False)
        print(ms.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
