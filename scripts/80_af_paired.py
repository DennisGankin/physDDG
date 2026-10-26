#!/usr/bin/env python
"""AF-mode Q1 + gate G5: how much do predictions move between the crystal and the AlphaFold model of the
same variant, and does accuracy against experiment drop?

  PYTHONPATH=. .venv/bin/python scripts/80_af_paired.py

Rows: data/processed/paired_af.csv, usable, AF model in the WT state (152 Ssym reverse rows where the AF model is
the mutation's end state are dropped). Quantities: fitted simple-descriptor models (B3+, B3, B1), the raw
descriptors they are built from, unit-weight stage-1/stage-2 packing free energies and the stage-1 per-term
energies. Outputs in results/af_mode/:
  paired_agreement.csv        AF vs xtal agreement per quantity (pooled + per-protein), overall / per dataset
  paired_agreement_bands.csv  the same by site pLDDT band and by local backbone RMSD band
  paired_accuracy.csv         accuracy against experiment, AF vs xtal, identical rows, paired protein bootstrap
  paired_delta_drivers.csv    |AF - xtal| prediction difference vs local RMSD / pLDDT / burial
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402
from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

S1_TERMS = ["s1_dF", "s1_dE_lj_atr", "s1_dE_lj_rep", "s1_dE_solv_apolar", "s1_dE_solv_polar", "s1_dE_hb_sc_bb",
            "s1_dE_hb_sc_sc", "s1_dE_elec_dh", "s1_dTS"]
DESCR = ["bur", "occn", "rsa", "cn", "de_env_bb", "de_rama"]
QUANT = list(C.SIMPLE_MODELS) + ["P1", "P1_s2", "s1_dF", "s2_dF"] + S1_TERMS[1:] + DESCR
RMSD_EDGES = [0, 0.3, 0.6, 1.0, 100]
RMSD_LABELS = ["<0.3", "0.3-0.6", "0.6-1.0", ">1.0"]


def build() -> pd.DataFrame:
    p = pd.read_csv(C.PROC / "paired_af.csv", low_memory=False)
    p = p[p.usable.astype(bool) & ~p.af_is_mutant_state.fillna(False).astype(bool)].reset_index(drop=True)
    p["pdb_resnum"] = C.norm_resnum(p.pdb_resnum)
    p["xtal_resnum"] = C.norm_resnum(p.xtal_resnum)
    af = p[C.KEYS + ["structure"]].copy()
    xt = p.assign(structure=p.xtal_structure, chain=p.xtal_chain, pdb_resnum=p.xtal_resnum)[C.KEYS + ["structure"]]
    for side, rows, tag in (("af", af, "paired_af"), ("xtal", xt, "paired_af_xtal")):
        f = C.simple(tag).merge(C.phys(tag, 1), on=C.KEYS + ["structure"], how="outer")
        f2 = C.phys(tag, 2)
        f2 = f2[C.KEYS + ["structure"] + [c for c in f2.columns if c.startswith("s2_")]]
        f = f.merge(f2, on=C.KEYS + ["structure"], how="outer")
        f = C.predict_simple(f)
        p1 = FT.load_model(C.OUT / "models" / "P1.json")
        f["P1"] = FT.predict(p1, f)
        g2 = f[[c for c in f.columns if c.startswith("s2_")]].rename(
            columns={c: c.replace("s2_", "s1_") for c in f.columns if c.startswith("s2_")})
        keep = [c for c in f.columns if not c.startswith(("s1_", "s2_"))]
        g2 = pd.concat([f[keep], g2], axis=1)
        f["P1_s2"] = FT.predict(p1, g2)
        f = f[[c for c in f.columns if c in QUANT + C.KEYS + ["structure"] + S1_TERMS]]
        f = f.rename(columns={c: f"{side}_{c}" for c in f.columns if c not in C.KEYS + ["structure"]})
        merged = C.attach(rows, f)
        for c in merged.columns:
            if c.startswith(side + "_"):
                p[c] = merged[c].to_numpy()
    p["plddt_band"] = C.plddt_band(p.plddt)
    p["rmsd_band"] = pd.cut(p.local_rmsd_10A, bins=RMSD_EDGES, labels=RMSD_LABELS, right=False)
    return p


def main():
    out = C.ensure_out()
    p = build()
    print(f"{len(p)} paired rows, {p.protein_id.nunique()} entities")

    # ---- agreement, overall / per dataset / per band
    rows = []
    for key, sub in [("all", p)] + [(f"ds={d}", p[p.dataset == d]) for d in sorted(p.dataset.unique())] \
            + [(f"plddt {b}", p[p.plddt_band == b]) for b in C.PLDDT_LABELS] \
            + [(f"rmsd {b}", p[p.rmsd_band == b]) for b in RMSD_LABELS]:
        for q in QUANT:
            a, x = sub.get(f"af_{q}"), sub.get(f"xtal_{q}")
            if a is None or x is None:
                continue
            a, x = a.to_numpy(float), x.to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(x)
            if ok.sum() < 20 or np.ptp(a[ok]) == 0 or np.ptp(x[ok]) == 0:
                continue
            per = [pearsonr(g.a, g.x)[0] for _, g in
                   pd.DataFrame({"a": a, "x": x, "pid": sub.protein_id.to_numpy()}).dropna().groupby("pid")
                   if len(g) >= 10 and g.a.nunique() > 1 and g.x.nunique() > 1]
            rows.append(dict(subset=key, quantity=q, n=int(ok.sum()), r=pearsonr(a[ok], x[ok])[0],
                             rho=spearmanr(a[ok], x[ok])[0], mean_abs_diff=float(np.mean(np.abs(a[ok] - x[ok]))),
                             sd_diff=float(np.std(a[ok] - x[ok])), sd_xtal=float(np.std(x[ok])),
                             n_prot=len(per), r_prot_median=float(np.median(per)) if per else np.nan,
                             frac_prot_ge_09=float(np.mean(np.array(per) >= 0.9)) if per else np.nan))
    ag = pd.DataFrame(rows)
    ag[ag.subset.isin(["all", "ds=fireprot", "ds=s669", "ds=ssym"])].to_csv(out / "paired_agreement.csv", index=False)
    ag[~ag.subset.isin(["all", "ds=fireprot", "ds=s669", "ds=ssym"])].to_csv(out / "paired_agreement_bands.csv",
                                                                            index=False)
    print(ag[ag.subset == "all"].round(3).to_string(index=False))

    # ---- accuracy against experiment on identical rows (paired bootstrap over proteins)
    acc = []
    for key, sub in [("all", p)] + [(f"ds={d}", p[p.dataset == d]) for d in sorted(p.dataset.unique())] \
            + [(f"plddt {b}", p[p.plddt_band == b]) for b in C.PLDDT_LABELS] \
            + [(f"rmsd {b}", p[p.rmsd_band == b]) for b in RMSD_LABELS]:
        for q in list(C.SIMPLE_MODELS) + ["P1", "P1_s2", "s1_dF", "s2_dF"]:
            d = sub[[f"af_{q}", f"xtal_{q}", "ddg", "protein_id", "dataset"]].dropna()
            if len(d) < 30:
                continue
            d = d.rename(columns={f"af_{q}": "af", f"xtal_{q}": "xtal"})
            pb = E.paired_bootstrap(d.assign(cluster=np.nan), "af", "xtal", target="ddg", n_boot=400,
                                    cluster_col="cluster", min_n=10)
            r = dict(subset=key, quantity=q, n=len(d), n_prot=d.protein_id.nunique())
            for m in ("spearman", "pearson", "spearman_prot_median"):
                if m in pb:
                    r[f"{m}_af"] = E.evaluate(d, "af", "ddg")[m]
                    r[f"{m}_xtal"] = E.evaluate(d, "xtal", "ddg")[m]
                    r[f"{m}_delta"] = pb[m]["delta"]
                    r[f"{m}_lo"], r[f"{m}_hi"] = pb[m]["lo"], pb[m]["hi"]
            acc.append(r)
    ac = pd.DataFrame(acc)
    ac.to_csv(out / "paired_accuracy.csv", index=False)
    print(ac[ac.subset == "all"].round(3).to_string(index=False))

    # ---- what drives |AF - xtal|
    drv = []
    for q in list(C.SIMPLE_MODELS) + ["P1", "P1_s2", "s1_dF", "s2_dF", "bur", "s1_dE_lj_rep", "s1_dE_lj_atr"]:
        a, x = p.get(f"af_{q}"), p.get(f"xtal_{q}")
        if a is None or x is None:
            continue
        d = pd.DataFrame({"ad": (a - x).abs(), "rmsd": p.local_rmsd_10A, "plddt": p.plddt,
                          "ca_dev": p.ca_dev_site, "bur": p.xtal_bur, "band": p.rmsd_band}).dropna(subset=["ad"])
        row = dict(quantity=q, n=len(d), mean_abs_diff=float(d.ad.mean()))
        for c in ("rmsd", "plddt", "ca_dev", "bur"):
            m = d[[c, "ad"]].dropna()
            row[f"rho_{c}"] = spearmanr(m[c], m.ad)[0] if len(m) > 20 else np.nan
        for b in RMSD_LABELS:
            row[f"mad_rmsd_{b}"] = float(d.ad[d.band == b].mean()) if (d.band == b).any() else np.nan
        drv.append(row)
    pd.DataFrame(drv).to_csv(out / "paired_delta_drivers.csv", index=False)
    print(pd.DataFrame(drv).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
