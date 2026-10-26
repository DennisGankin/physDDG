#!/usr/bin/env python
"""AF-mode: pLDDT shrinkage with a burial-matched control, and the G5 verdict table.

  PYTHONPATH=. .venv/bin/python scripts/86_af_shrink_gates.py

The Domainome analysis (scripts/82) showed that site pLDDT is strongly confounded with burial, so a pLDDT
attenuation fitted on Domainome can be a burial recalibration in disguise. This script therefore fits the same
two-parameter logistic twice -- once in pLDDT, once in 100*(1-RSA) -- on the same held-in data
(abundance_dms + the 'fit' half of the Domainome clusters) and compares the gains on the held-out half.
The fitted pLDDT shrinkage is then applied to Megascale-test (never fitted on) for the second half of gate G5.

Outputs: results/af_mode/shrink_control.csv, results/af_mode/gates.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.af_common as C  # noqa: E402
from physddg import afmode as A  # noqa: E402


PREDS = ["B3+", "B3", "B4", "P1", "s1dF"]


def common_row_deltas(d: pd.DataFrame, out: Path, y: str = "mscore", group: str = "protein_id",
                      half: str | None = "report", name: str | None = None) -> pd.DataFrame:
    """Treatment deltas on IDENTICAL rows. Trimming deletes some measured sites, so a plain comparison mixes the
    treatment with an implicit abstention; here every treatment of a predictor is scored on the rows where all of
    them are defined (and the abstention effect is reported separately as `rows_dropped`)."""
    sub = d[d.half == half] if (half and "half" in d) else d
    rows = []
    preds = {}
    for c in sub.columns:
        if "__" in c:
            p_, t_ = c.split("__", 1)
            preds.setdefault(p_, []).append(t_)
    for p_, ts in preds.items():
        cols = [f"{p_}__{t}" for t in ts]
        ok = sub[cols].notna().all(axis=1)
        s2 = sub[ok]
        base = C.per_protein_spearman(s2, f"{p_}__off", y, group)
        for c in cols:
            sc = C.per_protein_spearman(s2, c, y, group)
            common = base.dropna().index.intersection(sc.dropna().index)
            rows.append(dict(pred=p_, treatment=c.split("__", 1)[1], n=int(ok.sum()),
                             rows_dropped=float(1 - sub[c].notna().mean()),
                             median_rho=float(np.nanmedian(sc)),
                             delta_common=float(np.nanmedian(sc[common]) - np.nanmedian(base[common]))))
    t = pd.DataFrame(rows)
    t.to_csv(out / (name or f"treat_common_rows_{half}.csv"), index=False)
    return t


def megascale_common_rows(out: Path) -> pd.DataFrame:
    f = C.FEAT / "af" / "treated_megascale.parquet"
    if not f.exists():
        return pd.DataFrame()
    m = pd.read_parquet(f)
    m = m[m.domain_origin == "natural"] if "domain_origin" in m else m
    m = m.assign(mscore=m.ddg, half="report")
    return common_row_deltas(m, out, y="mscore", half=None, name="treat_megascale_common_rows.csv")


def main():
    out = C.ensure_out()
    d = pd.read_parquet(C.FEAT / "af" / "treated_domainome.parquet")
    d["half"] = C.split_halves(d)
    ab = pd.read_parquet(C.FEAT / "af" / "treated_abundance_dms.parquet")
    pms = C.FEAT / "af" / "treated_megascale.parquet"
    ms = None
    if pms.exists():
        ms = pd.read_parquet(pms)
        ms = C.attach(ms, C.phys("megascale", 1)[C.KEYS + ["ctx_bfactor", "ctx_rsa"]], on_structure=False)
    fit, rep = d[d.half == "fit"], d[d.half == "report"]
    rows, gates = [], []
    for p in PREDS:
        col = f"{p}__off"
        if col not in d:
            continue
        for var, name in (("plddt_site", "pLDDT"), ("bur100", "burial")):
            def x(t):
                return t.plddt_site.to_numpy(float) if var == "plddt_site" \
                    else 100.0 * (1.0 - t.rsa.to_numpy(float))
            pool = dict(pred=np.r_[fit[col].to_numpy(float), ab[col].to_numpy(float)],
                        xv=np.r_[x(fit), x(ab)], y=np.r_[fit.mscore.to_numpy(float), ab.mscore.to_numpy(float)],
                        g=np.r_[fit.protein_id.to_numpy(str), ab.protein_id.to_numpy(str)])
            cm = float(np.nanmean(pool["pred"]))
            best = A.fit_shrink(pool["pred"], pool["xv"], pool["y"], pool["g"], c_grid=(0.0, cm))
            r = dict(pred=p, variable=name, x0=best["x0"], s=best["s"], c=round(best["c"], 3))
            for half, sub in (("fit", fit), ("report", rep)):
                base = C.per_protein_spearman(sub, col, "mscore")
                q = sub.assign(sh=A.shrink(sub[col], x(sub), best["x0"], best["s"], best["c"]))
                s_ = C.per_protein_spearman(q, "sh", "mscore")
                common = base.dropna().index.intersection(s_.dropna().index)
                r[f"rho_{half}"] = float(np.nanmedian(s_))
                r[f"base_{half}"] = float(np.nanmedian(base))
                r[f"delta_{half}"] = float(np.nanmedian(s_[common]) - np.nanmedian(base[common]))
            # effect on Megascale-test (never fitted on)
            if ms is not None and col in ms and var == "plddt_site":
                m = ms[ms.domain_origin == "natural"] if "domain_origin" in ms else ms
                xv = m.ctx_bfactor.to_numpy(float) if "ctx_bfactor" in m else m.plddt.to_numpy(float)
                b0 = C.per_protein_spearman(m, col, "ddg")
                mm = m.assign(sh=A.shrink(m[col], xv, best["x0"], best["s"], best["c"]))
                b1 = C.per_protein_spearman(mm, "sh", "ddg")
                common = b0.dropna().index.intersection(b1.dropna().index)
                r["ms_base"] = float(np.nanmedian(b0[common]))
                r["ms_shrunk"] = float(np.nanmedian(b1[common]))
                r["ms_delta"] = r["ms_shrunk"] - r["ms_base"]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "shrink_control.csv", index=False)
    print(t.round(3).to_string(index=False))
    cr = common_row_deltas(d, out)
    print("Domainome, identical rows (report half):")
    print(cr.round(3).to_string(index=False))
    mc = megascale_common_rows(out)
    if len(mc):
        print("Megascale-test natural, identical rows:")
        print(mc.round(3).to_string(index=False))
    gates_table(t, out)


def gates_table(shr: pd.DataFrame, out: Path):
    """G5 verdicts: Domainome (report half) delta >= +0.02 AND Megascale-test delta >= -0.005."""
    dom = pd.read_csv(out / "treat_common_rows_report.csv").rename(columns={"delta_common": "delta_vs_first"})
    dom["delta_lo"] = dom["delta_hi"] = np.nan
    boot = out / "treat_domainome_boot.csv"                      # CIs from scripts/84, when available
    if boot.exists():
        b = pd.read_csv(boot)
        if "pred" in b:
            dom = dom.merge(b[["pred", "treatment", "delta_lo", "delta_hi"]], on=["pred", "treatment"],
                            how="left", suffixes=("_drop", ""))
            dom = dom.drop(columns=[c for c in dom.columns if c.endswith("_drop")])
    ms = pd.read_csv(out / "treat_megascale.csv") if (out / "treat_megascale.csv").exists() else None
    rows = []
    for _, r in dom.iterrows():
        if r.treatment == "off" or not np.isfinite(r.get("delta_vs_off", r.get("delta_vs_first", np.nan))):
            continue
        dd = r.get("delta_vs_first", np.nan)
        msd = np.nan
        if ms is not None and isinstance(r.get("pred"), str):
            m = ms[(ms.subset == "natural") & (ms.pred == r.pred)]
            a = m[m.treatment == r.treatment].median_rho
            b = m[m.treatment == "off"].median_rho
            if len(a) and len(b):
                msd = float(a.iloc[0] - b.iloc[0])
        rows.append(dict(pred=r.get("pred"), treatment=r.treatment, dom_delta=dd, dom_lo=r.delta_lo,
                         dom_hi=r.delta_hi, ms_delta=msd,
                         verdict="ship" if (dd >= 0.02 and (not np.isfinite(msd) or msd >= -0.005))
                         else ("neutral" if dd > -0.005 else "harmful")))
    for _, r in shr.iterrows():
        rows.append(dict(pred=r.pred, treatment=f"shrink({r.variable})", dom_delta=r.delta_report,
                         dom_lo=np.nan, dom_hi=np.nan, ms_delta=r.get("ms_delta", np.nan),
                         verdict="ship" if (r.delta_report >= 0.02 and
                                            (not np.isfinite(r.get("ms_delta", np.nan)) or
                                             r.get("ms_delta", 0) >= -0.005)) else "below gate"))
    g = pd.DataFrame(rows)
    g.to_csv(out / "gates.csv", index=False)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
