#!/usr/bin/env python
"""ENM site descriptors (physddg/enm.py) for every structure of every benchmark, cached at
data/features/enm_sites_<name>.parquet, plus descriptor diagnostics by structure source and their correlation with
the static burial/packing descriptors (features_extra site table).

  PYTHONPATH=. .venv/bin/python scripts/120_enm_sites.py [--refresh]
Outputs: results/enm_flexibility/descriptor_by_source.csv, descriptor_corr.csv, timing.csv
"""
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import enm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "enm_flexibility"
NAMES = ["megascale", "fireprot", "s669", "s461", "ssym", "domainome", "abundance_dms", "paired_af", "paired_af_xtal"]
RAW = ["gnm_msf", "gnm_msf_slow", "anm_msf", "cen_msf", "stiff_local", "stiff_all", "prs_eff", "prs_sens", "dfi",
       "gnm_hinge", "gnm_msf_plddt"]
STATIC = ["cn", "ncb10", "ncb14", "natom8", "rsa", "hse_up", "scsc_contacts", "void_frac", "plddt"]


def source(name, st, xs):
    if name == "megascale":
        return "megascale_af2" if xs.get(st, 0) > 0 else "megascale_design(B=0)"
    if name in ("domainome", "abundance_dms", "paired_af"):
        return f"{name}_af"
    return f"{name}_xtal"


def main(refresh=False):
    OUT.mkdir(parents=True, exist_ok=True)
    timing, by_src, corr = [], [], []
    for name in NAMES:
        f = FEAT / f"extra_sites_{name}.parquet"
        if not f.exists():
            continue
        xs = pd.read_parquet(f)
        structs = sorted(xs.structure.unique())
        t = time.time()
        e = enm.enm_site_table(structs, name, n_jobs=48, refresh=refresh)
        dt = time.time() - t
        timing.append({"bench": name, "structures": len(structs), "residues": len(e), "seconds": dt,
                       "failed": len(structs) - e.structure.nunique()})
        print(timing[-1], flush=True)
        mp = xs.groupby("structure").mean_plddt.first().to_dict()
        e["source"] = [source(name, s, mp) for s in e.structure]
        for src, g in e.groupby("source"):
            r = {"bench": name, "source": src, "n_struct": g.structure.nunique(), "n_res": len(g),
                 "median_len": float(g.groupby("structure").n_res.first().median())}
            for c in RAW:
                v = np.log(g[c].clip(lower=1e-12)) if c != "gnm_hinge" else g[c]
                r[f"{c}_log_median"], r[f"{c}_log_iqr"] = float(v.median()), float(v.quantile(.75) - v.quantile(.25))
            by_src.append(r)
        m = e.merge(xs[["structure", "chain", "pdb_resnum"] + STATIC].assign(pdb_resnum=lambda d: d.pdb_resnum.astype(str)),
                    on=["structure", "chain", "pdb_resnum"], how="left")
        for src, g in m.groupby("source"):
            for c in [x + "_z" for x in RAW if x != "gnm_msf_plddt"] + ["gnm_msf_plddt_z"]:
                if g[c].notna().sum() < 100:
                    continue
                row = {"bench": name, "source": src, "descriptor": c}
                for s in STATIC + ["gnm_msf_z", "degree"]:
                    ok = g[c].notna() & g[s].notna()
                    row[s] = float(g.loc[ok, [c, s]].rank().corr().iloc[0, 1]) if ok.sum() > 100 else np.nan
                corr.append(row)
    pd.DataFrame(timing).to_csv(OUT / "timing.csv", index=False)
    pd.DataFrame(by_src).to_csv(OUT / "descriptor_by_source.csv", index=False)
    c = pd.DataFrame(corr)
    c.to_csv(OUT / "descriptor_corr.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(by_src)[["bench", "source", "n_struct", "median_len", "gnm_msf_log_median", "anm_msf_log_median",
                                     "stiff_local_log_median", "cen_msf_log_median"]].round(2).to_string(index=False))
        print(c[c.bench.isin(["megascale", "fireprot", "domainome"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main("--refresh" in sys.argv)
