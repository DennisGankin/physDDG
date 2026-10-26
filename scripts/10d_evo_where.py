#!/usr/bin/env python
"""Evolution-aware extension, step 4: where do the profile terms help? (Megascale-test predictions of scripts/10c;
no fitting, nothing feeds back into selection.)

Per protein: Spearman of site means (sites with >= 5 variants; 'site level') and Spearman of site-centred values
('within site'), then the median over proteins; per stratum (natural / designed, buried RSA < 0.25 / exposed,
site N_eff bins). Also the per-protein change B5evo - B5 for every test protein.
Outputs results/evo_extension/where_site_within.csv, per_protein_megascale.csv.
  PYTHONPATH=. .venv/bin/python scripts/10d_evo_where.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "results" / "evo_extension"
MODELS = ["B5", "B5evo", "B3+", "B3+evo", "Evo-only", "FoldX"]


def per_protein(g, m):
    s = g.pdb_resnum.astype(str)
    sm = g.groupby(s)[["ddg", m]].mean()[g.groupby(s).size() >= 5]
    yc = g.ddg - g.groupby(s).ddg.transform("mean")
    pc = g[m] - g.groupby(s)[m].transform("mean")
    return dict(all=spearmanr(g[m], g.ddg)[0], site=spearmanr(sm[m], sm.ddg)[0] if len(sm) > 4 else np.nan,
                within=spearmanr(pc, yc)[0])


def main():
    d = pd.read_parquet(OUT / "preds_megascale_test.parquet").dropna(subset=["ddg"])
    rsa = d.rsa.to_numpy(float)
    n = d.evo_neff_site.to_numpy(float)
    strata = {"all": np.ones(len(d), bool), "natural": (d.domain_origin == "natural").to_numpy(),
              "designed": (d.domain_origin == "designed").to_numpy(), "buried(rsa<0.25)": rsa < 0.25,
              "exposed(rsa>=0.25)": rsa >= 0.25, "natural & buried": (d.domain_origin == "natural").to_numpy() & (rsa < 0.25),
              "natural & exposed": (d.domain_origin == "natural").to_numpy() & (rsa >= 0.25),
              "site neff<10": n < 10, "site neff 10-100": (n >= 10) & (n < 100), "site neff>=100": n >= 100}
    rows = []
    for sn, mk in strata.items():
        dd = d[mk]
        for m in MODELS:
            if m not in dd:
                continue
            pp = pd.DataFrame([per_protein(g, m) for _, g in dd.groupby("protein_id") if len(g) >= 50])
            if pp.empty:
                continue
            rows.append({"stratum": sn, "model": m, "n_rows": len(dd), "n_prot": len(pp),
                         "rho_prot_median": pp["all"].median(), "site_rho_prot_median": pp.site.median(),
                         "within_rho_prot_median": pp.within.median()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "where_site_within.csv", index=False)
    w = t.pivot_table(index="stratum", columns="model", values=["site_rho_prot_median", "within_rho_prot_median"], sort=False)
    print(w.round(3).to_string())
    pr = []
    for p, g in d.groupby("protein_id"):
        r = {"protein_id": p, "origin": g.domain_origin.iloc[0], "neff": g.evo_neff.iloc[0],
             "homolog_in_megascale_train": float(g.homolog_in_megascale_train.max()), "n": len(g)}
        for m in MODELS:
            if m in g:
                q = per_protein(g, m)
                r.update({f"{m}": q["all"], f"{m}_site": q["site"], f"{m}_within": q["within"]})
        pr.append(r)
    pr = pd.DataFrame(pr)
    pr["d_B5evo_B5"] = pr.B5evo - pr.B5
    pr.to_csv(OUT / "per_protein_megascale.csv", index=False)
    for tag, g in (("all", pr), ("natural", pr[pr.origin == "natural"]), ("designed", pr[pr.origin == "designed"]),
                   ("natural strict", pr[(pr.origin == "natural") & (pr.homolog_in_megascale_train == 0)]),
                   ("natural with train homologue", pr[(pr.origin == "natural") & (pr.homolog_in_megascale_train > 0)])):
        print(f"{tag:30s} n={len(g):2d}  mean d={g.d_B5evo_B5.mean():+.4f}  median d={g.d_B5evo_B5.median():+.4f}  "
              f"up {(g.d_B5evo_B5 > 0).sum()}/{len(g)}")


if __name__ == "__main__":
    main()
