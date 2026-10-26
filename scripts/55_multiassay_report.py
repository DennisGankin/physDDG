#!/usr/bin/env python
"""Multi-assay comparison (critique #2 I1 / A1 / A2) computed from the saved per-variant predictions.

Adds what the eval stage only produces for its `mains` list: paired-bootstrap deltas, per-protein win/loss
against FoldX, and mean next to median per-protein rho, for EVERY model including the joint fits and the
Delta-ref diagnostics. Strict subsets are per model (critique #2 3.4): a model fitted with FireProt-trainval
must also exclude FireProt-trainval homologues.

  PYTHONPATH=. .venv/bin/python scripts/55_multiassay_report.py
Writes results/ladder_physics/multiassay_{summary,deltas}.csv.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from physddg import eval as E  # noqa: E402

OUT = ROOT / "results" / "ladder_physics"
NBOOT = 200
MODELS = ["B3+", "B5", "B6g", "B6gs1", "B6gaf", "B6gj", "B6gjs1", "B6g-refzero", "B6g-noref", "B6g-floor", "B6x",
          "FoldX", "FoldX-clip", "tmol-refit", "KORPM", "DDGun3D"]
VIEWS = [("megascale", "domain_origin==natural", "spearman_prot_median", "MS natural"),
         ("megascale", "domain_origin==natural,STRICT", "spearman_prot_median", "MS natural-strict"),
         ("megascale", "", "spearman_prot_median", "MS all"),
         ("fireprot", "", "pearson", "FireProt"),
         ("s669", "STRICT", "pearson", "S669"), ("s461", "STRICT", "pearson", "S461"),
         ("domainome", "", "spearman_prot_median", "Domainome"),
         ("domainome", "plddt>=70,x_dist_term>=3", "spearman_prot_median", "Domainome flag-kept"),
         ("abundance_dms", "", "spearman_prot_median", "abundance_dms")]
JOINT = {"B6gj", "B6gjs1", "B4j", "B5j", "tmol-refit", "KORPM", "DDGun3D"}
MS_ONLY = "homolog_in_megascale_train==0"
BOTH = "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"


def strict_for(model):
    return BOTH if model in JOINT else MS_ONLY


def per_protein(df, col, target):
    y = df.ddg if target == "ddg" else -df.score
    d = pd.DataFrame({"p": df.protein_id.to_numpy(), "x": df[col].to_numpy(float), "y": y.to_numpy(float)}).dropna()
    out = {}
    for prot, g in d.groupby("p"):
        if len(g) >= 10 and g.x.nunique() > 1 and g.y.nunique() > 1:
            out[prot] = spearmanr(g.x, g.y)[0]
    return pd.Series(out, dtype=float)


def load_bench(bench: str) -> pd.DataFrame:
    b = pd.read_parquet(OUT / f"preds_{bench}.parquet")
    if "homolog_in_megascale_train" in b:
        return b
    t = E.load_benchmark(bench)
    if bench == "domainome":
        h = pd.read_csv(ROOT / "data" / "processed" / "homology_domainome.csv")
        t = t.merge(h[["protein_id"] + [c for c in h.columns if c.startswith("homolog_in")]], on="protein_id", how="left")
    cols = [c for c in ("homolog_in_megascale_train", "homolog_in_fireprot_trainval") if c in t]
    key = list(E.KEYS)
    if cols:
        t = E._norm(t, key)[key + cols].drop_duplicates(key)
        b = E._norm(b, key).merge(t, on=key, how="left")
    if bench == "domainome" and "x_dist_term" not in b:        # distance to the construct terminus (flag rule)
        from physddg import features_extra as X
        st = X.site_table(b.structure.unique(), bench)[["structure", "chain", "pdb_resnum", "dist_term"]]
        st = st.rename(columns={"dist_term": "x_dist_term"})
        st["pdb_resnum"] = E._norm_resnum(st["pdb_resnum"])
        b = E._norm(b, key).merge(st.drop_duplicates(["structure", "chain", "pdb_resnum"]),
                                  on=["structure", "chain", "pdb_resnum"], how="left")
    return b


def one_cell(bench, expr, metric, label, model, b, have):
    target = "ddg" if b.ddg.notna().any() else "score"
    e = expr.replace("STRICT", strict_for(model)).strip(",")
    d = E.apply_subset(b, e) if e else b
    keep = [c for c in have if b[c].notna().mean() >= 0.95]
    d = d[d[keep].notna().all(axis=1)] if keep else d              # identical rows (D17)
    if len(d) < 20:
        return [], []
    r = E.evaluate(d, model, target=target)
    ci = E.bootstrap_ci(d, model, target=target, n_boot=NBOOT)
    row = {"view": label, "bench": bench, "model": model, "n": r["n"], "n_prot": r["n_proteins"],
           "value": r[metric], "lo": ci.get(metric, (np.nan, np.nan))[0], "hi": ci.get(metric, (np.nan, np.nan))[1],
           "median_prot": r["spearman_prot_median"], "mean_prot": r["spearman_prot_mean"], "metric": metric,
           "strict_rule": e or "all"}
    if "FoldX" in d and model != "FoldX":
        a, f = per_protein(d, model, target), per_protein(d, "FoldX", target)
        j = pd.concat([a.rename("a"), f.rename("b")], axis=1).dropna()
        row["win_vs_foldx"], row["n_prot_pairs"] = int((j.a > j.b).sum()), int(len(j))
    deltas = []
    for ref in ("B5", "B6g", "FoldX", "FoldX-clip"):
        if ref in d and ref != model:
            pb = E.paired_bootstrap(d, model, ref, target=target, n_boot=NBOOT)
            if metric in pb:
                deltas.append({"view": label, "model": model, "vs": ref, "metric": metric, "n": len(d),
                               **{k: pb[metric][k] for k in ("delta", "lo", "hi")}})
    return [row], deltas


def main():
    cache = {}
    jobs = []
    for bench, expr, metric, label in VIEWS:
        if not (OUT / f"preds_{bench}.parquet").exists():
            continue
        if bench not in cache:
            cache[bench] = load_bench(bench)
        b = cache[bench]
        have = [m for m in MODELS if m in b and b[m].notna().any()]
        for m in have:
            jobs.append((bench, expr, metric, label, m, b, have))
    print(f"{len(jobs)} (view, model) cells", flush=True)
    res = Parallel(n_jobs=12, verbose=5)(delayed(one_cell)(*j) for j in jobs)
    rows = [r for rr, _ in res for r in rr]
    deltas = [d for _, dd in res for d in dd]
    s, dl = pd.DataFrame(rows), pd.DataFrame(deltas)
    s.to_csv(OUT / "multiassay_summary.csv", index=False)
    dl.to_csv(OUT / "multiassay_deltas.csv", index=False)
    piv = s.pivot_table(index="model", columns="view", values="value", sort=False).reindex(
        [m for m in MODELS if m in set(s.model)])
    print(piv[[v[3] for v in VIEWS if v[3] in piv.columns]].round(3).to_string())
    print("\nKey paired deltas (95% cluster-bootstrap CI):")
    key = dl[dl.model.isin(["B6g", "B6gj", "B6gjs1", "B6g-refzero"]) & dl["vs"].isin(["B5", "B6g", "FoldX-clip"])]
    print(key.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
