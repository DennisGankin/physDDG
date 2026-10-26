"""Audit (read-only): independent re-computation of headline metrics from results/ladder_physics/preds_*.parquet,
joined back to the canonical tables in data/processed (not via physddg.eval). Prints:
  - row alignment checks (ddg in preds == ddg in canonical table on KEYS; duplicates; split)
  - per-protein median Spearman / pooled Spearman / Pearson per model, on (a) each model's own rows and (b) the
    intersection of rows where all compared models have a prediction
  - Domainome per-domain Spearman vs -score with min-variant thresholds 10 / 50 and ties check
  - sign spot check (buried hydrophobic -> charged)
Usage: PYTHONPATH=. .venv/bin/python scripts/99_audit_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/ladder_physics"
K = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]
MODELS = ["B4", "B5", "B3+", "FoldX", "tmol-refit", "KORPM", "DDGun3D"]


def norm(d):
    d = d.copy()
    d["pdb_resnum"] = d["pdb_resnum"].astype(str).str.replace(r"\.0$", "", regex=True)
    for k in K:
        d[k] = d[k].astype(str)
    return d


def prot_med(d, col, y="ddg", min_n=10):
    v = [spearmanr(g[col], g[y])[0] for _, g in d.groupby("protein_id") if len(g) >= min_n and g[col].nunique() > 1]
    return float(np.median(v)), len(v)


def metrics(d, col, y="ddg", min_n=10):
    d = d[d[col].notna() & d[y].notna()]
    pm, npr = prot_med(d, col, y, min_n)
    return dict(n=len(d), n_prot=npr, prot_med=round(pm, 4), spearman=round(spearmanr(d[col], d[y])[0], 4),
                pearson=round(pearsonr(d[col], d[y])[0], 4))


def table(d, models, y="ddg", min_n=10):
    ms = [m for m in models if m in d]
    own = pd.DataFrame({m: metrics(d, m, y, min_n) for m in ms}).T
    inter = d.dropna(subset=ms)
    common = pd.DataFrame({m: metrics(inter, m, y, min_n) for m in ms}).T
    return own, common


def main():
    pd.set_option("display.width", 200)
    # ---------------- Megascale
    p = norm(pd.read_parquet(R / "preds_megascale.parquet"))
    c = norm(pd.read_parquet(ROOT / "data/processed/megascale.parquet"))
    c = c[c.split == "test"]
    m = p.merge(c[K + ["ddg", "domain_origin", "homolog_in_megascale_train", "dG_wt"]], on=K, how="left",
                suffixes=("", "_canon"))
    print("MS rows", len(p), "canonical test rows", len(c), "dup keys", p.duplicated(K).sum(),
          "unmatched", m.ddg_canon.isna().sum(), "max|ddg-ddg_canon|", np.nanmax(np.abs(m.ddg - m.ddg_canon)),
          "domain_origin mismatch", (m.domain_origin != m.domain_origin_canon).sum())
    for sub, q in [("natural", m[m.domain_origin_canon == "natural"]),
                   ("natural_strict", m[(m.domain_origin_canon == "natural") & (m.homolog_in_megascale_train == 0)]),
                   ("all", m)]:
        own, com = table(q, MODELS)
        print(f"\n== Megascale-test {sub}: own rows\n{own}\n-- common rows\n{com}")
    # natural proteins count, and proteins with homologs
    nat = m[m.domain_origin_canon == "natural"]
    print("natural proteins", nat.protein_id.nunique(), "of which with MS-train homolog",
          nat[nat.homolog_in_megascale_train > 0].protein_id.nunique())
    # ---------------- FireProt
    p = norm(pd.read_parquet(R / "preds_fireprot.parquet"))
    c = norm(pd.read_csv(ROOT / "data/processed/fireprot.csv", low_memory=False))
    c = c[c.split == "test"]
    m = p.merge(c[K + ["ddg"]].drop_duplicates(K), on=K, how="left", suffixes=("", "_canon"))
    print("\nFP rows", len(p), "canon", len(c), "dup", p.duplicated(K).sum(), "max|dd|",
          np.nanmax(np.abs(m.ddg - m.ddg_canon)))
    own, com = table(m, MODELS, min_n=10)
    print(f"== FireProt-test own rows\n{own}\n-- common rows\n{com}")
    # ---------------- Domainome
    p = norm(pd.read_parquet(R / "preds_domainome.parquet"))
    p["y"] = -p["score"]
    print("\nDomainome rows", len(p), "domains", p.protein_id.nunique(), "dup", p.duplicated(K).sum())
    for mn in (10, 50):
        own, com = table(p, ["B4", "B5", "B3+", "FoldX"], y="y", min_n=mn)
        print(f"== Domainome per-domain (min_n={mn}) own rows\n{own}\n-- common rows\n{com}")
    # ---------------- sign spot check
    p = pd.read_parquet(R / "preds_megascale.parquet")
    s = p[(p.rsa < 0.1) & p.wt.isin(list("ILVFM")) & p.mut.isin(list("DEKR"))]
    print("\nburied hydrophobic->charged (MS-test):", len(s), "mean ddg", s.ddg.mean().round(2), "frac ddg>0",
          (s.ddg > 0).mean().round(3), "mean B4", s.B4.mean().round(2), "frac B4>0", (s.B4 > 0).mean().round(3),
          "mean FoldX", s.FoldX.mean().round(2))
    print(s.sample(20, random_state=0)[K + ["rsa", "ddg", "B4", "B5", "FoldX"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
