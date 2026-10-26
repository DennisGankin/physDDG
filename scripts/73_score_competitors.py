#!/usr/bin/env python
"""Score results/competitors/<method>_<bench>.csv with scripts/30_evaluate.py (physddg/eval.py): all rows and strict
(homology-controlled) subsets, antisymmetry on Ssym. Strict subsets use benchmark columns plus the KORPM-training
overlap columns written by 74_korpm_leakage.py (merged into a temporary data root under data/work/score_root).
Per-run JSON in results/competitors/metrics/, aggregated table in results/competitors/summary.csv (+ markdown on stdout).
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comp_common import BENCHES, KEYS, ROOT, load_bench  # noqa: E402

COMP = ROOT / "results/competitors"
TMP = ROOT / "data/work/score_root"
SPLIT = {"megascale": "test", "fireprot": "test"}
# tag -> subset expression (applied by apply_subset to the augmented benchmark)
SUBSETS = {
    "all": None,
    "strict_ms": "homolog_in_megascale_train==0",
    "strict_korpm": "homolog_in_korpm_train==0",
    "strict_both": "homolog_in_megascale_train==0,homolog_in_korpm_train==0",
    "novel_variant": "exact_in_korpm_train==0",
}


def build_root(bench, methods):
    (TMP / "data/processed").mkdir(parents=True, exist_ok=True)
    df = load_bench(bench)  # same rows as the predictions (test split etc.)
    if bench == "domainome":  # homology columns live in a separate table
        h = pd.read_csv(ROOT / "data/processed/homology_domainome.csv")
        df = df.merge(h[["protein_id", "homolog_in_megascale_train", "homolog_in_fireprot_trainval"]], on="protein_id", how="left")
    ann = pd.read_csv(ROOT / "data/work/leak" / f"{bench}_ann.csv", dtype={"pdb_resnum": str})
    df = df.merge(ann, on=KEYS, how="left")
    for c in ("homolog_in_korpm_train", "exact_in_korpm_train", "pdb_in_korpm_train"):
        df[c] = df[c].fillna(0)
    # rows covered by every method (domainome: DDGun only ran on a subset of constructs)
    cov = None
    for m in methods:
        p = COMP / f"{m}_{bench}.csv"
        if p.exists():
            k = pd.read_csv(p, dtype={"pdb_resnum": str}).dropna(subset=["ddg_pred"])[KEYS].drop_duplicates()
            k[f"in_{m}"] = 1
            df = df.merge(k, on=KEYS, how="left")
            df[f"in_{m}"] = df[f"in_{m}"].fillna(0).astype(int)
    if bench == "domainome" and (COMP / f"ddgun3d_{bench}.csv").exists():
        df["in_ddgun_subset"] = df["in_ddgun3d"]
    keep = [c for c in df.columns if bench != "domainome" or c in KEYS + ["dataset", "split", "cluster", "score", "ddg", "pair_id", "direction"]
            or c.startswith(("homolog_", "exact_", "pdb_in", "in_"))]
    df = df[keep]
    df.to_parquet(TMP / "data/processed" / f"{bench}.parquet", index=False)


def run(job):
    m, b, tag, subset = job
    out = COMP / "metrics" / f"{m}_{b}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/30_evaluate.py"), "--bench", b, "--pred", str(COMP / f"{m}_{b}.csv"),
           "--root", str(TMP), "--out", str(out), "--n-boot", "300"]
    if b in SPLIT:
        cmd += ["--split", SPLIT[b]]
    if subset:
        cmd += ["--subset", subset]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return job, None, r.stderr[-300:]
    return job, json.load(open(out)), r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["korpm", "ddgun3d", "ddgun"])
    ap.add_argument("--bench", nargs="+", default=BENCHES)
    a = ap.parse_args()
    jobs = []
    for b in a.bench:
        build_root(b, a.methods)
        for m in a.methods:
            if not (COMP / f"{m}_{b}.csv").exists():
                continue
            for tag, sub in SUBSETS.items():
                if b == "domainome":
                    sub = "in_ddgun_subset==1" + ("," + sub if sub else "") if (COMP / "ddgun3d_domainome.csv").exists() else sub
                jobs.append((m, b, tag, sub))
            if b == "domainome":  # KORPM/DDGun on the full / common subset
                pass
    rows = []
    with ThreadPoolExecutor(16) as ex:
        for (m, b, tag, sub), res, log in ex.map(run, jobs):
            if res is None:
                print("FAILED", m, b, tag, log)
                continue
            mt = res["predictors"]["main"]["metrics"]
            ci = res["predictors"]["main"]["ci95"]
            r = dict(method=m, bench=b, subset=tag, n=mt["n"], n_proteins=mt["n_proteins"], spearman=mt["spearman"],
                     spearman_lo=ci["spearman"][0], spearman_hi=ci["spearman"][1], pearson=mt["pearson"],
                     spearman_prot_median=mt["spearman_prot_median"], rmse=mt["rmse"], auroc_destab=mt["auroc_destab"],
                     sign_acc=mt["sign_acc"], coverage=res["merge"]["main"]["coverage"])
            an = res["predictors"]["main"].get("antisymmetry")
            if an:
                r |= dict(anti_r=an["r_pearson"], anti_bias=an["bias"], pearson_direct=an["pearson_direct"],
                          pearson_reverse=an["pearson_reverse"])
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["bench", "method", "subset"])
    out = COMP / "summary.csv"
    if out.exists() and a.bench != BENCHES or out.exists() and a.methods != ["korpm", "ddgun3d", "ddgun"]:
        old = pd.read_csv(out)
        old = old[~old.set_index(["method", "bench"]).index.isin(df.set_index(["method", "bench"]).index)]
        df = pd.concat([old, df]).sort_values(["bench", "method", "subset"])
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["spearman_lo", "spearman_hi", "sign_acc"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def markdown(path=COMP / "summary.csv", out=COMP / "summary.md"):
    """Markdown tables for notes/competitors.md from summary.csv (python scripts/73_score_competitors.py --md-only)."""
    d = pd.read_csv(path)
    names = {"korpm": "KORPM", "ddgun3d": "DDGun3D", "ddgun": "DDGun (seq)"}
    f = lambda v, p=3: "" if pd.isna(v) else f"{v:.{p}f}"  # noqa: E731
    lines = []
    for sub, title in (("all", "All rows"), ("strict_ms", "Strict: no Megascale-train homologue (homolog_in_megascale_train==0)"),
                       ("strict_both", "Strict: no Megascale-train AND no KORPM-training homologue"),
                       ("novel_variant", "No exact variant (same PDB, chain, position, WT, mutant) in KORPM's training set")):
        lines += [f"**{title}**", "", "| bench | method | n | prot | Spearman [95% CI] | per-protein median Sp | Pearson | RMSE | AUROC(ddg>1) | Ssym r(dir,-rev) | Ssym bias |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in BENCHES:
            for m in ("korpm", "ddgun3d", "ddgun"):
                r = d[(d.bench == b) & (d.method == m) & (d["subset"] == sub)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(f"| {b} | {names[m]} | {int(r.n)} | {int(r.n_proteins)} | {f(r.spearman)} [{f(r.spearman_lo,2)}, {f(r.spearman_hi,2)}] | "
                             f"{f(r.spearman_prot_median)} | {f(r.pearson)} | {f(r.rmse,2)} | {f(r.auroc_destab)} | {f(r.get('anti_r'))} | {f(r.get('anti_bias'))} |")
        lines.append("")
    Path(out).write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--md-only" in sys.argv:
        markdown()
    else:
        main()
        markdown()
