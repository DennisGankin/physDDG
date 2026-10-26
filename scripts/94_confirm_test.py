#!/usr/bin/env python
"""Residual analysis, step 4: ONE confirmation look at held-out sets for the features accepted by scripts/93
(results/residual_analysis/final_features_{strict,relaxed}.json), plus transfer checks. Nothing here feeds back into selection.

- B5 and B5+added (each refit on all Megascale-train, same config) evaluated on identical rows of:
  Megascale-test (all 28 proteins, natural 21, strict, natural_strict), FireProt-test, Domainome (per-domain rho vs score).
  Paired cluster bootstrap for the difference. FoldX on the same rows for reference.
- Also each accepted feature added alone (to see which one transfers).
- G6-style: B5+added refit on FireProt-train only vs Megascale-train only (weights, cluster-bootstrap SEs).
Both selection outcomes (strict rule = primary, relaxed rule = secondary) were fixed before this script was run.
Outputs results/residual_analysis/confirm_test.csv, confirm_paired.csv, transfer_new_terms.csv.
  PYTHONPATH=. .venv/bin/python scripts/94_confirm_test.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import features_extra as X  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = ROOT / "results" / "residual_analysis"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C93 = load("c93", ROOT / "scripts" / "93_candidate_features.py")
L52 = C93.L52
L51 = L52.L


def with_extra(d, name):
    st = X.site_table(d.structure.unique(), name).drop(columns=["structure_abs"]).add_prefix("x_").rename(
        columns={"x_structure": "structure", "x_chain": "chain", "x_pdb_resnum": "pdb_resnum"})
    d = d.assign(pdb_resnum=d.pdb_resnum.astype(str), chain=d.chain.astype(str))
    n0 = len(d)
    d = d.merge(st, on=["structure", "chain", "pdb_resnum"], how="left")
    assert len(d) == n0
    print(name, "site descriptor coverage", round(d.x_ncb10.notna().mean(), 4))
    return pd.concat([d, X.candidate_features(d)], axis=1)


def add_scale(df, model_base, extra):
    for s in [e for e in extra if e.startswith("scale:")]:
        df[s] = C93.core_of(model_base, df) * df[s.split(":", 1)[1]].to_numpy(float)
    return df


def fit_model(trd, extra, base=None):
    feats = list(C93.BASE_FEATS) + list(extra)
    if any(e.startswith("scale:") for e in extra):
        trd = add_scale(trd.copy(), base, extra)
    return FT.fit(trd, C93.cfg_for(feats), name="+".join(extra) or "B5")


def main():
    strict = json.loads((OUT / "final_features_strict.json").read_text())["added"]
    relaxed = json.loads((OUT / "final_features_relaxed.json").read_text())["added"]
    added = list(dict.fromkeys(strict + relaxed))
    print("accepted (strict rule):", strict, " (relaxed rule):", relaxed)
    tr, _, fp = L52.train_tables()
    o = C93.OPTS
    trd = with_extra(L52.derive(tr, 2, o), "megascale")
    base = fit_model(trd, [])
    variants = {"B5": [], "B5+strict": strict, "B5+relaxed": relaxed}
    for a in added:
        variants[f"B5+{a}"] = [a]
    variants = {k: v for k, v in variants.items() if k == "B5" or v}
    models = {k: (base if not v else fit_model(trd, v, base)) for k, v in variants.items()}
    for k, m in models.items():
        print(k, {f: round(m["weights"][f], 4) for f in variants[k]})
    rows, pairs = [], []
    subsets = {"megascale": {"all (28)": "", "natural (21)": "domain_origin==natural", "strict": "homolog_in_megascale_train==0",
                             "natural_strict": "domain_origin==natural,homolog_in_megascale_train==0"},
               "fireprot": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
               "domainome": {"all": "", "plddt70": "plddt>=70"}}
    for bench, subs in subsets.items():
        b = L51.bench_table(bench)
        b = L52.add_phys(b, bench)
        b["cxxc"] = L52.cxxc_flags(b)
        b = with_extra(L52.derive(b, 2, o), bench)
        for k, m in models.items():
            bb = add_scale(b.copy(), base, variants[k])
            b[k] = FT.predict(m, bb)
        b = L52.attach_competitors(b, bench)
        for sub, expr in subs.items():
            d = E.apply_subset(b, expr)
            target = "ddg" if d.ddg.notna().any() else "score"
            print(bench, sub, len(d), flush=True)
            for k in list(models) + (["FoldX"] if "FoldX" in d else []):
                dd = d[d[k].notna()] if k != "FoldX" else d
                r = E.evaluate(dd, k, target=target)
                rows.append({"bench": bench, "subset": sub, "model": k, **{x: r[x] for x in ("n", "n_proteins", "spearman", "spearman_prot_median", "pearson")}})
            for k in ("B5+strict", "B5+relaxed"):
                if k not in models:
                    continue
                pb = E.paired_bootstrap(d, k, "B5", target=target, n_boot=200 if bench == "domainome" else 1000)
                pairs.append({"bench": bench, "subset": sub, "model": k, "vs": "B5",
                              **{f"d_{x}": pb[x]["delta"] for x in ("spearman", "spearman_prot_median", "pearson") if x in pb},
                              **{f"d_{x}_ci": f"[{pb[x]['lo']:+.3f}, {pb[x]['hi']:+.3f}]" for x in ("spearman", "spearman_prot_median", "pearson") if x in pb}})
    t, p = pd.DataFrame(rows), pd.DataFrame(pairs)
    t.to_csv(OUT / "confirm_test.csv", index=False)
    p.to_csv(OUT / "confirm_paired.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))
    # ---------------------------------------------------------------- G6-style transfer of the new weights
    fpd = with_extra(L52.derive(fp.assign(struct_source="xtal"), 2, o), "fireprot")
    feats = list(C93.BASE_FEATS) + [a for a in added if not a.startswith("scale:")]
    fits = {}
    for src, d in (("megascale", trd), ("fireprot", fpd)):
        fits[src] = FT.fit(d, C93.cfg_for(feats), n_boot=100, n_jobs=32, name=f"B5+added_{src}")
    unsigned = {}
    for src, d in (("megascale", trd), ("fireprot", fpd)):     # sign-free refit to detect flips hidden by bounds
        c = C93.cfg_for(feats)
        c.signs = {k: v for k, v in c.signs.items() if k not in added}
        unsigned[src] = FT.fit(d, c)
    rr = []
    for f in feats:
        wa, wb = fits["megascale"]["weights"][f], fits["fireprot"]["weights"][f]
        sa, sb = fits["megascale"]["se"]["weights"][f], fits["fireprot"]["se"]["weights"][f]
        rr.append({"feature": f, "new": f in added, "w_megascale": wa, "se_megascale": sa, "w_fireprot": wb, "se_fireprot": sb,
                   "ratio_over_1.7": (wb / wa / 1.7) if wa else np.nan,
                   "w_megascale_unsigned": unsigned["megascale"]["weights"][f], "w_fireprot_unsigned": unsigned["fireprot"]["weights"][f]})
    rr = pd.DataFrame(rr)
    rr.to_csv(OUT / "transfer_new_terms.csv", index=False)
    print(rr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
