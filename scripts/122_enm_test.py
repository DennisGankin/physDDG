#!/usr/bin/env python
"""Single final test look for the ENM flexibility terms (candidates fixed by scripts/121 on train CV + val only),
G6-style transfer (Megascale-only vs FireProt-only fit of the new weight) and the Ssym antisymmetry diagnostic.

  PYTHONPATH=. .venv/bin/python scripts/122_enm_test.py --stage test|transfer|antisym|all
Candidates (per stage 2 = on B6g, stage 1 = on B6gs1): `primary` (pre-registered: GNM MSF, per-structure z,
multiplicative on the engine dE), `best` (highest d_val_natural + d_cv_natural in screen.csv), `add` (additive
GNM-MSF site offset, ablation), `ctrl` (static control: per-structure contact-number rank in the same mulC form). Outputs (results/enm_flexibility/): test_metrics.csv, test_paired.csv,
test_strata.csv, transfer.csv, antisymmetry.csv, models/*.json.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

_spec = importlib.util.spec_from_file_location("enm121", Path(__file__).with_name("121_enm_select.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
P, L = S.P, S.P.L
OUT = S.OUT
BENCH = {"megascale": ["all", "natural", "natural_strict"], "fireprot": ["all", "strict"], "s669": ["strict"],
         "s461": ["strict"], "domainome": ["all", "flag_kept"], "abundance_dms": ["all"]}


def candidates():
    t = pd.read_csv(OUT / "screen.csv")
    out = {}
    for stage in (2, 1):
        s = t[(t.stage == stage) & (t.desc != "BASE")].copy()
        s["score"] = s.d_val_natural + s.d_cv_natural
        b = s.sort_values("score", ascending=False).iloc[0]
        out[stage] = {"primary": S.PRIMARY, "best": (b.desc, b.norm, b.form), "add": ("gnm_msf", "z", "add"),
                    "ctrl": ("ctrl_cn", "rank", "mulC")}
    return out


def fit_models():
    """Fit every candidate on the full Megascale-train (B6g/B6gs1 config) and save."""
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    models = {}
    for stage, cands in candidates().items():
        tr, _, _, cfg, ecols, o = S.load(stage)
        tag = "B6g" if stage == 2 else "B6gs1"
        base = FT.fit(tr, cfg, name=tag)
        base["stage"], base["derive_opts"] = stage, o
        models[tag] = base
        for k, (de, no, fo) in cands.items():
            m = S.fit_term(tr, cfg, S.flex(tr, de, no), fo, ecols, name=f"{tag}+enm_{k}")
            m.update(stage=stage, derive_opts=o, flex_desc=de, flex_norm=no, ecols=ecols)
            models[f"{tag}+enm_{k}"] = m
            FT.save_model(m, OUT / "models" / f"{tag}+enm_{k}.json")
            print(f"{tag}+enm_{k}", (de, no, fo), S.new_weights(m), flush=True)
    return models


def bench_frame(bench):
    b = P.add_extra(P.add_phys(L.bench_table(bench), bench, P.EVAL_CAP), bench)
    b["cxxc"] = P.cxxc_flags(b)
    return b


def predict_all(models, b, bench):
    out, cache = pd.DataFrame(index=b.index), {}
    for name, m in models.items():
        st = m["stage"]
        if st not in cache:
            d = S.add_enm(P.derive(b, st, m["derive_opts"]), bench)
            S.add_glog([d])
            cache[st] = d
        d = cache[st]
        if m.get("flex_form"):
            out[name] = S.predict_term(m, d, S.flex(d, m["flex_desc"], m["flex_norm"]), m["ecols"])
        else:
            out[name] = FT.predict(m, d)
    return out


def _eval(bench, sub, d, col, base, n_boot):
    target = "ddg" if d.ddg.notna().any() else "score"
    r = E.evaluate(d, col, target=target)
    row = {"bench": bench, "subset": sub, "model": col, **{k: r[k] for k in L.METRICS}, **L.site_decomp(d, col, target)}
    pr = None
    if col != base:
        pb = E.paired_bootstrap(d, col, base, target=target, n_boot=n_boot)
        pr = {"bench": bench, "subset": sub, "model": col, "vs": base, "n": len(d)}
        for k in ("spearman_prot_median", "spearman", "pearson"):
            if k in pb:
                pr.update({f"d_{k}": pb[k]["delta"], f"d_{k}_lo": pb[k]["lo"], f"d_{k}_hi": pb[k]["hi"]})
    return row, pr


def stage_test(n_boot=1000):
    models = fit_models()
    tasks, strata_rows = [], []
    for bench, subs in BENCH.items():
        b = bench_frame(bench)
        Pm = predict_all(models, b, bench)
        b = pd.concat([b, Pm], axis=1)
        keep = list(dict.fromkeys(P.KS + ["ddg", "score", "cluster", "domain_origin", "rsa", "dvol", "plddt", "ctx_bfactor",
                                          "struct_source", "split", "x_dist_term", "homolog_in_megascale_train",
                                          "homolog_in_fireprot_trainval", "near_hetatm"] + list(models)))
        b = b[[c for c in keep if c in b]]
        b.to_parquet(OUT / f"preds_{bench}.parquet", index=False)
        for sub in subs:
            d = E.apply_subset(b, P.SUBSETS[bench][sub])
            d = d[d[list(models)].notna().all(axis=1)]
            for col in models:
                base = "B6g" if models[col]["stage"] == 2 else "B6gs1"
                tasks.append((bench, sub, d[list(dict.fromkeys(P.KEYS + ["ddg", "cluster", col, base] +
                                                               (["score"] if "score" in d else [])))], col, base,
                              200 if bench == "domainome" else n_boot))
        if bench in ("megascale", "fireprot"):
            for sname, mask in P.strata(b).items():
                g = b[mask].dropna(subset=list(models))
                for col in models:
                    if len(g) < 30:
                        continue
                    r = E.evaluate(g, col)
                    strata_rows.append({"bench": bench, "stratum": sname, "model": col, "n": r["n"],
                                        "spearman_prot_median": r["spearman_prot_median"], "spearman": r["spearman"]})
    res = Parallel(n_jobs=32)(delayed(_eval)(*t) for t in tasks)
    met = pd.DataFrame([r[0] for r in res])
    met.to_csv(OUT / "test_metrics.csv", index=False)
    pr = pd.DataFrame([r[1] for r in res if r[1] is not None])
    pr.to_csv(OUT / "test_paired.csv", index=False)
    st = pd.DataFrame(strata_rows)
    st.to_csv(OUT / "test_strata.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(met[["bench", "subset", "model", "n", "spearman_prot_median", "spearman", "pearson"] +
                  [c for c in met.columns if c.startswith("site") or c.startswith("within")]].round(4).to_string(index=False))
        print(pr.round(4).to_string(index=False))
        print(st[st.bench == "megascale"].pivot(index="stratum", columns="model", values="spearman_prot_median").round(3).to_string())


def stage_transfer(n_boot=100):
    rows = []
    for stage, cands in candidates().items():
        tr, _, fp, cfg, ecols, _ = S.load(stage)
        for k, (de, no, fo) in cands.items():
            for src, d in (("megascale", tr), ("fireprot", fp)):
                for constrained in (True,):
                    m = S.fit_term(d, cfg, S.flex(d, de, no), fo, ecols, n_boot=n_boot, name=f"{k}_{src}")
                    for f, w in S.new_weights(m).items():
                        rows.append({"stage": stage, "cand": k, "desc": de, "norm": no, "form": fo, "train": src,
                                     "feature": f, "w": w, "se": m["se"]["weights"][f]})
    t = pd.DataFrame(rows)
    w = t.pivot_table(index=["stage", "cand", "desc", "norm", "form", "feature"], columns="train", values=["w", "se"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["ratio_over_1.7"] = w.w_fireprot / w.w_megascale / 1.7
    w["sign_flip"] = np.sign(w.w_fireprot) != np.sign(w.w_megascale)
    w["gt3x"] = (w["ratio_over_1.7"].abs() > 3) | (w["ratio_over_1.7"].abs() < 1 / 3)
    w.to_csv(OUT / "transfer.csv", index=False)
    print(w.round(4).to_string(index=False))


def stage_antisym():
    """Ssym diagnostic (Megascale-only models): additive offset vs multiplicative flexibility."""
    models = {p.stem: FT.load_model(p) for p in sorted((OUT / "models").glob("*.json"))}
    for stage in (2, 1):
        tr, _, _, cfg, _, o = S.load(stage)
        tag = "B6g" if stage == 2 else "B6gs1"
        m = FT.fit(tr, cfg, name=tag)
        m["stage"], m["derive_opts"] = stage, o
        models[tag] = m
    b = bench_frame("ssym")
    b = pd.concat([b, predict_all(models, b, "ssym")], axis=1)
    rows = []
    for c in models:
        a = E.antisymmetry(b, c)
        rows.append({"model": c, **{k: a[k] for k in ("n_pairs", "r_pearson", "bias", "pearson_direct", "pearson_reverse")}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "antisymmetry.csv", index=False)
    print(t.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    for s in (["test", "transfer", "antisym"] if a.stage == "all" else [a.stage]):
        {"test": stage_test, "transfer": stage_transfer, "antisym": stage_antisym}[s]()


if __name__ == "__main__":
    main()
