#!/usr/bin/env python
"""Evolution-aware extension ('PhysDDG+evo'), step 3: N_eff-gated profile terms added to B5 / B3+; selection on
Megascale train-CV + val only, then one evaluation pass on the held-out benchmarks.

This is a DIFFERENT INFORMATION SOURCE from the structure-only headline model: the profile features come from an
hhblits search of UniClust30 (2018_08), which contains homologues of every natural benchmark protein by design.
It is offered as an optional mode, never mixed into the headline numbers.

Models (all physddg/fit.py, B5 configuration of results/ladder_physics/selection.json: Huber + censoring, 20 shrunk
reference energies, physics sign constraints; evo weights constrained <= 0 because every evo_ column is oriented
'higher = more tolerated'):
  B5        feature_set('full', opts_s2) of scripts/52 (refit here on Megascale-train)
  B3+       scripts/51 B3+ (10 structure-statistics weights + reference energies; refit here)
  B5evo / B3+evo   base + gate * evo features (gate in [0, 1] from the per-site N_eff; gate * x = 0 where no profile)
  Evo-only  reference energies + gated evo features (no structure)
Gates: apriori = clip(log10(neff_site)/2, 0, 1) (fixed before any fit); none = 1; hard10 = neff_site >= 10;
       expN = 1 - exp(-(neff_site - 1)/N), N in {3, 10, 30} (the 'fitted' gate: N chosen on val/CV).
Selection (pre-set): score = delta val-natural per-protein median rho + delta CV-natural per-protein median rho vs the
base, among candidates with CV improved in >= 4/5 folds; among candidates within 0.003 of the best score the one with
fewer weights wins. Feature set first (apriori gate), then the gate on the chosen set. The B3+evo model reuses the
B5evo choice (no separate selection).

  PYTHONPATH=. .venv/bin/python scripts/10c_evo_eval.py --stage select|final|eval|all
Outputs results/evo_extension/: selection.csv, selection.json, models/*.json, metrics.csv, paired.csv, strata.csv,
site_decomp.csv, main_table.md.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = ROOT / "results" / "evo_extension"
FEAT = ROOT / "data" / "features"
SCR = ROOT / "data" / "work" / "evo"
K = 5
KEYS = list(E.KEYS)
KS = KEYS + ["structure"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L52 = _load("lp52", ROOT / "scripts" / "52_ladder_physics.py")
L51 = L52.L
SEL = L52.load_sel()
OPTS = SEL["opts_s2"]
B5_FEATS, B5_EXTRA = L52.feature_set("full", OPTS)
SEL51 = json.loads((ROOT / "results/baselines_ladder/selection.json").read_text())
B3P_FEATS = L51.specs(SEL51["env"])["B3+"][0]

EVO_RAW = ["evo_lor", "evo_pssm", "evo_entropy", "evo_nonwt", "evo_gap", "evo_hyd_lor", "evo_vol_lor", "evo_hyd_z",
           "evo_vol_z", "evo_blprof"]
GATES = ["apriori", "none", "hard10", "exp3", "exp10", "exp30"]
SETS = {"lor": ["evo_lor"], "pssm": ["evo_pssm"], "blprof": ["evo_blprof"], "lor+ent": ["evo_lor", "evo_entropy"],
        "lor+ent+gap": ["evo_lor", "evo_entropy", "evo_gap"],
        "lor+classes": ["evo_lor", "evo_hyd_lor", "evo_vol_z"],
        "full6": ["evo_lor", "evo_entropy", "evo_gap", "evo_hyd_lor", "evo_vol_z", "evo_blprof"]}


def gate_values(neff_s: pd.Series, kind: str) -> np.ndarray:
    n = neff_s.to_numpy(float)
    ok = np.isfinite(n)
    n = np.where(ok, n, 1.0)
    if kind == "apriori":
        g = np.clip(np.log10(n) / 2.0, 0, 1)
    elif kind == "none":
        g = np.ones_like(n)
    elif kind == "hard10":
        g = (n >= 10).astype(float)
    elif kind.startswith("exp"):
        g = 1.0 - np.exp(-(n - 1.0) / float(kind[3:]))
    return np.where(ok, g, 0.0)


def gcol(x, gate):
    return f"g{gate}:{x}"


def add_evo(d: pd.DataFrame, name: str, gates=GATES) -> pd.DataFrame:
    """Merge evo_<name>.parquet on KEYS+structure and materialise gated columns (0 where no profile)."""
    ev = pd.read_parquet(FEAT / f"evo_{name}.parquet")
    ev["pdb_resnum"] = E._norm_resnum(ev["pdb_resnum"])
    ev = ev.drop_duplicates(KS)
    d = d.assign(pdb_resnum=E._norm_resnum(d["pdb_resnum"]), chain=d["chain"].astype(str))
    n0 = len(d)
    cols = [c for c in ev.columns if c.startswith("evo_")]
    d = d.drop(columns=[c for c in cols if c in d]).merge(ev[KS + cols], on=KS, how="left")
    assert len(d) == n0
    d["evo_found"] = d["evo_found"].fillna(False).astype(bool)
    for gk in gates:
        g = gate_values(d["evo_neff_site"].where(d.evo_found), gk)
        for x in EVO_RAW:
            d[gcol(x, gk)] = np.where(d.evo_found, g * d[x].fillna(0).to_numpy(float), 0.0)
    return d


def cfg_for(base: str, feats: list) -> FT.FitConfig:
    if base == "B5":
        c = L52.make_cfg(feats, SEL, B5_EXTRA)
    else:   # B3+ / Evo-only: scripts/51 objective (huber, censor, rows, ref_l2), unconstrained structural weights
        c = FT.FitConfig(features=list(feats), ref=True, loss=SEL51["loss"], censor=SEL51["censor"], within=SEL51["within"],
                         rank_weight=SEL51["rank_weight"], ref_l2=SEL51["ref_l2"], ref_dataset="megascale")
    for f in feats:
        if f.startswith("g") and ":evo_" in f:
            c.signs[f] = -1
    return c


def base_feats(base):
    return {"B5": list(B5_FEATS), "B3+": list(B3P_FEATS), "Evo": []}[base]


# ----------------------------------------------------------------------------- tables
def train_val():
    """Megascale train / val with B5 (stage-2 derived) and B3+ (simple) features + evo. Slim columns."""
    tr, va, _ = L52.train_tables()
    trd, vad = L52.derive(tr, 2, OPTS), L52.derive(va, 2, OPTS)
    extra = [f for f in B3P_FEATS if f not in trd]
    out = []
    for d, split in ((trd, "train"), (vad, "val")):
        s = L51.megascale_train_table(split)[KS + extra].drop_duplicates(KS)
        s["pdb_resnum"] = E._norm_resnum(s["pdb_resnum"])
        d = d.assign(pdb_resnum=E._norm_resnum(d["pdb_resnum"])).merge(s, on=KS, how="left")
        d = add_evo(d, "megascale")
        keep = set(KS) | set(B5_FEATS) | set(B3P_FEATS) | {"y", "lo", "hi", "sigma_meas", "dataset", "cluster", "domain_origin",
                                                         "rsa", "evo_found", "evo_neff_site", "evo_neff"} | {
            c for c in d.columns if c.startswith("g") and ":evo_" in c}
        out.append(d[[c for c in d.columns if c in keep]].reset_index(drop=True))
    return out


# ----------------------------------------------------------------------------- selection
_TV = None


def _tv():
    global _TV
    if _TV is None:
        _TV = pickle.loads((SCR / "_tv.pkl").read_bytes())
    return _TV


def rho_prot(df, col):
    return E.evaluate(df.assign(ddg=df.y), col)["spearman_prot_median"]


def run_candidate(label, base, extra):
    import torch
    torch.set_num_threads(2)
    tr, va, folds = _tv()
    feats = base_feats(base) + list(extra)
    cfg = cfg_for(base, feats)
    oof = np.full(len(tr), np.nan)
    for j in range(K):
        m = FT.fit(tr[folds != j], cfg)
        oof[folds == j] = FT.predict(m, tr[folds == j])
    m = FT.fit(tr, cfg)
    pv = FT.predict(m, va)
    t = tr.assign(oof=oof)
    pt = t[t.y.notna()]
    nat = pt[pt.domain_origin == "natural"]
    v = va.assign(pred=pv)
    v = v[v.y.notna()]
    return {"label": label, "base": base, "extra": "+".join(extra), "n_new_w": len(extra),
            "cv_nat_prot": rho_prot(nat, "oof"), "cv_all_prot": rho_prot(pt, "oof"),
            "cv_des_prot": rho_prot(pt[pt.domain_origin != "natural"], "oof"),
            "cv_nat_fold": [rho_prot(g, "oof") for _, g in nat.groupby(folds[nat.index.to_numpy()])],
            "val_nat_prot": rho_prot(v[v.domain_origin == "natural"], "pred"), "val_all_prot": rho_prot(v, "pred"),
            "val_des_prot": rho_prot(v[v.domain_origin != "natural"], "pred"),
            "cv_pearson": float(np.corrcoef(pt.oof, pt.y)[0, 1]),
            "weights": {f: round(m["weights"][f], 4) for f in extra}}


def score(t: pd.DataFrame, ref: dict) -> pd.DataFrame:
    t = t.copy()
    for k in ("val_nat_prot", "val_all_prot", "cv_nat_prot", "cv_all_prot", "val_des_prot", "cv_des_prot"):
        t[f"d_{k}"] = t[k] - ref[k]
    t["folds_up"] = [int(np.sum(np.array(f) > np.array(ref["cv_nat_fold"]) + 1e-9)) for f in t.cv_nat_fold]
    t["score"] = t.d_val_nat_prot + t.d_cv_nat_prot
    return t


def pick(t: pd.DataFrame) -> pd.Series:
    ok = t[t.folds_up >= 4]
    if ok.empty:
        return None
    best = ok.score.max()
    near = ok[ok.score >= best - 0.003].sort_values(["n_new_w", "score"], ascending=[True, False])
    return near.iloc[0]


def stage_select():
    OUT.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    tr, va = train_val()
    print("train", len(tr), "val", len(va), "evo coverage train", round(tr.evo_found.mean(), 4), "val", round(va.evo_found.mean(), 4),
          flush=True)
    folds = FT.group_folds(tr["cluster"], K, 0)
    (SCR / "_tv.pkl").write_bytes(pickle.dumps((tr, va, folds)))
    # round 1: bases + feature sets with the a-priori gate (B5), plus the evo-only model and B3+ references
    cands = [("B5", "B5", []), ("B3+", "B3+", [])]
    cands += [(f"B5+{k}[apriori]", "B5", [gcol(x, "apriori") for x in v]) for k, v in SETS.items()]
    cands += [(f"B3++{k}[apriori]", "B3+", [gcol(x, "apriori") for x in v]) for k, v in SETS.items() if k in ("lor", "lor+ent", "full6")]
    cands += [(f"Evo:{k}[apriori]", "Evo", [gcol(x, "apriori") for x in v]) for k, v in SETS.items() if k in ("lor", "lor+ent", "full6")]
    res = Parallel(n_jobs=min(20, len(cands)))(delayed(run_candidate)(*c) for c in cands)
    t = pd.DataFrame(res)
    ref = t[t.label == "B5"].iloc[0].to_dict()
    t = score(t, ref)
    b5 = t[(t.base == "B5") & (t.label != "B5")]
    best = pick(b5)
    print(t.drop(columns=["cv_nat_fold"]).round(4).to_string(index=False), flush=True)
    if best is None:
        chosen_set = "lor"
        print("no feature set passes the 4/5-fold rule; using 'lor' for the gate study", flush=True)
    else:
        chosen_set = best.label.split("+", 1)[1].split("[")[0]
    print("chosen set:", chosen_set, flush=True)
    # round 2: gate on the chosen set
    c2 = [(f"B5+{chosen_set}[{g}]", "B5", [gcol(x, g) for x in SETS[chosen_set]]) for g in GATES if g != "apriori"]
    res2 = Parallel(n_jobs=len(c2))(delayed(run_candidate)(*c) for c in c2)
    t2 = score(pd.DataFrame(res2), ref)
    t = pd.concat([t, t2], ignore_index=True)
    gate_rows = t[t.label.str.startswith(f"B5+{chosen_set}[")]
    gbest = pick(gate_rows)
    gate = gbest.label.split("[")[1].rstrip("]") if gbest is not None else "apriori"
    # ties go to the a-priori gate (no extra choice made)
    ap = gate_rows[gate_rows.label.str.endswith("[apriori]")].iloc[0]
    if gbest is not None and gbest.score - ap.score < 0.003:
        gate = "apriori"
    t.drop(columns=["cv_nat_fold"]).to_csv(OUT / "selection.csv", index=False)
    print(t2.drop(columns=["cv_nat_fold"]).round(4).to_string(index=False), flush=True)
    sel = {"set": chosen_set, "features": SETS[chosen_set], "gate": gate,
           "rule": "score = d val_nat + d cv_nat vs B5; folds_up >= 4; within 0.003 -> fewer weights; gate ties -> apriori"}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selection", sel, flush=True)


# ----------------------------------------------------------------------------- final fits
def model_specs():
    sel = json.loads((OUT / "selection.json").read_text())
    ev = [gcol(x, sel["gate"]) for x in sel["features"]]
    ev_ap = [gcol(x, "apriori") for x in sel["features"]]
    sp = {"B5": ("B5", []), "B5evo": ("B5", ev), "B3+": ("B3+", []), "B3+evo": ("B3+", ev), "Evo-only": ("Evo", ev),
          "B5evo-nogate": ("B5", [gcol(x, "none") for x in sel["features"]]),
          "Evo-only-full6": ("Evo", [gcol(x, "apriori") for x in SETS["full6"]])}
    if sel["gate"] != "apriori":
        sp["B5evo-apriori"] = ("B5", ev_ap)
    if sel["gate"] == "none":
        sp.pop("B5evo-nogate")
    return sp


def stage_final():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    tr, va, folds = pickle.loads((SCR / "_tv.pkl").read_bytes())
    for name, (base, ev) in model_specs().items():
        feats = base_feats(base) + ev
        m = FT.fit(tr, cfg_for(base, feats), n_boot=50 if name in ("B5evo", "B3+evo") else 0, n_jobs=25, name=name,
                   provenance={"train": "megascale train (Dataset 3 + censored)", "evo": ev, "base": base,
                               "evo_features": "data/features/evo_megascale.parquet (scripts/10b)",
                               "selection": "results/evo_extension/selection.csv (Megascale-val natural + train grouped CV)"})
        FT.save_model(m, OUT / "models" / f"{name}.json")
        se = m.get("se", {}).get("weights", {})
        print(name, {k: f"{v:.3f}" + (f"±{se[k]:.3f}" if k in se else "") for k, v in m["weights"].items() if ":evo_" in k},
              "conv", m["converged"], flush=True)


# ----------------------------------------------------------------------------- evaluation
SUBSETS = {"megascale": {"all": "", "natural": "domain_origin==natural", "designed": "domain_origin==designed",
                         "strict": "homolog_in_megascale_train==0",
                         "natural_strict": "domain_origin==natural,homolog_in_megascale_train==0"},
           "fireprot": {"all": "", "strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
           "s669": {"strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
           "s461": {"strict": "homolog_in_megascale_train==0,homolog_in_fireprot_trainval==0"},
           "ssym": {"direct": "direction==direct"},
           "domainome": {"all": "", "strict": "homolog_in_megascale_train==0", "plddt<50": "plddt<50",
                         "plddt50-70": "plddt>=50,plddt<70", "plddt70-90": "plddt>=70,plddt<90", "plddt>=90": "plddt>=90"},
           "abundance_dms": {"all": ""}}
MAIN = ["B5", "B5evo", "B3+", "B3+evo", "Evo-only", "Evo-only-full6", "B5evo-nogate", "B5evo-apriori"]
COMPS = ["FoldX", "DDGun3D", "KORPM"]


def bench_frame(bench, models):
    b = L51.bench_table(bench)
    b = L52.add_phys(b, bench)
    b["cxxc"] = L52.cxxc_flags(b)
    b = L52.derive(b, 2, OPTS)
    b = add_evo(b, bench)
    for k, m in models.items():
        b[k] = FT.predict(m, b)
    b = L52.attach_competitors(b.drop(columns=["B3+"], errors="ignore"), bench).rename(columns={"B3+": "B3+(51)"})
    b["B3+"] = FT.predict(models["B3+"], b)
    return b


def strata_masks(b):
    st = {"all": np.ones(len(b), bool)}
    if "domain_origin" in b:
        st["natural"], st["designed"] = (b.domain_origin == "natural").to_numpy(), (b.domain_origin == "designed").to_numpy()
    rsa = b["rsa"].to_numpy(float) if "rsa" in b else np.full(len(b), np.nan)
    st["buried(rsa<0.25)"], st["exposed(rsa>=0.25)"] = rsa < 0.25, rsa >= 0.25
    n = b["evo_neff_site"].to_numpy(float)
    st["neff_site<10"], st["neff_site10-100"], st["neff_site>=100"] = n < 10, (n >= 10) & (n < 100), n >= 100
    st["no_profile"] = ~b.evo_found.to_numpy()
    return st


def _eval_one(bench, sub, d, model, target, n_boot):
    dd = d[d[model].notna()]
    r = E.evaluate(dd, model, target=target)
    row = {"bench": bench, "subset": sub, "model": model, **{k: r[k] for k in ("n", "n_proteins", "spearman",
                                                                                 "spearman_prot_median", "pearson")}}
    row.update(L51.site_decomp(dd, model, target))
    return row


def stage_eval(n_boot=500):
    models = {p.stem: FT.load_model(p) for p in sorted((OUT / "models").glob("*.json"))}
    rows, pairs, strata = [], [], []
    for bench, subs in SUBSETS.items():
        b = bench_frame(bench, models)
        print(bench, len(b), "evo coverage", round(b.evo_found.mean(), 4), flush=True)
        cols = [m for m in MAIN if m in b] + [c for c in COMPS if c in b]
        for sub, expr in subs.items():
            d = E.apply_subset(b, expr)
            if len(d) == 0:
                continue
            target = "ddg" if d.ddg.notna().any() else "score"
            common = d.dropna(subset=[c for c in cols if c != "B5evo-apriori" and c in ("B5", "B5evo", "B3+", "B3+evo")])
            res = Parallel(n_jobs=8)(delayed(_eval_one)(bench, sub, d, m, target, n_boot) for m in cols)
            rows += res
            nb = 200 if bench == "domainome" else n_boot
            for a, c in (("B5evo", "B5"), ("B3+evo", "B3+"), ("B5evo", "FoldX"), ("B5", "FoldX"), ("B5evo", "B3+evo"),
                         ("B5evo", "Evo-only"), ("B5evo", "DDGun3D")):
                if a not in d or c not in d:
                    continue
                both = d[d[a].notna() & d[c].notna()]
                if len(both) < 30:
                    continue
                pb = E.paired_bootstrap(both, a, c, target=target, n_boot=nb)
                pr = {"bench": bench, "subset": sub, "model": a, "vs": c, "n_both": len(both)}
                for k in ("spearman", "spearman_prot_median", "pearson"):
                    if k in pb:
                        pr.update({f"d_{k}": pb[k]["delta"], f"d_{k}_lo": pb[k]["lo"], f"d_{k}_hi": pb[k]["hi"]})
                pairs.append(pr)
            if sub in ("all", "direct", "strict") and bench in ("megascale", "fireprot", "domainome") or (bench, sub) == ("megascale", "natural"):
                for sname, mask in strata_masks(d).items():
                    dd = d[mask]
                    if len(dd) < 200 or dd.protein_id.nunique() < 3:
                        continue
                    for m in ("B5", "B5evo", "Evo-only", "FoldX"):
                        if m in dd:
                            r = _eval_one(bench, sub, dd, m, target, 0)
                            strata.append({"stratum": sname, **r})
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
        pd.DataFrame(pairs).to_csv(OUT / "paired.csv", index=False)
        pd.DataFrame(strata).to_csv(OUT / "strata.csv", index=False)
        if bench == "megascale":
            keep = KS + ["ddg", "domain_origin", "rsa", "evo_found", "evo_neff_site", "evo_neff", "cluster",
                         "homolog_in_megascale_train"] + [m for m in MAIN if m in b] + [c for c in COMPS if c in b]
            b[[c for c in keep if c in b]].to_parquet(OUT / "preds_megascale_test.parquet", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(pairs).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select", "final", "eval", "all"])
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    if a.stage in ("select", "all"):
        stage_select()
    if a.stage in ("final", "all"):
        stage_final()
    if a.stage in ("eval", "all"):
        stage_eval(a.n_boot)


if __name__ == "__main__":
    main()
