#!/usr/bin/env python
"""Residual analysis, step 3: test candidate interpretable features added to B5 (linear, physddg/fit.py).

Protocol (train/val only; no test rows): B5 configuration (selection.json opts_s2, Huber + censoring, 20 shrunk
reference energies, physics sign constraints) + candidate column(s) from physddg.features_extra.candidate_features.
Scores: grouped 5-fold CV on Megascale-train (same cluster folds as fit.cross_validate, seed 0) -> OOF per-protein
median Spearman (natural domains, the selection metric of scripts/52) and per fold; fit on all train -> Megascale-val
natural per-protein median Spearman. Acceptance rule (pre-set): val natural gain >= 0.004 AND CV natural per-protein
rho improved in >= 4/5 folds. Greedy forward selection over passing candidates (best val+CV first), re-testing the rest.
"scale:<flag>" candidates multiply the fold's fitted B5 core (sum_k w_k f_k, no reference energies) by a context flag
(one extra weight each = context-specific rescaling of all existing terms).
Outputs results/residual_analysis/candidates_round*.csv, forward.csv, final_features.json.
  PYTHONPATH=. .venv/bin/python scripts/93_candidate_features.py
"""
import importlib.util
import json
import os
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
from physddg import features_extra as X  # noqa: E402
from physddg import fit as FT  # noqa: E402

OUT = ROOT / "results" / "residual_analysis"
K = 5
SIGN_POS = {"helix_prop_int", "helix_prop_all", "helix_prop_exp", "beta_prop", "beta_prop_centre", "beta_prop_exp",
            "ncap_pref", "ncap_any", "helix_dipole", "ccap_gly_loss", "to_gly_ss", "overpack_bur", "overpack_strand_bur",
            "overpack_dense", "rep_lin20_bur", "rep_log_strand", "rep_log_bur", "cavity_nonvoid", "catpi_loss",
            "aro_stack_loss", "saltbridge_loss", "charge_like_env", "schb_loss", "schb_loss_any", "scsc_loss", "bur_dhyd", "bur2_dhyd", "polar_burial"}
SINGLE = ["helix_prop_int", "helix_prop_all", "helix_prop_exp", "beta_prop", "beta_prop_centre", "beta_prop_exp",
          "ncap_pref", "ncap_any", "helix_dipole", "ccap_gly_loss", "to_gly_ss", "to_gly_loop", "from_gly_nonposphi",
          "pro_posphi_exp", "to_pro_strand", "to_pro_helix_int", "from_pro_any", "overpack_bur", "overpack_strand_bur",
          "overpack_dense", "rep_lin20_bur", "rep_log_strand", "rep_log_bur", "cavity_nonvoid", "catpi_loss",
          "aro_stack_loss", "saltbridge_loss", "charge_like_env", "schb_loss", "schb_loss_any", "site_ncb14",
          "site_hse_up", "site_natom8", "site_void", "site_lowplddt", "site_term3", "site_term_dist", "site_hbsc",
          "site_ncap", "site_strand_centre", "site_scsc", "scsc_loss", "scsc_bur", "site_hbsc_bur",
          "bur_dhyd", "bur2_dhyd", "polar_burial"]
SCALES = ["flag_helix", "flag_strand", "flag_buried", "flag_term3", "flag_lowplddt", "flag_loop"]
GROUPS = {"scale:helix+strand": ["scale:flag_helix", "scale:flag_strand"],
          "scale:buried+loop": ["scale:flag_buried", "scale:flag_loop"]}


def load52():
    spec = importlib.util.spec_from_file_location("lp52", ROOT / "scripts" / "52_ladder_physics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L52 = load52()
SEL = L52.load_sel()
OPTS = SEL["opts_s2"]
BASE_FEATS, BASE_EXTRA = L52.feature_set("full", OPTS)


def cfg_for(feats):
    c = L52.make_cfg(feats, SEL, BASE_EXTRA)
    for f in feats:
        if f in SIGN_POS:
            c.signs[f] = 1
    return c


def load_tables():
    d = pd.read_parquet(OUT / "base.parquet")
    cf = X.candidate_features(d)
    d = pd.concat([d.drop(columns=[c for c in cf.columns if c in d.columns]), cf], axis=1)
    assert not d.columns.duplicated().any()
    tr, va = d[d.part == "train"].reset_index(drop=True), d[d.part == "val"].reset_index(drop=True)
    return tr, va


def core_of(model, df):
    c = np.zeros(len(df))
    for f, w in model["weights"].items():
        c += w * df[f].to_numpy(float)
    return c


class Evaluator:
    def __init__(self, tr, va):
        self.tr, self.va = tr, va
        self.folds = FT.group_folds(tr["cluster"], K, 0)
        self.nat_tr = (tr.domain_origin == "natural").to_numpy()
        self._core = None

    def base_cores(self):
        """B5 core per fold (fitted on fold-train) for train rows, and full-train core for val rows."""
        if self._core is None:
            res = Parallel(n_jobs=K + 1)(delayed(_fit)(self.tr[self.folds != j] if j < K else self.tr, BASE_FEATS) for j in range(K + 1))
            core_tr = np.zeros((K, len(self.tr)))
            for j in range(K):
                core_tr[j] = core_of(res[j], self.tr)
            self._core = (core_tr, core_of(res[K], self.tr), core_of(res[K], self.va))
        return self._core

    def frames(self, extra, j):
        """Train/eval frames for fold j (j = K: full train -> val) with scale columns materialised."""
        sc = [e for e in extra if e.startswith("scale:")]
        if j < K:
            a, b = self.tr[self.folds != j].copy(), self.tr[self.folds == j].copy()
        else:
            a, b = self.tr.copy(), self.va.copy()
        if sc:
            core_tr, core_full_tr, core_va = self.base_cores()
            ca = core_tr[j][self.folds != j] if j < K else core_full_tr
            cb = core_tr[j][self.folds == j] if j < K else core_va
            for s in sc:
                fl = s.split(":", 1)[1]
                a[s], b[s] = ca * a[fl].to_numpy(float), cb * b[fl].to_numpy(float)
        return a, b


def _fit(df, feats):
    import torch
    torch.set_num_threads(2)
    return FT.fit(df, cfg_for(feats))


def rho_prot(df, col):
    return E.evaluate(df.assign(ddg=df.y), col)["spearman_prot_median"]


EV = None
SLIM = {"path": OUT / "_slim_tables.pkl"}     # scratch (git-ignored): slim train/val frames + base cores for worker processes


def _worker_ev(path):
    global EV
    if EV is None:
        import pickle
        tr, va, cores = pickle.loads(Path(path).read_bytes())
        EV = Evaluator(tr, va)
        EV._core = cores
    return EV


def _run(args, path):
    return run_candidate(_worker_ev(path), *args)


def run_candidate(ev, label, extra, base_extra=()):
    import torch
    torch.set_num_threads(2)
    feats = list(BASE_FEATS) + list(base_extra) + list(extra)
    oof = np.full(len(ev.tr), np.nan)
    for j in range(K):
        a, b = ev.frames(list(base_extra) + list(extra), j)
        m = FT.fit(a, cfg_for(feats))
        oof[ev.folds == j] = FT.predict(m, b)
    a, b = ev.frames(list(base_extra) + list(extra), K)
    m = FT.fit(a, cfg_for(feats))
    pv = FT.predict(m, b)
    t = ev.tr.assign(oof=oof)
    pt = t[t.y.notna()]
    nat = pt[pt.domain_origin == "natural"]
    v = b.assign(pred=pv)
    v = v[v.y.notna()]
    r = {"label": label, "extra": "+".join(extra), "n_new_w": len(extra),
         "cv_nat_prot": rho_prot(nat, "oof"), "cv_all_prot": rho_prot(pt, "oof"),
         "cv_nat_fold": [rho_prot(g, "oof") for _, g in nat.groupby(ev.folds[nat.index.to_numpy()])],
         "val_nat_prot": rho_prot(v[v.domain_origin == "natural"], "pred"), "val_all_prot": rho_prot(v, "pred"),
         "val_prot_rho": {p: float(g.pred.corr(g.y, method="spearman")) for p, g in v.groupby("protein_id")},
         "val_nat_mean": float(np.mean([g.pred.corr(g.y, method="spearman") for _, g in v[v.domain_origin == "natural"].groupby("protein_id")])),
         "cv_pearson": float(np.corrcoef(pt.oof, pt.y)[0, 1]),
         "weights": {f: round(m["weights"][f], 4) for f in extra if f in m["weights"]}}
    return r


def score_round(ev, cands, base_extra, ref):
    res = Parallel(n_jobs=min(24, len(cands)))(delayed(_run)((lab, ex, base_extra), str(SLIM["path"])) for lab, ex in cands)
    t = pd.DataFrame(res)
    t["d_val_nat"] = t.val_nat_prot - ref["val_nat_prot"]
    t["d_val_all"] = t.val_all_prot - ref["val_all_prot"]
    t["d_cv_nat"] = t.cv_nat_prot - ref["cv_nat_prot"]
    t["d_cv_all"] = t.cv_all_prot - ref["cv_all_prot"]
    t["folds_up"] = [int(np.sum(np.array(f) > np.array(ref["cv_nat_fold"]) + 1e-9)) for f in t.cv_nat_fold]
    t["d_val_nat_mean"] = t.val_nat_mean - ref["val_nat_mean"]
    t["val_prot_up"] = [int(sum(r[p] > ref["val_prot_rho"][p] + 1e-9 for p in r)) for r in t.val_prot_rho]
    t["pass"] = (t.d_val_nat >= 0.004) & (t.folds_up >= 4)
    t["pass_relaxed"] = (t[["d_val_nat", "d_val_all", "d_val_nat_mean"]].max(axis=1) >= 0.004) & (t.folds_up >= 4) & (t.d_cv_nat >= 0.004)
    t["score"] = t.d_val_nat + t.d_cv_nat
    return t.sort_values("score", ascending=False)


def main():
    global EV
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="strict", choices=["strict", "relaxed"])
    rule = ap.parse_args().rule
    passcol = "pass" if rule == "strict" else "pass_relaxed"
    tr, va = load_tables()
    keep = set(BASE_FEATS) | set(SINGLE) | set(SCALES) | {"y", "lo", "hi", "sigma_meas", "dataset", "cluster", "protein_id",
                                                          "wt", "mut", "domain_origin", "chain", "pdb_resnum", "site"}
    tr, va = tr[[c for c in tr.columns if c in keep]], va[[c for c in va.columns if c in keep]]
    ev = EV = Evaluator(tr, va)
    ev.base_cores()
    import pickle
    SLIM["path"] = OUT / f"_slim_tables_{rule}.pkl"
    SLIM["path"].write_bytes(pickle.dumps((tr, va, ev._core)))
    cands = [(c, [c]) for c in SINGLE] + [(f"scale:{s}", [f"scale:{s}"]) for s in SCALES] + list(GROUPS.items())
    base_extra, rows = [], []
    ref = run_candidate(ev, "B5 (base)", [])
    print("BASE", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ref.items() if k != "val_prot_rho"}, flush=True)
    rows.append({**ref, "round": 0, "accepted": True})
    rnd = 1
    while True:
        t = score_round(ev, cands, base_extra, ref)
        t["round"] = rnd
        t.to_csv(OUT / f"candidates_{rule}_round{rnd}.csv", index=False)
        show = ["label", "n_new_w", "val_nat_prot", "d_val_nat", "d_val_all", "d_val_nat_mean", "val_prot_up", "cv_nat_prot", "d_cv_nat",
                "d_cv_all", "folds_up", "pass", "pass_relaxed", "weights"]
        print(f"\n== round {rnd} (base + {base_extra})\n", t[show].round(4).to_string(index=False), flush=True)
        ok = t[t[passcol]]
        if not len(ok):
            break
        best = ok.iloc[0]
        new = [x for x in best.extra.split("+")]
        base_extra = base_extra + new
        ref = run_candidate(ev, f"+{best.label}", [], base_extra)
        rows.append({**ref, "round": rnd, "accepted": True, "added": best.label})
        print("ACCEPT", best.label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ref.items() if k != "val_prot_rho"}, flush=True)
        added = set(new)
        cands = [(lab, ex) for lab, ex in cands if not (set(ex) & added)]
        rnd += 1
        if rnd > 8:
            break
    pd.DataFrame(rows).drop(columns=["val_prot_rho"]).to_csv(OUT / f"forward_{rule}.csv", index=False)
    (OUT / f"final_features_{rule}.json").write_text(json.dumps({"base": BASE_FEATS, "added": base_extra, "rule": rule}, indent=1))
    print("FINAL added:", base_extra)


if __name__ == "__main__":
    main()
