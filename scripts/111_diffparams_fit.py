#!/usr/bin/env python
"""Fit physical parameters theta (physddg/diffparams.py, 32 log-scale multipliers around the literature values)
end to end through the differentiable Stage-1 site scan + Boltzmann averaging, with the B6gs1 linear model
solved exactly for every theta (variable projection; envelope-theorem gradient).

  PYTHONPATH=. .venv/bin/python scripts/111_diffparams_fit.py --data megascale --lams 0.01 --folds none,0,1 --gpus 0

Objective (B6gs1 conventions, scripts/52 + physddg/fit.py): Megascale-train, Huber + interval (censored) loss,
row weights per dataset (equal variants per domain in the subsample = per-domain balancing), 20 shrunk reference
energies (D7, ref_l2), sign constraints, the B6g feature groups for the non-energy features (unchanged columns).
  L(theta) = min_w loss(w; X(theta)) + lam * sum theta^2
Training data: 'site-balanced' subsample of Megascale-train: per domain up to N_SITES random sites x up to
PER_SITE random variants at each (all 239 domains). Folds: grouped by homology cluster (fit.group_folds, seed 0,
the folds of physddg.fit.cross_validate) on the subsample; for fold j theta is fitted on the other folds and the
held-out subsample is monitored every iteration (early stopping is chosen later from these traces by CV only).
--data fireprot fits theta on FireProt-train only (all rows; G6-style transfer test of the physics).
--boot B: B cluster-bootstrap refits (row-weight multiplicities, warm start from --theta0) for SEs.
Outputs: results/diff_params/fits/<data>_lam<lam>_<fold|full|bootK>.json (theta trajectory, objective, held-out
monitor, final weights). Logs in logs/111_*.log.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from physddg import diffparams as D  # noqa: E402
from physddg import eval as EV  # noqa: E402
from physddg import fit as FT  # noqa: E402
from physddg.topology import AA_INDEX  # noqa: E402

_spec = importlib.util.spec_from_file_location("l52", ROOT / "scripts" / "52_ladder_physics.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

OUT = ROOT / "results" / "diff_params"
N_SITES, PER_SITE = 20, 3
THCOLS = ["e_lj_atr", "e_lj_rep", "e_solv_apolar", "e_solv_polar", "e_hb_sc_bb", "e_hb_sc_sc", "e_hb_bb_bb",
          "e_elec_dh", "dTS", "dprior_avg", "dref_apolar", "dref_polar"]
KT = 0.593                     # predict.KT (dTS = KT * dS_site)


def model_setup():
    sel = M.load_sel()
    o = dict(sel["opts_s1"], cxxc=False)
    feats, extra = M.feature_set("b6g", o)
    assert set(THCOLS) <= set(feats)
    return sel, o, feats, extra


def subsample(df: pd.DataFrame, n_sites=N_SITES, per_site=PER_SITE, seed=0) -> pd.DataFrame:
    """Site-balanced subsample: per protein up to n_sites random sites, up to per_site random rows each."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("protein_id", sort=True):
        sites = g.groupby(["chain", "pdb_resnum"]).indices
        keys = sorted(sites)
        pick = rng.choice(len(keys), min(n_sites, len(keys)), replace=False)
        for k in pick:
            ix = sites[keys[k]]
            keep.extend(g.index[rng.choice(ix, min(per_site, len(ix)), replace=False)])
    return df.loc[sorted(keep)].reset_index(drop=True)


def row_features(Gcat: torch.Tensor, gw: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
    """Model columns THCOLS from group features (D.GCOLS order) = scripts/52.derive (slog_rep) of the s1 columns."""
    d = Gcat[gm] - Gcat[gw]
    nT = len(D.TERMS)
    rep = d[:, 1:2]
    return torch.cat([d[:, :1], torch.sign(rep) * torch.log1p(rep.abs()), d[:, 2:nT], KT * d[:, nT:nT + 1],
                      d[:, nT + 1:nT + 4]], 1)


class Objective:
    """Variable-projection objective over theta for one training table (rows of `sub`)."""

    def __init__(self, sub: pd.DataFrame, pool: D.ThetaPool, feats, extra, sel):
        self.sub, self.pool, self.feats = sub, pool, feats
        self.cfg = M.make_cfg(feats, sel, extra)
        keys = list(pool.info)
        self.keys = keys
        off, gidx = 0, {}
        self.slices = {}
        for k in keys:
            for i, (c, r, a) in enumerate(pool.info[k]):
                gidx[(k, c, r, a)] = off + i
            self.slices[k] = slice(off, off + len(pool.info[k]))
            off += len(pool.info[k])
        self.ng = off
        st = sub.structure.astype(str).to_numpy()
        ch, rn = sub.chain.astype(str).to_numpy(), sub.pdb_resnum.astype(str).to_numpy()
        self.gw = torch.as_tensor([gidx[(s, c, r, w)] for s, c, r, w in zip(st, ch, rn, sub.wt)])
        self.gm = torch.as_tensor([gidx[(s, c, r, m)] for s, c, r, m in zip(st, ch, rn, sub.mut)])
        self.cache = {}

    def G(self, theta):
        key = tuple(np.round(np.asarray(theta, float), 10))
        if key not in self.cache:
            g = self.pool.G(theta)
            self.cache = {key: torch.as_tensor(np.concatenate([g[k] for k in self.keys]))}
        return self.cache[key]

    def table(self, theta) -> pd.DataFrame:
        X = row_features(self.G(theta), self.gw, self.gm).numpy()
        return self.sub.assign(**{c: X[:, j] for j, c in enumerate(THCOLS)})

    def value_grad(self, theta, train: np.ndarray, lam: float, mult: np.ndarray | None = None, x0=None):
        """Objective (training rows `train` bool mask) and its theta-gradient; returns (f, g, x*, prob)."""
        Gc = self.G(theta).clone().requires_grad_(True)
        X = row_features(Gc, self.gw, self.gm)
        d = self.sub.assign(**{c: X[:, j].detach().numpy() for j, c in enumerate(THCOLS)})[train].reset_index(drop=True)
        prob = FT._Problem(d, self.cfg)
        assert len(prob.df) == int(train.sum()), "rows dropped by the fitter"
        rw = prob.base_w * (mult[train] if mult is not None else 1.0)
        x, _ = prob.solve(rw, x0=x0)
        th = torch.tensor(x, dtype=torch.float64)
        Xs = prob.X.clone().requires_grad_(True)
        prob.X = Xs
        loss = prob.loss(th, rw)
        (gX,) = torch.autograd.grad(loss, Xs)
        cols = [self.feats.index(c) for c in THCOLS]
        gXraw = torch.zeros_like(X, dtype=torch.float64)
        gXraw[torch.as_tensor(np.flatnonzero(train))] = (gX[:, cols] / torch.as_tensor(prob.sd_x[cols]))
        (gG,) = torch.autograd.grad(X, Gc, grad_outputs=gXraw.float())
        tr_keys = set(self.sub.structure[train].astype(str))
        cot = {k: gG[self.slices[k]].numpy() for k in self.keys if k in tr_keys}
        g = self.pool.vjp(theta, cot)
        pen, gpen = D.penalty(theta, lam)
        prob.X = Xs.detach()
        return float(loss.detach()) + pen, g + gpen, x, prob


def heldout_metrics(obj: Objective, theta, prob, x, mask) -> dict:
    """Held-out subsample rows (inference mode, no nuisance) with the training-fold weights."""
    if mask.sum() == 0:
        return {}
    m = prob.unpack(x)
    m = {"intercept": m["intercept"], "weights": m["weights"], "ref": m["ref"]}
    d = obj.table(theta)[mask]
    d = d[d.y.notna()].assign(ddg=lambda t: t.y)
    d = d.assign(pred=FT.predict(m, d))
    r = EV.evaluate(d, "pred")
    nat = d[d.domain_origin == "natural"]
    rn = EV.evaluate(nat, "pred") if len(nat) else {"spearman_prot_median": np.nan}
    return {"ho_prot": r["spearman_prot_median"], "ho_nat_prot": rn["spearman_prot_median"], "ho_pooled": r["spearman"],
            "ho_pearson": r["pearson"]}


def run_fit(obj: Objective, train, lam, label, theta0=None, mult=None, maxiter=60, bound=1.0, log=None):
    log = log or (lambda s: print(s, flush=True))
    theta0 = np.zeros(D.NP) if theta0 is None else np.asarray(theta0, float)
    state = {"x0": None, "n_eval": 0, "last": None}
    traj = []
    t0 = time.time()

    def fg(theta):
        f, g, x, prob = obj.value_grad(theta, train, lam, mult, state["x0"])
        state["x0"], state["n_eval"], state["last"] = x, state["n_eval"] + 1, (theta.copy(), x, prob, f)
        return f, g

    def cb(theta):
        th_, x, prob, f = state["last"]
        if not np.allclose(th_, theta):
            f, _, x, prob = obj.value_grad(theta, train, lam, mult, state["x0"])
        ho = heldout_metrics(obj, theta, prob, x, ~train)
        rec = {"iter": len(traj) + 1, "n_eval": state["n_eval"], "obj": f, "pen": D.penalty(theta, lam)[0],
               "theta": np.asarray(theta).tolist(), "sec": time.time() - t0, **ho}
        traj.append(rec)
        log(f"[{label}] it {rec['iter']:3d} eval {rec['n_eval']:3d} obj {f:.6f} pen {rec['pen']:.5f} "
            + " ".join(f"{k} {v:.4f}" for k, v in ho.items()) + f" |th|max {np.abs(theta).max():.3f} {rec['sec']:.0f}s")

    f0, g0, x0, prob0 = obj.value_grad(theta0, train, lam, mult, None)
    ho0 = heldout_metrics(obj, theta0, prob0, x0, ~train)
    traj.append({"iter": 0, "n_eval": 1, "obj": f0, "pen": D.penalty(theta0, lam)[0], "theta": theta0.tolist(),
                 "sec": time.time() - t0, **ho0})
    log(f"[{label}] it   0 obj {f0:.6f} " + " ".join(f"{k} {v:.4f}" for k, v in ho0.items()))
    state["x0"] = x0
    res = minimize(fg, theta0, jac=True, method="L-BFGS-B", bounds=[(-bound, bound)] * D.NP, callback=cb,
                   options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxcor": 20})
    th = res.x
    f, g, x, prob = obj.value_grad(th, train, lam, mult, state["x0"])
    return {"label": label, "lam": lam, "theta": th.tolist(), "obj": f, "grad_norm": float(np.linalg.norm(g)),
            "message": str(res.message), "n_iter": int(res.nit), "n_eval": state["n_eval"], "trajectory": traj,
            "weights": prob.unpack(x), "names": D.NAMES, "sec": time.time() - t0}


def boot_mult(sub: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros(len(sub))
    for _, g in sub.groupby("dataset"):
        cl = g["cluster"].astype(str).to_numpy()
        u = np.unique(cl)
        cnt = pd.Series(rng.choice(u, len(u))).value_counts()
        m[g.index] = pd.Series(cl).map(cnt).fillna(0).to_numpy()
    return m


def training_table(data: str, seed: int = 0) -> pd.DataFrame:
    tr, _, fp = M.train_tables()
    sel, o, feats, extra = model_setup()
    if data == "megascale":
        base = tr
    elif data == "fireprot":
        base = fp
    else:
        raise ValueError(data)
    d = M.derive(base, 1, o)
    fixed = [f for f in feats if f not in THCOLS]
    ok = d[fixed].notna().all(1) & (d.y.notna() | d.lo.notna() | d.hi.notna())
    d = d[ok].reset_index(drop=True)
    return subsample(d, seed=seed) if data == "megascale" else d


def fold_map(data: str, sub: pd.DataFrame) -> np.ndarray:
    """Fold of each subsample row = fold of its cluster in fit.cross_validate on the FULL training table
    (group_folds over all rows of derive(train), seed 0), so CV on full rows never sees a theta fitted on that fold."""
    tr, _, fp = M.train_tables()
    full = tr if data == "megascale" else fp
    f = FT.group_folds(full["cluster"], 5, 0)
    cmap = dict(zip(full["cluster"].astype(str), f))
    return sub["cluster"].astype(str).map(cmap).to_numpy()


def jobs_for(sub: pd.DataFrame) -> list:
    jobs = []
    for st, g in sub.groupby("structure", sort=True):
        groups = {}
        for c, r, w, m in zip(g.chain.astype(str), g.pdb_resnum.astype(str), g.wt, g.mut):
            groups.setdefault((c, r), set()).update([w, m])
        path = st if Path(st).is_absolute() else str(ROOT / st)
        jobs.append((st, path, groups))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="megascale", choices=["megascale", "fireprot"])
    ap.add_argument("--lams", default="0.01")
    ap.add_argument("--folds", default="none", help="comma list of fold ids and/or 'none' (= all training rows)")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--boot-start", type=int, default=0)
    ap.add_argument("--theta0", default="", help="fit json whose theta warm-starts bootstrap refits")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--check", action="store_true", help="only verify theta=0 subsample columns vs scripts/52.derive")
    a = ap.parse_args()
    (OUT / "fits").mkdir(parents=True, exist_ok=True)
    sel, o, feats, extra = model_setup()
    sub = training_table(a.data)
    folds = fold_map(a.data, sub)
    print(f"{a.data}: {len(sub)} rows, {sub.protein_id.nunique()} proteins, {sub.cluster.nunique()} clusters, "
          f"{sub.groupby(['structure', 'chain', 'pdb_resnum']).ngroups} sites", flush=True)
    t0 = time.time()
    pool = D.ThetaPool(jobs_for(sub), [int(g) for g in a.gpus.split(",")])
    print(f"pool ready in {time.time() - t0:.0f}s, {sum(len(v) for v in pool.info.values())} groups", flush=True)
    obj = Objective(sub, pool, feats, extra, sel)
    try:
        if a.check:
            t = obj.table(np.zeros(D.NP))
            for c in THCOLS:
                print(c, float(np.abs(t[c] - sub[c]).max()), float(np.abs(sub[c]).max()))
            t1 = time.time()
            f, g, _, _ = obj.value_grad(np.zeros(D.NP), np.ones(len(sub), bool), 0.0)
            print("obj", f, "grad", dict(zip(D.NAMES, np.round(g, 5))), f"{time.time() - t1:.1f}s")
            return
        lams = [float(x) for x in a.lams.split(",")]
        if a.boot:
            base = json.loads(Path(a.theta0).read_text()) if a.theta0 else None
            for lam in lams:
                for b in range(a.boot_start, a.boot_start + a.boot):
                    label = f"{a.data}_lam{lam:g}_boot{b}"
                    r = run_fit(obj, np.ones(len(sub), bool), lam, label, theta0=base["theta"] if base else None,
                                mult=boot_mult(sub, 1000 + b), maxiter=a.maxiter)
                    (OUT / "fits" / f"{label}.json").write_text(json.dumps(r))
            return
        for lam in lams:
            for fs in a.folds.split(","):
                train = np.ones(len(sub), bool) if fs == "none" else folds != int(fs)
                label = f"{a.data}_lam{lam:g}_" + ("full" if fs == "none" else f"fold{fs}")
                r = run_fit(obj, train, lam, label, maxiter=a.maxiter)
                r["fold"] = None if fs == "none" else int(fs)
                (OUT / "fits" / f"{label}.json").write_text(json.dumps(r))
                print(f"saved {label}: obj {r['obj']:.6f} iters {r['n_iter']} evals {r['n_eval']} {r['sec']:.0f}s",
                      flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
