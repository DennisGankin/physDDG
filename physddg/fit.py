"""Fitter for few-weight interpretable ddG models over generic feature tables.

Model (inference):  ddg = w0 + sum_k w_k f_k + r[mut] - r[wt],   r = r_prior + delta_shared
Training adds per-dataset nuisance (dropped at inference, decision D7/D8):
  mu_i = a_d * (core_i + delta_d[mut] - delta_d[wt]) + b_d        (reference dataset: a = 1, b = 0)
Losses (per row, then weighted mean):
  gauss  Gaussian / interval-censored (Tobit) NLL, s^2 = sigma_d^2 + sigma_meas_i^2 (sigma_d fitted per dataset);
         point rows use y, interval rows (lo, hi) use log(Phi((hi-mu)/s) - Phi((lo-mu)/s)) (one side may be infinite)
  huber  Huber(delta) on y - mu; one-sided hinge-Huber for interval rows
  + rank_weight * within-protein pairwise logistic surrogate over pairs with |y_i - y_j| > rank_margin
Dataset weights: total weight of dataset d proportional to its number of homology clusters (or sqrt / rows);
within a dataset rows, clusters or proteins (per-domain balancing) weighted equally. Sign constraints via L-BFGS-B bounds. Deterministic.
Uncertainty: bootstrap over clusters (stratified by dataset) as row-weight multiplicities, warm-started refits.

Input table (one row per variant): `dataset`, `cluster`, `protein_id`, `wt`, `mut`, target `y` (NaN allowed for
interval rows), optional `lo`, `hi`, `sigma_meas`, plus numeric feature columns. `prepare_targets` builds y/lo/hi/
sigma_meas from the canonical Megascale / FireProt / generic tables.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .topology import AA1, AA_INDEX

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


@dataclass
class FitConfig:
    features: list
    ref: bool = False                 # fit delta_shared (20 reference energies)
    ref_prior: dict | None = None     # aa -> prior r (kcal/mol); zeros if None
    ref_l2: float = 1e-2              # L2 on delta_shared (on the weighted-mean-loss scale)
    ref_nuisance: tuple = ()          # datasets with their own delta_d (dropped at inference)
    nuis_l2: float = 1e-3             # L2 on delta_d
    scale_offset: bool = True         # per-dataset a_d, b_d for non-reference datasets
    ref_dataset: str | None = None    # dataset with a=1, b=0 (default: first in sorted order)
    loss: str = "gauss"               # gauss | huber
    huber_delta: float = 1.0
    censor: bool = True               # use interval rows (lo/hi); else interval-only rows are dropped
    rank_weight: float = 0.0
    rank_margin: float = 0.5
    rank_pairs: int = 1000            # sampled pairs per protein
    rank_tau: float = 0.5             # logistic temperature (kcal/mol)
    weighting: str = "clusters"       # dataset total weight: clusters | sqrt_clusters | rows
    within: str = "rows"              # within dataset: rows | clusters | proteins (equal weight per cluster / protein)
    signs: dict = field(default_factory=dict)  # feature -> +1 / -1
    l2: float = 1e-6                  # tiny ridge on standardised weights
    intercept: bool = True
    seed: int = 0


# ----------------------------------------------------------------------------- targets
def prepare_targets(df: pd.DataFrame, kind: str | None = None, floor_censor: bool = False) -> pd.DataFrame:
    """Add y / lo / hi / sigma_meas. kind: megascale (CI -> sigma; censored rows -> intervals; optional
    flag_mut_floor rows as lower bounds), fireprot (flag_censored -> lower bound), else point targets from ddg."""
    d = df.copy()
    kind = kind or str(d["dataset"].iloc[0])
    d["y"] = d["ddg"].astype(float)
    d["lo"], d["hi"] = np.nan, np.nan
    d["sigma_meas"] = 0.0
    if kind == "megascale":
        ci_m = d.get("dG_mut_ci95", pd.Series(np.nan, index=d.index)).fillna(0)
        ci_w = d.get("dG_wt_ci95", pd.Series(np.nan, index=d.index)).fillna(0)
        d["sigma_meas"] = np.sqrt((ci_m / 3.92) ** 2 + (ci_w / 3.92) ** 2)
        cens = d.get("censor", pd.Series("none", index=d.index)).fillna("none")
        iv = cens.isin(["mut_unfolded_lt-1", "mut_stable_gt5"])
        d.loc[iv, "lo"], d.loc[iv, "hi"], d.loc[iv, "y"] = d.loc[iv, "ddg_lo"], d.loc[iv, "ddg_hi"], np.nan
        d = d[(cens == "none") | iv]
        if floor_censor and "flag_mut_floor" in d:
            f = d["flag_mut_floor"].fillna(False).astype(bool) & d["y"].notna()
            d.loc[f, "lo"], d.loc[f, "hi"], d.loc[f, "y"] = d.loc[f, "y"], np.inf, np.nan
    elif kind == "fireprot" and "flag_censored" in d:
        f = d["flag_censored"].fillna(False).astype(bool)
        d.loc[f, "lo"], d.loc[f, "hi"], d.loc[f, "y"] = d.loc[f, "y"], np.inf, np.nan
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------- core
class _Problem:
    """Tensors + parameter layout for one fit configuration on one training table."""

    def __init__(self, df: pd.DataFrame, cfg: FitConfig):
        self.cfg = cfg
        d = df.copy()
        if not cfg.censor:
            d = d[d["y"].notna()]
        for c in ("lo", "hi"):
            if c not in d:
                d[c] = np.nan
        if "sigma_meas" not in d:
            d["sigma_meas"] = 0.0
        d = d[d["y"].notna() | d["lo"].notna() | d["hi"].notna()]
        d = d.dropna(subset=list(cfg.features)).reset_index(drop=True)
        self.df = d
        self.datasets = sorted(d["dataset"].astype(str).unique())
        self.refd = cfg.ref_dataset if cfg.ref_dataset in self.datasets else self.datasets[0]
        self.di = d["dataset"].astype(str).map({k: i for i, k in enumerate(self.datasets)}).to_numpy()
        X = d[list(cfg.features)].to_numpy(float)
        self.mu_x = np.zeros(X.shape[1])          # no centring: keep zero-feature = zero-effect meaning
        self.sd_x = np.where(X.std(0) > 0, X.std(0), 1.0)
        T = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
        self.X = T(X / self.sd_x)
        self.iw = T(d["wt"].map(AA_INDEX).to_numpy(), torch.long)
        self.im = T(d["mut"].map(AA_INDEX).to_numpy(), torch.long)
        y = d["y"].to_numpy(float)
        self.point = T(np.isfinite(y), torch.bool)
        self.y = T(np.nan_to_num(y))
        self.lo = T(np.where(self.point.numpy(), -np.inf, d["lo"].fillna(-np.inf).to_numpy(float)))
        self.hi = T(np.where(self.point.numpy(), np.inf, d["hi"].fillna(np.inf).to_numpy(float)))
        self.s2m = T(d["sigma_meas"].fillna(0).to_numpy(float) ** 2)
        self.dit = T(self.di, torch.long)
        self.prior = T([(cfg.ref_prior or {}).get(a, 0.0) for a in AA1])
        self.base_w = self._row_weights()
        self.pairs = self._rank_pairs() if cfg.rank_weight > 0 else None
        # parameter layout: name -> (slice, lower, upper)
        lay, n = {}, 0

        def add(name, size, lo=None, hi=None):
            nonlocal n
            lay[name] = (slice(n, n + size), lo, hi)
            n += size
        K = X.shape[1]
        add("w0", 1 if cfg.intercept else 0)
        add("w", K)
        add("dref", 20 if cfg.ref else 0)
        self.nuis = [ds for ds in cfg.ref_nuisance if ds in self.datasets]
        add("dnuis", 20 * len(self.nuis))
        self.other = [ds for ds in self.datasets if ds != self.refd] if cfg.scale_offset else []
        add("a", len(self.other), 0.05, 20.0)
        add("b", len(self.other))
        add("logsig", len(self.datasets) if cfg.loss == "gauss" else 0, np.log(0.05), np.log(10.0))
        self.lay, self.n = lay, n
        bounds = [(None, None)] * n
        for name, (sl, lo, hi) in lay.items():
            for j in range(sl.start, sl.stop):
                bounds[j] = (lo, hi)
        for k, f in enumerate(cfg.features):
            s = cfg.signs.get(f, 0)
            j = lay["w"][0].start + k
            bounds[j] = (0.0, None) if s > 0 else (None, 0.0) if s < 0 else (None, None)
        self.bounds = bounds

    def _row_weights(self) -> np.ndarray:
        d, cfg = self.df, self.cfg
        w = np.zeros(len(d))
        tot = {}
        for ds, g in d.groupby("dataset"):
            nc = g["cluster"].nunique()
            tot[ds] = {"clusters": nc, "sqrt_clusters": np.sqrt(nc), "rows": len(g)}[cfg.weighting]
        z = sum(tot.values())
        for ds, g in d.groupby("dataset"):
            if cfg.within in ("clusters", "proteins"):
                col = "cluster" if cfg.within == "clusters" else "protein_id"
                cnt = g[col].map(g[col].value_counts())
                w[g.index] = tot[ds] / z / g[col].nunique() / cnt.to_numpy()
            else:
                w[g.index] = tot[ds] / z / len(g)
        return w

    def _rank_pairs(self):
        d, cfg = self.df, self.cfg
        rng = np.random.default_rng(cfg.seed)
        I, J, S = [], [], []
        pt = self.point.numpy()
        for _, g in d[pt].groupby("protein_id"):
            idx, yy = g.index.to_numpy(), g["y"].to_numpy()
            if len(idx) < 2:
                continue
            a = rng.integers(0, len(idx), cfg.rank_pairs * 3)
            b = rng.integers(0, len(idx), cfg.rank_pairs * 3)
            dy = yy[a] - yy[b]
            ok = np.abs(dy) > cfg.rank_margin
            a, b, dy = a[ok][:cfg.rank_pairs], b[ok][:cfg.rank_pairs], dy[ok][:cfg.rank_pairs]
            I.append(idx[a]); J.append(idx[b]); S.append(np.sign(dy))
        if not I:
            return None
        I, J, S = map(np.concatenate, (I, J, S))
        return torch.as_tensor(I), torch.as_tensor(J), torch.as_tensor(S, dtype=torch.float64)

    def get(self, th, name):
        return th[self.lay[name][0]]

    def core(self, th, X=None, iw=None, im=None):
        X = self.X if X is None else X
        iw = self.iw if iw is None else iw
        im = self.im if im is None else im
        out = X @ self.get(th, "w")
        if self.cfg.intercept:
            out = out + self.get(th, "w0")[0]
        r = self.prior + (self.get(th, "dref") if self.cfg.ref else 0.0)
        return out + r[im] - r[iw]

    def mu(self, th):
        m = self.core(th)
        if self.nuis:
            dn = self.get(th, "dnuis").view(len(self.nuis), 20)
            for k, ds in enumerate(self.nuis):
                sel = self.dit == self.datasets.index(ds)
                m = torch.where(sel, m + dn[k][self.im] - dn[k][self.iw], m)
        if self.other:
            a, b = self.get(th, "a"), self.get(th, "b")
            A = torch.ones(len(self.datasets)); B = torch.zeros(len(self.datasets))
            idx = torch.as_tensor([self.datasets.index(ds) for ds in self.other])
            A = A.index_put((idx,), a); B = B.index_put((idx,), b)
            m = A[self.dit] * m + B[self.dit]
        return m

    def loss(self, th, rw):
        cfg = self.cfg
        m = self.mu(th)
        if cfg.loss == "gauss":
            s = torch.sqrt(torch.exp(2 * self.get(th, "logsig"))[self.dit] + self.s2m)
            zp = (self.y - m) / s
            lp = 0.5 * zp ** 2 + torch.log(s) + 0.5 * np.log(2 * np.pi)
            # finite sentinels for missing bounds (avoid 0*NaN gradients through log_ndtr(+-inf))
            hl, hh = torch.isfinite(self.lo), torch.isfinite(self.hi)
            a = torch.where(hl, (torch.nan_to_num(self.lo, neginf=0.0) - m) / s, torch.full_like(m, -30.0))
            b = torch.where(hh, (torch.nan_to_num(self.hi, posinf=0.0) - m) / s, torch.full_like(m, 30.0))
            lb, la = torch.special.log_ndtr(b), torch.special.log_ndtr(a)
            li = -(lb + torch.log1p(-torch.exp(torch.clamp(la - lb, max=-1e-12))))
            l = torch.where(self.point, lp, li)
        else:
            h = cfg.huber_delta
            hub = lambda r: torch.where(r.abs() <= h, 0.5 * r ** 2, h * (r.abs() - 0.5 * h))
            rl = torch.clamp(torch.nan_to_num(self.lo, neginf=-1e9) - m, min=0)
            rh = torch.clamp(m - torch.nan_to_num(self.hi, posinf=1e9), min=0)
            l = torch.where(self.point, hub(self.y - m), hub(rl) + hub(rh))
        rwt = torch.as_tensor(rw)
        obj = (rwt * l).sum() / rwt.sum()
        if self.pairs is not None:
            I, J, S = self.pairs
            c = self.core(th)
            pw = rwt[I] * rwt[J]
            obj = obj + cfg.rank_weight * (pw * torch.nn.functional.softplus(-(c[I] - c[J]) * S / cfg.rank_tau)).sum() / pw.sum()
        obj = obj + cfg.l2 * (self.get(th, "w") ** 2).sum()
        if cfg.ref:
            obj = obj + cfg.ref_l2 * (self.get(th, "dref") ** 2).sum()
        if self.nuis:
            obj = obj + cfg.nuis_l2 * (self.get(th, "dnuis") ** 2).sum()
        return obj

    def init(self):
        th = np.zeros(self.n)
        if self.other:
            th[self.lay["a"][0]] = 1.0
        if self.cfg.loss == "gauss":
            th[self.lay["logsig"][0]] = np.log(1.0)
        return th

    def solve(self, rw=None, x0=None, maxiter=3000):
        rw = self.base_w if rw is None else rw

        def fg(x):
            th = torch.tensor(x, requires_grad=True)
            f = self.loss(th, rw)
            f.backward()
            return float(f.detach()), th.grad.numpy().copy()
        x0 = self.init() if x0 is None else x0
        r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=self.bounds,
                     options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-12, "gtol": 1e-8})
        return r.x, r

    def unpack(self, x) -> dict:
        """Parameters in raw feature units."""
        g = lambda n: x[self.lay[n][0]]
        out = {"intercept": float(g("w0")[0]) if self.cfg.intercept else 0.0,
               "weights": dict(zip(self.cfg.features, (g("w") / self.sd_x).tolist())),
               "ref": dict(zip(AA1, (self.prior.numpy() + (g("dref") if self.cfg.ref else 0.0)).tolist())) if self.cfg.ref else {},
               "nuisance": {"scale": dict(zip(self.other, g("a").tolist())), "offset": dict(zip(self.other, g("b").tolist())),
                            "sigma": dict(zip(self.datasets, np.exp(g("logsig")).tolist())) if self.cfg.loss == "gauss" else {},
                            "delta_ref": {ds: dict(zip(AA1, g("dnuis").reshape(len(self.nuis), 20)[k].tolist()))
                                          for k, ds in enumerate(self.nuis)}}}
        return out

    def boot_weights(self, rng) -> np.ndarray:
        """Row-weight multiplicities from resampling clusters within each dataset."""
        m = np.zeros(len(self.df))
        for _, g in self.df.groupby("dataset"):
            cl = g["cluster"].astype(str).to_numpy()
            u = np.unique(cl)
            cnt = pd.Series(rng.choice(u, len(u))).value_counts()
            m[g.index] = pd.Series(cl).map(cnt).fillna(0).to_numpy()
        return self.base_w * m


def _boot_one(prob, x0, seed):
    torch.set_num_threads(1)
    rw = prob.boot_weights(np.random.default_rng(seed))
    x, _ = prob.solve(rw, x0=x0)
    return x


def fit(df: pd.DataFrame, cfg: FitConfig, n_boot: int = 0, n_jobs: int = 8, name: str = "model",
        provenance: dict | None = None) -> dict:
    """Fit; returns a JSON-able model dict (weights in raw units, bootstrap-over-cluster SEs if n_boot > 0)."""
    prob = _Problem(df, cfg)
    x, res = prob.solve()
    model = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), "git_commit": _git(),
             "config": _jsonable(asdict(cfg)), **prob.unpack(x),
             "converged": bool(res.success), "objective": float(res.fun), "message": str(res.message),
             "train": {ds: {"n_rows": int((g := prob.df[prob.df.dataset == ds]).shape[0]),
                            "n_point": int(g.y.notna().sum()), "n_interval": int(g.y.isna().sum()),
                            "n_proteins": int(g.protein_id.nunique()), "n_clusters": int(g.cluster.nunique()),
                            "total_weight": float(prob.base_w[g.index].sum())} for ds in prob.datasets},
             "provenance": provenance or {}}
    if n_boot:
        from joblib import Parallel, delayed
        xs = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(prob, x, cfg.seed * 100003 + b) for b in range(n_boot))
        ups = [prob.unpack(xb) for xb in xs]
        model["se"] = {"intercept": float(np.std([u["intercept"] for u in ups], ddof=1)),
                       "weights": {f: float(np.std([u["weights"][f] for u in ups], ddof=1)) for f in cfg.features},
                       "ref": {a: float(np.std([u["ref"][a] for u in ups], ddof=1)) for a in AA1} if cfg.ref else {},
                       "n_boot": n_boot, "unit": "cluster (stratified by dataset)"}
        model["boot_weights"] = {f: [u["weights"][f] for u in ups] for f in cfg.features}
    return model


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    """Inference-mode prediction (no dataset nuisance). NaN where a feature is NaN."""
    p = np.full(len(df), float(model["intercept"]))
    for f, w in model["weights"].items():
        p = p + w * df[f].to_numpy(float)
    if model.get("ref"):
        r = model["ref"]
        p = p + df["mut"].map(r).to_numpy(float) - df["wt"].map(r).to_numpy(float)
    return p


# ----------------------------------------------------------------------------- CV
def group_folds(groups, k: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic grouped folds: groups shuffled with `seed`, then greedily balanced by row count."""
    g = pd.Series(np.asarray(groups).astype(str))
    sizes = g.value_counts()
    rng = np.random.default_rng(seed)
    order = sizes.index.to_numpy()[rng.permutation(len(sizes))]
    order = sorted(order, key=lambda c: -sizes[c])
    load, fold = np.zeros(k), {}
    for c in order:
        j = int(np.argmin(load)); fold[c] = j; load[j] += sizes[c]
    return g.map(fold).to_numpy()


def cross_validate(df: pd.DataFrame, cfg: FitConfig, k: int = 5, seed: int = 0, group_col: str = "cluster") -> pd.DataFrame:
    """Grouped K-fold OOF inference-mode predictions (column `oof`, `fold`) for every row of df."""
    folds = group_folds(df[group_col], k, seed)
    out = df.copy()
    out["oof"], out["fold"] = np.nan, folds
    for j in range(k):
        m = fit(df[folds != j], cfg)
        te = folds == j
        out.loc[te, "oof"] = predict(m, df[te])
    return out


def cv_metrics(oof: pd.DataFrame, by_fold: bool = True) -> dict:
    """Per-protein median Spearman / pooled Spearman / Pearson / RMSE on point rows; mean +- sem over folds."""
    from .eval import evaluate
    pt = oof[oof["y"].notna() & oof["oof"].notna()].assign(ddg=lambda d: d["y"])
    keys = ("spearman_prot_median", "spearman", "pearson", "rmse")
    res = {k: evaluate(pt, "oof")[k] for k in keys}
    if by_fold:
        per = pd.DataFrame([{k: evaluate(g, "oof")[k] for k in keys} for _, g in pt.groupby("fold")])
        res.update({f"{k}_sem": float(per[k].std(ddof=1) / np.sqrt(len(per))) for k in keys})
    return res


# ----------------------------------------------------------------------------- io
def _git() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_model(model: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(model), indent=1))


def load_model(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
