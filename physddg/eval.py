"""Evaluation harness: load benchmarks, merge predictions, metrics, bootstrap, antisymmetry.

Conventions (CONVENTIONS.md): ddg in kcal/mol, positive = destabilising. A prediction must follow the
same sign (flip it before merging otherwise). Datasets with only a proxy `score` (lower = destabilising,
e.g. Domainome fitness) are scored against -score (target="score").
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("protein_id", "chain", "pdb_resnum", "wt", "mut")
POOLED = ("spearman", "pearson", "rmse", "mae", "rmse_fit", "mae_fit", "auroc_destab", "auroc_stab", "sign_acc")
PROT = ("spearman_prot_median", "spearman_prot_mean")
ALL_METRICS = ("n", "n_proteins", "n_prot_eval") + POOLED + PROT


# ----------------------------------------------------------------------------- loading / merging
def load_benchmark(name: str, root: Path = ROOT) -> pd.DataFrame:
    base = Path(root) / "data" / "processed"
    for ext, rd in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = base / f"{name}{ext}"
        if p.exists():
            df = rd(p)
            df["pdb_resnum"] = _norm_resnum(df["pdb_resnum"])
            return df
    raise FileNotFoundError(f"no {base}/{name}.parquet|csv")


def _norm_resnum(s: pd.Series) -> pd.Series:
    """String resnum, '4.0' -> '4' (float-typed columns), keeps insertion codes ('52A')."""
    return s.astype(str).str.strip().str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _norm(df: pd.DataFrame, keys) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        out[k] = _norm_resnum(out[k]) if k == "pdb_resnum" else out[k].astype(str).str.strip()
    return out


def _shift_resnum(s: pd.Series, k: int) -> pd.Series:
    m = s.str.extract(r"^(-?\d+)(.*)$")
    return (m[0].astype(float) + k).astype("Int64").astype(str) + m[1]


def _coverage(bench: pd.DataFrame, pred: pd.DataFrame, keys, col) -> float:
    pk = pred.loc[pred[col].notna(), list(keys)].drop_duplicates()
    hit = bench[list(keys)].merge(pk, on=list(keys), how="left", indicator=True)["_merge"] == "both"
    return float(hit.mean())


def merge_predictions(bench: pd.DataFrame, pred: pd.DataFrame, pred_col: str = "ddg_pred", keys=KEYS,
                      out_col: str = "pred", min_coverage: float = 0.99, shifts=range(-3, 4)) -> pd.DataFrame:
    """Inner-join predictions onto benchmark rows; coverage info in result.attrs['merge_info'].

    Warns loudly if < min_coverage of benchmark rows get a (non-NaN) prediction, and tests
    +-k shifts of integer pdb_resnum to flag a likely position-offset bug (never applied silently).
    """
    keys = tuple(keys)
    b, p = _norm(bench, keys), _norm(pred[list(keys) + [pred_col]], keys)
    ndup = int(p.duplicated(list(keys)).sum())
    if ndup:
        warnings.warn(f"{ndup} duplicate prediction keys; keeping first")
        p = p.drop_duplicates(list(keys))
    cov = _coverage(b, p, keys, pred_col)
    info = {"n_bench": len(b), "n_pred": len(p), "coverage": cov, "dup_pred_keys": ndup, "offset": None}
    if cov < min_coverage:
        msg = f"LOW COVERAGE: only {cov:.1%} of {len(b)} benchmark rows matched a prediction (need >= {min_coverage:.0%})."
        if "pdb_resnum" in keys:
            best = max((k for k in shifts if k), key=lambda k: _coverage(
                b, p.assign(pdb_resnum=_shift_resnum(p["pdb_resnum"], k)), keys, pred_col))
            cov_k = _coverage(b, p.assign(pdb_resnum=_shift_resnum(p["pdb_resnum"], best)), keys, pred_col)
            if cov_k > cov + 0.01:
                info["offset"] = {"shift": best, "coverage_shifted": cov_k}
                msg += (f" LIKELY POSITION OFFSET: shifting prediction pdb_resnum by {best:+d} gives "
                        f"{cov_k:.1%} coverage vs {cov:.1%}.")
        print("!" * 8, msg, file=sys.stderr)
        warnings.warn(msg)
    p = p.rename(columns={pred_col: out_col})
    m = b.merge(p, on=list(keys), how="inner")
    m[out_col] = pd.to_numeric(m[out_col], errors="coerce")
    m = m[m[out_col].notna()].reset_index(drop=True)
    m.attrs["merge_info"] = info
    return m


# ----------------------------------------------------------------------------- subsets
_OPS = [("==", np.equal), ("!=", np.not_equal), (">=", np.greater_equal), ("<=", np.less_equal),
        (">", np.greater), ("<", np.less)]


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return float(v)
    except ValueError:
        return v


def apply_subset(df: pd.DataFrame, expr: str | None) -> pd.DataFrame:
    """Filter rows with 'col==val', 'col!=val', 'col>=3', ... (several joined by ',' = AND)."""
    if not expr:
        return df
    mask = np.ones(len(df), bool)
    for term in expr.split(","):
        for sym, fn in _OPS:
            if sym in term:
                col, val = (x.strip() for x in term.split(sym, 1))
                v = _coerce(val)
                c = df[col]
                if isinstance(v, bool) or c.dtype == bool:
                    mask &= fn(c.astype(bool).to_numpy(), bool(v))
                elif isinstance(v, float) and pd.api.types.is_numeric_dtype(c):
                    mask &= fn(c.to_numpy(float), v)
                else:
                    mask &= fn(c.astype(str).to_numpy(), val)
                break
        else:
            raise ValueError(f"bad subset term {term!r}")
    return df[mask].reset_index(drop=True)


def add_bins(df: pd.DataFrame, col: str, edges, name: str | None = None) -> pd.DataFrame:
    """Add categorical column `<col>_bin` (e.g. pLDDT bins) with right-open intervals [e_i, e_i+1)."""
    edges = [-np.inf, *edges, np.inf]
    lab = [f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)]
    out = df.copy()
    out[name or f"{col}_bin"] = pd.cut(out[col], edges, right=False, labels=lab).astype(str)
    return out


# ----------------------------------------------------------------------------- metrics
def _safe(fn, a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(fn(a, b)[0])


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC, ties averaged; NaN if a class is empty."""
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _pooled(y: np.ndarray, p: np.ndarray, target: str, min_pos: int = 10) -> dict:
    """Pooled metrics. y is ddg (target='ddg') or -score (target='score', rank metrics only)."""
    out = dict.fromkeys(POOLED, np.nan)
    out["spearman"] = _safe(spearmanr, p, y)
    out["pearson"] = _safe(pearsonr, p, y)
    if target != "ddg":
        return out
    out["rmse"] = float(np.sqrt(np.mean((p - y) ** 2)))
    out["mae"] = float(np.mean(np.abs(p - y)))
    if len(y) > 2 and np.ptp(p) > 0:  # in-sample linear recalibration y ~ a*p + b (optimistic; reported separately)
        a, b = np.polyfit(p, y, 1)
        res = y - (a * p + b)
        out["rmse_fit"], out["mae_fit"] = float(np.sqrt(np.mean(res**2))), float(np.mean(np.abs(res)))
    for k, pos in (("auroc_destab", y > 1.0), ("auroc_stab", y < -0.5)):
        out[k] = auroc(p if k == "auroc_destab" else -p, pos) if pos.sum() >= min_pos else np.nan
    nz = y != 0
    out["sign_acc"] = float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else np.nan
    return out


def _prot_spearman(y, p, prot, min_n) -> pd.Series:
    d = pd.DataFrame({"y": y, "p": p, "g": prot})
    return d.groupby("g", sort=False).apply(
        lambda g: _safe(spearmanr, g.p.to_numpy(), g.y.to_numpy()) if len(g) >= min_n else np.nan, include_groups=False)


def _target(df: pd.DataFrame, target: str | None, y_col: str, score_col: str):
    if target is None:
        target = "ddg" if df[y_col].notna().any() else "score"
    y = df[y_col].to_numpy(float) if target == "ddg" else -df[score_col].to_numpy(float)
    return target, y


def evaluate(df: pd.DataFrame, pred_col: str = "pred", target: str | None = None, y_col: str = "ddg",
             score_col: str = "score", protein_col: str = "protein_id", min_n: int = 10) -> dict:
    """All metrics on rows of `df`. target='score' scores against -score (rank metrics only)."""
    target, y = _target(df, target, y_col, score_col)
    p = df[pred_col].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, prot = y[ok], p[ok], df[protein_col].to_numpy()[ok]
    ps = _prot_spearman(y, p, prot, min_n)
    out = {"n": int(ok.sum()), "n_proteins": int(len(np.unique(prot))), "n_prot_eval": int(ps.notna().sum()), "target": target}
    out.update(_pooled(y, p, target))
    out["spearman_prot_median"], out["spearman_prot_mean"] = float(ps.median()), float(ps.mean())
    return out


# ----------------------------------------------------------------------------- bootstrap
class _Boot:
    """Pre-split data so bootstrap replicates (resampled groups) are cheap."""

    def __init__(self, df, pred_col, target, y_col, score_col, protein_col, cluster_col, min_n):
        self.target, y = _target(df, target, y_col, score_col)
        p = df[pred_col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        df, self.y, self.p = df[ok], y[ok], p[ok]
        prot = df[protein_col].to_numpy()
        grp = df[cluster_col].to_numpy() if cluster_col and cluster_col in df else prot
        self.psp = _prot_spearman(self.y, self.p, prot, min_n)
        gi = pd.Series(np.arange(len(grp))).groupby(grp, sort=False)
        self.rows = {g: ix.to_numpy() for g, ix in gi}
        # protein -> spearman values per group (a cluster may hold several proteins)
        self.pvals = {g: self.psp.reindex(pd.unique(prot[r])).to_numpy() for g, r in self.rows.items()}
        self.groups = np.array(list(self.rows))

    def metrics(self, groups) -> dict:
        idx = np.concatenate([self.rows[g] for g in groups])
        out = _pooled(self.y[idx], self.p[idx], self.target)
        pv = np.concatenate([self.pvals[g] for g in groups])
        pv = pv[np.isfinite(pv)]
        out["spearman_prot_median"] = float(np.median(pv)) if len(pv) else np.nan
        out["spearman_prot_mean"] = float(np.mean(pv)) if len(pv) else np.nan
        return out


def _draws(groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(groups, len(groups)) for _ in range(n_boot)]


def bootstrap_ci(df, pred_col="pred", target=None, n_boot=1000, seed=0, cluster_col="cluster", y_col="ddg",
                 score_col="score", protein_col="protein_id", min_n=10, alpha=0.05) -> dict:
    """{metric: (lo, hi)} percentile CI, resampling `cluster_col` (if present, else proteins) with replacement."""
    b = _Boot(df, pred_col, target, y_col, score_col, protein_col, cluster_col, min_n)
    reps = pd.DataFrame([b.metrics(g) for g in _draws(b.groups, n_boot, seed)])
    q = reps.quantile([alpha / 2, 1 - alpha / 2])
    return {k: (float(q[k].iloc[0]), float(q[k].iloc[1])) for k in reps if reps[k].notna().any()}


def paired_bootstrap(df, pred_a, pred_b, target=None, n_boot=1000, seed=0, cluster_col="cluster", y_col="ddg",
                     score_col="score", protein_col="protein_id", min_n=10, alpha=0.05) -> dict:
    """Delta = metric(A) - metric(B) on identical rows, same resampled groups for both.

    Returns {metric: dict(delta, lo, hi, p)}; p = 2*min(P(delta<=0), P(delta>=0)) over replicates
    (two-sided bootstrap fraction, NOT a calibrated test; for RMSE/MAE lower is better).
    """
    both = df[df[pred_a].notna() & df[pred_b].notna()]
    A, B = (_Boot(both, c, target, y_col, score_col, protein_col, cluster_col, min_n) for c in (pred_a, pred_b))
    full = A.groups
    d0 = {k: A.metrics(full)[k] - B.metrics(full)[k] for k in A.metrics(full)}
    reps = pd.DataFrame([{k: a[k] - b_[k] for k in a} for g in _draws(full, n_boot, seed)
                         for a, b_ in [(A.metrics(g), B.metrics(g))]])
    out = {}
    for k in reps:
        r = reps[k].dropna()
        if len(r) and np.isfinite(d0[k]):
            out[k] = dict(delta=float(d0[k]), lo=float(r.quantile(alpha / 2)), hi=float(r.quantile(1 - alpha / 2)),
                          p=float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean()))))
    return out


# ----------------------------------------------------------------------------- antisymmetry
def antisymmetry(df: pd.DataFrame, pred_col: str = "pred", pair_col: str = "pair_id", dir_col: str = "direction",
                 y_col: str = "ddg") -> dict:
    """Direct/reverse pairs (Ssym-style). `direction` in {direct, reverse} (also dir/rev, 1/-1).

    r_pearson / r_spearman: correlation of pred(direct) with -pred(reverse) (1 = perfectly antisymmetric);
    bias: mean(pred_dir + pred_rev) (0 = unbiased; positive = both directions predicted destabilising);
    the same three numbers on experimental ddg ('exp_*', for reference) and per-direction accuracy.
    """
    d = df.copy()
    dr = d[dir_col].astype(str).str.lower().str[:3]
    d["_d"] = np.where(dr.isin(["dir", "1", "1.0"]), "dir", np.where(dr.isin(["rev", "-1", "-1."]), "rev", "?"))
    if (d["_d"] == "?").any():
        raise ValueError(f"unrecognised {dir_col} values: {sorted(df[dir_col].astype(str).unique())[:6]}")
    w = d.pivot_table(index=pair_col, columns="_d", values=[pred_col, y_col], aggfunc="first")
    w = w.dropna()
    pd_, pr_ = w[(pred_col, "dir")].to_numpy(float), w[(pred_col, "rev")].to_numpy(float)
    yd, yr = w[(y_col, "dir")].to_numpy(float), w[(y_col, "rev")].to_numpy(float)
    return {"n_pairs": int(len(w)),
            "r_pearson": _safe(pearsonr, pd_, -pr_), "r_spearman": _safe(spearmanr, pd_, -pr_),
            "bias": float(np.mean(pd_ + pr_)) if len(w) else np.nan,
            "exp_r_pearson": _safe(pearsonr, yd, -yr), "exp_bias": float(np.mean(yd + yr)) if len(w) else np.nan,
            "pearson_direct": _safe(pearsonr, pd_, yd), "pearson_reverse": _safe(pearsonr, pr_, yr)}
