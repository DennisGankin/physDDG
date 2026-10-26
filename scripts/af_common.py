"""Shared loading / prediction helpers for the AlphaFold-mode experiments (scripts 80-85).

Predictors used throughout (no fitted physics model exists yet; see notes/af_mode.md):
  B3+, B3, B1     fitted simple-descriptor baselines (results/baselines_ladder/models/*.json)
  s1dF, s2dF      unit-weight stage-1 / stage-2 packing free energy from physddg/mutate.py
  P1              small physics model fitted here on Megascale-train only (scripts/83_fit_phys_ref.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import eval as E  # noqa: E402
from physddg import fit as FT  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "af_mode"
LADDER = ROOT / "results" / "baselines_ladder"
KEYS = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]
SIMPLE_MODELS = ("B3+", "B3", "B1")
PLDDT_EDGES = (0, 50, 70, 90, 101)
PLDDT_LABELS = ("<50", "50-70", "70-90", ">90")


def norm_resnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(pdb_resnum=norm_resnum(df.pdb_resnum), chain=df.chain.astype(str),
                     wt=df.wt.astype(str), mut=df.mut.astype(str), protein_id=df.protein_id.astype(str))


def load_feat(path: Path, prefix: str = "") -> pd.DataFrame:
    f = pd.read_parquet(path)
    f = _norm(f).drop(columns=[c for c in ("site_found", "wt_match") if c in f])
    f = f.drop_duplicates(KEYS + ["structure"])
    if prefix:
        f = f.rename(columns={c: prefix + c for c in f.columns if c not in KEYS + ["structure"]})
    return f


def simple(name: str, feat_dir: Path = FEAT, prefix: str = "") -> pd.DataFrame:
    return load_feat(feat_dir / f"simple_{name}.parquet", prefix)


def phys(name: str, stage: int, feat_dir: Path = FEAT, prefix: str = "") -> pd.DataFrame:
    return load_feat(feat_dir / f"phys_s{stage}_{name}.parquet", prefix)


def attach(base: pd.DataFrame, feats: pd.DataFrame, on_structure: bool = True) -> pd.DataFrame:
    on = KEYS + (["structure"] if on_structure and "structure" in feats else [])
    b = _norm(base)
    m = b.merge(feats, on=on, how="left", suffixes=("", "_f"))
    assert len(m) == len(b), (len(m), len(b))
    return m


def models(names=SIMPLE_MODELS) -> dict:
    return {n: FT.load_model(LADDER / "models" / f"{n}.json") for n in names}


def predict_simple(df: pd.DataFrame, mdl: dict | None = None, suffix: str = "") -> pd.DataFrame:
    """Add fitted simple-descriptor predictions (columns `<name><suffix>`) to df (needs the simple features)."""
    mdl = mdl or models()
    out = df.copy()
    for name, m in mdl.items():
        out[name + suffix] = FT.predict(m, df)
    return out


def plddt_band(x) -> pd.Categorical:
    return pd.cut(pd.to_numeric(x, errors="coerce"), bins=PLDDT_EDGES, labels=list(PLDDT_LABELS), right=False)


def per_protein_spearman(df: pd.DataFrame, pred: str, y: str, group: str = "protein_id", min_n: int = 10) -> pd.Series:
    from scipy.stats import spearmanr
    d = df[[pred, y, group]].dropna()

    def f(g):
        if len(g) < min_n or g[pred].nunique() < 2 or g[y].nunique() < 2:
            return np.nan
        return spearmanr(g[pred], g[y])[0]
    return d.groupby(group, sort=False)[[pred, y]].apply(f)


def median_rho(df: pd.DataFrame, pred: str, y: str, group: str = "protein_id", min_n: int = 10) -> float:
    s = per_protein_spearman(df, pred, y, group, min_n)
    return float(np.nanmedian(s)) if len(s) else np.nan


def boot_median_rho(df: pd.DataFrame, preds: list[str], y: str, group: str = "protein_id",
                    cluster: str | None = None, n_boot: int = 500, seed: int = 0, min_n: int = 10) -> pd.DataFrame:
    """Per-group Spearman for several predictors on identical rows + cluster bootstrap of the median (paired)."""
    per = {p: per_protein_spearman(df, p, y, group, min_n) for p in preds}
    idx = per[preds[0]].index
    for p in preds[1:]:
        idx = idx.intersection(per[p].dropna().index)
    idx = idx.intersection(per[preds[0]].dropna().index)
    tab = pd.DataFrame({p: per[p].reindex(idx) for p in preds})
    if cluster and cluster in df:
        cl = df.drop_duplicates(group).set_index(group)[cluster].reindex(idx).fillna(pd.Series(idx, index=idx))
    else:
        cl = pd.Series(idx, index=idx)
    groups = cl.to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rows = {p: [] for p in preds}
    gi = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        pick = np.concatenate([gi[g] for g in rng.choice(uniq, len(uniq))])
        for p in preds:
            rows[p].append(np.nanmedian(tab[p].to_numpy()[pick]))
    res = []
    for p in preds:
        b = np.array(rows[p])
        res.append(dict(pred=p, n_domains=int(len(idx)), median_rho=float(np.nanmedian(tab[p])),
                        lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5))))
    out = pd.DataFrame(res)
    base = np.array(rows[preds[0]])
    for p in preds[1:]:
        d = np.array(rows[p]) - base
        out.loc[out.pred == p, "delta_vs_first"] = float(np.nanmedian(tab[p]) - np.nanmedian(tab[preds[0]]))
        out.loc[out.pred == p, "delta_lo"] = float(np.nanpercentile(d, 2.5))
        out.loc[out.pred == p, "delta_hi"] = float(np.nanpercentile(d, 97.5))
        out.loc[out.pred == p, "delta_p"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


def domainome(with_feats: bool = True, stage: tuple = (1, 2)) -> pd.DataFrame:
    """Domainome rows + site pLDDT + simple/physics features + ladder predictions."""
    cols = ["protein_id", "chain", "pdb_resnum", "wt", "mut", "structure", "score", "score_sigma", "cluster",
            "plddt", "construct_plddt_mean", "construct_frac_plddt_lt70", "uniprot", "domainome_position",
            "af_full_structure", "af_full_resnum", "seq_pos", "ddg_mochi_scaled"]
    d = pd.read_parquet(PROC / "domainome.parquet", columns=cols)
    d = _norm(d[d.structure.notna()]).reset_index(drop=True)
    d["mscore"] = -d.score
    if with_feats:
        d = attach(d, simple("domainome"))
        for s in stage:
            d = attach(d, phys("domainome", s))
        d = predict_simple(d)
    return d


def split_halves(d: pd.DataFrame, seed: int = 0, cluster_col: str = "cluster") -> np.ndarray:
    """Random half of the Domainome *clusters* (Pfam families) -> 'fit', the rest -> 'report'.
    Every AF-specific parameter is chosen on the 'fit' half (+ abundance_dms) and reported on the other."""
    cl = np.array(sorted(d[cluster_col].dropna().unique()))
    rng = np.random.default_rng(seed)
    fit = set(rng.choice(cl, len(cl) // 2, replace=False))
    return np.where(d[cluster_col].isin(fit), "fit", "report")


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    (FEAT / "af").mkdir(parents=True, exist_ok=True)
    return OUT
