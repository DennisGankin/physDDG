"""Elastic network model (ENM) site-flexibility descriptors from a single structure (no fitting, no DL).

Models (Bahar, Atilgan & Erman 1997; Atilgan et al. 2001), kT = gamma = 1 units:
  GNM   Kirchhoff matrix G_ij = -gamma_ij for |r_ij| <= rc (10 A, CA nodes), G_ii = -sum_j G_ij.
        MSF_i = [G^+]_ii (all non-zero modes) and the slow-mode part sum_{k<=K} u_ki^2 / lambda_k (K = 10).
  ANM   3N x 3N Hessian, H_ij = -gamma_ij e_ij e_ij^T for |r_ij| <= rc (15 A, CA nodes); covariance C = H^+
        (the 6 rigid-body zero modes removed). MSF_i = tr C_ii.
  derived from the ANM covariance C (blocks C_ij, 3x3):
    stiffness   effective spring constant of pair (i, j) along e_ij, k_ij = 1 / <(delta d_ij)^2>,
                <(delta d_ij)^2> = e^T (C_ii + C_jj - C_ij - C_ji) e (Eyal & Bahar 2008, Biophys J 94:3424;
                ProDy calcMechStiff). Per residue: mean over j within 10 A (stiff_local) and over all j (stiff_all).
    PRS         perturbation-response scanning (Atilgan & Atilgan 2009; General et al. 2014): mean squared response of
                i to a unit force of random direction at j, P_ij = ||C_ij||_F^2 / 3 (analytic isotropic average).
                Row-normalised by the self response, Q_ij = P_ij / P_ii: effectiveness_i = mean_j Q_ij (perturbing i
                moves the rest), sensitivity_j = mean_i Q_ij (j responds to perturbations elsewhere).
                dfi_i = sum_j sqrt(P_ij) / mean_i(sum_j sqrt(P_ij)) (dynamic flexibility index, Gerek et al. 2013).
    hinge       1 - |u_1i| / max|u_1| on the slowest GNM mode (hinge residues sit at its nodes).
  CA+centroid   ANM on CA and side-chain centroid nodes (CB onwards; Gly has no centroid node), cutoff 12 A;
                cen_msf = MSF of the residue's centroid node (CA node for Gly).
  pLDDT option  springs scaled by confidence, gamma_ij = sqrt(p_i p_j), p = clip(pLDDT/100, 0.1, 1) (AF inputs only):
                gnm_msf_plddt. Without pLDDT the column is NaN.
  degree        number of CA within 10 A (the GNM diagonal) - kept as the static reference descriptor.

Disconnected components or under-constrained nodes give extra (near-)zero modes; every pseudo-inverse drops
eigenvalues below `tol * lambda_max`, so isolated nodes get large-but-finite fluctuations from the remaining modes
and are reported as NaN only if a node has no contact at all.

API
  enm_descriptors(protein_or_coords, plddt=None, ...) -> dict of per-residue arrays (length N, residue order)
  normalise(desc, groups) -> per-structure log / z / rank columns
  enm_site_table(structures, name) -> cached per-residue table (data/features/enm_sites_<name>.parquet)
Everything is numpy/scipy; a 540-residue protein takes ~1 s (dominated by the 1,620 x 1,620 ANM eigh).
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "features"

# descriptors where larger = more flexible (log taken before z-scoring); stiffness-type ones are larger = stiffer
FLEX = ("gnm_msf", "gnm_msf_slow", "gnm_msf_plddt", "anm_msf", "cen_msf", "dfi", "prs_sens")
STIFF = ("stiff_local", "stiff_all", "prs_eff", "degree")
OTHER = ("gnm_hinge",)
ALL = FLEX + STIFF + OTHER


# ----------------------------------------------------------------------------- network matrices
def _gamma(xyz: np.ndarray, cutoff: float, w: np.ndarray | None):
    d = cdist(xyz, xyz)
    con = (d <= cutoff) & ~np.eye(len(xyz), dtype=bool)
    g = con.astype(float)
    if w is not None:
        g = g * np.sqrt(np.outer(w, w))
    return d, con, g


def kirchhoff(xyz: np.ndarray, cutoff: float = 10.0, w: np.ndarray | None = None) -> np.ndarray:
    """GNM Kirchhoff (graph Laplacian) matrix; w = optional per-node spring scale (gamma_ij = sqrt(w_i w_j))."""
    _, _, g = _gamma(np.asarray(xyz, float), cutoff, w)
    return np.diag(g.sum(1)) - g


def hessian(xyz: np.ndarray, cutoff: float = 15.0, w: np.ndarray | None = None) -> np.ndarray:
    """ANM Hessian (3N x 3N) with uniform (or confidence-scaled) springs."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    d, con, g = _gamma(xyz, cutoff, w)
    diff = xyz[:, None, :] - xyz[None, :, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(con[..., None], diff / np.where(d > 0, d, 1.0)[..., None], 0.0)
    off = -g[..., None, None] * e[..., :, None] * e[..., None, :]          # [N,N,3,3]
    diag = -off.sum(1)                                                      # [N,3,3]
    off[np.arange(n), np.arange(n)] = diag
    return off.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def _pinv_eig(m: np.ndarray, tol: float = 1e-8):
    lam, u = np.linalg.eigh(m)
    keep = lam > tol * max(lam.max(), 1e-12)
    return lam, u, keep


def gnm(xyz: np.ndarray, cutoff: float = 10.0, k_slow: int = 10, w: np.ndarray | None = None) -> dict:
    """GNM spectrum, MSF (all non-zero modes), slow-mode MSF (lowest k_slow non-zero modes), slowest mode."""
    K = kirchhoff(xyz, cutoff, w)
    lam, u, keep = _pinv_eig(K)
    lk, uk = lam[keep], u[:, keep]
    msf = (uk ** 2 / lk).sum(1)
    ks = min(k_slow, len(lk))
    msf_slow = (uk[:, :ks] ** 2 / lk[:ks]).sum(1)
    return {"eigvals": lam, "n_zero": int((~keep).sum()), "msf": msf, "msf_slow": msf_slow,
            "mode1": uk[:, 0] if len(lk) else np.full(len(K), np.nan), "degree": np.diag(K).copy()}


def anm_cov(xyz: np.ndarray, cutoff: float = 15.0, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """ANM covariance C = H^+ (3N x 3N), eigenvalues, number of dropped (zero) modes."""
    H = hessian(xyz, cutoff, w)
    lam, u, keep = _pinv_eig(H)
    C = (u[:, keep] / lam[keep]) @ u[:, keep].T
    return C, lam, int((~keep).sum())


def _blocks(C: np.ndarray) -> np.ndarray:
    n = C.shape[0] // 3
    return C.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)                      # [N,N,3,3]


def mech_stiffness(C: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Pairwise effective spring constants k_ij = 1 / <(delta d_ij)^2> (Eyal & Bahar 2008); diagonal NaN."""
    B = _blocks(C)
    xyz = np.asarray(xyz, float)
    diff = xyz[None, :, :] - xyz[:, None, :]
    d = np.linalg.norm(diff, axis=-1)
    n = len(xyz)
    e = diff / np.where(d > 0, d, 1.0)[..., None]
    D = B[np.arange(n), np.arange(n)]                                        # [N,3,3]
    var = (np.einsum("ija,iab,ijb->ij", e, D, e) + np.einsum("ija,jab,ijb->ij", e, D, e)
           - 2 * np.einsum("ija,ijab,ijb->ij", e, B, e))
    with np.errstate(divide="ignore"):
        k = 1.0 / np.where(var > 1e-12, var, np.nan)
    k[np.arange(n), np.arange(n)] = np.nan
    return k


def prs(C: np.ndarray) -> dict:
    """Perturbation-response scanning from the covariance (isotropic unit forces, analytic average)."""
    B = _blocks(C)
    P = (B ** 2).sum((-1, -2)) / 3.0                                       # [N,N], symmetric
    Q = P / np.diag(P)[:, None]
    s = np.sqrt(P).sum(1)
    return {"P": P, "effectiveness": Q.mean(1), "sensitivity": Q.mean(0), "dfi": s / s.mean()}


# ----------------------------------------------------------------------------- nodes
def _ca_centroid(prot):
    ca = prot.xyz14[:, 1].astype(float)
    sc = prot.xyz14[:, 4:].astype(float)
    m = prot.mask14[:, 4:]
    cnt = m.sum(1)
    cen = np.where(cnt[:, None] > 0, np.nansum(np.where(m[..., None], sc, 0.0), 1) / np.maximum(cnt, 1)[:, None], np.nan)
    return ca, cen


def enm_descriptors(x, plddt=None, cutoff_gnm: float = 10.0, cutoff_anm: float = 15.0, cutoff_cen: float = 12.0,
                    k_slow: int = 10, local_r: float = 10.0, centroid: bool = True) -> dict:
    """Per-residue ENM descriptors. x: physddg.structure.Protein (all residues, all chains) or CA coords [N,3].
    plddt: per-residue pLDDT (0-100) for the confidence-scaled GNM variant (AF models only), else None."""
    if hasattr(x, "xyz14"):
        ca, cen = _ca_centroid(x)
    else:
        ca, cen = np.asarray(x, float), None
    n = len(ca)
    out = {}
    g = gnm(ca, cutoff_gnm, k_slow)
    out["gnm_msf"], out["gnm_msf_slow"], out["degree"] = g["msf"], g["msf_slow"], g["degree"]
    m1 = np.abs(g["mode1"])
    out["gnm_hinge"] = 1.0 - m1 / m1.max() if np.isfinite(m1).all() and m1.max() > 0 else np.full(n, np.nan)
    if plddt is not None:
        p = np.clip(np.asarray(plddt, float) / 100.0, 0.1, 1.0)
        out["gnm_msf_plddt"] = gnm(ca, cutoff_gnm, k_slow, w=p)["msf"]
    else:
        out["gnm_msf_plddt"] = np.full(n, np.nan)
    C, _, _ = anm_cov(ca, cutoff_anm)
    B = _blocks(C)
    out["anm_msf"] = np.einsum("iaa->i", B[np.arange(n), np.arange(n)])
    k = mech_stiffness(C, ca)
    d = cdist(ca, ca)
    loc = (d <= local_r) & ~np.eye(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        out["stiff_local"] = np.nanmean(np.where(loc, k, np.nan), 1) if n > 1 else np.full(n, np.nan)
        out["stiff_all"] = np.nanmean(k, 1) if n > 1 else np.full(n, np.nan)
    pr = prs(C)
    out["prs_eff"], out["prs_sens"], out["dfi"] = pr["effectiveness"], pr["sensitivity"], pr["dfi"]
    if centroid and cen is not None:
        has = np.isfinite(cen[:, 0])
        nodes = np.vstack([ca, cen[has]])
        Cc, _, _ = anm_cov(nodes, cutoff_cen)
        msf_nodes = np.einsum("iaa->i", _blocks(Cc)[np.arange(len(nodes)), np.arange(len(nodes))])
        cm = msf_nodes[:n].copy()
        cm[has] = msf_nodes[n:]
        out["cen_msf"] = cm
    else:
        out["cen_msf"] = np.full(n, np.nan)
    iso = out["degree"] == 0
    for kk in out:
        out[kk] = np.asarray(out[kk], float)
        out[kk][iso] = np.nan
    return out


# ----------------------------------------------------------------------------- normalisation
def normalise(df: pd.DataFrame, group: str = "structure", cols=ALL) -> pd.DataFrame:
    """Add per-structure normalised versions: <c>_log (log of the positive raw value), <c>_z (z-score of the log
    within the structure; hinge z-scored as is) and <c>_rank (within-structure percentile in [0, 1])."""
    out = df.copy()
    for c in cols:
        if c not in out:
            continue
        v = out[c].astype(float)
        lv = np.log(v.clip(lower=1e-12)) if c != "gnm_hinge" else v
        out[f"{c}_log"] = lv
        g = lv.groupby(out[group])
        out[f"{c}_z"] = (lv - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        out[f"{c}_rank"] = g.rank(pct=True)
    return out


def _is_plddt(b: np.ndarray) -> bool:
    """B-factor column looks like pLDDT (AF model): all in [0, 100] and mean in [30, 100]."""
    return bool(np.all((b >= 0) & (b <= 100)) and 30 <= b.mean() <= 100)


def site_descriptors(path: str) -> pd.DataFrame:
    from . import structure as S
    p = S.load_protein(path, keep_het=False)
    b = p.bfactor.astype(float)
    af = _is_plddt(b)
    d = enm_descriptors(p, plddt=b if af else None)
    t = pd.DataFrame({"chain": p.chain, "pdb_resnum": p.resnum, **d})
    t["bf_as_plddt"] = af
    t["n_res"] = len(p)
    return t


def _one(path):
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):                 # one BLAS thread per worker (oversubscription is 100x slower)
            return site_descriptors(path)
    except Exception as e:  # noqa: BLE001
        print("ENM FAILED", path, e, flush=True)
        return None


def enm_site_table(structures, name: str, n_jobs: int = 32, refresh: bool = False) -> pd.DataFrame:
    """ENM descriptors (raw + per-structure log/z/rank) for all residues of the given structures, cached at
    data/features/enm_sites_<name>.parquet; `structure` equals the input strings."""
    cache = CACHE / f"enm_sites_{name}.parquet"
    structures = sorted(set(map(str, structures)))
    old = pd.read_parquet(cache) if cache.exists() and not refresh else None
    todo = structures if old is None else [s for s in structures if s not in set(old.structure)]
    if todo:
        paths = [s if Path(s).is_absolute() else str(ROOT / s) for s in todo]
        with Pool(n_jobs) as pool:
            res = pool.map(_one, paths, chunksize=1)
        parts = [normalise(r.assign(structure=s)) for s, r in zip(todo, res) if r is not None]
        new = pd.concat(parts, ignore_index=True)
        new["pdb_resnum"] = new["pdb_resnum"].astype(str)
        old = new if old is None else pd.concat([old, new], ignore_index=True)
        old.to_parquet(cache, index=False)
    return old[old.structure.isin(structures)].reset_index(drop=True)
