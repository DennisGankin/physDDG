"""Fast per-variant structure descriptors (no side-chain modelling) for baselines B1-B3+ and as fit.py inputs.

Per site (from E1's physddg.structure / physddg.rotlib, imported read-only):
  phi, psi, ss3/ss8 (DSSP), cn8/cn/cn12 (smooth CB contact number, 8/10/12 A, sigmoid width 1 A), ncb10, rsa (freesasa),
  bfac (CA B-factor = pLDDT for AF models), bfac_z (per-structure z-score), nh_hb (backbone N-H donates a DSSP H-bond,
  E < -0.5 kcal/mol), cis (peptide bond preceding the site is cis), in_ss (Cys in a disulfide), prev/next residue.
Per variant (sign: positive = destabilising, as ddg):
  de_env, de_env_bb, de_env_rsa  E_env(mut|env) - E_env(wt|env), PDB statistical potential (kT units) for the
                                 (CN bin x SS3), (CN x SS3 x phi/psi basin), (RSA bin x SS3) environments
  de_rama       E_rama(mut) - E_rama(wt) at the site's phi/psi (per-aa maps; pre-Pro map if residue i+1 is Pro)
  de_prepro     X->P: residue i-1 becomes pre-Pro, E_prepro(i-1) - E_aa(i-1)(i-1); P->X: the negative; else 0
  dhyd          hyd(wt) - hyd(mut), Fauchere-Pliska octanol pi in kcal/mol (x 2.303RT)
  dvol, dvol_loss, dvol_gain  V(wt)-V(mut) (Zamyatnin, A^3/100), its positive / negative parts
  occ, occn     Occ(i) = sum_j V_j exp(-d^2/2 3.5^2) (heavy atoms of other residues around CB), occn = occ/OCC_SCALE;
                occ_dvol_loss / occ_dvol_gain / occ_dhyd = occn x (PoPMuSiC-style cavity / overpacking / hydrophobic)
  bur           1 - clip(rsa, 0, 1);  bur_dhyd, bur_dvol_loss, bur_dvol_gain, bur_dscent (bur x (S_mut - S_wt), nats)
  to_pro, from_pro, pro_nhhb (X->P where the WT N-H is H-bonded), cis_pro_lost, to_gly, gly_posphi (G->X at phi>0),
  ss_broken (Cys in a disulfide mutated)
Caches: data/features/simple_<dataset>.parquet (scripts/50_features_simple.py).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import structure as S
from .rotlib import PDBStats
from .topology import AA1, AA_INDEX

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
SITE_KEYS = ["structure", "chain", "pdb_resnum"]
KEYS = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]

_FP_PI = dict(A=0.31, R=-1.01, N=-0.60, D=-0.77, C=1.54, Q=-0.22, E=-0.64, G=0.00, H=0.13, I=1.80, L=1.70, K=-0.99,
              M=1.23, F=1.79, P=0.72, S=-0.04, T=0.26, W=2.25, Y=0.96, V=1.22)   # Fauchere & Pliska 1983
HYD = {a: 1.364 * v for a, v in _FP_PI.items()}                                    # kcal/mol (2.303 RT, 298 K)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6,
           M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)  # Zamyatnin 1972, A^3

OCC_SCALE = 500.0   # A^3; Occ / OCC_SCALE ~ 1 for a fully buried site (set from the Megascale/FireProt Occ distribution)
NUMERIC = ["de_env", "de_env_bb", "de_env_rsa", "de_rama", "de_prepro", "dhyd", "dvol", "dvol_loss", "dvol_gain",
           "bur", "bur_dhyd", "bur_dvol_loss", "bur_dvol_gain", "bur_dscent", "to_pro", "from_pro", "pro_nhhb",
           "cis_pro_lost", "to_gly", "gly_posphi", "ss_broken", "occ", "occn", "occ_dvol_loss", "occ_dvol_gain", "occ_dhyd",
           "rsa", "cn", "cn8", "cn12", "ncb10", "phi", "psi", "bfac", "bfac_z", "nh_hb", "helix", "strand"]


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def _nh_hbond(p: S.Protein, cutoff: float = -0.5) -> np.ndarray:
    """[L] True if backbone N-H of residue j donates a Kabsch-Sander H-bond (E < cutoff) to any C=O, |i-j| >= 2."""
    from scipy.spatial import cKDTree
    n, c, o = (p.xyz14[:, k].astype(np.float64) for k in (0, 2, 3))
    h = S.backbone_h(p, dssp_style=True)
    ok = np.isfinite(h).all(1) & np.isfinite(o).all(1)
    out = np.zeros(len(p), bool)
    pr = cKDTree(np.nan_to_num(p.xyz14[:, 1].astype(np.float64))).query_pairs(9.0, output_type="ndarray")
    if not len(pr):
        return out
    pr = np.concatenate([pr, pr[:, ::-1]])
    i, j = pr[:, 0], pr[:, 1]                    # acceptor C=O of i, donor N-H of j
    keep = (np.abs(i - j) >= 2) & ok[i] & ok[j]
    i, j = i[keep], j[keep]
    d = lambda x, y: np.linalg.norm(x[i] - y[j], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = 0.084 * 332.0 * (1 / d(o, n) + 1 / d(c, h) - 1 / d(o, h) - 1 / d(c, n))
    out[np.unique(j[e < cutoff])] = True
    return out


def occupancy(p: S.Protein, sigma: float = 3.5, cutoff: float = 12.0) -> np.ndarray:
    """Occ(i) = sum_j V_j exp(-d_ij^2 / 2 sigma^2) over heavy atoms j of other residues, d from the site's CB
    (ideal CB for Gly); V_j = CHARMM19/EEF1 atomic volume (physddg.params, frozen literature values), A^3."""
    from scipy.spatial import cKDTree
    from .params import COLS, type_table
    at = p.atoms()
    vol = type_table()[at["type_id"], COLS.index("volume")].astype(np.float64)
    cb = S.cb_coords(p).astype(np.float64)
    tree = cKDTree(at["xyz"].astype(np.float64))
    occ = np.zeros(len(p))
    for i, nb in enumerate(tree.query_ball_point(cb, cutoff)):
        nb = np.asarray(nb, np.int64)
        nb = nb[at["res_index"][nb] != i]
        d2 = ((at["xyz"][nb] - cb[i]) ** 2).sum(1)
        occ[i] = (vol[nb] * np.exp(-d2 / (2 * sigma ** 2))).sum()
    return occ


def site_table(path: str) -> pd.DataFrame:
    """Per-residue descriptors of one structure file (all chains)."""
    p = S.annotate(S.load_protein(_resolve(path), keep_het=False))
    a, L = p.ann, len(p)
    cb = S.cb_coords(p)
    br = a["chain_break"]
    nxt_ok = np.r_[~br[1:], False]                        # residue i+1 bonded to i
    prev_ok = ~br
    seq = np.array(list(p.seq))
    ss_res = np.zeros(L, bool)
    for x, y in a["disulfides"]:
        ss_res[[x, y]] = True
    bf = p.bfactor.astype(float)
    sd = bf.std()
    df = pd.DataFrame({
        "structure": path, "chain": p.chain, "pdb_resnum": p.resnum, "aa": seq,
        "phi": a["phi"], "psi": a["psi"], "ss3": a["ss3"], "ss8": a["ss8"],
        "cn": a["cn"], "cn8": S.smooth_cn(cb, r0=8.0), "cn12": S.smooth_cn(cb, r0=12.0), "ncb10": a["ncb10"],
        "rsa": a["rsa"], "occ": occupancy(p), "bfac": bf, "bfac_z": (bf - bf.mean()) / sd if sd > 0 else 0.0,
        "nh_hb": _nh_hbond(p), "cis": a["cis"] & prev_ok, "in_ss": ss_res,
        "next_aa": np.where(nxt_ok, np.roll(seq, -1), "-"), "prev_aa": np.where(prev_ok, np.roll(seq, 1), "-"),
        "prev_phi": np.where(prev_ok, np.roll(a["phi"], 1), np.nan), "prev_psi": np.where(prev_ok, np.roll(a["psi"], 1), np.nan),
        "missing_sc": p.missing_sc,
    })
    return df


def _safe_site_table(path):
    try:
        return site_table(path)
    except Exception as e:  # noqa: BLE001  (report, never silently drop)
        print(f"site_table FAILED {path}: {e!r}", file=sys.stderr)
        return None


def site_tables(paths, n_jobs: int = 48) -> pd.DataFrame:
    paths = sorted(set(paths))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    with Pool(min(n_jobs, len(paths))) as pool:
        res = pool.map(_safe_site_table, paths, chunksize=1)
    return pd.concat([r for r in res if r is not None], ignore_index=True)


@lru_cache(maxsize=1)
def _stats() -> PDBStats:
    return PDBStats()


def variant_features(rows: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """rows: KEYS + structure (+ anything); sites: site_tables(). Returns rows' KEYS/structure + features (row-aligned)."""
    st = _stats()
    r = rows[KEYS + ["structure"]].copy()
    r["pdb_resnum"] = r["pdb_resnum"].astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    f = r.merge(sites, on=SITE_KEYS, how="left")
    assert len(f) == len(r)
    found = f["aa"].notna().to_numpy()
    wt_ok = (f["aa"] == f["wt"]).to_numpy()
    f["site_found"], f["wt_match"] = found, wt_ok
    g = f[found].copy()
    iw = g.wt.map(AA_INDEX).to_numpy()
    im = g.mut.map(AA_INDEX).to_numpy()
    ix = np.arange(len(g))
    ss3 = g.ss3.to_numpy()
    for col, var in (("de_env", "cn_ss"), ("de_env_bb", "cn_ss_bb"), ("de_env_rsa", "rsa_ss")):
        E = st.env_energy(ss3, cn=g.cn.to_numpy(), rsa=g.rsa.to_numpy(), phi=g.phi.to_numpy(), psi=g.psi.to_numpy(), variant=var)
        g[col] = E[ix, im] - E[ix, iw]
    R = st.rama_energy_all_aa(g.phi.to_numpy(), g.psi.to_numpy(), next_is_pro=(g.next_aa == "P").to_numpy())
    g["de_rama"] = np.nan_to_num(R[ix, im] - R[ix, iw])
    # pre-Pro at i-1
    pa = g.prev_aa.to_numpy()
    ok = ~np.isin(pa, ["G", "P", "-"]) & np.isfinite(g.prev_phi.to_numpy()) & np.isfinite(g.prev_psi.to_numpy())
    e_pp = np.zeros(len(g)); e_own = np.zeros(len(g))
    if ok.any():
        e_pp[ok] = st.rama_energy("prepro", g.prev_phi.to_numpy()[ok], g.prev_psi.to_numpy()[ok])
        e_own[ok] = st.rama_energy(pa[ok], g.prev_phi.to_numpy()[ok], g.prev_psi.to_numpy()[ok])
    sgn = np.where((g.mut == "P") & (g.wt != "P"), 1.0, np.where((g.wt == "P") & (g.mut != "P"), -1.0, 0.0))
    g["de_prepro"] = np.nan_to_num(sgn * (e_pp - e_own))
    hw, hm = g.wt.map(HYD), g.mut.map(HYD)
    vw, vm = g.wt.map(VOL) / 100, g.mut.map(VOL) / 100
    g["dhyd"] = hw - hm
    g["dvol"] = vw - vm
    g["dvol_loss"], g["dvol_gain"] = g.dvol.clip(lower=0), (-g.dvol).clip(lower=0)
    g["bur"] = 1 - g.rsa.clip(0, 1)
    for c in ("dhyd", "dvol_loss", "dvol_gain"):
        g[f"bur_{c}"] = g.bur * g[c]
    g["occn"] = (g.occ / OCC_SCALE).clip(0, 1.5)
    g["occ_dvol_loss"], g["occ_dvol_gain"] = g.occn * g.dvol_loss, g.occn * g.dvol_gain
    g["occ_dhyd"] = g.occn * g.dhyd
    sce = st.sc_entropy("sc_entropy_rotameric", "all")
    g["bur_dscent"] = g.bur * (sce[im] - sce[iw])
    to_pro = (g.mut == "P") & (g.wt != "P")
    g["to_pro"], g["from_pro"] = to_pro.astype(float), ((g.wt == "P") & (g.mut != "P")).astype(float)
    g["pro_nhhb"] = (to_pro & g.nh_hb.astype(bool)).astype(float)
    g["cis_pro_lost"] = ((g.wt == "P") & g.cis.astype(bool)).astype(float)
    g["to_gly"] = ((g.mut == "G") & (g.wt != "G")).astype(float)
    g["gly_posphi"] = ((g.wt == "G") & (g.mut != "G") & (g.phi > 0)).astype(float)
    g["ss_broken"] = ((g.wt == "C") & (g.mut != "C") & g.in_ss.astype(bool)).astype(float)
    g["helix"], g["strand"] = (g.ss3 == "H").astype(float), (g.ss3 == "E").astype(float)
    g["nh_hb"] = g.nh_hb.astype(float)
    out = f[KEYS + ["structure", "site_found", "wt_match", "ss3", "ss8"]].copy()
    for c in NUMERIC:
        out[c] = np.nan
        out.loc[found, c] = g[c].to_numpy(float)
    return out


# ----------------------------------------------------------------------------- dataset loaders
def dataset_rows(name: str) -> pd.DataFrame:
    """Benchmark rows needing features, with KEYS + structure. 'megascale' includes megascale_censored rows;
    'paired_af' = AF side, 'paired_af_xtal' = crystal side of data/processed/paired_af.csv (usable rows)."""
    base = ROOT / "data" / "processed"
    if name == "megascale":
        a = pd.read_parquet(base / "megascale.parquet", columns=KEYS + ["structure"])
        b = pd.read_csv(base / "megascale_censored.csv", usecols=KEYS + ["structure"])
        return pd.concat([a, b], ignore_index=True)
    if name.startswith("paired_af"):
        d = pd.read_csv(base / "paired_af.csv", low_memory=False)
        d = d[d.usable.astype(bool)]
        if name == "paired_af_xtal":
            d = d.assign(structure=d.xtal_structure, chain=d.xtal_chain, pdb_resnum=d.xtal_resnum)
        return d[KEYS + ["structure"]]
    p = base / f"{name}.parquet"
    d = pd.read_parquet(p, columns=KEYS + ["structure"]) if p.exists() else pd.read_csv(base / f"{name}.csv", low_memory=False)
    return d.loc[d.structure.notna(), KEYS + ["structure"]]


def build(name: str, n_jobs: int = 48, force: bool = False) -> pd.DataFrame:
    out = FEAT_DIR / f"simple_{name}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    rows = dataset_rows(name)
    rows = rows.assign(pdb_resnum=rows.pdb_resnum.astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True))
    rows = rows.drop_duplicates(KEYS + ["structure"]).reset_index(drop=True)
    sites = site_tables(rows.structure.unique(), n_jobs)
    feats = variant_features(rows, sites)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    return feats


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT_DIR / f"simple_{name}.parquet")
