"""Extra site / substitution descriptors for the residual analysis (notes/residual_analysis.md, scripts 9x_).

Site descriptors (per residue of the WT structure; no side-chain modelling; computed from the structure only):
  burial      ncb6..ncb14 (CB counts at 6/8/10/12/14 A), ncen8/10/12 (side-chain centroid counts), hse_up/hse_down
              (HSE-beta, 13 A), rsa / rsa_sc (freesasa), cn (smooth CB contact number)
  packing     natom6 / natom8 (heavy atoms of other residues within 6 / 8 A of the side-chain centroid),
              void_frac (fraction of grid points within 5 A of the centroid that lie outside vdW+0.5 A of every atom
              and are not solvent-accessible by a 1.4 A probe -> internal empty space), scsc_contacts (side-chain
              heavy atoms of other residues within 4.5 A of the site's side-chain atoms), scsc_res (distinct residues)
  hbond       hb_sc_any / hb_sc_bb / hb_sc_sc: WT side-chain polar atoms within 3.5 A of a polar atom of another
              residue (counts of partner atoms; no angles), hb_bb_nh / hb_bb_co (backbone N-H / C=O DSSP-bonded)
  ss_pos      ss8, helix flags: ncap (residue before the first H), n1..n3, c1..c3, ccap (residue after the last H),
              helix_int (>= 3 from both ends), helix_len, helix_pos (fraction along helix); strand: strand_edge /
              strand_centre (bridged to one / two other strand segments), strand_len; loop_len
  termini     dist_nterm, dist_cterm, dist_term (min), rel_pos, term3/term6 flags
  plddt       plddt (CA B-factor), plddt_win (+-3 window mean), plddt_z (per-structure z), plddt_min_win
  charge      q_site (formal charge of WT side chain), q_env8 / q_env12 (net formal charge of other side chains
              within 8 / 12 A of the side-chain centroid), n_pos6 / n_neg6 (charged groups within 6 A),
              sb_partners (opposite charges within 4.0 A of a WT charged atom), same_q6 (like charges within 6 A)
  aromatic    n_aro6 / n_aro8 (aromatic ring centroids within 6.5 / 8 A of the centroid), aro_stack (WT aromatic:
              rings within 7 A weighted by |cos| of the normal angle -> stacking / T-shaped), catpi (WT aromatic:
              K NZ / R CZ within 6 A of the ring centroid; WT K/R: rings within 6 A of NZ/CZ)
  backbone    phi, psi, basin (alpha / beta / ppII / alphaL / other), cis_prev
  domain      L, frac_helix, frac_strand, mean_plddt (per structure)

Substitution scales (verification status as of 2026-10-26, fitting lead; see notes/ladder_physics.md "B6"):
  HELIX_PS  Pace & Scholtz 1998, Biophys J 75:422 (PMID 9649402), helix propensity of solvent-exposed mid-helix
            residues, kcal/mol relative to Ala. The 19 non-Pro values were VERIFIED against the paper's abstract.
            Pro = 3.16 is the commonly cited table value and was NOT verified (the abstract excludes Pro).
            Not used by any shipped model (helix propensity failed selection in the residual analysis).
  BETA_MK   Minor & Kim 1994, Nature 367:660, beta-sheet propensity. Typed from memory by the analyst and NOT
            verified; Pro set to -3.0 by hand. Not used by any shipped model (failed selection). Do not cite.
  NCAP_KT   N-cap preference, now a statistical potential derived from the PISCES train set by
            scripts/53_ncap_stats.py -> physddg/data/ncap_pref.json: E(aa) = -ln P(aa | N-cap)/P(aa), kT units,
            negative = favoured (method of Richardson & Richardson 1988, Science 240:1648). Ranking
            S < D < N < T < P < G < H < C < ... < L < V < I reproduces the literature ranking (Doig & Baldwin 1995,
            Protein Sci 4:1325: Asn/Asp/Ser/Thr/Cys best, Gly favoured). It REPLACES the analyst's hand-typed
            "Doig-Baldwin-type" table, whose numbers could not be verified (the paper's table was not retrievable)
            and whose ranking contradicted the paper (Gln was placed mid-scale; the paper has Gln worst).
  helix dipole  no scale: Delta q (formal charge, His neutral) x (+1 at helix N1-N3, -1 at C1-C3). Physical basis:
            Hol, van Duijnen & Berendsen 1978 (Nature 273:443) helix dipole; Shoemaker et al. 1987 (Nature 326:563)
            charged residues at helix ends. Unit = elementary charge; the weight (kcal/mol per charge) is fitted.
D15 (state-function differences): ncap_pref and helix_dipole are differences S(mut, site) - S(wt, site).
  site_scsc and site_hbsc_bur are WT-site main effects (positive for every substitution, like `bur`) and are NOT
  state-function differences; their D15-compliant counterparts are scsc_dvol = (contacts/30) x (V_wt - V_mut)/100
  and hbsc_dpolar = partners x burial x (polar(wt) - polar(mut)).
"""
from __future__ import annotations

import json as _json

from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import structure as S
from .topology import TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "features"

HELIX_PS = dict(A=0.00, L=0.21, R=0.21, M=0.24, K=0.26, Q=0.39, E=0.40, I=0.41, W=0.49, S=0.50, Y=0.53, F=0.54,
                H=0.61, V=0.61, N=0.65, T=0.66, C=0.68, D=0.69, G=1.00, P=3.16)
BETA_MK = dict(T=1.1, I=1.0, Y=0.96, F=0.86, V=0.82, M=0.72, S=0.70, W=0.54, C=0.52, L=0.51, R=0.45, K=0.27, Q=0.23,
               E=0.01, N=-0.08, H=-0.02, A=0.0, D=-0.94, G=-1.2, P=-3.0)
_NCAP_FILE = Path(__file__).resolve().parent / "data" / "ncap_pref.json"
NCAP_KT = _json.loads(_NCAP_FILE.read_text())["energy_kT"] if _NCAP_FILE.exists() else {}
NCAP_DB = NCAP_KT          # backwards-compatible name used by scripts/93_*; now the verified PDB-derived scale
CHARGE = dict(D=-1, E=-1, K=1, R=1)
AROMATIC = set("FWY")
RING = {"F": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "Y": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "W": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"], "H": ["CG", "ND1", "CD2", "CE1", "NE2"]}
CHG_ATOMS = {"D": ["OD1", "OD2"], "E": ["OE1", "OE2"], "K": ["NZ"], "R": ["NE", "NH1", "NH2"]}
CATION_ATOM = {"K": "NZ", "R": "CZ"}
VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def _atom(p, i, name):
    t = TEMPLATES[p.seq[i]]
    if name not in t.atoms:
        return None
    k = t.index(name)
    return p.xyz14[i, k] if p.mask14[i, k] else None


def _ring(p, i):
    pts = [x for x in (_atom(p, i, n) for n in RING.get(p.seq[i], [])) if x is not None]
    if len(pts) < 3:
        return None, None
    pts = np.asarray(pts, np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[2]


def _basin(phi, psi):
    b = np.full(len(phi), "other", "U6")
    b[(phi < 0) & (psi > -100) & (psi < 45)] = "alpha"
    b[(phi < -100) & ((psi > 90) | (psi < -150))] = "beta"
    b[(phi >= -100) & (phi < 0) & ((psi > 90) | (psi < -150))] = "ppII"
    b[phi > 0] = "alphaL"
    return b


def _segments(mask):
    """[(start, end_inclusive)] runs of True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def site_descriptors(path: str) -> pd.DataFrame:
    p = S.annotate(S.load_protein(path, keep_het=False))
    L = len(p)
    a = p.ann
    at = p.atoms()
    ri, nm, xyz = at["res_index"], at["name"], at["xyz"].astype(np.float64)
    elem = np.array([n[0] for n in nm])
    is_sc = ~np.isin(nm, ["N", "CA", "C", "O", "OXT"])
    polar = np.isin(elem, ["N", "O"])
    tree = cKDTree(xyz)
    cb = S.cb_coords(p).astype(np.float64)
    sc = np.where(p.mask14[:, 4:, None], p.xyz14[:, 4:], 0.0).astype(np.float64)
    cnt = p.mask14[:, 4:].sum(1, keepdims=True)
    cen = np.where(cnt > 0, sc.sum(1) / np.maximum(cnt, 1), p.xyz14[:, 1])
    out = {"pdb_resnum": p.resnum, "chain": p.chain, "aa_struct": list(p.seq)}
    # burial
    tcb = cKDTree(cb)
    for r in (6, 8, 10, 12, 14):
        out[f"ncb{r}"] = np.array([len(v) for v in tcb.query_ball_point(cb, r)]) - 1
    tcen = cKDTree(cen)
    for r in (8, 10, 12):
        out[f"ncen{r}"] = np.array([len(v) for v in tcen.query_ball_point(cen, r)]) - 1
    out["hse_up"], out["hse_down"], out["cn"] = a["hse_up"], a["hse_down"], a["cn"]
    out["rsa"] = np.clip(a["rsa"], 0, 1.5)
    maxsc = np.array([max(1.0, float(a["sasa_sc"][i])) for i in range(L)])
    out["sasa_sc"] = a["sasa_sc"]
    # packing / contacts / H-bonds
    natom6, natom8, scsc, scres, hb_any, hb_bb, hb_sc = (np.zeros(L) for _ in range(7))
    for i in range(L):
        for r, arr in ((6, natom6), (8, natom8)):
            nb = tree.query_ball_point(cen[i], r)
            arr[i] = int(np.sum(ri[nb] != i))
        mine = np.flatnonzero((ri == i) & is_sc)
        if len(mine):
            nb = set()
            for k in mine:
                nb.update(tree.query_ball_point(xyz[k], 4.5))
            nb = np.array(sorted(nb), int)
            nb = nb[(ri[nb] != i) & is_sc[nb]]
            scsc[i], scres[i] = len(nb), len(np.unique(ri[nb]))
            pm = mine[polar[mine]]
            hits = set()
            for k in pm:
                for j in tree.query_ball_point(xyz[k], 3.5):
                    if ri[j] != i and abs(ri[j] - i) >= 1 and polar[j]:
                        hits.add(j)
            hits = np.array(sorted(hits), int)
            hb_any[i] = len(hits)
            if len(hits):
                hb_bb[i] = int(np.sum(~is_sc[hits])); hb_sc[i] = int(np.sum(is_sc[hits]))
    out.update(natom6=natom6, natom8=natom8, scsc_contacts=scsc, scsc_res=scres, hb_sc_any=hb_any, hb_sc_bb=hb_bb,
               hb_sc_sc=hb_sc)
    # void fraction: grid within 5 A of centroid, empty (outside vdW + 0.5) and buried (no 1.4 A probe sphere fits
    # with >= 30% solid angle open -> approximated: point has >= 55 atoms within 10 A)
    g = np.mgrid[-5:5.01:1.0, -5:5.01:1.0, -5:5.01:1.0].reshape(3, -1).T
    g = g[np.linalg.norm(g, axis=1) <= 5.0]
    rad = np.array([VDW.get(e, 1.7) for e in elem])
    void = np.zeros(L)
    for i in range(L):
        pts = cen[i] + g
        d, j = tree.query(pts, k=8)
        empty = np.all(d > rad[j] + 0.5, axis=1)
        dense = np.array([len(v) for v in tree.query_ball_point(pts, 10.0)]) >= 55
        void[i] = float(np.mean(empty & dense))
    out["void_frac"] = void
    # backbone H-bonds (DSSP)
    from .features_simple import _nh_hbond
    out["hb_bb_nh"] = _nh_hbond(p).astype(int)
    # secondary-structure position
    ss8, ss3 = a["ss8"], a["ss3"]
    out["ss8"], out["ss3"] = ss8, ss3
    hx = ss8 == "H"
    f = {k: np.zeros(L) for k in ("ncap", "n1", "n2", "n3", "c1", "c2", "c3", "ccap", "helix_int", "helix_len",
                                  "helix_pos", "strand_edge", "strand_centre", "strand_len", "loop_len")}
    for s0, e0 in _segments(hx):
        n = e0 - s0 + 1
        if s0 - 1 >= 0:
            f["ncap"][s0 - 1] = 1
        if e0 + 1 < L:
            f["ccap"][e0 + 1] = 1
        for k in range(3):
            if s0 + k <= e0:
                f[f"n{k + 1}"][s0 + k] = 1
            if e0 - k >= s0:
                f[f"c{k + 1}"][e0 - k] = 1
        for i in range(s0, e0 + 1):
            f["helix_len"][i] = n
            f["helix_pos"][i] = (i - s0) / max(n - 1, 1)
            f["helix_int"][i] = float(i - s0 >= 3 and e0 - i >= 3)
    st = ss3 == "E"
    segs = _segments(st)
    seg_id = np.full(L, -1)
    for k, (s0, e0) in enumerate(segs):
        seg_id[s0:e0 + 1] = k
        f["strand_len"][s0:e0 + 1] = e0 - s0 + 1
    ca = p.xyz14[:, 1].astype(np.float64)
    tca = cKDTree(ca)
    for i in np.flatnonzero(st):
        part = {seg_id[j] for j in tca.query_ball_point(ca[i], 5.5) if st[j] and seg_id[j] != seg_id[i] and abs(j - i) > 2}
        f["strand_centre" if len(part) >= 2 else "strand_edge"][i] = 1
    for s0, e0 in _segments(ss3 == "L"):
        f["loop_len"][s0:e0 + 1] = e0 - s0 + 1
    out.update(f)
    # termini (chain-wise)
    idx = np.arange(L)
    dn = np.zeros(L); dc = np.zeros(L)
    for ch in np.unique(p.chain):
        m = np.flatnonzero(p.chain == ch)
        dn[m] = idx[m] - m[0]; dc[m] = m[-1] - idx[m]
    out.update(dist_nterm=dn, dist_cterm=dc, dist_term=np.minimum(dn, dc), rel_pos=dn / np.maximum(dn + dc, 1),
               term3=(np.minimum(dn, dc) < 3).astype(int), term6=(np.minimum(dn, dc) < 6).astype(int))
    # pLDDT / B
    b = p.bfactor.astype(np.float64)
    win = np.array([b[max(0, i - 3):i + 4].mean() for i in range(L)])
    wmin = np.array([b[max(0, i - 3):i + 4].min() for i in range(L)])
    out.update(plddt=b, plddt_win=win, plddt_min_win=wmin, plddt_z=(b - b.mean()) / (b.std() + 1e-6))
    # charges
    qpts, qval = [], []
    for i in range(L):
        aa = p.seq[i]
        if aa in CHG_ATOMS:
            pts = [x for x in (_atom(p, i, n) for n in CHG_ATOMS[aa]) if x is not None]
            if pts:
                qpts.append((i, np.mean(pts, 0), CHARGE[aa], np.asarray(pts)))
    q_site = np.array([CHARGE.get(x, 0) for x in p.seq], float)
    q8, q12, npos6, nneg6, sbp, same6 = (np.zeros(L) for _ in range(6))
    for i in range(L):
        for (j, c, q, pts) in qpts:
            if j == i:
                continue
            d = np.linalg.norm(c - cen[i])
            if d < 8:
                q8[i] += q
            if d < 12:
                q12[i] += q
            if d < 6:
                npos6[i] += q > 0; nneg6[i] += q < 0
                if q_site[i] != 0 and np.sign(q) == np.sign(q_site[i]):
                    same6[i] += 1
    for (i, ci, qi, pi) in qpts:
        for (j, cj, qj, pj) in qpts:
            if j != i and qi * qj < 0 and np.min(np.linalg.norm(pi[:, None] - pj[None], axis=-1)) < 4.0:
                sbp[i] += 1
    out.update(q_site=q_site, q_env8=q8, q_env12=q12, n_pos6=npos6, n_neg6=nneg6, sb_partners=sbp, same_q6=same6)
    # aromatics / cation-pi
    rings = {i: _ring(p, i) for i in range(L) if p.seq[i] in AROMATIC}
    rings = {i: v for i, v in rings.items() if v[0] is not None}
    cats = {i: _atom(p, i, CATION_ATOM[p.seq[i]]) for i in range(L) if p.seq[i] in CATION_ATOM}
    cats = {i: v for i, v in cats.items() if v is not None}
    aro6, aro8, stack, catpi = (np.zeros(L) for _ in range(4))
    for i in range(L):
        for j, (c, nrm) in rings.items():
            if j == i:
                continue
            d = np.linalg.norm(c - cen[i])
            aro6[i] += d < 6.5; aro8[i] += d < 8
        if i in rings:
            ci, ni = rings[i]
            for j, (c, nrm) in rings.items():
                if j != i:
                    d = np.linalg.norm(c - ci)
                    if d < 7.0:
                        stack[i] += abs(float(ni @ nrm)) * 0.5 + 0.5   # parallel 1.0, T-shaped 0.5
            for j, x in cats.items():
                if j != i and np.linalg.norm(x - ci) < 6.0:
                    # cation above the ring face (angle to normal < 45 deg) counts fully, else half
                    v = (x - ci) / np.linalg.norm(x - ci)
                    catpi[i] += 1.0 if abs(float(v @ ni)) > 0.7 else 0.5
        if i in cats:
            for j, (c, nrm) in rings.items():
                if j != i and np.linalg.norm(cats[i] - c) < 6.0:
                    v = (cats[i] - c) / np.linalg.norm(cats[i] - c)
                    catpi[i] += 1.0 if abs(float(v @ nrm)) > 0.7 else 0.5
    out.update(n_aro6=aro6, n_aro8=aro8, aro_stack=stack, catpi=catpi)
    # backbone
    phi, psi = np.nan_to_num(a["phi"], nan=-180.0), np.nan_to_num(a["psi"], nan=180.0)
    out.update(phi=phi, psi=psi, basin=_basin(phi, psi))
    om = a.get("omega")
    out["cis_prev"] = (np.abs(np.nan_to_num(om, nan=180.0)) < 30).astype(int) if om is not None else np.zeros(L, int)
    # domain-level
    out.update(L=np.full(L, L), frac_helix=np.full(L, float(np.mean(ss3 == "H"))),
               frac_strand=np.full(L, float(np.mean(ss3 == "E"))), mean_plddt=np.full(L, float(b.mean())))
    d = pd.DataFrame(out)
    d.insert(0, "structure_abs", str(path))
    return d


def _one(path):
    try:
        return site_descriptors(path)
    except Exception as e:  # noqa: BLE001
        print("FAILED", path, e, flush=True)
        return None


def site_table(structures, name: str, n_jobs: int = 32, refresh: bool = False) -> pd.DataFrame:
    """Site descriptors for all residues of the given structures (cached at data/features/extra_sites_<name>.parquet).
    Returned with a `structure` column equal to the input strings."""
    cache = CACHE / f"extra_sites_{name}.parquet"
    structures = sorted(set(map(str, structures)))
    old = pd.read_parquet(cache) if cache.exists() and not refresh else None
    todo = structures if old is None else [s for s in structures if s not in set(old.structure)]
    if todo:
        paths = [s if Path(s).is_absolute() else str(ROOT / s) for s in todo]
        with Pool(n_jobs) as pool:
            res = pool.map(_one, paths, chunksize=1)
        parts = [r.assign(structure=s) for s, r in zip(todo, res) if r is not None]
        new = pd.concat(parts, ignore_index=True)
        new["pdb_resnum"] = new["pdb_resnum"].astype(str)
        old = new if old is None else pd.concat([old, new], ignore_index=True)
        old.to_parquet(cache, index=False)
    return old[old.structure.isin(structures)].reset_index(drop=True)


# ----------------------------------------------------------------------------- candidate per-variant features
def candidate_features(d: pd.DataFrame) -> pd.DataFrame:
    """Candidate interpretable terms (positive value = predicted destabilising when the weight is positive).
    Needs wt, mut, x_<site descriptor> columns (prefix x_ from site_table), dvol (V_wt - V_mut, A^3/100) and
    s2_dE_lj_rep. Returns a frame of new columns (same index)."""
    o = pd.DataFrame(index=d.index)
    wt, mut = d["wt"], d["mut"]
    m = lambda tab, s: s.map(tab).astype(float)
    q = lambda s: s.map(CHARGE).fillna(0).astype(float)
    hx = (d["x_ss8"] == "H").astype(float)
    hint = d["x_helix_int"].astype(float)
    nterm = (d["x_n1"] + d["x_n2"] + d["x_n3"]).clip(0, 1).astype(float)
    cterm = (d["x_c1"] + d["x_c2"] + d["x_c3"]).clip(0, 1).astype(float)
    strand = (d["x_ss3"] == "E").astype(float)
    bur = (1 - d["x_rsa"].clip(0, 1)).astype(float)
    posphi = (d["x_phi"] > 0).astype(float)
    # secondary-structure propensities
    dh = m(HELIX_PS, mut) - m(HELIX_PS, wt)
    o["helix_prop_int"] = dh * hint
    o["helix_prop_all"] = dh * hx
    o["helix_prop_exp"] = dh * hx * (1 - bur)
    db = m(BETA_MK, wt) - m(BETA_MK, mut)
    o["beta_prop"] = db * strand
    o["beta_prop_centre"] = db * d["x_strand_centre"].astype(float)
    o["beta_prop_exp"] = db * strand * (1 - bur)
    # helix capping / dipole
    o["ncap_pref"] = (m(NCAP_DB, mut) - m(NCAP_DB, wt)) * d["x_ncap"].astype(float)
    o["ncap_any"] = d["x_ncap"].astype(float) * (wt != mut).astype(float) * wt.isin(list("STDN")).astype(float)
    o["helix_dipole"] = (q(mut) - q(wt)) * (nterm - cterm)
    o["ccap_gly_loss"] = d["x_ccap"].astype(float) * (wt == "G").astype(float) * (mut != "G").astype(float)
    # backbone entropy / Gly / Pro
    o["to_gly_ss"] = (mut == "G").astype(float) * (1 - posphi) * (hx + strand).clip(0, 1)
    o["to_gly_loop"] = (mut == "G").astype(float) * (1 - posphi) * (1 - (hx + strand).clip(0, 1))
    o["from_gly_nonposphi"] = (wt == "G").astype(float) * (1 - posphi)
    o["pro_posphi_exp"] = (mut == "P").astype(float) * posphi * (1 - bur)
    o["to_pro_strand"] = (mut == "P").astype(float) * strand
    o["to_pro_helix_int"] = (mut == "P").astype(float) * hint
    o["from_pro_any"] = (wt == "P").astype(float)
    # overpacking (small -> large) with burial / packing / strand context
    gain = (-d["dvol"]).clip(lower=0).astype(float)
    rep = d["s2_dE_lj_rep"].astype(float).clip(lower=0)
    o["overpack_bur"] = gain * bur
    o["overpack_strand_bur"] = gain * bur * strand
    o["overpack_dense"] = gain * (d["x_ncb14"].astype(float) / 40.0)
    o["rep_lin20_bur"] = rep.clip(upper=20) * bur
    o["rep_log_strand"] = np.log1p(rep) * strand
    o["rep_log_bur"] = np.log1p(rep) * bur
    # hydrophobic transfer x burial (Fauchere-Pliska octanol scale, kcal/mol; >0 = hydrophobicity lost)
    from .features_simple import HYD
    dhyd = m(HYD, wt) - m(HYD, mut)
    o["bur_dhyd"] = dhyd * bur
    o["bur2_dhyd"] = dhyd * bur ** 2
    o["polar_burial"] = (-m(HYD, mut)).clip(lower=0) * bur * (d["x_rsa"] < 0.25).astype(float)
    # cavity: volume loss into a site with little pre-existing void
    loss = d["dvol"].clip(lower=0).astype(float)
    o["cavity_nonvoid"] = loss * bur * (1 - d["x_void_frac"].astype(float))
    # specific side-chain interactions lost (WT-structure geometry)
    aro_w, aro_m = wt.isin(list("FWY")), mut.isin(list("FWY"))
    cat_w, cat_m = wt.isin(list("KR")), mut.isin(list("KR"))
    o["catpi_loss"] = d["x_catpi"].astype(float) * ((aro_w & ~aro_m) | (cat_w & ~cat_m)).astype(float)
    o["aro_stack_loss"] = d["x_aro_stack"].astype(float) * (aro_w & ~aro_m).astype(float)
    qw, qm = q(wt), q(mut)
    o["saltbridge_loss"] = d["x_sb_partners"].astype(float) * ((qw != 0) & (qm != qw)).astype(float) * (1 + (qm == -qw).astype(float))
    o["charge_like_env"] = qm * d["x_q_env8"].astype(float) - qw * d["x_q_env8"].astype(float)   # >0: like charges
    polar_m = mut.isin(list("STNQHDEKRYW"))
    o["schb_loss"] = d["x_hb_sc_any"].astype(float).clip(upper=4) * (~polar_m).astype(float)
    o["schb_loss_any"] = d["x_hb_sc_any"].astype(float).clip(upper=4) * (wt != mut).astype(float)
    # site main effects (site sensitivity not captured by RSA burial)
    nonc = (wt != mut).astype(float)
    # pLDDT: design models (and crystal structures) carry no pLDDT; treat B-factor columns of non-AF models as "no
    # information" (pLDDT 100) so that the low-pLDDT terms are not a designed-protein indicator
    is_af = d["x_mean_plddt"].astype(float).between(30, 100) & (d["x_plddt"].astype(float).max() <= 100)
    if "struct_source" in d:
        is_af &= d["struct_source"].astype(str).str.contains("af", case=False) | d["struct_source"].isna()
    plddt = d["x_plddt"].astype(float).where(is_af & (d["x_mean_plddt"] > 0), 100.0)
    o["site_ncb14"] = d["x_ncb14"].astype(float) / 40.0 * nonc
    o["site_hse_up"] = d["x_hse_up"].astype(float) / 30.0 * nonc
    o["site_natom8"] = d["x_natom8"].astype(float) / 150.0 * nonc
    o["site_void"] = d["x_void_frac"].astype(float) * nonc
    o["site_lowplddt"] = (1 - plddt.clip(0, 100) / 100.0) * nonc
    o["site_term3"] = d["x_term3"].astype(float) * nonc
    o["site_term_dist"] = np.exp(-d["x_dist_term"].astype(float) / 3.0) * nonc
    o["site_hbsc"] = d["x_hb_sc_any"].astype(float).clip(upper=4) * nonc
    o["site_ncap"] = d["x_ncap"].astype(float) * nonc
    o["site_strand_centre"] = d["x_strand_centre"].astype(float) * nonc
    o["site_scsc"] = d["x_scsc_contacts"].astype(float).clip(upper=30) / 30.0 * nonc
    o["scsc_loss"] = d["x_scsc_contacts"].astype(float).clip(upper=30) / 30.0 * loss
    o["scsc_bur"] = d["x_scsc_contacts"].astype(float).clip(upper=30) / 30.0 * bur * nonc
    o["site_hbsc_bur"] = d["x_hb_sc_any"].astype(float).clip(upper=4) * bur * nonc
    # D15 state-function counterparts of the two site-only terms (antisymmetric on a fixed structure)
    polar_w = wt.isin(list("STNQHDEKRYW")).astype(float)
    o["scsc_dvol"] = d["x_scsc_contacts"].astype(float).clip(upper=30) / 30.0 * d["dvol"].astype(float)
    o["hbsc_dpolar"] = d["x_hb_sc_any"].astype(float).clip(upper=4) * bur * (polar_w - polar_m.astype(float))
    # flags used for per-context scaling of the B5 core (handled in scripts/93)
    o["flag_helix"], o["flag_strand"], o["flag_buried"] = hx, strand, (d["x_rsa"] < 0.25).astype(float)
    o["flag_term3"] = d["x_term3"].astype(float)
    o["flag_lowplddt"] = (1 - plddt.clip(0, 100) / 100.0)
    o["flag_loop"] = 1 - (hx + strand).clip(0, 1)
    return o
