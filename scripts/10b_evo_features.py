#!/usr/bin/env python
"""Evolution-aware extension ('PhysDDG+evo'), step 2: interpretable per-site / per-variant profile features.

Input: data/features/evo_units.parquet (scripts/10a) + the cached hhblits MSAs data/work/ddgun/hhblits_cache/<key>.blast
(query-anchored, query = row 0, one column per residue of the structure chain, in DDGun's residue order `nums`).
No learned model, no embedding: counting statistics of the alignment only.

Alignment statistics (per unit)
  sequence weights  w_i = 1 / #{j : id(i,j) >= 0.8}, id = identical aligned residues / min(non-gap length of i, j);
                    self always counted (query included)
  evo_nseq          number of sequences in the (hhblits-filtered: -id 90 -diff 1000) result MSA, incl. query
  evo_neff          sum_i w_i (global effective number of sequences); evo_neff_L = evo_neff / L
Per site (column)
  f_a      weighted amino-acid frequencies over non-gap rows
  neff_s   evo_neff_site = sum of weights of non-gap rows (>= 1 because of the query)
  p_a      pseudocount profile (PSI-BLAST, Altschul et al. 1997): p = (alpha f + beta g) / (alpha + beta),
           g_a = sum_b f_b q(a|b) (BLOSUM62 conditional), alpha = neff_s - 1, beta = 10.
           With no homologues p = q(.|wt), so the log-odds below reduce to BLOSUM62.
Feature columns (ALL ORIENTED SO THAT HIGHER = MORE TOLERATED, i.e. expected to be LESS destabilising;
in a ddG model with positive = destabilising their fitted weights must be <= 0):
  evo_lor       log p(mut) - log p(wt)                                        (site-independent log-ratio, nats)
  evo_pssm      log(p(mut)/bg(mut)) - log(p(wt)/bg(wt))                        (PSSM log-odds difference; = evo_lor
                + a pure (wt,mut) term that the fitted reference energies absorb)
  evo_entropy   Shannon entropy of f at the site (nats; 0 = invariant)          (site-level)
  evo_naa       exp(evo_entropy), effective number of amino acids at the site   (site-level)
  evo_nonwt     1 - f(wt)                                                       (site-level)
  evo_gap       weighted gap fraction at the site                               (site-level)
  evo_hyd_lor   log P_class(mut) - log P_class(wt), classes apolar AVILMFWC / polar STNQYHGP / charged DEKR (from p)
  evo_vol_lor   same with volume classes GAS / CDPNT / EVQH / MILKR / FYW
  evo_hyd_z     -|KD(mut) - <KD>_site| / max(sd_KD, 0.5)    (Kyte-Doolittle; distance from the site's hydropathy range)
  evo_vol_z     -|V(mut) - <V>_site| / max(sd_V, 10 A^3)    (Zamyatnin volumes)
  evo_blprof    sum_a f_a [B62(a,mut) - B62(a,wt)]           (DDGun's profile-weighted BLOSUM62 substitution score)
  evo_blosum    B62(wt,mut)                                  (no profile; reference)
  evo_gate      clip(log10(evo_neff_site) / 2, 0, 1): 0 with only the query, 0.5 at neff 10, 1 at >= 100 (fixed a priori)
Ssym reverse rows are scored on the mutant structure with that structure's own MSA (as DDGun).
Output: data/features/evo_<dataset>.parquet keyed (protein_id, chain, pdb_resnum, wt, mut) + structure;
        results/evo_extension/msa_coverage.csv (one row per unit).
  PYTHONPATH=. .venv/bin/python scripts/10b_evo_features.py
"""
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
from Bio.Align import substitution_matrices  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from physddg import eval as E  # noqa: E402

CACHE = ROOT / "data/work/ddgun/hhblits_cache"
FEAT = ROOT / "data/features"
OUT = ROOT / "results/evo_extension"
AA = "ARNDCQEGHILKMFPSTWYV"
AI = {a: i for i, a in enumerate(AA)}
# BLOSUM62 background frequencies (Henikoff & Henikoff 1992, blosum62.qij marginals)
BG = np.array([.074, .052, .045, .054, .025, .034, .054, .074, .026, .068, .099, .058, .025, .047, .039, .057, .051,
               .013, .032, .073])
BG = BG / BG.sum()
_B = substitution_matrices.load("BLOSUM62")
B62 = np.array([[_B[a][b] for b in AA] for a in AA], float)
QJ = BG[:, None] * BG[None, :] * 2.0 ** (B62 / 2.0)
QJ = QJ / QJ.sum()
QC = QJ / QJ.sum(0, keepdims=True)            # QC[a, b] = q(a | b)
KD = np.array([1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2])
VOL = np.array([88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7,
                89.0, 116.1, 227.8, 193.6, 140.0])
HYD_CLS = {**{a: 0 for a in "AVILMFWC"}, **{a: 1 for a in "STNQYHGP"}, **{a: 2 for a in "DEKR"}}
VOL_CLS = {**{a: 0 for a in "GAS"}, **{a: 1 for a in "CDPNT"}, **{a: 2 for a in "EVQH"}, **{a: 3 for a in "MILKR"},
           **{a: 4 for a in "FYW"}}
BETA = 10.0
BENCHES = ["megascale", "fireprot", "s669", "s461", "ssym", "s2648", "domainome", "abundance_dms"]


def read_msa(key: str, L: int) -> np.ndarray:
    """int8 matrix [N, L]: 0..19 amino acids, 20 = gap / unknown. Row 0 = query."""
    seqs, cur = [], []
    for line in open(CACHE / f"{key}.blast"):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    lut = np.full(256, 20, np.int8)
    for a, i in AI.items():
        lut[ord(a)] = i
    seqs = [s for s in seqs if len(s) == L]
    return lut[np.frombuffer("".join(seqs).encode(), np.uint8)].reshape(len(seqs), L)


def seq_weights(M: np.ndarray, thr: float = 0.8) -> np.ndarray:
    N, L = M.shape
    oh = np.zeros((N, L * 20), np.float32)
    r, c = np.nonzero(M < 20)
    oh[r, c * 20 + M[r, c]] = 1.0
    ln = (M < 20).sum(1).astype(np.float32)
    ident = (oh @ oh.T) / np.maximum(np.minimum(ln[:, None], ln[None, :]), 1.0)
    np.fill_diagonal(ident, 1.0)
    return 1.0 / (ident >= thr).sum(1)


def site_profiles(key: str, seq: str) -> dict:
    L = len(seq)
    M = read_msa(key, L)
    assert M.shape[0] >= 1 and "".join(AA[x] if x < 20 else "X" for x in M[0]) == "".join(a if a in AI else "X" for a in seq)
    w = seq_weights(M)
    oh = np.zeros((M.shape[0], L, 21))
    oh[np.arange(M.shape[0])[:, None], np.arange(L)[None, :], M] = 1.0
    cnt = np.einsum("n,nla->la", w, oh)               # [L, 21]
    neff_s = cnt[:, :20].sum(1)
    gap = cnt[:, 20] / w.sum()
    f = cnt[:, :20] / np.maximum(neff_s[:, None], 1e-9)
    g = f @ QC.T                                       # g_a = sum_b f_b q(a|b)
    alpha = np.maximum(neff_s - 1.0, 0.0)[:, None]
    p = (alpha * f + BETA * g) / (alpha + BETA)
    ent = -(f * np.log(np.where(f > 0, f, 1.0))).sum(1)
    mh, mv = f @ KD, f @ VOL
    sh = np.sqrt(np.maximum(f @ KD ** 2 - mh ** 2, 0))
    sv = np.sqrt(np.maximum(f @ VOL ** 2 - mv ** 2, 0))
    hyd_p = np.stack([p[:, [AI[a] for a in AA if HYD_CLS[a] == k]].sum(1) for k in range(3)], 1)
    vol_p = np.stack([p[:, [AI[a] for a in AA if VOL_CLS[a] == k]].sum(1) for k in range(5)], 1)
    blp = f @ B62                                      # [L, 20]: sum_a f_a B(a, x)
    return dict(L=L, nseq=M.shape[0], neff=float(w.sum()), f=f, p=p, neff_s=neff_s, gap=gap, ent=ent, mh=mh, mv=mv,
                sh=sh, sv=sv, hyd_p=hyd_p, vol_p=vol_p, blp=blp)


def unit_features(rec: dict, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """rows: variant rows of one (structure, chain); returns feature frame aligned with rows.index + unit summary."""
    nums = json.loads(rec["nums"])
    pos = {n: i for i, n in enumerate(nums)}
    P = site_profiles(rec["key"], rec["seq"])
    out = pd.DataFrame(index=rows.index, dtype=float)
    i = rows["pdb_resnum"].map(pos)
    ok = i.notna() & rows.wt.isin(list(AI)) & rows.mut.isin(list(AI))
    ok &= np.array([bool(o) and rec["seq"][int(k)] == w for o, k, w in zip(ok, i.fillna(0), rows.wt)], bool)
    ii = i[ok].astype(int).to_numpy()
    iw, im = rows.wt[ok].map(AI).to_numpy(), rows.mut[ok].map(AI).to_numpy()
    p, f = P["p"], P["f"]
    lp = np.log(p)
    col = {}
    col["evo_lor"] = lp[ii, im] - lp[ii, iw]
    col["evo_pssm"] = col["evo_lor"] - np.log(BG[im]) + np.log(BG[iw])
    col["evo_entropy"] = P["ent"][ii]
    col["evo_naa"] = np.exp(P["ent"][ii])
    col["evo_nonwt"] = 1.0 - f[ii, iw]
    col["evo_gap"] = P["gap"][ii]
    hc = np.array([HYD_CLS[a] for a in AA]); vc = np.array([VOL_CLS[a] for a in AA])
    col["evo_hyd_lor"] = np.log(P["hyd_p"][ii, hc[im]]) - np.log(P["hyd_p"][ii, hc[iw]])
    col["evo_vol_lor"] = np.log(P["vol_p"][ii, vc[im]]) - np.log(P["vol_p"][ii, vc[iw]])
    col["evo_hyd_z"] = -np.abs(KD[im] - P["mh"][ii]) / np.maximum(P["sh"][ii], 0.5)
    col["evo_vol_z"] = -np.abs(VOL[im] - P["mv"][ii]) / np.maximum(P["sv"][ii], 10.0)
    col["evo_blprof"] = P["blp"][ii, im] - P["blp"][ii, iw]
    col["evo_blosum"] = B62[iw, im]
    col["evo_neff_site"] = P["neff_s"][ii]
    col["evo_gate"] = np.clip(np.log10(P["neff_s"][ii]) / 2.0, 0.0, 1.0)
    for k, v in col.items():
        out.loc[ok, k] = v
    out["evo_nseq"], out["evo_neff"], out["evo_neff_L"] = P["nseq"], P["neff"], P["neff"] / P["L"]
    out["evo_found"] = ok.to_numpy()
    summ = dict(structure=rec["structure"], chain=rec["chain"], key=rec["key"], L=P["L"], nseq=P["nseq"], neff=P["neff"],
                neff_L=P["neff"] / P["L"], neff_site_median=float(np.median(P["neff_s"])),
                frac_sites_neff10=float((P["neff_s"] >= 10).mean()), datasets=rec.get("dataset", ""))
    return out, summ


def bench_rows(b: str) -> pd.DataFrame:
    d = E.load_benchmark(b)
    if b == "megascale":
        c = pd.read_csv(ROOT / "data/processed/megascale_censored.csv", low_memory=False)
        c["pdb_resnum"] = E._norm_resnum(c["pdb_resnum"])
        d = pd.concat([d, c], ignore_index=True)
    d = d[d.structure.notna()]
    d = d.assign(chain=d.chain.astype(str))[list(E.KEYS) + ["structure"]].drop_duplicates()
    return d.reset_index(drop=True)


def _job(rec, rows):
    try:
        return unit_features(rec, rows)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"FAIL {rec['structure']} {rec['chain']}: {e!r}\n")
        return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U = pd.read_parquet(FEAT / "evo_units.parquet")
    U = U[U.has_msa].set_index(["structure", "chain"])
    summ = {}
    for b in BENCHES:
        d = bench_rows(b)
        groups = [(U.loc[(s, c)].to_dict() | {"structure": s, "chain": c}, g) for (s, c), g in d.groupby(["structure", "chain"])
                  if (s, c) in U.index]
        res = Parallel(n_jobs=24)(delayed(_job)(r, g) for r, g in groups)
        feats = pd.concat([x for x, _ in res if x is not None])
        for _, s in res:
            if s is not None:
                summ[(s["structure"], s["chain"])] = s
        o = d.join(feats, how="left")
        o["evo_found"] = o["evo_found"].fillna(False).astype(bool)
        o.to_parquet(FEAT / f"evo_{b}.parquet", index=False)
        print(f"{b}: {len(o)} rows, {o.protein_id.nunique()} proteins, feature coverage {o.evo_found.mean():.4f}", flush=True)
    S = pd.DataFrame(list(summ.values()))
    ms = pd.read_parquet(ROOT / "data/processed/megascale.parquet", columns=["structure", "protein_id", "domain_origin"])
    org = ms.drop_duplicates("structure").set_index("structure").domain_origin
    S["origin"] = S.structure.map(org).fillna("natural")   # only Megascale has designed domains
    S.to_csv(OUT / "msa_coverage.csv", index=False)
    coverage_summary(S)


THR = (2, 10, 30, 100, 1000)


def coverage_summary(S: pd.DataFrame):
    """Fraction of units (structure chains) with global N_eff >= thresholds, per dataset (and origin for Megascale)."""
    rows = []
    for b in BENCHES:
        g0 = S[S.datasets.str.split(",").map(lambda x: b in x)]
        parts = [("all", g0)] + ([("natural", g0[g0.origin == "natural"]), ("designed", g0[g0.origin == "designed"])]
                                 if b == "megascale" else [])
        for tag, g in parts:
            r = {"dataset": b, "origin": tag, "n_units": len(g), "median_neff": g.neff.median(),
                 "median_frac_sites_neff10": g.frac_sites_neff10.median()}
            r.update({f"frac_neff>={t}": (g.neff >= t).mean() for t in THR})
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "msa_coverage_summary.csv", index=False)
    print(T.round(3).to_string(index=False))
    sh = S[S.neff < 10].sort_values("neff")
    print(f"\nshallow units (N_eff < 10): {len(sh)}")
    print(sh[["structure", "chain", "L", "nseq", "neff", "origin", "datasets"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
