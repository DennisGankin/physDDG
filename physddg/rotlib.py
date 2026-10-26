"""PDB-derived backbone-dependent rotamer library, Ramachandran potentials, side-chain entropies and
environment-dependent amino-acid potentials (derivation: scripts/40_build_rotlib.py; data: physddg/data/*.npz).

Rotamer definition (Dunbrack-style)
- sp3 chis: 3 bins g+ [0,120), t [120,240), g- [240,360). Pro: chi1 sign (2 puckers), chi2 follows (1 bin).
- Non-rotameric terminal chis (sp3-sp2: Asn/Asp chi2, Gln/Glu chi3, Phe/Tyr/His/Trp chi2) are discretised
  finely in the spirit of Dunbrack 2010: 30 deg bins over the symmetry-folded range (Asp/Glu [-90,90),
  Phe/Tyr [-30,150): 6 bins; Asn/Gln/His/Trp [-180,180): 12 bins). Chi means/sds are kept per bin.
- Backbone dependence: probabilities, chi means and sds on a 10 deg (phi, psi) grid (centres -175..175),
  estimated by von Mises kernel smoothing (sigma 10 deg) with hierarchical back-off to a sigma-30 deg estimate
  and then to the backbone-independent distribution (pseudo-counts ALPHA for probabilities, BETA for moments).
- Rotamers with < MIN_COUNT observations are pruned (their mass is dropped, probabilities not renormalised).
All energies/entropies are dimensionless (units of kT / nats); multiply by kT = 0.593 kcal/mol at 298 K.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import topology as T

DATA = Path(__file__).resolve().parent / "data"
GRID = np.arange(-175.0, 180.0, 10.0)          # 36 bin centres
SIG_FINE, SIG_COARSE = 10.0, 30.0
ALPHA, BETA, MIN_COUNT = 10.0, 5.0, 5
SD_FLOOR = 3.0


def chi_spec(aa: str) -> list[tuple[float, float, int]]:
    """Per chi: (lo, period, nbins); a value is folded into [lo, lo+period) and binned uniformly."""
    spec = []
    for k in range(1, T.N_CHI[aa] + 1):
        if aa == "P":
            spec.append((-180.0, 360.0, 2) if k == 1 else (-180.0, 360.0, 1))
        elif T.NONROTAMERIC_CHI.get(aa) == k:
            if (aa, k) in T.CHI_SYMMETRIC:
                spec.append(((-30.0 if aa in "FY" else -90.0), 180.0, 6))
            else:
                spec.append((-180.0, 360.0, 12))
        else:
            spec.append((0.0, 360.0, 3))
    return spec


def fold(x, lo, period):
    return (np.asarray(x) - lo) % period + lo


def chi_bins(aa: str, chis: np.ndarray) -> np.ndarray:
    """[N, nchi] bin index per chi (chis in degrees, [N, >=nchi])."""
    spec = chi_spec(aa)
    out = np.zeros((len(chis), len(spec)), np.int64)
    for k, (lo, per, nb) in enumerate(spec):
        out[:, k] = np.minimum(((fold(chis[:, k], lo, per) - lo) / (per / nb)).astype(np.int64), nb - 1)
    return out


def rot_index(aa: str, bins: np.ndarray) -> np.ndarray:
    """Mixed-radix rotamer id from per-chi bins."""
    idx = np.zeros(len(bins), np.int64)
    for k, (_, _, nb) in enumerate(chi_spec(aa)):
        idx = idx * nb + bins[:, k]
    return idx


def _vm(grid, x, sig):
    """von Mises kernel weights [len(grid), len(x)] (peak 1)."""
    kappa = 1.0 / np.radians(sig) ** 2
    return np.exp(kappa * (np.cos(np.radians(grid[:, None] - x[None])) - 1.0))


def _smooth_sums(phi, psi, feats, sig):
    """sum_k w_phi(g1,k) w_psi(g2,k) feats[k, m]  -> [36, 36, m] (torch on GPU when available)."""
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.as_tensor(_vm(GRID, phi, sig), device=dev, dtype=torch.float64)
        b = torch.as_tensor(_vm(GRID, psi, sig), device=dev, dtype=torch.float64)
        f = torch.as_tensor(feats, device=dev, dtype=torch.float64)
        out = torch.zeros(36, 36, f.shape[1], device=dev, dtype=torch.float64)
        for s in range(0, len(phi), 20000):
            w = a[:, None, s:s + 20000] * b[None, :, s:s + 20000]
            out += torch.einsum("ijk,km->ijm", w, f[s:s + 20000])
        return out.cpu().numpy()
    except ImportError:
        w = _vm(GRID, phi, sig)[:, None] * _vm(GRID, psi, sig)[None]
        return np.einsum("ijk,km->ijm", w, feats)


# ----------------------------------------------------------------------------- derivation
def derive_rotlib(df) -> dict[str, np.ndarray]:
    """Backbone-dependent rotamer library from per-residue records (columns aa, phi, psi, chi1..4)."""
    out = {"grid": GRID, "params": np.array([SIG_FINE, SIG_COARSE, ALPHA, BETA, MIN_COUNT, SD_FLOOR])}
    for aa in T.AA1:
        nchi = T.N_CHI[aa]
        if nchi == 0:
            continue
        d = df[df.aa == aa]
        chis = d[[f"chi{k}" for k in range(1, 5)]].to_numpy(np.float64)
        ok = np.isfinite(chis[:, :nchi]).all(1)
        d, chis = d[ok], chis[ok]
        spec = chi_spec(aa)
        X = np.stack([fold(chis[:, k], lo, per) for k, (lo, per, _) in enumerate(spec)], 1)
        rid = rot_index(aa, chi_bins(aa, chis))
        nrot_all = int(np.prod([s[2] for s in spec]))
        cnt = np.bincount(rid, minlength=nrot_all)
        keep = np.flatnonzero(cnt >= MIN_COUNT)
        remap = -np.ones(nrot_all, np.int64)
        remap[keep] = np.arange(len(keep))
        sel = remap[rid] >= 0
        r, X, phi, psi = remap[rid[sel]], X[sel], d.phi.to_numpy()[sel], d.psi.to_numpy()[sel]
        n, R = len(r), len(keep)
        oh = np.zeros((n, R))
        oh[np.arange(n), r] = 1.0
        feats = np.concatenate([oh, (oh[:, :, None] * X[:, None]).reshape(n, -1),
                                (oh[:, :, None] * X[:, None] ** 2).reshape(n, -1)], 1)
        tot = len(rid)                                            # incl. pruned: their mass is dropped
        g_cnt = feats[:, :R].sum(0)
        g_p = g_cnt / tot
        g_m1 = feats[:, R:R + R * nchi].sum(0).reshape(R, nchi) / g_cnt[:, None]
        g_m2 = feats[:, R + R * nchi:].sum(0).reshape(R, nchi) / g_cnt[:, None]
        levels = {}
        for sig in (SIG_COARSE, SIG_FINE):
            S = _smooth_sums(phi, psi, feats, sig)
            Nall = _smooth_sums(d.phi.to_numpy(), d.psi.to_numpy(), np.ones((len(d), 1)), sig)[..., 0]
            levels[sig] = (S[..., :R], S[..., R:R + R * nchi].reshape(36, 36, R, nchi),
                           S[..., R + R * nchi:].reshape(36, 36, R, nchi), Nall)
        p, m1, m2 = g_p, g_m1, g_m2
        for sig in (SIG_COARSE, SIG_FINE):
            C, S1, S2, Nall = levels[sig]
            p = (C + ALPHA * p) / (Nall[..., None] + ALPHA)
            m1 = (S1 + BETA * m1) / (C[..., None] + BETA)
            m2 = (S2 + BETA * m2) / (C[..., None] + BETA)
        sd = np.sqrt(np.maximum(m2 - m1 ** 2, SD_FLOOR ** 2))
        g_sd = np.sqrt(np.maximum(g_m2 - g_m1 ** 2, SD_FLOOR ** 2))
        bins = np.stack(np.unravel_index(keep, [s[2] for s in spec]), 1)
        out.update({
            f"{aa}_spec": np.array(spec, np.float64), f"{aa}_bins": bins.astype(np.int8),
            f"{aa}_count": g_cnt.astype(np.int64), f"{aa}_n": np.array(tot),
            f"{aa}_logp": np.log(np.maximum(p, 1e-12)).astype(np.float16),
            f"{aa}_mean": m1.astype(np.float16), f"{aa}_sd": sd.astype(np.float16),
            f"{aa}_p_global": g_p, f"{aa}_mean_global": g_m1, f"{aa}_sd_global": g_sd,
        })
    return out


SS3 = "HEL"
CN_NBINS = 8
RSA_EDGES = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, np.inf])
ENV_ALPHA = 50.0


def bb_basin(phi, psi) -> np.ndarray:
    """0 = alpha-R (phi<0, -100<psi<=50), 1 = beta/PPII (phi<0 otherwise), 2 = phi>0 (alpha-L etc.)."""
    phi, psi = np.asarray(phi), np.asarray(psi)
    return np.where(phi > 0, 2, np.where((psi > -100) & (psi <= 50), 0, 1))


def derive_pdbstats(df_bb, df_sc, lib) -> dict[str, np.ndarray]:
    """Ramachandran maps, entropies, environment potentials. df_bb: backbone-quality residues; df_sc: side-chain
    quality residues (for entropies)."""
    out: dict[str, np.ndarray] = {"grid": GRID}
    # ---- Ramachandran (kernel density, sigma 10 deg, 10 deg grid)
    nxt_pro = (df_bb.next_aa == "P").to_numpy()
    aa = df_bb.aa.to_numpy()
    phi, psi = df_bb.phi.to_numpy(np.float64), df_bb.psi.to_numpy(np.float64)

    def density(m):
        S = _smooth_sums(phi[m], psi[m], np.ones((int(m.sum()), 1)), SIG_FINE)[..., 0]
        S = S + 1e-3 * S.mean()                      # floor: ~0.1% of the mean density
        return S / S.sum()

    ref = density(np.ones(len(aa), bool))
    classes = {"general": ~np.isin(aa, ["G", "P"]) & ~nxt_pro, "gly": aa == "G", "pro": aa == "P",
               "prepro": ~np.isin(aa, ["G", "P"]) & nxt_pro}
    names, maps, counts = [], [], []
    for a in T.AA1:
        classes[a] = (aa == a) & ~nxt_pro
    for name, m in classes.items():
        names.append(name)
        maps.append(density(m))
        counts.append(int(m.sum()))
    maps = np.stack(maps)
    out["rama_names"] = np.array(names)
    out["rama_counts"] = np.array(counts)
    out["rama_logp"] = np.log(maps).astype(np.float32)                 # ln P(phi,psi | class) per grid cell
    out["rama_ref_logp"] = np.log(ref).astype(np.float32)               # all residues
    out["rama_energy"] = (-(np.log(maps) - np.log(ref))).astype(np.float32)   # -ln P(.|class)/P(.)
    # ---- side-chain conformational entropy (nats): marginal per SS class, conditional on (phi,psi)
    ent = np.full((20, 4), np.nan)
    ent_rot = np.full((20, 4), np.nan)
    ent_cond = np.full((20, 4), np.nan)
    for ia, a in enumerate(T.AA1):
        if T.N_CHI[a] == 0:
            ent[ia] = ent_rot[ia] = ent_cond[ia] = 0.0
            continue
        d = df_sc[df_sc.aa == a]
        chis = d[[f"chi{k}" for k in range(1, 5)]].to_numpy(np.float64)
        ok = np.isfinite(chis[:, :T.N_CHI[a]]).all(1)
        d, chis = d[ok], chis[ok]
        bins = chi_bins(a, chis)
        rid = rot_index(a, bins)
        nonrot = T.NONROTAMERIC_CHI.get(a)
        rid_rot = rot_index(a, np.where(np.arange(bins.shape[1]) == (nonrot or 0) - 1, 0, bins))
        gi = np.clip(((d.phi.to_numpy() + 180) // 10).astype(int), 0, 35)
        gj = np.clip(((d.psi.to_numpy() + 180) // 10).astype(int), 0, 35)
        P = np.exp(lib[f"{a}_logp"].astype(np.float64))[gi, gj]          # [n, R]
        P = P / P.sum(1, keepdims=True)
        h_cond = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
        for c, sel in enumerate([np.ones(len(d), bool)] + [(d.ss3 == s).to_numpy() for s in SS3]):
            if sel.sum() < 20:
                continue
            for arr, ids in ((ent, rid), (ent_rot, rid_rot)):
                q = np.bincount(ids[sel]) / sel.sum()
                q = q[q > 0]
                arr[ia, c] = -(q * np.log(q)).sum()
            ent_cond[ia, c] = h_cond[sel].mean()
    out["entropy_classes"] = np.array(["all", "H", "E", "L"])
    out["sc_entropy"] = ent               # marginal, library rotamer bins (incl. fine non-rotameric bins)
    out["sc_entropy_rotameric"] = ent_rot  # marginal, rotameric chis only (terminal non-rotameric chi ignored)
    out["sc_entropy_bbdep"] = ent_cond     # mean over residues of entropy of P(rot | phi, psi)
    # ---- environment-dependent amino-acid potentials
    aidx = np.array([T.AA_INDEX[x] for x in aa])
    ss = np.array([SS3.index(s) for s in df_bb.ss3])
    cn = df_bb.cn.to_numpy()
    cn_edges = np.quantile(cn, np.linspace(0, 1, CN_NBINS + 1))
    cn_edges[0], cn_edges[-1] = -np.inf, np.inf
    cnb = np.clip(np.searchsorted(cn_edges, cn, side="right") - 1, 0, CN_NBINS - 1)
    rsa = df_bb.rsa.to_numpy()
    rsab = np.clip(np.searchsorted(RSA_EDGES, rsa, side="right") - 1, 0, len(RSA_EDGES) - 2)
    bas = bb_basin(phi, psi)
    p_aa = np.bincount(aidx, minlength=20) / len(aidx)
    out["p_aa"] = p_aa
    out["cn_edges"], out["rsa_edges"] = cn_edges, RSA_EDGES
    for name, keys, shape in (("cn_ss", (cnb, ss), (CN_NBINS, 3)), ("cn_ss_bb", (cnb, ss, bas), (CN_NBINS, 3, 3)),
                              ("rsa_ss", (rsab, ss), (len(RSA_EDGES) - 1, 3))):
        flat = np.ravel_multi_index(keys, shape)
        N = np.bincount(flat * 20 + aidx, minlength=int(np.prod(shape)) * 20).reshape(*shape, 20).astype(np.float64)
        P = (N + ENV_ALPHA * p_aa) / (N.sum(-1, keepdims=True) + ENV_ALPHA)
        out[f"env_{name}_counts"] = N.astype(np.int64)
        out[f"env_{name}_energy"] = (-np.log(P / p_aa)).astype(np.float32)
    pos = phi > 0
    n_pos = np.bincount(aidx[pos], minlength=20)
    out["p_aa_given_posphi"] = (n_pos + 1) / (pos.sum() + 20)
    out["p_posphi_given_aa"] = (n_pos + 0.5) / (np.bincount(aidx, minlength=20) + 1)
    out["posphi_energy"] = (-np.log(out["p_aa_given_posphi"] / p_aa)).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- query API
def _interp_weights(phi, psi):
    """Bilinear weights over the 4 surrounding grid centres (periodic): (i0, i1, j0, j1, wi, wj)."""
    fi = (np.asarray(phi, np.float64) + 175.0) / 10.0
    fj = (np.asarray(psi, np.float64) + 175.0) / 10.0
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    return i0 % 36, (i0 + 1) % 36, j0 % 36, (j0 + 1) % 36, fi - i0, fj - j0


def _bilinear(table, phi, psi):
    i0, i1, j0, j1, wi, wj = _interp_weights(phi, psi)
    sh = (-1,) + (1,) * (table.ndim - 2)
    wi, wj = wi.reshape(sh), wj.reshape(sh)
    return ((1 - wi) * (1 - wj) * table[i0, j0] + wi * (1 - wj) * table[i1, j0]
            + (1 - wi) * wj * table[i0, j1] + wi * wj * table[i1, j1])


def _nearest(phi, psi):
    i = np.clip(((np.asarray(phi) + 180) // 10).astype(int), 0, 35)
    j = np.clip(((np.asarray(psi) + 180) // 10).astype(int), 0, 35)
    return i, j


class RotamerLibrary:
    """Backbone-dependent rotamer library. `query` returns candidate rotamers per backbone (phi, psi)."""

    def __init__(self, path: str | Path = DATA / "rotlib_bbdep.npz"):
        z = np.load(path)
        self.lib = {k: z[k] for k in z.files}

    def n_rot(self, aa: str) -> int:
        return 1 if T.N_CHI[aa] == 0 else len(self.lib[f"{aa}_bins"])

    def tables(self, aa: str, phi, psi):
        """logp [B,R], mean [B,R,nchi], sd [B,R,nchi]; NaN phi/psi -> backbone-independent values."""
        L = self.lib
        phi, psi = np.atleast_1d(np.asarray(phi, np.float64)), np.atleast_1d(np.asarray(psi, np.float64))
        bad = ~(np.isfinite(phi) & np.isfinite(psi))
        ph, ps = np.where(bad, 0.0, phi), np.where(bad, 0.0, psi)
        p = _bilinear(np.exp(L[f"{aa}_logp"].astype(np.float64)), ph, ps)
        i, j = _nearest(ph, ps)
        mean = L[f"{aa}_mean"][i, j].astype(np.float64)
        sd = L[f"{aa}_sd"][i, j].astype(np.float64)
        if bad.any():
            p[bad] = L[f"{aa}_p_global"]
            mean[bad] = L[f"{aa}_mean_global"]
            sd[bad] = L[f"{aa}_sd_global"]
        return np.log(np.maximum(p, 1e-12)), mean, sd

    def query(self, aa: str, phi, psi, max_r: int | None = None, cum: float | None = None,
              expand: bool = False) -> dict[str, np.ndarray]:
        """Top rotamers for residue type `aa` at backbones (phi[B], psi[B]).

        Rotamers are sorted by P(rot | phi, psi); kept while the cumulative probability *before* a rotamer is
        < cum (if given) and up to max_r. expand=True adds chi1 (and chi2) +-1 sd children (x3 or x9) with
        log-weights from a discretised Gaussian. Returns chis [B,R',4] (deg), logp [B,R'], mask [B,R'],
        rot [B,R'] (library rotamer index), sd [B,R',4]."""
        B = len(np.atleast_1d(phi))
        if T.N_CHI[aa] == 0:
            z = np.zeros((B, 1, 4))
            return {"chis": z, "logp": np.zeros((B, 1)), "mask": np.ones((B, 1), bool),
                    "rot": np.zeros((B, 1), np.int64), "sd": z.copy()}
        logp, mean, sd = self.tables(aa, phi, psi)
        nchi = mean.shape[-1]
        order = np.argsort(-logp, axis=1, kind="stable")
        R = order.shape[1] if max_r is None else min(max_r, order.shape[1])
        order = order[:, :R]
        lp = np.take_along_axis(logp, order, 1)
        mask = np.ones_like(lp, bool)
        if cum is not None:
            pr = np.exp(lp)
            before = np.cumsum(pr, 1) - pr
            mask = before < cum * np.exp(logp).sum(1, keepdims=True)
        chis = np.zeros((B, R, 4))
        sds = np.zeros((B, R, 4))
        chis[..., :nchi] = np.take_along_axis(mean, order[..., None], 1)
        sds[..., :nchi] = np.take_along_axis(sd, order[..., None], 1)
        rot = order
        if expand:
            nexp = min(nchi, 2)
            steps = np.array(np.meshgrid(*[[0, -1, 1]] * nexp, indexing="ij")).reshape(nexp, -1).T   # [E, nexp]
            lw1 = np.log(np.array([1.0, np.exp(-0.5), np.exp(-0.5)]) / (1 + 2 * np.exp(-0.5)))
            lw = np.where(steps == 0, lw1[0], lw1[1]).sum(1)
            E = len(steps)
            chis = np.repeat(chis[:, :, None], E, 2)
            chis[..., :nexp] += steps[None, None] * sds[:, :, None, :nexp]
            lp = (lp[:, :, None] + lw[None, None]).reshape(B, R * E)
            mask = np.repeat(mask, E, 1)
            rot = np.repeat(rot, E, 1)
            sds = np.repeat(sds, E, 1)
            chis = chis.reshape(B, R * E, 4)
        return {"chis": chis.astype(np.float32), "logp": np.where(mask, lp, -np.inf).astype(np.float32),
                "mask": mask, "rot": rot, "sd": sds.astype(np.float32)}

    def assign(self, aa: str, chis: np.ndarray) -> np.ndarray:
        """Library rotamer index of observed chis [N,>=nchi] (-1 if the bin combination was pruned)."""
        if T.N_CHI[aa] == 0:
            return np.zeros(len(chis), np.int64)
        bins = chi_bins(aa, np.asarray(chis, np.float64))
        spec = [s[2] for s in chi_spec(aa)]
        lib_ids = rot_index(aa, self.lib[f"{aa}_bins"].astype(np.int64))
        lut = -np.ones(int(np.prod(spec)), np.int64)
        lut[lib_ids] = np.arange(len(lib_ids))
        return lut[rot_index(aa, bins)]


class PDBStats:
    """Ramachandran potentials, side-chain entropies and environment potentials (physddg/data/pdbstats.npz)."""

    def __init__(self, path: str | Path = DATA / "pdbstats.npz"):
        z = np.load(path)
        self.d = {k: z[k] for k in z.files}
        self.rama_names = list(self.d["rama_names"])

    def rama_energy(self, cls, phi, psi, absolute: bool = False) -> np.ndarray:
        """E_rama = -ln P(phi,psi|cls)/P(phi,psi) (or -ln P(phi,psi|cls) per 10x10 deg cell if absolute), bilinear.
        cls: a 1-letter aa or one of general/gly/pro/prepro; scalar or array [B] of class names."""
        tab = self.d["rama_logp"] if absolute else -self.d["rama_energy"]
        cls = np.broadcast_to(np.asarray(cls), np.shape(phi))
        ci = np.array([self.rama_names.index(c) for c in np.atleast_1d(cls)])
        phi, psi = np.atleast_1d(phi), np.atleast_1d(psi)
        e = np.full(len(phi), np.nan)
        ok = np.isfinite(phi) & np.isfinite(psi)
        if ok.any():
            i0, i1, j0, j1, wi, wj = _interp_weights(phi[ok], psi[ok])
            c = ci[ok]
            v = ((1 - wi) * (1 - wj) * tab[c, i0, j0] + wi * (1 - wj) * tab[c, i1, j0]
                 + (1 - wi) * wj * tab[c, i0, j1] + wi * wj * tab[c, i1, j1])
            e[ok] = -v
        return e

    def rama_energy_all_aa(self, phi, psi, next_is_pro=None) -> np.ndarray:
        """[B,20] E_rama for every residue type at each backbone; next_is_pro -> pre-Pro map for non-Gly/Pro."""
        phi, psi = np.atleast_1d(phi), np.atleast_1d(psi)
        out = np.stack([self.rama_energy(a, phi, psi) for a in T.AA1], 1)
        if next_is_pro is not None and np.any(next_is_pro):
            pp = self.rama_energy("prepro", phi, psi)
            cols = [i for i, a in enumerate(T.AA1) if a not in "GP"]
            m = np.asarray(next_is_pro, bool)
            out[np.ix_(m, cols)] = pp[m, None]
        return out

    def env_energy(self, ss3, cn=None, rsa=None, phi=None, psi=None, variant: str = "cn_ss") -> np.ndarray:
        """[B,20] E_env(aa | env) = -ln P(aa|env)/P(aa). variant: cn_ss (needs cn), cn_ss_bb (cn, phi, psi),
        rsa_ss (rsa). ss3 in 'HEL' (DSSP 3-state); cn = smooth CB contact number (structure.smooth_cn)."""
        ss = np.array([SS3.index(s) for s in np.atleast_1d(ss3)])
        E = self.d[f"env_{variant}_energy"]
        if variant.startswith("cn"):
            b = np.clip(np.searchsorted(self.d["cn_edges"], np.atleast_1d(cn), side="right") - 1, 0, CN_NBINS - 1)
        else:
            b = np.clip(np.searchsorted(self.d["rsa_edges"], np.atleast_1d(rsa), side="right") - 1, 0,
                        len(RSA_EDGES) - 2)
        if variant == "cn_ss_bb":
            return E[b, ss, bb_basin(np.nan_to_num(phi, nan=-60.0), np.nan_to_num(psi, nan=140.0))]
        return E[b, ss]

    def sc_entropy(self, kind: str = "sc_entropy_rotameric", ss: str = "all") -> np.ndarray:
        """[20] side-chain conformational entropy (nats) in AA1 order for SS class all/H/E/L."""
        return self.d[kind][:, ["all", "H", "E", "L"].index(ss)]
