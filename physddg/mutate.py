"""Mutation / packing engine: per-site physics features for all 20 amino acids (WT included, identical path).

For a protein and a set of sites, every site is evaluated for all 20 residue types with library rotamers at the
site's (phi, psi) (no native "include current" at the site, D6). WT and mutant go through the same code, so
ddE(WT->WT) = 0 exactly, and ddE(wt->mut) = feat[site, mut] - feat[site, wt].

Stage 1 (fixed environment, `stage1`): each rotamer r of residue type a at site i gets
    E_r = intra(r) + E(r, protein without residue i)            [per term, energy.py]
    U_r = sum_k E_r,k (packing terms, unit weights)  - kT ln p(r | phi, psi)
Site free energy F(a) = -kT ln sum_r exp(-U_r / kT); Boltzmann-averaged terms <E_k>; site entropy
S = -sum w ln w; min-energy versions for ablation.

Stage 2 (neighbour flexibility, `stage2`): flexible set N(i) = side chains (residues with chis, not Pro) having an atom
within `nb_radius` of any candidate side-chain atom of any of the 20 residue types at i (so the set is identical for WT
and every mutant). Neighbours carry their native side chain (exact coordinates) + library rotamers (identical sets in
all states). Batched mean-field over (site rotamers) x (neighbour candidates) for all (site, aa) at once; outputs the
mean-field free energy of the local region (site + neighbours + their interactions with the fixed rest) and
mean-field-averaged per-term energies. WT-WT neighbour tables (candidate vs native, candidate vs candidate) are
computed once per protein and shared by all sites.

CB: native CB kept for all targets when WT is not Gly (Pro<->X: ideal CB of the target, Pro CB geometry differs);
WT Gly -> ideal CB. Repulsion: rep_scale 0.9 (D14); per-residue cap = cap0 + cap1 * (1 - burial) (D9).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import energy as E
from . import topology as T
from .builder import build_sidechains, ideal_cb
from .params import EnergyParams
from .rotlib import PDBStats, RotamerLibrary

AA1 = T.AA1
G_IDX, P_IDX = T.AA_INDEX["G"], T.AA_INDEX["P"]


@dataclass
class MutateConfig:
    kT: float = 0.593                    # kcal/mol (298 K)
    cum: float = 0.97                    # site rotamers: cumulative library probability kept ...
    max_rot: int = 30                    # ... capped at max_rot per residue type
    expand: bool = False                 # +-1 sd chi1/chi2 expansion of site rotamers (x3 / x9)
    rep_scale: float = 0.9               # D14
    cap0: float = 3.0                    # repulsion cap per atom pair (kcal/mol) ...
    cap1: float = 0.0                    # ... + cap1 * (1 - burial)  (D9 site-dependent cap)
    pack_terms: tuple = ("lj_atr", "lj_rep", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb",
                         "elec_dh", "disulf")
    # stage 2
    nb_radius: float = 6.0               # neighbour side chain atom within this of any site candidate atom
    nb_max: int = 16
    nb_cum: float = 0.9
    nb_max_rot: int = 12
    s2_max_rot: int = 20                 # site rotamers per residue type in stage 2 (top by library probability)
    mf_iters: int = 30
    mf_damp: float = 0.5
    site_chunk: int = 48
    pair_chunk: int = 120_000            # block pairs per pair_energy call (~8 KB each)
    params: EnergyParams = field(default_factory=lambda: EnergyParams(rep_scale=0.9))

    def __post_init__(self):
        self.params = replace(self.params, rep_scale=self.rep_scale)


_LIB: RotamerLibrary | None = None
_STATS: PDBStats | None = None


def library() -> RotamerLibrary:
    global _LIB
    if _LIB is None:
        _LIB = RotamerLibrary()
    return _LIB


def pdbstats() -> PDBStats:
    global _STATS
    if _STATS is None:
        _STATS = PDBStats()
    return _STATS


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


class SiteScan:
    """All-20-aa site evaluation for one protein (structure.Protein, annotated)."""

    def __init__(self, prot, cfg: MutateConfig | None = None, device="cuda"):
        self.prot, self.cfg, self.dev = prot, cfg or MutateConfig(), torch.device(device)
        p = self.cfg.params
        self.terms = E.term_names(p)
        self.pack_idx = torch.tensor([self.terms.index(t) for t in self.cfg.pack_terms], device=self.dev)
        b = E.blocks_from_protein(prot, self.dev, p)
        b.cap = self.cfg.cap0 + self.cfg.cap1 * (1.0 - b.burial)
        self.blocks = b
        self.L = len(prot)
        with torch.no_grad():
            self.env = E.make_env(b, None, p)
        if "phi" not in prot.ann:
            from .structure import annotate
            annotate(prot, do_sasa=False)
        self.phi, self.psi = prot.ann["phi"], prot.ann["psi"]
        self.bb = torch.tensor(np.nan_to_num(prot.xyz14[:, :4]), device=self.dev, dtype=torch.float32)
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------------------- candidate construction
    def _cb(self, sites: np.ndarray, a: int) -> torch.Tensor:
        """CB to use for residue type a at sites: native unless WT is Gly or exactly one of (WT, a) is Pro."""
        bb = self.bb[sites]
        ideal = ideal_cb(bb[:, 0], bb[:, 1], bb[:, 2], a if a != G_IDX else None)
        wt = torch.as_tensor(self.prot.aa[sites], device=self.dev)
        nat_ok = torch.as_tensor(self.prot.mask14[sites, 4], device=self.dev) & (wt != G_IDX) \
            & ((wt == P_IDX) == (a == P_IDX))
        return torch.where(nat_ok[:, None], self.bb.new_tensor(np.nan_to_num(self.prot.xyz14[sites, 4])), ideal)

    def site_candidates(self, sites: np.ndarray, max_rot: int | None = None, cum: float | None = None,
                        expand: bool | None = None) -> dict:
        """Library rotamers for all 20 aa at `sites`: flat lists (site_local, aa, logp normalised per (site, aa)),
        coordinates xyz14 [N,14,3]."""
        c = self.cfg
        lib = library()
        out = {k: [] for k in ("s", "a", "logp", "xyz", "chis")}
        for a in range(20):
            q = lib.query(AA1[a], self.phi[sites], self.psi[sites], max_r=max_rot or c.max_rot,
                          cum=c.cum if cum is None else cum, expand=c.expand if expand is None else expand)
            lp = np.where(q["mask"], q["logp"], -np.inf).astype(np.float64)
            lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)              # renormalise over kept rotamers
            chis = torch.as_tensor(q["chis"], device=self.dev, dtype=torch.float32)
            xyz, _ = build_sidechains(self.bb[sites], a, chis, cb=self._cb(sites, a))
            si, ri = np.nonzero(q["mask"])
            out["s"].append(si); out["a"].append(np.full(len(si), a))
            out["logp"].append(lp[si, ri]); out["chis"].append(q["chis"][si, ri])
            out["xyz"].append(xyz[torch.as_tensor(si, device=self.dev), torch.as_tensor(ri, device=self.dev)])
        res = {k: np.concatenate(v) for k, v in out.items() if k != "xyz"}
        res["xyz"] = torch.cat(out["xyz"])
        order = np.lexsort((np.arange(len(res["s"])), res["a"], res["s"]))      # group by site, then aa
        for k in ("s", "a", "logp", "chis"):
            res[k] = res[k][order]
        res["xyz"] = res["xyz"][torch.as_tensor(order, device=self.dev)]
        return res

    def _blocks(self, res_idx: np.ndarray, aa: np.ndarray, xyz14: torch.Tensor) -> E.Blocks:
        """Blocks for candidate side chains placed at residues res_idx (backbone/OXT/pos/... from the protein)."""
        b = self.blocks
        ri = torch.as_tensor(res_idx, device=self.dev)
        aa_t = torch.as_tensor(aa, device=self.dev, dtype=torch.long)
        tm = E.tables(self.dev)["valid"][aa_t]                                     # [N, 15] template slots
        mask = tm.clone()
        mask[:, E.OXT] = b.mask[ri, E.OXT]
        xyz = torch.cat([torch.nan_to_num(xyz14), b.xyz[ri, E.OXT][:, None]], 1)
        xyz[:, :4] = b.xyz[ri, :4]
        mask[:, :4] = b.mask[ri, :4]
        return E.Blocks(xyz=xyz, aa=aa_t, mask=mask, pos=b.pos[ri], prev_c=b.prev_c[ri], nterm=b.nterm[ri],
                        burial=b.burial[ri], cap=b.cap[ri])

    def _energies(self, mov: E.Blocks, exclude: np.ndarray, include_ref: bool = False) -> torch.Tensor:
        """intra + energy vs the environment without residue(s) exclude[n] (or [n, k]). Candidates sharing an
        exclusion set must be contiguous; one env_energy call (1-D env mask, shared neighbour list) per set.
        (A fully batched multi-site variant was 2x slower: GPU-bound, larger padded neighbour lists.)"""
        p = self.cfg.params
        f = E.features(mov, p)
        out = E.intra_energy(mov, p, include_ref, f)
        ex = np.asarray(exclude).reshape(len(mov.aa), -1)
        new = np.r_[True, (ex[1:] != ex[:-1]).any(1), True]
        bounds = np.flatnonzero(new)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            em = torch.ones(self.L, dtype=torch.bool, device=self.dev)
            em[torch.as_tensor(ex[lo], device=self.dev)] = False
            sl = slice(int(lo), int(hi))
            out[sl] = out[sl] + E.env_energy(mov[sl], self.env, p, env_mask=em, f={k: v[sl] for k, v in f.items()})
        return out

    def _pairs(self, a: E.Blocks, b: E.Blocks, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
        """pair energies between a[ia[k]] and b[ib[k]] (flat lists), chunked -> [K, T]."""
        p = self.cfg.params
        out = torch.zeros(len(ia), len(self.terms), device=self.dev)
        for s in range(0, len(ia), self.cfg.pair_chunk):
            sl = slice(s, s + self.cfg.pair_chunk)
            out[sl] = E.pair_energy(a[ia[sl]], b[ib[sl]], p)
        return out

    # ------------------------------------------------------------------------- stage 1
    @torch.no_grad()
    def stage1(self, sites, return_rotamers: bool = False) -> dict:
        sites = np.asarray(sites, np.int64)
        c, kT = self.cfg, self.cfg.kT
        nT = len(self.terms)
        res = {k: [] for k in ("F", "E_avg", "E_min", "U_min", "S_site", "prior_avg", "n_rot", "chi_best", "ref")}
        rots = []
        t0 = time.time()
        for s0 in range(0, len(sites), c.site_chunk):
            ss = sites[s0:s0 + c.site_chunk]
            cand = self.site_candidates(ss)
            mov = self._blocks(ss[cand["s"]], cand["a"], cand["xyz"])
            e = self._energies(mov, ss[cand["s"]])                                          # [N, T]
            red = self._reduce(e, cand, len(ss))
            red["ref"] = self._ref(mov, cand, len(ss))
            for k in res:
                res[k].append(red[k])
            if return_rotamers:
                rots.append({"site": ss[cand["s"]], "aa": cand["a"], "logp": cand["logp"], "chis": cand["chis"],
                             "E": e.cpu().numpy()})
        _sync(self.dev)
        self.timing["stage1"] = time.time() - t0
        out = {k: np.concatenate(v) for k, v in res.items()}
        out["sites"] = sites
        if return_rotamers:
            out["rotamers"] = {k: np.concatenate([r[k] for r in rots]) for k in rots[0]}
        return out

    def _reduce(self, e: torch.Tensor, cand: dict, nsite: int) -> dict:
        """Boltzmann reduction over rotamers per (site, aa)."""
        kT = self.cfg.kT
        g = torch.as_tensor(cand["s"] * 20 + cand["a"], device=self.dev)
        nG = nsite * 20
        prior = -kT * torch.as_tensor(cand["logp"], device=self.dev, dtype=torch.float32)
        U = e[:, self.pack_idx].sum(1) + prior                                                  # [N]
        # padded [G, Rmax] layout (rotamers of a group are contiguous: candidates sorted by site, aa)
        counts = torch.bincount(g, minlength=nG)
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(g), device=self.dev) - start[g]
        R = int(counts.max())
        Up = torch.full((nG, R), float("inf"), device=self.dev)
        Up[g, rank] = U
        logZ = torch.logsumexp(-Up / kT, 1)
        w = torch.exp(-Up / kT - logZ[:, None])                                                 # [G, R]
        wf = w[g, rank]
        Eavg = self._group_sum(e * wf[:, None], g, rank, nG, R)
        best = torch.argmin(Up, 1)
        idx_best = start + best
        ent = -(torch.where(w > 0, w * torch.log(w.clamp(min=1e-30)), torch.zeros_like(w))).sum(1)
        return {"F": (-kT * logZ).view(nsite, 20).cpu().numpy(),
                "E_avg": Eavg.view(nsite, 20, -1).cpu().numpy(),
                "E_min": e[idx_best].view(nsite, 20, -1).cpu().numpy(),
                "U_min": Up.min(1).values.view(nsite, 20).cpu().numpy(),
                "S_site": ent.view(nsite, 20).cpu().numpy(),
                "prior_avg": self._group_sum((prior * wf)[:, None], g, rank, nG, R)[:, 0].view(nsite, 20).cpu().numpy(),
                "n_rot": counts.view(nsite, 20).cpu().numpy(),
                "chi_best": cand["chis"][idx_best.cpu().numpy()].reshape(nsite, 20, 4)}

    def _ref(self, mov: E.Blocks, cand: dict, nsite: int) -> np.ndarray:
        """EEF1 reference solvation (sum of dG_ref, apolar / polar atoms) per (site, aa) [S, 20, 2]; excluded from
        all energies above (unfolded-state reference, D7) and returned separately."""
        f = E.features(mov, self.cfg.params)
        ref = f["dg_ref"] * f["m"].float()
        r2 = torch.stack([(ref * (1 - f["polar"])).sum(-1), (ref * f["polar"]).sum(-1)], -1)     # [N, 2]
        out = torch.zeros(nsite * 20, 2, device=self.dev)
        g = torch.as_tensor(cand["s"] * 20 + cand["a"], device=self.dev)
        first = np.r_[True, (cand["s"][1:] != cand["s"][:-1]) | (cand["a"][1:] != cand["a"][:-1])]
        fi = torch.as_tensor(np.flatnonzero(first), device=self.dev)
        out[g[fi]] = r2[fi]
        return out.view(nsite, 20, 2).cpu().numpy()

    def _group_sum(self, x: torch.Tensor, g, rank, nG: int, R: int) -> torch.Tensor:
        """Deterministic per-group sum of x [N, k] (padded layout, plain sum over the rotamer axis)."""
        pad = torch.zeros(nG, R, x.shape[1], device=self.dev, dtype=x.dtype)
        pad[g, rank] = x
        return pad.sum(1)


    # ------------------------------------------------------------------------- stage 2
    def neighbour_sets(self, sites: np.ndarray, cand: dict) -> list[np.ndarray]:
        """Flexible neighbours per site: residues with chis (not Pro, complete side chain) having a movable side-chain
        atom (beyond CB) within nb_radius of any candidate atom (CB onwards) of any residue type at the site.
        Sorted by that distance, at most nb_max. Identical for all 20 aa at a site (union of WT/mutant neighbourhoods)."""
        c = self.cfg
        prot = self.prot
        nchi = np.array([T.N_CHI[a] for a in prot.seq])
        ok = (nchi > 0) & (prot.aa != P_IDX) & ~prot.missing_sc
        mov = np.zeros((self.L, T.MAX_ATOMS), bool)
        mov[:, 5:] = prot.mask14[:, 5:]
        mov &= ok[:, None]
        rj, sj = np.nonzero(mov)
        Q = torch.as_tensor(np.nan_to_num(prot.xyz14[rj, sj]), device=self.dev)
        rj_t = torch.as_tensor(rj, device=self.dev)
        out = []
        xyz = cand["xyz"]
        for k, s in enumerate(sites):
            sel = torch.as_tensor(np.flatnonzero(cand["s"] == k), device=self.dev)
            P = xyz[sel][:, 4:].reshape(-1, 3)
            P = P[torch.isfinite(P).all(1)]
            if P.shape[0] == 0 or Q.shape[0] == 0:
                out.append(np.zeros(0, np.int64))
                continue
            near = (Q - self.bb[s, 1]).norm(dim=-1) < c.nb_radius + 25.0
            dmin = torch.cdist(Q[near], P).min(1).values
            dres = torch.full((self.L,), float("inf"), device=self.dev)
            dres = dres.scatter_reduce(0, rj_t[near], dmin, "amin", include_self=True)
            dres[s] = float("inf")
            d_np = dres.cpu().numpy()
            idx = np.flatnonzero(d_np < c.nb_radius)
            idx = idx[np.lexsort((idx, d_np[idx]))][:c.nb_max]
            out.append(np.sort(idx))
        return out

    def neighbour_candidates(self, J: np.ndarray) -> dict:
        """Candidates for residues J: native side chain (index 0, exact coordinates) + library rotamers (native CB).
        Returns padded tensors: blocks [nJ, C], valid [nJ, C], prior (-kT ln p, renormalised) [nJ, C], E1 [nJ, C, T]
        (intra + vs protein without the residue)."""
        c, kT = self.cfg, self.cfg.kT
        lib = library()
        prot = self.prot
        C = 1 + c.nb_max_rot
        nJ = len(J)
        xyz = torch.zeros(nJ, C, 14, 3, device=self.dev)
        valid = np.zeros((nJ, C), bool)
        logp = np.full((nJ, C), -np.inf)
        xyz[:, 0] = torch.as_tensor(np.nan_to_num(prot.xyz14[J]), device=self.dev)
        valid[:, 0] = True
        for a in np.unique(prot.aa[J]):
            rows = np.flatnonzero(prot.aa[J] == a)
            jj = J[rows]
            q = lib.query(AA1[a], self.phi[jj], self.psi[jj], max_r=c.nb_max_rot, cum=c.nb_cum)
            R = q["chis"].shape[1]
            cb = self.bb.new_tensor(np.nan_to_num(prot.xyz14[jj, 4]))
            built, _ = build_sidechains(self.bb[jj], int(a), torch.as_tensor(q["chis"], device=self.dev,
                                                                              dtype=torch.float32), cb=cb)
            xyz[torch.as_tensor(rows, device=self.dev)[:, None], torch.arange(1, R + 1, device=self.dev)] = \
                torch.nan_to_num(built)
            valid[rows, 1:R + 1] = q["mask"]
            logp[rows, 1:R + 1] = np.where(q["mask"], q["logp"], -np.inf)
            # native: probability of its library rotamer bin at its own backbone (or the smallest kept one)
            lpt, _, _ = lib.tables(AA1[a], self.phi[jj], self.psi[jj])
            rid = lib.assign(AA1[a], np.nan_to_num(prot.ann["chi"][jj]))
            lp_nat = np.where(rid >= 0, lpt[np.arange(len(jj)), np.maximum(rid, 0)], np.nan)
            lp_min = np.where(q["mask"], q["logp"], np.inf).min(1)
            logp[rows, 0] = np.where(np.isfinite(lp_nat), lp_nat, lp_min)
        logp = logp - np.logaddexp.reduce(logp, axis=1, keepdims=True)
        jr, cr = np.nonzero(valid)
        blk = self._blocks(J[jr], prot.aa[J[jr]], xyz[torch.as_tensor(jr, device=self.dev),
                                                     torch.as_tensor(cr, device=self.dev)])
        e = self._energies(blk, J[jr])
        E1 = torch.zeros(nJ, C, len(self.terms), device=self.dev)
        E1[torch.as_tensor(jr, device=self.dev), torch.as_tensor(cr, device=self.dev)] = e
        flat = -np.ones((nJ, C), np.int64)
        flat[jr, cr] = np.arange(len(jr))
        return {"J": J, "blocks": blk, "flat": flat, "valid": valid, "E1": E1,
                "prior": torch.as_tensor(np.where(valid, -kT * logp, 0.0), device=self.dev, dtype=torch.float32)}

    @torch.no_grad()
    def stage2(self, sites) -> dict:
        sites = np.asarray(sites, np.int64)
        c, kT = self.cfg, self.cfg.kT
        nT = len(self.terms)
        t0 = time.time()
        # pass 1: site candidates and flexible neighbour sets
        chunks = []
        for s0 in range(0, len(sites), c.site_chunk):
            ss = sites[s0:s0 + c.site_chunk]
            cand = self.site_candidates(ss, max_rot=c.s2_max_rot)
            chunks.append((ss, cand, self.neighbour_sets(ss, cand)))
        nbs = [n for _, _, nl in chunks for n in nl]
        J = np.unique(np.concatenate(nbs + [np.zeros(0, np.int64)]))
        jpos = {j: i for i, j in enumerate(J)}
        NC = self.neighbour_candidates(J) if len(J) else None
        Cmax = 1 + c.nb_max_rot
        # shared WT-WT tables: neighbour candidate vs native residue k, candidate vs candidate
        pn_keys, pp_keys = set(), set()
        for s, nl in zip(sites, nbs):
            V = [s] + nl.tolist()
            for j in nl:
                pn_keys.update((int(j), int(k)) for k in V if k != j)
            pp_keys.update((int(a), int(b)) for i, a in enumerate(nl) for b in nl[i + 1:])
        pn_keys, pp_keys = sorted(pn_keys), sorted(pp_keys)
        pn_id = {k: i for i, k in enumerate(pn_keys)}
        pp_id = {k: i for i, k in enumerate(pp_keys)}
        PN = torch.zeros(len(pn_keys), Cmax, nT, device=self.dev)
        PP = torch.zeros(len(pp_keys), Cmax, Cmax, nT, device=self.dev)
        if pn_keys:
            ia, ib, dst = [], [], []
            for i, (j, k) in enumerate(pn_keys):
                f = NC["flat"][jpos[j]]
                cc = np.flatnonzero(f >= 0)
                ia.append(f[cc]); ib.append(np.full(len(cc), k)); dst.append(np.stack([np.full(len(cc), i), cc], 1))
            ia, ib, dst = np.concatenate(ia), np.concatenate(ib), np.concatenate(dst)
            e = self._pairs(NC["blocks"], self.blocks, torch.as_tensor(ia, device=self.dev),
                            torch.as_tensor(ib, device=self.dev))
            PN[torch.as_tensor(dst[:, 0], device=self.dev), torch.as_tensor(dst[:, 1], device=self.dev)] = e
        if pp_keys:
            ia, ib, dst = [], [], []
            for i, (j, k) in enumerate(pp_keys):
                fj, fk = NC["flat"][jpos[j]], NC["flat"][jpos[k]]
                cj, ck = np.flatnonzero(fj >= 0), np.flatnonzero(fk >= 0)
                gj, gk = np.meshgrid(cj, ck, indexing="ij")
                ia.append(fj[gj.ravel()]); ib.append(fk[gk.ravel()])
                dst.append(np.stack([np.full(gj.size, i), gj.ravel(), gk.ravel()], 1))
            ia, ib, dst = np.concatenate(ia), np.concatenate(ib), np.concatenate(dst)
            e = self._pairs(NC["blocks"], NC["blocks"], torch.as_tensor(ia, device=self.dev),
                            torch.as_tensor(ib, device=self.dev))
            d_ = torch.as_tensor(dst, device=self.dev)
            PP[d_[:, 0], d_[:, 1], d_[:, 2]] = e
        self.timing["stage2_tables"] = time.time() - t0
        # pass 2: per site chunk, site one-body + site-neighbour pairs + mean field
        res = {k: [] for k in ("F", "E_avg", "S_local", "prior_avg", "n_rot", "n_nb", "chi_best", "ref",
                               "nb_native_frac")}
        for ss, cand, nl in chunks:
            red = self._stage2_chunk(ss, cand, nl, NC, jpos, PN, pn_id, PP, pp_id)
            for k in res:
                res[k].append(red[k])
        _sync(self.dev)
        self.timing["stage2"] = time.time() - t0
        out = {k: np.concatenate(v) for k, v in res.items()}
        out["sites"] = sites
        return out

    def _stage2_chunk(self, ss, cand, nl, NC, jpos, PN, pn_id, PP, pp_id) -> dict:
        c, kT = self.cfg, self.cfg.kT
        dev = self.dev
        nT = len(self.terms)
        S = len(ss)
        Cmax = 1 + c.nb_max_rot
        K = max(1, max(len(n) for n in nl))
        mov = self._blocks(ss[cand["s"]], cand["a"], cand["xyz"])
        e1 = self._energies(mov, ss[cand["s"]])                                      # [N, T] vs protein - site
        N = len(cand["s"])
        # site-candidate x neighbour-candidate pairs
        Esn = torch.zeros(N, K, Cmax, nT, device=dev)
        ia, ib, dst = [], [], []
        for k_, (s, n) in enumerate(zip(ss, nl)):
            rows = np.flatnonzero(cand["s"] == k_)
            for jl, j in enumerate(n):
                f = NC["flat"][jpos[j]]
                cc = np.flatnonzero(f >= 0)
                gr, gc = np.meshgrid(rows, cc, indexing="ij")
                ia.append(gr.ravel()); ib.append(f[gc.ravel()])
                dst.append(np.stack([gr.ravel(), np.full(gr.size, jl), gc.ravel()], 1))
        if ia:
            ia, ib, dst = np.concatenate(ia), np.concatenate(ib), np.concatenate(dst)
            e = self._pairs(mov, NC["blocks"], torch.as_tensor(ia, device=dev), torch.as_tensor(ib, device=dev))
            d_ = torch.as_tensor(dst, device=dev)
            Esn[d_[:, 0], d_[:, 1], d_[:, 2]] = e
        # site one-body vs the fixed rest: subtract interactions with the native flexible neighbours (cand 0)
        h0 = e1 - Esn[:, :, 0].sum(1)                                                 # [N, T]
        prior0 = -kT * torch.as_tensor(cand["logp"], device=dev, dtype=torch.float32)
        # neighbour one-body vs the fixed rest, per site
        hj = torch.zeros(S, K, Cmax, nT, device=dev)
        pj = torch.zeros(S, K, Cmax, device=dev)
        vj = torch.zeros(S, K, Cmax, dtype=torch.bool, device=dev)
        vj[:, :, 0] = True                                                             # padded nb: one null state
        Wnn = torch.zeros(S, K, Cmax, K, Cmax, nT, device=dev)
        for k_, (s, n) in enumerate(zip(ss, nl)):
            V = [s] + n.tolist()
            for jl, j in enumerate(n):
                ji = jpos[j]
                sub = sum(PN[pn_id[(int(j), int(k))]] for k in V if k != j)
                hj[k_, jl] = NC["E1"][ji] - sub
                pj[k_, jl] = NC["prior"][ji]
                vj[k_, jl] = torch.as_tensor(NC["valid"][ji], device=dev)
                for kl in range(jl + 1, len(n)):
                    w = PP[pp_id[(int(j), int(n[kl]))]]
                    Wnn[k_, jl, :, kl] = w
                    Wnn[k_, kl, :, jl] = w.transpose(0, 1)
        # padded (site, aa) layout
        g = torch.as_tensor(cand["s"] * 20 + cand["a"], device=dev)
        nG = S * 20
        counts = torch.bincount(g, minlength=nG)
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(N, device=dev) - start[g]
        R = int(counts.max())
        pk = self.pack_idx
        sb = torch.arange(nG, device=dev) // 20                                       # site of each batch row
        H0 = torch.zeros(nG, R, nT, device=dev); H0[g, rank] = h0
        P0 = torch.zeros(nG, R, device=dev); P0[g, rank] = prior0
        V0 = torch.zeros(nG, R, dtype=torch.bool, device=dev); V0[g, rank] = True
        ESN = torch.zeros(nG, R, K, Cmax, nT, device=dev); ESN[g, rank] = Esn
        U0 = torch.where(V0, H0[..., pk].sum(-1) + P0, torch.full_like(P0, float("inf")))
        Uj = torch.where(vj, hj[..., pk].sum(-1) + pj, torch.full_like(pj, float("inf")))[sb]   # [B, K, C]
        Wsn = ESN[..., pk].sum(-1)                                                     # [B, R, K, C]
        Wn = Wnn[..., pk].sum(-1)                                                      # [S, K, C, K, C]
        q0 = torch.softmax(-U0 / kT, -1)
        qj = torch.softmax(-Uj / kT, -1)
        dmp = c.mf_damp
        for _ in range(c.mf_iters):
            nn_f = torch.einsum("sald,skcld->sakc", qj.view(S, 20, K, Cmax), Wn).reshape(nG, K, Cmax)
            fj = Uj + torch.einsum("br,brkc->bkc", q0, Wsn) + nn_f
            qj = dmp * qj + (1 - dmp) * torch.softmax(-fj / kT, -1)
            f0 = U0 + torch.einsum("bkc,brkc->br", qj, Wsn)
            q0 = dmp * q0 + (1 - dmp) * torch.softmax(-f0 / kT, -1)
        xlogx = lambda q: torch.where(q > 0, q * torch.log(q.clamp(min=1e-30)), torch.zeros_like(q))
        ent = -(xlogx(q0).sum(-1) + xlogx(qj).sum((-1, -2)))
        q0z, qjz = torch.where(V0, q0, 0.0), torch.where(vj[sb], qj, 0.0)
        Hjb = hj[sb]
        E_avg = (torch.einsum("br,brt->bt", q0z, H0) + torch.einsum("bkc,bkct->bt", qjz, Hjb)
                 + torch.einsum("br,bkc,brkct->bt", q0z, qjz, ESN)
                 + 0.5 * (qjz.view(S, 20, K, Cmax, 1) * torch.einsum("sald,skcldt->sakct", qjz.view(S, 20, K, Cmax),
                                                                       Wnn)).sum((2, 3)).reshape(nG, nT))
        prior_avg = (q0z * P0).sum(-1) + (qjz * pj[sb]).sum((-1, -2))
        F = E_avg[:, pk].sum(-1) + prior_avg - kT * ent
        best = torch.argmax(torch.where(V0, q0, -1.0), -1)
        chi_best = cand["chis"][(start + best).cpu().numpy()].reshape(S, 20, 4)
        nb_nat = (qjz.argmax(-1) == 0).float()
        nn = torch.as_tensor([len(n) for n in nl], device=dev).float()[sb]
        kmask = torch.arange(K, device=dev)[None] < nn[:, None]
        nb_frac = torch.where(nn > 0, (nb_nat * kmask).sum(-1) / nn.clamp(min=1), torch.full_like(nn, float("nan")))
        v = lambda x: x.view(S, 20, *x.shape[1:]).cpu().numpy()
        return {"F": v(F), "E_avg": v(E_avg), "S_local": v(ent), "prior_avg": v(prior_avg),
                "n_rot": counts.view(S, 20).cpu().numpy(), "n_nb": np.array([len(n) for n in nl]),
                "chi_best": chi_best, "ref": self._ref(mov, cand, S), "nb_native_frac": v(nb_frac)}


# ============================================================================= statistical / context features
def stat_features(prot, sites, kT: float = 0.593) -> dict:
    """Per (site, aa) statistical terms in kcal/mol (PDB-derived, rotlib.PDBStats) and per-site context.
    rama: -kT ln P(phi,psi|aa)/P(phi,psi) (pre-Pro map if residue i+1 is Pro); rama_prev: residue i-1 under the
    pre-Pro map if aa == Pro, its own map otherwise (0 if i-1 is not bonded); env_*: -kT ln P(aa|env)/P(aa);
    sc_entropy: kT * rotameric side-chain entropy for the site's SS class; posphi: -kT ln P(aa|phi>0)/P(aa) if phi>0."""
    st = pdbstats()
    a = prot.ann
    sites = np.asarray(sites)
    S = len(sites)
    phi, psi = a["phi"][sites], a["psi"][sites]
    brk = a["chain_break"]
    nxt = np.array([(i + 1 < len(prot)) and not brk[i + 1] and prot.seq[i + 1] == "P" for i in sites])
    out = {"rama": kT * np.nan_to_num(st.rama_energy_all_aa(phi, psi, next_is_pro=nxt))}
    prev_ok = np.array([i > 0 and not brk[i] for i in sites])
    rp = np.zeros((S, 20))
    pi = np.maximum(sites - 1, 0)
    own = np.nan_to_num(np.array([st.rama_energy(prot.seq[j], [a["phi"][j]], [a["psi"][j]])[0] for j in pi]))
    pre = np.nan_to_num(np.array([st.rama_energy("prepro" if prot.seq[j] not in "GP" else prot.seq[j],
                                                 [a["phi"][j]], [a["psi"][j]])[0] for j in pi]))
    rp[:] = own[:, None]
    rp[:, P_IDX] = pre
    out["rama_prev"] = kT * np.where(prev_ok[:, None], rp, 0.0)
    ss3 = a["ss3"][sites]
    out["env_cn_ss"] = kT * st.env_energy(ss3, cn=a["cn"][sites])
    out["env_cn_ss_bb"] = kT * st.env_energy(ss3, cn=a["cn"][sites], phi=phi, psi=psi, variant="cn_ss_bb")
    if "rsa" in a:
        out["env_rsa_ss"] = kT * st.env_energy(ss3, rsa=a["rsa"][sites], variant="rsa_ss")
    ssi = {"H": 1, "E": 2, "L": 3}
    ent = st.d["sc_entropy_rotameric"]
    out["sc_entropy"] = kT * np.stack([ent[:, ssi[x]] for x in ss3])
    out["posphi"] = kT * np.where((np.nan_to_num(phi) > 0)[:, None], st.d["posphi_energy"][None], 0.0)
    ctx = {"phi": phi, "psi": psi, "ss3": ss3, "cn": a["cn"][sites], "ncb10": a["ncb10"][sites],
           "bfactor": prot.bfactor[sites], "hse_up": a["hse_up"][sites]}
    if "rsa" in a:
        ctx["rsa"] = a["rsa"][sites]
    return {"per_aa": out, "context": ctx}
