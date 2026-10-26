"""Differentiable Stage-1 site scan w.r.t. a small vector of physical parameters (engineer E3).

The frozen literature parameters of physddg/params.py (EEF1, CHARMM19 vdW, ...) and a few term hyper-parameters
are re-parameterised multiplicatively, theta = log(value / literature value), theta = 0 reproduces the shipped
engine. The Stage-1 protocol is unchanged (mutate.SiteScan: library rotamers at the site, fixed native
environment, Boltzmann average over rotamers); rotamer coordinates and candidate sets do not depend on theta,
so they are built once per site and only the energies + the Boltzmann reduction are re-evaluated.

Parameters (PARAMS, 32): per-type EEF1 solvation strength (scales dG_free and dG_ref of the type together),
a global EEF1 lambda scale, vdW radius and well-depth scales per type class (apolar C, aromatic C, polar N/O, S;
the radius also sets the EEF1 R_i, as in EEF1), multipliers of the H-bond / electrostatic terms inside the
packing energy U (they only shape the Boltzmann weights: the linear model already rescales the features), the
Debye length, the Boltzmann temperature kT of the rotamer ensemble, the weight of the rotamer prior in U, the
repulsive-radius scale (rep_scale, D14) and the repulsion cap (D9 cap0).

Hooks used: energy.features reads `EnergyParams.extra["type_table"]` when present (default: absent -> frozen table,
identical code path); EnergyParams.rep_scale / debye and Blocks.cap accept tensors. Nothing else in E1/E2 code
is changed. Features produced are exactly the Stage-1 columns of scripts/55 (s1_dE_<term>, s1_dTS,
s1_dprior_avg, s1_dref_apolar/polar, s1_dF) for the terms used by the B6gs1 model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint

from . import energy as E
from . import topology as T
from .mutate import MutateConfig, SiteScan, _float32
from .params import COLS, CHARMM19, TYPE_MAP, type_table

KT_LIB = 0.593            # mutate.MutateConfig.kT (prior energy -kT ln p and Boltzmann temperature)
TYPES = list(T.ATOM_TYPES)
C_APOLAR = ["CA", "C_bb", "CH1", "CH2", "CH3", "C_sp2"]
POLAR = ["N_bb", "N_pro", "O_bb", "N_amide", "N_his", "N_trp", "N_lys", "N_arg", "O_amide", "O_carboxyl", "O_hydroxyl"]
SULF = ["S_cys", "S_met"]

# name -> (kind, member atom types or None, unit / literature reference text)
PARAMS: dict[str, tuple] = {}
for _n, _m in [("N_bb", ["N_bb", "N_pro"]), ("O_bb", ["O_bb"]), ("CH1E", ["CA", "CH1"]), ("CH2", ["CH2"]),
               ("CH3", ["CH3"]), ("C_aro", ["C_aro"]), ("N_amide", ["N_amide"]), ("N_ring", ["N_his", "N_trp"]),
               ("N_lys", ["N_lys"]), ("N_arg", ["N_arg"]), ("O_amide", ["O_amide"]), ("O_carboxyl", ["O_carboxyl"]),
               ("O_hydroxyl", ["O_hydroxyl"]), ("S", SULF)]:
    PARAMS[f"solv_{_n}"] = ("solv", _m)
PARAMS["eef1_lambda"] = ("lam", TYPES)
for _n, _m in [("Capolar", C_APOLAR), ("Caro", ["C_aro"]), ("NO", POLAR), ("S", SULF)]:
    PARAMS[f"rad_{_n}"] = ("rad", _m)
for _n, _m in [("Capolar", C_APOLAR), ("Caro", ["C_aro"]), ("NO", POLAR), ("S", SULF)]:
    PARAMS[f"eps_{_n}"] = ("eps", _m)
for _n in ("hb_bb_bb", "hb_sc_bb", "hb_sc_sc", "elec_dh"):
    PARAMS[f"U_{_n}"] = ("U", None)
PARAMS["debye"] = ("debye", None)
PARAMS["kT"] = ("kT", None)
PARAMS["U_prior"] = ("prior", None)
PARAMS["rep_scale"] = ("rep_scale", None)
PARAMS["rep_cap"] = ("cap", None)
NAMES = list(PARAMS)
NP = len(NAMES)
# columns produced per (site, aa) group: 8 Boltzmann-averaged terms, site entropy S, <prior>, ref (2), F
TERMS = ("lj_atr", "lj_rep", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb", "elec_dh")
GCOLS = [f"E_{t}" for t in TERMS] + ["S_site", "prior_avg", "ref_apolar", "ref_polar", "F"]


def literature_values(cfg: MutateConfig | None = None) -> dict:
    """Reference (theta = 0) value of every parameter, in its natural unit (kcal/mol, Å, dimensionless)."""
    cfg = cfg or MutateConfig()
    p = cfg.params
    out = {}
    for n, (kind, mem) in PARAMS.items():
        if kind == "solv":
            out[n] = CHARMM19[TYPE_MAP[mem[0]][0]][4]            # dG_free of the first member (kcal/mol)
        elif kind == "lam":
            out[n] = 3.5
        elif kind == "rad":
            out[n] = CHARMM19[TYPE_MAP[mem[0]][0]][0]            # Rmin/2 (Å)
        elif kind == "eps":
            out[n] = CHARMM19[TYPE_MAP[mem[0]][0]][1]            # well depth (kcal/mol)
        elif kind in ("U", "prior"):
            out[n] = 1.0
        elif kind == "debye":
            out[n] = p.debye
        elif kind == "kT":
            out[n] = cfg.kT
        elif kind == "rep_scale":
            out[n] = cfg.rep_scale
        elif kind == "cap":
            out[n] = cfg.cap0
    return out


def _multiplier_index():
    """For each type-table column, the parameter index scaling each atom type (-1 = none)."""
    nt, nc = len(TYPES), len(COLS)
    idx = -np.ones((nt, nc), np.int64)
    col = {c: i for i, c in enumerate(COLS)}
    for k, n in enumerate(NAMES):
        kind, mem = PARAMS[n]
        if mem is None:
            continue
        for t in mem:
            ti = TYPES.index(t)
            cs = {"solv": ("dg_free", "dg_ref"), "lam": ("lam",), "rad": ("rmin_half", "rmin14_half"),
                  "eps": ("eps", "eps14")}[kind]
            for c in cs:
                assert idx[ti, col[c]] < 0, (t, c)
                idx[ti, col[c]] = k
    return idx


_MIDX = _multiplier_index()


@dataclass
class Physical:
    """Everything the engine needs for a given theta (tensors on one device)."""
    params: object               # EnergyParams with type_table / rep_scale / debye overrides
    cap: torch.Tensor            # scalar repulsion cap (kcal/mol per atom pair)
    umult: torch.Tensor          # [len(pack_terms)] multipliers of the packing terms in U
    kT: torch.Tensor
    wprior: torch.Tensor


def physical(theta: torch.Tensor, cfg: MutateConfig, terms: tuple, pack_terms: tuple) -> Physical:
    """theta [NP] (log scale, differentiable) -> engine parameters."""
    dev = theta.device
    base = torch.as_tensor(type_table(), device=dev)
    midx = torch.as_tensor(_MIDX, device=dev)
    ex = torch.exp(theta)
    mult = torch.where(midx >= 0, ex[midx.clamp(min=0)], torch.ones_like(base))
    tab = base * mult
    g = lambda n: ex[NAMES.index(n)]
    p = replace(cfg.params, rep_scale=cfg.rep_scale * g("rep_scale"), debye=cfg.params.debye * g("debye"),
                extra={**cfg.params.extra, "type_table": tab})
    um = torch.ones(len(pack_terms), device=dev)
    for j, t in enumerate(pack_terms):
        if f"U_{t}" in PARAMS:
            um = um.index_put((torch.tensor([j], device=dev),), g(f"U_{t}").reshape(1))
    return Physical(params=p, cap=cfg.cap0 * g("rep_cap"), umult=um, kT=cfg.kT * g("kT"), wprior=g("U_prior"))


def lean_config(chunk_elems: int = 1 << 22, **kw) -> MutateConfig:
    """Stage-1 config with only the terms the B6gs1 model / packing energy use (no lj_rep_soft, elec dh only).
    Terms are computed independently, so the remaining terms are identical to the full config's. chunk_elems:
    slot pairs per kernel chunk (memory; does not change results beyond float32 summation order)."""
    base = MutateConfig(rep_soft=0.0, **kw)
    base.params = replace(base.params, elec_modes=("dh",), chunk_elems=chunk_elems)
    base.pack_terms = tuple(t for t in base.pack_terms if t in E.term_names(base.params))
    return base


class SiteSet:
    """Fixed rotamer candidates for the needed (site, residue type) groups of one protein + differentiable
    Stage-1 evaluation. groups: {residue index: iterable of aa indices}. `sc` reuses an existing SiteScan."""

    @_float32
    def __init__(self, prot, groups: dict, cfg: MutateConfig | None = None, device="cuda", sc: SiteScan | None = None):
        self.cfg = cfg = cfg or lean_config()
        self.dev = torch.device(device)
        self.sc = sc = sc if sc is not None else SiteScan(prot, cfg, device)
        self.terms = sc.terms
        self.tidx = torch.tensor([self.terms.index(t) for t in TERMS], device=self.dev)
        self.pidx = torch.tensor([self.terms.index(t) for t in cfg.pack_terms], device=self.dev)
        sites = np.array(sorted(groups), np.int64)
        self.sites = sites
        cand = sc.site_candidates(sites)
        want = np.zeros((len(sites), 20), bool)
        for k, s in enumerate(sites):
            want[k, list(groups[s])] = True
        keep = want[cand["s"], cand["a"]]
        self.cs, self.ca = cand["s"][keep], cand["a"][keep]
        self.logp = torch.as_tensor(cand["logp"][keep], device=self.dev, dtype=torch.float32)
        self.excl = sites[self.cs]
        self.mov = sc._blocks(self.excl, self.ca, cand["xyz"][torch.as_tensor(np.flatnonzero(keep), device=self.dev)])
        # group layout: candidates are sorted by (site, aa) -> contiguous groups
        gkey = self.cs * 20 + self.ca
        first = np.r_[True, gkey[1:] != gkey[:-1]]
        self.ng = int(first.sum())
        self.group_site, self.group_aa = sites[self.cs[first]], self.ca[first]
        g = torch.as_tensor(np.cumsum(first) - 1, device=self.dev)
        counts = torch.bincount(g, minlength=self.ng)
        start = torch.cumsum(counts, 0) - counts
        self.g, self.rank = g, torch.arange(len(g), device=self.dev) - start[g]
        self.R = int(counts.max()) if self.ng else 0
        self.first = torch.as_tensor(np.flatnonzero(first), device=self.dev)
        self.n_rot = counts
        self.bounds = np.flatnonzero(np.r_[True, self.excl[1:] != self.excl[:-1], True])

    def group_index(self) -> dict:
        return {(int(s), int(a)): i for i, (s, a) in enumerate(zip(self.group_site, self.group_aa))}

    def _neighbours(self, env: E.Env):
        """theta-independent per-rotamer env neighbour lists [N, K] (as env_energy: per site, the env blocks within
        cutoff + extents of all candidates of that site, the site itself excluded; padded with the sentinel)."""
        sc, p = self.sc, self.cfg.params
        xd, md = self.mov.xyz, self.mov.mask
        lists, ks = [], []
        for lo, hi in zip(self.bounds[:-1], self.bounds[1:]):
            em = torch.ones(sc.L, dtype=torch.bool, device=self.dev)
            em[int(self.excl[lo])] = False
            ca = xd[lo, 1]
            reach = torch.where(md[lo:hi][..., None], xd[lo:hi] - ca, torch.zeros_like(xd[lo:hi])).norm(dim=-1).max()
            nb = E.env_neighbors(env, ca, reach, p, em)
            lists.append(nb); ks.append(len(nb))
        K = max(ks)
        nbp = torch.full((len(lists), K), env.n, dtype=torch.long, device=self.dev)
        for i, nb in enumerate(lists):
            nbp[i, :len(nb)] = nb
        site_of = torch.as_tensor(np.repeat(np.arange(len(lists)), np.diff(self.bounds)), device=self.dev)
        self._nb = nbp[site_of]                                                              # [N, K]
        self._qmask = torch.as_tensor(env.q["res"].cpu().numpy()[None, :] != self.excl[:, None], device=self.dev)
        # sort rotamers by slot width within chunks (as env_energy) -> less padding
        width = (md * torch.arange(1, E.A + 1, device=self.dev)).max(-1).values
        self._order = torch.argsort(width, stable=True)

    def energies(self, ph: Physical) -> torch.Tensor:
        """Per-rotamer per-term energies [N, n_terms] (intra + vs the protein without the site), as
        SiteScan._energies, with theta-dependent parameters (environment features rebuilt with them). All sites
        are evaluated together (per-rotamer neighbour lists, chunked) instead of one env_energy call per site."""
        sc, p = self.sc, ph.params
        rig = torch.as_tensor(sc.rigidity, device=self.dev)
        blocks = replace(sc.blocks, cap=ph.cap + self.cfg.cap1 * rig)
        env = E.make_env(blocks, None, p)
        if getattr(self, "_nb", None) is None:
            with torch.no_grad():
                self._neighbours(env)
        mov = replace(self.mov, cap=blocks.cap[torch.as_tensor(self.excl, device=self.dev)])
        f = E.features(mov, p)
        intra = E.intra_energy(mov, p, False, f)
        N, K = self._nb.shape
        se = env.sg
        parts, idxs = [], []
        step = max(1, p.chunk_elems // (K * E.A * E.A))
        def chunk(ii):
            fm = {k: v[ii][:, None] for k, v in f.items()}
            fe = E._gather_f(env.f, self._nb[ii])
            return E._env_terms(fm, fe, env.q, self._qmask[ii], se, None, p)
        ckpt = torch.is_grad_enabled()          # keep only one chunk's activations alive (recomputed in backward)
        for s0 in range(0, N, step):
            ii = self._order[s0:s0 + step]
            parts.append(torch.utils.checkpoint.checkpoint(chunk, ii, use_reentrant=False) if ckpt else chunk(ii))
            idxs.append(ii)
        ev = torch.cat(parts, 0)
        out = torch.zeros_like(ev).index_put((torch.cat(idxs),), ev)
        return intra + out

    def energies_loop(self, ph: Physical) -> torch.Tensor:
        """Reference implementation: one env_energy call per site (exactly SiteScan._energies)."""
        sc, p = self.sc, ph.params
        rig = torch.as_tensor(sc.rigidity, device=self.dev)
        blocks = replace(sc.blocks, cap=ph.cap + self.cfg.cap1 * rig)
        env = E.make_env(blocks, None, p)
        mov = replace(self.mov, cap=blocks.cap[torch.as_tensor(self.excl, device=self.dev)])
        f = E.features(mov, p)
        out = E.intra_energy(mov, p, False, f)
        parts = []
        for lo, hi in zip(self.bounds[:-1], self.bounds[1:]):
            em = torch.ones(sc.L, dtype=torch.bool, device=self.dev)
            em[int(self.excl[lo])] = False
            sl = slice(int(lo), int(hi))
            parts.append(E.env_energy(mov[sl], env, p, env_mask=em, f={k: v[sl] for k, v in f.items()}))
        return out + torch.cat(parts, 0)

    def ref(self, ph: Physical) -> torch.Tensor:
        """EEF1 reference solvation per group [G, 2] (apolar, polar), as SiteScan._ref."""
        f = E.features(self.mov[self.first], ph.params)
        r = f["dg_ref"] * f["m"].float()
        return torch.stack([(r * (1 - f["polar"])).sum(-1), (r * f["polar"]).sum(-1)], -1)

    def _pad(self, x, fill=0.0):
        pad = torch.full((self.ng, self.R, *x.shape[1:]), fill, device=self.dev, dtype=x.dtype)
        return pad.index_put((self.g, self.rank), x)

    def forward(self, ph: Physical) -> torch.Tensor:
        """Group features [G, len(GCOLS)] (differentiable in theta through ph)."""
        if self.ng == 0:
            return torch.zeros(0, len(GCOLS), device=self.dev)
        e = self.energies(ph)
        prior = -KT_LIB * self.logp
        U = (e[:, self.pidx] * ph.umult).sum(1) + ph.wprior * prior
        valid = self._pad(torch.ones_like(U, dtype=torch.bool), False)
        lg = torch.where(valid, -self._pad(U) / ph.kT, torch.full_like(valid, -float("inf"), dtype=U.dtype))
        logZ = torch.logsumexp(lg, 1)
        w = torch.exp(lg - logZ[:, None])
        wf = w[self.g, self.rank]
        Eavg = self._pad(e[:, self.tidx] * wf[:, None]).sum(1)
        ent = -(torch.where(w > 0, w * torch.log(w.clamp(min=1e-30)), torch.zeros_like(w))).sum(1)
        pav = self._pad(prior * wf).sum(1)
        F = -ph.kT * logZ
        return torch.cat([Eavg, ent[:, None], pav[:, None], self.ref(ph), F[:, None]], 1)


def theta_tensor(theta, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(theta, np.float32), device=device)


@_float32
def full_scan(prot, sites, theta, cfg: MutateConfig | None = None, device="cuda") -> dict:
    """No-grad Stage-1 scan of all 20 aa at `sites` with parameters theta -> arrays in the layout of
    SiteScan.stage1 (F, E_avg [S, 20, len(TERMS)], S_site, prior_avg, ref, n_rot) for predict.engine_columns."""
    cfg = cfg or lean_config()
    sites = np.asarray(sites, np.int64)
    out = {k: [] for k in ("F", "E_avg", "S_site", "prior_avg", "ref", "n_rot")}
    nT = len(TERMS)
    with torch.no_grad():
        sc = SiteScan(prot, cfg, device)
        ph = physical(theta_tensor(theta, device), cfg, sc.terms, cfg.pack_terms)
        for s0 in range(0, len(sites), cfg.site_chunk):
            ss = sites[s0:s0 + cfg.site_chunk]
            st = SiteSet(prot, {int(s): range(20) for s in ss}, cfg, device, sc=sc)
            G = st.forward(ph).cpu().numpy()
            n = len(ss)
            assert st.ng == 20 * n and (st.group_site == np.repeat(ss, 20)).all()
            out["E_avg"].append(G[:, :nT].reshape(n, 20, nT))
            out["S_site"].append(G[:, nT].reshape(n, 20))
            out["prior_avg"].append(G[:, nT + 1].reshape(n, 20))
            out["ref"].append(G[:, nT + 2:nT + 4].reshape(n, 20, 2))
            out["F"].append(G[:, nT + 4].reshape(n, 20))
            out["n_rot"].append(st.n_rot.cpu().numpy().reshape(n, 20))
    z = {k: np.concatenate(v) for k, v in out.items()}
    z.update(chain=prot.chain[sites], resnum=prot.resnum[sites], aa=np.array(list(prot.seq))[sites],
             terms=np.array(TERMS), sites=sites)
    return z


def penalty(theta: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """L2 (log scale) toward the literature values: lam * sum theta^2."""
    t = np.asarray(theta, float)
    return float(lam * (t ** 2).sum()), 2 * lam * t


def describe(theta: np.ndarray, se: np.ndarray | None = None, cfg: MutateConfig | None = None) -> list[dict]:
    """Parameter table rows: literature value, fitted value (natural units), scale exp(theta), SE of the scale."""
    lit = literature_values(cfg)
    rows = []
    for k, n in enumerate(NAMES):
        kind, mem = PARAMS[n]
        s = math.exp(float(theta[k]))
        r = {"param": n, "kind": kind, "members": ",".join(mem) if mem and kind != "lam" else "", "literature": lit[n],
             "fitted": lit[n] * s, "scale": s, "theta": float(theta[k])}
        if se is not None:
            r["scale_se"] = float(se[k])
        rows.append(r)
    return rows


# ============================================================================= multi-GPU worker pool (training)
def _pool_worker(conn, gpu: int, jobs: list, cfg_kw: dict):
    """Persistent GPU worker: holds SiteSets for its proteins; answers 'G' (features) and 'vjp' (theta gradient)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch as _t
    _t.set_num_threads(2)
    from .structure import annotate, load_protein
    cfg = lean_config(**cfg_kw)
    sets, info = {}, {}
    for key, path, groups in jobs:
        prot = annotate(load_protein(path, keep_het=False))
        idx = {(c, r): i for i, (c, r) in enumerate(zip(prot.chain.tolist(), prot.resnum.tolist()))}
        g = {}
        for (c, r), aas in groups.items():
            if (c, r) in idx:
                g.setdefault(idx[(c, r)], set()).update(int(T.AA_INDEX[a]) for a in aas)
        st = SiteSet(prot, g, cfg, "cuda")
        sets[key] = st
        info[key] = [(str(prot.chain[s]), str(prot.resnum[s]), str(T.AA1[a])) for s, a in zip(st.group_site, st.group_aa)]
    conn.send(("ready", info))
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "stop":
            break
        try:
            th = theta_tensor(msg[1], "cuda")
            if cmd == "G":
                out = {}
                with _t.no_grad():
                    for key, st in sets.items():
                        ph = physical(th, cfg, st.terms, cfg.pack_terms)
                        out[key] = st.forward(ph).double().cpu().numpy()
                conn.send(("ok", out))
            elif cmd == "vjp":
                cot = msg[2]
                thg = th.clone().requires_grad_(True)
                grad = _t.zeros_like(thg)
                for key, c in cot.items():
                    st = sets[key]
                    ph = physical(thg, cfg, st.terms, cfg.pack_terms)
                    G = st.forward(ph)
                    (g,) = _t.autograd.grad(G, thg, grad_outputs=_t.as_tensor(c, device=G.device, dtype=G.dtype))
                    grad += g
                conn.send(("ok", grad.double().cpu().numpy()))
        except Exception as e:  # noqa: BLE001
            import traceback
            conn.send(("error", traceback.format_exc()))


class ThetaPool:
    """One persistent worker process per GPU, proteins split by candidate count. jobs: list of
    (key, structure path, {(chain, resnum): set of aa letters}). After start, `info[key]` lists the groups
    (chain, resnum, aa) in the order of the feature rows returned by `G`."""

    def __init__(self, jobs: list, gpus: list, cfg_kw: dict | None = None, weights: dict | None = None):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        load = np.zeros(len(gpus))
        parts = [[] for _ in gpus]
        for j in sorted(jobs, key=lambda j: -(weights or {}).get(j[0], sum(len(v) for v in j[2].values()))):
            k = int(np.argmin(load))
            parts[k].append(j)
            load[k] += (weights or {}).get(j[0], sum(len(v) for v in j[2].values()))
        self.conns, self.procs = [], []
        for g, part in zip(gpus, parts):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_pool_worker, args=(b, g, part, cfg_kw or {}), daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)
        self.info, self.owner = {}, {}
        for k, c in enumerate(self.conns):
            tag, info = c.recv()
            if tag != "ready":
                raise RuntimeError(info)
            self.info.update(info)
            for key in info:
                self.owner[key] = k

    def _gather(self, msgs):
        for c, m in zip(self.conns, msgs):
            c.send(m)
        res = []
        for c in self.conns:
            tag, val = c.recv()
            if tag != "ok":
                raise RuntimeError(val)
            res.append(val)
        return res

    def G(self, theta) -> dict:
        out = {}
        for r in self._gather([("G", np.asarray(theta, np.float32))] * len(self.conns)):
            out.update(r)
        return out

    def vjp(self, theta, cot: dict) -> np.ndarray:
        per = [{} for _ in self.conns]
        for key, c in cot.items():
            per[self.owner[key]][key] = np.asarray(c, np.float32)
        res = self._gather([("vjp", np.asarray(theta, np.float32), p) for p in per])
        return np.sum(res, 0)

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=30)


# ============================================================================= whole-table feature computation
def _load_annotated(job):
    """(structure, sites) -> (structure, annotated Protein | None, sites | error) for the CPU pool."""
    from .structure import annotate, load_protein
    structure, sites = job
    path = structure if structure.startswith("/") else str(Path(__file__).resolve().parents[1] / structure)
    try:
        return structure, annotate(load_protein(path, keep_het=False)), sites
    except Exception as e:  # noqa: BLE001
        return structure, None, repr(e)


def scan_worker(gpu: int, jobs: list, theta, cdir: str, cpus: int):
    """One GPU process: full 20-aa Stage-1 scans at `theta`, cached per structure as .npz in `cdir`
    (file name = md5 of the structure path, the convention of scripts/55)."""
    import hashlib
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from concurrent.futures import ProcessPoolExecutor
    import torch as _t
    from .structure import annotate, load_protein
    _t.backends.cuda.matmul.allow_tf32 = False

    del annotate, load_protein
    with ProcessPoolExecutor(cpus) as ex:
        for structure, p, sites in ex.map(_load_annotated, jobs, chunksize=1):
            out = Path(cdir) / (hashlib.md5(structure.encode()).hexdigest() + ".npz")
            if p is None:
                np.savez(out, failed=np.array([str(sites)]))
                continue
            key = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
            idx = np.array(sorted({key[s] for s in sites if s in key}), np.int64)
            np.savez(out, **(full_scan(p, idx, theta, device="cuda") if len(idx) else {}))


def scan_structures(structures: dict, theta, cdir, gpus, cpus: int = 8):
    """Scan the structures that are not cached yet, one worker process per GPU (greedy balance by site count)."""
    import hashlib
    import multiprocessing as mp
    from pathlib import Path as _P
    cdir = _P(cdir)
    todo = [(s, sorted(v)) for s, v in structures.items()
            if not (cdir / (hashlib.md5(s.encode()).hexdigest() + ".npz")).exists()]
    if not todo:
        return
    todo.sort(key=lambda j: -len(j[1]))
    load, parts = [0] * len(gpus), [[] for _ in gpus]
    for j in todo:
        k = int(np.argmin(load))
        parts[k].append(j)
        load[k] += len(j[1])
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=scan_worker, args=(g, part, np.asarray(theta), str(cdir), cpus))
             for g, part in zip(gpus, parts) if part]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"scan worker failed with exit code {pr.exitcode}")
