"""PhysDDG energy function (engineer E2): GPU-batched, differentiable, term-decomposed.

Torch is used only as a GPU array / autograd library. Units: Å, kcal/mol, e. Float32.

Representation ("residue blocks"): every residue is a block of A = 15 atom slots (topology template order,
N CA C O side chain..., slot 14 = OXT). A `Blocks` object holds coordinates [..., R, A, 3] plus residue
type, presence mask, chain-aware sequence position, the previous residue's C (for the amide H), termini
flags, a burial value and a repulsion cap per residue. Energies between two blocks are evaluated densely
over the A x A slot pairs (masked), which makes all reductions plain tensor sums (bitwise deterministic
forward, no scatter/atomics) and lets the same code serve
  - whole-structure per-residue decomposition       `decompose(blocks)`
  - one residue (many variants/rotamers) vs a fixed environment   `env_energy(mov, env)`
  - rotamer-rotamer pair energies (broadcast)        `pair_energy(a, b)`
  - intra-residue energy                             `intra_energy(a)`
  - a local system of R flexible residues            `local_energy(mov[B,R], env)` -> one-body, two-body
All functions return per-term energies [..., n_terms] (never pre-summed); `term_names(params)` gives order.

Terms (see params.EnergyParams for constants):
  lj_atr      LJ attractive branch (r >= Rmin: eps[(Rmin/r)^12 - 2(Rmin/r)^6], -eps inside), switched 6->8 Å
  lj_rep      LJ repulsive branch eps(u-1)^2 (u=(Rmin/r)^6) for r < Rmin; linear above `cap` kcal/mol/pair
              (cap may be a per-residue tensor: Blocks.cap, pair cap = max of the two atoms' caps)
  solv_apolar EEF1 (Lazaridis-Karplus 1999) solvation of C/S atoms: sum_i [dG_ref_i - sum_j f_i(r_ij) V_j]
  solv_polar  same for N/O atoms (dG_ref included in intra / one-body energies when include_ref=True)
  hb_sc_bb    geometric H-bond (heavy-atom D..A distance x donor angle x acceptor angle), sc-bb pairs
  hb_sc_sc    same, sc-sc pairs
  hb_bb_bb    DSSP electrostatic H-bond energy (Kabsch & Sander 1983) between backbone N-H and C=O
              (amide H on the C(i-1)/CA bisector), floored at -3 kcal/mol, |i-j| >= 2
  elec_<mode> Coulomb between formal charges (topology; His neutral), modes: dh (Debye-Hückel, 0.15 M,
              eps 80; default), bur (DH-screened with burial-dependent eps_ij from 4 (both buried) to 80),
              dd (eps = 4r); CHARMM switch 10->12 Å
  disulf      -g(d_SG-SG) g(CB-SG-SG) g(SG-SG-CB) g(|chi_SS|) (Gaussians), for SG-SG < 3 Å
Exclusions: pairs 1-2 and 1-3 bonds apart excluded; 1-4 pairs use CHARMM19 1-4 vdW and are scaled by w14;
intra-residue pairs whose distance does not depend on any chi angle are excluded (rigid groups only add per-aa
constants), as are backbone-backbone pairs of peptide-bonded neighbours (fixed by the backbone); H-bonds need
>= 4 bonds separation; electrostatics excludes intra-residue pairs; SG-SG < 2.5 Å treated as a covalent bond.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from . import topology as topo
from .params import COLS, EnergyParams, type_table

A = 15                      # slots per residue block
OXT = 14
SR_TERMS = ("lj_atr", "lj_rep", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb")
FAR = 1.0e4                 # coordinate used for absent atoms
BIG2 = 1.0e4                # r^2 assigned to pairs outside all cutoffs (r = 100 Å)
C_IDX = topo.AA_INDEX["C"]


def term_names(p: EnergyParams | None = None) -> tuple[str, ...]:
    p = p or EnergyParams()
    soft = ("lj_rep_soft",) if p.rep_soft > 0 else ()
    return SR_TERMS + soft + tuple(f"elec_{m}" for m in p.elec_modes) + ("disulf",)


# ============================================================================= template tables
def slot_names(aa: str) -> list[str]:
    names = list(topo.TEMPLATES[aa].atoms)
    return names + [""] * (OXT - len(names)) + ["OXT"]


def residue_bonds(aa: str) -> list[tuple[str, str]]:
    """Intra-residue heavy-atom bonds incl. C-OXT (topology.BONDS, E1; formerly a private table here)."""
    return list(topo.BONDS[aa]) + [("C", "OXT")]


def _graph_dist(aa: str) -> np.ndarray:
    """All-pairs bond-graph distance [A, A] within one residue (inf for absent slots)."""
    s = {nm: i for i, nm in enumerate(slot_names(aa)) if nm}
    D = np.full((A, A), np.inf)
    np.fill_diagonal(D, 0)
    for a, b in residue_bonds(aa):
        D[s[a], s[b]] = D[s[b], s[a]] = 1
    for k in range(A):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def _downstream(aa: str, b: str, c: str) -> set[int]:
    """Slots moved by rotation about bond b-c (c side, c excluded)."""
    s = {nm: i for i, nm in enumerate(slot_names(aa)) if nm}
    nb = {i: set() for i in s.values()}
    for x, y in residue_bonds(aa):
        nb[s[x]].add(s[y]); nb[s[y]].add(s[x])
    seen, todo = {s[c]}, [s[c]]
    while todo:
        u = todo.pop()
        for v in nb[u]:
            if v != s[b] and v not in seen:
                seen.add(v); todo.append(v)
    return seen - {s[c]}


def build_tables() -> dict[str, np.ndarray]:
    n = len(topo.AA1)
    T = {
        "valid": np.zeros((n, A), bool), "type": np.zeros((n, A), np.int64), "charge": np.zeros((n, A), np.float32),
        "don": np.zeros((n, A), bool), "acc": np.zeros((n, A), bool), "dcone": np.zeros((n, A), bool),
        "aring": np.zeros((n, A), bool), "b1": np.full((n, A), -1, np.int64), "b2": np.full((n, A), -1, np.int64),
        "sep_intra": np.zeros((n, A, A), np.int8), "sep_pep": np.zeros((n, n, A, A), np.int8),
        "sep_ss": np.zeros((A, A), np.int8),
        "is_bb": np.array([k < 4 or k == OXT for k in range(A)]),
    }
    D = {}
    for k, aa in enumerate(topo.AA1):
        t = topo.TEMPLATES[aa]
        names = slot_names(aa)
        s = {nm: i for i, nm in enumerate(names) if nm}
        na = len(t.atoms)
        T["valid"][k, :na] = True
        T["valid"][k, OXT] = True
        T["type"][k, :na] = t.type_ids
        T["type"][k, OXT] = topo.TYPE_INDEX[topo.OXT_TYPE]
        T["charge"][k, :na] = t.charges
        roles = dict(topo.hbond_roles(aa))
        roles["OXT"] = topo.HB_BACKBONE["OXT"]
        for nm, hb in roles.items():
            i = s[nm]
            T["don"][k, i] = hb.donor > 0
            T["acc"][k, i] = hb.acceptor
            T["dcone"][k, i] = hb.don_geom in ("rot", "sp2_2H")
            T["aring"][k, i] = hb.acc_hyb == "ring"
            T["b1"][k, i] = -2 if hb.base == "-C" else s[hb.base]
            T["b2"][k, i] = s.get(hb.base2, -1)
        # bond-graph separations; intra pairs rigid w.r.t. all chis are excluded (sep := 0)
        Dk = _graph_dist(aa)
        D[k] = Dk
        sep = np.clip(Dk, 0, 4)
        sep[~np.isfinite(Dk)] = 0
        moving = [_downstream(aa, q[1], q[2]) for q in topo.CHI_ATOMS[aa]] if aa != "P" else []
        axes = [{s[q[1]], s[q[2]]} for q in topo.CHI_ATOMS[aa]] if aa != "P" else []
        varies = np.zeros((A, A), bool)
        for mv, ax in zip(moving, axes):
            inm = np.zeros(A, bool); inm[list(mv)] = True
            onax = np.zeros(A, bool); onax[list(ax)] = True
            varies |= (inm[:, None] & ~inm[None, :] & ~onax[None, :]) | (inm[None, :] & ~inm[:, None] & ~onax[:, None])
        sep[~varies | (sep < 3)] = 0
        T["sep_intra"][k] = sep.astype(np.int8)
    iC, iN, iSG = 2, 0, 5
    for k in range(n):
        for l in range(n):
            d = D[k][:, iC][:, None] + 1 + D[l][iN, :][None, :]
            T["sep_pep"][k, l] = np.where(np.isfinite(d), np.clip(d, 0, 4), 0).astype(np.int8)
    d = D[C_IDX][:, iSG][:, None] + 1 + D[C_IDX][iSG, :][None, :]
    T["sep_ss"] = np.where(np.isfinite(d), np.clip(d, 0, 4), 0).astype(np.int8)
    return T


_TAB_NP = build_tables()
_TAB_CACHE: dict = {}


def tables(device) -> dict[str, torch.Tensor]:
    key = str(device)
    if key not in _TAB_CACHE:
        t = {k: torch.as_tensor(v, device=device) for k, v in _TAB_NP.items()}
        t["param"] = torch.as_tensor(type_table(), device=device)
        _TAB_CACHE[key] = t
    return _TAB_CACHE[key]


# ============================================================================= residue blocks
@dataclass
class Blocks:
    """Residue blocks with arbitrary leading batch dims [...]: one residue per [...] element."""
    xyz: torch.Tensor      # [..., A, 3] float
    aa: torch.Tensor       # [...] long, index into topology.AA1
    mask: torch.Tensor     # [..., A] bool, atom present
    pos: torch.Tensor      # [...] long, chain-aware position: peptide-bonded neighbours differ by exactly 1
    prev_c: torch.Tensor   # [..., 3] C of residue i-1 (NaN if none: chain start / break)
    nterm: torch.Tensor    # [...] bool, carries the N-terminal +1 charge
    burial: torch.Tensor   # [...] float 0..1 (CB neighbour count based), for burial-dependent dielectric
    cap: torch.Tensor | None = None   # [...] repulsion cap (kcal/mol per atom pair); None -> params.rep_cap

    def __getitem__(self, idx) -> "Blocks":
        return Blocks(**{k: (None if v is None else v[idx]) for k, v in self.__dict__.items()})

    @property
    def shape(self):
        return self.aa.shape

    @property
    def device(self):
        return self.xyz.device

    def to(self, device) -> "Blocks":
        return Blocks(**{k: (None if v is None else v.to(device)) for k, v in self.__dict__.items()})


def burial_from_ca_cb(xyz: torch.Tensor, mask: torch.Tensor, p: EnergyParams) -> torch.Tensor:
    """Per-residue burial 0..1 from the number of CB (CA for Gly) within 10 Å, linear between burial_n."""
    cb = torch.where(mask[:, 4:5], xyz[:, 4], xyz[:, 1])
    d2 = ((cb[:, None] - cb[None]) ** 2).sum(-1)
    n = (d2 < 100.0).sum(1).float() - 1
    lo, hi = p.burial_n
    return ((n - lo) / (hi - lo)).clamp(0, 1)


def blocks_from_arrays(xyz14, mask14, oxt, aa, chain, device="cpu", params: EnergyParams | None = None,
                       termini: bool = True, max_cn: float = 2.0) -> Blocks:
    """Build whole-protein Blocks [L] from atom14 arrays (structure.Protein layout)."""
    p = params or EnergyParams()
    xyz14 = np.asarray(xyz14, np.float32)
    L = len(aa)
    xyz = np.concatenate([xyz14, np.asarray(oxt, np.float32)[:, None]], 1)
    mask = np.concatenate([np.asarray(mask14, bool), ~np.isnan(np.asarray(oxt)[:, :1])], 1)
    xyz = np.where(mask[..., None], np.nan_to_num(xyz), 0.0).astype(np.float32)
    chain = np.asarray(chain)
    brk = np.ones(L, bool)
    if L > 1:
        dcn = np.linalg.norm(xyz[1:, 0] - xyz[:-1, 2], axis=-1)
        brk[1:] = (chain[1:] != chain[:-1]) | ~(dcn <= max_cn) | ~mask[1:, 0] | ~mask[:-1, 2]
    pos = np.arange(L) + 2 * np.cumsum(brk)
    prev_c = np.full((L, 3), np.nan, np.float32)
    prev_c[1:] = np.where(brk[1:, None], np.nan, xyz[:-1, 2])
    chain_start = np.ones(L, bool)
    chain_start[1:] = chain[1:] != chain[:-1]
    t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
    xyz_t, mask_t = t(xyz, torch.float32), t(mask)
    return Blocks(xyz=xyz_t, aa=t(np.asarray(aa), torch.long), mask=mask_t, pos=t(pos, torch.long),
                  prev_c=t(prev_c, torch.float32), nterm=t(chain_start & termini),
                  burial=burial_from_ca_cb(xyz_t, mask_t, p))


def blocks_from_protein(prot, device="cpu", params: EnergyParams | None = None, termini: bool = True) -> Blocks:
    """Adapter for physddg.structure.Protein (E1)."""
    return blocks_from_arrays(prot.xyz14, prot.mask14, prot.oxt, prot.aa, prot.chain, device, params, termini)


# ============================================================================= per-slot features
def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def features(b: Blocks, p: EnergyParams) -> dict[str, torch.Tensor]:
    """Per-slot features [..., A, *] for energy evaluation (differentiable in b.xyz)."""
    T = tables(b.device)
    aa = b.aa
    m = b.mask & T["valid"][aa]
    x = torch.where(m[..., None], b.xyz, torch.full_like(b.xyz, FAR))
    t = T["type"][aa]
    q = T["charge"][aa].clone()
    q[..., 0] = q[..., 0] + b.nterm.float()
    cterm = m[..., OXT]
    q[..., 3] = q[..., 3] - 0.5 * cterm.float()
    q[..., OXT] = q[..., OXT] - 0.5 * cterm.float()
    q = q * m.float()
    don, acc = T["don"][aa] & m, T["acc"][aa] & m
    dcone, aring = T["dcone"][aa].clone(), T["aring"][aa]
    b1, b2 = T["b1"][aa].clone(), T["b2"][aa]
    # amide N without a previous C (chain start/break): NH3+-like cone donor with base CA
    has_prev = ~torch.isnan(b.prev_c[..., 0])
    no_prev_n = (b1[..., 0] == -2) & ~has_prev
    b1[..., 0] = torch.where(no_prev_n, torch.ones_like(b1[..., 0]), b1[..., 0])
    dcone[..., 0] = dcone[..., 0] | no_prev_n
    pc = torch.where(has_prev[..., None], b.prev_c, x[..., 1, :])          # NaN-free
    xe = torch.cat([x, pc[..., None, :]], -2)                               # slot A = previous C
    idx1 = torch.where(b1 == -2, torch.full_like(b1, A), b1.clamp(min=0))
    idx2 = b2.clamp(min=0)
    g = lambda idx: torch.gather(xe, -2, idx[..., None].expand(*idx.shape, 3))
    x1, x2 = g(idx1), g(idx2)
    u1, u2 = _unit(x1 - x), _unit(x2 - x)
    bis = -_unit(u1 + u2)                       # external bisector (H direction / ring lone pair)
    dvec = torch.where(dcone[..., None], -u1, bis)          # cone: base->donor axis; else D->H
    avec = torch.where(aring[..., None], bis, -u1)          # base->acceptor direction
    # optional per-type parameter table override (E3, physddg/diffparams.py: differentiable refit of the physical
    # parameters); absent by default -> the frozen literature table, identical code path
    tab = p.extra.get("type_table") if p.extra else None
    par = (T["param"] if tab is None else tab)[t]           # [..., A, n_cols]
    cap = b.cap if b.cap is not None else torch.full_like(b.burial, p.rep_cap)
    f = dict(x=x, m=m, t=t, q=q, don=don, acc=acc, dcone=dcone, dvec=dvec, avec=avec,
             bb=T["is_bb"].expand_as(m), cap=cap[..., None].expand_as(q), bur=b.burial[..., None].expand_as(q),
             aa=aa, pos=b.pos)
    for i, c in enumerate(COLS):
        f[c] = par[..., i]
    return f


# ============================================================================= pair geometry helpers
def _switch(r2: torch.Tensor, on: float, off: float) -> torch.Tensor:
    """CHARMM energy switch S(r) on r^2: 1 below `on`, 0 above `off`, C1-smooth in between."""
    on2, off2 = on * on, off * off
    s = (off2 - r2) ** 2 * (off2 + 2 * r2 - 3 * on2) / (off2 - on2) ** 3
    return torch.where(r2 <= on2, torch.ones_like(r2), torch.where(r2 >= off2, torch.zeros_like(r2), s))


def _ramp(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    """Smoothstep from 0 (x <= x0) to 1 (x >= x1); x0 > x1 gives a decreasing ramp."""
    t = ((x - x0) / (x1 - x0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _cosd(a: float) -> float:
    return math.cos(math.radians(a))


def _pair_sep(fa: dict, fb: dict) -> torch.Tensor:
    """Bond separation codes [..., A, A] (int8, 0 = excluded, 3 = 1-4, 4 = far) between two different residues."""
    T = tables(fa["x"].device)
    na, nb = fa["x"].shape[-2], fb["x"].shape[-2]               # slot widths (may be trimmed, see _trim)
    aa_a, aa_b = fa["aa"], fb["aa"]
    dpos = fb["pos"] - fa["pos"]
    shape = torch.broadcast_shapes(aa_a.shape, aa_b.shape)
    aa_a, aa_b, dpos = aa_a.expand(shape), aa_b.expand(shape), dpos.expand(shape)
    far = torch.full((*shape, na, nb), 4, dtype=torch.int8, device=aa_a.device)
    fwd = T["sep_pep"][aa_a, aa_b][..., :na, :nb]                              # a precedes b
    bwd = T["sep_pep"][aa_b, aa_a].transpose(-1, -2)[..., :na, :nb]            # b precedes a
    adj = (dpos.abs() == 1)[..., None, None]
    sep = torch.where((dpos == 1)[..., None, None], fwd, torch.where((dpos == -1)[..., None, None], bwd, far))
    # backbone-backbone pairs of peptide-bonded neighbours: fixed by the backbone, aa-independent -> excluded
    bbbb = T["is_bb"][:na, None] & T["is_bb"][None, :nb]
    sep = torch.where(adj & bbbb, torch.zeros_like(sep), sep)
    # disulfide bond (both Cys, SG-SG < ss_bond_max)
    if na > 5 and nb > 5:
        sga, sgb = fa["x"][..., 5, :], fb["x"][..., 5, :]
        ss = (aa_a == C_IDX) & (aa_b == C_IDX) & (((sga - sgb) ** 2).sum(-1) < 2.5 ** 2) & fa["m"][..., 5] & fb["m"][..., 5]
        sep = torch.where(ss[..., None, None], torch.minimum(sep, T["sep_ss"][:na, :nb]), sep)
    return sep


SLOT_KEYS = ("x", "m", "t", "q", "don", "acc", "dcone", "dvec", "avec", "bb", "cap", "bur") + COLS


VEC_KEYS = ("x", "dvec", "avec")


def _trim(f: dict, n: int) -> dict:
    """Keep the first n atom slots of every per-slot feature (empty slot tails cost compute only)."""
    if n >= A:
        return f
    return {k: (v.narrow(-2 if k in VEC_KEYS else -1, 0, n) if k in SLOT_KEYS else v) for k, v in f.items()}


def _width(m: torch.Tensor) -> int:
    """1 + index of the last present slot over all leading dims of mask m [..., A]."""
    pres = m.reshape(-1, m.shape[-1]).any(0)
    return int(torch.nonzero(pres).max().item()) + 1 if bool(pres.any()) else 1


# ============================================================================= short-range core
def _core(fa: dict, fb: dict, sep: torch.Tensor, nonadj: torch.Tensor, p: EnergyParams) -> torch.Tensor:
    """Short-range terms summed over slot pairs. fa/fb fields [..., A, *] (broadcastable leading dims),
    sep [..., A, A] separation codes, nonadj [...] bool (|i-j| >= 2, enables backbone-backbone H-bonds).
    Returns [..., len(SR_TERMS) (+1 if p.rep_soft > 0: lj_rep_soft)]."""
    ea = lambda k: fa[k][..., :, None]          # [..., A, 1]
    eb = lambda k: fb[k][..., None, :]          # [..., 1, A]
    d = fb["x"][..., None, :, :] - fa["x"][..., :, None, :]          # [..., A, A, 3], a -> b
    r2 = (d * d).sum(-1)
    cut = max(p.lj_off, p.solv_off)
    pm = ea("m") & eb("m") & (sep >= 3)
    near = pm & (r2 < cut * cut)
    r2 = torch.where(near, r2, torch.full_like(r2, BIG2))
    r = r2.sqrt()
    nearf = near.float()
    # ---- Lennard-Jones
    hbpair = (ea("don") & eb("acc")) | (ea("acc") & eb("don"))
    is14 = sep == 3
    rmin = torch.where(is14, ea("rmin14_half") + eb("rmin14_half"), ea("rmin_half") + eb("rmin_half"))
    rmin = torch.where(hbpair, rmin.clamp(max=p.hb_rmin), rmin)
    eps = torch.where(is14, ea("eps14") * eb("eps14"), ea("eps") * eb("eps")).sqrt()
    w = torch.where(is14, torch.full_like(r2, p.w14), torch.ones_like(r2)) * nearf
    rmin2 = rmin * rmin
    s6 = (rmin2 / r2) ** 3
    atr = torch.where(r2 < rmin2, -eps, eps * (s6 * s6 - 2 * s6)) * _switch(r2, p.lj_on, p.lj_off)
    rr2 = rmin2 * (p.rep_scale ** 2)
    cap = torch.maximum(ea("cap"), eb("cap"))
    uc = 1 + (cap / eps).sqrt()                                # (Rr/rc)^6 at the cap point
    rc2 = rr2 * uc ** (-1.0 / 3.0)
    s6r = (rr2 / torch.maximum(r2, rc2)) ** 3
    slope = 12 * eps * (uc - 1) * uc ** (7.0 / 6.0) / rr2.sqrt()  # |dE/dr| at rc
    rep = torch.where(r2 < rr2, eps * (s6r - 1) ** 2, torch.zeros_like(r2)) \
        + torch.where(r2 < rc2, slope * (rc2.sqrt() - r), torch.zeros_like(r2))
    # ---- EEF1 solvation: atom a loses -f_a(r) V_b, atom b loses -f_b(r) V_a
    ga, gb = _eef1(fa, fb, r2, near, p)
    pa, pb = ea("polar"), eb("polar")
    solv_polar = ga * pa + gb * pb
    solv_apolar = ga * (1 - pa) + gb * (1 - pb)
    # ---- geometric H-bond (side chains); donor/acceptor angle windows
    hbm = near & (sep >= 4)
    u = d / r[..., None]
    fd = _ramp(r, p.hb_d_min[0], p.hb_d_min[1]) * _ramp(r, p.hb_d_max[1], p.hb_d_max[0])

    def f_don(c, cone):
        b0, b1 = p.hb_don_bis
        c0, c1, c2, c3 = p.hb_don_cone
        bis = _ramp(c, _cosd(b0), _cosd(b1))
        con = _ramp(c, _cosd(c3), _cosd(c2)) * _ramp(c, _cosd(c0), _cosd(c1))
        return torch.where(cone, con, bis)

    fA = lambda c: _ramp(c, _cosd(p.hb_acc[0]), _cosd(p.hb_acc[1]))
    cda = (fa["dvec"][..., :, None, :] * u).sum(-1)      # donor a: dvec_a . (a->b)
    cdb = -(fb["dvec"][..., None, :, :] * u).sum(-1)     # donor b: dvec_b . (b->a)
    caa = (fa["avec"][..., :, None, :] * u).sum(-1)      # acceptor a: avec_a . (a->b)
    cab = -(fb["avec"][..., None, :, :] * u).sum(-1)     # acceptor b: avec_b . (b->a)
    f_ab = torch.where(ea("don") & eb("acc"), f_don(cda, ea("dcone")) * fA(cab), torch.zeros_like(r))
    f_ba = torch.where(eb("don") & ea("acc"), f_don(cdb, eb("dcone")) * fA(caa), torch.zeros_like(r))
    hb = -torch.maximum(f_ab, f_ba) * fd * hbm.float()
    bba, bbb = ea("bb"), eb("bb")
    hb_scbb = hb * (bba ^ bbb).float()
    hb_scsc = hb * (~bba & ~bbb).float()
    # ---- backbone-backbone: DSSP electrostatic model on N(slot 0) vs O/OXT(slots 3, 14)
    hb_bb = _dssp_pair(fa, fb, nonadj, p) + _dssp_pair(fb, fa, nonadj, p)
    out = torch.stack([(atr * w).sum((-1, -2)), (rep * w).sum((-1, -2)), solv_apolar.sum((-1, -2)),
                       solv_polar.sum((-1, -2)), hb_scbb.sum((-1, -2)), hb_scsc.sum((-1, -2))], -1)
    shape = out.shape[:-1]
    out = torch.cat([out, hb_bb.expand(shape)[..., None]], -1)
    if p.rep_soft > 0:     # log-saturating repulsion on the uncapped branch (extra feature term, after hb_bb_bb)
        raw = torch.where(r2 < rr2, eps * ((rr2 / r2.clamp(min=0.01)) ** 3 - 1) ** 2, torch.zeros_like(r2))
        soft = p.rep_soft * torch.log1p(raw / p.rep_soft)
        out = torch.cat([out, (soft * w).sum((-1, -2))[..., None]], -1)
    return out


def _eef1(fa: dict, fb: dict, r2: torch.Tensor, near: torch.Tensor, p: EnergyParams):
    """EEF1 pair solvation changes [..., A, A]: ga = -f_a(r) V_b (for atom a), gb = -f_b(r) V_a (for atom b),
    f_i(r) = dG_free_i / (2 pi^1.5 lambda_i r^2) exp(-((r - R_i)/lambda_i)^2), switched solv_on -> solv_off."""
    ea = lambda k: fa[k][..., :, None]
    eb = lambda k: fb[k][..., None, :]
    k = 1.0 / (2.0 * math.pi ** 1.5)
    rs2 = r2.clamp(min=1.0)
    rs = rs2.sqrt()
    fa_ = k * ea("dg_free") / (ea("lam") * rs2) * torch.exp(-((rs - ea("rmin_half")) / ea("lam")) ** 2)
    fb_ = k * eb("dg_free") / (eb("lam") * rs2) * torch.exp(-((rs - eb("rmin_half")) / eb("lam")) ** 2)
    sw = _switch(r2, p.solv_on, p.solv_off)
    zero = torch.zeros_like(r2)
    return torch.where(near, -fa_ * eb("volume") * sw, zero), torch.where(near, -fb_ * ea("volume") * sw, zero)


def _dssp_pair(fd: dict, fa: dict, nonadj: torch.Tensor, p: EnergyParams) -> torch.Tensor:
    """DSSP energy for donor N-H of block fd vs acceptor C=O (O and OXT) of block fa, floored, switched."""
    n = fd["x"][..., 0, :]
    h = n + p.nh_len * fd["dvec"][..., 0, :]
    okd = fd["don"][..., 0] & ~fd["dcone"][..., 0]                     # needs a placed amide H
    tot = 0.0
    for s in (3, OXT):
        if s >= fa["x"].shape[-2]:
            continue
        o = fa["x"][..., s, :]
        c = o - p.co_len * fa["avec"][..., s, :]
        ok = okd & fa["acc"][..., s] & nonadj
        r_on2 = ((o - n) ** 2).sum(-1)
        ok = ok & (r_on2 < p.dssp_off ** 2)
        big = torch.full_like(r_on2, 100.0)
        dist = lambda a, b: torch.where(ok, ((a - b) ** 2).sum(-1), big).clamp(min=0.25).sqrt()
        e = p.dssp_q * (1 / dist(o, n) + 1 / dist(c, h) - 1 / dist(o, h) - 1 / dist(c, n))
        e = e.clamp(min=p.dssp_floor, max=0.0) * _switch(torch.where(ok, r_on2, big), p.dssp_on, p.dssp_off)
        tot = tot + torch.where(ok, e, torch.zeros_like(e))
    return tot


# ============================================================================= electrostatics & disulfide
def _elec(xa, qa, ba, xb, qb, bbur, valid, p: EnergyParams) -> torch.Tensor:
    """Coulomb between point charges; inputs broadcast to [..., Ma, Mb]; returns [..., n_modes] summed."""
    d = xb[..., None, :, :] - xa[..., :, None, :]
    r2 = (d * d).sum(-1)
    ok = valid & (r2 < p.elec_off ** 2)
    r2 = torch.where(ok, r2, torch.full_like(r2, BIG2))
    sw = _switch(r2, p.elec_on, p.elec_off) * ok.float()
    r = r2.clamp(min=p.elec_rmin ** 2).sqrt()
    qq = qa[..., :, None] * qb[..., None, :] * p.coulomb * sw
    out = []
    for mode in p.elec_modes:
        if mode == "dd":
            e = qq / (p.eps_dd * r * r)
        elif mode == "dh":
            e = qq * torch.exp(-r / p.debye) / (p.eps_dh * r)
        elif mode == "bur":
            eps = p.eps_in + (p.eps_out - p.eps_in) * (1 - (ba[..., :, None] * bbur[..., None, :]).clamp(min=0).sqrt())
            e = qq * torch.exp(-r / p.debye) / (eps * r)
        else:
            raise ValueError(mode)
        out.append(e.sum((-1, -2)))
    return torch.stack(out, -1)


def _gauss(x, mu, sd):
    return torch.exp(-0.5 * ((x - mu) / sd) ** 2)


def _disulf(sga, cba, sgb, cbb, ok, p: EnergyParams) -> torch.Tensor:
    """Disulfide bonus (<= 0) for SG/CB pairs [..., 3] (broadcast), masked by ok."""
    v = sgb - sga
    dd = (v * v).sum(-1)
    ok = ok & (dd < 9.0)
    v = torch.where(ok[..., None], v, torch.ones_like(v))
    d = (v * v).sum(-1).sqrt()
    b1 = sga - cba
    b3 = cbb - sgb
    ang = lambda x, y: torch.rad2deg(torch.acos(((x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1) + 1e-8)).clamp(-0.9999, 0.9999)))
    th1 = ang(-b1, v)
    th2 = ang(-v, b3)
    n1 = torch.cross(b1, v, dim=-1)
    n2 = torch.cross(v, b3, dim=-1)
    m1 = torch.cross(n1, _unit(v), dim=-1)
    chi = torch.rad2deg(torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1) + 1e-8))
    e = -_gauss(d, *p.ss_d) * _gauss(th1, *p.ss_ang) * _gauss(th2, *p.ss_ang) * _gauss(chi.abs(), *p.ss_dih)
    return torch.where(ok, e, torch.zeros_like(e))


def _block_pair_all(fa: dict, fb: dict, p: EnergyParams) -> torch.Tensor:
    """All terms between two different residues (broadcast leading dims) -> [..., n_terms]."""
    fa, fb = _trim(fa, _width(fa["m"])), _trim(fb, _width(fb["m"]))
    sep = _pair_sep(fa, fb)
    dpos = (fb["pos"] - fa["pos"]).abs()
    sr = _core(fa, fb, sep, dpos >= 2, p)
    qv = (fa["q"] != 0)[..., :, None] & (fb["q"] != 0)[..., None, :] & (sep >= 4)
    el = _elec(fa["x"], fa["q"], fa["bur"], fb["x"], fb["q"], fb["bur"], qv, p)
    if fa["x"].shape[-2] > 5 and fb["x"].shape[-2] > 5:
        ok = (fa["aa"] == C_IDX) & (fb["aa"] == C_IDX) & fa["m"][..., 5] & fb["m"][..., 5]
        ss = _disulf(fa["x"][..., 5, :], fa["x"][..., 4, :], fb["x"][..., 5, :], fb["x"][..., 4, :], ok, p)
    else:
        ss = sr.new_zeros(sr.shape[:-1])
    shape = torch.broadcast_shapes(sr.shape[:-1], el.shape[:-1], ss.shape)
    return torch.cat([sr.expand(*shape, -1), el.expand(*shape, -1), ss.expand(shape)[..., None]], -1)


_core_eager = _core


def enable_compile(on: bool = True) -> None:
    """Fuse the short-range kernel with torch.compile (Triton). ~2x faster on large batches, ~40 s compile per
    new shape pattern, results differ from eager by float32 rounding (~1e-5 relative). Off by default."""
    global _core
    _core = torch.compile(_core_eager, dynamic=True) if on else _core_eager


# ============================================================================= public API
def intra_energy(b: Blocks, p: EnergyParams | None = None, include_ref: bool = True,
                 f: dict | None = None) -> torch.Tensor:
    """Intra-residue energy [..., n_terms]: chi-dependent non-bonded pairs (+ EEF1 dG_ref of all atoms)."""
    p = p or EnergyParams()
    f = f if f is not None else features(b, p)
    T = tables(b.device)
    w = _width(f["m"])
    ft = _trim(f, w)
    sep = T["sep_intra"][f["aa"]][..., :w, :w]
    sep = sep * torch.triu(torch.ones(w, w, dtype=torch.int8, device=b.device), 1)
    sr = _core(ft, ft, sep, torch.zeros_like(f["aa"], dtype=torch.bool), p)
    if include_ref:
        ref = f["dg_ref"] * f["m"].float()
        sr = sr.clone()
        sr[..., 2] = sr[..., 2] + (ref * (1 - f["polar"])).sum(-1)
        sr[..., 3] = sr[..., 3] + (ref * f["polar"]).sum(-1)
    n_el = len(p.elec_modes)
    return torch.cat([sr, sr.new_zeros(*sr.shape[:-1], n_el + 1)], -1)


def pair_energy(a: Blocks, b: Blocks, p: EnergyParams | None = None) -> torch.Tensor:
    """Energy between residue blocks a and b (different residues; leading dims broadcast) -> [..., n_terms].
    E.g. rotamer-rotamer tables: a[Na, 1], b[1, Nb] -> [Na, Nb, n_terms]."""
    p = p or EnergyParams()
    return _block_pair_all(features(a, p), features(b, p), p)


def _chunks(n: int, per_item: int, p: EnergyParams):
    step = max(1, p.chunk_elems // max(per_item, 1))
    for s in range(0, n, step):
        yield slice(s, min(n, s + step))


def _gather_f(f: dict, idx: torch.Tensor) -> dict:
    """Index the leading residue dim of feature dict f with idx (any shape)."""
    return {k: v[idx] for k, v in f.items()}


@dataclass
class Env:
    """Fixed environment for local energies: per-slot features of E residues (+ one empty sentinel block at
    index E) and flat lists of its charged atoms and Cys SG atoms (electrostatics / disulfide)."""
    f: dict
    ca: torch.Tensor          # [E+1, 3]
    reach: torch.Tensor       # [E+1] max distance of a present atom from CA
    res_index: torch.Tensor   # [E] index of each env residue in the source Blocks
    q: dict                   # charged atoms: x [Nc,3], q, bur, aa, slot, pos, res (env block index)
    sg: dict                  # Cys SG: x [Ns,3], cb [Ns,3], res

    @property
    def n(self) -> int:
        return self.ca.shape[0] - 1


def make_env(blocks: Blocks, exclude=None, p: EnergyParams | None = None) -> Env:
    """Environment from whole-protein Blocks [L] without residues `exclude` (indices or bool mask)."""
    p = p or EnergyParams()
    L = blocks.shape[0]
    keep = torch.ones(L, dtype=torch.bool, device=blocks.device)
    if exclude is not None:
        keep[torch.as_tensor(exclude, device=blocks.device)] = False
    idx = torch.nonzero(keep).squeeze(1)
    f = features(blocks[idx], p)
    E_ = idx.shape[0]
    pad = {}
    for k, v in f.items():           # sentinel = copy of block 0 with no atoms, far away
        z = v[:1].clone()
        if k == "x":
            z.fill_(FAR)
        elif k == "pos":
            z.fill_(-1000)
        elif v.dtype == torch.bool:
            z.fill_(False)
        elif k == "q":
            z.zero_()
        pad[k] = torch.cat([v, z], 0)
    ca = pad["x"][:, 1]
    dist = torch.where(pad["m"][..., None], pad["x"] - ca[:, None], torch.zeros_like(pad["x"])).norm(dim=-1)
    r, s_ = torch.nonzero(f["q"] != 0, as_tuple=True)
    q = dict(x=f["x"][r, s_], q=f["q"][r, s_], bur=f["bur"][r, s_], aa=f["aa"][r], slot=s_, pos=f["pos"][r], res=r)
    rc = torch.nonzero((f["aa"] == C_IDX) & f["m"][:, 5]).squeeze(1)
    sg = dict(x=f["x"][rc, 5], cb=f["x"][rc, 4], res=rc)
    del E_
    return Env(f=pad, ca=ca, reach=dist.max(-1).values, res_index=idx, q=q, sg=sg)


def env_neighbors(env: Env, ca: torch.Tensor, reach: torch.Tensor | float, p: EnergyParams,
                  env_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Indices [..., K] of env blocks whose atoms can come within the short-range cutoff of a residue at `ca`
    [..., 3] with extent `reach` (padded with the sentinel index E)."""
    cut = max(p.lj_off, p.solv_off)
    d = (ca[..., None, :] - env.ca[:-1]).norm(dim=-1)                     # [..., E]
    ok = d < cut + env.reach[:-1] + (reach[..., None] if torch.is_tensor(reach) else reach)
    if env_mask is not None:
        ok = ok & env_mask
    K = max(int(ok.sum(-1).max().item()) if ok.numel() else 0, 1)
    order = torch.argsort((~ok).to(torch.int8), dim=-1, stable=True)[..., :K]
    return torch.where(torch.gather(ok, -1, order), order, torch.full_like(order, env.n))


def env_energy(mov: Blocks, env: Env, p: EnergyParams | None = None, env_mask: torch.Tensor | None = None,
               f: dict | None = None) -> torch.Tensor:
    """Energy of residues `mov` [...] (e.g. [B] variants/rotamers of a site) with a fixed environment
    -> [..., n_terms]. env_mask: [E] or [..., E] bool, env residues to keep (per batch element).

    Elements are grouped by site (same `pos`); per site the env blocks within cutoff + extents are gathered
    once and broadcast over all variants of that site; electrostatics uses the env's flat charged-atom list."""
    p = p or EnergyParams()
    f = f if f is not None else features(mov, p)
    lead = mov.shape
    fl = {k: v.reshape(-1, *v.shape[len(lead):]) for k, v in f.items()}
    n = fl["aa"].shape[0]
    nt = len(term_names(p))
    out = torch.zeros(n, nt, device=mov.device, dtype=fl["x"].dtype)
    if n == 0:
        return out.reshape(*lead, nt)
    em = None
    if env_mask is not None:
        em = env_mask if env_mask.dim() == 1 else env_mask.reshape(-1, env.n)
    xd, md = fl["x"].detach(), fl["m"]
    with torch.no_grad():
        _, inv = torch.unique(fl["pos"], return_inverse=True)      # group = site (chain-aware position)
    for g in range(int(inv.max()) + 1):
        idx = torch.nonzero(inv == g).squeeze(1)
        ca_g = xd[idx[0], 1]
        reach_g = torch.where(md[idx][..., None], xd[idx] - ca_g, torch.zeros_like(xd[idx])).norm(dim=-1).max()
        em_g = None if em is None else (em if em.dim() == 1 else em[idx].any(0))
        nb = env_neighbors(env, ca_g, reach_g, p, em_g)                                    # [K]
        fe = _gather_f(env.f, nb[None])                                                    # [1, K, ...]
        dq = (env.q["x"] - ca_g).norm(dim=-1) < p.elec_off + reach_g
        qsel = torch.nonzero(dq & (True if em_g is None else em_g[env.q["res"]])).squeeze(1)
        qe = {k: v[qsel] for k, v in env.q.items()}
        ssel = torch.nonzero((env.sg["x"] - ca_g).norm(dim=-1) < 3.0 + reach_g).squeeze(1)
        se = {k: v[ssel] for k, v in env.sg.items()}
        width = (md[idx] * torch.arange(1, A + 1, device=md.device)).max(-1).values
        idx = idx[torch.argsort(width, stable=True)]                    # similar sizes share a chunk (trimming)
        for sl in _chunks(idx.shape[0], nb.numel() * A * A, p):
            ii = idx[sl]
            fm = {k: v[ii][:, None] for k, v in fl.items()}                               # [b, 1, ...]
            fe_b, qm_b, sm_b = fe, None, None
            if em is not None and em.dim() > 1:
                emb = em[ii]
                fe_b = dict(fe, m=fe["m"] & torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], 1)[:, nb][..., None])
                qm_b, sm_b = emb[:, qe["res"]], emb[:, se["res"]]
            out[ii] = _env_terms(fm, fe_b, qe, qm_b, se, sm_b, p)
    return out.reshape(*lead, nt)


def _env_terms(fm: dict, fe: dict, qe: dict, qmask, se: dict, smask, p: EnergyParams) -> torch.Tensor:
    """fm [b, 1, ...] moving residues; fe [1|b, K, ...] env blocks; qe flat env charges [Nc]; se env SG [Ns];
    qmask [b, Nc] / smask [b, Ns] optional per-element env masks. Returns [b, n_terms]."""
    fms, fes = _trim(fm, _width(fm["m"])), _trim(fe, _width(fe["m"]))                    # drop empty slot tails
    sep = _pair_sep(fms, fes)
    nonadj = (fe["pos"] - fm["pos"]).abs() >= 2
    sr = _core(fms, fes, sep, nonadj, p).sum(1)                                            # [b, 7]
    b = sr.shape[0]
    # electrostatics: compress the moving residue's charged slots, vs the env's flat charged atoms
    qm = fm["q"][:, 0]                                                                     # [b, A]
    nq = int((qm != 0).sum(-1).max().item())
    if nq > 0 and qe["q"].numel() > 0:
        o = torch.argsort((qm == 0).to(torch.int8), dim=-1, stable=True)[:, :nq]           # [b, nq]
        xm = torch.gather(fm["x"][:, 0], 1, o[..., None].expand(-1, -1, 3))
        q_ = torch.gather(qm, 1, o)
        bm = torch.gather(fm["bur"][:, 0], 1, o)
        T = tables(qm.device)
        aa_m, pos_m = fm["aa"][:, 0], fm["pos"][:, 0]
        dpos = qe["pos"][None, :] - pos_m[:, None]                                          # [b, Nc]
        sf = T["sep_pep"][aa_m[:, None, None], qe["aa"][None, None, :], o[:, :, None], qe["slot"][None, None, :]]
        sb = T["sep_pep"][qe["aa"][None, None, :], aa_m[:, None, None], qe["slot"][None, None, :], o[:, :, None]]
        sep_q = torch.where((dpos == 1)[:, None], sf, torch.where((dpos == -1)[:, None], sb, torch.full_like(sf, 4)))
        valid = (q_ != 0)[:, :, None] & (sep_q >= 4)
        if qmask is not None:
            valid = valid & qmask[:, None, :]
        el = _elec(xm, q_, bm, qe["x"][None], qe["q"][None], qe["bur"][None], valid, p)   # [b, n_modes]
    else:
        el = sr.new_zeros(b, len(p.elec_modes))
    # disulfide
    if se["x"].shape[0] > 0:
        ok = ((fm["aa"][:, 0] == C_IDX) & fm["m"][:, 0, 5])[:, None].expand(-1, se["x"].shape[0])
        if smask is not None:
            ok = ok & smask
        ss = _disulf(fm["x"][:, 0, 5][:, None], fm["x"][:, 0, 4][:, None], se["x"][None], se["cb"][None], ok, p).sum(1)
    else:
        ss = sr.new_zeros(b)
    return torch.cat([sr, el, ss[:, None]], -1)


def local_energy(mov: Blocks, env: Env, p: EnergyParams | None = None, env_mask: torch.Tensor | None = None,
                 include_ref: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Local system of R flexible residues per batch element: mov [B, R].

    Returns (one_body [B, R, n_terms], two_body [B, R, R, n_terms]):
      one_body[b, r] = intra(r) + E(r, fixed environment)   (rotamer vs fixed env; for packers)
      two_body[b, r, s] = E(r, s), symmetric, zero diagonal  (rotamer-rotamer)
    Total local energy = one_body.sum(1) + two_body.sum((1, 2)) / 2."""
    p = p or EnergyParams()
    f = features(mov, p)
    one = intra_energy(mov, p, include_ref, f) + env_energy(mov, env, p, env_mask, f)
    B, R = mov.shape
    nt = one.shape[-1]
    two = one.new_zeros(B, R, R, nt)
    if R > 1:
        iu = torch.triu_indices(R, R, 1, device=mov.device)
        fa = {k: v[:, iu[0]] for k, v in f.items()}
        fb = {k: v[:, iu[1]] for k, v in f.items()}
        e = _block_pair_all(fa, fb, p)                                            # [B, npairs, nt]
        two = two.index_put((torch.arange(B, device=mov.device)[:, None], iu[0][None], iu[1][None]), e)
        two = two + two.transpose(1, 2)
    return one, two


@dataclass
class Decomposition:
    one_body: torch.Tensor    # [L, n_terms] intra-residue (+ dG_ref)
    pair: torch.Tensor        # [L, L, n_terms] symmetric residue-pair energies, zero diagonal
    names: tuple

    @property
    def per_residue(self) -> torch.Tensor:
        """[L, n_terms]: one-body + half of all pair energies (sums to the total)."""
        return self.one_body + 0.5 * self.pair.sum(1)

    @property
    def total(self) -> torch.Tensor:
        return self.one_body.sum(0) + 0.5 * self.pair.sum((0, 1))


def residue_pairs(blocks: Blocks, p: EnergyParams, elec: bool = False) -> torch.Tensor:
    """Candidate residue pairs i<j [Q, 2] whose atoms can come within the term cutoff."""
    f_m = blocks.mask
    ca = blocks.xyz[:, 1]
    reach = torch.where(f_m[..., None], blocks.xyz - ca[:, None], torch.zeros_like(blocks.xyz)).norm(dim=-1).max(-1).values
    cut = p.elec_off if elec else max(p.lj_off, p.solv_off)
    d = torch.cdist(ca, ca)
    ok = torch.triu(d < cut + reach[:, None] + reach[None, :], 1)
    return torch.nonzero(ok)


def decompose(blocks: Blocks, p: EnergyParams | None = None, include_ref: bool = True) -> Decomposition:
    """Whole-structure per-residue decomposition for Blocks [L]."""
    p = p or EnergyParams()
    f = features(blocks, p)
    L = blocks.shape[0]
    names = term_names(p)
    nt = len(names)
    one = intra_energy(blocks, p, include_ref, f)
    pair = one.new_zeros(L, L, nt)
    with torch.no_grad():
        pr = residue_pairs(blocks, p, elec=True)            # superset (elec cutoff >= short-range cutoff)
    chunks = []
    for sl in _chunks(pr.shape[0], A * A, p):
        i, j = pr[sl, 0], pr[sl, 1]
        chunks.append(_block_pair_all(_gather_f(f, i), _gather_f(f, j), p))
    if chunks:
        e = torch.cat(chunks, 0)
        pair = pair.index_put((pr[:, 0], pr[:, 1]), e)
        pair = pair + pair.transpose(0, 1)
    return Decomposition(one_body=one, pair=pair, names=names)


def total_energy(blocks: Blocks, p: EnergyParams | None = None) -> torch.Tensor:
    return decompose(blocks, p).total


def atom_solvation(blocks: Blocks, p: EnergyParams | None = None) -> torch.Tensor:
    """EEF1 solvation free energy of every atom slot [L, A] (kcal/mol): dG_ref_i - sum_j f_i(r_ij) V_j,
    with the same exclusions as the energy terms (0 for absent slots). Useful as a burial descriptor."""
    p = p or EnergyParams()
    f = features(blocks, p)
    T = tables(blocks.device)
    L = blocks.shape[0]
    out = f["dg_ref"] * f["m"].float()

    def acc(fa, fb, sep):
        d = fb["x"][..., None, :, :] - fa["x"][..., :, None, :]
        r2 = (d * d).sum(-1)
        near = fa["m"][..., :, None] & fb["m"][..., None, :] & (sep >= 3) & (r2 < p.solv_off ** 2)
        r2 = torch.where(near, r2, torch.full_like(r2, BIG2))
        return _eef1(fa, fb, r2, near, p)

    sep_i = T["sep_intra"][f["aa"]]
    ga, _ = acc(f, f, sep_i)                       # full (not triangular): each atom's own losses
    out = out + ga.sum(-1)
    with torch.no_grad():
        pr = residue_pairs(blocks, p)
    for sl in _chunks(pr.shape[0], A * A, p):
        i, j = pr[sl, 0], pr[sl, 1]
        fa, fb = _gather_f(f, i), _gather_f(f, j)
        ga, gb = acc(fa, fb, _pair_sep(fa, fb))
        out = out.index_add(0, i, ga.sum(-1)).index_add(0, j, gb.sum(-2))
    return out


# ============================================================================= single-residue statistical terms
KT = 0.5925   # kcal/mol at 298.15 K (statistical potentials in nats are converted with this)


class StatTerms:
    """Single-residue statistical terms from E1's PDB statistics (physddg/data/pdbstats.npz, rotlib.PDBStats),
    as differentiable torch lookups, in kcal/mol (nats x KT).

    rama(aa, phi, psi, next_is_pro=None)  E = -kT ln[P(phi,psi|aa)/P(phi,psi)], periodic bilinear on the 10° grid
                                         (cell centres -175..175); pre-Pro map for non-Gly/Pro if next_is_pro
    env(aa, ss3, cn)                     E_env = -kT ln[P(aa|cn bin, SS)/P(aa)] (D4), ss3 in {0:H,1:E,2:L}
    sc_entropy(aa, ss="all")             T*S_conf of the side chain (rotameric entropy) [kcal/mol]
    Identical numbers to rotlib.PDBStats (times KT), see tests/test_energy.py::test_stat_terms."""

    def __init__(self, path=None, device="cpu", kT: float = KT, sc_kind: str = "sc_entropy_rotameric"):
        from .rotlib import PDBStats
        st = PDBStats(path) if path is not None else PDBStats()
        d = st.d
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.kT = kT
        names = list(st.rama_names)
        self.rama_grid = f(d["rama_energy"]) * kT                                     # [24, 36, 36]
        self.rama_row = torch.tensor([names.index(a) for a in topo.AA1], device=device)
        self.prepro_row = names.index("prepro")
        self.env_tab = f(d["env_cn_ss_energy"]) * kT                                  # [n_cn, 3, 20]
        self.cn_edges = f(d["cn_edges"])
        self.sc_tab = f(d[sc_kind]) * kT                                              # [20, 4]
        self.ss_names = [str(x) for x in d["entropy_classes"]]

    @classmethod
    def from_library(cls, device="cpu"):
        return cls(device=device)

    def rama(self, aa: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor, next_is_pro=None) -> torch.Tensor:
        row = self.rama_row[aa]
        if next_is_pro is not None:
            gp = (aa == topo.AA_INDEX["G"]) | (aa == topo.AA_INDEX["P"])
            row = torch.where(torch.as_tensor(next_is_pro, device=row.device) & ~gp,
                              torch.full_like(row, self.prepro_row), row)
        g = self.rama_grid
        n = g.shape[-1]
        x = (phi + 175.0) / 10.0
        y = (psi + 175.0) / 10.0
        x0, y0 = torch.floor(x), torch.floor(y)
        tx, ty = x - x0, y - y0
        i0, j0 = x0.long() % n, y0.long() % n
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        v = lambda i, j: g[row, i, j]
        return (v(i0, j0) * (1 - tx) * (1 - ty) + v(i1, j0) * tx * (1 - ty) + v(i0, j1) * (1 - tx) * ty
                + v(i1, j1) * tx * ty)

    def env(self, aa: torch.Tensor, ss3: torch.Tensor, cn: torch.Tensor) -> torch.Tensor:
        b = (torch.searchsorted(self.cn_edges, cn.contiguous(), right=True) - 1).clamp(0, self.env_tab.shape[0] - 1)
        return self.env_tab[b, ss3, aa]

    def sc_entropy(self, aa: torch.Tensor, ss: str = "all") -> torch.Tensor:
        return self.sc_tab[aa, self.ss_names.index(ss)]
