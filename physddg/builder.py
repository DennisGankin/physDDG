"""GPU-vectorised NeRF side-chain builder (torch used as an array library only).

build_sidechains(bb, aa, chis) builds all side-chain heavy atoms for any mix of amino acids in one call:
  bb   [B,3,3] or [B,4,3]  backbone N, CA, C (, O) per position
  aa   int (same aa everywhere) or LongTensor [B] / [B,R] (index into topology.AA1)
  chis [B,R,4] degrees (unused chis ignored)
returns xyz [B,R,14,3] in topology template (atom14) order with backbone copied in (O = NaN if not given),
and mask [B,R,14] of atoms that exist for the residue type. Geometry: topology.IDEAL_IC (per-aa CB included).

Also: ideal_cb (Gly -> X), amide_h (backbone H, bisector rule), donor_h (polar H with determined geometry).
"""
from __future__ import annotations

import numpy as np
import torch

from . import topology as T

N_STEPS = T.MAX_SC_ATOMS          # 10 build steps (Trp)
DUMMY = T.MAX_ATOMS               # scratch slot index 14


def _tables():
    tgt = np.full((20, N_STEPS), DUMMY, np.int64)
    par = np.zeros((20, N_STEPS, 3), np.int64)
    geo = np.zeros((20, N_STEPS, 3), np.float64)       # bond, angle, torsion offset (deg)
    chi = np.zeros((20, N_STEPS), np.int64)            # 0 = fixed torsion, k = chi_k
    mask = np.zeros((20, T.MAX_ATOMS), bool)
    for ia, a in enumerate(T.AA1):
        t = T.TEMPLATES[a]
        mask[ia, :len(t.atoms)] = True
        spec = ([T.CB_SPEC] if a != "G" else []) + T.IC_SPEC[a]
        for s, (atom, A, B, C, k) in enumerate(spec):
            tgt[ia, s] = t.index(atom)
            par[ia, s] = [t.index(A), t.index(B), t.index(C)]
            geo[ia, s] = T.IDEAL_IC[a][atom]
            chi[ia, s] = k
    geo[..., 1:] = np.radians(geo[..., 1:])
    return tgt, par, geo, chi, mask


_TGT, _PAR, _GEO, _CHI, _MASK = _tables()
_CACHE: dict = {}


def _dev_tables(device, dtype):
    key = (str(device), dtype)
    if key not in _CACHE:
        _CACHE[key] = (torch.as_tensor(_TGT, device=device), torch.as_tensor(_PAR, device=device),
                       torch.as_tensor(_GEO, device=device, dtype=dtype), torch.as_tensor(_CHI, device=device),
                       torch.as_tensor(_MASK, device=device))
    return _CACHE[key]


def nerf(a, b, c, bond, angle, torsion):
    """Place D from A,B,C ([...,3]) with |CD|=bond, angle BCD, dihedral ABCD (radians, broadcastable)."""
    bc = torch.nn.functional.normalize(c - b, dim=-1)
    n = torch.nn.functional.normalize(torch.cross(b - a, bc, dim=-1), dim=-1)
    m = torch.cross(n, bc, dim=-1)
    st = torch.sin(angle)
    d0, d1, d2 = -bond * torch.cos(angle), bond * st * torch.cos(torsion), bond * st * torch.sin(torsion)
    return c + d0[..., None] * bc + d1[..., None] * m + d2[..., None] * n


def build_sidechains(bb: torch.Tensor, aa, chis: torch.Tensor):
    B, R = chis.shape[:2]
    dev, dt = chis.device, chis.dtype
    tgt, par, geo, chi_i, amask = _dev_tables(dev, dt)
    if isinstance(aa, int):
        aa_t = torch.full((B, R), aa, dtype=torch.long, device=dev)
    else:
        aa_t = torch.as_tensor(aa, device=dev).long()
        aa_t = aa_t[:, None].expand(B, R) if aa_t.dim() == 1 else aa_t
    X = torch.zeros(B, R, T.MAX_ATOMS + 1, 3, device=dev, dtype=dt)
    X[:, :, :3] = bb[:, None, :3].to(dt)
    X[:, :, 3] = bb[:, None, 3].to(dt) if bb.shape[1] > 3 else float("nan")
    chis_r = torch.deg2rad(torch.cat([torch.zeros_like(chis[..., :1]), chis], -1))   # index 0 = no chi
    for s in range(N_STEPS):
        tg = tgt[aa_t, s]                                  # [B,R]
        if bool((tg == DUMMY).all()):
            break
        p = par[aa_t, s]                                   # [B,R,3]
        g = geo[aa_t, s]                                   # [B,R,3]
        tor = g[..., 2] + torch.gather(chis_r, -1, chi_i[aa_t, s][..., None])[..., 0]
        pa = torch.gather(X, 2, p[..., None].expand(B, R, 3, 3))
        d = nerf(pa[:, :, 0], pa[:, :, 1], pa[:, :, 2], g[..., 0], g[..., 1], tor)
        X = X.scatter(2, tg[..., None, None].expand(B, R, 1, 3), d[:, :, None])
    xyz = X[:, :, :T.MAX_ATOMS]
    mask = amask[aa_t]
    return xyz.masked_fill(~mask[..., None], float("nan")), mask


def ideal_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor, aa: int | None = None) -> torch.Tensor:
    """Ideal CB from backbone atoms [...,3] (generic CB_IC, or the given aa's CB geometry)."""
    bond, ang, tor = T.IDEAL_IC[T.AA1[aa]]["CB"] if aa is not None else T.CB_IC
    f = lambda v: torch.tensor(v, dtype=n.dtype, device=n.device)
    return nerf(c, n, ca, f(bond), f(np.radians(ang)), f(np.radians(tor)))


def amide_h(c_prev: torch.Tensor, n: torch.Tensor, ca: torch.Tensor, d: float = 1.01) -> torch.Tensor:
    """Backbone amide H on the external bisector of C(i-1)-N and CA-N."""
    u = torch.nn.functional.normalize(n - c_prev, dim=-1) + torch.nn.functional.normalize(n - ca, dim=-1)
    return n + d * torch.nn.functional.normalize(u, dim=-1)


def donor_h(D: torch.Tensor, B1: torch.Tensor, B2: torch.Tensor, geom: str, d: float = 1.01) -> torch.Tensor:
    """Polar H for donors whose H is fixed by heavy atoms (see topology.HB.don_geom).
    bisector -> [...,1,3]; sp2_2H -> [...,2,3] (H cis and trans to B2 about the B1-D bond)."""
    if geom == "bisector":
        return amide_h(B1, D, B2, d)[..., None, :]
    if geom == "sp2_2H":
        ang = torch.tensor(np.radians(120.0), dtype=D.dtype, device=D.device)
        h = [nerf(B2, B1, D, torch.tensor(d, dtype=D.dtype, device=D.device), ang,
                  torch.tensor(t, dtype=D.dtype, device=D.device)) for t in (0.0, np.pi)]
        return torch.stack(h, -2)
    raise ValueError(f"H position not determined for geometry {geom!r}")
