"""Tests for physddg.energy / physddg.params (engineer E2). Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_energy.py"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from physddg import energy as E
from physddg import topology as topo
from physddg.params import CHARMM19, TYPE_MAP, EnergyParams, type_table
from physddg.structure import load_protein

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PDB = Path("/workspace/data/fireprot_pdbs")
P = EnergyParams()
NAMES = E.term_names(P)
T_IDX = {n: i for i, n in enumerate(NAMES)}


@pytest.fixture(scope="module")
def small():
    prot = load_protein(PDB / "1BRF.pdb")
    return prot, E.blocks_from_protein(prot, DEV, P)


@pytest.fixture(scope="module")
def ss_prot():
    prot = load_protein(PDB / "1DEC.pdb")      # 3 disulfides
    return prot, E.blocks_from_protein(prot, DEV, P)


def to64(b):
    return E.Blocks(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.__dict__.items()})


# ----------------------------------------------------------------------------- parameters / tables
def test_param_table_complete():
    t = type_table()
    assert t.shape[0] == len(topo.ATOM_TYPES) and np.isfinite(t).all()
    assert set(TYPE_MAP.values()) and all(c in CHARMM19 for c, _ in TYPE_MAP.values())
    lam = dict(zip(topo.ATOM_TYPES, t[:, 5]))
    assert lam["N_lys"] == 6.0 and lam["O_carboxyl"] == 6.0 and lam["CH3"] == 3.5


def test_bonds_consistent_with_e1_internal_coordinates():
    for aa, spec in topo.IC_SPEC.items():
        bonds = {frozenset(b) for b in E.residue_bonds(aa)}
        for atom, _, _, c, _ in spec:
            assert frozenset((atom, c)) in bonds, (aa, atom, c)


def test_sep_tables():
    T = E.build_tables()
    s = {nm: i for i, nm in enumerate(E.slot_names("F")) if nm}
    f = topo.AA_INDEX["F"]
    assert T["sep_intra"][f, s["CD1"], s["CE2"]] == 0          # rigid ring pair excluded
    assert T["sep_intra"][f, s["N"], s["CG"]] == 3             # chi1-dependent 1-4 pair kept
    assert T["sep_intra"][f, s["O"], s["CD1"]] == 4
    l = topo.AA_INDEX["L"]
    assert T["sep_pep"][l, l, 2, 0] == 1 and T["sep_pep"][l, l, 3, 1] == 3   # C(i)-N(i+1), O(i)-CA(i+1)
    p_ = topo.AA_INDEX["P"]
    assert (T["sep_intra"][p_] == 0).all()                     # proline ring is rigid here


# ----------------------------------------------------------------------------- invariances / determinism
def test_rigid_motion_invariance(small):
    _, b = small
    tot = E.decompose(to64(b), P).total
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, device=DEV))[0]
    b2 = to64(b)
    b2.xyz = (b2.xyz @ R.T + torch.tensor([5.0, -3.0, 11.0], device=DEV, dtype=torch.float64)) * b2.mask[..., None]
    b2.prev_c = b2.prev_c @ R.T + torch.tensor([5.0, -3.0, 11.0], device=DEV, dtype=torch.float64)
    tot2 = E.decompose(b2, P).total
    assert torch.allclose(tot, tot2, atol=1e-6)


def test_deterministic(small):
    _, b = small
    a, c = E.decompose(b, P), E.decompose(b, P)
    assert torch.equal(a.pair, c.pair) and torch.equal(a.one_body, c.one_body)


def test_decomposition_sums(small):
    _, b = small
    d = E.decompose(b, P)
    assert torch.allclose(d.per_residue.sum(0), d.total, atol=1e-2)
    assert torch.allclose(d.pair, d.pair.transpose(0, 1))
    assert torch.isfinite(d.pair).all() and torch.isfinite(d.one_body).all()


# ----------------------------------------------------------------------------- local API == decomposition
def test_local_matches_decomposition(small):
    prot, b = small
    d = E.decompose(b, P)
    for i in range(0, len(prot), 6):
        env = E.make_env(b, [i], P)
        one, two = E.local_energy(b[torch.tensor([[i]], device=DEV)], env, P)
        assert torch.allclose(one[0, 0], d.one_body[i] + d.pair[i].sum(0), atol=1e-4)
        assert (two == 0).all()
    i, j = 10, 30
    env = E.make_env(b, [i, j], P)
    one, two = E.local_energy(b[torch.tensor([[i, j]], device=DEV)], env, P)
    assert torch.allclose(two[0, 0, 1], d.pair[i, j], atol=1e-5)
    assert torch.allclose(one[0, 0], d.one_body[i] + d.pair[i].sum(0) - d.pair[i, j], atol=1e-4)


def test_env_mask_multisite(small):
    prot, b = small
    d = E.decompose(b, P)
    L = len(prot)
    sites = torch.arange(0, L, 2, device=DEV)
    env = E.make_env(b, None, P)
    em = torch.ones(len(sites), L, dtype=torch.bool, device=DEV)
    em[torch.arange(len(sites)), sites] = False
    e = E.env_energy(b[sites], env, P, env_mask=em) + E.intra_energy(b[sites], P)
    assert torch.allclose(e, d.one_body[sites] + d.pair[sites].sum(1), atol=1e-4)


def test_pair_energy_broadcast(small):
    _, b = small
    a = b[torch.tensor([3, 4, 5], device=DEV)][:, None]
    c = b[torch.tensor([20, 21], device=DEV)][None, :]
    assert E.pair_energy(a, c, P).shape == (3, 2, len(NAMES))


def test_variant_batch_same_as_individual(small):
    """Batched variants (rotations of the side chain) equal one-at-a-time evaluation."""
    prot, b = small
    i = next(k for k, a in enumerate(prot.seq) if a in "FYWL" and 3 < k < len(prot) - 3)
    env = E.make_env(b, [i], P)
    mov = b[torch.full((6,), i, device=DEV)]
    sc = torch.zeros_like(mov.mask)
    sc[:, 5:14] = True
    torch.manual_seed(1)
    mov.xyz = mov.xyz + 0.4 * torch.randn_like(mov.xyz) * (sc & mov.mask)[..., None]
    batched = E.env_energy(mov, env, P)
    single = torch.stack([E.env_energy(mov[k:k + 1], env, P)[0] for k in range(6)])
    assert torch.allclose(batched, single, atol=1e-4)


# ----------------------------------------------------------------------------- functional forms
def _two_atom(r, t1="CH3", t2="CH3", cap=None, sep=4):
    """Short-range terms between two isolated atoms of given types at distance r (as slots of two Ala blocks)."""
    p = EnergyParams() if cap is None else EnergyParams(rep_cap=cap)
    x = torch.full((2, E.A, 3), E.FAR, dtype=torch.float64)
    x[0, 4] = torch.tensor([0.0, 0, 0]); x[1, 4] = torch.tensor([r, 0, 0], dtype=torch.float64)
    f = {"x": x, "m": torch.zeros(2, E.A, dtype=torch.bool)}
    f["m"][:, 4] = True
    tid = [topo.TYPE_INDEX[t1], topo.TYPE_INDEX[t2]]
    tab = torch.as_tensor(type_table(), dtype=torch.float64)
    f["t"] = torch.zeros(2, E.A, dtype=torch.long); f["t"][0, 4], f["t"][1, 4] = tid[0], tid[1]
    for k, c in enumerate(E.COLS):
        f[c] = tab[f["t"], k]
    for k in ("don", "acc", "dcone", "bb"):
        f[k] = torch.zeros(2, E.A, dtype=torch.bool)
    f["dvec"] = f["avec"] = torch.zeros(2, E.A, 3, dtype=torch.float64)
    f["cap"] = torch.full((2, E.A), p.rep_cap, dtype=torch.float64)
    fa = {k: v[0] for k, v in f.items()}; fb = {k: v[1] for k, v in f.items()}
    sepm = torch.full((E.A, E.A), sep, dtype=torch.int8)
    return E._core(fa, fb, sepm, torch.tensor(True), p)


def test_lj_split_and_cap():
    t = E.term_names(P)
    rmin = 2 * 2.165
    eps = 0.1811
    atr_at_min = _two_atom(rmin)[t.index("lj_atr")].item()
    assert atr_at_min == pytest.approx(-eps, rel=1e-6)
    assert _two_atom(rmin)[t.index("lj_rep")].item() == pytest.approx(0.0, abs=1e-9)
    assert _two_atom(rmin + 0.5)[t.index("lj_rep")].item() == 0.0
    assert _two_atom(8.5)[t.index("lj_atr")].item() == 0.0                       # switched off
    # cap: value at r_c equals cap, linear (constant slope) inside, continuous across r_c
    cap = 3.0
    uc = 1 + math.sqrt(cap / eps)
    rc = rmin * uc ** (-1 / 6)
    e = lambda r: _two_atom(r, cap=cap)[1].item()
    assert e(rc) == pytest.approx(cap, rel=1e-6)
    s1, s2 = (e(rc - 0.5) - e(rc - 0.4)) / 0.1, (e(rc - 1.0) - e(rc - 0.9)) / 0.1
    assert s1 == pytest.approx(s2, rel=1e-6)
    assert (e(rc - 1e-6) - e(rc + 1e-6)) / 2e-6 == pytest.approx((e(rc + 2e-6) - e(rc + 4e-6)) / 2e-6, rel=1e-3)
    assert e(1.0) < 40 and e(1.0) > cap


def test_site_dependent_cap(small):
    prot, b = small
    i = 12
    env = E.make_env(b, [i], P)
    mov = b[torch.full((2,), i, device=DEV)]
    mov.xyz = mov.xyz.clone()
    mov.xyz[:, 4:14] = mov.xyz[:, 4:14] + torch.tensor([0.0, 0.0, 1.5], device=DEV) * mov.mask[:, 4:14, None]
    mov.cap = torch.tensor([1.0, 10.0], device=DEV)
    e = E.env_energy(mov, env, P)
    rep = T_IDX["lj_rep"]
    assert e[0, rep] < e[1, rep]
    assert torch.equal(e[0, :rep], e[1, :rep])


def test_eef1_isolated_atom_reference():
    # isolated atoms: no solvation change, only the reference enters (via intra_energy include_ref)
    out = _two_atom(20.0, "N_lys", "O_carboxyl")
    assert out[2].item() == 0 and out[3].item() == 0
    # close polar pair: desolvation (positive change for polar atoms with negative dG_free)
    assert _two_atom(3.5, "O_hydroxyl", "O_hydroxyl")[3].item() > 0
    # apolar CH3 near CH3: dG_free > 0 -> solvation change negative (burial favourable)
    assert _two_atom(4.0, "CH3", "CH3")[2].item() < 0


def test_atom_solvation_matches_terms(small):
    _, b = small
    d = E.decompose(b, P)
    s = E.atom_solvation(b, P)
    pol = E.features(b, P)["polar"]
    assert (s * (1 - pol)).sum().item() == pytest.approx(d.total[T_IDX["solv_apolar"]].item(), abs=1e-2)
    assert (s * pol).sum().item() == pytest.approx(d.total[T_IDX["solv_polar"]].item(), abs=1e-2)


def test_proline_no_backbone_donor(small):
    prot, b = small
    f = E.features(b, P)
    pro = torch.tensor([a == "P" for a in prot.seq], device=DEV)
    assert not f["don"][pro, 0].any()
    assert f["don"][~pro, 0].sum() > 0


def test_backbone_hbonds_present(small):
    _, b = small
    d = E.decompose(b, P)
    assert d.total[T_IDX["hb_bb_bb"]] < -20


def test_disulfide(ss_prot):
    prot, b = ss_prot
    d = E.decompose(b, P)
    k = T_IDX["disulf"]
    assert d.total[k] < -1.5                         # three native disulfides, each close to -1
    i, j = 6, 14                                     # bonded SG pair: no LJ clash
    assert d.pair[i, j, T_IDX["lj_rep"]] < 1.0


def test_electrostatics_sign_and_switch():
    p = EnergyParams()
    x = torch.tensor([[0.0, 0, 0]], dtype=torch.float64)
    for r, sign in ((5.0, -1), (13.0, 0)):
        y = torch.tensor([[r, 0, 0]], dtype=torch.float64)
        e = E._elec(x, torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64),
                    y, torch.tensor([-1.0], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.bool), p)
        assert (np.sign(e.numpy()) == sign).all()
    assert abs(e[0].item()) == 0                     # beyond the 12 Å switch: exactly zero


# ----------------------------------------------------------------------------- gradients
def _fd_check(b, terms, n=25, seed=0):
    b = to64(b)
    x0 = b.xyz.detach().clone().requires_grad_(True)

    def energy(x):
        return E.decompose(E.Blocks(**{**b.__dict__, "xyz": x}), P).total

    tot = energy(x0)
    g = torch.autograd.grad(tot[[T_IDX[t] for t in terms]].sum(), x0)[0]
    pres = torch.nonzero(b.mask)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sel = pres[torch.randperm(len(pres), generator=gen)[:n].to(pres.device)]
    h = 1e-5
    for i, a in sel.tolist():
        for c in range(3):
            xp, xm = x0.detach().clone(), x0.detach().clone()
            xp[i, a, c] += h; xm[i, a, c] -= h
            with torch.no_grad():
                num = (energy(xp) - energy(xm))[[T_IDX[t] for t in terms]].sum() / (2 * h)
            assert num.item() == pytest.approx(g[i, a, c].item(), abs=1e-5, rel=1e-4)


def test_gradients_all_terms(ss_prot):
    _, b = ss_prot
    _fd_check(b, list(NAMES), n=40)


def test_gradients_disulfide_atoms(ss_prot):
    prot, b = ss_prot
    b64 = to64(b)
    x0 = b64.xyz.detach().clone().requires_grad_(True)
    tot = E.decompose(E.Blocks(**{**b64.__dict__, "xyz": x0}), P).total[T_IDX["disulf"]]
    g = torch.autograd.grad(tot, x0)[0]
    i = 6
    h = 1e-5
    for a in (4, 5):
        for c in range(3):
            xp, xm = x0.detach().clone(), x0.detach().clone()
            xp[i, a, c] += h; xm[i, a, c] -= h
            num = (E.decompose(E.Blocks(**{**b64.__dict__, "xyz": xp}), P).total[T_IDX["disulf"]]
                   - E.decompose(E.Blocks(**{**b64.__dict__, "xyz": xm}), P).total[T_IDX["disulf"]]) / (2 * h)
            assert num.item() == pytest.approx(g[i, a, c].item(), abs=1e-6, rel=1e-4)


def test_gradients_sidechain_hbonds():
    """FD check on the atoms of the residue pair with the strongest sc-sc and sc-bb H-bond."""
    prot = load_protein(PDB / "1BNL.pdb")
    b = to64(E.blocks_from_protein(prot, DEV, P))
    d = E.decompose(b, P)
    x0 = b.xyz.detach().clone().requires_grad_(True)
    energy = lambda x: E.decompose(E.Blocks(**{**b.__dict__, "xyz": x}), P).total
    for term in ("hb_sc_sc", "hb_sc_bb"):
        k = T_IDX[term]
        i, j = np.unravel_index(int(torch.argmin(d.pair[..., k]).item()), d.pair.shape[:2])
        assert d.pair[i, j, k] < -0.3
        g = torch.autograd.grad(energy(x0)[k], x0)[0]
        h = 1e-5
        for r in (i, j):
            for a in torch.nonzero(b.mask[r]).squeeze(1).tolist()[4:]:
                for c in range(3):
                    xp, xm = x0.detach().clone(), x0.detach().clone()
                    xp[r, a, c] += h; xm[r, a, c] -= h
                    with torch.no_grad():
                        num = (energy(xp)[k] - energy(xm)[k]) / (2 * h)
                    assert num.item() == pytest.approx(g[r, a, c].item(), abs=1e-5, rel=1e-4)
