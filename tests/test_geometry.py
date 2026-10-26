"""Tests for the geometry engine (topology, structure, builder, rotamer library, PDB statistics)."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch

from physddg import topology as T
from physddg.builder import amide_h, build_sidechains, donor_h, ideal_cb as ideal_cb_t
from physddg.structure import (annotate, backbone_h, dihedral, ideal_cb, load_protein, side_chain_chis,
                               smooth_cn)

UBQ = Path("/rsyncPDB/data/structures/divided/mmCIF/ub/1ubq.cif.gz")
needs_pdb = pytest.mark.skipif(not UBQ.exists(), reason="local PDB mirror not available")


@pytest.fixture(scope="module")
def ubq():
    return annotate(load_protein(UBQ))


# ----------------------------------------------------------------------------- topology
def test_vocabulary_and_templates():
    assert T.ATOM_TYPES == ["N_bb", "N_pro", "CA", "C_bb", "O_bb", "CH1", "CH2", "CH3", "C_aro", "C_sp2",
                            "N_amide", "N_his", "N_trp", "N_lys", "N_arg", "O_amide", "O_carboxyl",
                            "O_hydroxyl", "S_cys", "S_met"]
    assert T.MAX_ATOMS == 14
    for a, t in T.TEMPLATES.items():
        assert t.atoms[:4] == T.BACKBONE and all(x in T.TYPE_INDEX for x in t.types)
        for name, hb in T.hbond_roles(a).items():
            assert name in t.atoms
            for b in (hb.base, hb.base2):
                assert b == "-C" or b in t.atoms
        built = {"N", "CA", "C", "O"} | ({"CB"} if a != "G" else set())
        for atom, A, B, C, k in T.IC_SPEC[a]:
            assert {A, B, C} <= built and k <= T.N_CHI[a]
            built.add(atom)
        assert built == set(t.atoms)
    assert T.atom_type("P", "N") == "N_pro" and T.atom_type("I", "CD1") == "CH3" and T.atom_type("T", "CG2") == "CH3"
    assert T.atom_type("W", "CH2") == "C_aro" and T.atom_type("G", "OXT") == "O_carboxyl"
    assert abs(sum(T.atom_charges("R").values()) - 1) < 1e-9 and sum(T.atom_charges("D").values()) == -1
    assert sum(T.atom_charges("H").values()) == 0 and sum(T.atom_charges("H", his_charged=True).values()) == 1


def test_dihedral_known():
    p = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]])
    assert abs(dihedral(*p) - (-90.0)) < 1e-6 or abs(dihedral(*p) - 90.0) < 1e-6
    p2 = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert abs(dihedral(*p2)) < 1e-6


# ----------------------------------------------------------------------------- structure loading
@needs_pdb
def test_load_ubiquitin(ubq):
    assert len(ubq) == 76 and ubq.seq.startswith("MQIFVKTLTG") and ubq.resnum[0] == "1"
    assert not ubq.missing_sc.any() and np.isfinite(ubq.oxt).sum() <= 3
    ss = "".join(ubq.ann["ss3"])
    assert ss[23:33] == "H" * 10 and ss[1:7] == "E" * 6
    assert ubq.ann["rsa"].max() < 1.6 and (ubq.ann["rsa"] < 0.05).sum() > 10
    assert np.isnan(ubq.ann["phi"][0]) and np.isnan(ubq.ann["psi"][-1]) and not ubq.ann["cis"].any()
    at = ubq.atoms()
    assert len(at["xyz"]) == ubq.mask14.sum() + np.isfinite(ubq.oxt[:, 0]).sum()


@needs_pdb
def test_pdb_quirks(tmp_path):
    """MSE, altlocs (highest occupancy wins), insertion codes, NMR multi-model, waters/metals."""
    st = gemmi.read_structure(str(UBQ))
    st.remove_hydrogens()
    ch = st[0][0]
    met = ch[0]
    met.name, met.het_flag = "MSE", "H"
    for a in met:
        if a.name == "SD":
            a.name, a.element = "SE", gemmi.Element("Se")
    ch[9].seqid = gemmi.SeqId(9, "A")                     # G10 -> 9A
    leu = ch[7]                                           # L8: add altloc B (occ 0.6) shifted by 1 A
    extra = []
    for a in leu:
        if a.name in ("CD1", "CD2"):
            a.altloc, a.occ = "A", 0.4
            b = a.clone()
            b.altloc, b.occ = "B", 0.6
            b.pos = gemmi.Position(a.pos.x + 1.0, a.pos.y, a.pos.z)
            extra.append(b)
    for b in extra:
        leu.add_atom(b)
    zn = gemmi.Residue()
    zn.name, zn.het_flag, zn.seqid = "ZN", "H", gemmi.SeqId(101, " ")
    at = gemmi.Atom()
    at.name, at.element, at.pos = "ZN", gemmi.Element("Zn"), gemmi.Position(0, 0, 0)
    zn.add_atom(at)
    ch.add_residue(zn)
    model2 = st[0].clone()
    model2.num = 2
    for r in model2[0]:
        for a in r:
            a.pos = gemmi.Position(a.pos.x + 50, a.pos.y, a.pos.z)
    st.add_model(model2)
    st.setup_entities()
    f = tmp_path / "quirks.pdb"
    st.write_pdb(str(f))
    p = load_protein(f)
    ref = load_protein(UBQ)
    assert len(p) == 76 and p.seq == ref.seq and p.modified[0] and p.resname[0] == "MSE"
    assert p.mask14[0, :8].all() and np.allclose(p.xyz14[0, 6], ref.xyz14[0, 6], atol=1e-3)   # SE -> SD
    assert p.resnum[9] == "9A" and p.index("A", "9A") == 9
    assert p.altloc[7] and np.allclose(p.xyz14[7, 6, 0], ref.xyz14[7, 6, 0] + 1.0, atol=1e-3)  # altloc B kept
    assert np.allclose(p.xyz14[:, 1], ref.xyz14[:, 1], atol=1e-3)                               # model 1 only
    assert "ZN" in p.het_resname.tolist()


@needs_pdb
def test_missing_sidechain_flag(tmp_path):
    st = gemmi.read_structure(str(UBQ))
    st.remove_hydrogens()
    lys = st[0][0][5]                                     # K6: delete NZ
    for i in reversed(range(len(lys))):
        if lys[i].name == "NZ":
            del lys[i]
    f = tmp_path / "miss.pdb"
    st.write_pdb(str(f))
    p = load_protein(f)
    assert p.missing_sc[5] and p.missing_sc.sum() == 1


# ----------------------------------------------------------------------------- geometry builders
@needs_pdb
def test_rebuild_native_sidechains(ubq):
    # 1UBQ (1987 refinement) deviates more from modern ideal geometry than the PISCES test set (median 0.10 / 0.04 A)
    chi = side_chain_chis(ubq)
    bb = torch.tensor(ubq.xyz14[:, :4], dtype=torch.float64)
    for cb, tol in ((None, 0.25), (torch.tensor(ubq.xyz14[:, 4], dtype=torch.float64), 0.15)):
        X, m = build_sidechains(bb, torch.tensor(ubq.aa), torch.tensor(np.nan_to_num(chi))[:, None], cb=cb)
        X = X[:, 0].numpy()
        ok = ubq.mask14 & m[:, 0].numpy()
        ok[:, :4] = False
        d = np.linalg.norm(X - ubq.xyz14, axis=-1)
        rmsd = np.sqrt(np.where(ok, d ** 2, 0).sum(1) / np.maximum(ok.sum(1), 1))[ok.any(1)]
        assert np.median(rmsd) < tol
    for a, names in (("V", ("CG1", "CG2")), ("T", ("OG1", "CG2")), ("I", ("CG1", "CG2", "CD1")), ("L", ("CD1", "CD2"))):
        rows = np.flatnonzero(ubq.aa == T.AA_INDEX[a])
        for nm in names:      # naming / chirality conventions: each named atom lands on its native counterpart
            assert np.median(d[rows, T.TEMPLATES[a].index(nm)]) < 0.3, (a, nm)


@needs_pdb
def test_ideal_cb_and_amide_h(ubq):
    n, ca, c = (ubq.xyz14[:, k].astype(np.float64) for k in range(3))
    cb = ideal_cb(n, ca, c)
    nong = ubq.aa != T.AA_INDEX["G"]
    assert np.median(np.linalg.norm(cb[nong] - ubq.xyz14[nong, 4], axis=-1)) < 0.15     # L-chirality
    cbt = ideal_cb_t(torch.tensor(n), torch.tensor(ca), torch.tensor(c)).numpy()
    assert np.allclose(cb, cbt, atol=1e-6)
    h = backbone_h(ubq)
    ht = amide_h(torch.tensor(np.roll(c, 1, 0)), torch.tensor(n), torch.tensor(ca)).numpy()
    ok = np.isfinite(h[:, 0])
    assert ok.sum() == 76 - 1 - (ubq.aa[1:] == T.AA_INDEX["P"]).sum()
    assert np.allclose(h[ok], ht[ok], atol=1e-5)
    assert np.allclose(np.linalg.norm(h[ok] - n[ok], axis=-1), 1.01, atol=1e-4)
    # H roughly opposite to the carbonyl O of i-1 (trans peptide): angle O(i-1)...C(i-1)/N-H antiparallel
    o_prev = np.roll(ubq.xyz14[:, 3], 1, 0)
    cosang = np.sum((h - n) * (o_prev - np.roll(c, 1, 0)), -1)[ok]
    assert (cosang < 0).mean() > 0.95
    two = donor_h(torch.tensor(n[:3]), torch.tensor(ca[:3]), torch.tensor(c[:3]), "sp2_2H")
    assert two.shape == (3, 2, 3)


def test_builder_batching_mixed_aa():
    bb = torch.tensor([[[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0]]] * 20, dtype=torch.float64)
    chis = torch.tensor([[[-60.0, 180.0, 60.0, 180.0]] * 3] * 20, dtype=torch.float64)
    X, m = build_sidechains(bb, torch.arange(20), chis)
    assert X.shape == (20, 3, 14, 3) and m.shape == (20, 3, 14)
    for i, a in enumerate(T.AA1):
        assert m[i, 0].sum() == len(T.TEMPLATES[a].atoms)
        assert torch.isfinite(X[i, :, :3]).all() and torch.isfinite(X[i, :, 4:len(T.TEMPLATES[a].atoms)]).all()
        Xs, _ = build_sidechains(bb[:1], i, chis[:1])                  # scalar aa path gives the same result
        assert torch.allclose(Xs[0, 0, 4:len(T.TEMPLATES[a].atoms)], X[i, 0, 4:len(T.TEMPLATES[a].atoms)])
    # bond lengths from the IC table are reproduced
    k = T.AA_INDEX["K"]
    t = T.TEMPLATES["K"]
    assert abs(torch.linalg.norm(X[k, 0, t.index("NZ")] - X[k, 0, t.index("CE")]).item() - T.IDEAL_IC["K"]["NZ"][0]) < 1e-6
    # chi round trip
    xyz = X[k, 0].numpy()
    got = [dihedral(*(xyz[t.index(n)] for n in q)) for q in t.chis]
    assert np.allclose(got, [-60, 180, 60, 180], atol=1e-6) or np.allclose(np.abs(got), [60, 180, 60, 180], atol=1e-6)


def test_smooth_cn_numpy_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3)) * 8
    a = smooth_cn(x)
    xt = torch.tensor(x, requires_grad=True)
    b = smooth_cn(xt)
    assert np.allclose(a, b.detach().numpy(), atol=1e-4)
    b.sum().backward()
    assert torch.isfinite(xt.grad).all()


# ----------------------------------------------------------------------------- rotamer library / statistics
def test_rotamer_library():
    from physddg.rotlib import RotamerLibrary
    lib = RotamerLibrary()
    lp, mean, sd = lib.tables("L", [-63.0, -120.0, np.nan], [-41.0, 130.0, np.nan])
    assert np.all(np.exp(lp).sum(1) > 0.97) and np.all(np.exp(lp).sum(1) < 1.0001)
    q = lib.query("V", [-63.0], [-41.0], max_r=3)
    assert abs(q["chis"][0, 0, 0] - 175) < 15                         # Val in helix: t rotamer dominates
    q = lib.query("F", np.full(5, -120.0), np.full(5, 130.0), cum=0.9, expand=True)
    assert q["chis"].shape[1] == 9 * lib.n_rot("F") and q["mask"].sum(1).min() >= 9
    assert np.all(np.diff(q["logp"][0, ::9][q["mask"][0, ::9]]) <= 1e-6)    # sorted base rotamers
    chis = np.array([[-65.0, 175.0, 0, 0], [180.0, 65.0, 0, 0]])
    r = lib.assign("L", chis)
    assert (r >= 0).all() and r[0] != r[1]
    assert lib.query("G", [0.0], [0.0])["chis"].shape == (1, 1, 4)


def test_pdbstats():
    from physddg.rotlib import PDBStats
    s = PDBStats()
    assert s.rama_energy("gly", [80.0], [10.0])[0] < s.rama_energy("general", [80.0], [10.0])[0]
    assert s.rama_energy("pro", [-120.0], [130.0])[0] > s.rama_energy("pro", [-65.0], [140.0])[0]
    E = s.env_energy(["H", "L"], cn=[30.0, 8.0])
    assert E.shape == (2, 20)
    assert E[0, T.AA_INDEX["L"]] < E[0, T.AA_INDEX["K"]] and E[1, T.AA_INDEX["K"]] < E[1, T.AA_INDEX["L"]]
    assert s.env_energy(["E"], cn=[20.0], phi=[-120.0], psi=[130.0], variant="cn_ss_bb").shape == (1, 20)
    assert s.env_energy(["E"], rsa=[0.0], variant="rsa_ss").shape == (1, 20)
    assert s.d["p_aa_given_posphi"].argmax() == T.AA_INDEX["G"]
    assert s.sc_entropy()[T.AA_INDEX["R"]] > s.sc_entropy()[T.AA_INDEX["V"]] > 0
    allaa = s.rama_energy_all_aa([-63.0, -120.0], [-41.0, 130.0], next_is_pro=[False, True])
    assert allaa.shape == (2, 20) and np.isfinite(allaa).all()
