"""Tests for the mutation / packing engine (physddg/mutate.py)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from physddg import energy as E
from physddg import topology as T
from physddg.mutate import MutateConfig, SiteScan, stat_features
from physddg.structure import annotate, load_protein

PDB = Path("/workspace/data/megascale_pdbs/1A32.pdb")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not PDB.exists(), reason="test structure not available")


@pytest.fixture(scope="module")
def prot():
    return annotate(load_protein(PDB))


@pytest.fixture(scope="module")
def scan(prot):
    return SiteScan(prot, MutateConfig(), DEV)


def test_stage1_wt_zero_deterministic_batch_independent(scan, prot):
    sites = np.array([3, 10, 25, 40])
    a = scan.stage1(sites)
    b = scan.stage1(sites)
    for k in ("F", "E_avg", "E_min", "S_site", "ref"):
        assert np.array_equal(a[k], b[k]), k                        # bitwise deterministic
    c = scan.stage1(sites[1:2])                                     # same site, different batch composition
    assert np.allclose(a["F"][1], c["F"][0], atol=1e-4) and np.allclose(a["E_avg"][1], c["E_avg"][0], atol=1e-4)
    wt = prot.aa[sites]
    dF = a["F"] - a["F"][np.arange(len(sites)), wt][:, None]
    assert np.all(dF[np.arange(len(sites)), wt] == 0.0)
    assert a["F"].shape == (4, 20) and a["E_avg"].shape == (4, 20, len(scan.terms))
    assert (a["n_rot"][:, T.AA_INDEX["G"]] == 1).all() and (a["n_rot"][:, T.AA_INDEX["R"]] > 5).all()
    # F <= U_min (free energy below the best state) and F >= U_min - kT ln n
    assert np.all(a["F"] <= a["U_min"] + 1e-4)
    assert np.all(a["F"] >= a["U_min"] - scan.cfg.kT * np.log(a["n_rot"]) - 1e-4)


def test_stage1_matches_direct_energy(scan, prot):
    """Boltzmann average for a single-rotamer residue (Ala) equals intra + env energy of that residue."""
    s = 10
    r = scan.stage1(np.array([s]))
    cand = scan.site_candidates(np.array([s]))
    ia = np.flatnonzero(cand["a"] == T.AA_INDEX["A"])
    mov = scan._blocks(np.array([s]), np.array([T.AA_INDEX["A"]]), cand["xyz"][torch.as_tensor(ia, device=DEV)])
    em = torch.ones(len(prot), dtype=torch.bool, device=DEV)
    em[s] = False
    p = scan.cfg.params
    e = (E.intra_energy(mov, p, include_ref=False) + E.env_energy(mov, scan.env, p, env_mask=em)).cpu().numpy()[0]
    assert np.allclose(r["E_avg"][0, T.AA_INDEX["A"]], e, atol=1e-4)


def test_stage2_reduces_to_stage1_without_neighbours(prot):
    cfg = MutateConfig(nb_radius=0.0)
    sc = SiteScan(prot, cfg, DEV)
    sites = np.array([5, 30])
    s2 = sc.stage2(sites)
    s1 = SiteScan(prot, replace(cfg, max_rot=cfg.s2_max_rot), DEV).stage1(sites)
    assert (s2["n_nb"] == 0).all()
    assert np.allclose(s2["F"], s1["F"], atol=2e-3)
    assert np.allclose(s2["E_avg"], s1["E_avg"], atol=2e-3)


def test_stage2_wt_zero_and_determinism(scan, prot):
    sites = np.array([8, 20])
    a = scan.stage2(sites)
    b = scan.stage2(sites)
    for k in ("F", "E_avg", "S_local"):
        assert np.array_equal(a[k], b[k]), k
    assert (a["n_nb"] > 0).all()
    c = scan.stage2(sites[:1])
    assert np.allclose(a["F"][0], c["F"][0], atol=1e-3)
    assert np.isfinite(a["F"]).all()


def test_neighbour_sets_identical_for_all_aa(scan):
    sites = np.array([12])
    cand = scan.site_candidates(sites, max_rot=scan.cfg.s2_max_rot)
    nb = scan.neighbour_sets(sites, cand)[0]
    assert 12 not in nb and len(nb) <= scan.cfg.nb_max
    seq = scan.prot.seq
    assert all(T.N_CHI[seq[j]] > 0 and seq[j] != "P" for j in nb)


def test_cb_rule(scan, prot):
    gly = np.flatnonzero(prot.aa == T.AA_INDEX["G"])
    non = np.flatnonzero((prot.aa != T.AA_INDEX["G"]) & (prot.aa != T.AA_INDEX["P"]))[:3]
    cb_nat = scan._cb(non, T.AA_INDEX["L"]).cpu().numpy()
    assert np.allclose(cb_nat, prot.xyz14[non, 4], atol=1e-5)               # native CB kept
    if len(gly):
        cb = scan._cb(gly[:1], T.AA_INDEX["L"]).cpu().numpy()[0]
        n, ca = prot.xyz14[gly[0], 0], prot.xyz14[gly[0], 1]
        assert abs(np.linalg.norm(cb - ca) - T.IDEAL_IC["L"]["CB"][0]) < 1e-3 and np.linalg.norm(cb - n) > 2.0


def test_stat_features(prot):
    sf = stat_features(prot, np.arange(len(prot)))
    for k, v in sf["per_aa"].items():
        assert v.shape == (len(prot), 20) and np.isfinite(v).all(), k
    assert (sf["per_aa"]["rama_prev"][0] == 0).all()


def test_stage2_frozen_env_reverse(prot):
    """Without neighbours every environment row equals stage-1 F, so reverse = -forward exactly; with neighbours the
    diagonal (residue in its own relaxed environment) is finite and WT->WT is zero."""
    sites = np.array([5, 30])
    s2 = SiteScan(prot, MutateConfig(nb_radius=0.0), DEV).stage2(sites)
    F = s2["F_env"]
    assert F.shape == (2, 20, 20) and s2["E_env"].shape[:3] == (2, 20, 20)
    assert np.allclose(F, F[:, :1, :], atol=1e-5)                     # env rows identical (no flexible residues)
    w = prot.aa[sites]
    for k in range(2):
        fz = F[k, w[k], :] - F[k, w[k], w[k]]
        rev = np.array([F[k, m, w[k]] - F[k, m, m] for m in range(20)])
        assert np.allclose(rev, -fz, atol=1e-4)
    s2n = SiteScan(prot, MutateConfig(), DEV).stage2(sites)
    assert np.isfinite(s2n["F_env"]).all() and np.isfinite(s2n["E_env"]).all()
