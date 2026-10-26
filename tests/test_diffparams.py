"""Tests for the differentiable-parameter wrapper (physddg/diffparams.py, E3) and its hook in energy.features.
Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_diffparams.py"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from physddg import diffparams as D
from physddg import energy as E
from physddg.mutate import MutateConfig, SiteScan
from physddg.params import EnergyParams
from physddg.predict import engine_columns
from physddg.structure import annotate, load_protein

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
MS = "/workspace/data/megascale_pdbs/1A32.pdb"


@pytest.fixture(scope="module")
def prot():
    return annotate(load_protein(MS, keep_het=False))


def test_hook_absent_is_identical(prot):
    """No override -> the frozen table; an override equal to the frozen table gives bitwise identical energies."""
    p = EnergyParams()
    b = E.blocks_from_protein(prot, DEV, p)
    d0 = E.decompose(b, p).per_residue
    p1 = replace(p, extra={"type_table": E.tables(DEV)["param"].clone()})
    d1 = E.decompose(b, p1).per_residue
    assert torch.equal(d0, d1)


@pytest.mark.skipif(not (ROOT / "data/features/phys_s1_megascale.parquet").exists(), reason="run scripts/55 first")
def test_default_stage1_bitwise_unchanged(prot):
    """E1's SiteScan.stage1 with the default config reproduces the committed-pipeline feature file exactly
    (computed before the hook existed): the hook is backward compatible bit for bit."""
    f = pd.read_parquet(ROOT / "data/features/phys_s1_megascale.parquet")
    g = f[f.structure == MS].iloc[:200]
    key = {(c, r): i for i, (c, r) in enumerate(zip(prot.chain.tolist(), prot.resnum.tolist()))}
    sites = np.array(sorted({key[(c, r)] for c, r in zip(g.chain, g.pdb_resnum)}))
    sc = SiteScan(prot, MutateConfig(), DEV)
    res = sc.stage1(sites)
    z = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    z.update(chain=prot.chain[sites], resnum=prot.resnum[sites], aa=np.array(list(prot.seq))[sites], terms=np.array(sc.terms))
    cols = engine_columns(z, g, 1)
    for c in ["s1_dF", "s1_dTS", "s1_dprior_avg", "s1_dE_lj_atr", "s1_dE_lj_rep", "s1_dE_solv_polar", "s1_dE_hb_sc_bb"]:
        assert np.array_equal(cols[c].to_numpy(float), g[c].to_numpy(float)), c


def test_theta_zero_reproduces_stage1(prot):
    f = pd.read_parquet(ROOT / "data/features/phys_s1_megascale.parquet") if (ROOT / "data/features/phys_s1_megascale.parquet").exists() else None
    sites = np.arange(5, 25)
    z = D.full_scan(prot, sites, np.zeros(D.NP), device=DEV)
    sc = SiteScan(prot, MutateConfig(), DEV)
    ref = sc.stage1(sites)
    nT = len(D.TERMS)
    terms = list(sc.terms)
    idx = [terms.index(t) for t in D.TERMS]
    assert np.allclose(z["E_avg"], ref["E_avg"][..., idx], atol=2e-3, rtol=1e-4)
    assert np.allclose(z["F"], ref["F"], atol=2e-3)
    assert np.allclose(z["S_site"], ref["S_site"], atol=1e-4)
    assert np.allclose(z["prior_avg"], ref["prior_avg"], atol=1e-4)
    assert np.array_equal(z["ref"], ref["ref"])
    assert z["E_avg"].shape[-1] == nT
    del f


def test_batched_equals_loop_and_gradient(prot):
    rng = np.random.default_rng(0)
    sites = rng.choice(np.arange(2, len(prot) - 2), 8, replace=False)
    groups = {int(s): {int(prot.aa[s]), 3, 9, 15} for s in sites}
    cfg = D.lean_config()
    st = D.SiteSet(prot, groups, cfg, DEV)
    th = torch.as_tensor(rng.normal(0, 0.1, D.NP), device=DEV, dtype=torch.float32)
    with torch.no_grad():
        ph = D.physical(th, cfg, st.terms, cfg.pack_terms)
        a, b = st.energies_loop(ph), st.energies(ph)
    assert (a - b).abs().max().item() < 1e-3
    thg = th.clone().requires_grad_(True)
    G = st.forward(D.physical(thg, cfg, st.terms, cfg.pack_terms))
    ct = torch.randn_like(G)
    (g,) = torch.autograd.grad((G * ct).sum(), thg)
    assert torch.isfinite(g).all()
    h = 1e-3
    for k in (0, 15, 16, 25, 28, 30):          # solv_N_bb, rad_Capolar, rad_Caro, U_elec, kT, rep_scale
        e = torch.zeros_like(th)
        e[k] = h
        with torch.no_grad():
            fp = (st.forward(D.physical(th + e, cfg, st.terms, cfg.pack_terms)) * ct).sum().item()
            fm = (st.forward(D.physical(th - e, cfg, st.terms, cfg.pack_terms)) * ct).sum().item()
        fd = (fp - fm) / (2 * h)
        assert abs(fd - g[k].item()) <= 0.02 * abs(g[k].item()) + 0.05, (D.NAMES[k], fd, g[k].item())
