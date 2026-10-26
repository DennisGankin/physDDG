"""Analytic checks for physddg/enm.py (GNM/ANM spectra, zero modes, invariances, derived descriptors)."""
import numpy as np
import pytest

from physddg import enm


def chain(n, b=3.8):
    return np.c_[np.arange(n) * b, np.zeros(n), np.zeros(n)]


def ring(n, b=3.8):
    r = b / (2 * np.sin(np.pi / n))
    t = 2 * np.pi * np.arange(n) / n
    return np.c_[r * np.cos(t), r * np.sin(t), np.zeros(n)]


def blob(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) * 4.0
    return x


def test_gnm_linear_chain_spectrum():
    n = 12
    g = enm.gnm(chain(n), cutoff=4.0)                       # nearest neighbours only -> path-graph Laplacian
    ref = np.sort(2 - 2 * np.cos(np.pi * np.arange(n) / n))
    assert np.allclose(np.sort(g["eigvals"]), ref, atol=1e-10)
    assert g["n_zero"] == 1
    assert np.allclose(g["msf"], np.diag(np.linalg.pinv(enm.kirchhoff(chain(n), 4.0))))
    assert g["msf"][0] == pytest.approx(g["msf"][-1]) and g["msf"][0] > g["msf"][n // 2]   # ends most mobile


def test_gnm_ring_spectrum_and_uniform_msf():
    n = 10
    g = enm.gnm(ring(n), cutoff=4.0)
    ref = np.sort(2 - 2 * np.cos(2 * np.pi * np.arange(n) / n))
    assert np.allclose(np.sort(g["eigvals"]), ref, atol=1e-10)
    assert np.allclose(g["msf"], g["msf"].mean())
    # cycle-graph Laplacian pseudo-inverse diagonal: (n^2 - 1) / (12 n)
    assert g["msf"][0] == pytest.approx((n ** 2 - 1) / (12 * n))


def test_gnm_disconnected_components():
    x = np.vstack([chain(5), chain(5) + [100, 0, 0]])
    g = enm.gnm(x, cutoff=4.0)
    assert g["n_zero"] == 2 and np.isfinite(g["msf"]).all()


def test_anm_rigid_body_zero_modes():
    x = blob()
    H = enm.hessian(x, cutoff=15.0)
    assert np.allclose(H, H.T)
    n = len(x)
    c = x - x.mean(0)
    vecs = [np.tile(np.eye(3)[k], n) for k in range(3)]                       # translations
    for k in range(3):                                                         # infinitesimal rotations
        ax = np.eye(3)[k]
        vecs.append(np.cross(ax, c).ravel())
    for v in vecs:
        assert np.abs(H @ v).max() < 1e-9
    lam = np.linalg.eigvalsh(H)
    assert (lam < 1e-8 * lam.max()).sum() == 6
    C, _, nz = enm.anm_cov(x)
    assert nz == 6


def test_invariance_under_rigid_motion():
    x = blob(40, 1)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    y = x @ R.T + [5.0, -3.0, 11.0]
    a, b = enm.enm_descriptors(x), enm.enm_descriptors(y)
    for k in ("gnm_msf", "gnm_msf_slow", "anm_msf", "stiff_local", "stiff_all", "prs_eff", "prs_sens", "dfi", "degree"):
        assert np.allclose(a[k], b[k], rtol=1e-6, atol=1e-9), k


def test_derived_descriptors():
    x = blob(50, 2)
    C, _, _ = enm.anm_cov(x)
    k = enm.mech_stiffness(C, x)
    off = ~np.eye(len(x), dtype=bool)
    assert np.allclose(k[off], k.T[off]) and (k[off] > 0).all()
    p = enm.prs(C)
    assert np.allclose(p["P"], p["P"].T)
    d = enm.enm_descriptors(x)
    assert np.corrcoef(np.log(d["gnm_msf"]), np.log(d["anm_msf"]))[0, 1] > 0.5      # GNM and ANM agree
    assert np.corrcoef(np.log(d["gnm_msf"]), d["degree"])[0, 1] < 0                  # flexible = few contacts
    assert np.isnan(d["gnm_msf_plddt"]).all() and np.isnan(d["cen_msf"]).all()       # coords only, no pLDDT


def test_plddt_weighting():
    x = blob(40, 3)
    a = enm.enm_descriptors(x, plddt=np.full(40, 100.0))
    assert np.allclose(a["gnm_msf_plddt"], a["gnm_msf"])
    p = np.full(40, 95.0)
    p[:5] = 40.0
    b = enm.enm_descriptors(x, plddt=p)
    r = b["gnm_msf_plddt"] / b["gnm_msf"]
    assert r[:5].mean() > r[5:].mean()                     # weak springs at low confidence -> more mobile


def test_protein_input(tmp_path):
    from pathlib import Path
    from physddg import structure as S
    root = Path(__file__).resolve().parents[1]
    f = root / "data" / "structures" / "xtal" / "fireprot" / "1A23_A.pdb"
    if not f.exists():
        pytest.skip("structure not available")
    from threadpoolctl import threadpool_limits
    p = S.load_protein(f, keep_het=False)
    with threadpool_limits(4):
        d = enm.enm_descriptors(p)
    assert all(len(v) == len(p) for v in d.values())
    assert np.isfinite(d["cen_msf"]).all() and np.isfinite(d["gnm_msf"]).all()
