"""Tests for the AlphaFold-mode treatments (physddg/afmode.py). CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import afmode as A  # noqa: E402
from physddg.structure import annotate, load_protein  # noqa: E402

PDBS = sorted((ROOT / "data" / "structures" / "domainome" / "af_construct").glob("*.pdb"))


@pytest.fixture(scope="module")
def prot():
    if not PDBS:
        pytest.skip("no AF construct crops on disk (run scripts/22_domainome_structures.py)")
    return annotate(load_protein(PDBS[0], keep_het=False))


def test_window_mean_edges():
    x = np.arange(10.0)
    assert np.allclose(A.window_mean(x, 1), x)
    w = A.window_mean(x, 3)
    assert np.isclose(w[5], 5.0) and np.isclose(w[0], 0.5) and np.isclose(w[-1], 8.5)
    ch = np.array(["A"] * 5 + ["B"] * 5)
    w2 = A.window_mean(x, 5, ch)
    assert np.isclose(w2[4], np.mean(x[2:5])) and np.isclose(w2[5], np.mean(x[5:8]))


def test_cap_monotone_in_plddt(prot):
    cfg = A.CapConfig(cap_lo=0.5, cap_hi=3.0, p_lo=50, p_hi=90)
    pl = A.plddt(prot)
    cap = A.cap_from_plddt(prot, cfg)
    assert cap.min() >= 0.5 - 1e-6 and cap.max() <= 3.0 + 1e-6
    order = np.argsort(pl)
    assert np.all(np.diff(cap[order]) >= -1e-6)          # non-decreasing in pLDDT


def test_trim_keeps_structure_consistent(prot):
    keep = A.trim_mask(prot, A.TrimConfig(below=101.0, window=1, packed_exempt=False, min_keep=1))
    assert keep.sum() == 0 or keep.sum() == len(prot)     # min_keep fallback: everything or nothing dropped
    keep = A.trim_mask(prot, A.TrimConfig(below=70.0, window=1, packed_exempt=False))
    q = A.trim_protein(prot, keep)
    assert len(q) == keep.sum()
    assert list(q.chain) == [c for c, m in zip(prot.chain, keep) if m]
    # dihedrals of kept residues are preserved from the full model (trimming removes contacts, not geometry)
    assert np.allclose(np.nan_to_num(q.ann["phi"]), np.nan_to_num(prot.ann["phi"][keep]))
    assert q.ann["nterm_orig"].sum() <= 1 + (prot.chain[1:] != prot.chain[:-1]).sum()
    # burial is recomputed, and can only drop when neighbours are deleted
    if keep.sum() < len(prot):
        assert q.ann["cn"].mean() <= prot.ann["cn"][keep].mean() + 1e-6


def test_packed_exemption_keeps_more(prot):
    base = A.TrimConfig(below=70.0, window=1, packed_exempt=False)
    assert A.trim_mask(prot, base).sum() <= A.trim_mask(prot, A.TrimConfig(
        below=70.0, window=1, packed_exempt=True, packed_contacts=3.0)).sum()


def test_write_trimmed_pdb_roundtrip(prot, tmp_path):
    keep = A.trim_mask(prot, A.TrimConfig(below=70.0, window=1, packed_exempt=False))
    keys = {k for k, m in zip(prot.keys, keep) if m}
    out = tmp_path / "trim.pdb"
    n = A.write_trimmed_pdb(prot.path, keys, out)
    q = load_protein(out, keep_het=False)
    assert n == len(q) == keep.sum()
    assert q.seq == "".join(np.array(list(prot.seq))[keep])
    assert np.allclose(np.nan_to_num(q.xyz14), np.nan_to_num(prot.xyz14[keep]))


def test_shrink_and_fit():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    pred = y + rng.normal(scale=0.3, size=400)
    pl = np.r_[np.full(200, 95.0), np.full(200, 40.0)]
    pred[200:] = rng.normal(size=200)                     # low-confidence half is pure noise
    w = A.logistic_weight(pl, 70, 10)
    assert w[0] > 0.9 and w[-1] < 0.1
    s = A.shrink(pred, pl, 70, 10, 0.0)
    assert np.corrcoef(s[:200], pred[:200])[0, 1] > 0.999      # within a confidence band: pure rescaling
    assert np.all(np.abs(s[200:]) < 0.1 * np.abs(pred[200:]) + 1e-12)   # w(40) = 0.047
    best = A.fit_shrink(pred, pl, y, groups=np.zeros(400, int))
    assert best["score"] >= 0


def test_burial_helpers():
    assert A.cn_to_burial(8.0) == 0.0 and A.cn_to_burial(24.0) == 1.0
    b = A.burial_swap(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([95.0, 40.0]), 70, 90)
    assert b[0] == 1.0 and b[1] == 0.0
    assert A.blend(np.array([2.0]), np.array([0.0]), 0.25) == 0.5


def test_flags_shapes(prot):
    f = A.flags(prot)
    n = len(prot)
    assert all(len(v) == n for v in f.values())
    assert f["low_plddt"].dtype == bool
    assert f["terminal"][0] and f["terminal"][-1]
    sub = A.flags(prot, np.arange(5))
    assert len(sub["plddt"]) == 5 and np.allclose(sub["plddt"], f["plddt"][:5])
