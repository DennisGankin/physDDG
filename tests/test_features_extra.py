"""Regression test: vectorised features_extra.site_descriptors == original loop implementation (_site_descriptors_v1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physddg import features_extra as X

ROOT = Path(__file__).resolve().parents[1]
STRUCTS = (
    ["/workspace/data/megascale_pdbs/" + f for f in ("1PSE.pdb", "1GYZ.pdb", "1A32.pdb", "1AOY.pdb")]
    + [str(ROOT / "data/structures/xtal/fireprot" / f) for f in ("1A23_A.pdb", "1CQW_A.pdb", "1EY0_A.pdb", "1LZ1_A.pdb")]
    + [str(ROOT / "data/structures/domainome/af_construct/A0A2R8Y422_PF00240_2.pdb"),
       "/workspace/data/yeast_af/pdbs/P20486.pdb", "/workspace/data/yeast_af/pdbs/O13297.pdb",
       str(ROOT / "data/structures/bench/6q62A.pdb")]
)


@pytest.mark.parametrize("path", STRUCTS, ids=[Path(s).stem for s in STRUCTS])
def test_site_descriptors_identical(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not available")
    a, b = X._site_descriptors_v1(path), X.site_descriptors(path)
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fiub":
            x, y = x.astype(float), y.astype(float)
            assert (np.isnan(x) == np.isnan(y)).all(), c
            assert np.nanmax(np.abs(x - y), initial=0.0) <= 1e-6, c
        else:
            assert (x == y).all(), c
