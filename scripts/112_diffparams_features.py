#!/usr/bin/env python
"""Stage-1 engine features for whole benchmark tables at a fitted parameter vector theta (physddg/diffparams.py).

  PYTHONPATH=. .venv/bin/python scripts/112_diffparams_features.py --theta results/diff_params/fits/X.json \
      --datasets megascale fireprot --gpus 0 1 [--tag NAME]
  (--zero: theta = 0, i.e. the literature parameters through the same lean code path)

Same rows and column names as scripts/55 stage 1 (s1_dE_<term> for the 8 B6gs1 terms, s1_dTS, s1_dprior_avg,
s1_dref_apolar/polar, s1_dF, s1_nrot_*; `d_*`/`ctx_*` statistical columns are NOT recomputed: they do not depend
on theta and are merged from the standard feature files). Output: data/features/diffparams/phys_s1_<dataset>_<tag>.parquet
(git-ignored); per-structure cache data/features/diffparams/cache_<tag>/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("p55", ROOT / "scripts" / "55_phys_features.py")
P55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P55)

FEAT = ROOT / "data" / "features" / "diffparams"
KEYS = P55.KEYS


def theta_of(path: str | None, zero: bool = False, it: int | None = None) -> np.ndarray:
    from physddg import diffparams as D
    if zero:
        return np.zeros(D.NP)
    r = json.loads(Path(path).read_text())
    if it is not None:
        tr = {t["iter"]: t for t in r["trajectory"]}
        return np.asarray(tr[it]["theta"], float)
    return np.asarray(r["theta"], float)


def tag_of(theta: np.ndarray) -> str:
    return hashlib.md5(json.dumps(np.round(np.asarray(theta, float), 6).tolist()).encode()).hexdigest()[:10]


def features(datasets, theta, gpus=(0, 1), cpus=8, tag: str | None = None, rows: dict | None = None) -> dict:
    """Compute (cached) and assemble per-dataset feature tables at theta. Returns {dataset: path}."""
    tag = tag or tag_of(theta)
    cdir = FEAT / f"cache_{tag}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "theta.json").write_text(json.dumps(np.asarray(theta, float).tolist()))
    rows = rows or {n: P55.dataset_rows(n) for n in datasets}
    structures: dict[str, set] = {}
    for r in rows.values():
        for s, c, n in r[["structure", "chain", "pdb_resnum"]].drop_duplicates().itertuples(index=False):
            structures.setdefault(s, set()).add((c, n))
    t0 = time.time()
    from physddg.diffparams import scan_structures
    scan_structures(structures, theta, cdir, list(gpus), cpus)
    print(f"[112] tag {tag}: {len(structures)} structures computed in {time.time() - t0:.0f}s", flush=True)
    from physddg.predict import engine_columns
    out = {}
    for n, r in rows.items():
        parts = []
        for structure, g in r.groupby("structure", sort=False):
            f = P55.cache_file(cdir, structure)
            z = dict(np.load(f, allow_pickle=False)) if f.exists() else {}
            if "failed" in z:
                z = {}
            parts.append(pd.concat([g[KEYS + ["structure"]], engine_columns(z, g, 1)], axis=1))
        df = pd.concat(parts).loc[r.index]
        p = FEAT / f"phys_s1_{n}_{tag}.parquet"
        df.to_parquet(p, index=False)
        out[n] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default=None)
    ap.add_argument("--iter", type=int, default=None, help="use the theta of this L-BFGS iteration of the fit json")
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["megascale", "fireprot"])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    th = theta_of(a.theta, a.zero, a.iter)
    res = features(a.datasets, th, a.gpus, a.cpus, a.tag)
    for n, p in res.items():
        print(n, p)


if __name__ == "__main__":
    main()
