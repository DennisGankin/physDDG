#!/usr/bin/env python
"""Simple structure descriptors per variant -> data/features/simple_<dataset>.parquet (git-ignored cache).

  PYTHONPATH=. .venv/bin/python scripts/50_features_simple.py [--datasets megascale fireprot ...] [--force]
Prints coverage (site found in structure, WT letter matches) per dataset; never drops rows silently.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physddg import features_simple as F  # noqa: E402

ALL = ["megascale", "fireprot", "s669", "s461", "ssym", "s2648", "domainome", "abundance_dms", "paired_af", "paired_af_xtal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in a.datasets:
        t = time.time()
        f = F.build(name, a.jobs, a.force)
        n = len(f)
        print(f"{name:15s} rows {n:7d}  site_found {f.site_found.mean():.4f}  wt_match {f.wt_match.mean():.4f}  "
              f"de_env non-NaN {f.de_env.notna().mean():.4f}  structures {f.structure.nunique()}  {time.time() - t:.0f}s",
              flush=True)
        bad = f[~f.wt_match]
        if len(bad):
            print(f"   WT mismatch/missing examples:\n{bad.head(5)[F.KEYS + ['structure']].to_string()}")


if __name__ == "__main__":
    main()
