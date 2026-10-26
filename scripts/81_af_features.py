#!/usr/bin/env python
"""Physics features (physddg/mutate.py) recomputed under an AlphaFold-mode treatment (physddg/afmode.py).

  PYTHONPATH=. .venv/bin/python scripts/81_af_features.py --rows domainome --variant trim50 --stage 1 --gpus 2 3

Row sets (`--rows`):
  domainome        Domainome construct crops (AF v4)           = the rows of data/features/phys_s*_domainome.parquet
  domainome_v6     the same variants on the AF v6 construct crops
  domainome_full   the same variants on the full-length AF model (not cropped to the assayed construct)
  paired_af        AF side of data/processed/paired_af.csv ; paired_af_xtal = crystal side
  abundance_dms    full-length AF models of the 7 abundance DMS assays

Treatments (`--variant`, physddg.afmode.PRESETS): off | trim50 | trim50_nopack | trim70 | mavisp | cap_plddt |
cap_soft | repack_all. Trimming deletes residues from the model before any energy is computed; the cap variants
set a per-residue repulsion cap from pLDDT; repack_all is the global-repacking ablation.

Output: data/features/af/phys_s<stage>_<rows>_<variant>.parquet (git-ignored), same columns as scripts/55.
Per-structure cache in data/features/cache/af_s<stage>_<variant>_<cfghash>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import features_simple as FS  # noqa: E402

FEAT = ROOT / "data" / "features"
KEYS = FS.KEYS
ROWSETS = ["domainome", "domainome_v6", "domainome_full", "paired_af", "paired_af_xtal", "abundance_dms",
           "megascale", "fireprot"]


def norm_resnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def rows_for(name: str) -> pd.DataFrame:
    if name in ("domainome_v6", "domainome_full"):
        d = pd.read_parquet(ROOT / "data/processed/domainome.parquet",
                            columns=KEYS + ["structure", "af_full_structure", "af_full_resnum"])
        d = d[d.structure.notna()]
        if name == "domainome_full":
            d = d[d.af_full_structure.notna()].assign(structure=lambda x: x.af_full_structure,
                                                      pdb_resnum=lambda x: x.af_full_resnum)
        else:
            v6 = pd.read_csv(ROOT / "data/processed/domainome_af_v4_v6.csv")
            v6 = v6[(v6.v6_status == "ok") & (v6.n_seq_diff.fillna(1) == 0)][["domain_ID", "v6_construct"]]
            d = d.merge(v6, left_on="protein_id", right_on="domain_ID", how="inner")
            d = d.assign(structure=lambda x: x.v6_construct)
        return d[KEYS + ["structure"]]
    r = FS.dataset_rows(name)
    return r[KEYS + ["structure"]]


def dataset_rows(name: str) -> pd.DataFrame:
    r = rows_for(name)
    r = r.assign(pdb_resnum=norm_resnum(r.pdb_resnum), chain=r.chain.astype(str))
    return r.drop_duplicates(KEYS + ["structure"]).reset_index(drop=True)


def cfg_tag(stage: int, variant: str) -> str:
    from physddg.afmode import PRESETS
    from physddg.mutate import MutateConfig
    d = asdict(MutateConfig())
    d["params"] = {k: v for k, v in d["params"].items() if k != "extra"}
    src = [ROOT / "physddg" / f for f in ("params.py", "energy.py", "builder.py", "topology.py", "afmode.py",
                                          "mutate.py")]
    src += [ROOT / "physddg" / "data" / f for f in ("rotlib_bbdep.npz", "pdbstats.npz")]
    h = hashlib.md5(b"".join(f.read_bytes() for f in src)).hexdigest()
    return hashlib.md5(json.dumps([stage, variant, asdict(PRESETS[variant]), d, h],
                                  sort_keys=True, default=str).encode()).hexdigest()[:8]


def struct_path(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def cache_file(cdir: Path, structure: str) -> Path:
    return cdir / (hashlib.md5(structure.encode()).hexdigest() + ".npz")


# ----------------------------------------------------------------------------- workers
def _load(job):
    structure, sites = job
    from physddg.structure import annotate, load_protein
    try:
        p = annotate(load_protein(struct_path(structure), keep_het=False))
    except Exception as e:  # noqa: BLE001
        return structure, None, repr(e)
    return structure, p, sites


def gpu_worker(gpu: int, jobs: list, stage: int, variant: str, cdir: str, cpus: int, log: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from physddg import afmode as A
    from physddg.mutate import MutateConfig, stat_features
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = MutateConfig()
    mode = A.PRESETS[variant]
    rows = []
    with ProcessPoolExecutor(cpus) as ex:
        for structure, p, sites in ex.map(_load, jobs, chunksize=1):
            out = cache_file(Path(cdir), structure)
            if p is None:
                print(f"[gpu{gpu}] load failed {structure}: {sites}", flush=True)
                np.savez(out, failed=np.array([str(sites)]))
                continue
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            n_before = len(p)
            if mode.trim is not None:
                keep = A.trim_mask(p, mode.trim)
                p = A.trim_protein(p, keep)
            if mode.repack_all:
                p = A.repack_all(p, cfg, "cuda", passes=2)
            key = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
            idx = np.array(sorted({key[s] for s in sites if s in key}), np.int64)
            if len(idx) == 0:
                np.savez(out, chain=np.zeros(0, "U4"), resnum=np.zeros(0, "U8"), aa=np.zeros(0, "U1"),
                         terms=np.zeros(0, "U1"), n_trimmed=np.array([n_before - len(p)]))
                continue
            cap = A.cap_from_plddt(p, mode.cap) if mode.cap is not None else None
            sc = A.site_scan(p, cfg, "cuda", cap=cap)
            res = sc.stage1(idx) if stage == 1 else sc.stage2(idx)
            torch.cuda.synchronize()
            dt = time.time() - t0
            st = stat_features(p, idx, cfg.kT)
            arrs = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
            arrs.update({f"stat_{k}": v for k, v in st["per_aa"].items()})
            arrs.update({f"ctx_{k}": np.asarray(v) for k, v in st["context"].items()})
            np.savez(out, chain=p.chain[idx], resnum=p.resnum[idx], aa=np.array(list(p.seq))[idx],
                     terms=np.array(sc.terms), n_trimmed=np.array([n_before - len(p)]), **arrs)
            rows.append(dict(structure=structure, L=len(p), n_trimmed=n_before - len(p), n_sites=len(idx),
                             seconds=dt, gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=gpu))
            if len(rows) % 50 == 0:
                print(f"[gpu{gpu}] {len(rows)}/{len(jobs)} last L={len(p)} sites={len(idx)} {dt:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(log, index=False)


def compute(structures: dict, stage: int, variant: str, cdir: Path, gpus: list[int], cpus: int):
    import multiprocessing as mp
    jobs = [(s, sorted(v)) for s, v in structures.items() if not cache_file(cdir, s).exists()]
    if not jobs:
        return
    jobs.sort(key=lambda j: -len(j[1]))
    shards = [jobs[i::len(gpus)] for i in range(len(gpus))]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=gpu_worker, args=(g, sh, stage, variant, str(cdir), max(1, cpus // len(gpus)),
                                                  str(cdir / f"runtime_gpu{g}.csv")))
             for g, sh in zip(gpus, shards) if sh]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"gpu worker failed ({pr.exitcode})")


# ----------------------------------------------------------------------------- assembly
def assemble(name: str, rows: pd.DataFrame, stage: int, cdir: Path) -> pd.DataFrame:
    """Reuse scripts/55_phys_features.py's assembler (identical cache layout and column naming)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s55", ROOT / "scripts" / "55_phys_features.py")
    s55 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s55)
    return s55.assemble(name, rows, stage, cdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="domainome", choices=ROWSETS)
    ap.add_argument("--variant", default="off")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--min-sites", type=int, default=0,
                    help="only structures with at least this many needed sites (cheap subset for ablations)")
    a = ap.parse_args()

    rows = dataset_rows(a.rows)
    structures: dict[str, set] = {}
    for s, c, r in zip(rows.structure, rows.chain, rows.pdb_resnum):
        structures.setdefault(s, set()).add((c, r))
    if a.min_sites > 0:
        structures = {k: v for k, v in structures.items() if len(v) >= a.min_sites}
        rows = rows[rows.structure.isin(structures)].reset_index(drop=True)
    cdir = FEAT / "cache" / f"af_s{a.stage}_{a.variant}_{cfg_tag(a.stage, a.variant)}"
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"{a.rows}/{a.variant}: {len(rows)} rows, {len(structures)} structures -> {cdir.name}")
    t0 = time.time()
    compute(structures, a.stage, a.variant, cdir, a.gpus, a.cpus)
    print(f"engine {time.time() - t0:.0f}s")
    out = assemble(a.rows, rows, a.stage, cdir)
    (FEAT / "af").mkdir(parents=True, exist_ok=True)
    p = FEAT / "af" / f"phys_s{a.stage}_{a.rows}_{a.variant}.parquet"
    out.to_parquet(p, index=False)
    print(f"{p}: {len(out)} rows, site_found {out.site_found.mean():.3f}")


if __name__ == "__main__":
    main()
