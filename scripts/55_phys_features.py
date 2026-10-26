#!/usr/bin/env python
"""Physics features per variant from the mutation/packing engine (physddg/mutate.py).

  PYTHONPATH=. .venv/bin/python scripts/55_phys_features.py --stage 1 [--datasets ...] [--gpus 0 1] [--cpus 24]

1. Collect benchmark rows (physddg.features_simple.dataset_rows: same row sets as the simple features; 'paired_af' =
   AF side, 'paired_af_xtal' = crystal side) and the unique (structure, chain, resnum) sites.
2. Per structure (one GPU process per GPU, CPU pool for loading + annotation): all 20 aa at every needed site
   (stage 1: fixed environment; stage 2: + neighbour flexibility). Cached per structure in
   data/features/cache/phys_s<stage>_<cfg>/<md5>.npz (git-ignored), so datasets sharing structures reuse work.
3. Assemble data/features/phys_s<stage>_<dataset>.parquet: KEYS + structure + flags + d<feature> = feat[mut] - feat[wt]
   (row order = deduplicated dataset_rows). WT letter is not required to match the structure (wt_match flag; e.g.
   reverse rows on AF models) -- features are defined for any (wt, mut) pair at a modelled site.
Runtime per structure and GPU memory are logged to results/mutate/runtime_s<stage>.csv.
"""
from __future__ import annotations

import argparse
import hashlib
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
from physddg import features_simple as FS  # noqa: E402
from physddg import topology as T  # noqa: E402

FEAT = ROOT / "data" / "features"
ALL = ["megascale", "fireprot", "s669", "s461", "ssym", "paired_af", "paired_af_xtal", "abundance_dms", "domainome"]
KEYS = FS.KEYS


VARIANTS = {  # D9 repulsion-cap settings (file suffix -> MutateConfig overrides); "" = defaults (= capB + soft term)
    "": {}, "capA": {"cap0": 1.0, "cap1": 0.0}, "capB": {"cap0": 3.0, "cap1": 0.0}, "capC": {"cap0": 1.0, "cap1": 2.0},
}


def make_cfg(variant: str):
    from physddg.mutate import MutateConfig
    return MutateConfig(**VARIANTS[variant])


def cfg_tag(stage: int, variant: str = "") -> str:
    from dataclasses import asdict
    d = asdict(make_cfg(variant))
    d["params"] = {k: v for k, v in d["params"].items() if k != "extra"}
    # energy / geometry sources and statistics tables also define the result (mutate.py is covered by the config)
    src = [ROOT / "physddg" / f for f in ("params.py", "energy.py", "builder.py", "topology.py")]
    src += [ROOT / "physddg" / "data" / f for f in ("rotlib_bbdep.npz", "pdbstats.npz")]
    h = hashlib.md5(b"".join(f.read_bytes() for f in src)).hexdigest()
    return hashlib.md5(json.dumps([stage, d, h], sort_keys=True, default=str).encode()).hexdigest()[:8]


def norm_resnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def dataset_rows(name: str) -> pd.DataFrame:
    r = FS.dataset_rows(name)
    r = r.assign(pdb_resnum=norm_resnum(r.pdb_resnum), chain=r.chain.astype(str))
    return r.drop_duplicates(KEYS + ["structure"]).reset_index(drop=True)


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


def gpu_worker(gpu: int, jobs: list, stage: int, cdir: str, cpus: int, log: str, variant: str = ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from physddg.mutate import SiteScan
    from physddg.predict import site_arrays
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = make_cfg(variant)
    rows = []
    with ProcessPoolExecutor(cpus) as ex:
        for structure, p, sites in ex.map(_load, jobs, chunksize=1):
            out = cache_file(Path(cdir), structure)
            if p is None:
                print(f"[gpu{gpu}] load failed {structure}: {sites}", flush=True)
                np.savez(out, failed=np.array(sites))
                continue
            key = {(c, r): i for i, (c, r) in enumerate(zip(p.chain.tolist(), p.resnum.tolist()))}
            idx = np.array(sorted({key[s] for s in sites if s in key}), np.int64)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            sc = SiteScan(p, cfg, "cuda")
            z = site_arrays(p, idx, sc, stage)
            torch.cuda.synchronize()
            dt = time.time() - t0
            np.savez(out, **z)
            rows.append(dict(structure=structure, L=len(p), n_sites=len(idx), n_variants=19 * len(idx), seconds=dt,
                             n_rot=int(z["n_rot"].sum()) if "n_rot" in z else -1,
                             gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=gpu))
            if len(rows) % 20 == 0:
                print(f"[gpu{gpu}] {len(rows)}/{len(jobs)} last L={len(p)} sites={len(idx)} {dt:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(log, index=False)


def compute(structures: dict, stage: int, cdir: Path, gpus: list[int], cpus: int, variant: str = ""):
    import multiprocessing as mp
    todo = [(s, sorted(v)) for s, v in structures.items() if not cache_file(cdir, s).exists()]
    if not todo:
        return
    todo.sort(key=lambda j: -len(j[1]))                         # big first, greedy balance by #sites
    load = [0] * len(gpus)
    parts = [[] for _ in gpus]
    for j in todo:
        k = int(np.argmin(load))
        parts[k].append(j)
        load[k] += len(j[1])
    ctx = mp.get_context("spawn")
    rdir = ROOT / "results" / "mutate"
    rdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for g, part in zip(gpus, parts):
        log = str(rdir / f"runtime_s{stage}{variant}_gpu{g}_{int(time.time())}.csv")
        pr = ctx.Process(target=gpu_worker, args=(g, part, stage, str(cdir), cpus, log, variant))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"GPU worker failed with exit code {pr.exitcode}")


# ----------------------------------------------------------------------------- assembly
def assemble(name: str, rows: pd.DataFrame, stage: int, cdir: Path) -> pd.DataFrame:
    """Variant columns per dataset row from the per-structure cache (physddg.predict.engine_columns)."""
    from physddg.predict import engine_columns
    parts = []
    for structure, g in rows.groupby("structure", sort=False):
        f = cache_file(cdir, structure)
        z = dict(np.load(f, allow_pickle=False)) if f.exists() else {}
        parts.append(pd.concat([g[KEYS + ["structure"]], engine_columns(z, g, stage)], axis=1))
    return pd.concat(parts).loc[rows.index]


def sanity(stage: int):
    from scipy.stats import spearmanr
    pre = "s1" if stage == 1 else "s2"
    lines = []
    for name, split in (("megascale", "val"), ("megascale", "test"), ("fireprot", "test"), ("s669", None),
                        ("ssym", None)):
        p = FEAT / f"phys_{pre}_{name}.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p)
        base = FS.dataset_rows(name)
        tab = pd.read_parquet(ROOT / "data/processed/megascale.parquet", columns=KEYS + ["ddg", "split"]) \
            if name == "megascale" else pd.read_csv(ROOT / f"data/processed/{name}.csv", low_memory=False)[KEYS + ["ddg", "split"]]
        tab = tab.assign(pdb_resnum=norm_resnum(tab.pdb_resnum), chain=tab.chain.astype(str))
        m = f.merge(tab.drop_duplicates(KEYS), on=KEYS, how="inner")
        if split:
            m = m[m.split == split]
        m = m[m.ddg.notna() & m[f"{pre}_dF"].notna()]
        del base
        res = {"dataset": name, "split": split or "all", "n": len(m)}
        for col in (f"{pre}_dF", f"{pre}_dU_min"):
            if col in m:
                res[col] = spearmanr(m[col], m.ddg).statistic
        res["dF+dref"] = spearmanr(m[f"{pre}_dF"] + m[f"{pre}_dref_apolar"] + m[f"{pre}_dref_polar"], m.ddg).statistic
        if name == "megascale":
            per = [spearmanr(g[f"{pre}_dF"], g.ddg).statistic for _, g in m.groupby("protein_id") if len(g) > 20]
            res["dF_per_protein_median"] = float(np.nanmedian(per))
        lines.append(res)
    df = pd.DataFrame(lines)
    out = ROOT / "results" / "mutate" / f"sanity_raw_dF_s{stage}.csv"
    df.round(4).to_csv(out, index=False)
    print(df.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--variants", nargs="+", default=[""], help="repulsion-cap variants: capA capB capC ('' = default)")
    ap.add_argument("--no-sanity", action="store_true")
    a = ap.parse_args()
    for v in a.variants:
        run(a, v)


def run(a, variant: str):
    sfx = f"_{variant}" if variant else ""
    cdir = FEAT / "cache" / f"phys_s{a.stage}{sfx}_{cfg_tag(a.stage, variant)}"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = {n: dataset_rows(n) for n in a.datasets}
    structures: dict[str, set] = {}
    for r in rows.values():
        for s, c, n in r[["structure", "chain", "pdb_resnum"]].drop_duplicates().itertuples(index=False):
            structures.setdefault(s, set()).add((c, n))
    n_sites = sum(len(v) for v in structures.values())
    print(f"stage {a.stage}: {len(structures)} structures, {n_sites} sites, cache {cdir}", flush=True)
    t0 = time.time()
    compute(structures, a.stage, cdir, a.gpus, a.cpus, variant)
    print(f"compute done in {time.time() - t0:.0f}s", flush=True)
    pre = "s1" if a.stage == 1 else "s2"
    for n, r in rows.items():
        f = assemble(n, r, a.stage, cdir)
        f.to_parquet(FEAT / f"phys_{pre}_{n}{sfx}.parquet", index=False)
        print(f"{n:15s} rows {len(f):7d} site_found {f.site_found.mean():.4f} wt_match {f.wt_match.mean():.4f}", flush=True)
    if not a.no_sanity and not variant:
        sanity(a.stage)


if __name__ == "__main__":
    main()
