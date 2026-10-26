#!/usr/bin/env python
"""End-to-end speed benchmark of the PhysDDG predictor (physddg.predict), wall-clock, including structure loading,
annotation, engine features, structure-statistics features and scoring.

(a) Full saturation scans (19 x L variants) of four single-chain X-ray structures (~60, 150, 300, 600 residues;
    data/structures/bench/, extracted from PISCES test entries by this script) for stage 1 (B4) and stage 2 (B5):
    1 GPU in-process (model loaded, CUDA warm; also the cold CLI start overhead) and N GPUs (sites sharded over
    one process per GPU, includes process spawn + CUDA init).
(b) Whole benchmark tables through run_jobs: Megascale-test (28k variants) and Domainome (536k), stage 1 / 2.
Output: results/mutate/speed_benchmark.csv (+ .md table).
Usage: PYTHONPATH=. python scripts/59_speed_benchmark.py [--gpus 0 1 2 3] [--skip-tables] [--domainome-stage2]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
BENCH = ROOT / "data" / "structures" / "bench"
PROTS = {"60": "7a38A", "150": "1zd0A", "300": "6q62A", "600": "3edfA"}
AF_PROTS = {"150": "/workspace/data/yeast_af/pdbs/P20486.pdb", "300": "/workspace/data/yeast_af/pdbs/P00546.pdb",
            "600": "/workspace/data/yeast_af/pdbs/P06775.pdb"}          # AlphaFold DB models (B-factor = pLDDT)
OUT = ROOT / "results" / "mutate"


def prepare():
    import gemmi
    import pisces_io as P
    BENCH.mkdir(parents=True, exist_ok=True)
    for pid in PROTS.values():
        f = BENCH / f"{pid}.pdb"
        if f.exists():
            continue
        st = gemmi.read_structure(str(P.structure_path(pid[:4])))
        st.remove_ligands_and_waters()
        st.remove_hydrogens()
        for i in range(len(st) - 1, 0, -1):
            del st[i]
        for n in [ch.name for ch in st[0]]:
            if n != pid[4:]:
                st[0].remove_chain(n)
        st.write_pdb(str(f))


def scans(gpus: list[int]) -> list[dict]:
    import torch
    from physddg.predict import PhysDDG, run_jobs, shard_sites
    rows = []
    for stage in (1, 2):
        m = PhysDDG(stage=stage, device=f"cuda:{gpus[0]}")
        m.predict(str(BENCH / f"{PROTS['60']}.pdb"), scan_all=True)            # warm-up
        for size, pid in PROTS.items():
            f = str(BENCH / f"{pid}.pdb")
            torch.cuda.synchronize()
            t0 = time.time()
            o = m.predict(f, scan_all=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rows.append(dict(test=f"scan_{size}", structure=pid, L=o[["chain", "pdb_resnum"]].drop_duplicates().shape[0],
                             stage=stage, n_gpu=1, variants=len(o), wall_s=dt, variants_per_s=len(o) / dt,
                             engine_s=m.timing["engine_s"], cpu_s=m.timing["load_s"] + m.timing["simple_s"]))
            print(rows[-1], flush=True)
            if len(gpus) > 1:
                prot = m.load(f)
                r = m._rows(prot, f, None, None, True, None)
                t0 = time.time()
                o = run_jobs(shard_sites(f, r, len(gpus)), gpus, None, stage, "xtal", 1)
                dt = time.time() - t0
                rows.append(dict(test=f"scan_{size}", structure=pid, L=rows[-1]["L"], stage=stage, n_gpu=len(gpus),
                                 variants=len(o), wall_s=dt, variants_per_s=len(o) / dt))
                print(rows[-1], flush=True)
    # AF mode (D18: trim + stage 2 + pLDDT attenuation + flags), 1 GPU
    m = PhysDDG(mode="af", device=f"cuda:{gpus[0]}")
    m.predict(AF_PROTS["150"], scan_all=True, sites="1-20")
    for size, f in AF_PROTS.items():
        torch.cuda.synchronize()
        t0 = time.time()
        o = m.predict(f, scan_all=True)
        dt = time.time() - t0
        rows.append(dict(test=f"af_scan_{size}", structure=Path(f).stem, L=o[["chain", "pdb_resnum"]].drop_duplicates().shape[0],
                         stage=2, mode="af", n_gpu=1, variants=len(o), wall_s=dt, variants_per_s=len(o) / dt,
                         withheld_frac=float(o.withheld.mean())))
        print(rows[-1], flush=True)
    # cold CLI start (python + torch import + CUDA init + model load), 60-residue protein, 3 mutations
    t0 = time.time()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "physddg"), "predict", "--structure",
                    str(BENCH / f"{PROTS['60']}.pdb"), "--mutations", "A90G", "--out", os.devnull],
                   check=False, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0])})
    rows.append(dict(test="cli_cold_start_3_mutations", stage=1, n_gpu=1, wall_s=time.time() - t0))
    return rows


def table(name: str, stage: int, gpus: list[int], cpus: int, procs_per_gpu: int = 1, mode: str = "xtal") -> dict:
    from physddg import features_simple as FS
    from physddg.predict import run_jobs
    r = FS.dataset_rows(name)
    if name == "megascale":
        ms = pd.read_parquet(ROOT / "data/processed/megascale.parquet", columns=["protein_id", "chain", "pdb_resnum",
                                                                                "wt", "mut", "split"])
        r = r.merge(ms[ms.split == "test"].drop(columns="split").drop_duplicates(), on=FS.KEYS)
    r = r.assign(pdb_resnum=r.pdb_resnum.astype(str), chain=r.chain.astype(str)).drop_duplicates(FS.KEYS + ["structure"])
    r["mutation"] = r.wt + r.pdb_resnum + r.mut
    jobs = [(s, g.drop(columns="structure").reset_index(drop=True)) for s, g in r.groupby("structure", sort=False)]
    gl = [g for g in gpus for _ in range(procs_per_gpu)]
    t0 = time.time()
    o = run_jobs(jobs, gl, None, stage, mode, cpus)
    dt = time.time() - t0
    res = dict(test=f"table_{name}" + ("_test" if name == "megascale" else ""), stage=stage, mode=mode, n_gpu=len(gpus),
               procs_per_gpu=procs_per_gpu, structures=len(jobs), variants=len(o), wall_s=dt, variants_per_s=len(o) / dt,
               failed=int(o.ddg.isna().sum()))
    print(res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--cpus", type=int, default=6)
    ap.add_argument("--skip-scans", action="store_true")
    ap.add_argument("--skip-tables", action="store_true")
    ap.add_argument("--domainome-stage2", action="store_true")
    a = ap.parse_args()
    prepare()
    rows = []
    if not a.skip_scans:
        rows += scans(a.gpus)
    if not a.skip_tables:
        rows.append(table("megascale", 1, a.gpus, a.cpus))
        rows.append(table("megascale", 2, a.gpus, a.cpus))
        rows.append(table("domainome", 1, a.gpus, a.cpus, procs_per_gpu=2))
        if a.domainome_stage2:
            rows.append(table("domainome", 2, a.gpus, a.cpus, procs_per_gpu=2))
            rows.append(table("domainome", 2, a.gpus, a.cpus, procs_per_gpu=2, mode="af"))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    from physddg.predict import default_weights
    df["weights"] = [Path(str(default_weights(int(s), m if isinstance(m, str) else "xtal"))).name if s == s else ""
                     for s, m in zip(df.stage, df.get("mode", pd.Series(["xtal"] * len(df))))]
    df.round(3).to_csv(OUT / "speed_benchmark.csv", index=False)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
