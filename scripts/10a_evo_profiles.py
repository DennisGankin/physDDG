#!/usr/bin/env python
"""Evolution-aware extension (optional mode 'PhysDDG+evo'), step 1: hhblits sequence profiles for every benchmark chain.

Same search as DDGun (Montanucci et al. 2019; scripts/72_run_ddgun.py): hhblits v3.3.0 vs UniClust30 2018_08,
-n 2 -e 1e-6 -cpu 4, query = the chain sequence of the structure file used for the benchmark (DDGun's parser), results
cached by sha1(sequence) in data/work/ddgun/hhblits_cache/<key>.blast (query-anchored MSA, insertions removed, the
hhblits result MSA filtered by its defaults -id 90 -diff 1000). Existing cache entries (written by the DDGun runs)
are reused unchanged; missing ones are searched with up to --jobs concurrent hhblits (4 threads each).

Units: every distinct (structure, chain) in megascale (all splits + censored rows), fireprot, s669, s461, ssym (incl.
reverse/mutant structures), s2648, domainome (construct AF crops), abundance_dms.
Output: data/features/evo_units.parquet (structure, chain, seq, nums (json list), key, datasets) -- git-ignored.
  PYTHONPATH=. .venv/bin/python scripts/10a_evo_profiles.py [--jobs 15]
"""
import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"          # the parse/search workers need no BLAS threads (RLIMIT_NPROC is shared)
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from physddg import eval as E  # noqa: E402

PY = ROOT / ".venv_ddgun/bin/python"
WORKER = ROOT / "scripts/10aw_evo_worker.py"
DB = ROOT / "data/raw/tools/uniclust30_2018_08/uniclust30_2018_08/uniclust30_2018_08"
CACHE = ROOT / "data/work/ddgun/hhblits_cache"
WORK = ROOT / "data/work/evo"
BENCHES = ["megascale", "fireprot", "s669", "s461", "ssym", "s2648", "domainome", "abundance_dms"]


def abs_structure(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def units() -> pd.DataFrame:
    rows = []
    for b in BENCHES:
        d = E.load_benchmark(b)
        if b == "megascale":
            c = pd.read_csv(ROOT / "data/processed/megascale_censored.csv", low_memory=False, usecols=["structure", "chain"])
            d = pd.concat([d[["structure", "chain"]], c], ignore_index=True)
        u = d[d.structure.notna()][["structure", "chain"]].drop_duplicates()
        u["dataset"] = b
        rows.append(u)
    u = pd.concat(rows, ignore_index=True)
    u["chain"] = u["chain"].astype(str)
    return u.groupby(["structure", "chain"], as_index=False).dataset.agg(lambda s: ",".join(sorted(set(s))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=15)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    (CACHE.parent / "evo_tmp").mkdir(parents=True, exist_ok=True)
    u = units()
    u["structure_abs"] = [str(abs_structure(s)) for s in u.structure]
    tsv = WORK / "units.tsv"
    u[["structure_abs", "chain"]].to_csv(tsv, sep="\t", header=False, index=False)
    out = WORK / "units_parsed.jsonl"
    t0 = time.time()
    # parse in 16 shards
    sh = 16
    lines = tsv.read_text().splitlines()
    procs = []
    for k in range(sh):
        part = WORK / f"units_{k}.tsv"
        part.write_text("\n".join(lines[k::sh]) + "\n")
        procs.append(subprocess.Popen([str(PY), str(WORKER), "parse", str(part), str(WORK / f"parsed_{k}.jsonl")]))
    for p in procs:
        p.wait()
    recs = []
    for k in range(sh):
        recs += [json.loads(x) for x in (WORK / f"parsed_{k}.jsonl").read_text().splitlines() if x.strip()]
        (WORK / f"units_{k}.tsv").unlink()
        (WORK / f"parsed_{k}.jsonl").unlink()
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    p = pd.DataFrame(recs)
    u = u.merge(p, on=["structure_abs", "chain"], how="left")
    print(f"parsed {len(u)} units in {time.time() - t0:.0f}s; errors: {(u.key == '').sum()}", flush=True)
    seqs = u[u.key != ""].drop_duplicates("key")[["key", "seq"]]
    have = {x.name.split(".")[0] for x in CACHE.glob("*.blast")}
    todo = seqs[~seqs.key.isin(have)].sort_values("seq", key=lambda s: -s.str.len())
    print(f"{len(seqs)} distinct sequences, {len(seqs) - len(todo)} cached, {len(todo)} to search", flush=True)

    def run(kv):
        key, seq = kv
        t = time.time()
        p = subprocess.run([str(PY), str(WORKER), "search", seq, key, str(CACHE), str(DB)], capture_output=True, text=True)
        ok = (CACHE / f"{key}.blast").exists()
        if not ok:
            sys.stderr.write(f"FAIL {key} len={len(seq)}: {p.stderr[-300:]}\n")
        return key, ok, time.time() - t

    done = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for key, ok, dt in ex.map(run, list(zip(todo.key, todo.seq))):
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  searched {done}/{len(todo)}  last {dt:.0f}s  elapsed {time.time() - t0:.0f}s", flush=True)
    u["nums"] = u["nums"].map(json.dumps)
    u["has_msa"] = u["key"].map(lambda k: (CACHE / f"{k}.blast").exists() if k else False)
    u.drop(columns=["structure_abs"]).to_parquet(ROOT / "data/features/evo_units.parquet", index=False)
    print("units with MSA:", u.has_msa.mean(), flush=True)


if __name__ == "__main__":
    main()
