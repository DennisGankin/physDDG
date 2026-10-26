#!/usr/bin/env python
"""Run DDGun (sequence) and DDGun3D (Montanucci et al. 2019) on the canonical benchmark tables.

One worker process (scripts/72w_ddgun_worker.py in .venv_ddgun) per structure; hhblits (4 threads each) against
UniClust30 2018_08. Writes results/competitors/ddgun3d_<bench>.csv and ddgun_<bench>.csv (KEYS + ddg_pred,
positive = destabilising; DDGun's own sign is positive = stabilising, so ddg_pred = -raw) + timing jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comp_common import BENCHES, KEYS, ROOT, abs_structure, load_bench  # noqa: E402

PY = ROOT / ".venv_ddgun/bin/python"
WORKER = ROOT / "scripts/72w_ddgun_worker.py"
DB = ROOT / "data/raw/tools/uniclust30_2018_08/uniclust30_2018_08/uniclust30_2018_08"
SCR = ROOT / "data/work/ddgun"
CACHE = SCR / "hhblits_cache"


def run_job(job):
    jid, spath, chain, rows = job
    d = SCR / jid
    d.mkdir(parents=True, exist_ok=True)
    muts = sorted({f"{w}{r}{m}" for _, _, r, w, m in rows})
    (d / "muts.txt").write_text("\n".join(muts) + "\n")
    out = d / "out.csv"
    t0 = time.time()
    p = subprocess.run([str(PY), str(WORKER), spath, chain, str(d / "muts.txt"), str(out), str(d / "w"), str(CACHE), str(DB)],
                       capture_output=True, text=True)
    wall = time.time() - t0
    if not out.exists():
        sys.stderr.write(f"FAIL {jid} {spath} {chain}: {p.stderr[-400:]}\n")
        return jid, None, None, wall
    res = pd.read_csv(out).set_index("mut")
    tm = json.load(open(str(out) + ".time.json"))
    return jid, res, tm, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=BENCHES)
    ap.add_argument("--domainome-n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=15, help="concurrent structures (each hhblits uses 4 threads)")
    ap.add_argument("--outdir", default=str(ROOT / "results/competitors"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N structures")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for b in a.bench:
        df = load_bench(b, a.domainome_n if b == "domainome" else None)
        df = df[df.structure.notna()]
        jobs = []
        for k, ((spath, chain), g) in enumerate(df.groupby(["structure", "chain"], sort=False)):
            rows = [(i, c, r, w, m) for i, c, r, w, m in zip(g.index, g.chain, g.pdb_resnum, g.wt, g.mut)]
            jobs.append((f"{b}_{k}", str(abs_structure(spath)), chain, rows))
        if a.limit:
            jobs = jobs[: a.limit]
        t0 = time.time()
        raw3, raws = pd.Series(float("nan"), index=df.index), pd.Series(float("nan"), index=df.index)
        tot = dict(t_hhblits_s=0.0, t_score_s=0.0, n_hhblits=0, n_failed=0)
        with ThreadPoolExecutor(a.jobs) as ex:
            for (jid, res, tm, wall), job in zip(ex.map(run_job, jobs), jobs):
                if res is None:
                    tot["n_failed"] += 1
                    continue
                tot["t_score_s"] += tm["t_score_s"]
                if not tm["hhblits_cached"]:
                    tot["t_hhblits_s"] += tm["t_hhblits_s"]
                    tot["n_hhblits"] += 1
                for i, c, r, w, m in job[3]:
                    key = f"{w}{r}{m}"
                    if key in res.index:
                        raw3[i] = res.at[key, "ddgun3d_raw"]
                        raws[i] = res.at[key, "ddgun_seq_raw"]
                import shutil
                shutil.rmtree(SCR / jid, ignore_errors=True)
        wall = time.time() - t0
        sel = df.index if not a.limit else df.index[raw3.notna() | raws.notna()]
        for name, raw in (("ddgun3d", raw3), ("ddgun", raws)):
            res = df.loc[sel, KEYS].copy()
            res[f"{name}_raw"] = raw[sel]
            res["ddg_pred"] = -raw[sel]  # DDGun: positive = stabilising
            for c in ("pair_id", "direction"):
                if c in df:
                    res[c] = df.loc[sel, c]
            res.to_csv(Path(a.outdir) / f"{name}_{b}.csv", index=False)
        n_ok = int(raw3.notna().sum())
        info = dict(bench=b, n=len(df), n_pred=n_ok, n_structures=len(jobs), wall_s=round(wall, 1), jobs=a.jobs,
                    **{k: round(v, 1) if isinstance(v, float) else v for k, v in tot.items()},
                    ms_per_mut_scoring=round(1000 * tot["t_score_s"] / max(1, n_ok), 2),
                    s_per_hhblits_search_4threads=round(tot["t_hhblits_s"] / max(1, tot["n_hhblits"]), 1))
        print(json.dumps(info), flush=True)
        with open(Path(a.outdir) / "ddgun_timing.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
