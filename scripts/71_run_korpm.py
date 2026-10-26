#!/usr/bin/env python
"""Run KORPM (Martin-Hernandez et al. 2023) on the canonical benchmark tables.

Output: results/competitors/korpm_<bench>.csv  (KEYS + ddg_pred, positive = destabilising) and a timing json.
KORPM's own convention is positive = STABILISING (README; verified below against S661/Ssym), so ddg_pred = -korpm.
One korpm process per structure (potential file read once per process), parallel over structures.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comp_common import (BENCHES, KEYS, ROOT, abs_structure, load_bench, sanitise_pdb, structure_seq_map)  # noqa: E402

KDIR = ROOT / "data/raw/tools/korpm"
EXE = KDIR / "sbg/bin/korpm_gcc"
POT = KDIR / "pot/korp6Dv1.bin"
WORK = ROOT / "data/work/korpm"


def run_one(args):
    """args: (job_id, structure_path, chain_rows) -> list of (row_index, korpm_raw), seconds, n_skipped"""
    jid, spath, rows = args
    d = WORK / jid
    d.mkdir(parents=True, exist_ok=True)
    mp = sanitise_pdb(spath, d / "s.pdb")
    seq = structure_seq_map(d / "s.pdb")
    lines, idx = [], []
    skipped = 0
    for i, ch, rn, wt, mut in rows:
        new = mp.get((ch, rn))
        if new is None or seq.get((ch, str(new)), None) != wt:  # residue absent or WT letter mismatch
            skipped += 1
            continue
        lines.append(f"s {wt}{ch}{new}{mut}\n")
        idx.append(i)
    if not lines:
        return [], 0.0, skipped
    (d / "in.txt").write_text("".join(lines))
    t0 = time.time()
    r = subprocess.run([str(EXE), str(d / "in.txt"), "--dir", str(d), "--score_file", str(POT), "-o", str(d / "out.txt")],
                       capture_output=True, text=True, cwd=str(KDIR))
    dt = time.time() - t0
    # the program expects <dir>/<id>.pdb with id the first column -> name file "s"
    out = []
    try:
        vals = [l.split() for l in (d / "out.txt").read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        vals = []
    if len(vals) != len(idx):
        sys.stderr.write(f"WARN {jid}: {len(vals)} outputs for {len(idx)} inputs; stderr tail: {r.stderr[-300:]}\n")
        return [], dt, skipped + len(idx)
    for i, v in zip(idx, vals):
        out.append((i, float(v[2])))
    for f in ("s.pdb", "in.txt", "out.txt"):
        (d / f).unlink(missing_ok=True)
    d.rmdir()
    return out, dt, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=BENCHES)
    ap.add_argument("--domainome-n", type=int, default=0, help="subset of constructs (0 = all)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--outdir", default=str(ROOT / "results/competitors"))
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for b in a.bench:
        df = load_bench(b, a.domainome_n or None)
        t0 = time.time()
        jobs = []
        for k, (spath, g) in enumerate(df.groupby("structure", sort=False)):
            rows = [(i, c, r, w, m) for i, c, r, w, m in zip(g.index, g.chain, g.pdb_resnum, g.wt, g.mut)]
            jobs.append((f"{b}_{k}", str(abs_structure(spath)), rows))
        # largest jobs first
        jobs.sort(key=lambda j: -len(j[2]))
        raw = pd.Series(float("nan"), index=df.index)
        cpu, skipped = 0.0, 0
        with ProcessPoolExecutor(a.workers) as ex:
            for out, dt, sk in ex.map(run_one, jobs, chunksize=1):
                cpu += dt
                skipped += sk
                for i, v in out:
                    raw[i] = v
        wall = time.time() - t0
        res = df[KEYS].copy()
        res["korpm_raw"] = raw
        res["ddg_pred"] = -raw  # KORPM: positive = stabilising -> flip to positive = destabilising
        for c in ("pair_id", "direction"):
            if c in df:
                res[c] = df[c]
        res.to_csv(Path(a.outdir) / f"korpm_{b}.csv", index=False)
        info = dict(bench=b, n=len(df), n_pred=int(raw.notna().sum()), n_skipped=skipped, n_structures=len(jobs),
                    wall_s=round(wall, 1), korpm_cpu_s=round(cpu, 1), korpm_ms_per_mut_cpu=round(1000 * cpu / max(1, raw.notna().sum()), 2),
                    workers=a.workers)
        print(json.dumps(info), flush=True)
        with open(Path(a.outdir) / "korpm_timing.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
