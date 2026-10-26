#!/usr/bin/env python
"""Worker for scripts/10a_evo_profiles.py (run with .venv_ddgun/bin/python: python 3.10 + DDGun's own PDB parser).

Two modes, both using the DDGun code path so the sequence (and hence the cache key) is identical to the one used by
scripts/72w_ddgun_worker.py:
  parse  UNITS_TSV OUT_JSONL     for every (structure, chain): DDGun chain sequence, residue numbers (same order),
                                 sha1(sequence) = key of data/work/ddgun/hhblits_cache/<key>.{blast,hssp}
  search SEQ KEY CACHEDIR DB      hhblits exactly as ddgun_3d.run_3d_pipeline (-n 2 -cpu 4 -e 1e-6, -opsi output,
                                 converted with ddgun's ali2fasta; then ali2prof.py -> .hssp so the DDGun cache stays
                                 complete). Writes <key>.blast / <key>.hssp atomically + <key>.search.json (timing).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

TOOL = "/workspace/stability_hackathon/data/raw/tools/ddgun"
sys.path.insert(0, TOOL)
sys.path.insert(0, TOOL + "/tools")
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]  # ali2prof.py uses `env python`
import ddgun_3d as D  # noqa: E402


def parse(units_tsv, out_jsonl):
    with open(out_jsonl, "w") as fo:
        for line in open(units_tsv):
            pdb, chain = line.rstrip("\n").split("\t")
            try:
                och = D.get_pdbchain(pdb, chain, D.at)
                seq = och.getSequence()
                nums = [str(x) for x in och.getPDBNumbers()]
                rec = dict(structure_abs=pdb, chain=chain, seq=seq, nums=nums, key=hashlib.sha1(seq.encode()).hexdigest())
            except Exception as e:  # noqa: BLE001
                rec = dict(structure_abs=pdb, chain=chain, seq="", nums=[], key="", error=repr(e)[:300])
            fo.write(json.dumps(rec) + "\n")


def search(seq, key, cache, db):
    if os.path.exists(f"{cache}/{key}.blast") and os.path.exists(f"{cache}/{key}.hssp"):
        return
    work = tempfile.mkdtemp(prefix=f"evo_{key[:8]}_", dir=os.path.join(cache, "..", "evo_tmp"))
    seqfile, blast = f"{work}/q.fasta", f"{work}/q.blast"
    with open(seqfile, "w") as f:
        f.write(f">{key}\n{seq}")
    t0 = time.time()
    cmd = [D.pblast, "-d", db, "-i", seqfile, "-opsi", blast + "x", "-n", "2", "-cpu", "4", "-e", "1e-06"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write("HHBLITS_ERROR " + p.stderr[-500:])
        sys.exit(4)
    D.ali2fasta(blast + "x", blast)
    t1 = time.time()
    p = subprocess.run([D.pprof, seqfile, blast, f"{work}/q.hssp"], capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write("HSSP_ERROR " + p.stderr[-500:])
        sys.exit(5)
    for ext in ("hssp", "blast"):   # hssp first: the cache counts as complete only when .blast exists too
        shutil.copy(f"{work}/q.{ext}", f"{cache}/{key}.{ext}.tmp{os.getpid()}")
        os.replace(f"{cache}/{key}.{ext}.tmp{os.getpid()}", f"{cache}/{key}.{ext}")
    json.dump(dict(t_hhblits_s=round(t1 - t0, 2), seqlen=len(seq)), open(f"{cache}/{key}.search.json", "w"))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "parse":
        parse(*sys.argv[2:4])
    else:
        search(*sys.argv[2:6])
