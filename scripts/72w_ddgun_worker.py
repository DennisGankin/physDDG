#!/usr/bin/env python
"""DDGun / DDGun3D worker for ONE structure chain (run with .venv_ddgun/bin/python, python 3.10 + biopython 1.79).

Imports the original ddgun_3d.py (data/raw/tools/ddgun) and uses its own pipeline functions (DSSP, hhblits vs UniClust30
2018_08 -> profile, features). Differences from the command-line tool, all for speed/precision only:
  * hhblits/HSSP results are cached by sequence hash (identical sequences are searched once);
  * residue environments are cached per site (the original recomputes them for each of the 19 mutants);
  * predictions are recomputed from the unrounded features (the CLI prints 1 decimal), using the CLI's formulas:
      DDGun3D = (0.18*KD + 0.20*BL - 0.29*Skolnick - 0.33*BV) * (1.1 - RSA/100)
      DDGun   = 0.27*KD + 0.30*BL - 0.43*Skolnick
    DDGun's own sign convention: positive = STABILISING ("Increase"); the caller flips it.
usage: 72w_ddgun_worker.py PDB CHAIN MUTFILE OUTCSV WORKDIR CACHEDIR DB
"""
import hashlib
import json
import os
import shutil
import sys
import time

TOOL = "/workspace/stability_hackathon/data/raw/tools/ddgun"
sys.path.insert(0, TOOL)
sys.path.insert(0, TOOL + "/tools")
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ["PATH"]  # ali2prof.py uses `env python`
import ddgun_3d as D  # noqa: E402


def cached_env(pdbfile, chain, l_mut, d=[0.0, 5.0], atoms=D.at):
    """Same as ddgun_3d.get_env_residue but each (wt,pos) site is computed once."""
    lres_env, lnres, site = {}, {}, {}
    och = D.pdb.readPDB(pdbfile, chain, atoms)
    lres = och.getPDBNumbers()
    for i, v_mut in l_mut.items():
        v_dat, v_res = [], []
        for mut in v_mut:
            wt, pos, new = mut[0], mut[1:-1], mut[-1]
            if (wt, pos) not in site:
                r_i = och.getResidueByNumber(pos)
                if not r_i or r_i.NameOne != wt:
                    site[(wt, pos)] = None
                else:
                    env = []
                    for j in lres:
                        if j == pos:
                            continue
                        dist = r_i.residueDistance(och.getResidueByNumber(j))
                        if d[0] <= dist <= d[1]:
                            env.append((och.getResidueByNumber(j).NameOne, j))
                    site[(wt, pos)] = env
            env = site[(wt, pos)]
            if env is None:
                sys.stderr.write(f"WARNING: Incorrect residue {mut[:-1]}\n")
                v_dat, v_res = [], []
                break
            v_res.append(wt + str(lres.index(pos) + 1) + new)
            v_dat.append(env)
        if v_dat:
            lres_env[i] = v_dat
        if v_res:
            lnres[i] = v_res
    return lres, lnres, lres_env


D.get_env_residue = cached_env


def main():
    pdb, chain, mutfile, outcsv, work, cache, db = sys.argv[1:8]
    os.makedirs(work, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    muts = [l.strip() for l in open(mutfile) if l.strip()]
    l_mut = {m: [m] for m in muts}
    t0 = time.time()
    och = D.get_pdbchain(pdb, chain, D.at)
    seq = och.getSequence()
    key = hashlib.sha1(seq.encode()).hexdigest()
    name = os.path.abspath(pdb).split("/")[-1]
    chainfile = f"{work}/{name}.{chain}"
    cached = os.path.exists(f"{cache}/{key}.hssp") and os.path.exists(f"{cache}/{key}.blast")
    if cached:
        shutil.copy(f"{cache}/{key}.hssp", chainfile + ".hssp")
        shutil.copy(f"{cache}/{key}.blast", chainfile + ".blast")
    t1 = time.time()
    chainfile, dsspfile, hsspfile = D.run_3d_pipeline(pdb, chain, D.pblast, db, 1.0e-6, work)
    t2 = time.time()
    if not cached:
        for ext in ("hssp", "blast"):
            shutil.copy(f"{chainfile}.{ext}", f"{cache}/{key}.{ext}.tmp{os.getpid()}")
            os.replace(f"{cache}/{key}.{ext}.tmp{os.getpid()}", f"{cache}/{key}.{ext}")
    pots = ["KYTJ820101", "HENS920102", "SKOJ970101", "BASU010101"]
    l_data, l_hssp, lres_env = D.get_muts_score(chainfile, chain, dsspfile, hsspfile, l_mut, pots, [0.0, 5.0], 2, work)
    t3 = time.time()
    rows = ["mut,s_kd,s_bl,s_prof,s_3d,rsa,ddgun3d_raw,ddgun_seq_raw"]
    for m, v in l_data.items():
        vm = v[0]
        p3 = (0.18 * vm[0] + 0.20 * vm[1] - 0.29 * vm[2] - 0.33 * vm[3]) * (1.1 - vm[5] * 0.01)
        ps = 0.27 * vm[0] + 0.30 * vm[1] - 0.43 * vm[2]
        rows.append(f"{m},{vm[0]:.6f},{vm[1]:.6f},{vm[2]:.6f},{vm[3]:.6f},{vm[5]:.3f},{p3:.6f},{ps:.6f}")
    open(outcsv, "w").write("\n".join(rows) + "\n")
    json.dump(dict(n_mut=len(muts), n_pred=len(l_data), hhblits_cached=cached, t_hhblits_s=round(t2 - t1, 2),
                   t_score_s=round(t3 - t2, 2), seqlen=len(seq)), open(outcsv + ".time.json", "w"))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
