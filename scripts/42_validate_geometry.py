"""Validate the geometry engine on held-out PISCES 'test' chains (not used for the rotamer library).

1. Builder: rebuild every well-ordered native side chain from native backbone + native chis with ideal geometry;
   heavy-atom RMSD (CB onwards; min over exact-symmetry name swaps).
2. Rotamer library coverage: fraction of native side chains with a candidate within 30 deg on all chis
   (2-fold symmetric terminal chis compared modulo 180) among the top-R (R = 1, 3, 5, 10, 20, all) and for
   cumulative-probability cut-offs, with / without +-1 sd chi1/chi2 expansion.
3. Builder throughput on GPU (positions x rotamers per second).
Outputs: results/geometry/validation.json, results/geometry/coverage_per_aa.csv, results/geometry/rmsd_per_aa.csv
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/42_validate_geometry.py [n_chains=all]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import pisces_io  # noqa: E402
from physddg import topology as T  # noqa: E402
from physddg.structure import backbone_dihedrals, load_protein, side_chain_chis  # noqa: E402

OUT = ROOT / "results" / "geometry"


def load_chain(job):
    pdb, chain = job
    try:
        p = load_protein(pisces_io.structure_path(pdb), chains=[chain], keep_het=False)
    except Exception:  # noqa: BLE001
        return None
    bd = backbone_dihedrals(p)
    nat = np.array([len(T.TEMPLATES[a].atoms) for a in p.seq])
    bsc = np.array([p.b14[i, 4:nat[i]].max(initial=0) for i in range(len(p))])
    good = (~p.altloc) & (~p.missing_sc) & (bsc < 40) & (p.b14[:, :4].max(1) < 40) & (~p.modified) \
        & (p.occ14[:, :4].min(1) >= 0.99)
    return {"aa": p.aa, "seq": p.seq, "xyz": p.xyz14, "mask": p.mask14, "phi": bd["phi"], "psi": bd["psi"],
            "chi": side_chain_chis(p), "good": good, "id": f"{pdb}{chain}"}


def ang_diff(a, b, period=360.0):
    d = np.abs((a - b + period / 2) % period - period / 2)
    return d


def builder_rmsd(chains, device, native_cb=False):
    import torch
    from physddg.builder import build_sidechains
    rows = []
    for c in chains:
        sel = c["good"] & (c["aa"] != T.AA_INDEX["G"])
        if not sel.any():
            continue
        bb = torch.tensor(c["xyz"][sel, :4], device=device, dtype=torch.float64)
        chi = torch.tensor(np.nan_to_num(c["chi"][sel]), device=device, dtype=torch.float64)[:, None]
        cb = torch.tensor(c["xyz"][sel, 4], device=device, dtype=torch.float64) if native_cb else None
        X, m = build_sidechains(bb, torch.tensor(c["aa"][sel], device=device), chi, cb=cb)
        X, m = X[:, 0].cpu().numpy(), m[:, 0].cpu().numpy()
        ref = c["xyz"]
        for k, i in enumerate(np.flatnonzero(sel)):
            a = c["seq"][i]
            t = T.TEMPLATES[a]
            idx = np.arange(4, len(t.atoms))
            best = np.inf
            swaps = [[]] + [T.SYMMETRIC_SWAPS.get(a, [])]
            for sw in swaps:
                perm = np.arange(len(t.atoms))
                for x, y in sw:
                    perm[t.index(x)], perm[t.index(y)] = t.index(y), t.index(x)
                best = min(best, float(np.sqrt(np.mean(np.sum((X[k, perm[idx]] - ref[i, idx]) ** 2, -1)))))
            rows.append((c["id"], a, best))
    return pd.DataFrame(rows, columns=["chain", "aa", "rmsd"])


def coverage(chains, lib):
    recs = []
    for a in T.AA1:
        nchi = T.N_CHI[a]
        if nchi == 0:
            continue
        phi, psi, chi = [], [], []
        for c in chains:
            s = c["good"] & (c["aa"] == T.AA_INDEX[a]) & np.isfinite(c["phi"]) & np.isfinite(c["psi"])
            phi.append(c["phi"][s]); psi.append(c["psi"][s]); chi.append(c["chi"][s])
        phi, psi, chi = np.concatenate(phi), np.concatenate(psi), np.concatenate(chi)
        if len(phi) == 0:
            continue
        per = np.array([180.0 if (a, k + 1) in T.CHI_SYMMETRIC else 360.0 for k in range(nchi)])
        for expand in (False, True):
            q = lib.query(a, phi, psi, expand=expand)
            d = ang_diff(q["chis"][:, :, :nchi], chi[:, None, :nchi], per)       # [B, R', nchi]
            hit = (d <= 30.0).all(-1) & q["mask"]
            E = 9 if (expand and nchi >= 2) else (3 if expand else 1)
            hit_base = hit.reshape(len(phi), -1, E).any(-1)                    # [B, R] per base rotamer
            first = np.where(hit_base.any(1), hit_base.argmax(1), 10 ** 6)
            pr = np.exp(q["logp"].reshape(len(phi), -1, E)[:, :, 0])                 # base rotamer probs (sorted)
            cum_before = np.cumsum(pr, 1) - pr
            tot = pr.sum(1, keepdims=True)
            rec = {"aa": a, "expand": expand, "n": len(phi), "n_rot_lib": lib.n_rot(a)}
            for R in (1, 3, 5, 10, 20):
                rec[f"top{R}"] = float((first < R).mean())
            rec["all"] = float((first < 10 ** 6).mean())
            for cum in (0.8, 0.9, 0.95, 0.99):
                kept = cum_before < cum * tot
                ok = first < 10 ** 6
                rec[f"cum{cum}"] = float(kept[np.flatnonzero(ok), first[ok]].sum() / len(phi))
                rec[f"cum{cum}_nrot"] = float(kept.sum(1).mean())
            recs.append(rec)
    return pd.DataFrame(recs)


def throughput(device, B=2000, R=40, reps=20):
    import torch
    from physddg.builder import build_sidechains
    g = torch.Generator(device="cpu").manual_seed(0)
    bb = torch.tensor([[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0], [2.15, -1.0, 0.3]], dtype=torch.float32)
    bb = (bb[None] + torch.randn(B, 1, 3, generator=g) * 10).to(device)
    out = {}
    for name, aa in (("mixed20", torch.randint(0, 20, (B, R), generator=g).to(device)), ("trp", T.AA_INDEX["W"]),
                     ("leu", T.AA_INDEX["L"])):
        chis = (torch.rand(B, R, 4, generator=g) * 360 - 180).to(device)
        for _ in range(3):
            build_sidechains(bb, aa, chis)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            build_sidechains(bb, aa, chis)
        torch.cuda.synchronize()
        out[name] = B * R * reps / (time.time() - t0)
    return out


def main(n_chains: int | None = None):
    import torch
    from physddg.rotlib import RotamerLibrary
    rows = [r for r in pisces_io.read_list(pisces_io.download()) if pisces_io.split_of(r["pdb"]) == "test"]
    rows = rows[:n_chains] if n_chains else rows
    with ProcessPoolExecutor(32) as ex:
        chains = [c for c in ex.map(load_chain, [(r["pdb"], r["chain"]) for r in rows], chunksize=4) if c]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    rm = builder_rmsd(chains, dev)
    rm["rmsd_native_cb"] = builder_rmsd(chains, dev, native_cb=True).rmsd.to_numpy()
    per_aa = rm.groupby("aa")[["rmsd", "rmsd_native_cb"]].describe(percentiles=[0.5, 0.95])
    per_aa = per_aa.loc[:, (slice(None), ["count", "50%", "mean", "95%", "max"])]
    per_aa.round(4).to_csv(OUT / "rmsd_per_aa.csv")
    cov = coverage(chains, RotamerLibrary())
    cov.round(4).to_csv(OUT / "coverage_per_aa.csv", index=False)
    w = cov.n / cov.groupby("expand").n.transform("sum")
    pooled = {e: {k: float((g[k] * w[g.index]).sum()) for k in g.columns if k.startswith(("top", "all", "cum"))}
              for e, g in cov.groupby("expand")}
    tp = throughput(dev) if dev == "cuda" else {}
    res = {"n_test_chains": len(chains), "n_sidechains": int(len(rm)),
           "rmsd_median": float(rm.rmsd.median()), "rmsd_mean": float(rm.rmsd.mean()),
           "rmsd_p95": float(rm.rmsd.quantile(0.95)), "rmsd_frac_lt_0.15": float((rm.rmsd < 0.15).mean()),
           "rmsd_native_cb_median": float(rm.rmsd_native_cb.median()),
           "rmsd_native_cb_p95": float(rm.rmsd_native_cb.quantile(0.95)),
           "coverage_pooled_noexpand": pooled[False], "coverage_pooled_expand": pooled[True],
           "builder_residues_per_s": tp}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(json.dumps(res, indent=1))
    print(per_aa.round(3))
    print(cov.round(3).to_string())


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
