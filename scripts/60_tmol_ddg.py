#!/usr/bin/env python
"""tmol (beta_nov2016 energy, Dunbrack packer, GPU) single-mutant ddG baseline + per-term dE features.

Runs in the SEPARATE venv .venv_tmol (requirements_tmol.txt), not the main .venv:
  CUDA_VISIBLE_DEVICES=3 .venv_tmol/bin/python scripts/60_tmol_ddg.py --dataset megascale --split test \
      [--proteins 1GYZ 1PSE | --n-proteins 5] [--min] [--seeds 1] [--shell 8] [--chunk 512] [--tag ...]
Output: results/tmol/<dataset>[_<tag>].parquet with KEYS (protein_id, chain, pdb_resnum, wt, mut) +
  ddg_reu       E(mut) - E(WT) total, tmol score units (~REU)
  ddg_pred      ddg_reu / 2.94 (Rosetta REU -> kcal/mol convention; positive = destabilising, like FoldX)
  d_<term>      weighted per-term dE (sum over terms = ddg_reu), terms of beta2016.sfxn that are non-zero
  status        ok | <error>
Protocol (identical for WT reference and every mutant at a site):
  * input chain, protein residues only (HETATM / water / ligands dropped), all-atom pose built by tmol (adds H).
  * shell = residues with any heavy atom within --shell A of any heavy atom of the WT site residue
    (Gly site: N, CA, C, O + virtual CB). Same shell for WT and all 19 mutants at that site.
  * PackerTask: everything outside the shell frozen; shell residues repack (identity fixed, Dunbrack rotamers +
    current rotamer); site residue: identity = WT (reference) or mut, Dunbrack rotamers only (no current rotamer,
    DESIGN D6 symmetric modelling). Simulated annealing with fixed seed(s); with --seeds K the min-energy result
    over K seeds is kept per pose (WT and mutant alike).
  * --min: then kinematic minimisation of side-chain torsions of all shell residues (LBFGS-Armijo, max 50 it).
  * score whole pose with beta2016 (weighted per term), ddG = E(mut) - E(WT ref of that site).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PDB_MIRROR_PATH", "")  # silence atomworks env warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]
AA1 = "ACDEFGHIKLMNPQRSTVWY"
ONE2THREE = dict(zip(AA1, "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
THREE2ONE = {v: k for k, v in ONE2THREE.items()}
REU_PER_KCAL = 2.94


def norm_resnum(s):
    return s.astype(str).str.strip().str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def load_table(dataset, split, table=None):
    base = ROOT / "data" / "processed"
    p = Path(table) if table else base / f"{dataset}.parquet"
    df = (pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)) if p.exists() \
        else pd.read_csv(base / f"{dataset}.csv")
    if split:
        df = df[df.split.isin(split.split(","))]
    df = df.copy()
    df["pdb_resnum"] = norm_resnum(df["pdb_resnum"])
    return df


# ----------------------------------------------------------------------------- structure / pose
def load_pose(path, chain, device):
    import biotite.structure as struc
    from tmol.io import atom_array_from_file, pose_stack_from_atom_array

    path = Path(path) if Path(path).is_absolute() else ROOT / path
    arr = atom_array_from_file(str(path))
    arr = arr[(arr.chain_id == chain) & (~arr.hetero) & np.isin(arr.res_name, list(THREE2ONE) + ["MSE"])]
    if (arr.res_name == "MSE").any():  # selenomethionine -> Met
        m = arr.res_name == "MSE"
        arr.res_name[m] = "MET"
        arr.atom_name[m & (arr.atom_name == "SE")] = "SD"
        arr.element[m & (arr.element == "SE")] = "S"
    # keep first altloc only (biotite already did); drop residues lacking backbone N/CA/C
    starts = struc.get_residue_starts(arr, add_exclusive_stop=True)
    keep = np.zeros(len(arr), bool)
    for s, e in zip(starts[:-1], starts[1:]):
        names = set(arr.atom_name[s:e])
        if {"N", "CA", "C"} <= names:
            keep[s:e] = True
    arr = arr[keep]
    pose = pose_stack_from_atom_array(arr, device)
    return pose, arr


def residue_geometry(arr):
    """Per residue: label 'resnum+icode', heavy-atom coords (+ virtual CB for Gly)."""
    import biotite.structure as struc

    starts = struc.get_residue_starts(arr, add_exclusive_stop=True)
    labels, heavy = [], []
    for s, e in zip(starts[:-1], starts[1:]):
        sub = arr[s:e]
        labels.append(f"{sub.res_id[0]}{sub.ins_code[0].strip()}")
        h = sub[sub.element != "H"]
        xyz = h.coord
        if sub.res_name[0] == "GLY":
            try:
                n, ca, c = (h.coord[h.atom_name == a][0] for a in ("N", "CA", "C"))
                b, cc = ca - n, c - ca
                cb = -0.58273431 * np.cross(b, cc) + 0.56802827 * b - 0.54067466 * cc + ca
                xyz = np.vstack([xyz, cb])
            except IndexError:
                pass
        heavy.append(xyz)
    return labels, heavy


def shell_for_site(heavy, site, cutoff):
    site_xyz = heavy[site]
    out = []
    for j, xyz in enumerate(heavy):
        d = np.sqrt(((xyz[:, None, :] - site_xyz[None]) ** 2).sum(-1)).min()
        if d <= cutoff:
            out.append(j)
    return out


# ----------------------------------------------------------------------------- packing / scoring
_DB = {}


def get_tools(device):
    if "sfxn" not in _DB:
        from tmol.database import ParameterDatabase
        from tmol.score import beta2016_score_function
        from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

        _DB["sfxn"] = beta2016_score_function(device)
        _DB["db"] = ParameterDatabase.get_default()
        _DB["dun"] = create_dunbrack_sampler_from_database(_DB["db"], device)
    return _DB["sfxn"], _DB["dun"]


def pack_chunk(wt_pose, jobs, shells, device, seed, do_min):
    """jobs: list of (block, name3). Returns weighted per-term scores [n_terms, n_jobs] (numpy), term names."""
    from tmol.pose import PoseStackBuilder
    from tmol.pack import PackerPalette, PackerTask, pack_rotamers
    from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler

    sfxn, dun = get_tools(device)
    n = len(jobs)
    stack = PoseStackBuilder.from_poses([wt_pose] * n, device)
    nb = stack.max_n_blocks
    shell_mask = torch.zeros((n, nb), dtype=torch.bool)
    site_mask = torch.zeros((n, nb), dtype=torch.bool)
    for i, (blk, _) in enumerate(jobs):
        shell_mask[i, shells[blk]] = True
        site_mask[i, blk] = True
    shell_mask, site_mask = shell_mask.to(device), site_mask.to(device)

    task = PackerTask(stack, PackerPalette())
    task.disable_packing_by_block_mask(~shell_mask)
    task.restrict_to_repacking(shell_mask & ~site_mask)
    for name3 in sorted({j[1] for j in jobs}):
        m = torch.zeros((n, nb), dtype=torch.bool)
        for i, (blk, nm) in enumerate(jobs):
            if nm == name3:
                m[i, blk] = True
        task.restrict_absent_name3s({name3}, m.to(device))
    inc = IncludeCurrentSampler()
    task.add_conformer_sampler(dun)
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(inc)
    task.disable_sampler_by_block_mask(inc, site_mask)
    packed = pack_rotamers(stack, sfxn, task, seed=seed)

    if do_min:
        from tmol.kinematics import FoldForest, MoveMap
        from tmol.optimization import run_kin_min

        ff = FoldForest.reasonable_fold_forest(packed)
        mm = MoveMap.from_pose_stack(packed)
        mm.set_move_all_sc_tors_for_blocks(shell_mask)
        packed = run_kin_min(packed, sfxn, ff, mm, optimizer_kwargs={"max_iter": 50})

    scorer = sfxn.render_whole_pose_scoring_module(packed)
    with torch.no_grad():
        terms = scorer(packed.coords, sum_terms=False, apply_weights=True)
    # sanity: site identity
    pbt = packed.packed_block_types
    got = [pbt.active_block_types[int(packed.block_type_ind64[i, blk])].name3 for i, (blk, _) in enumerate(jobs)]
    ok = [g == nm for g, (_, nm) in zip(got, jobs)]
    return terms.detach().cpu().numpy(), np.array(ok)


def term_names(sfxn):
    return [st.name for st in sfxn.all_score_types()]


def run_protein(pid, sub, device, a, log):
    t0 = time.time()
    struct, chain = sub.structure.iloc[0], sub.chain.iloc[0]
    pose, arr = load_pose(struct, chain, device)
    t_load = time.time() - t0
    bio_labels, bio_heavy = residue_geometry(arr)
    nb = int((pose.block_type_ind64[0] >= 0).sum())
    # residue labels of tmol blocks (tmol may skip residues, e.g. terminal ones lacking main-chain atoms)
    pi_lab = pose.pdb_info.residue_labels[0][:nb]
    pi_ic = pose.pdb_info.residue_insertion_codes[0][:nb]
    labels = [f"{int(r)}{(c or '').strip()}" for r, c in zip(pi_lab, pi_ic)]
    geo = dict(zip(bio_labels, bio_heavy))
    missing = [l for l in labels if l not in geo]
    if missing:
        raise RuntimeError(f"tmol blocks without biotite residue: {missing[:5]}")
    if nb != len(bio_labels):
        log(f"  {pid}: tmol kept {nb} of {len(bio_labels)} residues; dropped {[l for l in bio_labels if l not in set(labels)]}")
    heavy = [geo[l] for l in labels]
    lab2blk = {l: i for i, l in enumerate(labels)}
    pbt = pose.packed_block_types
    name3 = [pbt.active_block_types[int(pose.block_type_ind64[0, i])].name3 for i in range(nb)]

    sub = sub.copy()
    sub["block"] = sub.pdb_resnum.map(lab2blk)
    status = pd.Series("ok", index=sub.index)
    status[sub.block.isna()] = "resnum_not_in_structure"
    okrows = sub.block.notna()
    sub.loc[okrows, "struct_wt"] = [THREE2ONE.get(name3[int(b)], "X") for b in sub.block[okrows]]
    status[okrows & (sub.struct_wt != sub.wt)] = "wt_mismatch"
    use = sub[status == "ok"]
    sites = sorted(use.block.astype(int).unique())
    shells = {b: shell_for_site(heavy, b, a.shell) for b in sites}
    jobs = []
    for b in sites:
        jobs.append((b, name3[b]))  # WT reference (same protocol)
        for m in sorted(use[use.block == b].mut.unique()):
            jobs.append((b, ONE2THREE[m]))
    jobs = list(dict.fromkeys(jobs))
    t_prep = time.time() - t0

    sfxn, _ = get_tools(device)
    names = term_names(sfxn)
    best = None
    okflag = np.ones(len(jobs), bool)
    t1 = time.time()
    for seed in range(a.seeds):
        res = []
        for c0 in range(0, len(jobs), a.chunk):
            ch = jobs[c0:c0 + a.chunk]
            try:
                terms, ok = pack_chunk(pose, ch, shells, device, a.seed_base + seed, a.min)
            except Exception as e:  # fall back to per-job packing to isolate the failing job
                log(f"  chunk {c0} failed ({type(e).__name__}: {str(e)[:120]}); retrying one by one")
                terms = np.full((len(names), len(ch)), np.nan, np.float32)
                ok = np.zeros(len(ch), bool)
                for k, j in enumerate(ch):
                    try:
                        tt, oo = pack_chunk(pose, [j], shells, device, a.seed_base + seed, a.min)
                        terms[:, k], ok[k] = tt[:, 0], oo[0]
                    except Exception as e2:
                        log(f"    job {j} failed: {type(e2).__name__}: {str(e2)[:120]}")
            res.append(terms)
            okflag[c0:c0 + len(ch)] &= ok
        terms = np.concatenate(res, axis=1)
        if best is None:
            best = terms
        else:
            better = np.nansum(terms, 0) < np.nansum(best, 0)
            best[:, better] = terms[:, better]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_pack = time.time() - t1

    E = pd.DataFrame(best.T, columns=[f"d_{n}" for n in names])
    E["block"] = [j[0] for j in jobs]
    E["name3"] = [j[1] for j in jobs]
    E["ok"] = okflag
    dcols = [f"d_{n}" for n in names]
    is_wt = np.array([n3 == name3[b] for b, n3 in zip(E.block, E.name3)])
    wt_ref = E[is_wt].drop(columns="name3").set_index("block")
    out = sub[KEYS + ["block"]].copy()
    out["status"] = status
    out["name3"] = out.mut.map(ONE2THREE)
    okm = out.status == "ok"
    mrows = out.loc[okm, ["block", "name3"]].astype({"block": int}).merge(
        E, on=["block", "name3"], how="left").set_index(out.index[okm])
    wrows = wt_ref.reindex(mrows.block.to_numpy()).set_index(mrows.index)
    good = mrows.ok.fillna(False).astype(bool) & wrows.ok.fillna(False).astype(bool)
    out.loc[good[~good].index, "status"] = "pack_failed"
    gi = good[good].index
    D = mrows.loc[gi, dcols].to_numpy() - wrows.loc[gi, dcols].to_numpy()
    out[dcols] = np.nan
    out.loc[gi, dcols] = D
    out = out.drop(columns="name3")
    out["ddg_reu"] = out[dcols].sum(axis=1, min_count=1)
    out.loc[out.status != "ok", "ddg_reu"] = np.nan
    out["ddg_pred"] = out.ddg_reu / REU_PER_KCAL
    out["n_shell"] = out.block.map(lambda b: len(shells.get(int(b), [])) if pd.notna(b) else np.nan)
    n_mut = int((out.status == "ok").sum())
    info = dict(protein_id=pid, n_res=nb, n_rows=len(out), n_ok=n_mut, n_poses=len(jobs), t_load=t_load, t_prep=t_prep,
                t_pack=t_pack, t_total=time.time() - t0, mut_per_s=len(jobs) * a.seeds / max(t_pack, 1e-9))
    return out.drop(columns=["block"]), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="megascale")
    ap.add_argument("--split", default="test")
    ap.add_argument("--proteins", nargs="*")
    ap.add_argument("--n-proteins", type=int, default=0, help="first N proteins (sorted) after --proteins filter")
    ap.add_argument("--shell", type=float, default=8.0)
    ap.add_argument("--min", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--table", help="variant table (csv/parquet) instead of data/processed/<dataset> (e.g. speed tests)")
    ap.add_argument("--max-rows-per-protein", type=int, default=0, help="subsample sites (whole sites kept)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_table(a.dataset, a.split, a.table)
    pids = sorted(df.protein_id.unique())
    if a.proteins:
        pids = [p for p in pids if p in set(a.proteins)]
    if a.n_proteins:
        pids = pids[:a.n_proteins]
    tag = f"_{a.tag}" if a.tag else ""
    out_dir = ROOT / "results" / "tmol"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.dataset}{tag}.parquet"
    log_path = out_dir / f"{a.dataset}{tag}.log"
    logf = open(log_path, "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    import tmol
    log(f"# {time.strftime('%F %T')} tmol {tmol.__version__} torch {torch.__version__} device {device} "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else ''} args {vars(a)}")
    done = pd.read_parquet(out_path) if out_path.exists() else None
    parts = [done] if done is not None else []
    done_ids = set(done.protein_id) if done is not None else set()
    infos = []
    for pid in pids:
        if pid in done_ids:
            continue
        sub = df[df.protein_id == pid]
        if a.max_rows_per_protein and len(sub) > a.max_rows_per_protein:
            sites = sub.pdb_resnum.drop_duplicates().sample(frac=1, random_state=0)
            keep, n = [], 0
            for s in sites:
                k = (sub.pdb_resnum == s).sum()
                if n + k > a.max_rows_per_protein:
                    break
                keep.append(s)
                n += k
            sub = sub[sub.pdb_resnum.isin(keep)]
        try:
            out, info = run_protein(pid, sub, device, a, log)
        except Exception as e:
            log(f"{pid}: FAILED {type(e).__name__}: {str(e)[:300]}")
            continue
        infos.append(info)
        log(f"{pid}: res {info['n_res']} rows {info['n_rows']} ok {info['n_ok']} poses {info['n_poses']} "
            f"load {info['t_load']:.1f}s prep {info['t_prep']:.1f}s pack {info['t_pack']:.1f}s -> {info['mut_per_s']:.1f} poses/s; total {info['t_total']:.1f}s; "
            f"status {out.status.value_counts().to_dict()}")
        parts.append(out)
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)
        torch.cuda.empty_cache()
    if infos:
        I = pd.DataFrame(infos)
        speed_path = out_dir / f"speed_{a.dataset}{tag}.csv"
        I.to_csv(speed_path, mode="a", header=not speed_path.exists(), index=False)
        log(f"total poses {I.n_poses.sum()} in {I.t_pack.sum():.0f}s pack + {I.t_prep.sum():.0f}s prep: "
            f"{I.n_poses.sum() * a.seeds / I.t_pack.sum():.1f} poses/s packing")


if __name__ == "__main__":
    main()
