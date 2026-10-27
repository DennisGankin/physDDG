"""Tests for the mutation / packing engine (physddg/mutate.py)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from physddg import energy as E
from physddg import topology as T
from physddg.mutate import MutateConfig, SiteScan, stat_features
from physddg.structure import annotate, load_protein

PDB = Path("/workspace/data/megascale_pdbs/1A32.pdb")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not PDB.exists(), reason="test structure not available")


@pytest.fixture(scope="module")
def prot():
    return annotate(load_protein(PDB))


@pytest.fixture(scope="module")
def scan(prot):
    return SiteScan(prot, MutateConfig(), DEV)


def test_stage1_deterministic_batch_independent(scan, prot):
    sites = np.array([3, 10, 25, 40])
    a = scan.stage1(sites)
    b = scan.stage1(sites)
    for k in ("F", "E_avg", "E_min", "S_site", "ref"):
        assert np.array_equal(a[k], b[k]), k                        # bitwise deterministic
    c = scan.stage1(sites[1:2])                                     # same site, different batch composition
    assert np.allclose(a["F"][1], c["F"][0], atol=1e-4) and np.allclose(a["E_avg"][1], c["E_avg"][0], atol=1e-4)
    assert a["F"].shape == (4, 20) and a["E_avg"].shape == (4, 20, len(scan.terms))
    assert (a["n_rot"][:, T.AA_INDEX["G"]] == 1).all() and (a["n_rot"][:, T.AA_INDEX["R"]] > 5).all()
    # F <= U_min (free energy below the best state) and F >= U_min - kT ln n
    assert np.all(a["F"] <= a["U_min"] + 1e-4)
    assert np.all(a["F"] >= a["U_min"] - scan.cfg.kT * np.log(a["n_rot"]) - 1e-4)


def test_stage1_matches_direct_energy(scan, prot):
    """Boltzmann average for a single-rotamer residue (Ala) equals intra + env energy of that residue."""
    s = 10
    r = scan.stage1(np.array([s]))
    cand = scan.site_candidates(np.array([s]))
    ia = np.flatnonzero(cand["a"] == T.AA_INDEX["A"])
    mov = scan._blocks(np.array([s]), np.array([T.AA_INDEX["A"]]), cand["xyz"][torch.as_tensor(ia, device=DEV)])
    em = torch.ones(len(prot), dtype=torch.bool, device=DEV)
    em[s] = False
    p = scan.cfg.params
    e = (E.intra_energy(mov, p, include_ref=False) + E.env_energy(mov, scan.env, p, env_mask=em)).cpu().numpy()[0]
    assert np.allclose(r["E_avg"][0, T.AA_INDEX["A"]], e, atol=1e-4)


def test_stage2_reduces_to_stage1_without_neighbours(prot):
    cfg = MutateConfig(nb_radius=0.0)
    sc = SiteScan(prot, cfg, DEV)
    sites = np.array([5, 30])
    s2 = sc.stage2(sites)
    s1 = SiteScan(prot, replace(cfg, max_rot=cfg.s2_max_rot), DEV).stage1(sites)
    assert (s2["n_nb"] == 0).all()
    assert np.allclose(s2["F"], s1["F"], atol=2e-3)
    assert np.allclose(s2["E_avg"], s1["E_avg"], atol=2e-3)


def test_stage2_wt_zero_and_determinism(scan, prot):
    sites = np.array([8, 20])
    a = scan.stage2(sites)
    b = scan.stage2(sites)
    for k in ("F", "E_avg", "S_local"):
        assert np.array_equal(a[k], b[k]), k
    assert (a["n_nb"] > 0).all()
    c = scan.stage2(sites[:1])
    assert np.allclose(a["F"][0], c["F"][0], atol=1e-3)
    assert np.isfinite(a["F"]).all()


def test_neighbour_sets_identical_for_all_aa(scan):
    sites = np.array([12])
    cand = scan.site_candidates(sites, max_rot=scan.cfg.s2_max_rot)
    nb = scan.neighbour_sets(sites, cand)[0]
    assert 12 not in nb and len(nb) <= scan.cfg.nb_max
    seq = scan.prot.seq
    assert all(T.N_CHI[seq[j]] > 0 and seq[j] != "P" for j in nb)


def test_cb_rule(scan, prot):
    gly = np.flatnonzero(prot.aa == T.AA_INDEX["G"])
    non = np.flatnonzero((prot.aa != T.AA_INDEX["G"]) & (prot.aa != T.AA_INDEX["P"]))[:3]
    cb_nat = scan._cb(non, T.AA_INDEX["L"]).cpu().numpy()
    assert np.allclose(cb_nat, prot.xyz14[non, 4], atol=1e-5)               # native CB kept
    if len(gly):
        cb = scan._cb(gly[:1], T.AA_INDEX["L"]).cpu().numpy()[0]
        n, ca = prot.xyz14[gly[0], 0], prot.xyz14[gly[0], 1]
        assert abs(np.linalg.norm(cb - ca) - T.IDEAL_IC["L"]["CB"][0]) < 1e-3 and np.linalg.norm(cb - n) > 2.0


def test_stat_features(prot):
    sf = stat_features(prot, np.arange(len(prot)))
    for k, v in sf["per_aa"].items():
        assert v.shape == (len(prot), 20) and np.isfinite(v).all(), k
    assert (sf["per_aa"]["rama_prev"][0] == 0).all()


def test_stage2_frozen_env_reverse(prot):
    """Without neighbours every environment row equals stage-1 F, so reverse = -forward exactly; with neighbours the
    diagonal (residue in its own relaxed environment) is finite and WT->WT is zero."""
    sites = np.array([5, 30])
    s2 = SiteScan(prot, MutateConfig(nb_radius=0.0), DEV).stage2(sites)
    F = s2["F_env"]
    assert F.shape == (2, 20, 20) and s2["E_env"].shape[:3] == (2, 20, 20)
    assert np.allclose(F, F[:, :1, :], atol=1e-5)                     # env rows identical (no flexible residues)
    w = prot.aa[sites]
    for k in range(2):
        fz = F[k, w[k], :] - F[k, w[k], w[k]]
        rev = np.array([F[k, m, w[k]] - F[k, m, m] for m in range(20)])
        assert np.allclose(rev, -fz, atol=1e-4)
    s2n = SiteScan(prot, MutateConfig(), DEV).stage2(sites)
    assert np.isfinite(s2n["F_env"]).all() and np.isfinite(s2n["E_env"]).all()



def _perturbed(prot, res: int, shift=(0.7, -0.5, 0.4)):
    """Copy of prot with the side-chain atoms beyond CB of residue `res` translated (backbone and CB unchanged)."""
    import copy
    q = copy.deepcopy(prot)
    q.xyz14[res, 5:] = q.xyz14[res, 5:] + np.asarray(shift, np.float32)
    return q


@pytest.mark.parametrize("stage", [1, 2])
def test_site_native_side_chain_not_used(prot, stage):
    """Audit item 10: the mutated site is modelled from library rotamers only. Moving the WT side chain of the site
    (atoms beyond CB) must not change its features, while moving a neighbour's side chain must."""
    site = 25
    cfg = MutateConfig()
    run = lambda p: (SiteScan(p, cfg, DEV).stage1 if stage == 1 else SiteScan(p, cfg, DEV).stage2)(np.array([site]))
    base = run(prot)
    moved = run(_perturbed(prot, site))
    # stage 1: exact code-path independence; stage 2: neighbours' fixed-environment energies are obtained as
    # (vs whole protein) - (vs native flexible residues incl. the site), equal up to float32 rounding (~1e-5)
    tol = 1e-6 if stage == 1 else 1e-4
    assert np.abs(moved["F"] - base["F"]).max() <= tol
    assert np.abs(moved["E_avg"] - base["E_avg"]).max() <= tol
    ca = prot.xyz14[:, 1]
    d = np.linalg.norm(ca - ca[site], axis=1)
    nb = [j for j in np.argsort(d)[1:] if T.N_CHI[prot.seq[j]] >= 2 and not prot.missing_sc[j]][0]
    other = run(_perturbed(prot, nb, (1.5, 1.5, 0.0)))
    assert np.abs(other["F"] - base["F"]).max() > 1e-3


def test_predict_api_wt_to_wt_and_reproduces_fit():
    """Public API: WT->WT is exactly 0; predictions reproduce the fitted B4 model's stored predictions."""
    import pandas as pd
    from physddg.predict import PhysDDG
    preds = Path("results/ladder_physics/preds_megascale.parquet")
    from physddg.predict import MODELS
    m = PhysDDG(MODELS / "B4.json", device=DEV)
    out = m.predict(PDB, ["S1S", "P2P", "E3E", "E3A"])
    assert (out.ddg.to_numpy()[:3] == 0.0).all() and out.ddg.to_numpy()[3] != 0.0
    assert out.wt_match.all() and out.site_found.all()
    if preds.exists():
        P = pd.read_parquet(preds)
        g = P[P.protein_id == P.protein_id.iloc[0]]
        o = m.predict(g.structure.iloc[0], [f"{c}:{w}{r}{u}" for c, w, r, u in zip(g.chain, g.wt, g.pdb_resnum, g.mut)])
        assert np.nanmax(np.abs(o.ddg.to_numpy() - g["B4"].to_numpy())) < 1e-4


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_determinism_across_runs_and_gpus(prot):
    sites = np.array([7, 33])
    r = [SiteScan(prot, MutateConfig(), d).stage1(sites)["F"] for d in ("cuda:0", "cuda:0", "cuda:1")]
    assert np.array_equal(r[0], r[1]) and np.array_equal(r[0], r[2])
    r2 = [SiteScan(prot, MutateConfig(), d).stage2(sites)["F"] for d in ("cuda:0", "cuda:1")]
    assert np.array_equal(r2[0], r2[1])



def _stored(bench):
    import pandas as pd
    f = Path(f"results/ladder_physics/preds_{bench}.parquet")
    return pd.read_parquet(f) if f.exists() else None


@pytest.mark.parametrize("name,bench,n", [("B6gjs1", "megascale", 2), ("B6gj", "megascale", 2), ("B6gjaf", "megascale", 2),
                                          ("B6gjs1", "fireprot", 1), ("B6gj", "fireprot", 1),
                                          ("B6gjaf", "domainome", 1), ("B6gj", "domainome", 1),
                                          ("B6gs1", "megascale", 1), ("B6g", "megascale", 1)])
def test_predict_reproduces_shipped_models(name, bench, n):
    """predict() (exact training-time feature path incl. features_extra and the two-stage attenuation) reproduces
    the fitting agent's stored predictions to 1e-4 (joint models = shipped defaults; B6g = Megascale-only refs)."""
    from physddg.predict import MODELS, PhysDDG
    P = _stored(bench)
    if P is None or name not in P or not (MODELS / f"{name}.json").exists():
        pytest.skip(f"{name} predictions / weights not available")
    m = PhysDDG(MODELS / f"{name}.json", device=DEV)          # xtal mode, struct_source from the table
    for pid in P.protein_id.unique()[:n]:
        g = P[P.protein_id == pid]
        o = m.predict(g.structure.iloc[0], [f"{c}:{w}{r}{u}" for c, w, r, u in zip(g.chain, g.wt, g.pdb_resnum, g.mut)],
                      struct_source=g.struct_source.iloc[0])
        assert np.nanmax(np.abs(o.ddg.to_numpy() - g[name].to_numpy())) < 1e-4


def test_defaults_are_final_models_and_nuisance_ignored(tmp_path):
    """Defaults resolve to final_*.json (joint models); the nuisance block (FireProt scale/offset, Megascale delta_ref,
    also inside the AF model's scale_base) never affects predictions."""
    import json
    from physddg.predict import MODELS, PhysDDG, default_weights
    if not (MODELS / "final_stage1.json").exists():
        pytest.skip("final models not available")
    assert default_weights(1).name == "final_stage1.json" and default_weights(2).name == "final_stage2.json"
    assert default_weights(2, "af").name == "final_af.json"
    for name in ("final_stage1.json", "final_af.json"):
        m = json.loads((MODELS / name).read_text())
        bad = json.loads(json.dumps(m))
        for mm in [bad] + ([bad["scale_base"]] if "scale_base" in bad else []):
            nu = mm.get("nuisance") or {}
            for k in ("scale", "offset"):
                if isinstance(nu.get(k), dict):
                    nu[k] = {a: 99.0 for a in nu[k]}
            if isinstance(nu.get("delta_ref"), dict):
                nu["delta_ref"] = {a: ({x: 50.0 for x in v} if isinstance(v, dict) else 50.0)
                                   for a, v in nu["delta_ref"].items()}
        f = tmp_path / name
        f.write_text(json.dumps(bad))
        muts = ["E3A", "V4A", "Q5P", "S1S"]
        a = PhysDDG(MODELS / name, device=DEV).predict(PDB, muts)
        b = PhysDDG(f, device=DEV).predict(PDB, muts)
        assert np.array_equal(a.ddg.to_numpy(), b.ddg.to_numpy())
        assert b.ddg.to_numpy()[3] == 0.0


def test_crystal_input_never_attenuated():
    from physddg.predict import MODELS, PhysDDG
    w = MODELS / "final_af.json"
    if not w.exists():
        pytest.skip("AF weights not available")
    m = PhysDDG(w, device=DEV, mode="xtal")
    o = m.predict(PDB, ["E3A", "V4A", "Q5P"], terms=True)
    assert (o["c_scale_plddt"] == 0.0).all()
    a = PhysDDG(w, device=DEV, mode="af", trim=False).predict(PDB, ["E3A", "V4A", "Q5P"], terms=True)
    assert "withheld" in a and "withheld" not in o


def test_af_crop_equals_construct_file():
    """Cropping the full AF model to the assayed range reproduces predictions on the construct file exactly."""
    import pandas as pd
    from physddg.predict import PhysDDG
    f = Path("data/processed/domainome.parquet")
    if not f.exists():
        pytest.skip("domainome table not available")
    D = pd.read_parquet(f, columns=["protein_id", "structure", "af_full_structure", "af_full_resnum", "pdb_resnum",
                                    "wt", "mut"])
    G = D[D.protein_id == D.protein_id.iloc[0]]
    if not Path(G.af_full_structure.iloc[0]).exists():
        pytest.skip("full AF model not available")
    g = G.iloc[::25]
    fr = G.af_full_resnum.astype(int)
    m = PhysDDG(mode="af", device=DEV, trim=False)
    a = m.predict(g.structure.iloc[0], [f"{w}{r}{u}" for w, r, u in zip(g.wt, g.pdb_resnum, g.mut)])
    b = m.predict(g.af_full_structure.iloc[0], [f"{w}{r}{u}" for w, r, u in zip(g.wt, g.af_full_resnum.astype(int), g.mut)],
                  crop=f"A:{fr.min()}-{fr.max()}")
    assert b.wt_match.all() and np.allclose(a.ddg.to_numpy(), b.ddg.to_numpy(), atol=1e-6)


# ----------------------------------------------------------------------------- structure type (audit 2, I4 / D23)
XTAL_RAW = Path("/workspace/data/fireprot_pdbs/1A23.pdb")                       # crystal/NMR with PDB header
XTAL_HIGH_B = Path("data/structures/xtal/fireprot/1JNX_X.pdb")                  # headerless crystal, mean B 30-100
AF_MODEL = Path("/workspace/data/yeast_af/pdbs/P20486.pdb")                     # AlphaFold DB model (pLDDT)
MS_NO_PLDDT = Path("/workspace/data/megascale_pdbs/EEHEE_rd2_0391.pdb")         # Megascale model, empty B column


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"{p} not available")


def test_detect_structure_type():
    from physddg.predict import detect_structure_type, plddt_available
    _need(XTAL_RAW, XTAL_HIGH_B, AF_MODEL, MS_NO_PLDDT)
    for f, want in ((XTAL_RAW, "xtal"), (XTAL_HIGH_B, "xtal"), (AF_MODEL, "af"), (MS_NO_PLDDT, "xtal")):
        p = load_protein(f)
        assert detect_structure_type(f, p)[0] == want, f
    assert plddt_available(load_protein(AF_MODEL)) and not plddt_available(load_protein(MS_NO_PLDDT))


def test_crystal_never_attenuated_any_path():
    """A crystal whose mean B-factor lies in 30-100 (the case the old B-factor fallback called an AF model) gets no
    pLDDT attenuation with AF weights: in xtal mode, with structure_type='auto', and through features() called
    directly without any label."""
    from physddg.predict import MODELS, PhysDDG
    _need(XTAL_HIGH_B, MODELS / "final_af.json")
    p = load_protein(XTAL_HIGH_B)
    assert 30 <= p.bfactor.mean() <= 100 and p.b14[p.mask14].max() <= 100
    muts = [f"{a}{r}A" for a, r in zip(p.seq[20:26], p.resnum[20:26]) if a != "A"]
    for kw in ({"mode": "xtal"}, {"mode": "af"}):
        m = PhysDDG(MODELS / "final_af.json", device=DEV, **kw)
        o = m.predict(XTAL_HIGH_B, muts, terms=True, structure_type="auto" if kw["mode"] == "af" else None)
        assert (o["c_scale_plddt"] == 0).all() and (o["structure_type"] == "xtal").all()
        assert not o["flags"].str.contains("plddt").any()
    m = PhysDDG(MODELS / "final_af.json", device=DEV)
    prot = m.load(XTAL_HIGH_B)
    rows = m._rows(prot, str(XTAL_HIGH_B), muts, None, False, None)
    df = m.features(prot, str(XTAL_HIGH_B), rows)                 # building block, no label
    assert (df["flag_lowplddt"] == 0).all()


def test_af_model_attenuated_and_model_without_plddt_flagged():
    from physddg.predict import MODELS, PhysDDG
    _need(AF_MODEL, MS_NO_PLDDT, MODELS / "final_af.json")
    m = PhysDDG(MODELS / "final_af.json", device=DEV, mode="af", trim=False)
    p = load_protein(AF_MODEL)
    low = [i for i in range(len(p)) if p.bfactor[i] < 70 and p.seq[i] != "A"][:3]
    o = m.predict(AF_MODEL, [f"{p.seq[i]}{p.resnum[i]}A" for i in low], terms=True, structure_type="auto")
    assert (o["structure_type"] == "af").all() and (o["c_scale_plddt"] != 0).any()
    q = load_protein(MS_NO_PLDDT)
    muts = [f"{q.seq[i]}{q.resnum[i]}A" for i in range(5, 10) if q.seq[i] != "A"]
    for kw in ({"structure_type": "af"}, {"struct_source": "af2_model_tsuboyama"}):
        o = m.predict(MS_NO_PLDDT, muts, terms=True, **kw)
        assert (o["c_scale_plddt"] == 0).all() and o["flags"].str.contains("plddt_missing").all()
        assert not o["flags"].str.contains("low_plddt").any()
