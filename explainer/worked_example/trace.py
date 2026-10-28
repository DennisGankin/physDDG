"""Worked-example trace of PhysDDG on 1PSE (Megascale-test). Prints intermediates for each step."""
import json, sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/stability_hackathon")
from physddg.predict import PhysDDG, site_arrays, engine_columns, derive
from physddg.mutate import SiteScan, stat_features
from physddg import topology as T
OUT = "./"
S = "/workspace/stability_hackathon/examples/1PSE.pdb"
MUTS = ["V24A", "K29E", "G31A", "A61W"]
res = {}
t0 = time.time()
m = PhysDDG(device="cpu")                     # default final_stage2 (B6g)
prot = m.load(S)
a = prot.ann
print("annotation keys:", sorted(a.keys()))
key = {r: i for i, r in enumerate(prot.resnum.tolist())}
sites = np.array(sorted({key[x[1:-1]] for x in MUTS}))
# step 1: site context
ctx = []
for x in MUTS:
    i = key[x[1:-1]]
    ctx.append(dict(mut=x, wt=prot.seq[i], ss=str(a["ss3"][i]), dssp=str(a.get("dssp", a["ss3"])[i]) if "dssp" in a else "",
                    phi=round(float(a["phi"][i]), 1), psi=round(float(a["psi"][i]), 1),
                    rsa=round(float(a["rsa"][i]), 2), cn=round(float(a["cn"][i]), 1), plddt=round(float(prot.bfactor[i]), 1)))
res["context"] = ctx; print(pd.DataFrame(ctx))
# step 2: stage 1 (site rotamers, fixed environment)
sc = SiteScan(prot, m.cfg, "cpu")
cand = sc.site_candidates(sites)
s1 = sc.stage1(sites, return_rotamers=True)
terms = list(sc.terms); print("engine terms:", terms)
AA = T.AA1
s1tab = []
for x in MUTS:
    k = int(np.flatnonzero(sites == key[x[1:-1]])[0])
    for aa in (x[0], x[-1]):
        j = AA.index(aa)
        s1tab.append(dict(mut=x, aa=aa, n_rot=int(s1["n_rot"][k, j]), F=round(float(s1["F"][k, j]), 2),
                          U_min=round(float(s1["U_min"][k, j]), 2), S_site=round(float(s1["S_site"][k, j]), 2),
                          **{t: round(float(s1["E_avg"][k, j, ti]), 2) for ti, t in enumerate(terms)}))
res["stage1"] = s1tab; print(pd.DataFrame(s1tab).to_string())
# rotamer-level detail for V24 (WT) : energies and Boltzmann weights
R = s1["rotamers"]; k24 = int(np.flatnonzero(sites == key["24"])[0])
sel = (R["site"] == key["24"]) & (R["aa"] == AA.index("V"))
pack_idx = [terms.index(t) for t in m.cfg.pack_terms if t in terms]
U = R["E"][sel][:, pack_idx].sum(1) - 0.593 * R["logp"][sel]
w = np.exp(-(U - U.min()) / 0.593); w /= w.sum()
res["v24_rotamers"] = [dict(chi1=round(float(c[0]), 0), logp=round(float(lp), 2), U=round(float(u), 2), weight=round(float(ww), 3))
                       for c, lp, u, ww in zip(R["chis"][sel], R["logp"][sel], U, w)]
print(pd.DataFrame(res["v24_rotamers"]))
# step 3: stage 2 (flexible neighbours, mean field)
s2 = sc.stage2(sites)
s2tab = []
for x in MUTS:
    k = int(np.flatnonzero(sites == key[x[1:-1]])[0])
    for aa in (x[0], x[-1]):
        j = AA.index(aa)
        s2tab.append(dict(mut=x, aa=aa, n_nb=int(s2["n_nb"][k]), F=round(float(s2["F"][k, j]), 2),
                          S_local=round(float(s2["S_local"][k, j]), 2),
                          **{t: round(float(s2["E_avg"][k, j, ti]), 2) for ti, t in enumerate(terms)}))
res["stage2"] = s2tab; print(pd.DataFrame(s2tab).to_string())
nb = sc.neighbour_sets(sites, sc.site_candidates(sites, max_rot=m.cfg.s2_max_rot))
res["neighbours"] = {x: [prot.seq[j] + prot.resnum[j] for j in nb[int(np.flatnonzero(sites == key[x[1:-1]])[0])]] for x in MUTS}
print(res["neighbours"])
# step 4: statistics
st = stat_features(prot, sites, 0.593)
stt = []
for x in MUTS:
    k = int(np.flatnonzero(sites == key[x[1:-1]])[0]); wi, mi = AA.index(x[0]), AA.index(x[-1])
    stt.append(dict(mut=x, **{n: round(float(st["per_aa"][n][k, mi] - st["per_aa"][n][k, wi]), 2)
                               for n in ("env_cn_ss_bb", "rama", "rama_prev")}))
res["stats"] = stt; print(pd.DataFrame(stt))
# step 5+6: full predictor with features and contributions
t1 = time.time()
rows = m._rows(prot, S, MUTS, None, False, None)
m._stype, m._has_plddt = "xtal", False
df = m.features(prot, S, rows)
d = derive(df, m.stage, m.opts, m.caps)
W = m.model["weights"]
feat = d[["mutation"] + list(W)].copy()
res["features"] = json.loads(feat.round(3).to_json(orient="records"))
sco = m.score(df, terms=True)
out = pd.concat([rows[["mutation"]], sco], axis=1)
res["contrib"] = json.loads(out.round(3).to_json(orient="records"))
print(feat.round(3).T); print(out.round(3).T)
pr = m.predict(S, MUTS)
res["predict"] = json.loads(pr[["mutation", "ddg", "flags"]].round(3).to_json(orient="records"))
print(pr[["mutation", "ddg", "flags"]])
# measured + competitors
P = pd.read_parquet("/workspace/stability_hackathon/results/ladder_physics/preds_megascale.parquet")
P = P[P.protein_id == "1PSE"].assign(mutation=lambda q: q.wt + q.pdb_resnum + q.mut)
res["measured"] = json.loads(P[P.mutation.isin(MUTS)][["mutation", "ddg", "final_stage2", "FoldX-clip", "B3+"]].round(3).to_json(orient="records"))
res["model"] = {"intercept": m.model["intercept"], "weights": W, "ref": m.model["ref"]}
res["timing_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT + "trace.json", "w"), indent=1, default=float)
print("done", res["timing_s"])
