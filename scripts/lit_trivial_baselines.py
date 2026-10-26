# Trivial (non-physics) baselines on Megascale-test; literature-support analysis.
# Run: PYTHONPATH=. .venv/bin/python scripts/lit_trivial_baselines.py
"""Trivial baselines on Megascale-test (28 proteins of the ThermoMPNN split)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Polypeptide import protein_letters_3to1

fx=pd.read_csv("/foldx_native_clean/results/foldx_native_megascale.csv")
mg=pd.read_csv("/workspace/data/runtime_tables/megascale_runtime.csv")
mg=mg[mg.mut_type!="wt"].copy()
mg["entity_id"]=mg.WT_name.str.replace(".pdb","",regex=False)
m=mg.mut_type.str.fullmatch(r"[A-Z]\d+[A-Z]"); mg=mg[m.fillna(False)].copy()
mg["wt_aa"]=mg.mut_type.str[0]; mg["mut_aa"]=mg.mut_type.str[-1]
mg["position_1based"]=mg.mut_type.str[1:-1].astype(int)
mg["ddg_exp"]=-pd.to_numeric(mg.ddG_ML,errors="coerce")
mg=mg.dropna(subset=["ddg_exp"]).groupby(["entity_id","position_1based","wt_aa","mut_aa"],as_index=False).ddg_exp.mean()
df=fx[["entity_id","position_1based","wt_aa","mut_aa","chain","pdb_resnum","foldx_ddg"]].merge(
    mg,on=["entity_id","position_1based","wt_aa","mut_aa"])

# SASA per residue
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
        'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}  # Tien 2013 theoretical
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
VOL={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'E':138.4,'Q':143.8,'G':60.1,'H':153.2,
     'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
p=PDBParser(QUIET=True); sr=ShrakeRupley()
rsa={}
for pid in df.entity_id.unique():
    try:
        s=p.get_structure(pid,f"/workspace/data/megascale_pdbs/{pid}.pdb")[0]
    except Exception as e:
        print("miss",pid,e); continue
    sr.compute(s,level="R")
    for ch in s:
        for res in ch:
            try: aa=protein_letters_3to1[res.get_resname()]
            except KeyError: continue
            rsa[(pid,ch.id,str(res.id[1]).strip())]=res.sasa/MAXASA[aa]
df["rsa"]=[rsa.get((r.entity_id,r.chain,str(r.pdb_resnum)),np.nan) for r in df.itertuples()]
print("rsa coverage",df.rsa.notna().mean().round(3),"n",len(df))
d=df.dropna(subset=["rsa"]).copy()
d["burial"]=1-d.rsa.clip(0,1)
y=d.ddg_exp.to_numpy()
feats={
 "burial (1-RSA) alone": d.burial,
 "-RSA": -d.rsa,
 "burial x dKD (hydrophobicity loss)": d.burial*(d.wt_aa.map(KD)-d.mut_aa.map(KD)),
 "burial x |dVolume|": d.burial*(d.wt_aa.map(VOL)-d.mut_aa.map(VOL)).abs(),
 "burial x dVolume (shrink +)": d.burial*(d.wt_aa.map(VOL)-d.mut_aa.map(VOL)),
 "dKD alone": (d.wt_aa.map(KD)-d.mut_aa.map(KD)),
 "to-Pro / from-Gly indicator": (d.mut_aa=="P").astype(float)+ (d.wt_aa=="G").astype(float)*0,
 "FoldX": d.foldx_ddg,
}
for k,v in feats.items():
    v=np.asarray(v,float)
    per=[spearmanr(v[d.entity_id.values==q],y[d.entity_id.values==q]).statistic for q in d.entity_id.unique()]
    print(f"{k:38s} pooled {spearmanr(v,y).statistic: .3f}  per-protein median {np.nanmedian(per): .3f}")
# 3-feature linear, grouped CV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
X=np.column_stack([d.burial,d.burial*(d.wt_aa.map(KD)-d.mut_aa.map(KD)),
                   d.burial*(d.wt_aa.map(VOL)-d.mut_aa.map(VOL)),
                   (d.mut_aa=="P").astype(float),(d.wt_aa=="G").astype(float),
                   (d.wt_aa=="P").astype(float),(d.mut_aa=="G").astype(float)]).astype(float)
g=d.entity_id.values; pred=np.zeros_like(y)
for tr,te in GroupKFold(n_splits=5).split(X,y,g):
    pred[te]=LinearRegression().fit(X[tr],y[tr]).predict(X[te])
per=[spearmanr(pred[g==q],y[g==q]).statistic for q in d.entity_id.unique()]
print("7-feature trivial model (grouped CV): pooled %.3f per-protein median %.3f"%(spearmanr(pred,y).statistic,np.nanmedian(per)))
