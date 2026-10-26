# Mass-balance (unfolded-state) correction on top of FoldX, Megascale-test (literature-support).
# Run: PYTHONPATH=. .venv/bin/python scripts/lit_unfolded_state_mbc.py
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Polypeptide import protein_letters_3to1
AAs=list("ACDEFGHIKLMNPQRSTVWY")
fx=pd.read_csv("/foldx_native_clean/results/foldx_native_megascale.csv")
mg=pd.read_csv("/workspace/data/runtime_tables/megascale_runtime.csv")
mg=mg[mg.mut_type!="wt"].copy(); mg["entity_id"]=mg.WT_name.str.replace(".pdb","",regex=False)
m=mg.mut_type.str.fullmatch(r"[A-Z]\d+[A-Z]"); mg=mg[m.fillna(False)].copy()
mg["wt_aa"]=mg.mut_type.str[0]; mg["mut_aa"]=mg.mut_type.str[-1]
mg["position_1based"]=mg.mut_type.str[1:-1].astype(int)
mg["ddg_exp"]=-pd.to_numeric(mg.ddG_ML,errors="coerce")
mg=mg.dropna(subset=["ddg_exp"]).groupby(["entity_id","position_1based","wt_aa","mut_aa"],as_index=False).ddg_exp.mean()
df=fx.merge(mg,on=["entity_id","position_1based","wt_aa","mut_aa"])
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
p=PDBParser(QUIET=True); sr=ShrakeRupley(); rsa={}
for pid in df.entity_id.unique():
    s=p.get_structure(pid,f"/workspace/data/megascale_pdbs/{pid}.pdb")[0]; sr.compute(s,level="R")
    for ch in s:
        for res in ch:
            try: aa=protein_letters_3to1[res.get_resname()]
            except KeyError: continue
            rsa[(pid,ch.id,str(res.id[1]).strip())]=res.sasa/MAXASA[aa]
df["rsa"]=[rsa.get((r.entity_id,r.chain,str(r.pdb_resnum)),np.nan) for r in df.itertuples()]
df=df.dropna(subset=["rsa"]); y=df.ddg_exp.to_numpy(); g=df.entity_id.to_numpy()
O=np.zeros((len(df),20))
for j,a in enumerate(AAs):
    O[:,j]=(df.mut_aa.values==a).astype(float)-(df.wt_aa.values==a).astype(float)
bur=(1-df.rsa.clip(0,1)).to_numpy()[:,None]
S=df.foldx_ddg.to_numpy()[:,None]
def cv(X,label):
    pred=np.zeros_like(y)
    for tr,te in GroupKFold(n_splits=5).split(X,y,g):
        pred[te]=Ridge(alpha=1.0).fit(X[tr],y[tr]).predict(X[te])
    per=[spearmanr(pred[g==q],y[g==q]).statistic for q in np.unique(g)]
    print(f"{label:52s} pooled rho {spearmanr(pred,y).statistic:.3f}  r {pearsonr(pred,y).statistic:.3f}  per-prot med {np.nanmedian(per):.3f}  RMSE {np.sqrt(((pred-y)**2).mean()):.2f}")
    return pred
cv(S,"FoldX only (rescaled)")
cv(np.hstack([S,O]),"FoldX + 20-coef unfolded-state (MBC)")
cv(np.hstack([S,O,bur]),"FoldX + MBC + burial")
cv(np.hstack([S,O,bur,bur*O]),"FoldX + MBC + burial + burial x AA (40 coef)")
cv(np.hstack([O,bur,bur*O]),"no FoldX: MBC + burial + burial x AA")
