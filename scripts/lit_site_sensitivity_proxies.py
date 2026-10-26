# How well do simple structural proxies predict per-site mean ddG on Megascale-test (literature-support).
# Run: PYTHONPATH=. .venv/bin/python scripts/lit_site_sensitivity_proxies.py
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Polypeptide import protein_letters_3to1
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
fx=pd.read_csv("/foldx_native_clean/results/foldx_native_megascale.csv")
mg=pd.read_csv("/workspace/data/runtime_tables/megascale_runtime.csv")
mg=mg[mg.mut_type!="wt"].copy(); mg["entity_id"]=mg.WT_name.str.replace(".pdb","",regex=False)
m=mg.mut_type.str.fullmatch(r"[A-Z]\d+[A-Z]"); mg=mg[m.fillna(False)].copy()
mg["wt_aa"]=mg.mut_type.str[0]; mg["mut_aa"]=mg.mut_type.str[-1]; mg["position_1based"]=mg.mut_type.str[1:-1].astype(int)
mg["y"]=-pd.to_numeric(mg.ddG_ML,errors="coerce")
mg=mg.dropna(subset=["y"]).groupby(["entity_id","position_1based","wt_aa","mut_aa"],as_index=False).y.mean()
df=fx[["entity_id","position_1based","wt_aa","mut_aa","chain","pdb_resnum","foldx_ddg"]].merge(mg,on=["entity_id","position_1based","wt_aa","mut_aa"])
site=df.groupby(["entity_id","chain","pdb_resnum","wt_aa"],as_index=False).agg(y_site=("y","mean"),fx_site=("foldx_ddg","mean"),n=("y","size"))
p=PDBParser(QUIET=True); sr=ShrakeRupley()
feat={}
for pid in site.entity_id.unique():
    s=p.get_structure(pid,f"/workspace/data/megascale_pdbs/{pid}.pdb")[0]; sr.compute(s,level="R")
    res=[r for ch in s for r in ch if r.get_resname() in protein_letters_3to1]
    cb=[]
    for r in res:
        a = r['CB'] if 'CB' in r else (r['CA'] if 'CA' in r else None)
        cb.append(a.coord if a is not None else np.full(3,np.nan))
    cb=np.array(cb); D=np.linalg.norm(cb[:,None,:]-cb[None,:,:],axis=-1)
    wcn=np.nansum(1.0/(1.0+np.exp((D-9.0)/1.5)),axis=1)-0.5
    cn10=(np.nan_to_num(D,nan=99)<10).sum(1)-1
    for i,r in enumerate(res):
        aa=protein_letters_3to1[r.get_resname()]
        feat[(pid,r.get_parent().id,str(r.id[1]).strip())]=(r.sasa/MAXASA[aa], wcn[i], cn10[i])
site["rsa"]=[feat.get((r.entity_id,r.chain,str(r.pdb_resnum)),(np.nan,)*3)[0] for r in site.itertuples()]
site["wcn"]=[feat.get((r.entity_id,r.chain,str(r.pdb_resnum)),(np.nan,)*3)[1] for r in site.itertuples()]
site["cn10"]=[feat.get((r.entity_id,r.chain,str(r.pdb_resnum)),(np.nan,)*3)[2] for r in site.itertuples()]
site=site.dropna()
print("n sites",len(site))
for k,v in {"-RSA":-site.rsa,"weighted contact number (CB, sig 9A)":site.wcn,"contact number <10A":site.cn10,"FoldX site mean":site.fx_site}.items():
    per=[spearmanr(v[site.entity_id==q],site.y_site[site.entity_id==q]).statistic for q in site.entity_id.unique()]
    print(f"{k:38s} vs site-mean ddG: pooled {spearmanr(v,site.y_site).statistic:.3f} per-prot med {np.nanmedian(per):.3f}")
