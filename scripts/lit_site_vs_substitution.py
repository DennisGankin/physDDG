# Oracle decomposition of Megascale-test ddG into site and substitution effects (literature-support).
# Run: PYTHONPATH=. .venv/bin/python scripts/lit_site_vs_substitution.py
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
mg=pd.read_csv("/workspace/data/runtime_tables/megascale_runtime.csv")
mg=mg[mg.mut_type!="wt"].copy(); mg["entity_id"]=mg.WT_name.str.replace(".pdb","",regex=False)
m=mg.mut_type.str.fullmatch(r"[A-Z]\d+[A-Z]"); mg=mg[m.fillna(False)].copy()
mg["wt_aa"]=mg.mut_type.str[0]; mg["mut_aa"]=mg.mut_type.str[-1]
mg["pos"]=mg.mut_type.str[1:-1].astype(int)
mg["y"]=-pd.to_numeric(mg.ddG_ML,errors="coerce")
d=mg.dropna(subset=["y"]).groupby(["entity_id","pos","wt_aa","mut_aa"],as_index=False).y.mean()
# restrict to the 28 test proteins
test=set(pd.read_csv("/foldx_native_clean/results/foldx_native_megascale.csv").entity_id.unique())
d=d[d.entity_id.isin(test)].copy()
g=d.groupby(["entity_id","pos"])
s=g.y.transform("sum"); n=g.y.transform("count")
d["site_loo"]=(s-d.y)/(n-1)           # leave-one-out per-site mean
gm=d.groupby("mut_aa"); d["aa_loo"]=(gm.y.transform("sum")-d.y)/(gm.y.transform("count")-1)
d["both"]=d.site_loo+d.aa_loo-d.y.mean()
for k in ["site_loo","aa_loo","both"]:
    sub=d.dropna(subset=[k])
    per=[spearmanr(sub[k][sub.entity_id==p],sub.y[sub.entity_id==p]).statistic for p in sub.entity_id.unique()]
    print(f"{k:10s} pooled {spearmanr(sub[k],sub.y).statistic:.3f} per-prot med {np.nanmedian(per):.3f} n={len(sub)}")
print("n sites",d.groupby(['entity_id','pos']).ngroups,"mean muts/site",round(len(d)/d.groupby(['entity_id','pos']).ngroups,1))
