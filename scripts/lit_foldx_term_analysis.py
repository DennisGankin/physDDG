"""Which FoldX 5.1 energy terms carry ddG signal? (literature-support analysis)

Uses precomputed native FoldX 5.1 per-term decompositions (read-only, /foldx_native_clean)
and the Megascale (Tsuboyama 2023) ddG_ML table in /workspace/data/runtime_tables.

Outputs a small table to results/lit_foldx_terms/foldx_term_signal.csv and prints it.
Run:  PYTHONPATH=. .venv/bin/python scripts/lit_foldx_term_analysis.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FOLDX = "/foldx_native_clean/results/foldx_native_megascale.csv"
MEGA = "/workspace/data/runtime_tables/megascale_runtime.csv"
OUT = "results/lit_foldx_terms"

TERMS = [
    "backbone_hbond", "sidechain_hbond", "van_der_waals", "electrostatics",
    "solvation_polar", "solvation_hydrophobic", "van_der_waals_clashes",
    "entropy_sidechain", "entropy_mainchain", "sloop_entropy", "mloop_entropy",
    "cis_bond", "torsional_clash", "backbone_clash", "helix_dipole",
    "water_bridge", "disulfide", "electrostatic_kon", "partial_covalent_bonds",
    "energy_ionisation", "entropy_complex",
]


def main():
    fx = pd.read_csv(FOLDX)
    mg = pd.read_csv(MEGA)
    mg = mg[mg.mut_type != "wt"].copy()
    mg["entity_id"] = mg.WT_name.str.replace(".pdb", "", regex=False)
    # mut_type like "E1Q"; keep single substitutions only
    m = mg.mut_type.str.fullmatch(r"[A-Z]\d+[A-Z]")
    mg = mg[m.fillna(False)].copy()
    mg["wt_aa"] = mg.mut_type.str[0]
    mg["mut_aa"] = mg.mut_type.str[-1]
    mg["position_1based"] = mg.mut_type.str[1:-1].astype(int)
    mg["ddg_exp"] = -pd.to_numeric(mg.ddG_ML, errors="coerce")  # D1: destabilising positive
    mg = mg.dropna(subset=["ddg_exp"])
    mg = mg.groupby(["entity_id", "position_1based", "wt_aa", "mut_aa"], as_index=False).ddg_exp.mean()

    df = fx.merge(mg, on=["entity_id", "position_1based", "wt_aa", "mut_aa"], how="inner")
    print(f"merged {len(df)} variants, {df.entity_id.nunique()} proteins")

    rows = []
    for t in ["foldx_ddg"] + TERMS:
        x = df[t].to_numpy(float)
        if np.nanstd(x) == 0:
            rows.append(dict(term=t, frac_nonzero=0.0, spearman=np.nan, sd=0.0))
            continue
        rho = spearmanr(x, df.ddg_exp, nan_policy="omit").statistic
        rows.append(dict(term=t, frac_nonzero=float(np.mean(np.abs(x) > 1e-6)),
                         sd=float(np.nanstd(x)), spearman=float(rho)))
    res = pd.DataFrame(rows).sort_values("spearman", key=lambda s: -s.abs())

    # greedy forward selection on a plain linear fit (pooled Spearman, 5-fold grouped by protein)
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    X = df[TERMS].to_numpy(float)
    y = df.ddg_exp.to_numpy(float)
    g = df.entity_id.to_numpy()
    gkf = GroupKFold(n_splits=5)

    def cv_rho(cols):
        pred = np.zeros_like(y)
        for tr, te in gkf.split(X, y, g):
            lr = LinearRegression().fit(X[np.ix_(tr, cols)], y[tr])
            pred[te] = lr.predict(X[np.ix_(te, cols)])
        return spearmanr(pred, y).statistic

    chosen, hist = [], []
    remaining = list(range(len(TERMS)))
    best_so_far = -1
    for _ in range(min(10, len(TERMS))):
        scored = [(cv_rho(chosen + [c]), c) for c in remaining]
        scored.sort(reverse=True)
        r, c = scored[0]
        if r < best_so_far + 0.001:
            break
        best_so_far = r
        chosen.append(c)
        remaining.remove(c)
        hist.append((TERMS[c], round(r, 4)))
        print(f"  + {TERMS[c]:28s} cv pooled rho = {r:.4f}")

    os.makedirs(OUT, exist_ok=True)
    res.to_csv(f"{OUT}/foldx_term_signal.csv", index=False)
    pd.DataFrame(hist, columns=["term_added", "cv_pooled_spearman"]).to_csv(
        f"{OUT}/foldx_term_forward_selection.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
