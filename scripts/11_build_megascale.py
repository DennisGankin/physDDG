#!/usr/bin/env python
"""Build the Megascale (Tsuboyama 2023) tables.

Outputs (data/processed/):
  megascale.parquet / .csv.gz   Dataset-3 single substitutions on the 298 ThermoMPNN proteins (== on-disk runtime
                                table, verified 1:1), with dG_wt, dG_mut, CIs, flags, ThermoMPNN split.
  megascale_censored.csv        single substitutions of the same proteins that Tsuboyama EXCLUDED from Dataset 3:
                                dG '<-1' (mutant fully unfolded: ddg is only a lower bound), '>5' (above the stability
                                ceiling: ddg upper bound) and '-' (unreliable). ddg = NaN; interval in ddg_lo/ddg_hi.
  megascale_doubles.csv.gz      Dataset-3 double substitutions (+ censored doubles), with the component single ddgs.

Inputs (read-only): /workspace/data/runtime_tables/megascale_runtime.csv, /workspace/data/megascale_pdbs/,
  data/raw/tsuboyama/extracted/Processed_K50_dG_datasets/{Tsuboyama2023_Dataset2_Dataset3_20230416.csv,
  Single_DMS_list.csv} (Zenodo 7992926, scripts/10), data/raw/thermompnn/mega_splits.pkl.

Sign: Tsuboyama dG = unfolding free energy, ddG_ML = dG_ML(mut) - dG_ML(WT) (negative = destabilising; verified
below). Our convention (D1): ddg = dG_wt - dG_mut = -ddG_ML. Original kept in ddG_ML_orig.
Tsuboyama dynamic range: dG in [-1, 5] kcal/mol; outside it dG_ML is reported as '<-1' / '>5' and ddG_ML as '-'.
For Dataset-3 rows dG_ML == deltaG (unclipped estimate) exactly, so there is no clipping inside Dataset 3; censoring
appears only as (a) rows excluded from Dataset 3 (-> megascale_censored.csv) and (b) values close to the limits
(flags below).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ht_common import ROOT, cb_burial, read_structure  # noqa: E402

RUNTIME = "/workspace/data/runtime_tables/megascale_runtime.csv"
PDBDIR = "/workspace/data/megascale_pdbs"
TDIR = ROOT / "data/raw/tsuboyama/extracted/Processed_K50_dG_datasets"
TSU = TDIR / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
SPLITS = ROOT / "data/raw/thermompnn/mega_splits.pkl"
OUT = ROOT / "data/processed"
LOW, HIGH = -1.0, 5.0      # Tsuboyama dG dynamic range (kcal/mol)
MARGIN = 0.5               # flag dG within this distance of a limit
CI_LOWCONF = 0.5           # 95% CI width above which a dG is called low-confidence
NUM = ["dG_ML", "ddG_ML", "deltaG", "deltaG_95CI_high", "deltaG_95CI_low", "deltaG_95CI", "deltaG_t", "deltaG_c",
       "deltaG_t_95CI", "deltaG_c_95CI", "log10_K50_t", "log10_K50_c", "log10_K50unfolded_t", "log10_K50unfolded_c"]


def log(*a):
    print(*a, flush=True)


def to_num(df):
    for c in NUM:
        df[c + "_num"] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_singles(s, wtseq):
    p = s.str.extract(r"^([A-Z])(\d+)([A-Z])$")
    assert p.notna().all().all()
    return p[0].values, p[1].astype(int).values, p[2].values


def main():
    sp = pickle.load(open(SPLITS, "rb"))
    split = {n: k for k in ("train", "val", "test") for n in sp[k]}
    names = set(split)
    assert len(names) == 298

    t = pd.read_csv(TSU, dtype=str)
    t = t[t.WT_name.isin(names) & ~t.mut_type.str.contains("ins|del")].copy()
    t = to_num(t)
    t["in_dataset3"] = t.ddG_ML != "-"
    t["n_mut"] = np.where(t.mut_type == "wt", 0, t.mut_type.str.count(":") + 1)
    log(f"Tsuboyama rows for the 298 proteins (no indels): {len(t)}; by n_mut x in_dataset3:\n"
        + pd.crosstab(t.n_mut, t.in_dataset3).to_string())

    # ---------- verify the on-disk runtime table == Dataset-3 singles + wt -----------------------------------------
    r = pd.read_csv(RUNTIME)
    d3 = t[t.in_dataset3 & (t.n_mut <= 1)]
    key = lambda df, dd: df.WT_name + "|" + df.mut_type + "|" + df.aa_seq + "|" + dd.round(6).astype(str)  # noqa: E731
    assert len(r) == len(d3) and set(key(r, r.ddG_ML)) == set(key(d3, d3.ddG_ML_num)) and not key(r, r.ddG_ML).duplicated().any()
    log(f"runtime table ({len(r)} rows, {r.WT_name.nunique()} proteins) == Tsuboyama Dataset-3 singles+wt for the 298 "
        f"ThermoMPNN proteins (1:1 on WT_name|mut_type|aa_seq|ddG_ML). Its dG_ML column is NaN for "
        f"{r.dG_ML.isna().sum()} rows -> dG taken from Zenodo.")

    # ---------- WT reference ----------------------------------------------------------------------------------------
    wt = t[(t.n_mut == 0) & t.in_dataset3]
    m3 = t[t.in_dataset3 & (t.n_mut >= 1)]
    implied = (m3.dG_ML_num - m3.ddG_ML_num).groupby(m3.WT_name)
    assert implied.std().max() < 1e-9, "ddG_ML is not dG_mut - const per protein"
    W = pd.DataFrame({"dG_wt": implied.mean()})
    W["dG_wt_rep_median"] = wt.groupby("WT_name").dG_ML_num.median()
    W["dG_wt_rep_sd"] = wt.groupby("WT_name").dG_ML_num.std()
    W["dG_wt_n_rep"] = wt.groupby("WT_name").size()
    W["dG_wt_ci95"] = wt.groupby("WT_name").deltaG_95CI_num.median()
    sdl = pd.read_csv(TDIR / "Single_DMS_list.csv").set_index("name")
    W["wt_dg_med_dmslist"] = sdl.wt_dg_med.reindex(W.index)
    log(f"dG_wt (implied by ddG_ML) vs median of WT replicate rows: max|diff| {(W.dG_wt - W.dG_wt_rep_median).abs().max():.3f}; "
        f"vs Single_DMS_list.wt_dg_med: max|diff| {(W.dG_wt - W.wt_dg_med_dmslist).abs().max():.3f} "
        f"(n={W.wt_dg_med_dmslist.notna().sum()})")
    wtseq = wt.groupby("WT_name").aa_seq.first()
    assert (wt.groupby("WT_name").aa_seq.nunique() == 1).all()

    # ---------- structures ------------------------------------------------------------------------------------------
    S = {}
    for name in names:
        st = read_structure(f"{PDBDIR}/{name}")
        ch = st[0][0]
        assert len(st[0]) == 1
        seq = "".join(__import__("gemmi").find_tabulated_residue(res.name).one_letter_code.upper() for res in ch)
        nums = [f"{res.seqid.num}{res.seqid.icode.strip()}" for res in ch]
        assert seq == wtseq[name], f"{name}: structure sequence != Tsuboyama WT aa_seq"
        S[name] = dict(chain=ch.name, nums=nums, burial=cb_burial(st, ch.name))
    log("structures: 298/298 single-chain, observed sequence identical to the WT aa_seq, numbered 1..L contiguous")

    def site(names_, pos):
        return [S[n]["nums"][p - 1] for n, p in zip(names_, pos)]

    def common(df, wt_, pos, mt):
        seqs = df.WT_name.map(wtseq)
        ok = np.array([s[p - 1] == a for s, p, a in zip(seqs, pos, wt_)])
        return ok

    domain_origin = lambda pid: np.where(pid.str.contains(r"(?:^|_)\d[A-Z0-9]{3}$") &  # noqa: E731
                                         ~pid.str.contains("rd|TrROS|HEEH|EEHEE|EHEE|HHH"), "natural", "designed")

    def base_cols(df):
        pid = df.WT_name.str.replace(r"\.pdb$", "", regex=True)
        return {
            "dataset": "megascale", "protein_id": pid.values, "structure": (PDBDIR + "/" + df.WT_name).values,
            "chain": df.WT_name.map(lambda n: S[n]["chain"]).values,
        }

    def dg_cols(df):
        w = W.reindex(df.WT_name)
        return {
            "dG_wt": w.dG_wt.values, "dG_wt_ci95": w.dG_wt_ci95.values, "dG_wt_rep_sd": w.dG_wt_rep_sd.values,
            "dG_wt_n_rep": w.dG_wt_n_rep.values,
            "dG_mut": df.dG_ML_num.values,                      # NaN for '<-1' / '>5' / '-'
            "dG_mut_label": df.dG_ML.values,                    # raw Tsuboyama string ('<-1', '>5', '-' or number)
            "deltaG_unclipped": df.deltaG_num.values,           # Tsuboyama 'deltaG' (beyond range = unreliable)
            "dG_mut_ci95": df.deltaG_95CI_num.values,           # width of 95% CI (high - low)
            "dG_mut_ci_low": df.deltaG_95CI_low_num.values, "dG_mut_ci_high": df.deltaG_95CI_high_num.values,
            "dG_t": df.deltaG_t_num.values, "dG_c": df.deltaG_c_num.values,
            "dG_t_ci95": df.deltaG_t_95CI_num.values, "dG_c_ci95": df.deltaG_c_95CI_num.values,
            "log10_K50_t": df.log10_K50_t_num.values, "log10_K50_c": df.log10_K50_c_num.values,
            "log10_K50U_t": df.log10_K50unfolded_t_num.values, "log10_K50U_c": df.log10_K50unfolded_c_num.values,
        }

    def flags(o):
        o["flag_lowconf"] = (o.dG_mut_ci95 > CI_LOWCONF) | (o.dG_wt_ci95 > CI_LOWCONF)
        o["flag_mut_floor"] = o.dG_mut < LOW + MARGIN       # near unfolded floor: ddg ~ lower bound
        o["flag_mut_ceiling"] = o.dG_mut > HIGH - MARGIN    # near stability ceiling: stabilisation capped
        o["flag_wt_high"] = o.dG_wt > HIGH - 1.0            # WT dG > 4: stabilising mutations easily censored
        o["flag_wt_low"] = o.dG_wt < LOW + 2.0              # WT dG < 1: destabilising mutations easily censored
        o["flag_protease_disagree"] = (o.dG_t - o.dG_c).abs() > 1.0
        return o

    # ---------- singles (Dataset 3 and censored) --------------------------------------------------------------------
    singles = t[t.n_mut == 1].copy()
    singles["wt"], singles["seq_pos"], singles["mut"] = parse_singles(singles.mut_type, wtseq)
    ok = common(singles, singles.wt, singles.seq_pos, singles.mut)
    ok_seq = np.array([m == s[:p - 1] + b + s[p:] for m, s, p, b in
                       zip(singles.aa_seq, singles.WT_name.map(wtseq), singles.seq_pos, singles.mut)])
    log(f"singles: WT letter matches WT sequence {ok.mean():.4%}; mutant sequence = WT + that change {ok_seq.mean():.4%}")
    assert ok.all() and ok_seq.all()
    assert not singles.duplicated(["WT_name", "seq_pos", "mut"]).any()
    log("singles: no duplicate (protein, position, mutant) entries (incl. censored) -> nothing to average")
    o = pd.DataFrame({
        **base_cols(singles), "pdb_resnum": site(singles.WT_name, singles.seq_pos),
        "wt": singles.wt.values, "mut": singles.mut.values, "seq_pos": singles.seq_pos.values,
        "ddg": -singles.ddG_ML_num.values, "split": singles.WT_name.map(split).values, "cluster": "",
        "ddG_ML_orig": singles.ddG_ML_num.values, "in_dataset3": singles.in_dataset3.values, **dg_cols(singles),
        "stabilizing_mut_tsuboyama": singles.Stabilizing_mut.values,
        "in_train_s669": singles.WT_name.isin(set(sp["train_s669"])).values,  # ThermoMPNN train minus S669 homologues
        "WT_cluster_tsuboyama": singles.WT_cluster.values,
        "cb_nb": [S[n]["burial"][S[n]["nums"][p - 1]] for n, p in zip(singles.WT_name, singles.seq_pos)],
        "struct_source": "af2_model_tsuboyama",
        "source": "Tsuboyama2023 (zenodo 7992926) Dataset3; ThermoMPNN mega split",
    })
    o["domain_origin"] = domain_origin(o.protein_id)
    o = flags(o)
    # censoring interval in our convention (ddg = dG_wt - dG_mut)
    o["censor"] = np.select([o.in_dataset3, o.dG_mut_label == "<-1", o.dG_mut_label == ">5"],
                            ["none", "mut_unfolded_lt-1", "mut_stable_gt5"], "unreliable")
    o["ddg_lo"] = np.select([o.censor == "none", o.censor == "mut_unfolded_lt-1"], [o.ddg, o.dG_wt - LOW], np.nan)
    o["ddg_hi"] = np.select([o.censor == "none", o.censor == "mut_stable_gt5"], [o.ddg, o.dG_wt - HIGH], np.nan)
    o.loc[o.censor == "mut_unfolded_lt-1", "ddg_hi"] = np.inf
    o.loc[o.censor == "mut_stable_gt5", "ddg_lo"] = -np.inf

    # sign evidence
    stab = pd.crosstab(o.stabilizing_mut_tsuboyama[o.in_dataset3], o.ddG_ML_orig[o.in_dataset3] > 0)
    log("sign: Tsuboyama Stabilizing_mut vs (ddG_ML > 0):\n" + stab.to_string())
    hyd = o.wt.isin(list("ILVFMWY")) & o.mut.isin(list("AGPDK"))
    for lab, m in [("buried(>=16) hydrophobic->AGPDK", hyd & (o.cb_nb >= 16)),
                   ("exposed(<=8) hydrophobic->AGPDK", hyd & (o.cb_nb <= 8)), ("any->P", o.mut == "P"),
                   ("all", o.ddg.notna())]:
        x = o.ddg[m & o.in_dataset3]
        log(f"  sign {lab:34s} n={len(x):6d} mean ddg={x.mean():+.2f} frac>0={(x > 0).mean():.2f}")
    cz = o[~o.in_dataset3]
    log(f"censored singles: {len(cz)} rows: " + str(cz.censor.value_counts().to_dict()) +
        f"; buried(>=16) share among '<-1': {(cz[cz.censor == 'mut_unfolded_lt-1'].cb_nb >= 16).mean():.2f} vs "
        f"Dataset3: {(o[o.in_dataset3].cb_nb >= 16).mean():.2f}")

    main_t = o[o.in_dataset3].drop(columns=["in_dataset3"]).sort_values(["split", "protein_id", "seq_pos", "mut"])
    main_t = main_t.reset_index(drop=True)
    assert len(main_t) == len(r) - (r.mut_type == "wt").sum()
    OUT.mkdir(parents=True, exist_ok=True)
    main_t.to_parquet(OUT / "megascale.parquet", index=False)
    main_t.to_csv(OUT / "megascale.csv.gz", index=False)
    cen = o[~o.in_dataset3].sort_values(["split", "protein_id", "seq_pos", "mut"]).reset_index(drop=True)
    cen.to_csv(OUT / "megascale_censored.csv", index=False)

    # ---------- doubles ----------------------------------------------------------------------------------------------
    dbl = t[t.n_mut == 2].copy()
    pp = dbl.mut_type.str.split(":", expand=True)
    syn = (pp[0].str[0] == pp[0].str[-1]) | (pp[1].str[0] == pp[1].str[-1])
    log(f"doubles: {len(dbl)} rows; dropping {int(syn.sum())} with a synonymous component (e.g. E24E:S41R)")
    dbl = dbl[~syn]
    # replicate measurements of the same double (different library/codons): Dataset-3 replicates averaged
    n0 = len(dbl)
    dbl["rank"] = (~dbl.in_dataset3).astype(int)
    dbl = dbl.sort_values(["WT_name", "mut_type", "rank"])
    numc = [c for c in dbl.columns if c.endswith("_num")]
    g = dbl.groupby(["WT_name", "mut_type"], sort=False)
    first = g.first()
    d3m = dbl[dbl.in_dataset3].groupby(["WT_name", "mut_type"])[numc].mean()
    first.loc[d3m.index, numc] = d3m.values
    first["n_meas"] = g.size()
    first["n_meas_dataset3"] = g.in_dataset3.sum()
    first["in_dataset3"] = first.n_meas_dataset3 > 0
    dbl = first.reset_index()
    log(f"doubles: {n0} -> {len(dbl)} unique (protein, double) after averaging replicates "
        f"({int((dbl.n_meas > 1).sum())} had >1 measurement)")
    parts = dbl.mut_type.str.split(":", expand=True)
    rows = {}
    for k in (0, 1):
        w_, p_, m_ = parse_singles(parts[k], wtseq)
        assert common(dbl, w_, p_, m_).all()
        rows[k] = (w_, p_, m_)
    sd = o.set_index(["protein_id", "seq_pos", "mut"])
    pid = dbl.WT_name.str.replace(r"\.pdb$", "", regex=True)
    dd = pd.DataFrame({
        **base_cols(dbl), "mut_type": dbl.mut_type.values,
        "wt1": rows[0][0], "seq_pos1": rows[0][1], "mut1": rows[0][2], "pdb_resnum1": site(dbl.WT_name, rows[0][1]),
        "wt2": rows[1][0], "seq_pos2": rows[1][1], "mut2": rows[1][2], "pdb_resnum2": site(dbl.WT_name, rows[1][1]),
        "ddg": -dbl.ddG_ML_num.values, "split": dbl.WT_name.map(split).values, "in_dataset3": dbl.in_dataset3.values,
        "n_meas": dbl.n_meas.values, "n_meas_dataset3": dbl.n_meas_dataset3.values,
        **dg_cols(dbl),
    })
    for k in (1, 2):
        idx = pd.MultiIndex.from_arrays([pid.values, dd[f"seq_pos{k}"].values, dd[f"mut{k}"].values])
        dd[f"ddg{k}"] = sd.ddg.reindex(idx).values
        dd[f"censor{k}"] = sd.censor.reindex(idx).values
    dd["ddg_additive"] = dd.ddg1 + dd.ddg2
    dd["domain_origin"] = domain_origin(dd.protein_id)
    dd = flags(dd)
    dd["censor"] = np.select([dd.in_dataset3, dd.dG_mut_label == "<-1", dd.dG_mut_label == ">5"],
                             ["none", "mut_unfolded_lt-1", "mut_stable_gt5"], "unreliable")
    assert not dd.duplicated(["protein_id", "mut_type"]).any()
    dd.to_csv(OUT / "megascale_doubles.csv.gz", index=False)
    ok = dd[dd.in_dataset3 & dd.ddg1.notna() & dd.ddg2.notna()]
    log(f"doubles: {len(dd)} rows ({int(dd.in_dataset3.sum())} Dataset3), {dd.protein_id.nunique()} proteins; "
        f"both singles available for {len(ok)}; Pearson(ddg, ddg1+ddg2) = {np.corrcoef(ok.ddg, ok.ddg_additive)[0, 1]:.3f}")

    # ---------- summary ----------------------------------------------------------------------------------------------
    s = main_t.groupby("split").agg(rows=("ddg", "size"), proteins=("protein_id", "nunique"),
                                    natural=("domain_origin", lambda x: (x == "natural").mean()),
                                    ddg_mean=("ddg", "mean"), ddg_sd=("ddg", "std"),
                                    frac_gt1=("ddg", lambda x: (x > 1).mean()),
                                    floor=("flag_mut_floor", "mean"), ceiling=("flag_mut_ceiling", "mean"),
                                    lowconf=("flag_lowconf", "mean"))
    log("megascale.parquet by split:\n" + s.round(3).to_string())
    log(f"total {len(main_t)} rows, {main_t.protein_id.nunique()} proteins; domains: "
        f"{main_t.drop_duplicates('protein_id').domain_origin.value_counts().to_dict()}")
    log(f"censored file by split: {cen.groupby('split').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
