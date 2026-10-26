import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physddg import eval as E  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def synth(n_prot=12, n=40, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prot):
        for i in range(n):
            wt, mut = rng.choice(list(AA), 2, replace=False)
            rows.append(dict(protein_id=f"P{p}", chain="A", pdb_resnum=str(i + 1) + ("A" if i == 5 else ""),
                             wt=wt, mut=mut, ddg=rng.normal(1, 1.5), split="test" if p % 2 else "train",
                             cluster=f"c{p // 2}", plddt=rng.uniform(40, 100)))
    d = pd.DataFrame(rows)
    d["good"] = d.ddg + rng.normal(0, noise, len(d))
    d["bad"] = d.ddg + rng.normal(0, 3 * noise + 1, len(d))
    return d


def test_perfect_and_calibration():
    d = synth()
    d["pred"] = 2 * d.ddg + 1  # monotone + linear but badly calibrated
    m = E.evaluate(d, "pred")
    assert m["spearman"] == pytest.approx(1) and m["pearson"] == pytest.approx(1)
    assert m["rmse_fit"] == pytest.approx(0, abs=1e-9) and m["rmse"] > 1
    assert m["spearman_prot_median"] == pytest.approx(1) and m["n_prot_eval"] == 12
    assert m["auroc_destab"] == pytest.approx(1)  # rank-based, calibration-free
    assert m["sign_acc"] < 1  # raw sign uses uncalibrated values (documented)


def test_auroc_matches_sklearn_and_ties():
    d = synth()
    m = E.evaluate(d, "good")
    assert m["auroc_destab"] == pytest.approx(roc_auc_score(d.ddg > 1, d.good))
    assert m["auroc_stab"] == pytest.approx(roc_auc_score(d.ddg < -0.5, -d.good))
    assert E.auroc(np.zeros(6), np.array([1, 0, 1, 0, 1, 0], bool)) == 0.5
    assert np.isnan(E.auroc(np.arange(3.0), np.zeros(3, bool)))


def test_min_n_per_protein():
    d = synth(n_prot=3, n=40)
    d = pd.concat([d, d[d.protein_id == "P0"].head(5).assign(protein_id="tiny")])
    m = E.evaluate(d, "good", min_n=10)
    assert m["n_proteins"] == 4 and m["n_prot_eval"] == 3
    ref = np.mean([spearmanr(g.good, g.ddg)[0] for p, g in d.groupby("protein_id") if p != "tiny"])
    assert m["spearman_prot_mean"] == pytest.approx(ref)


def test_score_proxy_uses_negative_score():
    d = synth()
    d["score"] = -d.ddg  # fitness: lower = destabilising
    d["ddg"] = np.nan
    m = E.evaluate(d, "good")
    assert m["target"] == "score" and m["spearman"] > 0.8
    assert np.isnan(m["rmse"]) and np.isnan(m["auroc_destab"])  # no ddg-scale metrics for proxies
    assert m["spearman_prot_median"] > 0.8


def test_merge_full_coverage_and_float_resnum():
    d = synth()
    p = d[list(E.KEYS)].assign(ddg_pred=1.0).copy()
    ins = p.pdb_resnum.str.endswith("A")
    p["pdb_resnum"] = np.where(ins, p.pdb_resnum, d.pdb_resnum + ".0")  # float-typed columns give '4.0'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = E.merge_predictions(d, p)
    assert m.attrs["merge_info"]["coverage"] == 1 and len(m) == len(d)


@pytest.mark.parametrize("k", [1, -1, 2])
def test_offset_detection(k):
    d = synth()
    d["pdb_resnum"] = d.pdb_resnum.str.replace("A", "")  # plain ints
    p = d[list(E.KEYS)].assign(ddg_pred=d.good)
    p["pdb_resnum"] = (p.pdb_resnum.astype(int) + k).astype(str)  # buggy predictor numbering
    with pytest.warns(UserWarning, match="OFFSET"):
        m = E.merge_predictions(d, p)
    off = m.attrs["merge_info"]["offset"]
    assert off["shift"] == -k and off["coverage_shifted"] > 0.9  # pred_resnum + shift matches


def test_low_coverage_without_offset_is_flagged_but_no_offset():
    d = synth()
    p = d[list(E.KEYS)].assign(ddg_pred=1.0).iloc[::3]
    with pytest.warns(UserWarning, match="LOW COVERAGE"):
        m = E.merge_predictions(d, p)
    assert m.attrs["merge_info"]["offset"] is None and len(m) == pytest.approx(len(d) / 3, abs=1)


def test_subset_and_bins():
    d = synth()
    d["strict"] = d.protein_id.isin(["P1", "P3"])
    assert set(E.apply_subset(d, "split==test").split) == {"test"}
    assert set(E.apply_subset(d, "strict==true").protein_id) == {"P1", "P3"}
    s = E.apply_subset(d, "split==test,plddt>=70")
    assert (s.plddt >= 70).all() and (s.split == "test").all()
    b = E.add_bins(d, "plddt", [50, 70, 90])
    assert b.plddt_bin.nunique() == 4 and b.loc[b.plddt < 50, "plddt_bin"].eq("[-inf,50)").all()


def test_bootstrap_ci_and_paired():
    d = synth(n_prot=20)
    ci = E.bootstrap_ci(d, "good", n_boot=200)
    m = E.evaluate(d, "good")
    assert ci["spearman"][0] < m["spearman"] < ci["spearman"][1]
    pb = E.paired_bootstrap(d, "good", "bad", n_boot=200)
    assert pb["spearman"]["delta"] > 0 and pb["spearman"]["lo"] > 0 and pb["spearman"]["p"] < 0.05
    assert pb["rmse"]["delta"] < 0  # lower is better
    same = E.paired_bootstrap(d, "good", "good", n_boot=50)
    assert same["spearman"]["delta"] == 0 and same["spearman"]["p"] == 1


def test_bootstrap_uses_clusters_and_blank_cluster():
    d = synth()
    b = E._Boot(d, "good", None, "ddg", "score", "protein_id", "cluster", 10)
    assert len(b.groups) == 6  # 12 proteins in 6 clusters
    d.loc[d.protein_id == "P0", "cluster"] = ""  # blank -> own cluster
    d.loc[d.protein_id == "P1", "cluster"] = np.nan
    assert len(E._Boot(d, "good", None, "ddg", "score", "protein_id", "cluster", 10).groups) == 5 + 2


def _pairs(n=200, bias=0.0, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(1, 2, n)
    d = pd.DataFrame({"pair_id": np.r_[np.arange(n), np.arange(n)], "direction": ["direct"] * n + ["reverse"] * n,
                      "ddg": np.r_[x, -x]})
    d["pred"] = d.ddg + np.where(d.direction == "direct", bias / 2, bias / 2)
    return d


def test_antisymmetry_perfect_and_biased():
    a = E.antisymmetry(_pairs())
    assert a["n_pairs"] == 200 and a["r_pearson"] == pytest.approx(1) and a["bias"] == pytest.approx(0)
    b = E.antisymmetry(_pairs(bias=0.8))
    assert b["r_pearson"] == pytest.approx(1) and b["bias"] == pytest.approx(0.8)  # shift = bias, corr unaffected
    d = _pairs()
    d.loc[d.direction == "reverse", "pred"] = np.random.default_rng(3).normal(size=200)  # unrelated reverse
    assert abs(E.antisymmetry(d)["r_pearson"]) < 0.3
    d["direction"] = d.direction.map({"direct": "dir", "reverse": "rev"})
    assert E.antisymmetry(_pairs(), )["n_pairs"] == E.antisymmetry(d)["n_pairs"]
    with pytest.raises(ValueError):
        E.antisymmetry(d.assign(direction="x"))


def test_load_benchmark(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    d = synth(2, 5)
    d.to_csv(tmp_path / "data" / "processed" / "toy.csv", index=False)
    assert len(E.load_benchmark("toy", tmp_path)) == 10
    with pytest.raises(FileNotFoundError):
        E.load_benchmark("nope", tmp_path)
