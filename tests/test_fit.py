import numpy as np
import pandas as pd

from physddg import fit as FT

AA = list("ACDEFGHIKLMNPQRSTVWY")


def _synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"dataset": "A", "protein_id": rng.integers(0, 30, n).astype(str), "wt": rng.choice(AA, n),
                       "mut": rng.choice(AA, n), "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    df["cluster"] = df.protein_id.astype(int) // 2
    r = {a: rng.normal() * 0.5 for a in AA}
    df["truth"] = 1.5 * df.f1 - 0.7 * df.f2 + 0.3 + df.mut.map(r) - df.wt.map(r)
    df["y"] = df.truth + rng.normal(size=n) * 0.5
    c = df.y > 1.5
    df["lo"], df["hi"] = np.where(c, 1.5, np.nan), np.where(c, np.inf, np.nan)
    df.loc[c, "y"] = np.nan
    return df, r


def test_censored_recovery_and_predict():
    df, r = _synth()
    m = FT.fit(df, FT.FitConfig(["f1", "f2"], ref=True, ref_l2=1e-5))
    assert abs(m["weights"]["f1"] - 1.5) < 0.05 and abs(m["weights"]["f2"] + 0.7) < 0.05
    m_nc = FT.fit(df, FT.FitConfig(["f1", "f2"], ref=True, ref_l2=1e-5, censor=False))
    assert m_nc["weights"]["f1"] < m["weights"]["f1"]            # dropping censored rows biases towards 0
    p = FT.predict(m, df)
    assert np.corrcoef(p, df.truth)[0, 1] > 0.99


def test_sign_constraint_scale_offset_and_determinism():
    df, _ = _synth()
    m = FT.fit(df, FT.FitConfig(["f1", "f2"], signs={"f2": 1}))
    assert m["weights"]["f2"] == 0.0
    d2 = df.sample(2000, random_state=1).assign(dataset="B", lo=np.nan, hi=np.nan)
    d2["y"] = 2 * d2.truth + 1
    a = FT.fit(pd.concat([df, d2], ignore_index=True), FT.FitConfig(["f1", "f2"], ref=True, ref_dataset="A"))
    b = FT.fit(pd.concat([df, d2], ignore_index=True), FT.FitConfig(["f1", "f2"], ref=True, ref_dataset="A"))
    assert abs(a["nuisance"]["scale"]["B"] - 2) < 0.05 and abs(a["nuisance"]["offset"]["B"] - 1) < 0.1
    assert a["weights"] == b["weights"]
