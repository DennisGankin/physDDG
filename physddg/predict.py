"""PhysDDG predictor: structure + mutations -> ddG (kcal/mol, positive = destabilising).

    from physddg.predict import PhysDDG
    m = PhysDDG()                                   # B4 (stage 1) weights; PhysDDG(stage=2) -> B5
    df = m.predict("protein.pdb", ["A12G", "L45P"])  # or m.predict("x.cif", scan_all=True, sites="10-80")

Pipeline (identical code to the training features):
  1. structure.load_protein + annotate (CPU)
  2. engine features: mutate.SiteScan.stage1 / stage2 over the needed sites, all 20 residue types (GPU),
     -> `site_arrays` (same arrays scripts/55_phys_features.py caches) -> `engine_columns` (same columns)
  3. structure-statistics features: features_simple.site_table + variant_features (fitting agent's code)
  4. model transform (`derive`, mirrors scripts/52_ladder_physics.py) and linear model (physddg.fit.predict):
     ddg = intercept + sum_k w_k x_k + ref[mut] - ref[wt]; WT->WT is defined as exactly 0.
Per-term contributions w_k x_k (+ reference-energy and intercept terms) are returned with `terms=True`.
Flags (column `flags`, ';'-separated; empty = none): wt_mismatch, site_missing_atoms, missing_atoms_8A,
hetatm_5A, chain_break_10A, terminal (<= 2 residues from a chain end), low_plddt (<70, mode='af' only), and in AF mode
the AF-mode agent's flags (physddg.afmode.flags) when that module is available.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import topology as T

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "ladder_physics" / "models"
DEFAULT_WEIGHTS = {1: MODELS / "B4.json", 2: MODELS / "B5.json"}   # legacy; see default_weights()
KEYS = ["protein_id", "chain", "pdb_resnum", "wt", "mut"]
KT = 0.593
TERMS = ["lj_atr", "lj_rep", "solv_apolar", "solv_polar", "hb_sc_bb", "hb_sc_sc", "hb_bb_bb", "ELEC", "disulf"]
_MUT = re.compile(r"^(?:([A-Za-z0-9]+):)?([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$")


# ============================================================================= engine arrays / columns (shared with 55)
def site_arrays(prot, idx: np.ndarray, sc, stage: int) -> dict:
    """Per-site arrays for sites idx (all 20 aa): engine results + statistical terms + context. Exactly what
    scripts/55_phys_features.py caches per structure."""
    from .mutate import stat_features
    res = sc.stage1(idx) if stage == 1 else sc.stage2(idx)
    st = stat_features(prot, idx, sc.cfg.kT)
    z = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    z.update({f"stat_{k}": v for k, v in st["per_aa"].items()})
    z.update({f"ctx_{k}": np.asarray(v) for k, v in st["context"].items()})
    z.update(chain=prot.chain[idx], resnum=prot.resnum[idx], aa=np.array(list(prot.seq))[idx],
             terms=np.array(sc.terms))
    return z


def engine_columns(z: dict, g: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Variant columns d<x> = x(site, mut) - x(site, wt) for rows g (chain, pdb_resnum, wt, mut; index kept)."""
    pre = "s1" if stage == 1 else "s2"
    out = pd.DataFrame(index=g.index)
    if "chain" not in z:
        out["site_found"] = False
        return out
    key = {(c, r): i for i, (c, r) in enumerate(zip(z["chain"].tolist(), z["resnum"].tolist()))}
    si = np.array([key.get((c, r), -1) for c, r in zip(g.chain, g.pdb_resnum)])
    ok = si >= 0
    iw = g.wt.map(T.AA_INDEX).to_numpy()
    im = g.mut.map(T.AA_INDEX).to_numpy()
    s_, w_, m_ = si[ok], iw[ok], im[ok]
    out["site_found"] = ok
    out["wt_match"] = False
    out.loc[ok, "wt_match"] = z["aa"][s_] == g.wt.to_numpy()[ok]
    cols = {}

    def d(arr):
        return arr[s_, m_] - arr[s_, w_]

    terms = [str(t) for t in z["terms"]]
    cols[f"{pre}_dF"] = d(z["F"])
    for k in ("U_min", "prior_avg"):
        if k in z:
            cols[f"{pre}_d{k}"] = d(z[k])
    if "S_site" in z:
        cols[f"{pre}_dTS"] = KT * d(z["S_site"])
    if "S_local" in z:                                        # stage 2: mean-field entropy of the local region
        cols[f"{pre}_dTS"] = KT * d(z["S_local"])
    if "n_nb" in z:
        cols[f"{pre}_n_nb"] = z["n_nb"][s_]
        cols[f"{pre}_nb_native_frac_wt"] = z["nb_native_frac"][s_, w_]
    for j, t in enumerate(terms):
        cols[f"{pre}_dE_{t}"] = d(z["E_avg"][..., j])
        if "E_min" in z:
            cols[f"{pre}_dEmin_{t}"] = d(z["E_min"][..., j])
    cols[f"{pre}_dref_apolar"] = d(z["ref"][..., 0])
    cols[f"{pre}_dref_polar"] = d(z["ref"][..., 1])
    if "n_rot" in z:
        cols[f"{pre}_nrot_wt"] = z["n_rot"][s_, w_]
        cols[f"{pre}_nrot_mut"] = z["n_rot"][s_, m_]
    if "F_env" in z:     # stage 2: site scored in frozen mean-field environments (rows: env state, cols: residue)
        def fz(arr):     # forward wt->mut in the WT-relaxed environment
            return arr[s_, w_, m_] - arr[s_, w_, w_]

        def rv(arr):     # reverse mut->wt in the modelled mutant environment
            return arr[s_, m_, w_] - arr[s_, m_, m_]
        for tag, fn in (("fz", fz), ("rev", rv)):
            cols[f"{tag}_{pre}_dF"] = fn(z["F_env"])
            cols[f"{tag}_{pre}_dprior_avg"] = fn(z["prior_env"])
            cols[f"{tag}_{pre}_dTS"] = KT * fn(z["S_env"])
            for j, t in enumerate(terms):
                cols[f"{tag}_{pre}_dE_{t}"] = fn(z["E_env"][..., j])
        cols[f"sym_{pre}_dF"] = 0.5 * (cols[f"fz_{pre}_dF"] - cols[f"rev_{pre}_dF"])
    for k in z:
        if k.startswith("stat_"):
            cols["d_" + k[5:]] = d(z[k])
        elif k.startswith("ctx_"):
            cols["ctx_" + k[4:]] = z[k][s_]
    for c, v in cols.items():
        out[c] = np.nan if v.dtype.kind == "f" else None
        out.loc[ok, c] = v
    return out


# ============================================================================= model transform (mirrors 52.derive)
def derive(d: pd.DataFrame, stage: int, opts: dict, caps: dict | None = None) -> pd.DataFrame:
    """Model inputs from feature columns; mirrors scripts/52_ladder_physics.derive. Cap variants (opts['cap']):
    the engine is run with that variant's MutateConfig, so its columns carry no suffix here (52 merges several
    variants side by side with a `_<cap>` suffix; both resolve to the same numbers)."""
    d = d.copy()
    s = f"s{stage}"
    cap = opts.get("cap", "")
    sfx = f"_{cap}" if cap else ""
    mid = "dEmin" if (opts.get("avg") == "min" and stage == 1) else "dE"
    tr = opts["transform"]
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
    for t in TERMS:
        name = f"elec_{opts['elec']}" if t == "ELEC" else t
        src = f"{s}_{mid}_{name}"
        if t == "lj_rep" and opts.get("rep") == "soft":
            src = f"{s}_{mid}_lj_rep_soft{sfx}" if f"{s}_{mid}_lj_rep_soft{sfx}" in d else f"{s}_{mid}_lj_rep_soft"
        elif cap and f"{src}{sfx}" in d:
            src = f"{src}{sfx}"
        x = d[src].astype(float)
        if opts.get("rep") == "soft" and t == "lj_rep":
            pass                                           # already saturating: no extra transform
        elif tr == "slog_all" or (tr == "slog_rep" and t == "lj_rep"):
            x = slog(x)
        elif tr.startswith("clip_rep") and t == "lj_rep":
            c = float(tr[8:])
            x = x.clip(-c, c)
        elif tr == "clip_pct" and caps:
            x = x.clip(*caps[name])
        d[f"e_{name}"] = x
    if opts.get("pro") == "split":
        isp = ((d.wt == "P") | (d.mut == "P")).astype(float)
        d["e_lj_rep_pro"] = d["e_lj_rep"] * isp
        d["e_lj_rep"] = d["e_lj_rep"] * (1 - isp)
    d["dTS"], d["dprior_avg"] = d[f"{s}_dTS"], d[f"{s}_dprior_avg"]
    d["dref_apolar"], d["dref_polar"] = d[f"{s}_dref_apolar"], d[f"{s}_dref_polar"]
    fcol = f"{s}_dF{sfx}" if (cap and f"{s}_dF{sfx}" in d) else f"{s}_dF"
    d["dF"] = slog(d[fcol]) if tr != "none" else d[fcol]
    if "ss_broken" in d:
        d["ss_broken"] = d["ss_broken"].fillna(0)
        d["ss_broken_nc"] = d["ss_broken"] * (~d.get("cxxc", pd.Series(False, index=d.index)).astype(bool))
    return d


def cxxc(prot, rows: pd.DataFrame) -> pd.Series:
    """WT Cys in a C-X-X-C motif (only used by the test-informed '+C' model variants)."""
    key = {(c, r): i for i, (c, r) in enumerate(zip(prot.chain.tolist(), prot.resnum.tolist()))}
    out = []
    for c, r, w in zip(rows.chain, rows.pdb_resnum, rows.wt):
        i = key.get((c, r), -1)
        out.append(w == "C" and i >= 0 and any(0 <= i + k < len(prot) and prot.seq[i + k] == "C"
                                                and prot.chain[i + k] == c for k in (-3, 3)))
    return pd.Series(out, index=rows.index)


# ============================================================================= flags
def site_flags(prot, idx: np.ndarray, mode: str = "xtal") -> tuple[list[str], list[bool]]:
    """Per-site warning flags (see module docstring) and the AF-mode `withheld` indicator
    (pLDDT < 70 | terminal | exposed hydrophobic patch | metal site; always False in xtal mode)."""
    a = prot.ann
    ca = np.nan_to_num(prot.xyz14[:, 1])
    from .structure import cb_coords
    cb = np.nan_to_num(cb_coords(prot))
    miss = prot.missing_sc | prot.missing_o
    br = a["chain_break"].copy()
    starts = np.r_[True, prot.chain[1:] != prot.chain[:-1]]
    brk_res = (br & ~starts)
    brk_res = brk_res | np.r_[brk_res[1:], False]                 # both residues flanking a break
    ends = starts | np.r_[starts[1:], True]
    end_idx = np.flatnonzero(ends)
    het = prot.het_xyz[~np.isin(prot.het_resname, ["HOH", "WAT", "DOD"])] if len(prot.het_xyz) else np.zeros((0, 3))
    af = None
    if mode == "af":
        from . import afmode
        af = afmode.flags(prot, np.asarray(idx))
    out, withheld = [], []
    for k, i in enumerate(idx):
        f = []
        if prot.missing_sc[i] or not prot.mask14[i, :4].all():
            f.append("site_missing_atoms")
        dcb = np.linalg.norm(cb - cb[i], axis=1)
        if (miss & (dcb < 8.0) & (np.arange(len(prot)) != i)).any():
            f.append("missing_atoms_8A")
        if len(het):
            xs = prot.xyz14[i][prot.mask14[i]]
            if (np.linalg.norm(het[:, None] - xs[None], axis=-1) < 5.0).any():
                f.append("hetatm_5A")
        if (brk_res & (np.linalg.norm(ca - ca[i], axis=1) < 10.0)).any():
            f.append("chain_break_10A")
        if af is None:
            same = prot.chain[end_idx] == prot.chain[i]
            if len(end_idx) and (np.abs(end_idx[same] - i) <= 2).any():
                f.append("terminal")
            withheld.append(False)
        else:
            names = ("low_plddt", "very_low_plddt", "near_low_plddt", "terminal", "exposed_hydrophobic_patch", "metal_site")
            f += [n for n in names if bool(af[n][k])]
            withheld.append(bool(af["low_plddt"][k] or af["terminal"][k] or af["exposed_hydrophobic_patch"][k]
                                 or af["metal_site"][k]))
        out.append(";".join(f))
    return out, withheld


# ============================================================================= mutation parsing
def parse_mutations(muts, default_chain: str | None = None) -> pd.DataFrame:
    """'A12G', 'A52AG' (insertion code A), 'B:L45P' (chain B) -> DataFrame chain, pdb_resnum, wt, mut."""
    if isinstance(muts, str):
        muts = [m for m in re.split(r"[,\s;]+", muts) if m]
    rows = []
    for m in muts:
        g = _MUT.match(m.strip())
        if not g:
            raise ValueError(f"cannot parse mutation {m!r} (expected e.g. A12G, A52AG, B:L45P)")
        ch, wt, num, ic, mut = g.groups()
        rows.append({"chain": ch or default_chain, "pdb_resnum": f"{num}{ic}", "wt": wt, "mut": mut, "mutation": m})
    return pd.DataFrame(rows)


def parse_sites(spec: str | None) -> set | None:
    """'10-80,95,100A' -> set of residue-number strings (ranges expand to plain numbers)."""
    if not spec:
        return None
    out = set()
    for part in str(spec).split(","):
        m = re.match(r"^(-?\d+)-(-?\d+)$", part.strip())
        if m:
            out.update(str(x) for x in range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.strip():
            out.add(part.strip())
    return out


# ============================================================================= predictor
def default_weights(stage: int, mode: str = "xtal") -> Path:
    """Shipped model for (stage, mode): final_*.json (D22: joint Megascale+FireProt fits B6gjs1 / B6gj / B6gjaf; their
    `nuisance` blocks are never applied, fit.predict ignores them) if present, else the B6g family. The Megascale-only
    models are ref_*.json (`PhysDDG(weights='ref')`, CLI `--weights ref`)."""
    if mode == "af":
        cands = ["final_af.json", "B6gaf.json", "B6af.json"] if stage == 2 else ["final_af_s1.json", "B6gaf_s1.json",
                                                                                 "B6af_s1.json"]
    else:
        cands = ["final_stage2.json", "B6g.json", "B6.json", "B5.json"] if stage == 2 else \
            ["final_stage1.json", "B6gs1.json", "B6s1.json", "B4.json"]
    for c in cands:
        if (MODELS / c).exists():
            return MODELS / c
    raise FileNotFoundError(f"no model for stage {stage}, mode {mode} in {MODELS}")


AF_TRIM = dict(below=50.0, window=10, termini_below=70.0, packed_exempt=True)   # D18 (MAVISp-style)


def crop_mask(prot, spec: str) -> np.ndarray:
    """'A:10-120,B:5-60' or '10-120' (all chains) -> residues to keep (author numbering, integer part)."""
    keep = np.zeros(len(prot), bool)
    num = np.array([int(re.match(r"-?\d+", r).group()) for r in prot.resnum])
    for part in spec.split(","):
        part = part.strip()
        ch, rng = (part.split(":", 1) if ":" in part else (None, part))
        m = re.match(r"^(-?\d+)-(-?\d+)$", rng)
        if not m:
            raise ValueError(f"bad crop range {part!r}")
        sel = (num >= int(m.group(1))) & (num <= int(m.group(2)))
        if ch is not None:
            sel &= prot.chain == ch
        keep |= sel
    return keep


def _as_pdb(structure: str, tmp: Path) -> Path:
    """PDB-format copy for text-level trimming (mmCIF / gz inputs are converted with gemmi)."""
    s = str(structure)
    if s.endswith(".pdb") or s.endswith(".ent"):
        return Path(s)
    import gemmi
    st = gemmi.read_structure(s)
    st.setup_entities()
    out = tmp / (Path(s).name.split(".")[0] + ".pdb")
    st.write_pdb(str(out))
    return out


class PhysDDG:
    """Structure-based physics-inspired ddG predictor (see module docstring).

    mode='xtal': fixed input structure, no pLDDT treatment (attenuation is never applied).
    mode='af' (D18): optional construct crop, MAVISp-style trim of low-pLDDT windows / termini (packed-exempt),
    stage 2 (shell repack, AF side chains kept elsewhere), pLDDT attenuation of the AF model weights, AF-mode flags
    and the `withheld` indicator (pLDDT < 70 | terminal | exposed hydrophobic patch | metal site)."""

    def __init__(self, weights: str | Path | None = None, stage: int | None = None, device: str = "cuda",
                 mode: str = "xtal", cfg=None, trim: bool | None = None):
        import sys
        import torch
        if "physddg.fit" not in sys.modules:       # older fit.py set torch's default dtype at import: undo here
            old = torch.get_default_dtype()
            from . import fit  # noqa: F401
            torch.set_default_dtype(old)
        from . import fit as FT
        self._FT = FT
        if mode not in ("xtal", "af"):
            raise ValueError("mode must be 'xtal' or 'af'")
        if weights is None or str(weights) in ("final", "ref"):   # 'ref' = Megascale-only B6g family (ref_*.json)
            kind = "final" if weights is None else str(weights)
            weights = default_weights(stage or (2 if mode == "af" else 1), mode)
            if kind == "ref":
                weights = MODELS / weights.name.replace("final_", "ref_")
        self.model = FT.load_model(weights)
        self.weights_path = str(weights)
        self.stage = int(self.model.get("stage", stage or 1))
        if stage is not None and stage != self.stage:
            raise ValueError(f"weights {weights} were fitted on stage-{self.stage} features, not stage {stage}")
        self.opts = self.model["derive_opts"]
        self.caps = self.model.get("provenance", {}).get("caps")
        self.scale_kind = self.model.get("scale_kind")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA not available: running on CPU (slow)")
            device = "cpu"
        self.device, self.mode = device, mode
        self.trim = (mode == "af") if trim is None else trim
        from .mutate import CAP_VARIANTS, MutateConfig
        self.cfg = cfg or MutateConfig(**CAP_VARIANTS.get(self.opts.get("cap", "") or "", {}))
        self.timing: dict = {}

    # ------------------------------------------------------------------ building blocks
    def load(self, structure: str | Path, chain: str | None = None):
        from .structure import annotate, load_protein
        p = load_protein(structure, chains=None, keep_het=True)
        return annotate(p)

    def _rows(self, prot, structure, mutations, chain, scan_all, sites) -> pd.DataFrame:
        if scan_all:
            want = parse_sites(sites)
            sel = [(c, r, a) for c, r, a in zip(prot.chain, prot.resnum, prot.seq)
                   if (chain is None or c == chain) and (want is None or r in want)]
            rows = pd.DataFrame([{"chain": c, "pdb_resnum": r, "wt": a, "mut": m, "mutation": f"{a}{r}{m}"}
                                 for c, r, a in sel for m in T.AA1 if m != a])
        else:
            dch = chain or (prot.chain[0] if len(set(prot.chain.tolist())) == 1 else None)
            rows = parse_mutations(mutations, dch)
            if rows.chain.isna().any():
                raise ValueError("structure has several chains: give --chain or 'B:L45P' style mutations")
        rows["structure"] = str(structure)
        rows["protein_id"] = Path(str(structure)).stem
        return rows

    def features(self, prot, structure: str, rows: pd.DataFrame, site_tab=None,
                 struct_source: str | None = None) -> pd.DataFrame:
        """All model input columns for rows: engine (mutate.py), structure statistics (features_simple) and the
        B6 site / substitution terms (features_extra), merged exactly as scripts/52_ladder_physics.py does.
        site_tab: optional precomputed {'simple': features_simple.site_table, 'extra': features_extra.site_descriptors}."""
        from . import features_extra as X
        from . import features_simple as FS
        from .mutate import SiteScan
        t0 = time.time()
        key = {(c, r): i for i, (c, r) in enumerate(zip(prot.chain.tolist(), prot.resnum.tolist()))}
        idx = np.array(sorted({key[(c, r)] for c, r in zip(rows.chain, rows.pdb_resnum) if (c, r) in key}), np.int64)
        if "nterm_orig" in prot.ann:                 # trimmed / cropped model: no invented charged termini
            from . import afmode
            sc = afmode.site_scan(prot, self.cfg, self.device)
        else:
            sc = SiteScan(prot, self.cfg, self.device)
        z = site_arrays(prot, idx, sc, self.stage) if len(idx) else {}
        t1 = time.time()
        eng = engine_columns(z, rows, self.stage)
        site_tab = site_tab or {}
        sites = site_tab.get("simple")
        sites = sites if sites is not None else FS.site_table(str(structure))
        simple = FS.variant_features(rows[KEYS + ["structure"]], sites)
        simple.index = rows.index
        df = pd.concat([rows, eng, simple[[c for c in simple.columns if c not in rows.columns and c not in eng.columns]]],
                       axis=1)
        df = pd.concat([df, cxxc(prot, rows).rename("cxxc")], axis=1).copy()
        if struct_source is not None:
            df = df.assign(struct_source=struct_source)
        # B6 terms (features_extra), as 52.add_extra: x_-prefixed site descriptors + candidate_features
        ex = site_tab.get("extra")
        if ex is None:                               # reuse the loaded model unless it is a trimmed/cropped copy
            own = "rsa" in prot.ann and "trim_kept" not in prot.ann and "nterm_orig" not in prot.ann
            ex = X.site_descriptors(str(structure), prot=prot if own else None)
        ex = ex.drop(columns=["structure_abs"]).add_prefix("x_").rename(columns={"x_chain": "chain",
                                                                                "x_pdb_resnum": "pdb_resnum"})
        ex["pdb_resnum"] = ex["pdb_resnum"].astype(str)
        ex["chain"] = ex["chain"].astype(str)
        d = df.reset_index().merge(ex, on=["chain", "pdb_resnum"], how="left").set_index("index").copy()
        d.index.name = None
        miss = {c: np.nan for c in ("s1_dE_lj_rep", "s2_dE_lj_rep") if c not in d}
        d = d.assign(**miss)                         # candidate_features reads s2_dE_lj_rep (unused by shipped terms)
        cand = X.candidate_features(d)
        df = pd.concat([d, cand[[c for c in cand.columns if c not in d.columns]]], axis=1)
        self.timing.update(engine_s=t1 - t0, simple_s=time.time() - t1, n_sites=len(idx))
        self._idx, self._key = idx, key
        return df

    def _atten(self, d: pd.DataFrame) -> np.ndarray:
        if self.scale_kind == "plddt":
            return d["flag_lowplddt"].to_numpy(float)
        if self.scale_kind == "burial":
            return (1.0 - d["x_rsa"].clip(0, 1)).to_numpy(float)
        raise ValueError(f"unknown scale_kind {self.scale_kind!r}")

    def score(self, df: pd.DataFrame, terms: bool = False) -> pd.DataFrame:
        """Apply the linear model (two-stage when the model has `scale_kind`) to feature columns from `features`."""
        FT = self._FT
        d = derive(df, self.stage, self.opts, self.caps)
        if self.scale_kind:
            col = f"scale_{self.scale_kind}"
            d[col] = FT.predict(self.model["scale_base"], d) * self._atten(d)
        pred = FT.predict(self.model, d)
        out = pd.DataFrame({"ddg": pred}, index=df.index)
        same = (df.wt == df.mut).to_numpy()
        out.loc[same, "ddg"] = 0.0                                   # WT->WT is exactly zero by definition
        if terms:
            for f, w in self.model["weights"].items():
                out[f"c_{f}"] = np.where(same, 0.0, w * d[f].to_numpy(float))
            r = self.model.get("ref") or {}
            out["c_ref"] = np.where(same, 0.0, df.mut.map(r).to_numpy(float) - df.wt.map(r).to_numpy(float)) if r else 0.0
            out["c_intercept"] = np.where(same, 0.0, float(self.model["intercept"]))
        return out

    # ------------------------------------------------------------------ AF-mode preparation (D18)
    def _af_models(self, structure: str, prot, crop: str | None, tmp: Path):
        """(full-or-cropped model, its file, trimmed model, its file, keep mask relative to the cropped model)."""
        from . import afmode
        base_file = _as_pdb(structure, tmp)
        p1, f1 = prot, Path(structure)
        if crop:
            keep = crop_mask(prot, crop)
            p1 = afmode.trim_protein(prot, keep, preserve_dihedrals=False)   # = a construct file of that range
            p1.ann.pop("nterm_orig", None)           # construct boundaries are real termini (as in the assay)
            f1 = tmp / "crop.pdb"
            afmode.write_trimmed_pdb(base_file, set(p1.keys), f1)
        if not self.trim:
            return p1, f1, None, None
        keep2 = afmode.trim_mask(p1, afmode.TrimConfig(**AF_TRIM))
        if keep2.all():
            return p1, f1, None, None
        p2 = afmode.trim_protein(p1, keep2)
        if "nterm_orig" in p1.ann:                   # chain starts relative to the original model
            p2.ann["nterm_orig"] = np.asarray(p1.ann["nterm_orig"])[keep2]
        f2 = tmp / "trim.pdb"
        afmode.write_trimmed_pdb(base_file, set(p2.keys), f2)
        return p1, f1, p2, f2

    # ------------------------------------------------------------------ public API
    def predict(self, structure: str | Path, mutations=None, chain: str | None = None, scan_all: bool = False,
                sites: str | None = None, terms: bool = False, prot=None, rows: pd.DataFrame | None = None,
                site_tab=None, crop: str | None = None, struct_source: str | None = None) -> pd.DataFrame:
        """ddG for mutations ('A12G', ...) or a saturation scan (scan_all, optional sites '10-80').
        crop (AF mode): keep only these residues ('A:10-120') before anything else (assayed construct / domain).
        struct_source: overrides the structure-type label used by the pLDDT attenuation (default: 'af' in AF mode,
        'xtal' otherwise; only labels containing 'af' enable it)."""
        import tempfile
        t0 = time.time()
        prot = prot if prot is not None else self.load(structure, chain)
        t1 = time.time()
        if rows is None:
            rows = self._rows(prot, structure, mutations, chain, scan_all, sites)
            if scan_all and crop:                    # saturation scan of the cropped construct only
                keep = {k for k, m in zip(prot.keys, crop_mask(prot, crop)) if m}
                rows = rows[[(c, r) in keep for c, r in zip(rows.chain, rows.pdb_resnum)]].reset_index(drop=True)
        else:
            rows = rows.assign(structure=str(structure), chain=rows.chain.astype(str),
                               pdb_resnum=rows.pdb_resnum.astype(str))
            if "protein_id" not in rows:
                rows["protein_id"] = Path(str(structure)).stem
            if "mutation" not in rows:
                rows["mutation"] = rows.wt + rows.pdb_resnum + rows.mut
        src = struct_source or ("af" if self.mode == "af" else "xtal")
        trimmed_site = pd.Series(False, index=rows.index)
        with tempfile.TemporaryDirectory() as tmpd:
            if self.mode == "af" and (crop or self.trim):
                p1, f1, p2, f2 = self._af_models(str(structure), prot, crop, Path(tmpd))
                rows_feat = rows.assign(structure=str(f1))
                if p2 is not None:
                    kept = set(p2.keys)
                    trimmed_site = pd.Series([(c, r) not in kept for c, r in zip(rows.chain, rows.pdb_resnum)],
                                             index=rows.index)
                    parts = []
                    a = rows_feat[~trimmed_site].assign(structure=str(f2))
                    if len(a):
                        parts.append(self._run(p2, str(f2), a, None, src, terms))
                    b = rows_feat[trimmed_site]
                    if len(b):                       # site itself trimmed: score it on the untrimmed model, flagged
                        parts.append(self._run(p1, str(f1), b, None, src, terms))
                    out = pd.concat(parts).loc[rows.index]
                else:
                    out = self._run(p1, str(f1), rows_feat, None, src, terms)
            else:
                out = self._run(prot, str(structure), rows, site_tab, src, terms)
        out = out.drop(columns=["structure"], errors="ignore")
        if self.mode == "af":
            out.loc[trimmed_site.to_numpy(), "flags"] = (out.loc[trimmed_site.to_numpy(), "flags"] + ";site_trimmed").str.strip(";")
        self.timing.update(load_s=t1 - t0, total_s=time.time() - t0, n_variants=len(out))
        return out

    def _run(self, prot, structure: str, rows: pd.DataFrame, site_tab, src: str, terms: bool) -> pd.DataFrame:
        df = self.features(prot, structure, rows, site_tab, src)
        sc = self.score(df, terms)
        out = rows[[c for c in rows.columns if c != "structure"]].copy()
        out["ddg"] = sc["ddg"]
        out["site_found"] = df["site_found"].astype(bool)
        out["wt_match"] = df["wt_match"].astype(bool)
        fl, wh = site_flags(prot, self._idx, self.mode) if len(self._idx) else ([], [])
        fl, wh = dict(zip(self._idx.tolist(), fl)), dict(zip(self._idx.tolist(), wh))
        ix = [self._key.get((c, r), -1) for c, r in zip(rows.chain, rows.pdb_resnum)]
        out["flags"] = [("" if i < 0 else fl.get(i, "")) + ("" if wm else (";wt_mismatch" if i >= 0 else ";site_not_found"))
                        for i, wm in zip(ix, out.wt_match)]
        out["flags"] = out["flags"].str.strip(";")
        if self.mode == "af":
            out["withheld"] = [bool(wh.get(i, True)) for i in ix]
        if terms:
            out = pd.concat([out, sc.drop(columns="ddg")], axis=1)
        return out


def model_info(weights: str | Path) -> dict:
    return json.loads(Path(weights).read_text())


# ============================================================================= multi-GPU / multi-structure runner
def _prep(job):
    """CPU part of a job: load + annotate + structure-statistics site table (runs in a process pool)."""
    import os
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from . import features_extra as X
    from . import features_simple as FS
    from .structure import annotate, load_protein
    structure, rows = job
    t = time.time()
    try:
        prot = annotate(load_protein(structure, keep_het=True))
        tab = {"simple": FS.site_table(str(structure)), "extra": X.site_descriptors(str(structure), prot=prot)}
    except Exception as e:  # noqa: BLE001
        return structure, rows, None, None, repr(e), time.time() - t
    return structure, rows, prot, tab, None, time.time() - t


def _gpu_worker(gpu: int, jobs: list, weights, stage, mode, cpus: int, terms: bool, q):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from concurrent.futures import ProcessPoolExecutor
    outs = []
    try:
        m = PhysDDG(weights, stage, "cuda", mode)
        with ProcessPoolExecutor(max(1, cpus)) as ex:                 # CPU prep overlapped with GPU work
            for structure, rows, prot, tab, err, t_cpu in ex.map(_prep, jobs, chunksize=1):
                try:
                    if prot is None:
                        raise RuntimeError(f"load_failed:{err}")
                    o = m.predict(structure, rows=rows, prot=prot, site_tab=None if m.mode == "af" else tab,
                                  terms=terms)
                    o["gpu"], o["cpu_prep_s"], o["gpu_s"] = gpu, t_cpu, m.timing["engine_s"]
                except Exception as e:  # noqa: BLE001  (never lose the other structures)
                    o = rows.drop(columns=[c for c in ("structure",) if c in rows]).assign(
                        ddg=np.nan, flags=f"error:{e!r}"[:300])
                o["structure"] = structure
                outs.append(o)
    finally:
        q.put(pd.concat(outs, ignore_index=True) if outs else pd.DataFrame())


def run_jobs(jobs: list, gpus=(0,), weights=None, stage: int | None = None, mode: str = "xtal", cpus: int = 4,
             terms: bool = False) -> pd.DataFrame:
    """jobs: list of (structure, rows DataFrame with chain/pdb_resnum/wt/mut[/mutation/protein_id/...]).
    One process per GPU (greedy balance by number of distinct sites), CPU preparation in a pool per GPU process."""
    import multiprocessing as mp
    import os
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # inherited by spawned workers:
        os.environ.setdefault(v, "1")                                           # avoid 64 BLAS threads per process
    if len(gpus) == 1 and cpus <= 0:
        m = PhysDDG(weights, stage, f"cuda:{gpus[0]}", mode)
        return pd.concat([m.predict(s, rows=r, terms=terms).assign(structure=s) for s, r in jobs], ignore_index=True)
    size = [r[["chain", "pdb_resnum"]].drop_duplicates().shape[0] for _, r in jobs]
    parts, load = [[] for _ in gpus], [0] * len(gpus)
    for j in np.argsort(size)[::-1]:
        k = int(np.argmin(load))
        parts[k].append(jobs[j])
        load[k] += size[j]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gpu_worker, args=(g, pj, weights, stage, mode, cpus, terms, q))
             for g, pj in zip(gpus, parts) if pj]
    for pr in procs:
        pr.start()
    import queue
    res = []
    while len(res) < len(procs):
        try:
            res.append(q.get(timeout=5))
        except queue.Empty:
            dead = [pr for pr in procs if not pr.is_alive() and pr.exitcode not in (0, None)]
            if dead and q.empty():
                for pr in procs:
                    pr.terminate()
                raise RuntimeError(f"GPU worker(s) died: exit codes {[pr.exitcode for pr in dead]}")
    for pr in procs:
        pr.join()
    return pd.concat(res, ignore_index=True)


def shard_sites(structure, rows: pd.DataFrame, n: int) -> list:
    """Split one structure's rows into n jobs by site (contiguous blocks of sites)."""
    sites = rows[["chain", "pdb_resnum"]].drop_duplicates().reset_index(drop=True)
    blocks = np.array_split(np.arange(len(sites)), n)
    jobs = []
    for b in blocks:
        if len(b) == 0:
            continue
        keep = sites.iloc[b]
        jobs.append((structure, rows.merge(keep, on=["chain", "pdb_resnum"])))
    return jobs
