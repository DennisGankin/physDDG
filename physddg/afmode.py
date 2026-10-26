"""AlphaFold mode: confidence-aware treatments of a predicted input structure.

Everything here is a pure function of a `structure.Protein` (or of an already-computed feature / prediction table);
nothing is fitted inside this module. The treatments, with the labels used in notes/af_mode.md:

  (a) trim      delete residues whose (windowed) pLDDT is below a cutoff from the model before any energy is
                computed -- `trim_mask` + `trim_protein`. Deleting is the field convention (MAVISp: termini and
                linkers < 70; Janes/ProtVar: 10-residue windows < 50). `packed_exempt` implements the Williams
                2025 "near-predictive" exemption (a low-pLDDT segment that is genuinely packed is kept).
                Partial down-weighting of a band (50-70 at weight w) is obtained by linearly blending features
                computed with and without those residues -- `blend` (exact for the per-term pair energies of an
                additive potential, approximate for the Boltzmann free energy; see notes).
  (b) cap       pLDDT-modulated repulsion cap (D9 with flex(site) = confidence instead of burial):
                cap(i) = cap_lo + (cap_hi - cap_lo) * ramp(pLDDT_i; p_lo, p_hi), per residue, used as
                `Blocks.cap` (pair cap = max of the two atoms' caps) -- `cap_from_plddt`, `site_scan`.
  (c) repack    shell repack = the engine's stage 2 (site + 6 A neighbours, identical for WT and mutant).
                `repack_all` is the *ablation*: iterated greedy repacking of every side chain of the input
                (library rotamers only), which the literature says should hurt on AF backbones.
  (d) shrink    pLDDT-based shrinkage of the prediction towards a constant: ddG_AF = w*ddG + (1-w)*c,
                w = logistic((pLDDT - x0)/s) -- `shrink`, `fit_shrink`.
  (e) burial    backbone-only burial (Cbeta contact number / half-sphere exposure) substituted for SASA-based
                burial in a pLDDT band -- `cn_to_burial`, `burial_swap`.
  (f) flag      abstain / flag rules: low confidence, trimmed neighbourhood, exposed hydrophobic patch
                (missing partner), Cys/His metal-site cluster -- `flags`.

Sign convention: ddG positive = destabilising, as everywhere in this project.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import topology as T

HYDROPHOBIC = set("AVLIMFWYC")
KT = 0.593


# ============================================================================= pLDDT helpers
def plddt(prot) -> np.ndarray:
    """Per-residue pLDDT (CA B-factor of an AlphaFold model); NaN-free, clipped to [0, 100]."""
    b = np.asarray(prot.b14[:, 1], float)
    return np.clip(np.nan_to_num(b, nan=100.0), 0.0, 100.0)


def window_mean(x: np.ndarray, w: int, chain: np.ndarray | None = None) -> np.ndarray:
    """Centred running mean of `x` over `w` residues, not crossing chain boundaries (edges use what exists)."""
    x = np.asarray(x, float)
    if w <= 1:
        return x.copy()
    out = np.empty_like(x)
    ch = np.asarray(chain) if chain is not None else np.zeros(len(x), int)
    for c in np.unique(ch):
        idx = np.flatnonzero(ch == c)
        v = x[idx]
        k = min(w, len(v))
        pad = np.r_[np.full(k, np.nan), v, np.full(k, np.nan)]
        lo, hi = k, k + len(v)
        acc = np.zeros(len(v))
        cnt = np.zeros(len(v))
        for d in range(-(k // 2), k - k // 2):
            seg = pad[lo + d:hi + d]
            m = np.isfinite(seg)
            acc[m] += seg[m]
            cnt[m] += 1
        out[idx] = acc / np.maximum(cnt, 1)
    return out


def contact_density(prot, cutoff: float = 5.0, seq_window: int = 2) -> np.ndarray:
    """Heavy-atom contacts per heavy atom with residues further than `seq_window` in sequence (Williams 2025
    packing criterion: 'barbed wire' and 'pseudostructure' have few such contacts, 'near-predictive' is packed)."""
    from scipy.spatial import cKDTree
    a = prot.atoms()
    tree = cKDTree(a["xyz"])
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    ri, rj = a["res_index"][pairs[:, 0]], a["res_index"][pairs[:, 1]]
    same_chain = prot.chain[ri] == prot.chain[rj]
    keep = ~(same_chain & (np.abs(ri - rj) <= seq_window))
    n = np.zeros(len(prot))
    np.add.at(n, ri[keep], 1.0)
    np.add.at(n, rj[keep], 1.0)
    natoms = np.bincount(a["res_index"], minlength=len(prot)).astype(float)
    return n / np.maximum(natoms, 1)


# ============================================================================= (a) trimming
@dataclass
class TrimConfig:
    below: float = 50.0          # delete residues whose windowed pLDDT is below this
    window: int = 10             # window length (1 = per-residue; 10 = Janes/ProtVar rule)
    termini_below: float | None = None   # additionally delete terminal segments below this (MAVISp rule: 70)
    packed_exempt: bool = True   # keep a low-pLDDT segment that is genuinely packed (Williams 2025)
    packed_contacts: float = 3.0  # contacts per heavy atom above which a residue counts as packed
    # (3.0 = 10th percentile of the pLDDT > 90 residues of the Domainome constructs; medians by band:
    #  <50 0.0, 50-70 2.3, 70-90 5.4, >90 7.7 contacts/atom)
    min_keep: int = 20           # never trim below this many residues


def trim_mask(prot, cfg: TrimConfig | None = None) -> np.ndarray:
    """[L] bool: residues to KEEP."""
    cfg = cfg or TrimConfig()
    p = plddt(prot)
    pw = window_mean(p, cfg.window, prot.chain)
    drop = pw < cfg.below
    if cfg.termini_below is not None:
        term = np.zeros(len(prot), bool)
        for c in np.unique(prot.chain):
            idx = np.flatnonzero(prot.chain == c)
            low = pw[idx] < cfg.termini_below
            i = 0
            while i < len(idx) and low[i]:
                term[idx[i]] = True
                i += 1
            j = len(idx) - 1
            while j >= 0 and low[j]:
                term[idx[j]] = True
                j -= 1
        drop |= term
    if cfg.packed_exempt and drop.any():
        drop &= contact_density(prot) < cfg.packed_contacts
    keep = ~drop
    if keep.sum() < cfg.min_keep:
        keep = np.ones(len(prot), bool)
    return keep


def trim_protein(prot, keep: np.ndarray, preserve_dihedrals: bool = True):
    """Protein with only residues `keep`. Backbone dihedrals (phi/psi/omega) of the kept residues are preserved
    from the full model, so deleting a neighbour removes its *contacts* without corrupting the site's backbone;
    everything else (DSSP, burial, SASA) is recomputed on the trimmed model, which is the point of trimming.
    `ann['nterm_orig']` marks residues that were real chain starts, so the engine does not invent charged termini."""
    from dataclasses import fields

    from .structure import annotate
    keep = np.asarray(keep, bool)
    ann_full = dict(prot.ann)
    kw = {}
    for f in fields(prot):
        v = getattr(prot, f.name)
        if f.name == "ann":
            continue
        if f.name == "seq":
            kw[f.name] = "".join(np.array(list(prot.seq))[keep])
        elif isinstance(v, np.ndarray) and v.shape[:1] == (len(prot),):
            kw[f.name] = v[keep]
        else:
            kw[f.name] = v
    out = type(prot)(**kw)
    annotate(out, do_sasa=True)
    if preserve_dihedrals:
        for k in ("phi", "psi", "omega", "cis"):
            if k in ann_full and k in out.ann:
                out.ann[k] = np.asarray(ann_full[k])[keep]
    start = np.ones(len(prot), bool)
    start[1:] = prot.chain[1:] != prot.chain[:-1]
    out.ann["nterm_orig"] = start[keep]
    out.ann["trim_kept"] = keep
    return out


def write_trimmed_pdb(src: str | "Path", keep_keys: set, dst: str | "Path") -> int:
    """Write `src` without the residues outside `keep_keys` (a set of (chain, resnum+icode) as in Protein.keys).
    Pure text filtering, so the kept atoms are bit-identical to the input; used to feed the trimmed model to
    code that reads structures from disk (physddg.features_simple). Returns the number of residues written."""
    from pathlib import Path as _P
    seen = set()
    lines = []
    for ln in _P(src).read_text().splitlines(True):
        if ln.startswith(("ATOM", "HETATM", "ANISOU", "TER")):
            key = (ln[21], ln[22:27].strip())
            if key not in keep_keys:
                continue
            seen.add(key)
        lines.append(ln)
    _P(dst).parent.mkdir(parents=True, exist_ok=True)
    _P(dst).write_text("".join(lines))
    return len(seen)


# ============================================================================= (b) pLDDT-modulated cap
def ramp(x, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(x, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


@dataclass
class CapConfig:
    cap_lo: float = 0.5          # repulsion cap (kcal/mol per atom pair) at pLDDT <= p_lo
    cap_hi: float = 3.0          # ... and at pLDDT >= p_hi (3.0 = the engine default, D14)
    p_lo: float = 50.0
    p_hi: float = 90.0
    burial_c1: float = 0.0       # optional extra term: + burial_c1 * (burial - 0.5)


def cap_from_plddt(prot, cfg: CapConfig | None = None, burial: np.ndarray | None = None) -> np.ndarray:
    """Per-residue repulsion cap. Low confidence -> soft cap: a clash against an unreliable neighbour is more
    likely a model error than a real overlap (D9 with flex(site) = 1 - confidence)."""
    cfg = cfg or CapConfig()
    c = cfg.cap_lo + (cfg.cap_hi - cfg.cap_lo) * ramp(plddt(prot), cfg.p_lo, cfg.p_hi)
    if cfg.burial_c1 and burial is not None:
        c = c + cfg.burial_c1 * (np.asarray(burial, float) - 0.5)
    return np.clip(c, 0.05, 50.0)


def site_scan(prot, mcfg=None, device: str = "cuda", cap: np.ndarray | None = None, fix_termini: bool = True):
    """`mutate.SiteScan` with an optional per-residue repulsion cap override (and without inventing charged
    N-termini at trim boundaries). The environment is rebuilt after the override, as in SiteScan.__init__."""
    import torch

    from . import energy as E
    from .mutate import MutateConfig, SiteScan
    sc = SiteScan(prot, mcfg or MutateConfig(), device)
    changed = False
    if cap is not None:
        sc.blocks.cap = torch.as_tensor(np.asarray(cap, np.float32), device=sc.dev)
        changed = True
    if fix_termini and "nterm_orig" in prot.ann:
        sc.blocks.nterm = torch.as_tensor(np.asarray(prot.ann["nterm_orig"], bool), device=sc.dev)
        changed = True
    if changed:
        with torch.no_grad():
            sc.env = E.make_env(sc.blocks, None, sc.cfg.params)
    return sc


# ============================================================================= (c) repacking
def repack_all(prot, mcfg=None, device: str = "cuda", passes: int = 2, keep_backbone: bool = True):
    """ABLATION: replace every side chain of the input by the lowest-energy library rotamer of its own residue
    type (iterated greedy over all positions, native environment, same energy as the engine). Literature says
    this should not help on an AF backbone; `repack_all` exists to test that on our own energy."""
    import torch

    from .builder import build_sidechains
    from .mutate import MutateConfig
    from .structure import annotate
    mcfg = mcfg or MutateConfig()
    p = prot
    for _ in range(passes):
        sc = site_scan(p, mcfg, device)
        sites = np.arange(len(p))
        res = sc.stage1(sites)
        chis = res["chi_best"][np.arange(len(p)), p.aa]                     # [L,4] best rotamer of the WT type
        bb = torch.as_tensor(np.nan_to_num(p.xyz14[:, :4]), device=sc.dev, dtype=torch.float32)
        cb = torch.as_tensor(np.nan_to_num(p.xyz14[:, 4]), device=sc.dev, dtype=torch.float32)
        use_cb = torch.as_tensor(p.mask14[:, 4] & (p.aa != T.AA_INDEX["G"]), device=sc.dev)
        xyz, _ = build_sidechains(bb, p.aa, torch.as_tensor(chis, device=sc.dev, dtype=torch.float32)[:, None],
                                  cb=torch.where(use_cb[:, None], cb, torch.full_like(cb, float("nan"))))
        new = xyz[:, 0].cpu().numpy()[:, :14]
        xyz14 = p.xyz14.copy()
        mask14 = p.mask14.copy()
        nat = np.array([len(T.TEMPLATES[a].atoms) for a in p.seq])
        for i in range(len(p)):
            k = nat[i]
            if k > 4 and not np.isnan(new[i, 4:k]).any():
                xyz14[i, 4:k] = new[i, 4:k]
                mask14[i, 4:k] = True
        if keep_backbone:
            xyz14[:, :4] = p.xyz14[:, :4]
        from dataclasses import fields
        kw = {f.name: getattr(p, f.name) for f in fields(p) if f.name != "ann"}
        kw["xyz14"], kw["mask14"] = xyz14, mask14
        q = type(p)(**kw)
        annotate(q)
        q.ann["nterm_orig"] = p.ann.get("nterm_orig", np.r_[True, p.chain[1:] != p.chain[:-1]])
        p = q
    return p


# ============================================================================= (a') feature blending
def blend(a, b, w: float):
    """w * a + (1 - w) * b, elementwise, for numpy arrays or DataFrames with identical layout (used to
    down-weight a pLDDT band: a = low-pLDDT residues present, b = absent)."""
    return w * a + (1.0 - w) * b


# ============================================================================= (d) shrinkage
def logistic_weight(pl, x0: float = 70.0, s: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(pl, float) - x0) / max(s, 1e-6)))


def shrink(pred, pl, x0: float = 70.0, s: float = 10.0, c: float = 0.0) -> np.ndarray:
    """ddG_AF = w(pLDDT) * ddG + (1 - w) * c."""
    w = logistic_weight(pl, x0, s)
    return w * np.asarray(pred, float) + (1.0 - w) * c


def fit_shrink(pred, pl, y, groups=None, x0_grid=(40, 50, 60, 70, 80), s_grid=(3, 5, 10, 20), c_grid=(0.0,),
               objective: str = "spearman_group") -> dict:
    """Fit (x0, s, c) by grid search maximising the median within-group Spearman of `shrink(pred)` against y.
    Shrinkage is monotone in pred within a group only if w is constant there, so this can change ranking only
    across sites of different confidence -- which is the point."""
    from scipy.stats import spearmanr
    pred, pl, y = map(lambda v: np.asarray(v, float), (pred, pl, y))
    ok = np.isfinite(pred) & np.isfinite(pl) & np.isfinite(y)
    pred, pl, y = pred[ok], pl[ok], y[ok]
    g = np.asarray(groups)[ok] if groups is not None else np.zeros(len(y), int)
    uniq, inv = np.unique(g, return_inverse=True)
    best = None
    for x0 in x0_grid:
        for s in s_grid:
            for c in c_grid:
                q = shrink(pred, pl, x0, s, c)
                if objective == "spearman_group":
                    vals = [spearmanr(q[inv == k], y[inv == k])[0] for k in range(len(uniq)) if (inv == k).sum() >= 10]
                    score = float(np.nanmedian(vals)) if vals else np.nan
                else:
                    score = float(spearmanr(q, y)[0])
                if best is None or (np.isfinite(score) and score > best["score"]):
                    best = dict(x0=float(x0), s=float(s), c=float(c), score=score)
    return best


# ============================================================================= (e) backbone-only burial
def cn_to_burial(cn, a: float = 8.0, b: float = 24.0) -> np.ndarray:
    """Cbeta contact number -> burial in [0,1] (same linear map as energy.burial_from_ca_cb's default)."""
    return np.clip((np.asarray(cn, float) - a) / (b - a), 0.0, 1.0)


def hse_burial(hse_up, hse_down=None, a: float = 5.0, b: float = 30.0) -> np.ndarray:
    """Half-sphere exposure (CA neighbours in the CA->CB half-sphere) -> burial in [0,1]."""
    return np.clip((np.asarray(hse_up, float) - a) / (b - a), 0.0, 1.0)


def burial_swap(bur, alt_bur, pl, lo: float = 70.0, hi: float = 90.0, hard: bool = False) -> np.ndarray:
    """Use the backbone-only burial `alt_bur` where confidence is low, the SASA-based `bur` where it is high;
    linear cross-fade between `lo` and `hi` (hard=True: a step at `hi`)."""
    w = (np.asarray(pl, float) >= hi).astype(float) if hard else ramp(pl, lo, hi)
    return w * np.asarray(bur, float) + (1.0 - w) * np.asarray(alt_bur, float)


# ============================================================================= (f) flags / abstain
@dataclass
class FlagConfig:
    plddt_low: float = 70.0
    plddt_very_low: float = 50.0
    patch_radius: float = 10.0
    patch_min_hydrophobic: int = 4     # exposed hydrophobic neighbours within patch_radius
    patch_rsa: float = 0.35            # a residue counts as exposed above this relative SASA
    metal_radius: float = 8.0
    metal_min_ligands: int = 3         # Cys/His side chains clustered within metal_radius


def flags(prot, sites: np.ndarray | None = None, cfg: FlagConfig | None = None) -> dict:
    """Per-site flags for 'do not trust this number' / 'context is missing'.

    low_plddt / very_low_plddt : site confidence bands.
    near_low_plddt             : a residue with pLDDT < 50 within 8 A of the site (environment unreliable).
    exposed_hydrophobic_patch  : the site is an exposed hydrophobe in a cluster of exposed hydrophobes --
                                 the signature of a missing partner (interface / oligomer / membrane).
    metal_site                 : >= `metal_min_ligands` Cys/His side chains clustered near the site.
    terminal                   : within 3 residues of a chain end (crop artefact).
    """
    from scipy.spatial import cKDTree
    cfg = cfg or FlagConfig()
    sites = np.arange(len(prot)) if sites is None else np.asarray(sites)
    pl = plddt(prot)
    ann = prot.ann
    rsa = np.asarray(ann.get("rsa", np.full(len(prot), np.nan)), float)
    cb = ann.get("cb", None)
    if cb is None:
        from .structure import cb_coords
        cb = cb_coords(prot)
    tree = cKDTree(np.nan_to_num(cb))
    hyd = np.array([a in HYDROPHOBIC for a in prot.seq])
    exposed = rsa > cfg.patch_rsa
    cyshis = np.array([a in "CH" for a in prot.seq])
    out = {}
    out["plddt"] = pl[sites]
    out["low_plddt"] = pl[sites] < cfg.plddt_low
    out["very_low_plddt"] = pl[sites] < cfg.plddt_very_low
    near = tree.query_ball_point(np.nan_to_num(cb[sites]), 8.0)
    out["near_low_plddt"] = np.array([bool((pl[np.array(n, int)] < cfg.plddt_very_low).any()) if n else False
                                      for n in near])
    patch = tree.query_ball_point(np.nan_to_num(cb[sites]), cfg.patch_radius)
    n_hyd = np.array([int((hyd[np.array(n, int)] & exposed[np.array(n, int)]).sum()) if n else 0 for n in patch])
    out["n_exposed_hydrophobic"] = n_hyd
    out["exposed_hydrophobic_patch"] = (n_hyd >= cfg.patch_min_hydrophobic) & exposed[sites] & hyd[sites]
    met = tree.query_ball_point(np.nan_to_num(cb[sites]), cfg.metal_radius)
    out["n_cys_his"] = np.array([int(cyshis[np.array(n, int)].sum()) if n else 0 for n in met])
    out["metal_site"] = out["n_cys_his"] >= cfg.metal_min_ligands
    pos = np.arange(len(prot))
    start = np.r_[True, prot.chain[1:] != prot.chain[:-1]]
    end = np.r_[prot.chain[:-1] != prot.chain[1:], True]
    d_start = pos - np.maximum.accumulate(np.where(start, pos, -10**6))
    d_end = np.minimum.accumulate(np.where(end, pos, 10**6)[::-1])[::-1] - pos
    out["terminal"] = (np.minimum(d_start, d_end)[sites] < 3)
    return out


# ============================================================================= configuration bundle
@dataclass
class AFMode:
    """One named AF-mode configuration (what `scripts/82_af_features.py` and `scripts/84_af_treatments.py` run)."""
    name: str = "off"
    trim: TrimConfig | None = None
    cap: CapConfig | None = None
    stage: int = 1                      # 1 = site only, 2 = site + 6 A shell repack (treatment c)
    repack_all: bool = False            # ablation
    band_weight: float | None = None    # down-weight of the 50-70 band (blend of two trimmed runs)
    shrink_params: dict | None = None   # {'x0':..,'s':..,'c':..}
    burial_band: tuple | None = None    # (lo, hi) for the backbone-only burial swap

    def with_(self, **kw) -> "AFMode":
        return replace(self, **kw)


PRESETS = {
    "off": AFMode("off"),
    "trim50": AFMode("trim50", trim=TrimConfig(below=50.0, window=10, packed_exempt=True)),
    "trim50_nopack": AFMode("trim50_nopack", trim=TrimConfig(below=50.0, window=10, packed_exempt=False)),
    "trim70": AFMode("trim70", trim=TrimConfig(below=70.0, window=1, termini_below=70.0, packed_exempt=True)),
    "mavisp": AFMode("mavisp", trim=TrimConfig(below=50.0, window=10, termini_below=70.0, packed_exempt=True)),
    "cap_plddt": AFMode("cap_plddt", cap=CapConfig(cap_lo=0.5, cap_hi=3.0, p_lo=50.0, p_hi=90.0)),
    "cap_soft": AFMode("cap_soft", cap=CapConfig(cap_lo=0.25, cap_hi=1.5, p_lo=50.0, p_hi=90.0)),
    "repack_all": AFMode("repack_all", repack_all=True),
}
