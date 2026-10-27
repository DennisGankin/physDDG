#!/usr/bin/env python
"""Reproduce the manuscript figures from the committed result tables.

Inputs (read only):
  results/ladder_physics/multiassay_summary.csv   per-assay values + 95% cluster-bootstrap CIs (model card)
  results/ladder_physics/metrics.csv              per-subset metrics (Domainome pLDDT bands)
  results/ladder_physics/paired_af_agreement.csv  crystal vs AlphaFold agreement
  results/ladder_physics/final_selection_ablation.csv  drop-one-group ablation (val / CV / test)
  results/ladder_physics/models/final_stage2.json weights and cluster-bootstrap SEs
  results/ladder_physics/preds_megascale.parquet  per-variant predictions (per-protein Spearman)
  results/ladder_physics/preds_domainome.parquet  per-variant predictions (pLDDT bands)

Outputs: manuscript/figures/fig{1..5}_*.pdf and .png

Run:  .venv/bin/python manuscript/figures/make_figures.py
Colour roles (validated with the dataviz palette checker, light mode):
  PhysDDG models = blue #2a78d6, FoldX = orange #eb6834, joint multi-assay variant = aqua #1baf7a,
  every other baseline = grey. Identity is never carried by colour alone (labels / marker shapes).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "ladder_physics"
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LGREY, INK, INK2 = "#8a8984", "#c9c8c3", "#0b0b0b", "#52514e"
BLUE_L = "#cfe0f5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
W2 = 7.0  # double-column width (inch), Bioinformatics ~178 mm

LABEL = {
    "final_stage2": "PhysDDG default (Stage 2)", "final_stage1": "PhysDDG fast (Stage 1)",
    "final_af": "PhysDDG AF mode", "alt_multiassay_stage2": "joint multi-assay variant",
    "B3+": "simple baseline B3+", "FoldX-clip": "FoldX 5.1 (clipped)", "FoldX": "FoldX 5.1",
    "tmol-refit": "tmol, refitted", "KORPM": "KORPM", "DDGun3D": "DDGun3D",
}
COLOR = {"final_stage2": BLUE, "final_stage1": BLUE, "final_af": BLUE, "alt_multiassay_stage2": AQUA,
         "FoldX-clip": ORANGE, "FoldX": ORANGE}
MARK = {"final_stage2": "o", "final_stage1": "s", "final_af": "D", "alt_multiassay_stage2": "^",
        "FoldX-clip": "o", "FoldX": "o"}


def col(m):
    return COLOR.get(m, GREY)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------- Fig 1
def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=color, shrinkA=0, shrinkB=0))


def rotamers(ax, cx, cy, color, n=5, r=0.035, seed=0):
    """A site drawn as a fan of rotamer sticks: the ensemble that is Boltzmann-averaged."""
    rng = np.random.default_rng(seed)
    ax.plot([cx - 0.03, cx + 0.03], [cy, cy], color=INK2, lw=1.6, solid_capstyle="round")  # backbone
    for k in range(n):
        a = np.deg2rad(40 + k * 100 / (n - 1)) + rng.normal(0, 0.05)
        alpha = 1.0 if k == n // 2 else 0.45
        ax.plot([cx, cx + r * np.cos(a)], [cy, cy + r * 1.6 * np.sin(a)], color=color, lw=1.1,
                alpha=alpha, solid_capstyle="round")
        ax.plot(cx + r * np.cos(a), cy + r * 1.6 * np.sin(a), "o", ms=2.4, color=color, alpha=alpha)


def fig1():
    fig = plt.figure(figsize=(W2, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.965, "a  Input and AF mode", fontsize=7.5, fontweight="bold", va="top")
    ax.text(0.27, 0.965, "b  Same protocol for wild type and mutant at site i", fontsize=7.5,
            fontweight="bold", va="top")
    ax.text(0.745, 0.965, "c  Linear read-out", fontsize=7.5, fontweight="bold", va="top")

    # (a) input
    box(ax, 0.01, 0.70, 0.22, 0.20, "Structure (crystal or AF model)\n+ mutation list\n"
        "fixed backbone; heavy atoms", fc="#f4f4f2")
    box(ax, 0.01, 0.28, 0.225, 0.36, "", fc="white", ls="--")
    ax.text(0.12, 0.615, "AF mode only", ha="center", va="center", fontsize=6.3, fontweight="bold")
    ax.text(0.02, 0.44,
        "1. crop to assayed construct\n2. trim pLDDT<50 windows and\n    pLDDT<70 termini (packed: kept)\n"
        "3. Stage-2 shell repack,\n    AF side chains kept elsewhere\n4. flags: pLDDT<70, terminal,\n"
        "    exposed hydrophobic patch,\n    metal site\n5. pLDDT attenuation (1 weight)", ha="left", va="center",
        fontsize=5.9, linespacing=1.25)
    box(ax, 0.01, 0.06, 0.22, 0.18, "PDB-derived tables (PISCES)\nrotamer library P(rot | φ,ψ)\n"
        "Ramachandran, environment,\nN-cap potentials", fc="#f4f4f2", fs=6)
    arrow(ax, 0.23, 0.80, 0.275, 0.80)
    arrow(ax, 0.23, 0.15, 0.275, 0.33)

    # (b) WT / mutant panels
    for j, (lab, colr, seed) in enumerate([("wild type a = wt", INK2, 1), ("mutant a = mut", BLUE, 2)]):
        x0 = 0.28 + j * 0.235
        ax.add_patch(FancyBboxPatch((x0, 0.40), 0.215, 0.50, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    fc="#fbfbfa", ec=LGREY, lw=0.7))
        ax.text(x0 + 0.1075, 0.875, lab, ha="center", va="center", fontsize=6.8, color=colr, fontweight="bold")
        # environment atoms
        rng = np.random.default_rng(7)
        pts = rng.uniform([x0 + 0.015, 0.505], [x0 + 0.2, 0.84], size=(26, 2))
        keep = np.hypot((pts[:, 0] - (x0 + 0.1075)) / 0.6, pts[:, 1] - 0.62) > 0.075
        ax.plot(pts[keep, 0], pts[keep, 1], "o", ms=2.2, color=LGREY)
        # flexible neighbours (Stage 2)
        for (dx, dy) in [(-0.06, 0.73), (0.065, 0.735), (-0.065, 0.52), (0.06, 0.515)]:
            rotamers(ax, x0 + 0.1075 + dx, dy, "#9a9a95", n=3, r=0.018, seed=seed + int(dy * 100))
        rotamers(ax, x0 + 0.1075, 0.60, colr, n=5, r=0.03, seed=seed)
        ax.text(x0 + 0.1075, 0.43, "site: library rotamers only (no native)\n"
                "neighbours ≤16, native + library rotamers", ha="center", va="bottom", fontsize=5.4,
                color=INK2, linespacing=1.2)
    ax.text(0.5, 0.385, "", fontsize=6)
    box(ax, 0.28, 0.20, 0.45, 0.165, "", fc=BLUE_L, ec=BLUE)
    ax.text(0.505, 0.345, "Stage 1: Boltzmann sum over site rotamers in the fixed environment",
            ha="center", va="center", fontsize=6.2)
    ax.text(0.505, 0.29, r"$F(a)=-kT\,\ln\sum_r \exp(-U_r/kT)$,    $\langle E_k\rangle$ = Boltzmann average of term $k$",
            ha="center", va="center", fontsize=6.4)
    ax.text(0.505, 0.228, "Stage 2: site + ≤16 flexible neighbours, damped mean field (30 sweeps)",
            ha="center", va="center", fontsize=6.2)
    box(ax, 0.28, 0.035, 0.45, 0.145,
        "Terms, kept separate (their unit-weight sum U sets the Boltzmann weights):\n"
        "LJ attraction / repulsion (CHARMM19), EEF1 apolar / polar solvation,\n"
        "H-bonds sc–bb, sc–sc, bb–bb (DSSP), Debye–Hückel electrostatics,\n"
        r"side-chain entropy $\Delta TS$, rotamer prior $-kT\ln p(\mathrm{rot}|\phi,\psi)$", fc="white", fs=5.9)
    arrow(ax, 0.505, 0.40, 0.505, 0.365)

    # (c) read-out
    box(ax, 0.745, 0.58, 0.245, 0.32,
        r"$\Delta E_k = \langle E_k\rangle_{\mathrm{mut}} - \langle E_k\rangle_{\mathrm{wt}}$" "\n\n"
        r"$\widehat{\Delta\Delta G} = b + \sum_k w_k\,x_k + r_{\mathrm{mut}} - r_{\mathrm{wt}}$" "\n\n"
        "x = 10 engine terms + 10 structure\nstatistics (burial, cavity, Rama,\nenvironment, N-cap, helix dipole, …)",
        fc=BLUE_L, ec=BLUE, fs=6.2)
    box(ax, 0.745, 0.30, 0.245, 0.24,
        "Fit on Megascale-train only\nHuber loss; censored rows as\nintervals (one-sided Huber)\n"
        "r: 20 reference energies, L2-shrunk\nsigns fixed by physics; 41 numbers",
        fc="white", fs=6.0)
    box(ax, 0.745, 0.06, 0.245, 0.20,
        "Output: ΔΔG in kcal/mol\n(positive = destabilising)\n+ per-term contributions\n+ confidence flags",
        fc="#f4f4f2", fs=6.0)
    arrow(ax, 0.73, 0.29, 0.745, 0.62)
    arrow(ax, 0.8675, 0.30, 0.8675, 0.26)
    save(fig, "fig1_schematic")


# ----------------------------------------------------------------------------------------- Fig 2
def per_protein_rho(df, models):
    out = {}
    for m in models:
        out[m] = df.groupby("protein_id").apply(lambda g: spearmanr(g["ddg"], g[m])[0], include_groups=False)
    return pd.DataFrame(out)


def fig2():
    s = pd.read_csv(RES / "multiassay_summary.csv")
    order = ["final_af", "final_stage2", "final_stage1", "B3+", "FoldX-clip", "tmol-refit", "DDGun3D", "KORPM"]
    v = s[(s.view == "MS natural") & s.model.isin(order)].set_index("model").loc[order]

    p = pd.read_parquet(RES / "preds_megascale.parquet")
    nat = p[p.domain_origin == "natural"]
    rho = per_protein_rho(nat, ["final_stage2", "FoldX-clip"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        r = v.loc[m]
        a.plot([r.lo, r.hi], [yi, yi], color=col(m), lw=1.6, solid_capstyle="round")
        a.plot(r.value, yi, MARK.get(m, "o"), ms=5, color=col(m), mec="white", mew=0.8)
        a.text(r.hi + 0.008, yi, f"{r.value:.3f}", va="center", fontsize=6.2, color=INK2)
    a.set_yticks(y)
    a.set_yticklabels([LABEL[m] for m in order])
    a.set_xlim(0.42, 0.78)
    a.set_xlabel("per-protein median Spearman ρ (95% cluster-bootstrap CI)")
    a.grid(axis="x")
    a.set_axisbelow(True)
    a.set_title("a  Megascale-test, 21 natural proteins, 22,159 variants", loc="left", fontweight="bold")

    lim = (0.2, 0.95)
    b.plot(lim, lim, color=LGREY, lw=0.8, zorder=0)
    win = rho["final_stage2"] > rho["FoldX-clip"]
    b.plot(rho.loc[win, "FoldX-clip"], rho.loc[win, "final_stage2"], "o", ms=4.2, color=BLUE, mec="white",
           mew=0.6, label=f"PhysDDG higher ({win.sum()})")
    b.plot(rho.loc[~win, "FoldX-clip"], rho.loc[~win, "final_stage2"], "s", ms=4.2, color=ORANGE, mec="white",
           mew=0.6, label=f"FoldX higher ({(~win).sum()})")
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_aspect("equal")
    b.set_xlabel("FoldX 5.1 per-protein ρ")
    b.set_ylabel("PhysDDG default per-protein ρ")
    b.legend(frameon=False, loc="lower right", handletextpad=0.2)
    b.set_title("b  Same proteins, same AF2 structures", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_megascale")
    return rho


# ----------------------------------------------------------------------------------------- Fig 3
def fig3():
    s = pd.read_csv(RES / "multiassay_summary.csv")
    views = [("FireProt", "FireProt-test\nPearson r"), ("S669", "S669-strict\nPearson r"),
             ("S461", "S461-strict\nPearson r"), ("Domainome", "Domainome\nper-domain ρ"),
             ("Domainome strict", "Domainome strict\nper-domain ρ")]
    order = ["final_stage2", "final_stage1", "alt_multiassay_stage2", "B3+", "FoldX", "tmol-refit", "DDGun3D", "KORPM"]
    fig, axes = plt.subplots(1, len(views), figsize=(W2, 2.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (view, title) in zip(axes, views):
        v = s[(s.view == view)].set_index("model")
        for yi, m in zip(y, order):
            if m not in v.index:
                ax.text(0.5, yi, "not run", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        fontsize=5.5, color=LGREY)
                continue
            r = v.loc[m]
            sub = (view == "FireProt" and r.n < 347) or (view.startswith("Domainome") and m == "DDGun3D")
            c = col(m)
            ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.4, solid_capstyle="round", alpha=0.5 if sub else 1)
            ax.plot(r.value, yi, MARK.get(m, "o"), ms=4.2, color=c if not sub else "white", mec=c, mew=0.9)
        nn = v.loc["final_stage2"]
        nstr = f"{int(nn.n) / 1000:.0f}k" if nn.n >= 10000 else f"{int(nn.n)}"
        ax.set_title(f"{title}\nn = {nstr} / {int(nn.n_prot)} prot.", fontsize=6.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[m] for m in order])
    fig.text(0.5, -0.02, "value with 95% cluster-bootstrap CI; open marker = method covers a subset of rows "
             "(FireProt: FoldX 341, tmol 344 rows; Domainome: DDGun3D 47 / 28 domains)", ha="center",
             fontsize=6, color=INK2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_assays")


# ----------------------------------------------------------------------------------------- Fig 4
def cluster_boot_median(vals, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.Series(vals).groupby(np.asarray(groups)).apply(list)
    keys = g.index.to_numpy()
    meds = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        meds.append(np.median(np.concatenate([g.iloc[i] for i in pick])))
    return np.percentile(meds, [2.5, 97.5])


def fig4():
    d = pd.read_parquet(RES / "preds_domainome.parquet",
                        columns=["protein_id", "cluster", "score", "plddt", "final_af", "final_stage2", "FoldX", "B3+"])
    d = d.dropna(subset=["final_af", "final_stage2", "FoldX", "B3+", "plddt"])
    bands = [("≥90", d.plddt >= 90), ("70–90", (d.plddt >= 70) & (d.plddt < 90)),
             ("50–70", (d.plddt >= 50) & (d.plddt < 70)), ("<50", d.plddt < 50)]
    models = ["final_af", "final_stage2", "FoldX", "B3+"]
    rows = []
    for name, mask in bands:
        sub = d[mask]
        gb = sub.groupby("protein_id")
        ok = gb.size()
        ok = ok[ok >= 10].index
        sub = sub[sub.protein_id.isin(ok)]
        clus = sub.groupby("protein_id")["cluster"].first()
        for m in models:
            r = sub.groupby("protein_id").apply(lambda g: spearmanr(-g["score"], g[m])[0], include_groups=False)
            r = r.dropna()
            lo, hi = cluster_boot_median(r.values, clus.loc[r.index].values, n=500)
            rows.append(dict(band=name, model=m, med=np.median(r), lo=lo, hi=hi, n_dom=len(r), n_var=len(sub)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fig4_domainome_bands.csv", index=False)

    a_ag = pd.read_csv(RES / "paired_af_agreement.csv")
    a_ag = a_ag[a_ag.dataset == "all"].set_index("quantity")
    qs = [("s1_dF", "unfitted ΔF, Stage 1"), ("s2_dF", "unfitted ΔF, Stage 2"),
          ("final_stage1", "PhysDDG fast"), ("final_stage2", "PhysDDG default")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    off = {"final_af": -0.24, "final_stage2": -0.08, "FoldX": 0.08, "B3+": 0.24}
    lab = {"final_af": "PhysDDG AF mode", "final_stage2": "PhysDDG default", "FoldX": "FoldX 5.1", "B3+": "B3+"}
    clr = {"final_af": BLUE, "final_stage2": "#7fa9e0", "FoldX": ORANGE, "B3+": GREY}
    mk = {"final_af": "D", "final_stage2": "o", "FoldX": "s", "B3+": "v"}
    for m in models:
        tt = t[t.model == m]
        x = np.arange(len(bands)) + off[m]
        a.vlines(x, tt.lo, tt.hi, color=clr[m], lw=1.3)
        a.plot(x, tt.med, mk[m], ms=4, color=clr[m], mec="white", mew=0.6, label=lab[m], ls="none")
    a.axhline(0, color=INK2, lw=0.6)
    nvar = t.groupby("band", sort=False).n_var.first()
    a.set_xticks(range(len(bands)))
    a.set_xticklabels([f"{n}\n{nvar[n]:,} var." for n, _ in bands])
    a.set_xlabel("site pLDDT band")
    a.set_ylabel("per-domain median ρ vs −fitness")
    a.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.8, handletextpad=0.2)
    a.set_ylim(-0.2, 0.62)
    a.grid(axis="y")
    a.set_axisbelow(True)
    a.set_title("a  Domainome by site pLDDT (95% cluster-bootstrap CI)", loc="left", fontweight="bold")

    y = np.arange(len(qs))[::-1]
    for yi, (q, name) in zip(y, qs):
        r = a_ag.loc[q]
        b.plot([r.spearman_xtal_vs_ddg, r.spearman_af_vs_ddg], [yi, yi], color=LGREY, lw=1.4, zorder=0)
        b.plot(r.spearman_xtal_vs_ddg, yi, "s", ms=4.5, color=GREY, mec="white", mew=0.6,
               label="crystal input" if yi == y[0] else None)
        b.plot(r.spearman_af_vs_ddg, yi, "o", ms=4.5, color=BLUE, mec="white", mew=0.6,
               label="AlphaFold input" if yi == y[0] else None)
        b.text(0.65, yi, f"r(AF, xtal) = {r.pearson_af_vs_xtal:.2f}", va="center", fontsize=6, color=INK2)
    b.set_yticks(y)
    b.set_yticklabels([n for _, n in qs])
    b.set_xlim(0.35, 0.82)
    b.set_xlabel("pooled Spearman ρ vs measured ΔΔG")
    b.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.2)
    b.set_ylim(-0.5, len(qs) - 0.2)
    b.grid(axis="x")
    b.set_axisbelow(True)
    b.set_title("b  Paired crystal / AF set, 3,858 variants", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_afmode")
    return t


# ----------------------------------------------------------------------------------------- Fig 5
TERM_LABEL = {
    "e_lj_atr": "LJ attraction", "e_lj_rep": "LJ repulsion (signed log)", "e_solv_apolar": "EEF1 apolar",
    "e_solv_polar": "EEF1 polar", "e_hb_sc_bb": "H-bond sc–bb", "e_hb_sc_sc": "H-bond sc–sc",
    "e_hb_bb_bb": "H-bond bb–bb", "e_elec_dh": "Debye–Hückel", "dTS": "side-chain entropy ΔTS",
    "dprior_avg": "rotamer prior", "d_env_cn_ss_bb": "environment potential", "d_rama": "Ramachandran (i)",
    "d_rama_prev": "pre-Pro (i−1)", "bur": "burial 1−RSA", "occ_dvol_loss": "cavity (Occ×ΔV−)",
    "gly_posphi": "Gly→X at φ>0", "ncap_pref": "N-cap potential", "helix_dipole": "helix dipole",
    "pro_nhhb": "X→Pro, lost NH H-bond", "ss_broken": "disulfide broken",
}


def fig5():
    mdl = json.load(open(RES / "models" / "final_stage2.json"))
    w, se = mdl["weights"], mdl["se"]["weights"]
    terms = [t for t in TERM_LABEL if t in w]
    ab = pd.read_csv(RES / "final_selection_ablation.csv")
    drops = ab[ab.label.str.startswith("B6g − ")].copy()
    drops["name"] = drops.label.str.replace("B6g − ", "", regex=False)
    drops = drops.sort_values("d_cv_natural_prot")

    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(terms))[::-1]
    for yi, t in zip(y, terms):
        val, s = w[t], se.get(t, 0.0)
        c = BLUE if t.startswith("e_") or t in ("dTS", "dprior_avg") else GREY
        a.plot([val - 1.96 * s, val + 1.96 * s], [yi, yi], color=c, lw=1.4)
        a.plot(val, yi, "o", ms=3.6, color=c, mec="white", mew=0.5)
    a.axvline(0, color=INK2, lw=0.6)
    a.set_yticks(y)
    a.set_yticklabels([TERM_LABEL[t] for t in terms], fontsize=6)
    a.set_xlabel("weight (kcal/mol per unit) ± 1.96 SE")
    a.set_title("a  Fitted weights, default model", loc="left", fontweight="bold")
    a.text(0.98, 0.02, "blue: engine terms\ngrey: structure statistics", transform=a.transAxes,
           ha="right", va="bottom", fontsize=5.8, color=INK2)
    a.grid(axis="x")
    a.set_axisbelow(True)

    yy = np.arange(len(drops))[::-1]
    h = 0.27
    b.barh(yy + h, drops.d_val_natural_prot, height=h, color=BLUE, label="Megascale-val")
    b.barh(yy, drops.d_cv_natural_prot, height=h, color="#7fa9e0", label="grouped CV (train)")
    b.barh(yy - h, drops.d_test_natural_prot, height=h, color=LGREY, label="test (reported, not used)")
    b.axvline(0, color=INK2, lw=0.6)
    b.set_yticks(yy)
    b.set_yticklabels(drops.name, fontsize=6)
    b.set_xlabel("Δ per-protein ρ (natural) after dropping the group")
    b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=5.8,
             handlelength=1.2, columnspacing=0.8)
    b.set_title("b  Drop-one-group ablation", loc="left", fontweight="bold", pad=14)
    b.grid(axis="x")
    b.set_axisbelow(True)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig5_terms")


if __name__ == "__main__":
    fig1()
    rho = fig2()
    print("Fig2 wins:", int((rho["final_stage2"] > rho["FoldX-clip"]).sum()), "/", len(rho))
    fig3()
    t = fig4()
    print(t.round(3).to_string())
    fig5()
    print("written to", OUT)
