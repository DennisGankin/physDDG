"""Figures for the PhysDDG explainer. Writes static SVGs (fixed light colours, for GitHub) and
theme-aware SVG fragments (CSS variables, for the HTML page).

Run: python3 make_figures.py   (no dependencies beyond the standard library)
Data: per_protein_rho.json (recomputed from results/ladder_physics/preds_megascale.parquet),
../worked_example/trace.json (from ../worked_example/trace.py), numbers quoted from the model card.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
STATIC = dict(bg="#ffffff", ink="#1d2430", ink2="#4f5a6b", mute="#8a93a3", grid="#e3e7ee",
              phys="#2a78d6", foldx="#eb6834", other="#9aa3b2", lit="#c7cdd8", pos="#d9534f", neg="#1a9b70",
              meas="#1d2430", tint="#eef3fb")
THEMED = {k: f"var(--v-{k})" for k in STATIC}
FONT = "font-family=\"IBM Plex Sans, Helvetica, Arial, sans-serif\""


def svg(w, h, body, c, title):
    bg = f'<rect width="{w}" height="{h}" fill="{c["bg"]}"/>' if c is STATIC else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="{title}" {FONT}>{bg}{body}</svg>')


def t(x, y, s, c, size=12, anchor="start", weight=400, fill=None, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill or c["ink"]}" {extra}>{s}</text>')


# ---------------------------------------------------------------- ladder (Megascale-test natural)
LADDER = [  # label, per-protein median Spearman, kind, note
    ("Burial only (1 weight)", 0.528, "other", "B1"),
    ("Physics energies, unit weights", 0.557, "other", "stage-2 ΔF, nothing fitted"),
    ("FoldX 5.1", 0.629, "foldx", "clipped; same AF2 structures"),
    ("10-parameter statistics model", 0.648, "other", "B3+, no side chains"),
    ("PhysDDG fast (stage 1)", 0.678, "phys", "final_stage1"),
    ("PhysDDG default (stage 2)", 0.692, "phys", "final_stage2"),
    ("ThermoMPNN (deep learning)", 0.77, "lit", "published, not re-run here"),
]


def ladder(c):
    W, H, L, R, top, bh, gap = 760, 345, 250, 70, 40, 26, 14
    x0, x1 = 0.45, 0.80
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    b = [t(10, 20, "Per-protein median Spearman ρ, 21 natural Megascale-test domains (22,159 variants)", c, 12, fill=c["ink2"])]
    for v in (0.5, 0.6, 0.7, 0.8):
        b.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top-6}" y2="{top+len(LADDER)*(bh+gap)}" stroke="{c["grid"]}"/>')
        b.append(t(sx(v), top + len(LADDER) * (bh + gap) + 14, f"{v:.1f}", c, 11, "middle", fill=c["mute"]))
    for i, (lab, v, kind, note) in enumerate(LADDER):
        y = top + i * (bh + gap)
        col = c[kind]
        dash = ' stroke-dasharray="4 3"' if kind == "lit" else ""
        fill = c["bg"] if kind == "lit" else col
        b.append(f'<g class="hov"><title>{lab}: {v:.3f} ({note})</title>'
                 f'<rect x="{sx(x0):.1f}" y="{y}" width="{sx(v)-sx(x0):.1f}" height="{bh}" rx="3" fill="{fill}" '
                 f'stroke="{col}" stroke-width="{1.5 if kind=="lit" else 0}"{dash}/></g>')
        b.append(t(L - 10, y + bh / 2 + 4, lab, c, 12.5, "end", 600 if kind == "phys" else 400))
        b.append(t(sx(v) + 6, y + bh / 2 + 4, f"{v:.2f}" if kind != "lit" else "≈0.77", c, 12, fill=c["ink2"]))
    return svg(W, H, "".join(b), c, "Accuracy ladder on Megascale-test")


# ---------------------------------------------------------------- per-protein dumbbell
def dumbbell(c):
    rows = json.loads((HERE / "per_protein_rho.json").read_text())
    rows.sort(key=lambda r: r["final_stage2"] - r["FoldX-clip"])
    W, top, rh, L, R = 760, 46, 17, 70, 40
    H = top + len(rows) * rh + 40
    x0, x1 = 0.45, 0.85
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    b = [t(L, 16, "Each row is one held-out natural domain: FoldX (orange) → PhysDDG default (blue)", c, 12, fill=c["ink2"]),
         t(L, 32, "PhysDDG ranks better on 19 of 21 domains", c, 12, weight=600)]
    for v in (0.5, 0.6, 0.7, 0.8):
        b.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top-6}" y2="{top+len(rows)*rh}" stroke="{c["grid"]}"/>')
        b.append(t(sx(v), top + len(rows) * rh + 14, f"{v:.1f}", c, 11, "middle", fill=c["mute"]))
    b.append(t((L + W - R) / 2, H - 6, "per-protein Spearman ρ with measured ΔΔG", c, 11, "middle", fill=c["mute"]))
    for i, r in enumerate(rows):
        y = top + i * rh + rh / 2
        p, f = r["final_stage2"], r["FoldX-clip"]
        lc = c["mute"] if p < f else c["phys"]
        b.append(f'<g class="hov"><title>{r["protein_id"]}: PhysDDG {p:.3f}, FoldX {f:.3f}, '
                 f'10-param statistics {r["B3+"]:.3f}</title>'
                 f'<rect x="{L}" y="{y-rh/2:.1f}" width="{W-L-R}" height="{rh}" fill="{c["bg"]}" fill-opacity="0"/>'
                 f'<line x1="{sx(f):.1f}" x2="{sx(p):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{lc}" stroke-width="2"/>'
                 f'<circle cx="{sx(f):.1f}" cy="{y:.1f}" r="4.5" fill="{c["foldx"]}"/>'
                 f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="4.5" fill="{c["phys"]}"/></g>')
        b.append(t(L - 8, y + 4, r["protein_id"], c, 11, "end", fill=c["ink2"]))
    return svg(W, H, "".join(b), c, "Per-protein comparison with FoldX")


# ---------------------------------------------------------------- worked example contributions
GROUPS = [
    ("Physics energies", ["c_e_lj_atr", "c_e_lj_rep", "c_e_solv_apolar", "c_e_solv_polar", "c_e_hb_sc_bb",
                          "c_e_hb_sc_sc", "c_e_hb_bb_bb", "c_e_elec_dh", "c_dTS", "c_dprior_avg"]),
    ("Structure statistics", ["c_d_env_cn_ss_bb", "c_d_rama", "c_d_rama_prev", "c_gly_posphi", "c_ncap_pref",
                              "c_helix_dipole", "c_pro_nhhb", "c_ss_broken"]),
    ("Burial and cavity", ["c_bur", "c_occ_dvol_loss"]),
    ("Reference energies + offset", ["c_ref", "c_intercept"]),
]


def grouped(trace):
    meas = {m["mutation"]: m for m in trace["measured"]}
    out = []
    for r in trace["contrib"]:
        g = {name: sum(r[k] for k in ks) for name, ks in GROUPS}
        out.append(dict(mutation=r["mutation"], ddg=r["ddg"], groups=g, measured=meas[r["mutation"]]["ddg"],
                        foldx=meas[r["mutation"]]["FoldX-clip"]))
    return out


def contrib(c):
    tr = json.loads((HERE.parent / "worked_example" / "trace.json").read_text())
    rows = grouped(tr)
    W, panel_w, L, top = 760, 175, 10, 40
    H = 300
    lo, hi = -0.6, 4.0
    b = [t(10, 16, "Predicted ΔΔG split into contributions (kcal/mol, + = destabilising); ● measured, ◆ FoldX", c, 12, fill=c["ink2"])]
    names = [g[0] for g in GROUPS]
    for j, r in enumerate(rows):
        px = L + j * (panel_w + 15)
        sy = lambda v: top + 30 + (hi - v) / (hi - lo) * (H - top - 70)
        b.append(t(px + panel_w / 2, top + 10, r["mutation"], c, 14, "middle", 700))
        b.append(f'<line x1="{px}" x2="{px+panel_w}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="{c["mute"]}"/>')
        bw = 28
        for i, n in enumerate(names):
            v = r["groups"][n]
            x = px + 8 + i * (bw + 6)
            y0, y1 = sorted((sy(0), sy(v)))
            col = c["pos"] if v >= 0 else c["neg"]
            b.append(f'<g class="hov"><title>{r["mutation"]} · {n}: {v:+.2f}</title>'
                     f'<rect x="{x}" y="{y0:.1f}" width="{bw}" height="{max(y1-y0,1):.1f}" rx="2" fill="{col}"/></g>')
            b.append(t(x + bw / 2, (y0 - 4) if v >= 0 else (y1 + 12), f"{v:+.2f}", c, 9.5, "middle", fill=c["ink2"]))
        xs = px + 8 + 4 * (bw + 6) + 4
        for v, shape, lab in ((r["ddg"], "bar", "pred"), (r["measured"], "dot", "meas"), (min(r["foldx"], hi), "dia", "FoldX")):
            y = sy(v)
            if shape == "bar":
                b.append(f'<rect x="{xs-2}" y="{min(y,sy(0)):.1f}" width="10" height="{abs(sy(0)-y):.1f}" rx="2" fill="{c["phys"]}"><title>PhysDDG {v:.2f}</title></rect>')
            elif shape == "dot":
                b.append(f'<circle cx="{xs+3}" cy="{y:.1f}" r="5" fill="{c["meas"]}" stroke="{c["bg"]}" stroke-width="2"><title>measured {v:.2f}</title></circle>')
            else:
                b.append(f'<path d="M{xs+3},{y-6:.1f} l6,6 l-6,6 l-6,-6z" fill="{c["foldx"]}" stroke="{c["bg"]}" stroke-width="1.5"><title>FoldX (clipped) {r["foldx"]:.2f}</title></path>')
        b.append(t(px + panel_w / 2, H - 26, f'PhysDDG {r["ddg"]:.2f} · measured {r["measured"]:.2f}', c, 11, "middle", fill=c["ink"]))
        b.append(t(px + panel_w / 2, H - 12, f'FoldX {r["foldx"]:.2f}', c, 11, "middle", fill=c["ink2"]))
    # legend for the 4 groups
    lg = "".join(f'<tspan x="10" dy="0"></tspan>' for _ in [])
    b.append(t(10, H + 6, "Bars in each panel, left to right: " + " · ".join(names) + " · total", c, 11, fill=c["ink2"]))
    return svg(W, H + 14, "".join(b), c, "Worked example contributions")


# ---------------------------------------------------------------- pipeline overview (static, for Markdown)
STEPS = [
    ("1", "Describe each site", ["φ/ψ, helix/strand,", "burial, contacts", "(AF: crop, trim)"]),
    ("2", "Build side chains", ["library shapes for", "wild type AND all", "19 mutants alike"]),
    ("3", "Score + average", ["9 physics terms,", "Boltzmann average;", "stage 2: neighbours", "can also move"]),
    ("4", "Add PDB statistics", ["environment, Rama,", "cavity, N-cap,", "helix dipole"]),
    ("5", "Weighted sum", ["41 numbers fitted", "on 216k Megascale", "measurements"]),
    ("6", "Report + flag", ["ΔΔG, per-term", "parts, flags;", "AF: withhold", "pLDDT &lt; 70 etc."]),
]


def pipeline(c):
    W, H = 1040, 270
    bw, bh, y = 132, 140, 70
    gap = (W - 120 - 6 * bw - 60) / 5
    b = [f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{c["mute"]}"/></marker></defs>']
    b.append(f'<rect x="4" y="{y+20}" width="96" height="80" rx="6" fill="{c["bg"]}" stroke="{c["ink2"]}"/>')
    for i, s in enumerate(["Structure", "(PDB or AF)", "+ mutations"]):
        b.append(t(52, y + 48 + i * 17, s, c, 12, "middle", 600 if i == 0 else 400))
    x = 112
    b.append(f'<line x1="100" y1="{y+60}" x2="{x}" y2="{y+60}" stroke="{c["mute"]}" stroke-width="1.5" marker-end="url(#ar)"/>')
    for i, (n, title, lines) in enumerate(STEPS):
        fill = c["tint"] if n in "235" else c["bg"]
        b.append(f'<rect x="{x:.1f}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{c["phys"]}" stroke-width="1.3"/>')
        b.append(t(x + 10, y + 22, n, c, 15, weight=700, fill=c["phys"]))
        b.append(t(x + 10, y + 42, title, c, 12, weight=600))
        for j, l in enumerate(lines):
            b.append(t(x + 10, y + 66 + j * 16, l, c, 11, fill=c["ink2"]))
        nx = x + bw + gap
        if i < 5:
            b.append(f'<line x1="{x+bw:.1f}" y1="{y+60}" x2="{nx-2:.1f}" y2="{y+60}" stroke="{c["mute"]}" stroke-width="1.5" marker-end="url(#ar)"/>')
        x = nx
    ox = x - gap
    b.append(f'<line x1="{ox:.1f}" y1="{y+60}" x2="{ox+12:.1f}" y2="{y+60}" stroke="{c["mute"]}" stroke-width="1.5" marker-end="url(#ar)"/>')
    b.append(t(ox + 16, y + 56, "ΔΔG", c, 13, weight=700))
    b.append(t(ox + 16, y + 72, "kcal/mol", c, 10.5, fill=c["ink2"]))
    # brackets: per site batched on GPU (steps 2-3), per mutation (5-6)
    x2 = 112 + 1 * (bw + gap); x3 = 112 + 2 * (bw + gap) + bw
    b.append(f'<path d="M{x2:.1f},{y-10} v-8 H{x3:.1f} v8" fill="none" stroke="{c["mute"]}"/>')
    b.append(t((x2 + x3) / 2, y - 24, "per site, all 20 amino acids at once, batched on GPU", c, 11, "middle", fill=c["ink2"]))
    b.append(t(W / 2, y + bh + 40, "Shaded steps hold the parts that differ most from FoldX: symmetric wild-type modelling, "
               "ensemble averaging, data-fitted weights.", c, 11, "middle", fill=c["ink2"]))
    return svg(W, H, "".join(b), c, "PhysDDG pipeline overview")


# ---------------------------------------------------------------- journey timeline
TL = [  # hours since 10-26 12:00 UTC, id, label, kind (idea/neg/audit)
    (0.65, "TP2", "Outside critique\nrewrites design", "idea"),
    (0.95, "TP1", "Site-sensitivity\noracle", "idea"),
    (1.55, "TP3", "10-parameter model\nties FoldX", "idea"),
    (2.8, "TP4", "Forward/reverse\nbias fails", "neg"),
    (4.3, "TP5", "Repacking pays only\nfor small→large", "idea"),
    (5.8, "TP6", "AlphaFold: flag,\ndon't correct", "idea"),
    (7.2, "TP7", "Transfer beats\nhome-test gain", "idea"),
    (12.7, "TP8", "Audit 2 reverses\nshipped model", "audit"),
]


def timeline(c):
    W, H, L, R, y = 1040, 345, 40, 40, 190
    hmax = 22.0
    sx = lambda h: L + h / hmax * (W - L - R)
    col = {"idea": c["phys"], "neg": c["foldx"], "audit": c["pos"]}
    b = [f'<line x1="{L}" x2="{W-R}" y1="{y}" y2="{y}" stroke="{c["mute"]}" stroke-width="2"/>']
    for h in range(0, 23, 2):
        x = sx(h)
        b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y-4}" y2="{y+4}" stroke="{c["mute"]}"/>')
        lab_t = f"{(12+h)%24:02d}:00"
        if h == 0: lab_t = "10-26 12:00"
        if h == 12: lab_t = "10-27 00:00"
        b.append(t(x, y + 20, lab_t, c, 10.5, "middle", fill=c["mute"]))
    for i, (h, tid, lab, kind) in enumerate(TL):
        x = sx(h)
        up = i % 2 == 0
        lines = lab.split("\n")
        far = i in (2, 3)                      # staggered second lane for crowded neighbours
        d = 56 + (44 if far else 0)            # distance from axis to the near edge of the label block
        hgt = 16 + 13 * len(lines)
        if up:
            top = y - d - hgt
            b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y}" y2="{y-d:.1f}" stroke="{col[kind]}" stroke-width="1.3"/>')
        else:
            top = y + d
            b.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y}" y2="{top:.1f}" stroke="{col[kind]}" stroke-width="1.3"/>')
        b.append(f'<g class="hov"><title>{tid}: {" ".join(lines)}</title><circle cx="{x:.1f}" cy="{y}" r="7" fill="{col[kind]}" stroke="{c["bg"]}" stroke-width="2"/></g>')
        b.append(t(x, top + 12, tid, c, 12, "middle", 700, fill=col[kind]))
        for j, l in enumerate(lines):
            b.append(t(x, top + 26 + 13 * j, l, c, 11, "middle", fill=c["ink"]))
    # legend
    lx = L
    for k, lab in (("idea", "idea that shaped the method"), ("neg", "failure that taught a rule"), ("audit", "audit that reversed a decision")):
        b.append(f'<circle cx="{lx+5}" cy="20" r="5" fill="{col[k]}"/>')
        b.append(t(lx + 15, 24, lab, c, 11, fill=c["ink2"]))
        lx += 230
    b.append(t(W - R, 24, "dots are placed at the time of the decisive evidence", c, 11, "end", fill=c["mute"]))
    return svg(W, H, "".join(b), c, "Timeline of the eight turning points")


if __name__ == "__main__":
    out = {}
    for name, fn in (("ladder", ladder), ("per_protein", dumbbell), ("worked_example", contrib), ("pipeline", pipeline), ("timeline", timeline)):
        (HERE / f"{name}.svg").write_text(fn(STATIC))
        out[name] = fn(THEMED)
    (HERE / "themed_fragments.json").write_text(json.dumps(out))
    print("wrote", list(out))
