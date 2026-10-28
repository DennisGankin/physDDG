"""Builds explainer/physddg_explainer.html (single self-contained page) from figures/themed_fragments.json,
worked_example/trace.json and the content below. Run: python3 build_html.py"""
import json
from pathlib import Path

H = Path(__file__).parent
FR = json.loads((H / "figures" / "themed_fragments.json").read_text())
TR = json.loads((H / "worked_example" / "trace.json").read_text())

STEPS = [
    dict(n=1, title="Describe each site", short="angles, burial, secondary structure",
         what="Load the structure (heavy atoms, fixed backbone). For every residue compute backbone angles φ/ψ, helix/strand/loop, burial (relative solvent accessibility and neighbouring Cβ atoms) and position in a helix. In AlphaFold mode also crop to the assayed domain and trim long low-confidence segments.",
         why="Almost every later term depends on context: the same Val→Ala costs about 1.7 kcal/mol in the core and about 0 at the surface. Burial alone ranks mutations within a protein at ρ ≈ 0.53.",
         without="No burial, no structure-aware statistics, no way to spot unreliable AlphaFold regions.",
         code="physddg/structure.py (load_protein, annotate, dssp, sasa, burial); physddg/afmode.py; PhysDDG._af_models in physddg/predict.py",
         tps=[1, 6]),
    dict(n=2, title="Build every candidate side chain", short="wild type and mutant built alike",
         what="At the mutated position, take the typical side-chain shapes (rotamers) of all 20 amino acids at this position's backbone angles, from a library derived from 9,266 high-resolution PDB chains. Keep the likeliest shapes covering 97% of probability (at most 30) and build their atoms on the GPU. The wild-type side chain in the file is not reused.",
         why="Symmetry. If the wild type kept its real, perfectly packed side chain and the mutant got an approximate modelled one, every mutation would look destabilising from modelling error alone. Rebuilding both the same way cancels the error. A test checks that mutating a residue to itself gives exactly 0.",
         without="A destabilising bias, largest where the structure is best packed. The project measured +6 to +8 kcal/mol bias on forward/reverse pairs for unweighted energies.",
         code="SiteScan.site_candidates (physddg/mutate.py:154); physddg/rotlib.py; physddg/builder.py; scripts/40_build_rotlib.py",
         tps=[2, 4]),
    dict(n=3, title="Score the shapes and average them", short="9 physics terms, Boltzmann average",
         what="Score each shape against its surroundings with nine published force-field terms (van der Waals attraction and repulsion, apolar and polar desolvation, three kinds of hydrogen bond, screened electrostatics, disulfide) plus the shape's prior probability. Combine shapes with a Boltzmann average, F = −kT ln Σ exp(−U/kT), giving a free energy, the average of each term, and a side-chain entropy. Stage 1 keeps the surroundings frozen; stage 2 (default) also lets up to 16 neighbouring side chains within 6 Å adapt, by 30 rounds of mean-field iteration.",
         why="Averaging is robust where a single best shape is not: one wrong rotamer can create a 50 kcal/mol clash. Stage 2 lets the surroundings make room for a larger side chain, worth +0.06 ρ on buried small→large mutations on held-out data (+0.01 overall).",
         without="Unweighted energies alone rank at ρ 0.54–0.56, below FoldX. Without stage 2, buried small→large mutations are scored as rigid clashes.",
         code="SiteScan.stage1 / _reduce (physddg/mutate.py:223, :250); SiteScan.stage2 (:383); physddg/energy.py; physddg/params.py",
         tps=[2, 5]),
    dict(n=4, title="Add what known structures say", short="PDB statistics for position sensitivity",
         what="A few statistical terms estimated from PDB structures, never from ΔΔG data: an environment potential (how often an amino acid occurs at this burial, secondary structure and backbone region), Ramachandran terms, burial and cavity, helix N-cap preference, helix dipole, and a correction for glycines at positive φ.",
         why="Most of the predictable signal is how sensitive a position is, not which substitution it receives. Knowing only each position's average ΔΔG would give ρ ≈ 0.72. These terms are the cheapest estimate of that sensitivity; a 10-weight model built from them alone already ties FoldX (0.648 vs 0.629).",
         without="The physics energies must infer burial indirectly and the model loses about 0.05 ρ. No single group of terms dominates in the ablation.",
         code="stat_features (physddg/mutate.py:556); physddg/features_simple.py; physddg/features_extra.py; physddg/data/*.npz",
         tps=[1, 3, 7]),
    dict(n=5, title="Combine with a small fitted linear model", short="41 numbers, censoring-aware fit",
         what="ΔΔG = intercept + Σ w_k·x_k + r[mutant] − r[wild type]. The x_k are mutant-minus-wild-type differences of the terms above; the 20 r values are 'reference energies' standing for the unfolded state. The 41 numbers are fitted to 216k Megascale-train measurements with a robust Huber loss, censored measurements (11,363 that only say 'at least this destabilising') as intervals, weights per homology cluster, and physically expected signs.",
         why="Raw energy units are not ΔΔG units. The weights calibrate each term (repulsion is down-weighted and log-compressed because a fixed backbone exaggerates clashes). Reference energies are needed because ΔΔG compares folded with unfolded, which the structure does not show.",
         without="Unweighted sums sit below FoldX. Dropping censoring and cluster weights is a safeguard rather than a measured gain: the project found the point metrics insensitive to the loss.",
         code="physddg/fit.py (fit, cross_validate); derive and PhysDDG.score in physddg/predict.py; scripts/52_ladder_physics.py; scripts/57_export_final.py",
         tps=[2, 3, 8]),
    dict(n=6, title="Report, and say when not to trust it", short="contributions, flags, withheld",
         what="Per mutation: ΔΔG, the contribution of every term (they sum exactly to ΔΔG) and flags (wild-type mismatch, ligand or metal within 5 Å, chain break, terminal residue). For AlphaFold models a withheld flag marks pLDDT < 70, terminal residues, exposed hydrophobic patches and metal sites.",
         why="Below pLDDT 70 every method, FoldX included, is at ρ ≈ 0. Withholding those 14% of variants improves what remains by about +0.03 to +0.04 ρ on Domainome, more than any correction that was tried. This is selective prediction: it improves the trusted set, not the withheld numbers.",
         without="Users get confident-looking numbers for disordered or misfolded regions.",
         code="PhysDDG.predict / _run (physddg/predict.py:530, :589); site_flags (:177); afmode.flags (physddg/afmode.py:344)",
         tps=[6]),
]

TPS = [
    dict(n=1, time="10-26 12:57", title="Where does the accuracy live?", kind="idea",
         sit="Before any engine existed, what should a physics model be good at?",
         opt="Refit FoldX's 22 terms; build an engine and hope; or first find where the signal in the data is.",
         ev="A leave-one-out site-mean oracle (predict each mutation by the average of the other mutations at its site) reaches pooled ρ = 0.757 on Megascale-test. Refitting FoldX's 22 terms under protein-grouped cross-validation gave 0.541, worse than stock FoldX (0.569).",
         out="Every agent was asked to report site-level and within-site ranking separately. FoldX re-weighting was never attempted.",
         shaped=[1, 4], where="LITERATURE.md §2; scripts/lit_site_vs_substitution.py; commit a0003cd"),
    dict(n=2, time="10-26 12:40", title="An outside critique rewrites the design", kind="idea",
         sit="The PI's first design (about 13 terms, minimum-energy repacking, a soft-clipped link) was given to a fresh agent to critique. It answered in nine minutes.",
         opt="Accept or reject about ten proposals. The PI accepted nearly all and rejected one sequencing proposal (delay stage 2) because AlphaFold mode needed it.",
         ev="The critic argued: (1) the largest omission is an environment statistical potential; (2) reference energies are confounded with Megascale's own unfolded-state model; (3) censoring lives in ΔG space; (4) reusing the current rotamer silently breaks wild-type/mutant symmetry; (5) a separate entropy term is redundant with Boltzmann averaging.",
         out="Decisions D4–D10 and seven pre-registered gates, written before any benchmark number existed.",
         shaped=[2, 3, 4, 5], where="notes/critique_design_v0.md; DESIGN.md v1; commit caa78ca"),
    dict(n=3, time="10-26 13:33", title="10 parameters and no side chains match FoldX", kind="idea",
         sit="The fitting agent built a ladder of simple baselines before the engine was ready.",
         opt="Keep the planned gate (engine ≥ a 2-weight model + 0.03) or re-base it on the strongest simple model.",
         ev="'B3+' (environment potential, burial, Ramachandran, Pro/Gly indicators, 20 reference energies) scored 0.648 against FoldX 0.629 on Megascale-test, and 0.43 vs 0.41 on FireProt. The old gate was far too easy.",
         out="The engine now had to beat B3+ by 0.03 and FoldX with a confidence interval excluding zero. It beat B3+ (+0.044, CI [0.018, 0.071]) but on crystal benchmarks never clearly did, which became a stated limit. Two terms (cis-Pro, overpacking) were removed after a transfer check.",
         shaped=[4, 5], where="notes/baseline_ladder.md; decision D16; commits 2a6b530, d428000"),
    dict(n=4, time="10-26 14:48", title="Forward and reverse mutations disagree", kind="neg",
         sit="Unfitted energies scored A→B on one structure and B→A on another with correlation 0.04–0.13 and a +6 to +8 kcal/mol bias: each structure is adapted to the residue it contains.",
         opt="(a) build every term as a difference of one state function; (b) site-dependent repulsion caps; (c) score the reverse mutation in the modelled mutant environment; (d) drop wild-type-only terms.",
         ev="Fitting cut the bias from 8.0 to 1.4. Caps changed validation ρ by only 0.010. Symmetrising on one structure left r = 0.15. Dropping wild-type-only terms reached r = 0.59 at −0.044 ρ.",
         out="Accepted as a permanent weakness (r ≈ 0.24 vs 0.88 for KORPM). What survived is the rule that the wild type is rebuilt like the mutant, plus a test that WT→WT = 0.",
         shaped=[2], where="notes/engine_mutate.md 'v2 additions'; results/ladder_physics/antisymmetry*.csv; commits 054516a, 18dffdd, 794bb2c"),
    dict(n=5, time="10-26 16:20", title="Is neighbour repacking worth it?", kind="idea",
         sit="Stage 2 is the costliest component; the critic expected little gain.",
         opt="Drop it, or keep it for the class it was designed for.",
         ev="B4 (stage 1) 0.677 vs B5 (stage 2) 0.687. The +0.02 gate failed overall (+0.010) but passed on buried small→large mutations (+0.048; +0.058 out-of-fold, 5 of 5 folds). A backbone ensemble (9× cost) gave nothing.",
         out="Both stages ship: stage 1 as the fast default for large scans, stage 2 for accuracy and AlphaFold inputs. The backbone ensemble was dropped.",
         shaped=[3], where="notes/ladder_physics.md; commits 5942d30, fb91835"),
    dict(n=6, time="10-26 17:47", title="AlphaFold: flag, don't correct", kind="idea",
         sit="The brief asked for an AlphaFold-specific mode. The obvious design shrinks predictions where pLDDT is low.",
         opt="Shrink by pLDDT, repack all side chains, modulate the repulsion cap, or flag.",
         ev="On 3,858 paired variants the AlphaFold model was never worse than the crystal. Repacking everything hurt agreement (0.950 → 0.940). pLDDT shrinkage gained +0.022 but a burial-only control with the same two parameters gained +0.073, so it was a burial calibration. Withholding 14% of variants gained +0.038, more than all treatments together.",
         out="Trim, crop, keep AlphaFold side chains outside a 6 Å shell, and flag. A small pLDDT attenuation exists for AlphaFold inputs, but its weight's interval includes zero.",
         shaped=[1, 6], where="notes/af_mode.md; decision D18; commits 6b87e15, 50a67ee"),
    dict(n=7, time="10-26 19:12", title="Transfer beats home-test gain", kind="idea",
         sit="Residual analysis added terms; the best one improved the Megascale test but not other assays.",
         opt="Ship the best Megascale model, or the most transferable one.",
         ev="Of 51 candidate terms, N-cap preference and helix dipole passed pre-set rules (validation 0.733 → 0.749). 'site_scsc' passed the FoldX gate by the widest margin (+0.072) but cost FireProt −0.026 Pearson and sat at the edge of the transfer check. A hand-typed N-cap table contradicted its source and was replaced by one derived from 58,800 PDB N-caps.",
         out="The PI, citing the brief ('the physics must transfer'), shipped the model without the best term. It is indistinguishable from the previous rung (+0.005, CI [−0.004, +0.019]).",
         shaped=[4], where="notes/residual_analysis.md; decision D19; commits 9d5d501, fecda54, 7e8989e"),
    dict(n=8, time="10-27 00:07", title="A second audit reverses the shipped model", kind="audit",
         sit="After a critique, the PI shipped a joint Megascale + FireProt model that seemed to improve FireProt (0.482) and Domainome at no cost.",
         opt="Defend it, patch the numbers, or re-decide from scratch using only validation and cross-validation.",
         ev="The FireProt column covered 338 of 347 rows because a comparison method failed on exactly the two rows that discriminated the models. On all 347, the model's Pearson was 0.388, below simpler models; the whole advantage came from one weight fitted on about 8 rows. At least 12 evaluations had touched test data, at least 5 of which changed what ships. Several shipped files were stale hand-made aliases.",
         out="The 8-row term was dropped; the family was re-decided on validation (0.741 vs 0.730) and cross-validation (0.681 vs 0.665), so the Megascale-only family ships. 'Physics generalises across assays' was withdrawn.",
         shaped=[5], where="notes/audit_2.md; decision D23; commits f707673, d684399, 47a1b89"),
]

GLOSS = [
    ("ΔΔG", "Change in folding free energy caused by a mutation, kcal/mol. Here positive = destabilising. 1 kcal/mol is roughly one good hydrogen bond."),
    ("Rotamer", "One of the typical shapes of an amino-acid side chain (a few to about 30 per amino acid)."),
    ("Burial / RSA", "How hidden from water a residue is. Relative solvent accessibility is 0 when buried, 1 when exposed."),
    ("pLDDT", "AlphaFold's per-residue confidence, 0–100. Below 70 the local structure is often unreliable."),
    ("Megascale", "Tsuboyama et al. 2023: about 776,000 stability measurements on small domains. Its training split fits PhysDDG."),
    ("Per-protein ρ", "Rank (Spearman) correlation of predicted and measured ΔΔG within one protein, then the median over proteins."),
    ("FoldX", "The most widely used physics-based ΔΔG tool (Guerois 2002; Schymkowitz 2005)."),
    ("Held-out", "Proteins never used to fit weights or choose between models."),
]

INS_SCI = [
    ("Most predictable signal is which position, not which substitution.",
     "A held-out position-average oracle ranks mutations at per-protein ρ ≈ 0.72, above every physics method tested (FoldX 0.63, PhysDDG 0.69). In the worked example, burial and cavity carry 1.43 of V24A's 1.78 kcal/mol.", "TP1, TP3"),
    ("A 10-parameter statistical model without side chains already matches FoldX.",
     "0.648 vs 0.629 on Megascale, 0.43 vs 0.41 on FireProt. Atomistic energies add a real but modest +0.044 [0.018, 0.071].", "TP3"),
    ("Published force-field constants are already near-optimal for ΔΔG.",
     "Refitting 32 physical parameters end to end moved each by less than ×1.6 with no ranking gain; fits on two independent assays agreed (r = 0.80).", "Appendix"),
    ("AlphaFold models are at least as good as crystals at high confidence.",
     "On 3,858 paired variants the AlphaFold-based prediction was never worse and was better where the structures differ. The pLDDT effect is largely burial in disguise.", "TP6"),
    ("Cell abundance is not folding stability at the surface.",
     "Megascale amino-acid preferences transfer to Domainome in the core (ρ 0.84) but not at the surface (−0.09).", "TP3"),
]
INS_METH = [
    ("Model the wild type with exactly the procedure used for the mutant.",
     "Reusing the experimental wild-type side chain biased forward/reverse pairs by +8 kcal/mol before fitting.", "TP2, TP4"),
    ("Build trivial baselines first and move the bar to them.",
     "The 10-parameter model made the original gate meaningless; re-basing it changed what counted as progress.", "TP3"),
    ("For unreliable inputs, flag rather than correct, and test every correction against a matched control.",
     "Withholding 14% gained more than every correction; the best-looking correction matched a burial-only control.", "TP6"),
    ("Independent audits of an agent's own claims pay for themselves.",
     "They found a 338-vs-347-row comparison that hid the decisive rows, a rule built from 2 test rows, and at least 12 test-set looks. They reversed the shipped model.", "TP8"),
]
INS_ENG = [
    "Sign convention: positive ΔΔG = destabilising; Megascale ddG_ML is flipped at ingestion.",
    "Rotamer library derived by the project from PISCES chains (≤ 1.8 Å, 30% identity, 9,266 chains) because the Dunbrack 2010 library needs a licence application.",
    "Repulsion radii scaled ×0.9; repulsion is signed-log transformed before fitting; per-pair cap 3 kcal/mol.",
    "Stage 2: ≤ 16 neighbours within 6 Å, ≤ 12 neighbour rotamers plus native, 30 damped mean-field sweeps (damping 0.5).",
    "PyTorch is only a GPU array library; float64 in the fitter once leaked into the engine (fixed in commit 860d136).",
    "FoldX was run on Megascale's AlphaFold2 models after RepairPDB with default options (options were not recorded).",
    "On CPU the predictor is 50–100× slower (the worked example: 4 minutes for 4 sites).",
    "scripts/57_export_final.py --check detects stale shipped model files. Structure type (xtal|af|auto) must be explicit.",
]

DATA = dict(steps=STEPS, tps=TPS, gloss=GLOSS, sci=INS_SCI, meth=INS_METH, eng=INS_ENG, trace=TR, fig=FR)

CSS = r"""
:root{--bg:#f6f8fb;--panel:#ffffff;--ink:#1d2430;--ink2:#4a5566;--mute:#7a8597;--line:#dde3ec;--accent:#2a78d6;--accent-ink:#ffffff;--soft:#e8f0fb;
--warn:#d9534f;--ok:#1a9b70;--orange:#eb6834;
--v-bg:#ffffff;--v-ink:#1d2430;--v-ink2:#4f5a6b;--v-mute:#8a93a3;--v-grid:#e3e7ee;--v-phys:#2a78d6;--v-foldx:#eb6834;--v-other:#9aa3b2;--v-lit:#c7cdd8;--v-pos:#d9534f;--v-neg:#1a9b70;--v-meas:#1d2430;--v-tint:#eef3fb;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f131a;--panel:#161b24;--ink:#e6eaf0;--ink2:#b2bccb;--mute:#8793a7;--line:#2a3240;--accent:#5a9bf0;--accent-ink:#0f131a;--soft:#1b2535;--warn:#e8706c;--ok:#38b88d;--orange:#f08a5c;
--v-bg:#161b24;--v-ink:#e6eaf0;--v-ink2:#b2bccb;--v-mute:#7f8ba0;--v-grid:#2a3240;--v-phys:#5a9bf0;--v-foldx:#f08a5c;--v-other:#6d788b;--v-lit:#4a5466;--v-pos:#e8706c;--v-neg:#38b88d;--v-meas:#e6eaf0;--v-tint:#1b2535;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f131a;--panel:#161b24;--ink:#e6eaf0;--ink2:#b2bccb;--mute:#8793a7;--line:#2a3240;--accent:#5a9bf0;--accent-ink:#0f131a;--soft:#1b2535;--warn:#e8706c;--ok:#38b88d;--orange:#f08a5c;
--v-bg:#161b24;--v-ink:#e6eaf0;--v-ink2:#b2bccb;--v-mute:#7f8ba0;--v-grid:#2a3240;--v-phys:#5a9bf0;--v-foldx:#f08a5c;--v-other:#6d788b;--v-lit:#4a5466;--v-pos:#e8706c;--v-neg:#38b88d;--v-meas:#e6eaf0;--v-tint:#1b2535;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:16px/1.6 "IBM Plex Sans",system-ui,sans-serif;padding-block:0 4rem;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:980px;margin:0 auto}
h1,h2,h3{font-family:"Newsreader",Georgia,serif;font-weight:600;line-height:1.2;text-wrap:balance}
h1{font-size:2.1rem;margin:2.2rem 0 .4rem}h2{font-size:1.5rem;margin:2rem 0 .6rem}h3{font-size:1.15rem;margin:1.4rem 0 .4rem}
p{margin:.5rem 0}a{color:var(--accent)}
.lede{font-size:1.15rem;color:var(--ink);max-width:68ch}
.sub{color:var(--ink2);max-width:68ch}
header.top{padding-top:1rem;display:flex;gap:1rem;align-items:center;justify-content:space-between;flex-wrap:wrap}
.disc{font-size:.85rem;color:var(--ink2);background:var(--soft);border-radius:6px;padding:.6rem .8rem;margin:1rem 0}
nav.tabs{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);display:flex;gap:.25rem;border-bottom:1px solid var(--line);z-index:5;margin-top:1rem;overflow-x:auto}
nav.tabs button{font:inherit;font-weight:600;background:none;border:0;border-bottom:3px solid transparent;padding:.7rem 1rem;color:var(--ink2);cursor:pointer;white-space:nowrap}
nav.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent)}
button:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.theme{font:inherit;font-size:.85rem;background:none;border:1px solid var(--line);border-radius:6px;color:var(--ink2);padding:.3rem .6rem;cursor:pointer}
section.tab[hidden]{display:none}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:1rem 1.1rem;margin:1rem 0}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.8rem;margin:1rem 0}
.three>div{min-width:0}
.fig{background:var(--v-bg);border:1px solid var(--line);border-radius:8px;padding:.6rem;margin:1rem 0;overflow-x:auto}
.fig svg{min-width:640px;display:block}
.fig .hov:hover{opacity:.8}
.cap{font-size:.85rem;color:var(--ink2);margin-top:.3rem}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.4rem .6rem;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
td.n,th.n{text-align:right}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85em}
.pipe{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0}
.pipe button{flex:1 1 140px;min-width:0;font:inherit;text-align:left;background:var(--panel);border:1.5px solid var(--line);border-radius:8px;padding:.6rem .7rem;cursor:pointer;color:var(--ink)}
.pipe button .k{font-family:"Newsreader",serif;font-size:1.4rem;color:var(--accent);font-weight:600;display:block;line-height:1}
.pipe button b{display:block;font-size:.95rem;margin-top:.2rem}
.pipe button span.s{font-size:.8rem;color:var(--ink2)}
.pipe button[aria-pressed="true"]{border-color:var(--accent);background:var(--soft)}
.stepbox dl{display:grid;grid-template-columns:8.5rem 1fr;gap:.4rem 1rem;margin:.6rem 0}
.stepbox dt{font-weight:600;color:var(--ink2)}.stepbox dd{margin:0;min-width:0}
@media(max-width:560px){.stepbox dl{grid-template-columns:1fr}}
.chip{display:inline-block;font-size:.8rem;border:1px solid var(--accent);color:var(--accent);background:none;border-radius:99px;padding:.05rem .55rem;margin:0 .2rem .2rem 0;cursor:pointer;font-family:inherit}
.chip:hover{background:var(--soft)}
.pseudo{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem;overflow-x:auto;font-size:.82rem;line-height:1.5;white-space:pre}
.wex{display:flex;gap:.4rem;flex-wrap:wrap;margin:.6rem 0}
.wex button{font:inherit;font-size:.9rem;border:1.5px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:.35rem .8rem;cursor:pointer}
.wex button[aria-pressed="true"]{border-color:var(--accent);background:var(--soft);font-weight:600}
.tp{border:1px solid var(--line);border-radius:8px;background:var(--panel);margin:.6rem 0}
.tp summary{cursor:pointer;padding:.7rem 1rem;display:flex;gap:.8rem;align-items:baseline;flex-wrap:wrap;list-style:none}
.tp summary::-webkit-details-marker{display:none}
.tp summary::before{content:"▸";color:var(--mute);width:1rem}
.tp[open] summary::before{content:"▾"}
.tp .id{font-family:"Newsreader",serif;font-weight:700;font-size:1.1rem}
.tp .t{font-weight:600;flex:1 1 14rem}.tp .tm{font-size:.8rem;color:var(--mute)}
.tp .body{padding:0 1rem 1rem 2.6rem}
.tp .body dl{display:grid;grid-template-columns:7.5rem 1fr;gap:.35rem 1rem;margin:.3rem 0}
.tp .body dt{font-weight:600;color:var(--ink2)}.tp .body dd{margin:0;min-width:0}
@media(max-width:560px){.tp .body{padding-left:1rem}.tp .body dl{grid-template-columns:1fr}}
.kind-idea{color:var(--accent)}.kind-neg{color:var(--orange)}.kind-audit{color:var(--warn)}
.tp.flash{outline:2px solid var(--accent)}
.ins{margin:.4rem 0;padding:.6rem .9rem;background:var(--panel);border:1px solid var(--line);border-radius:8px}
.ins b{display:block}.ins .e{color:var(--ink2);font-size:.92rem}
.lim li{margin:.3rem 0}
footer{margin-top:3rem;font-size:.85rem;color:var(--mute)}
"""

JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function tab(name,push){
  $$('section.tab').forEach(s=>s.hidden=s.id!=='t-'+name);
  $$('nav.tabs button').forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===name));
  if(push!==false){try{history.replaceState(null,'','#'+name)}catch(e){}}
}
$$('nav.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
// theme
const root=document.documentElement;
$('#theme').onclick=()=>{const d=root.dataset.theme==='dark'||(!root.dataset.theme&&matchMedia('(prefers-color-scheme:dark)').matches);root.dataset.theme=d?'light':'dark';};
// figures
$$('[data-fig]').forEach(e=>e.innerHTML=D.fig[e.dataset.fig]);
// pipeline
const pipe=$('#pipe');
D.steps.forEach((s,i)=>{const b=document.createElement('button');b.innerHTML=`<span class="k">${s.n}</span><b>${esc(s.title)}</b><span class="s">${esc(s.short)}</span>`;b.onclick=()=>step(i,true);b.dataset.i=i;pipe.appendChild(b);});
function chips(arr,kind){return arr.map(n=>`<button class="chip" data-${kind}="${n}">${kind==='tp'?'Turning point '+n:'Step '+n}</button>`).join('')}
function step(i,scroll){
  const s=D.steps[i];$$('#pipe button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.i==i));
  $('#stepbox').innerHTML=`<h3>Step ${s.n}. ${esc(s.title)}</h3><dl><dt>What it does</dt><dd>${esc(s.what)}</dd><dt>Why it is needed</dt><dd>${esc(s.why)}</dd><dt>Without it</dt><dd>${esc(s.without)}</dd><dt>In the code</dt><dd class="mono">${esc(s.code)}</dd><dt>Produced by</dt><dd>${chips(s.tps,'tp')}</dd></dl>`;
  $$('#stepbox [data-tp]').forEach(c=>c.onclick=()=>gotoTP(+c.dataset.tp));
  if(scroll)$('#stepbox').scrollIntoView({block:'nearest',behavior:'smooth'});
}
// worked example
const T=D.trace,M=T.contrib.map(r=>r.mutation);
let wstep=1;
const WS=[['1','Context'],['2','Shapes (stage 1)'],['3','Neighbours (stage 2)'],['4','Statistics'],['5','Weighted sum']];
const wb=$('#wex');WS.forEach(([k,l])=>{const b=document.createElement('button');b.textContent=k+'. '+l;b.dataset.k=k;b.onclick=()=>wshow(+k);wb.appendChild(b);});
const f1=x=>(+x).toFixed(1),f2=x=>(+x).toFixed(2);
function tbl(h,rows,num){return `<div class="tablewrap"><table><thead><tr>${h.map((x,i)=>`<th class="${num&&i>=num?'n':''}">${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((x,i)=>`<td class="${num&&i>=num?'n':''}">${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
const GRP={phys:['c_e_lj_atr','c_e_lj_rep','c_e_solv_apolar','c_e_solv_polar','c_e_hb_sc_bb','c_e_hb_sc_sc','c_e_hb_bb_bb','c_e_elec_dh','c_dTS','c_dprior_avg'],stat:['c_d_env_cn_ss_bb','c_d_rama','c_d_rama_prev','c_gly_posphi','c_ncap_pref','c_helix_dipole','c_pro_nhhb','c_ss_broken'],bur:['c_bur','c_occ_dvol_loss'],ref:['c_ref','c_intercept']};
function wshow(k){
  wstep=k;$$('#wex button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k==k));let h='';
  if(k==1){h=`<p>1PSE is a 68-residue natural domain from the held-out Megascale test set, modelled by AlphaFold2. For each mutation the first step records where the site sits.</p>`+tbl(['mutation','structure','φ / ψ (°)','RSA','Cβ contacts','pLDDT'],T.context.map(c=>[c.mut,{E:'strand',L:'loop',H:'helix'}[c.ss]||c.ss,f1(c.phi)+' / '+f1(c.psi),f2(c.rsa),f1(c.cn),f1(c.plddt)]),3)+`<p class="cap">V24 and A61 are buried (RSA ≈ 0); K29 is exposed; G31 is a glycine at positive φ.</p>`}
  if(k==2){h=`<p>Free energies F in the engine's own units (kcal/mol, before any weighting), shapes kept after the 97% probability cut, and three of the nine terms. A61W: all 16 tryptophan shapes clash with the frozen surroundings.</p>`+tbl(['site','amino acid','shapes','F','attraction','repulsion','polar desolv.'],T.stage1.filter(r=>['V24A','G31A','A61W'].includes(r.mut)).map(r=>[r.mut.slice(1,-1),r.aa+(r.aa===r.mut[0]?' (wt)':''),r.n_rot,f1(r.F),f1(r.lj_atr),f1(r.lj_rep),f1(r.solv_polar)]),2)+`<p class="cap">V24 (wild-type valine): two shapes with Boltzmann weights ${T.v24_rotamers.map(r=>f2(r.weight)).join(' and ')} (energies ${T.v24_rotamers.map(r=>f1(r.U)).join(' and ')}).</p>`}
  if(k==3){const g=Object.fromEntries(T.stage2.map(r=>[r.mut+r.aa,r]));h=`<p>Stage 2 lets neighbouring side chains adapt (${T.neighbours.A61W.length} for A61W: ${T.neighbours.A61W.join(', ')}…). The same neighbour set is used for wild type and mutant.</p>`+tbl(['mutation','neighbours','repulsion wt','repulsion mut','difference','stage 1 difference'],M.map(m=>{const a=g[m+m[0]]||g[m+m[0]],b=g[m+m.slice(-1)];const s1=T.stage1.filter(r=>r.mut===m);return [m,a.n_nb,f1(a.lj_rep),f1(b.lj_rep),f1(b.lj_rep-a.lj_rep),f1(s1[1].lj_rep-s1[0].lj_rep)]}),1)+`<p class="cap">For A61W the repulsion difference falls from 51.6 to 21.2 once neighbours can move. After the fitted log-compression and weight it adds only +0.47 kcal/mol.</p>`}
  if(k==4){h=`<p>Terms estimated from PDB structures (mutant minus wild type).</p>`+tbl(['mutation','environment potential','Ramachandran','what it says'],T.stats.map(r=>[r.mut,f2(r.env_cn_ss_bb),f2(r.rama),{V24A:'buried strand valine; cavity term 0.48',K29E:'exposed loop, nothing special',G31A:'alanine rarely adopts φ = +92°, so statistics call it costly; glycine-at-positive-φ flag = 1',A61W:'buried loop, backbone angles fine'}[r.mut]]),1)}
  if(k==5){const meas=Object.fromEntries(T.measured.map(r=>[r.mutation,r]));
    const rows=T.contrib.map(r=>{const s=a=>a.reduce((x,k)=>x+r[k],0);const me=meas[r.mutation];return [r.mutation,f2(s(GRP.phys)),f2(s(GRP.stat)),f2(s(GRP.bur)),f2(s(GRP.ref)),'<b>'+f2(r.ddg)+'</b>','<b>'+f2(me.ddg)+'</b>',f2(me['FoldX-clip'])]});
    h=`<p>The 20 weighted contributions (they sum exactly to the prediction), grouped into four families.</p><div class="fig"><div data-fig="worked_example"></div></div>`+tbl(['mutation','physics energies','structure statistics','burial + cavity','reference + offset','PhysDDG','measured','FoldX'],rows,1)+`<ul class="lim"><li><b>V24A</b> works: a buried valine loses contacts and leaves a cavity; burial and cavity carry most of the answer. FoldX overshoots by 1.1.</li><li><b>K29E</b> works: an exposed charge swap is correctly near zero.</li><li><b>G31A</b> fails: statistics say alanine at positive φ costs about 1 kcal/mol, the measurement says it does not. This is a known failure class; FoldX is worse (3.9).</li><li><b>A61W</b> partly fails: buried small→large is the class where the fixed backbone exaggerates the clash (ρ 0.49 vs FoldX 0.55 on that class).</li></ul>`;
  }
  $('#wbody').innerHTML=h;$$('#wbody [data-fig]').forEach(e=>e.innerHTML=D.fig[e.dataset.fig]);
}
// turning points
const tl=$('#tps');
D.tps.forEach(t=>{const d=document.createElement('details');d.className='tp';d.id='tp'+t.n;
 d.innerHTML=`<summary><span class="id kind-${t.kind}">TP${t.n}</span><span class="t">${esc(t.title)}</span><span class="tm">${esc(t.time)} UTC</span></summary><div class="body"><dl><dt>Situation</dt><dd>${esc(t.sit)}</dd><dt>Options</dt><dd>${esc(t.opt)}</dd><dt>Evidence</dt><dd>${esc(t.ev)}</dd><dt>Outcome</dt><dd>${esc(t.out)}</dd><dt>Shaped</dt><dd>${chips(t.shaped,'step')}</dd><dt>Where</dt><dd class="mono">${esc(t.where)}</dd></dl></div>`;tl.appendChild(d);});
$$('#tps [data-step]').forEach(c=>c.onclick=()=>{tab('method');step(+c.dataset.step-1,true)});
function gotoTP(n){tab('journey');const d=$('#tp'+n);d.open=true;d.classList.add('flash');d.scrollIntoView({block:'center',behavior:'smooth'});setTimeout(()=>d.classList.remove('flash'),1800)}
$$('[data-tpjump]').forEach(c=>c.onclick=()=>gotoTP(+c.dataset.tpjump));
$$('#timeline text').forEach(t=>{const m=t.textContent.match(/^TP(\d)$/);if(m){t.style.cursor='pointer';t.onclick=()=>gotoTP(+m[1])}});
// insights, glossary
const insHTML=(a)=>a.map(([h,e,w])=>`<div class="ins"><b>${esc(h)}</b><span class="e">${esc(e)}</span> <span class="mono">${w.split(', ').map(x=>/^TP/.test(x)?`<a href="#" data-tpjump="${x.slice(2)}">${x}</a>`:esc(x)).join(', ')}</span></div>`).join('');
$('#sci').innerHTML=insHTML(D.sci);$('#meth').innerHTML=insHTML(D.meth);
$('#eng').innerHTML=D.eng.map(x=>`<li>${esc(x)}</li>`).join('');
$$('#t-insights [data-tpjump]').forEach(c=>c.onclick=e=>{e.preventDefault();gotoTP(+c.dataset.tpjump)});
$('#gloss').innerHTML=D.gloss.map(([a,b])=>`<tr><td><b>${esc(a)}</b></td><td>${esc(b)}</td></tr>`).join('');
step(0,false);wshow(1);
const h=(location.hash||'#method').slice(1);tab(['method','journey','insights'].includes(h)?h:'method',false);
"""

BODY = """
<div class="wrap">
<header class="top"><div class="mono" style="color:var(--mute)">PhysDDG explainer</div><button class="theme" id="theme" type="button">Light / dark</button></header>
<h1>PhysDDG: how a physics-based mutation-stability predictor was built</h1>
<p class="disc">This predictor was developed in an exercise to test whether AI agents can run a bioinformatics project: a PI agent managed 29 subagents on 26–27 October 2026. All numbers come from the project's files, rerun or recomputed where noted. Claims the project's own audits withdrew are not repeated here.</p>
<nav class="tabs" role="tablist"><button data-tab="method" role="tab">Method</button><button data-tab="journey" role="tab">How we got here</button><button data-tab="insights" role="tab">Insights</button></nav>

<section class="tab" id="t-method">
<p class="lede" style="margin-top:1.4rem"><b>PhysDDG takes a protein structure and a list of single amino-acid substitutions and predicts, for each, the change in folding stability (ΔΔG, kcal/mol; positive = destabilising), with every physical contribution shown, in about a millisecond per mutation on one GPU.</b></p>
<p class="sub">For each mutated position it builds the original and new side chain the same way in many possible shapes and scores them with classical force-field energies. It adds statistics from known structures (burial, which amino acids fit where, which backbone angles they tolerate). A weighted sum with 41 numbers fitted to 216,000 measurements gives the ΔΔG. On held-out proteins it ranks mutations better than FoldX, the standard physics-based tool (ρ 0.69 vs 0.63), but below deep-learning predictors (about 0.77).</p>
<div class="fig"><div data-fig="pipeline"></div><div class="cap">Click a step below, or follow the numbered steps. Shaded: the parts that differ most from FoldX.</div></div>

<h2>The six steps</h2>
<div class="pipe" id="pipe"></div>
<div class="panel stepbox" id="stepbox"></div>

<h2>How it differs from the standard approaches</h2>
<div class="tablewrap"><table><thead><tr><th></th><th>FoldX, Rosetta</th><th>Statistical (KORPM, DDGun)</th><th>Deep learning</th><th>PhysDDG</th></tr></thead><tbody>
<tr><td>Computes</td><td>one best mutant model, energy difference</td><td>lookup tables over environments</td><td>learned representation</td><td>physics averaged over many side-chain shapes, plus statistics</td></tr>
<tr><td>Fitted</td><td>~10–20 weights on ~1,000 classic data points</td><td>~10–20 weights, literature sets</td><td>millions of parameters</td><td>41 weights on 216k Megascale rows</td></tr>
<tr><td>269-residue scan</td><td>7–14 CPU-hours</td><td>seconds</td><td>seconds</td><td>3.6 s fast / 8 s default (1 GPU)</td></tr>
<tr><td>Interpretable</td><td>yes</td><td>partly</td><td>no</td><td>yes, per term</td></tr>
<tr><td>Megascale ρ</td><td>0.63</td><td>0.50–0.52</td><td>≈0.77 (published)</td><td><b>0.69</b></td></tr></tbody></table></div>
<p class="sub">Four choices make the difference: the wild type is modelled exactly like the mutant; shapes are averaged rather than picked; physics is mixed with statistics and the mix is fitted; and everything is batched on the GPU.</p>

<h2>Worked example: four real mutations in 1PSE</h2>
<p class="sub">Real intermediate values from running the shipped default model on a held-out protein. Across all 1,213 measured mutants of 1PSE, PhysDDG reaches ρ = 0.60, FoldX 0.49.</p>
<div class="wex" id="wex"></div><div class="panel" id="wbody"></div>

<h2>Pseudocode</h2>
<div class="pseudo">predict(structure, mutations, stage=2, mode="xtal"):
    prot = annotate(load(structure))                    # step 1
    for each mutated site i (batched on GPU):
        for aa in 20 amino acids:                       # wild type treated like a mutant
            R   = library_rotamers(aa, phi_i, psi_i)    # step 2: at most 30 shapes
            U_r = sum_terms E_term(r) - kT*ln p(r|phi,psi)   # step 3
            if stage == 2: mean_field_repack(site + up to 16 neighbours within 6 A)
            F[aa], E_term[aa], S[aa] = boltzmann_average(U_r)
        stats[aa] = env_potential, ramachandran, N-cap, ...  # step 4 (PDB-derived)
    for each mutation wt->mut at i:
        x   = (E_term, S, stats)[mut] - (E_term, S, stats)[wt]  (+ burial, cavity)
        ddg = b + w.x + r[mut] - r[wt]                  # step 5: 41 fitted numbers
        report ddg, per-term parts, flags               # step 6</div>

<h2>Results</h2>
<p class="sub">Fitted on Megascale-train only; every choice made on validation proteins and homology-grouped cross-validation. Held out: Megascale-test (28 domains), FireProt-test, S669, S461, Human Domainome, paired crystal/AlphaFold set.</p>
<div class="fig"><div data-fig="ladder"></div><div class="cap">Per-protein median Spearman ρ, 21 natural held-out domains, identical rows and structures. ThermoMPNN is a published number, not re-run.</div></div>
<div class="fig"><div data-fig="per_protein"></div><div class="cap">Recomputed from the stored predictions: PhysDDG ranks better than FoldX on 19 of 21 domains.</div></div>
<div class="panel"><ul class="lim">
<li><b>Ranking vs FoldX on Megascale:</b> +0.063, project CI [+0.006, +0.120] (a bootstrap I reran gives [−0.003, +0.120]). Direction is convincing (19/21), size uncertain.</li>
<li><b>Scale of predictions:</b> Pearson 0.70 vs 0.60 and RMSE 0.78 vs 1.85 kcal/mol (FoldX clipped to [−3, 8]). Strong.</li>
<li><b>Against the 10-parameter statistics model:</b> +0.044 [0.018, 0.071]. The physics adds a real, modest amount.</li>
<li><b>Other assays:</b> no clear win. FireProt Pearson 0.45 vs FoldX 0.42 (not significant); S669 0.34 and S461 0.55, level with DDGun3D and the 10-parameter model; Domainome 0.42 vs 0.43, and 0.39 vs 0.39 after removing training homologues.</li>
<li><b>Selection caveat:</b> at least 12 evaluations touched test data, 5 of which changed what ships. Treat differences ≤ 0.02 as optimistic.</li></ul></div>

<h2>Limitations and when not to use it</h2>
<div class="three"><div class="panel"><b>Shown not to work well</b><ul class="lim"><li>Forward and reverse predictions agree only at r ≈ 0.24 (KORPM 0.88).</li><li>Finding stabilising mutations (top-5 precision 0.15–0.23).</li><li>Buried small→large (0.49 vs FoldX 0.55).</li><li>pLDDT &lt; 70: no method works; flagged.</li><li>Crystal benchmarks: no clear gain over the 10-parameter model.</li></ul></div>
<div class="panel"><b>Assumptions</b><ul class="lim"><li>Fixed backbone; one structure; single substitutions.</li><li>Weights from one assay (small domains, 25 °C); other assays' ΔΔG are about 1.7× larger.</li><li>Megascale structures are AlphaFold2 models, so structure source and assay are confounded.</li></ul></div>
<div class="panel"><b>Not validated / do not use for</b><ul class="lim"><li>Generalisation across assays (claim withdrawn).</li><li>Ligand or metal sites, interfaces, multiple mutations, indels, pH or temperature, disordered regions.</li><li>Best accuracy on well-folded domains: use deep learning (≈0.77).</li></ul></div></div>

<h2>Terms</h2><div class="tablewrap"><table><tbody id="gloss"></tbody></table></div>
<h2>Ideas taken from the literature</h2>
<p class="sub">CHARMM19 and EEF1 energy parameters (Brooks 1983; Neria 1996; Lazaridis &amp; Karplus 1999). Separately weighted energy terms: FoldX (Guerois 2002; Schymkowitz 2005). Same protocol for wild type and mutant: Rosetta cartesian_ddg, shown to matter on model structures by Valanciute 2023. Rotamer libraries and priors: Dunbrack &amp; Karplus 1993, Shapovalov &amp; Dunbrack 2011 (PhysDDG derives its own from PISCES, Wang &amp; Dunbrack 2003). Mean-field packing: Koehl &amp; Delarue 1994; Kamisetty 2008. Environment potentials and volume terms: PoPMuSiC (Gilis &amp; Rooman 1997; Dehouck 2009). N-cap and helix dipole: Richardson 1988; Doig &amp; Baldwin 1995; Hol 1978; Shoemaker 1987. Unfolded-state mass balance: Rossi 2025. Censoring: Tsuboyama 2023, Tobin 1958, Huber 1964. AlphaFold handling: MAVISp trimming (Arnaudi) and keeping AlphaFold side chains (Vangaru 2025). Splits: ThermoMPNN (Dieckhaus 2024). Three ingredients (environment potential, Boltzmann averaging, censoring-aware loss) came from an AI reviewer agent (TP2). Citations come from the project's manuscript and were not re-verified here.</p>
</section>

<section class="tab" id="t-journey" hidden>
<p class="lede" style="margin-top:1.4rem"><b>The final method looks the way it does because of eight decisions, most forced by evidence that contradicted an early assumption.</b> Seven fell within about seven hours on 26 October; the eighth, an audit that reversed the shipped model, came the next night.</p>
<div class="fig"><div id="timeline" data-fig="timeline"></div><div class="cap">Click a label to open the turning point. Blue: an idea that shaped the method. Orange: a failure that taught a rule. Red: an audit that reversed a decision.</div></div>
<h2>Turning points</h2><div id="tps"></div>
<h2>Dead ends that taught something</h2>
<div class="tablewrap"><table><thead><tr><th>Attempt</th><th>Result</th><th>Lesson</th></tr></thead><tbody>
<tr><td>Refit FoldX's 22 terms</td><td>0.541 vs stock 0.569</td><td>re-weighting an existing term set is not where gains are</td></tr>
<tr><td>GPU Rosetta (tmol) as engine</td><td>raw 0.49, refit 0.60, 3.1 min per scan</td><td>kept as a baseline only</td></tr>
<tr><td>Backbone ensemble</td><td>no gain at 9× cost</td><td>local side-chain flexibility is enough</td></tr>
<tr><td>Elastic-network dynamics</td><td>correlates −0.85 to −0.91 with contact number</td><td>burial in disguise</td></tr>
<tr><td>Refit 32 physical constants</td><td>no ranking gain; all within ×/÷1.6; two assays agree (r = 0.80)</td><td>literature values are near a local optimum</td></tr>
<tr><td>Double mutants for coupling</td><td>no distance trend after removing the assay floor</td><td>little pairwise signal to capture</td></tr>
<tr><td>Evolution profile (1 weight)</td><td>+0.04 Megascale, +0.07 Domainome; vanishes on 14 strict proteins</td><td>different information source; never wired into the predictor</td></tr>
<tr><td>Joint Megascale+FireProt default</td><td>retracted by audit 2</td><td>a gain carried by 8 rows is not a gain</td></tr></tbody></table></div>
<h2>The research process</h2>
<div class="three">
<div class="panel"><b>Main agent (PI)</b><p>Wrote the brief into PROMPT.md, the first design, gates and every decision (D1–D23). Made the calls that mattered: accept the outside critique, move the bar to the 10-parameter model, choose transfer over home-test gain, retract the joint model. It also declared a gate passed that audit 1 showed failing, and twice shipped a term built on test rows or on 8 rows.</p></div>
<div class="panel"><b>Subagents (29)</b><p>Engine (geometry, rotamers, scan, predictor), energy terms, fitting (nine assignments over two days), AlphaFold. Three roles produced the evidence discipline: the design critic and two auditors. Others curated data, ran competitors, and wrote the manuscript and release.</p></div>
<div class="panel"><b>Productive vs circular</b><p>Early, cheap, independent checks changed the design each time. Circular: the shipped family flipped from Megascale-only to joint and back; the main ladder script was rerun at least ten times; repulsion-cap variants cost hours and landed in noise. Safety-classifier stops interrupted 16 of 29 agent logs without changing results.</p></div></div>
<div class="panel"><b>Where humans changed direction.</b> Rarely. The brief fixed the goals ("generalise across assays" is the sentence the PI cites when dropping the best-scoring term). Later messages were logistics: "keep going", "try again" after a usage limit, "are you done?", and release instructions (keep the full git history; remove personal identifiers).</div>
</section>

<section class="tab" id="t-insights" hidden>
<p class="lede" style="margin-top:1.4rem"><b>Most of what was learned is about the problem, not the code.</b> Ranked within each category by generality and surprise; links go to where each arose.</p>
<h2>Scientific insights</h2><div id="sci"></div>
<h2>Methodological insights</h2><div id="meth"></div>
<h2>Engineering details (reproduction only)</h2><ul class="lim" id="eng"></ul>
<p class="sub">The full tables with evidence and file pointers are in INSIGHTS.md.</p>
</section>
<footer>Built from the project record: RESEARCH_TREE.md, REPORT.md, notes/, transcripts, git history, and a rerun of the shipped model on 1PSE.</footer>
</div>
"""

html = f"""<title>PhysDDG Explainer</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;600&family=Newsreader:wght@600;700&display=swap">
<style>{CSS}</style>
{BODY}
<script type="application/json" id="data">{json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/")}</script>
<script>{JS}</script>
"""
out = H / "physddg_explainer.html"
out.write_text(html)
(H / "index.html").write_text(html)   # GitHub Pages entry point
print("wrote", out, "and index.html", len(html) // 1024, "KB")
