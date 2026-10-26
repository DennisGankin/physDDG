# Literature: fast side-chain packing, rotamer libraries, energy functions, GPU tools, runtimes

Scope: algorithms/data needed to build a fast, physics-inspired, GPU-batched ΔΔG engine (PhysDDG) without DL.
Compiled 2026-10-26. Tags: **[V]** number/statement read in the primary source (full text, abstract, official file or
README); **[S]** from an abstract or secondary summary only; **[U]** not verified (e.g. citation details from memory,
or primary text not accessible).

---

## 0. Bottom line

- **Packing**: classic rotamer packers are fast enough for local mutant repacking: FASPR packs 379 proteins in 34.3 s
  total (≈ 0.09 s/protein, DEE + tree decomposition) **[V]**; SCWRL4 86 % χ1 / 75 % χ1+2 within 40° **[V]**. DL packers are
  ~10–20 % better in RMSD but are DL (excluded) and, importantly, **no packer beats AF's own side chains on AF backbones**
  (Vangaru 2025, see lit_alphafold.md). For ΔΔG we only need to repack a 6–8 Å shell around one site.
- **Ready-made GPU physics exists**: **tmol** ("Rosetta on the GPU", PyTorch + CUDA, Apache-2.0, batched scoring,
  packing, minimisation, FastRelax with beta2016 score function; on PyPI) **[V]**. It is the closest prior art to the
  PhysDDG engine and a strong baseline/reference implementation (or dependency) — not DL.
- **Energy terms**: the two empirical ΔΔG workhorses are FoldX (occupancy-scaled transfer-energy terms; 2002 weights
  Wvdw 0.2, WsolvH 1.4, WsolvP 1.25, Wmc 1.0, Wsc 0.75 **[V]**; current docs: all 1 except Wvdw 0.33 **[S]**) and Rosetta
  REF2015 (19 weighted terms + 20 per-aa reference energies; weights copied below from the official weights file **[V]**).
- **Rotamer library licensing**: Dunbrack 2010 bbdep is CC BY 4.0 (free incl. commercial, attribution) but the official
  download needs a (free) application/approval **[V]** → **access step to flag**. Copies ship inside FASPR (`dun2010bbdep.bin`,
  MIT repo) and tmol (`dunbrack.bin`, Apache-2.0 repo) **[V]**. Richardson Top8000 rotamer distributions are on GitHub
  under CC BY 4.0 and Top2018 residue-filtered coordinates on Zenodo (CC) **[V]** — enough to derive our own library.
- **Runtimes to beat**: FoldX ≈ 34 CPU-s per variant (proteome-scale measurements) **[V]**; Rosetta cartesian_ddg
  ≈ 20+ CPU-min per variant (3 iterations, derived from RaSP Table 4) **[V-derived]**; DL tools: RaSP 0.4 s/position,
  ThermoMPNN full SSM 2 s (105 aa) to 8 s (693 aa) on one V100, Stability Oracle ~50 ms/residue **[V]**.
- **Cheap accuracy lever**: an unfolded-state/"mass-balance" per-aa term raised FoldX Pearson by +0.18 on S461
  (Rossi et al. 2025) **[V]** — zero runtime cost.

---

## 1. Side-chain packers

### 1.1 Non-DL packers (usable inside PhysDDG)

| Method | Algorithm | Accuracy (as reported) | Speed (as reported) | Ref |
|---|---|---|---|---|
| **SCWRL4** | bbdep 2010-type KDE library; samples around rotamers; anisotropic H-bond term; short-range soft vdW; k-DOP collision detection; **tree decomposition** of interaction graph; params fit in crystal environment | 379-protein set: **86 % χ1, 75 % χ1+2 within 40°**; 89 % / 80 % for well-ordered side chains | "comparable speed" to SCWRL3 (abstract) | Krivov GG, Shapovalov MV, Dunbrack RL. *Proteins* 77:778 (2009) doi:10.1002/prot.22488 **[V]** |
| **FASPR** | optimised scoring + deterministic search: **DEE + tree decomposition**; Dunbrack 2010 library | 69.1 % of all χ within 20° (SCWRL4 68.8, CISRR 68.6, RASP 67.8, SCATD 61.7) | **379 proteins in 34.3 s**, fastest tested | Huang X, Pearce R, Zhang Y. *Bioinformatics* 36:3758 (2020) doi:10.1093/bioinformatics/btaa234 **[V]**. Code MIT, C++ (`g++ -O3`), accepts a sequence to build mutants; https://github.com/tommyhuangthu/FASPR **[V]** |
| **OPUS-Rota** (2008) | OPUS-PSP-type potential, fast search | χ1 / χ1+2 +5.4 / +8.8 pts over SCWRL(3) | ≈ SCWRL speed | Lu M, Dousis AD, Ma J. *Protein Sci* 17:1576 (2008) doi:10.1110/ps.035022.108 **[V]** |
| **OPUS-Rota2** (2019) | improved OPUS-Rota, non-DL as far as the abstract says | more accurate than OPUS-Rota, SCWRL4, OSCAR-star; slightly below OSCAR-o | "2 orders of magnitude faster than OSCAR-o" | Xu G, Ma T, Du J, Wang Q, Ma J. *JCTC* 15:5154 (2019) doi:10.1021/acs.jctc.9b00309 **[S]** (no DL mentioned in abstract; volume/page **[U]**) |
| OPUS-Rota3/4/5 | **DL** (RotaNN DNN; Rota4 gradient-based with DL predictors; Rota5 3D-Unet + RotaFormer) — excluded | Rota3: 51.14 % χ within 20° on CAMEO-Hard61 vs OSCAR-star 50.87, SCWRL4 50.40, FASPR 49.85 | — | Xu G et al. *JCIM* 2020 doi:10.1021/acs.jcim.0c00951; *Brief Bioinform* 2022 doi:10.1093/bib/bbab529; *Structure* 2024 doi:10.1016/j.str.2024.03.015 **[S]** |
| **RASP** | CIS-RR clash-elimination + optimised terms + mixed search | comparable to best | "~40× faster than CIS-RR" | Miao Z, Cao Y, Jiang T. *Bioinformatics* 27:3117 (2011) doi:10.1093/bioinformatics/btr538 **[S]** (pages **[U]**) |
| **CIS-RR / CISRR** | clash-detection-guided iterative search + rotamer relaxation | low clashes | slow (motivated RASP) | Cao Y, Song L, Miao Z, Hu Y, Tian L, Jiang T. *Bioinformatics* 27:785 (2011) **[U]** |
| **OSCAR-star / OSCAR-o** | orientation-dependent energy; OSCAR-o optimised but slow | OSCAR-o among most accurate non-DL | OSCAR-o ~100× slower than OPUS-Rota2 | Liang S, Zheng D, Zhang C, Standley DM. *Bioinformatics* 27:2913 (2011); Liang S, Zhou Y, Grishin N, Standley DM. *J Comput Chem* 32:1680 (2011) **[U]** |
| **Sccomp** | solvation + contact-surface scoring | — | — | Eyal E, Najmanovich R, McConkey BJ, Edelman M, Sobolev V. *J Comput Chem* 25:712 (2004) **[U]** |
| **Rosetta packer** | Metropolis **simulated annealing** over rotamers with precomputed one-/two-body energy tables (interaction graph), plus rotamer trials / min-pack variants; REF2015 | best physics packer on core residues; fewest clashes among classic tools (tables below) | slowest of the classic tools (≈150× AttnPacker GPU time) | Kuhlman B, Baker D. *PNAS* 97:10383 (2000); Leaver-Fay A et al. *Methods Enzymol* 487:545 (2011) **[U cite]** |
| **TreePack** | tree decomposition (exact) | — | up to 90× faster than SCWRL3 on large proteins | Xu J, Berger B. *J ACM* 53:533 (2006) **[S]** |
| **Force Field X ManyBody** | many-body expansion (2-/3-body, AMOEBA + GK implicit solvent) + many-body DEE/Goldstein; GPU energies via OpenMM | used on AF2 models: clashscore 20.75 → 0.11 | GPU side-chain energies ~25× faster than first implementation | Gogal RA et al. *J Chem Phys* 2024 doi:10.1063/5.0214652 **[V]**; 25× from FFX website **[S]** |
| **OSPREY 3.0** | provable design: DEE/A*, continuous flexibility, K* | — | >100× faster than OSPREY 2; **GPU support adds >10× speedup** | Hallen MA … Donald BR. *J Comput Chem* 39:2494 (2018) doi:10.1002/jcc.25522 **[V abstract]** (pages **[U]**) |

**Search-algorithm references (combinatorial core)**
- Dead-end elimination: Desmet J, De Maeyer M, Hazes B, Lasters I. *Nature* 356:539 (1992) doi:10.1038/356539a0 **[S]**;
  Goldstein RF. *Biophys J* 66:1335 (1994) **[U]**; generalised DEE: Looger LL, Hellinga HW. *J Mol Biol* 307:429 (2001) **[U]**.
- A*: Leach AR, Lemon AP. *Proteins* 33:227 (1998) **[U]**.
- Tree decomposition: Xu & Berger 2006 (above); used by SCWRL4 and FASPR **[V]**.
- Belief propagation / message passing: Yanover C, Weiss Y. "Approximate inference and protein-folding." NIPS 2002
  **[U]**; Yanover C, Schueler-Furman O, Weiss Y. *J Comput Biol* 15:899 (2008) (learning energies for SC prediction)
  **[S]**; SPRINT toolbox: Fromer M … Linial M. *Bioinformatics* 26:2466 (2010) **[S]**; Kamisetty H, Xing EP,
  Langmead CJ. "Free energy estimates of all-atom protein structures using generalized belief propagation." *J Comput
  Biol* 15:755 (2008) **[S]** — BP gives side-chain conformational free energies (entropy) directly, relevant to ΔΔG.
  BP/mean-field are naturally batched (tensor ops over rotamer marginals) → GPU-friendly.
- Huang X, Pearce R, Zhang Y. "Toward the accuracy and speed of protein side-chain packing: a systematic study on
  rotamer libraries." *JCIM* 60:410 (2020) **[S, cited in FASPR README]**.

### 1.2 Benchmarks incl. DL packers (for context only; DL excluded from PhysDDG)

**AttnPacker paper** — McPartlon M, Xu J. *PNAS* 120:e2216438120 (2023) doi:10.1073/pnas.2216438120 (PMC10266014) **[V]**.
CASP13 native backbones (columns: side-chain RMSD all / surface / core Å; χ1–χ4 MAE °; all-χ-within-20° accuracy all /
surface / core %; clashes at 100/90/80 % vdW):

| Method | RMSD all | surf | core | χ1 | χ2 | χ3 | χ4 | Acc all | surf | core | clash100 | 90 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SCWRL4 | 0.934 | 1.200 | 0.597 | 27.64 | 28.97 | 49.57 | 61.54 | 56.2 | 45.2 | 71.2 | 115.3 | 20.6 | 4.6 |
| FASPR | 0.910 | 1.167 | 0.604 | 27.04 | 28.41 | 50.30 | 60.89 | 56.4 | 45.5 | 70.3 | 112.8 | 23.3 | 5.6 |
| Rosetta packer | 0.872 | 1.171 | 0.509 | 25.88 | 28.25 | 48.13 | 59.82 | 58.6 | 45.9 | 75.3 | 73.8 | 7.9 | 2.6 |
| DLPacker | 0.772 | 1.018 | 0.483 | 22.18 | 27.00 | 51.22 | 70.04 | 58.8 | 47.3 | 73.9 | 64.3 | 7.3 | 2.0 |
| AttnPacker | 0.669 | 0.881 | … | (best in all columns) | | | | | | | | | |

Relative wall time for all 83 CASP13 targets (Table 2): AttnPacker 1.0 (= 68 s on one RTX A6000), **FASPR 0.5**,
**SCWRL4 14.7**, DLPacker 124.4 (GPU), **Rosetta packer 151.7**; CPU tools on one AMD EPYC 7742 core.
⇒ per target ≈ FASPR 0.4 s, SCWRL4 12 s, Rosetta 124 s **[V-derived]**.

**Vangaru & Bhattacharya 2025** — *Brief Bioinform* 26:bbaf297 doi:10.1093/bib/bbaf297 (PMC12192453) **[V]**.
CASP14 native backbones (RMSD all/core/surface Å; χ1–χ4 MAE °; rotamer recovery RR % = all χ MAE < 20°; clashes 100/90/80 %):
- SCWRL4 1.04 / 0.61 / 1.26; χ1 32.22, χ2 31.65, χ3 50.21, χ4 55.10; RR 47.5; clashes 158.3 / 40.2 / 11.8
- FASPR 1.03 / 0.62 / 1.24; 31.97, 31.27, 49.43, 55.74; RR 47.8; 152.9 / 41.8 / 13.0
- PyRosetta packer 1.00 / 0.55 / 1.23; 30.98, 31.29, 49.31, 55.58; RR 48.9; 104.3 / 22.1 / 8.4
- DLPacker 0.90 / 0.50 / 1.11; RR 50.6; FlowPacker 0.80, PIPPack 0.79, AttnPacker 0.79 Å (all RMSD); FlowPacker,
  PIPPack, DiffPack have the highest RR. DL > classical on native backbones; Rosetta fewest clashes.
- On **AF2/AF3 backbones every packer is worse than AF's own side chains** (Wilcoxon, 792 tests; e.g. CASP15 AF2
  side chains 0.902 Å, χ1 MAE 28.05°).

**PIPPack** — Randolph NZ, Kuhlman B. *Proteins* 92:1220 (2024) doi:10.1002/prot.26705 **[V abstract + bioRxiv text]**:
~1,400-chain Top2018 test (B-factor filtered: χ with any atom B > 40 Å² discarded); competitive RR/RMSD with DLPacker,
AttnPacker, DiffPack and "fastest runtimes … for almost every protein size", runs on CPU or GPU; table values are images
in the preprint **[numbers U]**. Post-hoc MCMC resampling of clashing residues fixes clashes at negligible cost.

Other DL packers named in the brief: DLPacker (Misiura M, Shroff R, Thyer R, Kolomeisky AB. *Proteins* (2022)
doi:10.1002/prot.26311 **[S]**), DiffPack (Zhang Y et al., NeurIPS 2023 **[U]**), FlowPacker (Lee JS, Kim PM.
*Bioinformatics* 41:btaf010 (2025) doi:10.1093/bioinformatics/btaf010 **[S]**), H-Packer (Visani GM et al., arXiv
2311.09312, 2023 **[U]**). **ZymCTRL is an enzyme language model, not a packer.** Misiura 2022 runtime/accuracy table not
re-read **[U]**.

Takeaway for PhysDDG: accuracy differences between SCWRL4/FASPR/Rosetta are ≤ 0.04 Å RMSD and ≤ 1.5 pts RR —
noise relative to ΔΔG errors. Choose by speed/batchability: DEE + small-graph exact search (FASPR style) or batched
mean-field/BP on GPU.

---

## 2. Rotamer libraries

| Library | Content | Licence / access | Ref |
|---|---|---|---|
| **Dunbrack 2010 smoothed bbdep** | rotamer probabilities, mean χ and σ as smooth functions of (φ,ψ) via adaptive KDE / kernel regression; non-rotameric χ (Asn/Asp/Gln/Glu/His/Phe/Tyr/Trp terminal χ) as continuous densities; 10° φ/ψ grid text files (`ALL.bbdep.rotamers.lib`, per-residue `*.bbdep.densities.lib`); default 5 %-stepdown smoothing (also 0/2/10/20/25 %) | Page states **CC BY 4.0 since 2019-07-25, "same for both academia and industry … free, redistributable, and modifiable as long as you acknowledge the source"**; the linked licence text is written as a database licence (ODC-By-like). **Download page is provided "once you have applied for a 2010 library license and have been approved"** → needs a manual application (flag to user). | Shapovalov MV, Dunbrack RL. *Structure* 19:844 (2011) doi:10.1016/j.str.2011.03.019 (PMC3118414) **[V]**; http://dunbrack.fccc.edu/lab/bbdep2010 , `/lab/bbdep2010_lic`, `/lab/bbdep2010_tutorial` **[V]** |
| Copies of Dunbrack 2010 already redistributed | FASPR repo `dun2010bbdep.bin` (binary; MIT-licensed repo); tmol `tmol/database/default/scoring/dunbrack.bin` + `.yaml` (Apache-2.0 repo); Rosetta database (Rosetta non-commercial licence) | usable now under CC BY attribution; binary formats need a reader (FASPR C++ source / tmol Python loader) | GitHub trees **[V]** |
| Dunbrack 2002 bbdep / bbind | older discrete library (SCWRL3 era) | — | Dunbrack RL. *Curr Opin Struct Biol* 12:431 (2002) **[U]** |
| **Richardson Top8000 ("ultimate") rotamer distributions** | multi-dimensional χ distributions from ~8,000 chains, residue-filtered by density fit (~1 M residues); favored/allowed (0.3–2 %)/outlier contours; used in MolProbity/PHENIX | **GitHub rlabduke/reference_data, CC BY 4.0**: `Top8000_rotamer_central_values`, `Top8000_rotamer_pct_contour_grids`, `Top8000_rotamer_residues.csv`, chain lists at 50/70/90/95 % homology **[V]**; backbone-independent | Hintze BJ, Lewis SM, Richardson JS, Richardson DC. *Proteins* 84:1177 (2016) doi:10.1002/prot.25039 **[V]** |
| Lovell "penultimate" library | 153 backbone-independent rotamers, B-factor/density-filtered | — | Lovell SC, Word JM, Richardson JS, Richardson DC. *Proteins* 40:389 (2000) **[U]** |
| **Top2018 dataset** (to derive our own) | ~3 M residues in ~15,000 single-chain files; residue-level filtering for density support, no clashes/geometry outliers; main-chain+Cβ-filtered and full-residue-filtered versions, each at 4 identity levels | **Zenodo, Creative Commons, "freely available for all uses"** | Williams CJ, Richardson DC, Richardson JS. *Protein Sci* 31:290 (2022) doi:10.1002/pro.4239 (PMC8740842) **[V abstract]** (pages **[U]**) |

**Deriving a backbone-dependent library ourselves** (if the Dunbrack application is slow): take Top2018 full-residue-filtered
at 50–70 % identity (optionally B < 40 Å² per χ as in PIPPack); compute φ, ψ, χ1–4; assign rotamer bins by Top8000 central
values (or k-means on χ with periodic distance); estimate P(rot | φ,ψ) with a von Mises kernel on the torus (Dunbrack used
adaptive KDE) on a 10° grid; mean/σ of χ per bin by kernel-weighted circular statistics; treat terminal sp2 χ (Asn χ2,
Asp χ2, Phe/Tyr χ2, His χ2, Trp χ2, Gln/Glu χ3) as continuous with ~10–30° sampling. Expanded rotamers: ±1σ on χ1/χ2
(SCWRL4/Rosetta practice: "averaging over samples of conformations about the positions in the rotamer library" **[V]**).
Use −log P(rot|φ,ψ) as the Dunbrack energy term (Dunbrack tutorial: library "should not be used in a way where all
rotamers are treated equally" **[V]**).

---

## 3. Energy functions

### 3.1 FoldX (FOLD-X / FOLDEF)
Guerois R, Nielsen JE, Serrano L. *J Mol Biol* 320:369 (2002) doi:10.1016/S0022-2836(02)00442-4 **[V, PDF]**;
Schymkowitz J et al. *Nucleic Acids Res* 33:W382 (2005) doi:10.1093/nar/gki387 (PMC1160148) **[V]**;
Delgado J, Radusky LG, Cianferoni D, Serrano L. FoldX 5.0, *Bioinformatics* (2019) doi:10.1093/bioinformatics/btz184 **[S]**.

ΔG = Wvdw·ΔGvdw + WsolvH·ΔGsolvH + WsolvP·ΔGsolvP + ΔGwb + ΔGhbond + ΔGel + ΔGkon + Wmc·TΔSmc + Wsc·TΔSsc + ΔGclash
- **2002 fitted weights**: Wvdw = 0.2, WsolvH = 1.4, WsolvP = 1.25, Wmc = 1.0, Wsc = 0.75; ΔGhbond = −1.3 kcal/mol
  (polar–polar) and −1.4 kcal/mol (polar–charged) **[V]** (sign inferred; PDF extraction lost the minus glyph).
  Fitted on 339 deletion-type single mutants in 9 proteins (barnase, CI2, spectrin SH3, Src SH3, Sso7d, tenascin,
  FKBP, Ada2h, CheY) to avoid side-chain modelling; training R 0.70 (0.80 without 16 outliers, SD 0.75 kcal/mol);
  blind test on 667 mutants + 82 complex mutants; global R 0.83, SD 0.81 kcal/mol, slope 0.76 on 95 % of all 1,088
  mutants (1,030, outliers removed) **[V]**.
- **Current FoldX documentation**: "They are all 1, except for the van der Waals' contribution which is 0.33" **[S]**
  (foldxsuite.crg.eu, via search summary). Web-server paper: calibrated on > 1000 mutants, R 0.81, SD 0.46 kcal/mol **[V]**.
- Term definitions **[V]**: vdW and solvation from experimental transfer energies (water→vapour for vdW, water→organic
  for solvation), split into apolar/polar; explicit water bridges for waters making > 2 H-bonds; H-bonds from simple
  geometry with energies from double-mutant cycles; Coulomb with dielectric scaled by burial; helix-dipole pseudo-charges
  at helix termini and aromatic quadrupole charges; ΔSmc from φ/ψ statistics scaled by main-chain accessibility and
  H-bonding; ΔSsc from Abagyan side-chain entropy scaled by burial; clash term.
- **Occupancy scaling (key for speed)**: all solvation/vdW/electrostatic/entropy terms are multiplied per atom by
  Sfact = (Occ − Occmin)/(Occmax − Occmin), Occ = sum of fragmental volumes of atoms within **6 Å** (Holm & Sander solvent
  contact model) — chosen because it is much faster than SASA **[V]**. Trivially GPU-batched (one radius query).
- Defaults: T = 298 K (do not deviate > 10 K), ionic strength 0.05 M, "VdWDesign" switch for clash handling **[V]**.

### 3.2 Rosetta REF2015
Alford RF, Leaver-Fay A, … Gray JJ. *J Chem Theory Comput* 13:3031 (2017) doi:10.1021/acs.jctc.7b00125 (PMC5717763)
**[V via bioRxiv 10.1101/106054]**; weights from `database/scoring/weights/ref2015.wts` (RosettaCommons/rosetta GitHub,
Rosetta non-commercial licence) **[V]**:

| term | w | meaning |
|---|---|---|
| fa_atr | 1.0 | LJ attractive |
| fa_rep | 0.55 | LJ repulsive, **linearised for d < 0.6 σ** (softened) |
| fa_intra_rep | 0.005 | intra-residue repulsion |
| fa_sol | 1.0 | Lazaridis–Karplus isotropic desolvation (ΔGref removed) |
| fa_intra_sol_xover4 | 1.0 | intra-residue LK beyond 3 bonds |
| lk_ball_wtd | 1.0 | anisotropic polar desolvation (waters on ideal sites; w_aniso ~0.7, w_iso ~0.3) |
| fa_elec | 1.0 | Coulomb, sigmoidal ε(r) from 6 (core) to 80, constant below 1.45 Å, truncated/shifted at 5.5 Å |
| hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc | 1.0 each | orientation-dependent H-bond (fit to Top8000 polar contacts) |
| dslf_fa13 | 1.25 | disulfide |
| rama_prepro | 0.45 | Ramachandran (pre-proline aware) |
| omega | 0.4 | peptide planarity |
| p_aa_pp | 0.6 | P(aa | φ,ψ) |
| fa_dun | 0.7 | −log P(rot | φ,ψ) Dunbrack 2010 |
| pro_close | 1.25 | proline ring closure |
| yhh_planarity | 0.625 | Tyr OH planarity |
| ref | 1.0 | per-aa unfolded reference energies |

Reference energies (packing context, default; order A C D E F G H I K L M N P Q R S T V W Y):
1.32468, 3.25479, −2.14574, −2.72453, 1.21829, 0.79816, −0.30065, 2.30374, −0.71458, 1.66147, 1.65735, −1.34026,
−1.64321, −1.45095, −0.09474, −0.28969, 1.15175, 2.64269, 2.26099, 0.58223 **[V]**. (A "minimisation context" set
differs for A, C, M, P, R.) Rosetta energies are often scaled ÷2.9 to kcal/mol for cartesian_ddg (Cagiada 2023, citing
Park 2016) **[V]**.

LK atom parameters (Rosetta `atom_properties.txt`, columns LJ_RADIUS, LJ_WDEPTH, LK_DGFREE, LK_LAMBDA (=3.5 Å for all),
LK_VOLUME), e.g. CH3 2.0118/0.0626/7.293/3.5/25.855; Nbb 1.8025/0.1617/−9.969/3.5/15.992; OCbb 1.5406/0.1424/−8.007/3.5/12.196;
OOC LK_DGFREE −9.240; Nlys −20.865 **[V]** — a ready parameter table for an LK/EEF1-style solvation term.

### 3.3 Implicit solvation options
- **EEF1** (Lazaridis T, Karplus M. *Proteins* 35:133 (1999) doi:10.1002/(SICI)1097-0134(19990501)35:2<133::AID-PROT1>3.0.CO;2-N)
  **[V abstract]**: Gaussian solvent exclusion, group contributions from model compounds, CHARMM19 polar-H, **ionic side chains
  neutralised + distance-dependent dielectric**. Correlation length λ = 3.5 Å (6 Å for charged groups) **[U, from memory;
  Rosetta uses 3.5 Å for all atoms [V]]**. Pairwise, cutoff ~9 Å → cheap on GPU.
- **SASA-based atomic solvation parameters**: Eisenberg D, McLachlan AD. *Nature* 319:199 (1986); Wesson L, Eisenberg D.
  *Protein Sci* 1:227 (1992); fast analytic SASA LCPO (Weiser J, Shenkin PS, Still WC. *J Comput Chem* 20:217 (1999));
  Shrake–Rupley numerical SASA (*J Mol Biol* 79:351 (1973)) **[U cites]**. SASA is non-pairwise (needs neighbour
  overlaps); LCPO or occupancy/neighbour-count proxies are GPU-friendlier.
- **Generalized Born**: OBC (Onufriev A, Bashford D, Case DA. *Proteins* 55:383 (2004) doi:10.1002/prot.20033 **[S]**),
  GBn2 (Nguyen H, Roe DR, Simmerling C. *JCTC* 9:2020 (2013) doi:10.1021/ct3010485 **[S]**). Both in OpenMM (GPU).
  GB needs Born radii (pairwise descreening) then pairwise energy: O(N·k) with neighbour lists; more expensive and noisier
  for ΔΔG than LK unless averaged.

### 3.4 Hydrogen bonds and electrostatics
- Kortemme T, Morozov AV, Baker D. *J Mol Biol* 326:1239 (2003) doi:10.1016/S0022-2836(03)00021-4 **[V abstract]**:
  orientation-dependent HB from crystal-structure statistics (distance + angles at H and acceptor), beats Coulomb-only HB.
- Rosetta REF2015 hbond: polynomial fits on Top8000 contacts; separate sr/lr backbone, bb–sc, sc–sc weights **[V]**.
  Model with covalent-electrostatic hybrid: O'Meara MJ et al. *JCTC* 11:609 (2015) **[U]**.
- DREIDING 12-10 HB × cos⁴θ: Mayo SL, Olafson BD, Goddard WA. *J Phys Chem* 94:8897 (1990) **[U cite]**.
- FoldX: geometric HB with fixed energies (above) **[V]**.
- Coulomb with distance-dependent dielectric: EEF1 ε = r (neutralised groups); Rosetta sigmoidal ε(r) 6→80 with 5.5 Å
  cutoff **[V]**; FoldX burial-scaled ε + ionic strength **[V]**.
- LJ softening: Rosetta linear fa_rep below 0.6σ **[V]**; SCWRL4 short-range soft vdW **[V]**; FoldX "VdWDesign=false"
  ignores small overlaps **[V]**. Important on AF/side-chain-model inputs where small clashes are model error.

### 3.5 Unfolded state / reference model (cheap, high value)
- Rossi I, Barducci G, Sanavia T, Turina P, Capriotti E, Fariselli P. "Mass balance approximation of unfolding boosts
  potential-based protein stability predictions." *Protein Sci* 34:e70134 (2025) doi:10.1002/pro.70134 (PMC12023412) **[V]**:
  adding a per-amino-acid (wt/mut) unfolded-state term (fitted, or the Rose hydrophobicity scale) to "potential-like"
  predictors raised Pearson on S461 for FoldX **+0.18**, ProteinMPNN +0.18, Pythia +0.19 (0.41 → 0.56); methods
  already MBC-aware (Stability Oracle, PoPMuSiC) gain ~0; fitted aa coefficients correlate with the Rose scale; preserves
  antisymmetry; Pythia/MBC on Megascale PCC ≈ 0.70, RMSE 1.43. → PhysDDG should include 20 (or 2×20) reference
  parameters, like Rosetta `ref`.
- Kellogg EH, Leaver-Fay A, Baker D. *Proteins* 79:830 (2011) doi:10.1002/prot.22921 **[V abstract]**: similar ΔΔG accuracy
  across protocols with very different sampling *if* force-field resolution is matched to sampling; backbone sampling
  helps only when structural change is real; main error sources: desolvation vs buried polar balance, unfolded-state
  modelling.

---

## 4. GPU tooling and batched-energy tricks

| Tool | What | Notes | Ref |
|---|---|---|---|
| **tmol** | "Rosetta on the GPU": all-atom scoring (beta2016 = REF2015 successor), Dunbrack-based packer (`pack_rotamers`, PackerTask, samplers), minimisation, FastRelax; PyTorch + custom CPU/CUDA kernels; batched `PoseStack` padded to largest member; gradients through coordinates; CUDA-graph replay for fixed shapes; one process per GPU for multi-GPU | Apache-2.0, `pip install "tmol[cu12]"`, active (pushed 2026-10-23); no peer-reviewed paper found **[U]** | https://github.com/uw-ipd/tmol , docs https://uw-ipd.github.io/tmol/ **[V README/docs]** |
| **OpenMM 8** | GPU MD toolkit; implicit solvent (OBC/GBn2), custom forces via expressions, minimiser; Python API; ML-potential support (not needed) | fastest route to restrained Amber relax of AF models (AF2 itself uses OpenMM + Amber99sb) | Eastman P et al. *J Phys Chem B* (2024) doi:10.1021/acs.jpcb.3c06662 (PMC10846090) **[S]** |
| **TorchMD** | MD in PyTorch, AMBER params via parmed, autograd forces | **~60× slower than ACEMD3 on test systems, attributed to the lack of neighbour lists** — pure-PyTorch dense pair loops are slow; need cell/neighbour lists or local fixed-size environments | Doerr S et al. *JCTC* 17:2355 (2021) doi:10.1021/acs.jctc.0c01343 (PMC8486166) **[V]** |
| **JAX MD** | differentiable MD/energies in JAX with cell lists & neighbour lists | Apache-2.0 | Schoenholz SS, Cubuk ED. NeurIPS 2020, arXiv:1912.04232 **[U cite]**; https://github.com/jax-md/jax-md **[V repo]** |
| **DMFF** | differentiable force fields (JAX), incl. multipoles | LGPL-3.0 | https://github.com/deepmodeling/DMFF **[V repo]**; paper Wang X et al. *JCTC* 2023 **[U]** |
| **OSPREY 3.0** | GPU-accelerated continuous-flexibility design (>10× from GPU) | Java/Python | Hallen 2018 **[V abstract]** |
| **Force Field X** | AMOEBA/GK, many-body rotamer optimisation, GPU via OpenMM | Java | Gogal 2024 **[V]** |
| GPU FoldX-like tool | **none found** (FoldX is single-threaded CPU; MutateX parallelises by process) | — | Tiberti M et al. *Brief Bioinform* 2022 (MutateX) **[U]** |

Batched-evaluation tricks (synthesised; sources noted):
1. **Fixed-size local environments instead of global neighbour lists**: for single-site ΔΔG, gather the K nearest
   residues (e.g. all with any atom within 10–12 Å, padded to K≈32–64) around each site once; then all WT/mutant
   rotamer evaluations are dense tensors [n_mut, n_rot, K, atoms_per_res, 3]. Avoids the neighbour-list problem that
   slowed TorchMD; padding + masks as in tmol's PoseStack.
2. **Pair-energy tables** (Rosetta packer / FASPR / OSPREY design): precompute one-body (rotamer–backbone/fixed environment)
   and two-body (rotamer–rotamer) energies for the shell once per site; reuse across the 19 substitutions (only the
   mutated position's rotamer set changes). Then packing = small discrete optimisation (DEE + exhaustive/tree search, or
   batched mean-field/BP/ICM on GPU).
3. **Smooth, cutoff-truncated terms** (Rosetta 5.5 Å electrostatics, LK with ~6 Å, FoldX 6 Å occupancy) keep K small.
4. **Occupancy/neighbour-count burial** (FoldX) instead of SASA: one radius count per atom.
5. **Autograd for minimisation** of the shell (side-chain χ or Cartesian with restraints) with L-BFGS in PyTorch;
   CUDA graphs for repeated fixed-shape calls (tmol doc).
6. **Group proteins by size** to limit padding waste; shard by protein over GPUs, one process per GPU (tmol doc).
7. float32 everywhere; watch LJ r⁻¹² overflow → use Rosetta-style linearised repulsion.

---

## 5. Runtime reference points (ΔΔG tools)

| Tool | Runtime | Hardware / conditions | Source |
|---|---|---|---|
| **FoldX BuildModel** | ≈ 34 CPU-s per variant (24,187 Spike variants in 232 h; 161,825 variants/30 proteins in 64.5 CPU-days) | proteome-scale runs | Dunham AS, Beltrao P, AlQuraishi M. *Genome Biol* (2023) doi:10.1186/s13059-023-02948-3 **[V]** |
| FoldX (RaSP table) | SSM wall-clock: ELOB 107 aa 42,237 s; GCK 434 aa 309,762 s; F8 693 aa 559,050 s; 395–807 s per residue (19 mutants × 3 runs) | "parallelized" on 64 × 2.6 GHz Opteron cores — implies ~1,300 CPU-s/mutation if fully parallel, inconsistent with the 34 s above; treat as upper bound | Blaabjerg 2023 Table 4 **[V]** |
| **Rosetta cartesian_ddg** | SSM wall-clock ELOB 44,324 s, GCK 118,361 s, F8 133,178 s (192–414 s per residue, 3 iterations), pre-processing (relax) 677–17,211 s → ≈ 20–25 CPU-min per mutation if fully parallel | same 64-core server | Blaabjerg 2023 **[V, derived]** |
| Rosetta cartesian_ddg protocol | mutate, repack residue + neighbours within **6 Å**, Cartesian-minimise side chains within 6 Å and backbone of sequence neighbours; 3 runs (Park 2016) or until 2 lowest within 1 REU / best of 5 (Frenz 2020) | — | Frenz B et al. *Front Bioeng Biotechnol* 8:558247 (2020) doi:10.3389/fbioe.2020.558247 (PMC7579412) **[V]** |
| Rosetta flex ddG | 35 backrub models × 35,000 trials per mutant (binding ΔΔG) | CPU-hours per mutant **[U]** | Barlow KA et al. *J Phys Chem B* 122:5389 (2018) doi:10.1021/acs.jpcb.7b11367 **[S]**; settings from MAVISp methods **[V]** |
| PoPMuSiC 2.1 | all single mutations of a medium-size protein in < 1 min | web server | Dehouck Y et al. *BMC Bioinformatics* 12:151 (2011) (PMC3113940) **[S]** |
| DDGun / DDGun3D | untrained, very fast (linear combination of evolutionary/structure scores) | numbers not found **[U]** | Montanucci L et al. *NAR* 50:W222 (2022) doi:10.1093/nar/gkac325 **[S]** |
| MAESTRO | multi-agent statistical potentials, fast | numbers not found **[U]** | Laimer J et al. *BMC Bioinformatics* 16:116 (2015) **[U]** |
| EvoEF2 | physics-based design energy, C++, "accurate and fast" | numbers not found **[U]**; MIT | Huang X, Pearce R, Zhang Y. *Bioinformatics* 36:1135 (2020) doi:10.1093/bioinformatics/btz740 **[S]** |
| RaSP (DL) | 0.4 s per position after pre-processing (7–20 s), independent of length; 480–1,036× faster than Rosetta | V100 GPU | Blaabjerg 2023 **[V]** |
| ThermoMPNN (DL) | full SSM 2 s (105 aa) / 8 s (693 aa) | one V100, 8 CPUs | Dieckhaus H et al. *PNAS* 2024 doi:10.1073/pnas.2314853121 **[V]** |
| Stability Oracle (DL) | ~50 ms per residue (all 19 substitutions) | GPU | Diaz DJ et al. *Nat Commun* 2024 **[V]** |
| Sequence UNET (DL) | 420 ms per protein after 85 s initialisation | — | Dunham 2023 **[V]** |

Target for PhysDDG: FoldX-class or better accuracy at ≲ 10 ms per mutation on a GPU (i.e. ~3,000× FoldX CPU time),
which puts full human-proteome SSM (~2×10⁸ variants) at ~1 GPU-week across 4 GPUs.

---

## 6. Fast mutant-scanning tricks (what the field does; what to copy)

1. **Prepare the WT once** (repair/relax), then do only local work per mutation: FoldX RepairPDB → BuildModel; Rosetta
   relax once → cartesian_ddg (Frenz 2020: input relaxed 20× and best kept) **[V]**.
2. **Local shell repacking/minimisation only**: 6 Å shell in cartesian_ddg **[V]**; same treatment of WT and mutant
   (the WT is re-optimised with identical moves to cancel protocol bias) **[V]**.
3. **Position-level batching**: all 19 substitutions at a site share the environment, pair tables and WT reference;
   RaSP's constant 0.4 s/position (all 19 at once) shows the value of this grouping **[V]**.
4. **Replicates for stochastic protocols**: FoldX ×3–5 runs (MutateX/MAVISp default 5), Rosetta ×3 or convergence rule;
   a deterministic packer (FASPR-style DEE + tree search) removes this cost entirely.
5. **Free unfolded-state/reference terms** (Rosetta `ref`, mass-balance correction) — no structural work **[V]**.
6. **Antisymmetry check**: score the reverse mutation from the built mutant (Ssym-style) as a diagnostic; MBC preserves
   antisymmetry **[V]**.
7. **Skip what cannot matter**: residues with pLDDT < 50 or outside domains (trim, lit_alphafold.md); Gly/Pro special
   backbone handling only where needed (Frenz 2020 adds fragment sampling for Pro mutations — reduced egregious errors
   25.33 → 24.67 only) **[V]**.
8. **Cutoffs + float32 + dense padded tensors + CUDA graphs**, one process per GPU (§4).
