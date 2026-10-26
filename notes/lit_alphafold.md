# Literature: using AlphaFold / predicted structures for ΔΔG prediction

Scope: evidence relevant to a PhysDDG "AlphaFold mode" (physics-inspired, no DL in the predictor).
Compiled 2026-10-26 from primary sources (Europe PMC full text, bioRxiv JATS, publisher PDFs, GitHub, docs).
Tags: **[V]** = number read in the primary source by me; **[S]** = from abstract/secondary summary only;
**[U]** = unverified / could not locate the primary text. Sign: as in each paper (most use ΔΔG > 0 = destabilising).

---

## 0. Bottom line (for the impatient)

1. **High-confidence AF2 models are about as good as crystal structures as ΔΔG inputs**, consistently, for FoldX,
   Rosetta, RaSP, mCSM-type tools and ThermoMPNN-type tools (Akdel 2022; Blaabjerg 2023; Sora 2023; Diaz 2024;
   Gerasimavicius 2023). The loss is small (e.g. RaSP: mean Pearson of predictions xtal–xtal 0.97 vs xtal–AF 0.94).
2. **The loss is concentrated at pLDDT < 70 (backbone unreliable) and especially < 50** (Akdel 2022 Fig 3b;
   Tunyasuvunakool 2021; RaSP Table 3). But low-pLDDT residues also have genuinely smaller |ΔΔG| (mostly
   disordered/tolerant), so lower correlation there is partly a dynamic-range effect, not only model error.
3. **FoldX is more sensitive to input-structure errors than Rosetta cartesian_ddg** (Valanciute 2023: with homology
   models < ~45 % template identity FoldX and ddg_monomer fall to ~the structure-free null, cartesian_ddg stays
   near crystal performance). Relaxation/minimisation of the input therefore matters more for a FoldX-like (rigid,
   local) model — directly relevant to PhysDDG.
4. **Do not repack AF side chains with a generic packer on the AF backbone**: AF2/AF3 native side chains beat
   SCWRL4/FASPR/Rosetta/DL packers repacked on AF backbones (Vangaru & Bhattacharya 2025). Keep AF side chains for the
   WT environment; repack only the mutated site + shell.
5. **AF confidence itself is a strong but confounded signal**: ΔpLDDT (WT→mutant AF runs) barely tracks ΔΔG (Pak 2023,
   r = −0.17), but WT pLDDT at the site separates pathogenic/benign with AUROC 0.85 (Keskin Karakoyun 2023), better than
   stability predictors. pLDDT is a burial/order proxy — useful as a covariate/weight, dangerous as a leak.
6. **Missing context is the main systematic AF failure for stability**: metals (Domainome zinc fingers: all stability
   predictors fail), cofactors/ligands, oligomeric partners (Gerasimavicius 2022: full assemblies improve AUC
   0.652 → 0.677), membrane. AF3 can add ligands/ions; no ΔΔG benchmark of AF3-with-ligands found.
7. **Megascale (Tsuboyama 2023) "structures" are AF2 models**, trimmed by contact counts; designed domains were
   folded in single-sequence mode. So a model trained on Megascale is already trained on AF geometry; the AF mode
   matters most when *transferring* to large multi-domain natural proteins (Domainome, human proteome).
8. No paper found that quantifies "over-packed/over-idealised AF cores" for ΔΔG; and no head-to-head **AF3 vs AF2**
   ΔΔG benchmark found (as of 2026-10). Both are open questions we can test ourselves.

---

## 1. Large assessments: AF models vs experimental structures as ΔΔG input

### Akdel et al. 2022 — community assessment of AF2 applications
Akdel M, Pires DEV, … Lindorff-Larsen K, … Beltrao P. *Nat Struct Mol Biol* 29:1056–1067 (2022).
doi:10.1038/s41594-022-00849-w (PMC9663297). **[V]** (full text read)
- **DMS benchmark**: 33 proteins, 117,135 mutations; experimental structures for 31, AF2 for all 33. Predictors:
  FoldX, Rosetta, DynaMut2. "Correlation estimates … derived from the AF2 models consistently matched or were better
  than those derived from experimental models" (Fig 3a,b; per-protein −Pearson vs DMS).
- **Regions with pLDDT < 50 show lower concordance** (Fig 3b), but restricting to AF-only regions (no experimental
  model) at higher confidence still gives correlations comparable to experimental structures. Low-confidence regions
  have higher mutational tolerance in DMS (Fig 3c) → part of the drop is dynamic range.
- **Thermodynamic benchmark**: 2,648 single mutants, 121 (methods text says 132) proteins from ThermoMutDB; 11 predictors
  (I-Mutant, SAAFEC-SEQ, MUpro, mCSM, DUET, SDM, DynaMut, MAESTRO, ENCoM, DynaMut2, FoldX). AF2 models "comparable to
  those of experimental structures"; homology models degrade substantially below 40 % template identity (Fig 3d,
  Supp Fig 8). Exact per-method Pearson values are in the figure only (not extracted) **[U for numbers]**.
- AF2 models generated locally for the thermodynamic set; AFDB (July 2021) for the proteome analyses.
- Example of a confident side-chain error: Trp A111 (pLDDT 86.1) modelled backwards in eIF2B (Fig 6e).
- Domain parsing from PAE: Croll's `pae_to_domains` (graph community clustering on PAE),
  https://github.com/tristanic/pae_to_domains — used to extract rigid cores for molecular replacement.

### Blaabjerg et al. 2023 — RaSP (eLife 12:e82593, doi:10.7554/eLife.82593, PMC10266766) **[V]**
- Table 3: Pearson of RaSP ΔΔG predictions between two crystal structures (X1–X2) vs crystal–AF2, six proteins
  (PRMT5, PKM, FTH1, FTL, PSMA2, GNB1). X1–X2: 0.93–0.99; X–AF2: 0.89–0.97 (mean ρ̄ 0.97 vs 0.94).
  Split by AF pLDDT ≥ 90 vs < 90: high-pLDDT X–AF2 0.90–0.97; medium/low-pLDDT X–AF2 **0.66–0.96** (PRMT5 0.66/0.89,
  PSMA2 0.78/0.80 are the worst).
- Authors' caveat (response to reviewers): pLDDT correlates with structure class (loops/disorder have small ΔΔG), so
  "it is not trivial to assess to what extent prediction accuracy depends on pLDDT and why".
- Human proteome: ~300 M RaSP values on 23,391 AFDB v2 structures (pre-processed with Reduce + PDBFixer only, no trimming,
  no relax); fragments of large proteins kept as in AFDB.

### Valanciute et al. 2023 — stability predictions from homology models
Valanciute A, Nygaard L, Zschach H, Maglegaard Jepsen M, Lindorff-Larsen K, Stein A. *Comput Struct Biotechnol J*
(2023). doi:10.1016/j.csbj.2022.11.048 (PMC9729920). **[V]**
- 4 proteins, 344 mutations, Modeller models from 19 templates binned by identity; FoldX, Rosetta ddg_monomer,
  Rosetta cartesian_ddg.
- "Both FoldX and Rosetta ddg_monomer appear more sensitive to the input structure than cartesian_ddg"; at template
  identity < ~45 % (bins 4–5) FoldX and ddg_monomer are "at most only slightly better than the (second)
  structure-independent null model"; cartesian_ddg is ~unchanged for identity > 40 %.
- Null model: constant ΔΔG = 1.15 kcal/mol already gives MAE ≈ 1.1 kcal/mol on this set (reminder: MAE is weak).
- Cartesian protocol: constrained relax (coord constraints incl. side chains, cartesian, lbfgs_armijo_nonmonotone),
  3 iterations averaged. → Lesson for PhysDDG: local flexibility (minimisation in Cartesian space around the site)
  buys robustness to model error.

### Lihan, Lupyan, Oehme 2023 — FEP+ on homology models
*Protein Sci* 32:e4557. doi:10.1002/pro.4557 (PMC9878467). **[V via bioRxiv full text]**
- Not AF: FEP+ (OPLS4) on Prime homology models, 12 proteins / 316 mutations. Crystal R² 0.67, RMSE 1.12; homology
  (highest-identity) R² 0.58, RMSE 1.18; with local identity > ~40 % homology ≈ crystal (R² 0.63 vs 0.67).

### Pak & Ivankov 2022 — best templates vs models
*Bioinformatics* 38:4312. doi:10.1093/bioinformatics/btac515. **[S]** (abstract/announcement + search summary)
- Seven ΔΔG predictors (incl. FoldX); using the best available template directly is equal or better than a homology
  model built from it; for templates > 50 % identity homology-model error is ~1.5× the template error. AF models
  (CASP14) outperform best templates only at low identity (30–50 %). Structures were FoldX-RepairPDB'd. **[S]**

### Sora et al. 2023 — RosettaDDGPrediction (Papaleo lab)
Sora V, … Papaleo E. *Protein Sci* 32:e4527. doi:10.1002/pro.4527 (PMC9795540). **[V]**
- Case study p53 DBD (31 ThermoMutDB mutants): x-ray 2XWR vs AFDB model (res 91–289, pLDDT mostly > 70).
  Pearson 0.57–0.79 across protocols; x-ray runs 0.74–0.79; AF + talaris2014 (3 or 10 runs) 0.74–0.79;
  **AF + ref2015 cartddg: 0.57 (3 runs) / 0.68 (10 runs)**; cartddg2020 on AF "rescued" ref2015. Smallest MAE 0.90
  (x-ray, ref2015, 10 runs). → small n; protocol noise is comparable to the xtal/AF difference.

### Diaz et al. 2024 — Stability Oracle (DL, but informative)
*Nat Commun* (2024). doi:10.1038/s41467-024-49780-2 (PMC11266546). **[V]**
- ColabFold 2.3 (template-free, full UniProt sequence, best pLDDT of 5) for T2837 proteins; 3 structures removed
  (one failed, two US-score < 0.5; 50 mutations dropped). Using the AF **WT** structure: no change in classification,
  slightly lower regression metrics. Using AF-predicted **mutant** ("from/to") structures: 2–4 % drop.

### Gerasimavicius, Livesey, Marsh 2023 — stability predictors vs DMS
*Protein Sci* 32:e4688. doi:10.1002/pro.4688 (PMC10273344). **[V]**
- 9 predictors vs 49 DMS sets (170,940 variants); PDB structures preferred, AF2 (AFDB July 2021) only where no PDB
  coverage. FoldX and Rosetta best; both improve "considerably" with full complex structures; Foldetta consensus.
- Background claim they cite: AF2 is a suitable FoldX input even for proteins without homologs in training.

### Gerasimavicius, Livesey, Marsh 2022 — LOF/GOF/DN mechanisms
*Nat Commun* (2022). doi:10.1038/s41467-022-31686-6 (PMC9259657). **[V]**
- FoldX 5 on monomers vs full biological assemblies: pathogenic mean |ΔΔG| 3.39 → 3.89 kcal/mol; ClinVar vs gnomAD
  AUC **0.652 (monomer) → 0.677 (full assembly)**, p = 4.3e-71. HGMD external set used AF2 monomer models.
  → Missing partners in AF monomer models cost measurable signal.

### Keskin Karakoyun et al. 2023 — AF2 structures, cancer genes
*Front Genet* 14:1052383. doi:10.3389/fgene.2023.1052383 (PMC9988940). **[V/S]**
- 26 hereditary-cancer genes, >1000 ClinVar + 160 cohort variants; mCSM, MAESTRO, CUPSAT on AF2 structures, plus
  SAAF2EC-SEQ and MUpro. Stability predictors low–medium for pathogenicity (MUpro AUROC 0.534); **AF2 pLDDT alone
  AUROC 0.852**. Not a ΔΔG-accuracy study; shows pLDDT is a strong confounder in clinical benchmarks.

### McBride et al. 2023 (PRL) and McBride & Tlusty 2024 (bioRxiv)
- McBride JM, Polev K, Abdirasulov A, Reinharz V, Grzybowski BA, Tlusty T. "AlphaFold2 can predict single-mutation
  effects." *Phys Rev Lett* 131:218401 (2023). doi:10.1103/PhysRevLett.131.218401. **[V via bioRxiv 2022.04.14.488301]**
  Local distance difference (LDD) between AF2 models of WT and mutant correlates with PDB pairs differing by 1–3
  mutations (r ≈ 0.24 for one mutation) and weakly with GFP/mTagBFP2/PafA phenotypes. No ΔΔG.
- McBride JM, Tlusty T. "AI-predicted protein deformation encodes energy landscape perturbation." bioRxiv
  10.1101/2023.10.12.561990 (v2 2024). **[V]** 2,499 ThermoMutDB mutants; effective strain summed in a 15 Å sphere
  (averaged over many AF models; low-pLDDT residues excluded) vs ΔΔG: Spearman 0.57 in thermonuclease, 0.39–0.61 in
  the next four proteins, **0.36 pooled (0.41 vs |ΔΔG|)**. **FoldX 5 BuildModel (5 runs) on the top AF model: ρ = 0.26**
  pooled, i.e. worse than strain; a simple neighbour count (10 Å) gives ρ = 0.30. Single AF models are noisy; strain
  information decays beyond ~15 Å.

### Pak et al. 2023 — pLDDT change as a ΔΔG proxy
Pak MA, … Ivankov DN. *PLoS ONE* 18:e0282689. doi:10.1371/journal.pone.0282689 (PMC10019719). **[V]**
- 1,154 ThermoMutDB mutations / 73 proteins (< 250 aa); GFP 796 mutants. ΔpLDDT at site vs ΔΔG: PCC **−0.17**;
  Δ⟨pLDDT⟩: 0.02 (n.s.). 87 % of destabilising mutations have negative ΔpLDDT but magnitude does not track ΔΔG.
  Two-feature regression test r = 0.20. Conclusion: AF confidence changes are not a ΔΔG predictor.

### Buel & Walters 2022 — correspondence
Buel GR, Walters KJ. "Can AlphaFold2 predict the impact of missense mutations on structure?" *Nat Struct Mol Biol*
29:1–2 (2022). doi:10.1038/s41594-021-00714-2. **[S]** Argues AF2 is insensitive to point mutations (predicts WT-like
structure for destabilised/misfolding mutants); attributes this to reliance on MSA/PDB homology. Implication for us:
**never build mutant structures by re-running AF**; model the mutant physically on the WT AF backbone.

### Peccati, Alunno-Rufini, Jiménez-Osés 2023 — AF ensembles + Rosetta
*J Chem Inf Model* (2023). doi:10.1021/acs.jcim.2c01083 (PMC9930118). **[V]**
- Multi-mutant enzyme thermostability (LovD, LipA): single models give wildly varying ρ (−0.04 to −0.91 across five
  AF models); combining an ensemble ρ ≈ −0.84; AF-based ensembles beat crystal-based ones ("scaffold bias").
  → Averaging over several AF models (5 AF2 models / AF3 seeds) is a cheap robustness trick.

### Vliora, Tiberti, Papaleo 2026 — FoldX 5 vs 5.1 on AF models
*J Mol Graph Model* (2026) doi:10.1016/j.jmgm.2026.109624; bioRxiv 10.64898/2026.03.31.715598. **[V via summary of full
text]** 119 fold-non-redundant proteins, ~540k variants on AFDB models (MAVISp simple mode; RepairModel → MutateX →
BuildModel). FoldX5 vs 5.1: Pearson 0.956, MAE 0.52 kcal/mol, per-protein r 0.933 ± 0.077. Outliers (NUPR1 r 0.49,
TMEM127 0.42, TSC1 0.68) driven by low-pLDDT / conformationally different AF model versions (RMSD up to 3.8 Å between
AFDB releases). No experimental comparison. → AF model *version* and low-pLDDT regions alone can move FoldX outputs.

### Others referenced in the brief
- **Tsishyn et al.** (Rooman/Pucci): Tsishyn M, Hermans P, Rooman M, Pucci F. "Residue conservation and solvent
  accessibility are (almost) all you need for predicting mutational effects in proteins." *Bioinformatics* 41:btaf322
  (2025). doi:10.1093/bioinformatics/btaf322 (PMC12202862). **[V]** Uses ProteinGym AF models (RSA from AF);
  "essentially the same results" with alternative structures. The 2018 paper "Quantification of biases in predictions
  of protein stability changes upon mutations" (Pucci F, Bernaerts KV, Kwasigroch JM, Rooman M, *Bioinformatics*
  34:3659, doi:10.1093/bioinformatics/bty348) is about Ssym/antisymmetry bias, **not AF**. I found **no** Rooman/Pucci
  paper specifically on PoPMuSiC with AF structures **[U]**.
- **Iqbal et al. 2022** and **"Schmidt et al."**: could not identify a matching AF-ΔΔG paper **[U]**.
- **MAVISp** is Arnaudi M … Papaleo E (not Tiemann); see §4. Tiemann et al. 2023 (Lindorff-Larsen lab, membrane
  proteins with AF2 + Rosetta) not checked **[U]**.
- **ThermoMPNN** (Dieckhaus et al. 2024 *PNAS*, doi:10.1073/pnas.2314853121 [article number U]): trained on Megascale (AF
  models, §3) and tested on PDB structures (FireProt, S669, Ssym); i.e. a model trained on AF geometry transfers to
  crystal inputs. No explicit AF-vs-PDB ablation in the paper **[V]**. Domainome (Beltran 2025) found ThermoMPNN the best
  stability predictor on AF-derived constructs (median ρ 0.50; 0.57 excluding zinc fingers) **[V]**.
- **AlphaMissense** (Cheng J et al. *Science* 381:eadg7492, 2023, doi:10.1126/science.adg7492): pathogenicity, not ΔΔG;
  learns from AF-like structure internally; not usable in PhysDDG (DL) but a benchmark comparator. **[S]**
- **IFUM** (Lee H … Park H. *Nat Commun* 2026, doi:10.1038/s41467-026-68637-4, PMC12923840) **[V]**: DL absolute ΔG; on
  57 natural proteins PCC 0.53 overall → 0.66 for 45 with pLDDT > 80 → 0.97 for 10 with pLDDT > 90; Rosetta
  FastRelaxed AF3 models used for engineering sets; replacing them with ESMFold models significantly reduced Tm
  correlations; recommends AF3 when ESMFold pLDDT < 80. Retraining on AF3 inputs did not help on Megascale (ESMFold is
  already accurate for ≤ 80-aa domains).
- **Rossi et al. 2025** (*Protein Sci* 34:e70134, doi:10.1002/pro.70134, PMC12023412) **[V]**: ESM-IF1 and Pythia perform
  better on AF models than on x-ray structures, attributed to AF-heavy training sets (> 90 % AF). Physics-free models
  can acquire an "AF-geometry" bias.

---

## 2. What is wrong with AF models (structure accuracy facts)

| Property | Evidence | Source |
|---|---|---|
| Backbone accuracy vs pLDDT | median Cα error 0.6 Å (pLDDT > 90), 1.1 Å (80–90), 1.5 Å (70–80); median global Cα RMSD 1.0 Å → 0.4 Å after removing a smooth distortion field (215 structures) | Terwilliger et al. 2024 *Nat Methods*, doi:10.1038/s41592-023-02087-4 (PMC10776388) **[V]** |
| Side chains, confident regions | 20 % of side chains in pLDDT ≥ 70 regions differ from the deposited model by ≥ 1.5 Å RMSD; 7 % clearly incompatible with density; vs 6 % / 2 % between two crystal forms of the same protein (102 maps) | Terwilliger 2024 **[V]** |
| χ1 accuracy vs pLDDT | pLDDT > 90: χ1 within 40° **80 %** of the time (recent PDB, < 2.5 Å, B < 30); pLDDT > 70 ≈ correct backbone | Tunyasuvunakool et al. 2021 *Nature* 596:590, doi:10.1038/s41586-021-03828-1, Ext Data Fig 3 **[V]** |
| AF side chains vs repacking | CASP14/15: AF2/AF3 native side chains beat SCWRL4, FASPR, Rosetta packer, DLPacker, AttnPacker, PIPPack, FlowPacker, DiffPack repacked on the AF backbone (e.g. CASP15 AF2 side-chain RMSD 0.90 Å, χ1 MAE 28.1°); repacking changes are null or worse; pLDDT-weighted consensus repacking gives +1.5 % rotamer recovery vs AF2 only | Vangaru & Bhattacharya 2025 *Brief Bioinform* 26:bbaf297, doi:10.1093/bib/bbaf297 (PMC12192453) **[V]** |
| Stereochemistry / clashes | AF2 models of 218 deafness genes: MolProbity clashscore 20.75, MolProbity score 2.86 → 0.11 / 0.97 after FFX many-body rotamer optimisation + minimisation (relax status of inputs not stated) | Gogal et al. 2024 *J Chem Phys* (Force Field X), doi:10.1063/5.0214652 (PMC11223778) **[V]** |
| Low-pLDDT modes | below pLDDT 70: "barbed wire" (non-predictive, no packing, C–N–CA angle outliers near −4σ, cis/twisted peptides), "pseudostructure" (isolated SS-like, little packing, enriched in signal peptides), "near-predictive" (packed, protein-like; can be nearly correct; associated with conditional folding). Classified by packing contacts (Probe contacts per heavy atom in i±2 window) × validation-outlier density; Phenix tool | Williams CJ, Chen VB, Richardson DC, Richardson JS. bioRxiv 10.1101/2025.06.06.658382 (2025) **[V]** |
| AF3 specifics | diffusion model, atom-level pLDDT; hallucination of spurious order in disordered regions (mitigated by AF2 cross-distillation); chirality violations 4.4 % on PoseBusters; can model ligands/ions/nucleic acids | Abramson et al. 2024 *Nature* 630:493, doi:10.1038/s41586-024-07487-w (PMC11168924) **[V]** |
| Relaxation in AF pipeline | AF2 runs constrained Amber99sb relaxation (OpenMM) on the best model by default ("to improve local geometry"); AFDB proteome models went through this relax step; ColabFold offers optional `relax_amber` | Jumper 2021 Methods; AF README; Tunyasuvunakool 2021 Methods **[V]**. AF2 paper supplement states relaxation does not improve accuracy metrics but removes violations **[U — supplement not re-read]** |
| Large proteins | AFDB splits > 2,700 aa proteins into 1,400-aa overlapping fragments (human proteome); fragments are worse predictions | SNPeffect 5.0 (Janssen 2023); Williams 2025 **[V]** |
| Over-packing / over-idealisation | No quantitative packing-density or cavity study found. Indirect: one AF2 binding site was ~20 % smaller in volume than the crystal pocket (268 → 213 Å³, ligand-discovery paper) **[U, secondary summary]**; AF ignores ligands/waters, so apo pockets can be collapsed | — |

Side-chain *distributions*: AF2χ (Cagiada M, Thomasen FE, Ovchinnikov S, Deane CM, Lindorff-Larsen K; bioRxiv
10.1101/2025.04.16.649219, 2025) **[V]** extracts χ1/χ2 distributions from inner AF2 structure-module layers + Top8000
prior; inner-layer χ1 correlates better with NMR-restrained ensemble averages (ρ 0.83) than the final AF2 χ1 (0.69).
DL, so not usable inside PhysDDG, but it supports the idea that single AF rotamers understate heterogeneity —
a reason to Boltzmann-average over rotamers rather than trust one conformation.

---

## 3. Megascale and other benchmark structures are AF models

Tsuboyama K, … Rocklin GJ. "Mega-scale experimental analysis of protein folding stability in biology and design."
*Nature* 620:434 (2023). doi:10.1038/s41586-023-06328-6 (PMC10412457). **[V]**
- Natural domains: monomeric PDB entries 30–100 aa, then "predicted the structures … using AlphaFold (even though the PDB
  structures were known), and used the AlphaFold models to trim amino acids from the N- and C-termini that had a low
  number of contacts"; final ≤ 72 aa.
- "For most of the structural analysis, we used structural models predicted by AlphaFold … default parameters … model
  with the highest pLDDT. For designed sequences, we skipped a step for generating multiple sequence alignment"
  (single-sequence AF for Rosetta/trRosetta-hallucinated designs).
- Burial = Rosetta-style Cβ cone count (9 Å) on AF models.
- On disk: `/workspace/data/megascale_pdbs/*.pdb` are named by PDB ID but carry pLDDT in the B-factor column (values
  67–92 in the first file) and explicit hydrogens → consistent with (post-processed) AF models. **[V, local check]**
- Consequences: (i) PhysDDG parameters fitted on Megascale are fitted on AF geometry; (ii) Megascale cannot test an
  xtal→AF transfer; use FireProt/S669 proteins with both PDB and AF models (paired evaluation, RESEARCH_TREE 5.1);
  (iii) designed domains folded in single-sequence mode are idealised — over-idealisation effects, if any, would show up
  as a natural-vs-designed residual difference.

Domainome: Beltran A, Jiang X, Shen Y, Lehner B. "Site-saturation mutagenesis of 500 human protein domains." *Nature*
(2025). doi:10.1038/s41586-024-08370-4 (PMC11754108). **[V]** aPCA abundance for > 500,000 variants / 522 domains;
domains filtered as folded if > 10 % core residues and pLDDT > 50; best stability predictor ThermoMPNN (median ρ 0.50;
0.57 without zinc fingers, also 0.57 on domains non-homologous to Megascale); ESM1v and EVE 0.48. **All tested stability
predictors perform poorly on small zinc-finger domains that need metal binding** (Ext Data Fig 2a; 141 ZnF vs 348 other
domains). AF3 used only for case-study models.

---

## 4. Practices of large-scale "FoldX on AlphaFold" resources (what others did)

| Resource | Structure handling | Notes |
|---|---|---|
| **Jänes et al. 2024** (Beltrao lab; ProtVar) — "Predicted mechanistic impacts of human protein missense variants", bioRxiv 10.1101/2024.05.29.596373 **[V]** | AFDB human models; **remove regions with mean pLDDT < 50 over a 10-aa window**; FoldX BuildModel for all mutations at all positions; results reported for pLDDT > 70 (208 M scores); FGFR1 rebuilt from FGFR2 xtal domain orientation (AF had wrong inter-domain packing without the TM segment) | Data: https://ftp.ebi.ac.uk/pub/databases/ProtVar/ ; code https://github.com/jurgjn/af2genomics . The local `/af_data/af2.trim_bf*` + `af2.trim_bf.repairpdb` tree (and very likely `/foldx_native_clean`) follows this pipeline: pLDDT-window trimming ("trim_bf") then FoldX RepairPDB. Thresholds: FoldX > 2 = destabilising. |
| **MAVISp** — Arnaudi M … Papaleo E. *Protein Sci* (2026) doi:10.1002/pro.70548 (PMC13055194) **[V]** | AFDB models trimmed to remove **pLDDT < 70 termini and long disordered linkers**; long loops optionally replaced by short MODELLER stretches; cofactors re-added via AlphaFill/MODELLER in targeted studies; transmembrane regions excluded; Procheck QC with laxer thresholds for AF (≥ 70 % core Rama, ≥ 90 % core+allowed, ≤ 5 % disallowed) | FoldX5 BuildModel ×5 runs (MutateX) + Rosetta cartddg2020 (ref2015) or RaSP; since 2024 RaSP–FoldX consensus default; class thresholds ±3 kcal/mol (FoldX5.1 study). |
| **SNPeffect 5.0** — Janssen K … Rousseau F, Schymkowitz J. *BMC Bioinformatics* (2023), doi:10.1186/s12859-023-05407-9 (PMC10355034) **[V]** | AFDB human models; FoldX (v3.0 per text **[V]**); "recommended only to consider regions with pLDDT > 70"; "performing energy minimization of the side chains of each PDB structure is highly recommended before modeling with FoldX"; CATH domain context | Pipeline only (https://github.com/vibbits/snpeffect), not a precomputed catalogue. |
| **RaSP proteome** (Blaabjerg 2023) | AFDB v2, Reduce + PDBFixer, no trimming | ~300 M predictions, caveat on IDRs/membrane. |
| **Gerasimavicius 2022/2023** | PDB first (biological assemblies), AF2 monomers only to fill gaps | complexes improve signal. |

Consensus thresholds in the field: **pLDDT > 70 to report, trim terminal/linker segments < 70 (MAVISp) or windows < 50
(Jänes), side-chain minimisation/RepairPDB before scanning.** None of these choices was benchmarked against
experimental ΔΔG in the cited papers — they are conventions.

Runtime anchor (FoldX on AF): Dunham, Beltrao, AlQuraishi 2023 *Genome Biol* (Sequence UNET), doi:10.1186/s13059-023-02948-3 **[V]**: FoldX took
232 h for 24,187 Spike variants and 64.5 CPU-days for 161,825 variants in 30 proteins, i.e. **≈ 34 CPU-s per variant**.

---

## 5. Specific issues and what the evidence says

**pLDDT bands.** > 90: backbone ~0.6 Å, χ1 ~80 % correct → treat as crystal-quality. 70–90: backbone OK (1.1–1.5 Å
median), side chains often wrong (≥ 20 % differ) → environment counts OK, specific side-chain contacts/H-bonds less
reliable. 50–70: backbone unreliable; may be near-predictive if packed (Williams 2025). < 50: mostly disorder/barbed
wire; ΔΔG small anyway (Akdel Fig 3c). Quantitative ΔΔG-vs-pLDDT curves exist only coarsely (RaSP ≥ 90 / < 90; Akdel
< 50 / ≥ 50; IFUM protein-level > 80, > 90).

**Relaxation.** Direct evidence on FoldX RepairPDB / Rosetta relax / Amber relax *of AF models* vs experimental ΔΔG:
none found. Indirect: (i) Valanciute 2023: protocols that minimise locally (cartesian_ddg) are robust to model error,
FoldX is not; (ii) Sora 2023: cartddg2020 rescued ref2015 on an AF model; (iii) IFUM used Rosetta FastRelax of AF3
models; (iv) MAVISp/Jänes/SNPeffect all RepairPDB or minimise; (v) FFX shows large clash reductions; (vi) AF2's own
Amber relax is mild (restrained, removes clashes only). Also note that most FoldX/Rosetta papers relax xtal structures
too, so "relax" is not AF-specific. **Open question to test ourselves.**

**Trimming low-pLDDT residues / termini.** Universal practice (Tsuboyama by contacts; MAVISp < 70 termini; Jänes
windowed < 50). Rationale: disordered segments wrapped over the domain create artificial burial and spurious contacts
(barbed wire has few contacts, but pseudostructure/near-predictive can pack against the domain). For a contact/burial-
based energy (FoldX-like occupancy scaling) spurious contacts directly inflate ΔΔG.

**Domain cropping / PAE.** Croll `pae_to_domains` (used in Akdel 2022), AF PAE JSON in AFDB. Domainome assayed Pfam
constructs, so the evaluated structure should be the cropped construct (project memory: cropping removed the shift).
Inter-domain orientations can be wrong without context (FGFR1 example, Jänes 2024).

**Oligomers, ligands, cofactors, metals.** Complexes: Gerasimavicius 2022 (AUC +0.025), 2023 (FoldX/Rosetta improve on
DMS). Metals: Domainome zinc fingers — all predictors fail. AlphaFill (Hekkelman et al. 2023 *Nat Methods*
20:205, doi:10.1038/s41592-022-01685-y **[S]**) transplants ligands/metals into AF models; AF3 predicts them directly.

**AF mutant structures.** Don't: Buel & Walters 2022; Pak 2023; Stability Oracle mutant-AF −2–4 %; McBride shows the
signal exists only after averaging many AF runs (expensive; DL anyway).

**AF3 vs AF2.** Side-chain accuracy similar (Vangaru 2025: CASP15 AF2 0.90 Å vs AF3 0.95 Å all-residue RMSD; CASP14
1.07 vs 1.04). Hallucinated order in IDRs is a new risk for burial-based terms (check packing, not only pLDDT). No ΔΔG
benchmark found.

**ESMFold.** Lower accuracy than AF2 on larger proteins; fine for small domains (IFUM 2026). Nothing quantitative for
physics ΔΔG found.

---

## 6. Recommended PhysDDG "AF mode" (evidence-weighted proposal; each item testable on paired xtal/AF sets)

1. **Input**: prefer the experimental structure if ≥ 40–50 % identity template covers the site (Pak & Ivankov;
   Valanciute; Lihan); else AF2/AF3 model. Record `struct_source` and AFDB version.
2. **Confidence-aware trimming** (before any energy/burial calculation):
   - drop terminal segments and linkers with pLDDT < 70 (MAVISp rule), and any internal window with mean pLDDT < 50
     over 10 aa (Jänes rule) unless the segment is packed (≥ protein-typical contact density) — i.e. keep
     "near-predictive", drop "barbed wire"/"pseudostructure" (Williams 2025 criterion: contacts per heavy atom in i±2
     window + geometry outliers). Implement the packing test ourselves (no DL needed).
   - For Domainome-type data, crop to the assayed construct; otherwise split multi-domain models at PAE domain
     boundaries (community clustering of PAE, Croll) and score each mutation in its own domain plus any domain with
     confident inter-domain PAE (< ~10 Å) contacting it.
3. **Do not repack the whole protein.** Keep AF side chains for the WT environment (Vangaru 2025); repack only the
   mutated residue and a 6–8 Å shell (Rosetta cartesian_ddg uses 6 Å), for WT and mutant identically.
4. **Light, restrained relaxation** once per WT model: remove clashes and fix bad rotamers with a restrained
   minimisation (heavy-atom harmonic restraints, strength scaled by pLDDT — the ISOLDE idea in Akdel 2022 Methods:
   weaker restraints / disabled restraints for pLDDT < 50). Then local Cartesian minimisation around the site for
   both WT and mutant (cartesian_ddg-style), which is what made Rosetta robust to model error.
5. **pLDDT as a weight, not a feature that leaks**: shrink predicted ΔΔG toward the unfolded/neutral baseline as site
   pLDDT drops, e.g. ΔΔG_AF = w(pLDDT)·ΔΔG_phys + (1−w)·c, with w a logistic in pLDDT (2 parameters, fitted on paired
   sets with grouped CV). Report metrics stratified by pLDDT bands (> 90, 70–90, 50–70, < 50) and on pLDDT > 70 only,
   since the field reports there.
6. **Burial from contacts, robust to side-chain error**: in 70–90 bands use backbone/Cβ-based burial (Cβ count / half-
   sphere exposure) in preference to full-atom SASA, because AF backbones are much more reliable than its side chains.
7. **Ensemble averaging**: average ΔΔG over the 5 AF2 models (or AF3 seeds) where available (Peccati 2023; McBride &
   Tlusty) — cheap on GPU, and gives a per-mutation uncertainty.
8. **Missing context flags**: flag Cys/His clusters consistent with metal sites (and zinc-finger Pfams), known
   cofactor sites (AlphaFill), interface residues of known complexes; either add the ion/ligand (AF3/AlphaFill) or
   report separately. Membrane segments: exclude (MAVISp practice).
9. **Never compute mutant structures with AF**; never use ΔpLDDT as a ΔΔG signal.
10. **Validation design**: paired FireProt/S669 proteins with PDB and AF models (same mutations), metrics per pLDDT
    band, plus Domainome (constructs) for transfer; compare with FoldX native on the same trimmed AF inputs.

Open questions worth an experiment: (a) does Amber/our-forcefield restrained relax of AF inputs improve ΔΔG vs raw AF
and vs RepairPDB? (b) AF3 vs AF2 inputs; (c) is there a systematic over-packing signature (e.g. lower cavity volume,
higher contact density in AF vs paired xtal) that biases large→small mutations?
