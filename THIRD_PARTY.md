# Third-party data, parameters and tools

The MIT licence of this repository covers only our own work. The items below keep their own terms.
Licences marked "checked" were read from the Zenodo record or the GitHub licence field on 2026-10-27. Items
marked **to confirm** have no clear licence statement in the sources we could read; the repository owner should
confirm them before publication.

## Datasets (experimental values in `data/processed/` and `results/`)

| dataset | source | used for | licence |
|---|---|---|---|
| Megascale (Tsuboyama et al. 2023, *Nature* 620:434) | Zenodo 10.5281/zenodo.7992926 (processed K50 dG datasets, AlphaFold model PDBs) | training (train split), Megascale-test | CC BY 4.0 (checked) |
| ThermoMPNN dataset splits (Dieckhaus et al. 2024, *PNAS*) | github.com/Kuhlman-Lab/ThermoMPNN `dataset_splits/` | Megascale/FireProt train/val/test splits, clusters | MIT (checked) |
| FireProtDB (Stourac et al. 2021, *NAR* 49:D319), as curated by the ThermoMPNN authors | loschmidt.chemi.muni.cz/fireprotdb; ThermoMPNN data release | FireProt-train/val/test | **to confirm** (FireProtDB states it is a free non-profit service; no licence text found) |
| S669 (Pancotti et al. 2022, *Brief Bioinform* 23:bbab555) | copy in github.com/chaconlab/korpm | external test (strict subset) | **to confirm** (paper supplement; KORPM repository licence NPOSL-3.0) |
| S461 (Hernández et al. 2023, *Bioinformatics* 39:btad011) | github.com/chaconlab/korpm | external test (strict subset) | **to confirm** (repository licence NPOSL-3.0 per its LICENSE file; GitHub reports NOASSERTION) |
| Ssym (Pucci et al. 2018, *Bioinformatics* 34:3659) | github.com/chaconlab/korpm; ThermoNet copy (GPL-3.0) for cross-checks | antisymmetry diagnostic | **to confirm** |
| S2648 (Dehouck et al. 2009, *Bioinformatics* 25:2537) | github.com/mitiau/PROSTATA | data/processed only (not in the main tables) | repository Apache-2.0 (checked); original data **to confirm** |
| Domainome 1.0 (Beltran et al. 2024/2025) | Zenodo 10.5281/zenodo.13629491 and 10.5281/zenodo.14356805 | held-out DMS validation | CC BY 4.0 (checked, both records) |
| ProteinGym v1.3 abundance DMS (Notin et al. 2023) | marks.hms.harvard.edu/proteingym; github.com/OATML-Markslab/ProteinGym | held-out abundance validation (7 assays) | MIT (checked, repository); individual assays come from their original papers |

## Structures

| item | source | licence |
|---|---|---|
| PDB entries (crystal/NMR benchmark structures, PISCES set for the rotamer library and statistics) | wwPDB | CC0 1.0 (wwPDB policy) |
| PISCES culled list (Wang & Dunbrack 2003, *Bioinformatics* 19:1589) | dunbrack.fccc.edu/pisces | free to use; please cite |
| AlphaFold DB models (Varadi et al. 2024, *NAR*) | alphafold.ebi.ac.uk | CC BY 4.0 |
| `examples/1PSE.pdb` | Megascale AlphaFold model set (Zenodo 7992926) | CC BY 4.0 |

`physddg/data/rotlib_bbdep.npz`, `pdbstats.npz` and `ncap_pref.json` are our own statistics, derived from PISCES-culled
PDB chains by `scripts/40_build_rotlib.py` and `scripts/53_ncap_stats.py`. They contain no third-party library
(we did not use the Dunbrack 2010 rotamer library, which requires a separate licence).

## Force-field parameters

`physddg/params.py` contains numerical constants from published work: CHARMM19 united-atom van der Waals
parameters (Brooks et al. 1983; Neria, Fischer & Karplus 1996) and EEF1 solvation parameters (Lazaridis & Karplus
1999, *Proteins* 35:133, Table I). They were checked against public copies of the parameter files (VMD plugin
copy of `param19.inp`, OSPREY3 `eef1parm.dat`, PLUMED `EEFSolv.cpp`), which `scripts/44_fetch_energy_param_sources.sh`
downloads; those files are not redistributed here. CHARMM itself is not used or distributed.

## Competitor tools and their predictions

| tool | licence | in this repository |
|---|---|---|
| FoldX 5.1 (Schymkowitz et al. 2005; Delgado et al. 2019) | academic licence, no redistribution of the binary | **binary and raw outputs not included.** Per-variant FoldX ΔΔG values are **not** included in this repository. Aggregate FoldX metrics (scores and confidence intervals) appear in `results/` and `notes/`. |
| KORPM (Hernández et al. 2023) | NPOSL-3.0 (repository LICENSE) | code not included; our KORPM predictions are in `results/competitors/` |
| DDGun / DDGun3D (Montanucci et al. 2019/2022) | GPL-2.0-or-later (README) | code not included; our predictions are in `results/competitors/` |
| tmol 0.2.0 (UW-IPD) | Apache-2.0 | not included (installed separately for the tmol comparison); our refitted tmol predictions are in `results/tmol/` |
| MMseqs2 | MIT | downloaded by `scripts/10_download_ht.py`, not included |
| HH-suite / UniClust30 (optional evolution mode) | GPL-3.0 / CC BY-SA 4.0 | downloaded by scripts `72a`/`72b`, not included |

Predictions made by running third-party tools are our outputs; we believe the tool licences do not restrict them,
but the owner should confirm this for FoldX in particular.

## Python dependencies

numpy, pandas, pyarrow, scipy, scikit-learn (BSD), biopython (Biopython licence), freesasa (MIT), gemmi (MPL-2.0),
PyTorch (BSD-3-Clause), pytest (MIT), pydssp (MIT, optional). They are installed from PyPI and not distributed here.
