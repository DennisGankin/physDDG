# Input data

You do **not** need any of this to run the predictor: the fitted models (`results/ladder_physics/models/`) and the
PDB-derived tables (`physddg/data/`) are in the repository. You need it to refit the weights or to rebuild the
benchmark tables.

Total download size is about 120 GB, dominated by the AlphaFold DB proteome archives and the local PDB mirror.

## Layout and paths

| location | content | in git? |
|---|---|---|
| `data/raw/` | everything the scripts download (Zenodo, GitHub, PISCES, ProteinGym, tools) | no |
| `data/processed/` | the curated variant tables and their dataset cards (`*.md`) | small tables yes, large ones regenerated |
| `data/structures/` | cleaned crystal chains and AlphaFold construct crops produced by the scripts | no |
| `data/features/` | cached features (about 11 GB) | no |

**The input locations are written into the scripts for the machine where the project ran.** Inputs that the
project did not download are read from these read-only paths: `/rsyncPDB` (local wwPDB rsync mirror),
`/af_data` (AlphaFold DB human v4), `/alphafold` (AlphaFold DB v6 proteome tars), `/workspace/data` (Megascale and
FireProt structures and runtime tables, yeast AlphaFold models), `/workspace/Domainome` (Domainome supplementary
table) and `/foldx_native_clean` (FoldX results; FoldX is not distributed). To rebuild on another machine, change
these constants in `scripts/ht_common.py`, `scripts/afdb_io.py`, `scripts/pisces_io.py`, `scripts/11_build_megascale.py`,
`scripts/12_build_fireprot.py`, `scripts/13_build_external.py`, `scripts/21`–`25` and `scripts/59_speed_benchmark.py`
(search for the paths above). There is no environment-variable configuration.

## Datasets, and the script that fetches or builds each one

| dataset | where it comes from | script |
|---|---|---|
| **Megascale** (Tsuboyama et al. 2023) ΔΔG and AlphaFold model structures | Zenodo [10.5281/zenodo.7992926](https://doi.org/10.5281/zenodo.7992926), file `Processed_K50_dG_datasets.zip` (1.0 GB); the structures come from the same record (`AlphaFold_model_PDBs.zip`) and are expected in `/workspace/data/megascale_pdbs/` | `10_download_ht.py`, then `11_build_megascale.py` |
| **ThermoMPNN splits** (train/validation/test, clusters, homologue lists) | `github.com/Kuhlman-Lab/ThermoMPNN`, folder `dataset_splits/` | `10_download_ht.py` |
| **FireProtDB** ΔΔG, as curated by the ThermoMPNN authors | [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/); the curated export `4_fireprotDB_bestpH.csv` and the structures are expected in `/workspace/data/` | `12_build_fireprot.py` |
| **S669, S461, Ssym, S2648** | copies in `github.com/chaconlab/korpm` (S669, S461, Ssym), `github.com/mitiau/PROSTATA` (S2648), `github.com/gersteinlab/ThermoNet` (Ssym cross-check); structures from the local PDB mirror | `10_download_ht.py`, then `13_build_external.py` |
| **Domainome 1.0** (Beltran et al.) aPCA fitness | Zenodo [10.5281/zenodo.13629491](https://doi.org/10.5281/zenodo.13629491); supplementary tables from Zenodo record 14356805. The extended data table is expected in `/workspace/Domainome/` | `20_download_domainome_supp.sh`, then `22`, `23`, `26` |
| **ProteinGym** abundance DMS (7 assays) | `marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip` plus the reference file from `github.com/OATML-Markslab/ProteinGym` | `25_abundance_dms.py` |
| **PISCES** culled PDB list (for the rotamer library and the statistics) | `dunbrack.fccc.edu/pisces/download/`; the chains themselves come from the local PDB mirror | `40_build_rotlib.py` (through `pisces_io.py`) |
| **AlphaFold DB** models | `alphafold.ebi.ac.uk`; we used local copies of the human v4 proteome and the v6 proteome archives. Single models are downloaded on demand by `afdb_io.py` | `21_index_af_tars.py`, `22`, `24`, `25` |
| **wwPDB** entries | a local rsync mirror (`/rsyncPDB/data/structures/divided/mmCIF/...`) | `ht_common.py`, `pisces_io.py` |
| **MMseqs2** (homology clustering) | GitHub release, downloaded into `data/raw/tools/` | `10_download_ht.py`, used by `14_homology.py` |
| **Force-field parameter sources** (for verification only) | public copies of `param19.inp`, `eef1parm.dat`, `EEFSolv.cpp` | `44_fetch_energy_param_sources.sh` |

Competitor tools (FoldX 5.1, KORPM, DDGun3D, tmol) are not distributed and are not needed for the shipped models;
scripts `70`–`72b` install KORPM and DDGun, and FoldX must be obtained under its own licence. See
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Building the tables

```bash
PYTHONPATH=. python scripts/10_download_ht.py          # ~1 h, mostly the Zenodo download
PYTHONPATH=. python scripts/11_build_megascale.py
PYTHONPATH=. python scripts/12_build_fireprot.py
PYTHONPATH=. python scripts/13_build_external.py
PYTHONPATH=. python scripts/14_homology.py             # MMseqs2 clusters and the homology flags
PYTHONPATH=. python scripts/15_validate_tables.py      # schema and sign checks
bash scripts/20_download_domainome_supp.sh && PYTHONPATH=. python scripts/22_domainome_structures.py
PYTHONPATH=. python scripts/23_domainome_table.py
PYTHONPATH=. python scripts/25_abundance_dms.py
```

Every table follows the schema in [CONVENTIONS.md](../CONVENTIONS.md) and has a card
(`data/processed/<dataset>.md`) with its source, filters, sign handling, counts and known issues. The full order,
with run times, is in [README_DEV.md](../README_DEV.md#reproducing-results).
