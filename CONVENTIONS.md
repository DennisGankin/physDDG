# Project conventions (all agents must follow)

## Layout
- `physddg/`        the predictor package (importable: `PYTHONPATH=. .venv/bin/python ...`)
- `scripts/`        numbered, rerunnable pipeline scripts (`NN_name.py` / `.sh`); every result must come from a script
- `data/raw/`       downloaded originals (git-ignored; every download done by a script in `scripts/` so it can be redone)
- `data/processed/` canonical benchmark tables (small ones committed, large ones git-ignored + regenerable)
- `data/structures/` derived structures (AF crops, cleaned PDBs); git-ignored, regenerable
- `results/`        metrics tables, figures, prediction CSVs (one subfolder per experiment)
- `notes/`          agent reports / experiment write-ups (markdown, short and factual)
- Read-only inputs, never modified: `/workspace/data`, `/workspace/Domainome`, `/foldx_native_clean`, `/af_data`, `/alphafold`, `/rsyncPDB`
- Only write inside `/workspace/stability_hackathon`.

## Environment
- Python: `/workspace/stability_hackathon/.venv/bin/python` (uv venv; add packages with `uv pip install --python .venv/bin/python <pkg>` and record them in `requirements.txt`).
- 64 CPUs, 4x RTX 4090 (24 GB). Parallelise (multiprocessing / joblib) and use GPUs (`CUDA_VISIBLE_DEVICES`) for heavy work. Do not hog all 4 GPUs without need; other agents share them.
- PyTorch may be used **only as a GPU array library** (tensor maths, batched energy evaluation, autograd for minimisation / fitting a handful of weights). No neural networks, no learned embeddings, no pretrained DL models anywhere in the predictor.

## Sign and unit convention (decision D1)
- `ddg` in kcal/mol, **positive = destabilising** (ΔΔG = ΔG_unfold(WT) − ΔG_unfold(mut)), same as FoldX.
- Datasets that use the opposite sign (Tsuboyama/Megascale `ddG_ML`, ProTherm-style tables) are flipped at ingestion; record the original column and the flip in the processing script.

## Canonical variant table schema (`data/processed/<dataset>.csv` or `.parquet`)
Required columns:
- `dataset`        e.g. megascale, fireprot, s669, ssym, domainome
- `protein_id`     stable identifier of the measured protein/construct
- `structure`      path (relative to project root or absolute for read-only inputs) of the structure file used
- `chain`          chain ID in that structure
- `pdb_resnum`     residue number in the structure file as string, including insertion code (e.g. "52A")
- `wt`, `mut`      one-letter amino acids (single missense only in the main tables)
- `ddg`            kcal/mol, positive = destabilising (NaN if only a proxy exists, e.g. fitness)
- `split`          train / val / test / external (see split notes per dataset)
- `cluster`        homology cluster id (for grouped CV / leakage control)
Optional: `uniprot`, `seq_pos` (1-based in the protein sequence), `score` (non-ddG proxy such as DMS fitness, with its own sign documented), `pH`, `temp`, `method`, `source`, `struct_source` (xtal/nmr/af/design), `plddt`.
Each dataset gets a short `data/processed/<dataset>.md` card: source/URL, version, filters, sign handling, counts, known issues.

## Evaluation
- Primary metrics: Spearman (pooled and per-protein median), Pearson, RMSE after no calibration, AUROC for ddg > 1 kcal/mol; for Ssym-type data the antisymmetry (Pearson of direct vs −reverse, mean bias).
- Uncertainty: bootstrap over proteins (or homology clusters), paired for model comparisons.
- Leakage: anything fitted must be evaluated on proteins with no homolog (>=25-30% id / same family) in the fitting set; report both all and strict subsets.

## Git
- Commit small, coherent units with clear messages. Do not commit large data, caches or the venv.
