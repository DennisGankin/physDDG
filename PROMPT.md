# Original task prompt (verbatim, 2026-10-26)

/goal You are the PI of a computational biophysics project. You manage a team of agents (Opus 5.5 or Sonnet 5.5 as fitting). You delegate, review, integrate and decide; you keep the research record. 

Inside /workspace/stability_hackathon Build PhysDDG a structure-based, very fast physics-inspired predictor of the effect of missense mutations on protein folding stability (ΔΔG). It must beat related state-of-the art models (like FoldX) clearly, and ideally close the gap to deep-learning predictors. The model should generalise across assays. It is not enough to fit one high-throughput assay; the physics should transfer. One key application will be to apply this on AlphaFold derived structures so the method could have specific considerations and a specific mode that improves predictions made on these types of structures. Consider also optimizations that will make the predictor faster, including GPU optimizations. It should not use any deep learning. Instead it could use things like force-field-style energy terms, rotamer packing, minimisation relaxation, and a modest number of interpretable parameters fitted to data. Use parameters you think are useful and interpretable. You will need to collect suitable datasets of carefully measured metrics of impact on stability and perhaps larger-scale data that may be noisier. We already have some data on disk. Get more data if you think it's useful. If you run into access issues, flag them, so we can fix them together. Design through repeated and independent critique: Search the current literature and available datasets and methods, and write LITERATURE.md.
  
You have access to 4 GPUs and many CPUs, when you run scripts, make sure to optimize for speed and parallelize when needed. 

Available inputs (read-only):

- data/
    fireprot_pdbs/      PDB structures for FireProtDB entries
    megascale_pdbs/     PDB structures for the Megascale (Tsuboyama) dataset
    yeast_af/pdbs/      AlphaFold structures for yeast proteins
    runtime_tables/     4_fireprotDB_bestpH.csv, megascale_runtime.csv (ddG tables)
    downloads/          original zips (AlphaFold_model_PDBs.zip, fireprot_upload.zip)
- Domainome/            Extended_data_Table_2_fitness_scores_normalized_domainranks.txt (Domainome DMS fitness)
- foldx_native_clean/         foldx ddG predictions human proteome
You also have access to The Alphafold database here: /af_data and here: /alphafold
And to PDB here:  /rsyncPDB

Workspace: write all new code, results and notes under stability_hackathon/.

Only work inside stability_hackathon. Do not touch the rest. You can reuse data that we have. If you download new data, that is not yet available, put it into stability_hackathon/data

Save the prompt to a file (PROMPT.md) and keep track of the research in a research tree (RESEARCH_TREE.md), keeping a Status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding. 

Your goal is to build the best possible predictor and you should keep trying until you genuinely run out of promising ideas. Use git to track progress. Keep all work reproducible, create scripts for all steps, so that they can always be rerun. Be critical. Flag and explain all important decisions clearly. Do the best you can. I believe in you!

## Path notes (resolved on this machine)
- Read-only inputs: /workspace/data, /workspace/Domainome, /foldx_native_clean (at filesystem root, not under /workspace)
- AlphaFold DB: /af_data (v4 human proteome tar + af2/ tree), /alphafold/v6 (49 proteome tars, v6)
- PDB mirror: /rsyncPDB/data/structures/{divided,all}
