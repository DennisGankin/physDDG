# ssym (Pucci et al. 2018, Bioinformatics 34:3659), direct + reverse

**What it is.** 342 direct mutations on 15 WT proteins. For each one, an experimental structure of the mutant exists, so the reverse mutation is evaluated on the mutant structure. In total: 684 rows on 357 structures.

**Schema additions requested by the coordinator.**
- `direction`: direct or reverse.
- `pair_id`: shared by a direct mutation and its reverse.
- `pdb_id`: the structure actually used (the WT entry for direct rows, the mutant entry for reverse rows).
- `protein_id`: the WT protein (e.g. 1BNI_A) for both directions, so that grouping by protein and homology works.

**Values and sources.** Three copies of Ssym were compared:
- `korpm/KORPM_Ssym.txt` (chaconlab/korpm)
- `thermonet/s_sym.txt` (gersteinlab/ThermoNet)
- the ACDC-NN cross-validation files `prostata/ssym_TS_dir_*.mut` (mitiau/PROSTATA)

KORPM's copy differs from the other two on **24/342 pairs**, for example 1L63 L91A: −2.6 in KORPM against 3.9 in the others. ThermoNet and ACDC agree on all 342. **The ThermoNet/ACDC value is used.** KORPM's value is kept as `ddg_korpm` in our sign convention.

**Sign.** ThermoNet's copy is positive = destabilising (used as is). The KORPM and ACDC copies are negative = destabilising. Direct mean +1.02. Reverse rows have ddg = −ddg(direct) exactly, so the antisymmetry check gives r = 1.000.

**Structures and mapping.**
- Local mirror, model 1, the mutated chain, minimal cleaning (as for s669).
- 684/684 rows map exactly, with the WT letter matching in both the WT and mutant structures.
- 1RN1 uses chain C.
- 4.1% of rows are near a HETATM.

**Homology.**
- 0 proteins have a Megascale-train homologue.
- 11/15 proteins (96.5% of rows) have a FireProt-trainval homologue (lysozymes, barnase, RNase, ...).

**Caveat.** **Ssym is not independent of FireProt/ProTherm-derived training data.**
