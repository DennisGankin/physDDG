# Example input / expected outputs

`1PSE.pdb` — a 68-residue Megascale domain (Tsuboyama et al. 2023 structure set), used for the fresh-clone smoke test.

```bash
scripts/physddg predict --structure examples/1PSE.pdb --mutations "I2A,V8A,Y17A,G20A,F49A" --terms --out m.csv
scripts/physddg predict --structure examples/1PSE.pdb --scan-all --out scan.csv
```
Expected outputs (default model `final_stage1.json`, B6gjs1, stage 1): `1PSE_mutations_expected.csv`,
`1PSE_scan_expected.csv`. A fresh install should reproduce `ddg` to ~1e-5 kcal/mol (GPU float32 rounding).
