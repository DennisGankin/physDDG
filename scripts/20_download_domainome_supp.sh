#!/usr/bin/env bash
# Download Domainome 1.0 supplementary tables (Beltran et al. 2025 Nature, Zenodo record 14356805, v5).
# Supplementary_Table_2 is identical in content to /workspace/Domainome/Extended_data_Table_2_* (used read-only);
# we fetch Table 1 (library design) and Table 4 (MoCHI homologue-averaged scaled ddG) only.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/raw/domainome
mkdir -p "$OUT"
BASE=https://zenodo.org/api/records/14356805/files
declare -A MD5=(
  [Supplementary_Table_1_library_design.txt]=9efcf1ae4f90e975f0c0719c79d136a6
  [Supplementary_Table_4_homolog_predictions.txt.zip]=82d4c7212579e59f973fe87748c63c92
)
for f in "${!MD5[@]}"; do
  if [ ! -s "$OUT/$f" ] || [ "$(md5sum "$OUT/$f" | cut -d' ' -f1)" != "${MD5[$f]}" ]; then
    curl -sSL --retry 3 -o "$OUT/$f" "$BASE/$f/content"
  fi
  echo "$(md5sum "$OUT/$f" | cut -d' ' -f1)  expected ${MD5[$f]}  $f"
done
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2], [n for n in zipfile.ZipFile(sys.argv[1]).namelist() if not n.startswith("__MACOSX")])" "$OUT/Supplementary_Table_4_homolog_predictions.txt.zip" "$OUT"
cd "$OUT"
ls -la
