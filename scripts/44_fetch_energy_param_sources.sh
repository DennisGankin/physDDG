#!/usr/bin/env bash
# 44: download the public copies of the literature parameter files used to verify physddg/params.py
#   param19.inp   CHARMM19 united-atom parameters (copy shipped in the VMD trunctraj plugin, CVS mirror)
#   eef1parm.dat  EEF1 Table I (Lazaridis & Karplus 1999) for CHARMM19 types, as shipped with OSPREY3
#   EEFSolv.cpp   PLUMED EEF1 implementation (Lazaridis-Karplus dG values for CHARMM22/36 types)
# tests/test_energy.py::test_params_match_sources compares params.py with these files (skips if absent).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/raw/param_sources
mkdir -p "$OUT"
curl -sSL -m 60 -o "$OUT/param19.inp" \
  https://raw.githubusercontent.com/thatchristoph/vmd-cvs-github/master/plugins/trunctraj/toppar/param19.inp
curl -sSL -m 60 -o "$OUT/eef1parm.dat" \
  https://raw.githubusercontent.com/donaldlab/OSPREY3/main/src/main/resources/config/eef1parm.dat
curl -sSL -m 60 -o "$OUT/EEFSolv.cpp" \
  https://raw.githubusercontent.com/plumed/plumed2/master/src/colvar/EEFSolv.cpp
sha256sum "$OUT"/* | tee "$OUT/SHA256SUMS"
