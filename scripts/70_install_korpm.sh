#!/bin/bash
# Install KORPM (Martin-Hernandez et al. 2023) under data/raw/tools/korpm: clone, fetch LFS potential, compile.
set -e
T=/workspace/stability_hackathon/data/raw/tools
mkdir -p $T; cd $T
[ -d korpm ] || git clone --depth 1 https://github.com/chaconlab/korpm
cd korpm
# the potential files are git-LFS pointers (134 bytes); git-lfs is not installed, use the media endpoint
for f in pot/korp6Dv1.bin pot/dunbrack.bin; do
  if [ $(stat -c %s $f) -lt 1000 ]; then
    curl -sL -o $f https://media.githubusercontent.com/media/chaconlab/korpm/main/$f
  fi
done
ls -l pot
cd sbg && ./compile_korpm.sh > ../compile.log 2>&1 || { tail -30 ../compile.log; exit 1; }
ls -l bin
