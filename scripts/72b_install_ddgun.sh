#!/bin/bash
# Install DDGun (Montanucci et al. 2019) under data/raw/tools/ddgun with its own venv (.venv_ddgun, python 3.10 +
# biopython 1.79 because ali2prof.py needs the legacy Bio.Blast.NCBIStandalone) and hh-suite (hhblits) built from source.
# The UniClust30 2018_08 database is fetched by 72a_download_uniclust30.sh and symlinked into ddgun/data.
set -e
R=/workspace/stability_hackathon
T=$R/data/raw/tools
cd $T
[ -d ddgun ] || git clone --depth 1 https://github.com/biofold/ddgun
cd $R
[ -d .venv_ddgun ] || uv venv --python 3.10 .venv_ddgun
uv pip install --python .venv_ddgun/bin/python "numpy<2" biopython==1.79 cmake pandas
cd $T/ddgun/utils
[ -d hh-suite ] || git clone --depth 1 --branch v3.3.0 https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
PATH=$R/.venv_ddgun/bin:$PATH cmake -DCMAKE_CXX_FLAGS="-include cstdint" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=.. .. > cmake.log 2>&1
make -j 32 > make.log 2>&1 && make install > install.log 2>&1
$T/ddgun/utils/hh-suite/bin/hhblits -h 2>&1 | head -3
