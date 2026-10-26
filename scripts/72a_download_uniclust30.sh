#!/bin/bash
# Download UniClust30 2018_08 (hhsuite format, 26.7 GB) used by DDGun, and extract. Resumable.
set -e
D=/workspace/stability_hackathon/data/raw/tools/uniclust30_2018_08
mkdir -p $D; cd $D
curl -L -C - --retry 20 --retry-delay 5 -o uniclust30_2018_08_hhsuite.tar.gz \
  https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
tar -xzf uniclust30_2018_08_hhsuite.tar.gz
ls $D/uniclust30_2018_08 | head
touch $D/DONE
