#!/usr/bin/env python
"""Download raw inputs for the data-HT pipeline into data/raw/ (idempotent).

Sources (all public, no auth):
  thermompnn/   ThermoMPNN dataset_splits (mega_splits.pkl, fireprot_splits.pkl, clusters, homologue m8)
                https://github.com/Kuhlman-Lab/ThermoMPNN/tree/main/dataset_splits
  tsuboyama/    Tsuboyama et al. 2023 Zenodo 10.5281/zenodo.7992926 (v2_230420):
                Processed_K50_dG_datasets.zip (1.0 GB) -> extracted
  korpm/        S669.txt, S461.txt, KORPM_Ssym.txt from github.com/chaconlab/korpm (Martin-Hernandez et al. 2023,
                the authors of S461)
  prostata/     S2648.csv from github.com/mitiau/PROSTATA (Dehouck 2009 S2648 with PDB chain, pH, T)
  thermonet/    s_sym.txt (Ssym, Pucci 2018, as distributed with ThermoNet) for cross-checking
  tools/        MMseqs2 release 18-8cc5c static binary (linux avx2)

Usage: .venv/bin/python scripts/10_download_ht.py [--skip-tsuboyama]
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GH = "https://raw.githubusercontent.com"
FILES = {
    "thermompnn": [f"{GH}/Kuhlman-Lab/ThermoMPNN/main/dataset_splits/{f}" for f in [
        "mega_splits.pkl", "fireprot_splits.pkl", "megascale_clusters.tsv", "fireprot_clusters.tsv",
        "megascale_vs_fireprot_homologues.m8", "s669_vs_megascale_homologues.m8"]],
    "korpm": [f"{GH}/chaconlab/korpm/main/{f}" for f in [
        "S461.txt", "S461_results/S669.txt", "KORPM_Ssym.txt", "S461_results/S461.xlsx", "README.md"]],
    "prostata": [f"{GH}/mitiau/PROSTATA/main/DATA/S2648.csv"],
    "thermonet": [f"{GH}/gersteinlab/ThermoNet/master/data/datasets/s_sym.txt"],
}
TSUBO = ("https://zenodo.org/api/records/7992926/files/Processed_K50_dG_datasets.zip/content",
         "Processed_K50_dG_datasets.zip")
MMSEQS = "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-avx2.tar.gz"


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {url}")
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(dest), url], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tsuboyama", action="store_true")
    a = ap.parse_args()
    for sub, urls in FILES.items():
        print(sub)
        for u in urls:
            fetch(u, RAW / sub / Path(u).name)
    if not a.skip_tsuboyama:
        print("tsuboyama")
        z = RAW / "tsuboyama" / TSUBO[1]
        fetch(TSUBO[0], z)
        out = RAW / "tsuboyama" / "extracted"
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        print("  extracted:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())[:20])
    print("mmseqs2")
    tgz = RAW / "tools" / "mmseqs-linux-avx2.tar.gz"
    fetch(MMSEQS, tgz)
    binp = RAW / "tools" / "mmseqs" / "bin" / "mmseqs"
    if not binp.exists():
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(RAW / "tools")], check=True)
    print("  ", subprocess.run([str(binp), "version"], capture_output=True, text=True).stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
