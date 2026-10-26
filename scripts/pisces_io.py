"""PISCES culled PDB list: download (once, into data/raw/pisces/) and map chains to the local PDB mirror.

Default set: cullpdb pc30, resolution <= 1.8 A, R <= 0.25, X-ray, chain length 40-10000 (latest list on the server).
Chains are split deterministically into 'train' (rotamer library / statistics) and 'test' (validation) by a
hash of the PDB id, so all chains of one entry fall on the same side.
"""
from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pisces"
URL = "https://dunbrack.fccc.edu/pisces/download/"
PATTERN = r"cullpdb_pc30\.0_res0\.0-1\.8_len40-10000_R0\.25_Xray_d[0-9_]+_chains[0-9]+"
MIRROR = Path("/rsyncPDB/data/structures/divided")


def download(pattern: str = PATTERN) -> Path:
    """Fetch the latest dated pisces_lists_*.tar.gz (individual list files return 403) and extract one list."""
    import tarfile
    RAW.mkdir(parents=True, exist_ok=True)
    have = sorted(p for p in RAW.iterdir() if re.fullmatch(pattern, p.name))
    if have:
        return have[-1]
    req = lambda u: urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req(URL), timeout=60).read().decode()
    tars = sorted(set(re.findall(r"pisces_lists_\d{4}_\d\d_\d\d\.tar\.gz", html)))
    if not tars:
        raise RuntimeError(f"no pisces_lists tarball at {URL}")
    tar_path = RAW / tars[-1]
    if not tar_path.exists():
        with urllib.request.urlopen(req(URL + tars[-1]), timeout=600) as r:
            tar_path.write_bytes(r.read())
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and re.fullmatch(pattern, Path(m.name).name)]
        if not members:
            raise RuntimeError(f"no list matching {pattern} in {tar_path}")
        m = sorted(members, key=lambda m: Path(m.name).name)[-1]
        out = RAW / Path(m.name).name
        out.write_bytes(tf.extractfile(m).read())
    return out


def read_list(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines()[1:]:
        f = line.split()
        if len(f) < 5:
            continue
        pdb, ch = f[0][:4].lower(), f[0][4:]
        rows.append({"pdb": pdb, "chain": ch, "length": int(f[1]), "method": f[2],
                     "resolution": float(f[3]), "rfactor": float(f[4])})
    return rows


def structure_path(pdb: str) -> Path | None:
    pdb = pdb.lower()
    for sub, name in (("mmCIF", f"{pdb}.cif.gz"), ("pdb", f"pdb{pdb}.ent.gz")):
        p = MIRROR / sub / pdb[1:3] / name
        if p.exists():
            return p
    return None


def split_of(pdb: str, test_frac: float = 0.1) -> str:
    h = int(hashlib.md5(pdb.lower().encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"
