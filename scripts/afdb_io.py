"""Shared helpers for the 2x_ data-AF scripts: locate and read AlphaFold DB models
from the local read-only stores (af2 tree, proteome tars via the offset index built by
21_index_af_tars.py) or, as a last resort, the EBI AFDB (cached under data/raw/afdb/).
"""
import gzip, io, os, re, subprocess, json
import pandas as pd
import gemmi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data/cache/af_tar_index")
RAW_AFDB = os.path.join(ROOT, "data/raw/afdb")
AF2_TSV = "/af_data/af2.tsv"
AF2_DIR = "/af_data"

# search order: (label, tar path); v4 human first (same release as the af2/ tree)
TARS = [
    ("v4_human_tar", "/af_data/UP000005640_9606_HUMAN_v4.tar"),
    ("v6_human", "/alphafold/v6/UP000005640_9606_HUMAN_v6.tar"),
    ("v6_swissprot", "/alphafold/v6/swissprot_pdb_v6.tar"),
    ("v6_yeast", "/alphafold/v6/UP000002311_559292_YEAST_v6.tar"),
    ("v6_ecoli", "/alphafold/v6/UP000000625_83333_ECOLI_v6.tar"),
    ("v6_mouse", "/alphafold/v6/UP000000589_10090_MOUSE_v6.tar"),
]
UA = "Mozilla/5.0 (X11; Linux x86_64) physddg-data-af"
FRAG_STEP = 200  # AFDB splits proteins >2700 aa into 1400-aa fragments every 200 aa
_MEMBER_RE = re.compile(r"AF-(.+?)-F(\d+)-model_v(\d+)\.pdb\.gz$")

_index = {}


def tar_index(label):
    """dict uniprot -> list of (frag, version, offset, size, member)."""
    if label in _index:
        return _index[label]
    path = dict(TARS)[label]
    f = os.path.join(IDX, os.path.basename(path).replace(".tar", ".tsv"))
    out = {}
    if os.path.exists(f):
        t = pd.read_csv(f, sep="\t")
        for m, o, s in zip(t.member, t.offset, t["size"]):
            mm = _MEMBER_RE.search(m)
            if mm:
                out.setdefault(mm.group(1), []).append((int(mm.group(2)), int(mm.group(3)), int(o), int(s), m))
    _index[label] = out
    return out


def read_tar_member(label, offset, size):
    with open(dict(TARS)[label], "rb") as fh:
        fh.seek(offset)
        return gzip.decompress(fh.read(size)).decode()


def _curl(url, dest=None, ua=True):
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "5"] + (["-A", UA] if ua else [])
    if dest:
        r = subprocess.run(cmd + ["-o", dest + ".part", url])
        if r.returncode == 0:
            os.replace(dest + ".part", dest)
        elif os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return r.returncode == 0
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def fetch_ebi(acc):
    """Download AF model(s) for an accession from the EBI AFDB (latest version), cached in data/raw/afdb/.
    Returns list of dict(frag, version, text, path, uniprot_start, uniprot_end).
    Notes: alphafold.ebi.ac.uk answers 403 without a browser-like User-Agent and 429 under bursts;
    newer AFDB entries (e.g. viral proteins) use IDs like AF-0000000365763033-model_v1 and may cover only a
    segment of the UniProt sequence (uniprotStart..uniprotEnd)."""
    os.makedirs(RAW_AFDB, exist_ok=True)
    meta_f = os.path.join(RAW_AFDB, f"{acc}.json")
    if not os.path.exists(meta_f):
        txt = None
        for ua in (True, False):
            txt = _curl(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", ua=ua)
            if txt and txt.strip().startswith("["):
                break
            txt = None
        if txt is None:
            # fall back to direct file URLs (single-fragment models, v6 then v4)
            for v in (6, 4):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
                fn = os.path.join(RAW_AFDB, os.path.basename(url))
                if os.path.exists(fn) or _curl(url, fn):
                    return [dict(frag=1, version=v, text=open(fn).read(), path=os.path.relpath(fn, ROOT),
                                 uniprot_start=1, uniprot_end=None)]
            print(f"EBI AFDB: no model reachable for {acc}", flush=True)
            return []
        open(meta_f, "w").write(txt)
    try:
        meta = json.load(open(meta_f))
    except Exception:
        return []
    out = []
    for e in meta:
        if e.get("uniprotAccession") != acc:
            continue
        url = e["pdbUrl"]
        fn = os.path.join(RAW_AFDB, os.path.basename(url))
        if not os.path.exists(fn) and not _curl(url, fn):
            print(f"EBI download failed for {url}", flush=True)
            continue
        mf = re.search(r"-F(\d+)-model", fn)
        mv = re.search(r"model_v(\d+)\.pdb", fn)
        out.append(dict(frag=int(mf.group(1)) if mf else 1, version=int(mv.group(1)) if mv else -1,
                        text=open(fn).read(), path=os.path.relpath(fn, ROOT),
                        uniprot_start=int(e.get("uniprotStart") or 1), uniprot_end=e.get("uniprotEnd")))
    return out


def structure_seq(pdb_text):
    """Return (seq, resnums, plddt per residue (CA B-factor)) of chain A of an AF model."""
    st = gemmi.read_pdb_string(pdb_text)
    ch = st[0][0]
    seq, nums, pl = [], [], []
    for r in ch:
        seq.append(gemmi.find_tabulated_residue(r.name).one_letter_code.upper())
        nums.append(r.seqid.num)
        ca = r.find_atom("CA", "*")
        pl.append(ca.b_iso if ca else float("nan"))
    return "".join(seq), nums, pl


def candidates(acc, sources=("af2_tree", "v4_human_tar", "v6_human", "v6_swissprot", "v6_yeast", "v6_ecoli", "v6_mouse"), ebi=True):
    """Yield candidate models for accession in priority order:
    dict(source, version, frag, n_frags, text, src_path, uniprot_offset); UniProt pos = model resnum + uniprot_offset."""
    for lab in sources:  # honour the given priority order
        if lab == "af2_tree":
            af = _af2()
            for r in af[af.uniprot_id == acc].itertuples():
                p = os.path.join(AF2_DIR, r.af2_pdb)
                if os.path.exists(p):
                    fr, nf = int(r.frag_id[1:]), int(r.n_frags)
                    yield dict(source="v4_human_af2tree", version=4, frag=fr, n_frags=nf, text=open(p).read(), src_path=p,
                               uniprot_offset=FRAG_STEP * (fr - 1) if nf > 1 else 0)
            continue
        ents = tar_index(lab).get(acc, [])
        nf = len(ents)
        for frag, ver, off, size, mem in sorted(ents):
            yield dict(source=lab, version=ver, frag=frag, n_frags=nf, text=read_tar_member(lab, off, size),
                       src_path=f"{dict(TARS)[lab]}::{mem}", uniprot_offset=FRAG_STEP * (frag - 1) if nf > 1 else 0)
    if ebi:
        ents = fetch_ebi(acc)
        nf = len(ents)
        for e in ents:
            # residue numbering of EBI models starts at 1 for the modelled segment
            off = e["uniprot_start"] - 1 if e["uniprot_start"] > 1 else (FRAG_STEP * (e["frag"] - 1) if nf > 1 else 0)
            yield dict(source="ebi_afdb", version=e["version"], frag=e["frag"], n_frags=nf, text=e["text"],
                       src_path=e["path"], uniprot_offset=off)


_af2_df = None


def _af2():
    global _af2_df
    if _af2_df is None:
        _af2_df = pd.read_csv(AF2_TSV, sep="\t", usecols=["uniprot_id", "frag_id", "n_frags", "af2_pdb"])
    return _af2_df
