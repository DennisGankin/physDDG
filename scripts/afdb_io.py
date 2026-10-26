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


def fetch_ebi(acc):
    """Download AF model(s) for accession from EBI AFDB (latest version). Returns list of (frag, version, pdb_text)."""
    os.makedirs(RAW_AFDB, exist_ok=True)
    meta_f = os.path.join(RAW_AFDB, f"{acc}.json")
    if not os.path.exists(meta_f):
        ok = False
        for ua in ([], ["-A", UA]):  # the API has answered 403 intermittently with either header
            r = subprocess.run(["curl", "-sSfL", *ua, "--retry", "3", f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip().startswith("["):
                open(meta_f, "w").write(r.stdout); ok = True; break
        if not ok:
            # fall back to direct file URLs (single-fragment models, v6 then v4)
            for v in (6, 4):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
                fn = os.path.join(RAW_AFDB, os.path.basename(url))
                if not os.path.exists(fn):
                    rr = subprocess.run(["curl", "-sSfL", "-A", UA, "--retry", "2", "-o", fn + ".part", url])
                    if rr.returncode != 0:
                        if os.path.exists(fn + ".part"):
                            os.remove(fn + ".part")
                        continue
                    os.replace(fn + ".part", fn)
                return [(1, v, open(fn).read(), os.path.relpath(fn, ROOT))]
            print(f"EBI AFDB: no model reachable for {acc}", flush=True)
            return []
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
        if not os.path.exists(fn):
            # NB: alphafold.ebi.ac.uk/files returns 403 without a browser-like User-Agent
            r = subprocess.run(["curl", "-sSfL", "-A", UA, "--retry", "3", "-o", fn + ".part", url])
            if r.returncode != 0:
                print(f"EBI download failed for {url}", flush=True)
                continue
            os.replace(fn + ".part", fn)
        mm = re.search(r"-F(\d+)-model_v(\d+)\.pdb", fn)
        out.append((int(mm.group(1)), int(mm.group(2)), open(fn).read(), os.path.relpath(fn, ROOT)))
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
    dict(source, version, frag, n_frags, text, src_path)."""
    if "af2_tree" in sources:
        af = _af2()
        rows = af[af.uniprot_id == acc]
        for r in rows.itertuples():
            p = os.path.join(AF2_DIR, r.af2_pdb)
            if os.path.exists(p):
                yield dict(source="v4_human_af2tree", version=4, frag=int(r.frag_id[1:]), n_frags=int(r.n_frags),
                           text=open(p).read(), src_path=p)
    for lab in sources:
        if lab == "af2_tree":
            continue
        ents = tar_index(lab).get(acc, [])
        nf = len(ents)
        for frag, ver, off, size, mem in sorted(ents):
            yield dict(source=lab, version=ver, frag=frag, n_frags=nf, text=read_tar_member(lab, off, size),
                       src_path=f"{dict(TARS)[lab]}::{mem}")
    if ebi:
        ents = fetch_ebi(acc)
        for frag, ver, text, fn in ents:
            yield dict(source="ebi_afdb", version=ver, frag=frag, n_frags=len(ents), text=text, src_path=fn)


_af2_df = None


def _af2():
    global _af2_df
    if _af2_df is None:
        _af2_df = pd.read_csv(AF2_TSV, sep="\t", usecols=["uniprot_id", "frag_id", "n_frags", "af2_pdb"])
    return _af2_df
