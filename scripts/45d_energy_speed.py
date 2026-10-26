"""45d: energy evaluation throughput on one RTX 4090 (float32).

(i)  whole-protein per-residue decomposition (decompose): ~300-residue crystal structure (1CQW, 295 res),
     forward only (no_grad) and forward+backward.
(ii) batched local systems with local_energy:
     a) 10k variants x 3 flexible residues (~40 moving atoms) vs ~2k environment atoms (one-body + two-body)
     b) 10k variants of ONE site with random amino acid and chis (E1 builder) vs the full protein environment
        (the mutation-scan / rotamer-scoring case)
Timings: median of repeated runs after warm-up, torch.cuda.synchronize around each.
Output: results/engine_energy/45d_speed.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import energy as E  # noqa: E402
from physddg.builder import build_sidechains  # noqa: E402
from physddg.params import EnergyParams  # noqa: E402
from physddg.structure import load_protein  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
p = EnergyParams()


def timeit(fn, n=10, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


rows = []
prot = load_protein("/workspace/data/fireprot_pdbs/1CQW.pdb")
b = E.blocks_from_protein(prot, dev, p)
L = len(prot)
nat = int(b.mask.sum())
with torch.no_grad():
    t = timeit(lambda: E.decompose(b, p))
rows.append(dict(test="decompose_forward", residues=L, atoms=nat, seconds=t, evals_per_s=1 / t))


def fb():
    x = b.xyz.clone().requires_grad_(True)
    E.decompose(E.Blocks(**{**b.__dict__, "xyz": x}), p).total.sum().backward()


t = timeit(fb)
rows.append(dict(test="decompose_forward_backward", residues=L, atoms=nat, seconds=t, evals_per_s=1 / t))

# (ii a) 3 flexible residues (~40 atoms) vs ~2k-atom environment, 10k variants (side chains jittered)
ca = b.xyz[:, 1]
cand = [i for i in range(1, L - 1) if prot.seq[i] in "FWYRKM"]
site = [cand[len(cand) // 2]]
d = (ca - ca[site[0]]).norm(dim=-1)
order = torch.argsort(d).tolist()
flex = [order[0]] + [j for j in order[1:] if prot.seq[j] in "FWYRKMLEQ"][:2]
n_mov = int(b.mask[flex].sum())
envres = [j for j in order if j not in flex]
cum = np.cumsum(b.mask[envres].sum(1).cpu().numpy())
envres = envres[: int(np.searchsorted(cum, 2000)) + 1]
keep = torch.zeros(L, dtype=torch.bool, device=dev)
keep[envres] = True
env = E.make_env(b, torch.nonzero(~keep).squeeze(1), p)
n_env = int(env.f["m"][:-1].sum())
B = 10000
mov = b[torch.tensor(flex, device=dev)[None].expand(B, -1)]
sc = torch.zeros_like(mov.mask)
sc[..., 4:14] = True
mov.xyz = mov.xyz + 0.3 * torch.randn_like(mov.xyz) * (sc & mov.mask)[..., None]
with torch.no_grad():
    t = timeit(lambda: E.local_energy(mov, env, p), n=5)
rows.append(dict(test="local_10k_x_3res_vs_env", residues=len(flex), atoms=n_mov, env_atoms=n_env, variants=B,
                 seconds=t, evals_per_s=B / t))

# (ii b) one site, 10k variants with random aa + chis built by E1's builder, vs full protein environment
i = site[0]
bb = b.xyz[i, :4][None]
aa = torch.randint(0, 20, (1, B), device=dev)
chis = torch.rand(1, B, 4, device=dev) * 360 - 180
xyz14, m14 = build_sidechains(bb, aa, chis)
xyz15 = torch.cat([torch.nan_to_num(xyz14[0]), torch.zeros(B, 1, 3, device=dev)], 1)
m15 = torch.cat([m14[0], torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1)
one = b[torch.full((B,), i, device=dev)]
mv = E.Blocks(xyz=xyz15, aa=aa[0], mask=m15, pos=one.pos, prev_c=one.prev_c, nterm=one.nterm, burial=one.burial)
env_full = E.make_env(b, [i], p)
with torch.no_grad():
    t = timeit(lambda: E.local_energy(mv[:, None], env_full, p), n=5)
rows.append(dict(test="local_10k_variants_1site_vs_protein", residues=1, atoms=float(m15.sum(1).float().mean()),
                 env_atoms=int(env_full.f["m"][:-1].sum()), variants=B, seconds=t, evals_per_s=B / t))
# same local tests with the torch.compile'd short-range kernel (compile time excluded: warm-up)
E.enable_compile(True)
with torch.no_grad():
    t = timeit(lambda: E.local_energy(mov, env, p), n=5)
    rows.append(dict(test="compiled_local_10k_x_3res_vs_env", residues=len(flex), atoms=n_mov, env_atoms=n_env,
                     variants=B, seconds=t, evals_per_s=B / t))
    t = timeit(lambda: E.local_energy(mv[:, None], env_full, p), n=5)
    rows.append(dict(test="compiled_local_10k_variants_1site_vs_protein", residues=1,
                     atoms=float(m15.sum(1).float().mean()), env_atoms=int(env_full.f["m"][:-1].sum()), variants=B,
                     seconds=t, evals_per_s=B / t))
E.enable_compile(False)
df = pd.DataFrame(rows)
print(torch.cuda.get_device_name(0))
print(df.to_string(index=False))
df.to_csv(ROOT / "results/engine_energy/45d_speed.csv", index=False)
