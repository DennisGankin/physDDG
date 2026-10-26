"""45a: gradient check of every energy term (autograd vs central finite differences) + determinism.

Float64 copies of real crystal-structure coordinates (1BNL, FireProt set) are perturbed by small random noise
(0.05 Å) so that atoms sit at generic points of the piecewise-smooth terms. For each term separately we compare
the autograd gradient wrt ~60 random present atom coordinates with central differences (h = 1e-4 Å).
Also: float32 forward repeated -> bitwise identical; float32 vs float64 relative error.
Output: results/engine_energy/45a_gradcheck.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physddg import energy as E  # noqa: E402
from physddg.params import EnergyParams  # noqa: E402
from physddg.structure import load_protein  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
p = EnergyParams()
prot = load_protein("/workspace/data/fireprot_pdbs/1BNL.pdb")
b32 = E.blocks_from_protein(prot, dev, p)
# small region: residues within the first 40 (keeps FD affordable); add noise
sub = b32[torch.arange(0, 40, device=dev)]
b = E.Blocks(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sub.__dict__.items()})
b.xyz = b.xyz + 0.05 * torch.randn_like(b.xyz) * b.mask[..., None]
b.prev_c = torch.cat([torch.full_like(b.prev_c[:1], float("nan")), b.xyz[:-1, 2]], 0)  # keep consistent
names = E.term_names(p)


def energy(xyz):
    bb = E.Blocks(**{**b.__dict__, "xyz": xyz,
                     "prev_c": torch.cat([torch.full_like(xyz[:1, 2], float("nan")), xyz[:-1, 2]], 0)})
    return E.decompose(bb, p).total


x0 = b.xyz.detach().clone().requires_grad_(True)
tot = energy(x0)
present = torch.nonzero(b.mask)
sel = present[torch.randperm(len(present), device=dev)[:60]]
rows = []
for k, nm in enumerate(names):
    g, = torch.autograd.grad(tot[k], x0, retain_graph=True)
    h = 1e-4
    num, ana = [], []
    for (i, a) in sel.tolist():
        for c in range(3):
            xp = x0.detach().clone(); xp[i, a, c] += h
            xm = x0.detach().clone(); xm[i, a, c] -= h
            with torch.no_grad():
                num.append(((energy(xp)[k] - energy(xm)[k]) / (2 * h)).item())
            ana.append(g[i, a, c].item())
    num, ana = np.array(num), np.array(ana)
    scale = max(np.abs(num).max(), 1e-8)
    rows.append(dict(term=nm, value=tot[k].item(), max_abs_grad=scale,
                     max_abs_err=np.abs(num - ana).max(), rel_err=np.abs(num - ana).max() / scale,
                     n_nonzero=int((np.abs(num) > 1e-6).sum())))
df = pd.DataFrame(rows)
# determinism (float32, whole protein) and precision
e1 = E.decompose(b32, p).pair
e2 = E.decompose(b32, p).pair
det = bool(torch.equal(e1, e2))
x32 = b32.xyz.clone().requires_grad_(True)
bb = E.Blocks(**{**b32.__dict__, "xyz": x32})
g1, = torch.autograd.grad(E.decompose(bb, p).total.sum(), x32)
torch.use_deterministic_algorithms(True)
x32b = b32.xyz.clone().requires_grad_(True)
g2, = torch.autograd.grad(E.decompose(E.Blocks(**{**b32.__dict__, "xyz": x32b}), p).total.sum(), x32b)
x32c = b32.xyz.clone().requires_grad_(True)
g3, = torch.autograd.grad(E.decompose(E.Blocks(**{**b32.__dict__, "xyz": x32c}), p).total.sum(), x32c)
torch.use_deterministic_algorithms(False)
b64 = E.Blocks(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b32.__dict__.items()})
t32, t64 = E.decompose(b32, p).total.double(), E.decompose(b64, p).total
print(df.to_string(index=False))
print(f"forward bitwise deterministic (float32): {det}")
print(f"gradient bitwise deterministic with torch.use_deterministic_algorithms(True): {bool(torch.equal(g2, g3))}"
      f"; default mode max |dg| = {(g1 - g2).abs().max().item():.2e}")
print("float32 vs float64 total abs diff per term:", (t32 - t64).abs().cpu().numpy().round(5))
out = ROOT / "results/engine_energy"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "45a_gradcheck.csv", index=False)
