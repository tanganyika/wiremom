"""Guided wave along one radial (radius 0.5 mm, bare or 0.75 mm PVC) above, resting
on and in average soil at 14.15 MHz, from the 2-D near field corrected onto bare
filaments 3 mm above and 3 mm below the surface (both, as a check)."""
import math, json, warnings
import numpy as np
from guided import ex_at, root, K0, EG, W
from xsec import solve
warnings.filterwarnings("ignore")
es = complex(EG); a, t_pvc, eps_pvc = 0.5e-3, 0.75e-3, 3.0
EPS0 = 8.8541878128e-12
refs = {z: solve(a, 0, 1, z, es)[0] for z in (3e-3, -3e-3)}
def beta(z_ref, dp, guess):
    dP = dp/(2*np.pi*EPS0)
    return root(lambda b: ex_at(b, z_ref, a, a) + 1j*b**2/W*dP, guess)/K0
rows = []
for label, t in (("bare", 0.0), ("PVC", t_pvc)):
    ro = a + t
    cases = [("2 mm above", ro + 2e-3), ("1 mm above", ro + 1e-3), ("0.25 mm above", ro + 0.25e-3)]
    cases += [(f"resting, {f:.0%} in soil", ro*math.cos(math.pi*f)) for f in (0.05, 0.15, 0.20, 0.25, 0.30, 0.50)]
    cases += [("just under (0.25 mm)", -(ro + 0.25e-3)), ("1 cm deep", -1e-2)]
    g = {3e-3: K0*1.6, -3e-3: K0*1.6}
    for name, zc in cases:
        p, _ = solve(a, t, eps_pvc, zc, es)
        bs = {}
        for z_ref in (3e-3, -3e-3):
            b = beta(z_ref, p - refs[z_ref], g[z_ref])
            g[z_ref] = b*K0
            bs[z_ref] = b
        rows.append({"wire": label, "case": name, "b_up": [bs[3e-3].real, bs[3e-3].imag], "b_dn": [bs[-3e-3].real, bs[-3e-3].imag]})
        b = bs[3e-3]
        print(f"{label:4} {name:24} beta/k0 {bs[3e-3]:.3f} / {bs[-3e-3]:.3f}   wavelength {2*np.pi/(b.real*K0):5.2f} m   loss {-b.imag*K0*8.686:5.2f} dB/m", flush=True)
json.dump(rows, open("wave_cases.json", "w"), indent=1)
