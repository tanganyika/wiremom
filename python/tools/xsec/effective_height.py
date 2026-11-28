"""Effective height of a wire's charge for coupling to an outside vertical field:
the vertical dipole of its far potential in the air, scaled by that of bare thin
wires at known heights in the air."""
import math
import numpy as np
from xsec import solve
F, EPS0 = 14.15e6, 8.8541878128e-12
ES = complex(13, -0.005 / (2 * math.pi * F * EPS0))

def dipole(a, t, eps_i, zc):
    out = {}
    solve(a, t, eps_i, zc, ES, kw_out=out)
    return out["dipole"]

if __name__ == "__main__":
    a = 0.5e-3
    for z in (6e-3, 3e-3, 1.5e-3, -1.5e-3, -3e-3, -6e-3):
        print(f"bare wire at {z*1e3:+.1f} mm: dipole {dipole(a, 0, 1, z):.5f}")
