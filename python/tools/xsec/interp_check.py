"""Antenna (20 m vertical, 16 x 5 m radials) from two corrected references,
interpolated in the effective height of the target's charge; checked against
wiremom modelling the target directly."""
import subprocess, sys, math
import numpy as np
from xsec import solve
from effective_height import dipole, ES
EFF = sys.argv[1]
a = 0.5e-3
SLOPE = dipole(a, 0, 1, 3e-3).real / 3e-3  # dipole per metre of height in the air

def heff(t, eps_i, zc):
    return dipole(a, t, eps_i, zc).real / SLOPE

def ant(z_ref, dp):
    r = subprocess.run([EFF, str(z_ref), str(dp.real), str(dp.imag)], capture_output=True, text=True).stdout.split()
    return complex(float(r[0]), float(r[1])), float(r[2])

def interp(t, eps_i, zc, lo=-3e-3, hi=3e-3):
    p = solve(a, t, eps_i, zc, ES)[0]
    out = []
    for z in (lo, hi):
        out.append((ant(z, p - solve(a, 0, 1, z, ES)[0]), heff(0, 1, z)))
    (zl, el), hl = out[0]
    (zh, eh), hh = out[1]
    h = heff(t, eps_i, zc)
    w = (h - hl) / (hh - hl)
    return zl + w * (zh - zl), el + w * (eh - el), h

if __name__ == "__main__":
    for zt, lo, hi in ((3e-3, -3e-3, 6e-3), (1.5e-3, -3e-3, 3e-3), (-1.5e-3, -3e-3, 3e-3), (6e-3, -3e-3, 3e-3)):
        z, e, h = interp(0, 1, zt, lo, hi)
        zt_, et = ant(zt, 0j)
        print(f"target bare wire at {zt*1e3:+.1f} mm (effective {h*1e3:+.2f} mm): interpolated {z.real:.1f}{z.imag:+.1f}j {100*e:.1f}%   wiremom directly {zt_.real:.1f}{zt_.imag:+.1f}j {100*et:.1f}%", flush=True)
