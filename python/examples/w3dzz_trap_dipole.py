"""W3DZZ five-band trap dipole (ARRL Antenna Handbook design, as used for the
coax-trap-form project): 32 ft inner sections, 60 pF / 8.2 uH traps, 22 ft outer
sections, fed with 75 ohm, 10 m above ground.

Run:  python3 examples/w3dzz_trap_dipole.py
      GROUND=pec python3 examples/w3dzz_trap_dipole.py     (or poor, "very good", ...)

The default is "average" soil (eps_r 13, 5 mS/m) via the Sommerfeld solution;
efficiency then includes the power the ground absorbs.  With these exact
dimensions 80/40/20 m work and 15/10 m resonate outside the bands, which nec2c
agrees with: real W3DZZ builds trim the outer wires for the high bands.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wiremom import Model, Structure  # noqa: E402

FT = 0.3048
H = 10.0  # height, m
A = 0.000814  # 14 AWG radius, m
CU = 5.8e7
INNER, OUTER = 32 * FT, 22 * FT
L_TRAP, C_TRAP, Q_TRAP = 8.2e-6, 60e-12, 100.0
F_TRAP = 1 / (2 * math.pi * math.sqrt(L_TRAP * C_TRAP))

st = Structure()
# the inner wire is one piece so the feed sits on its centre node
st.add_wire([0, -INNER, H], [0, INNER, H], A, 40, conductivity=CU)       # tag 1
st.add_wire([0, INNER, H], [0, INNER + OUTER, H], A, 14, conductivity=CU)   # tag 2
st.add_wire([0, -INNER, H], [0, -INNER - OUTER, H], A, 14, conductivity=CU)  # tag 3
m = Model(st, ground=os.environ.get("GROUND", "average"))
m.add_source(1, 20)
for tag in (2, 3):  # trap at the joint between inner and outer wire (outer wire node 0)
    m.add_load(tag, 0, l=L_TRAP, c=C_TRAP, r=Q_TRAP * 2 * math.pi * F_TRAP * L_TRAP, parallel=True)

print(f"trap resonance {F_TRAP / 1e6:.3f} MHz, {st.nsegments} segments")
bands = {"80 m": (3.4e6, 4.1e6), "40 m": (6.9e6, 7.5e6), "20 m": (13.8e6, 14.6e6),
         "15 m": (20.8e6, 21.8e6), "10 m": (27.5e6, 30.0e6)}
for band, (f0, f1) in bands.items():
    freqs = np.linspace(f0, f1, 36)
    sols = m.sweep(freqs)
    swr = np.array([s.vswr(75.0) for s in sols])
    i = int(np.argmin(swr))
    s = sols[i]
    z = s.input_impedance
    print(f"{band}: best SWR(75) {swr[i]:5.2f} at {freqs[i] / 1e6:6.3f} MHz   "
          f"Z = {z.real:7.1f} {z.imag:+7.1f}j   eff {100 * s.efficiency:5.1f}%   "
          f"broadside gain {s.gain_dbi(np.radians(45), np.radians(0))[0]:5.2f} dBi @45 deg elev")
