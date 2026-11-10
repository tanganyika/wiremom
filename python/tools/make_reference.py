"""Freeze reference results for the Rust tests (tests/data/reference.json).

Run from the python/ directory:  ../.venv/bin/python tools/make_reference.py

For every case in nec2_compare.cases() it stores the geometry, the Python
prototype's impedance and gains, and nec2c's (when nec2c is installed), plus
Sommerfeld remainder values at sample points for every soil preset.  The Rust
tests only read this file; they never run Python.
"""

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import nec2_compare as nc  # noqa: E402
from wiremom import ground as gnd  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "tests", "data", "reference.json")


def cplx(z):
    return [float(np.real(z)), float(np.imag(z))]


def case_json(case):
    loads = []
    for tag, kw in case.loads:
        loads.append({"wire": tag, "node": case.wires[tag - 1].nseg // 2, "r": kw.get("r", 0.0),
                      "l": kw.get("l", 0.0), "c": kw.get("c"), "parallel": bool(kw.get("parallel"))})
    tag, where = case.feed
    g = case.ground
    ground = {"kind": "free"} if g is None else {"kind": "pec"} if g == "pec" else \
        {"kind": "real", "eps_r": g[0], "sigma": g[1]}
    zp, gp = nc.wiremom_solve(case)
    out = {
        "name": case.name, "freq": case.freq, "ground": ground,
        "wires": [{"start": list(w.start), "end": list(w.end), "radius": w.radius, "nseg": w.nseg,
                   "sigma": w.sigma} for w in case.wires],
        "source": {"wire": tag, "node": case.wires[tag - 1].nseg // 2 if where == "centre" else 0},
        "loads": loads,
        "directions_deg": [list(d) for d in case.directions],  # (theta from zenith, phi)
        "python": {"z": cplx(zp), "gain_dbi": [float(x) for x in gp]},
    }
    if shutil.which("nec2c"):
        zn, gn = nc.nec2c_solve(case)
        out["nec2c"] = {"z": cplx(zn), "gain_dbi": [float(x) for x in gn]}
    return out


def dyadic_json():
    f = 14.2e6
    pts = [(0.0, 0.05), (0.3, 0.1), (5.0, 0.2), (1.0, 8.0), (20.0, 3.0), (60.0, 1.0), (0.01, 40.0)]
    rows = []
    for name, gr in gnd.PRESETS.items():
        eps, g0 = gr.eps_c(f), gr.gamma0(f)
        for rho, h in pts:
            d = gnd.reflected_dyadic(rho, h, f, eps, (g0, -g0))
            rows.append({"ground": name, "eps_r": gr.eps_r, "sigma": gr.sigma, "freq": f,
                         "rho": rho, "h": h, "d": [cplx(x) for x in d]})
    return rows


def main():
    data = {"cases": [case_json(c) for c in nc.cases()], "sommerfeld_remainder": dyadic_json()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {os.path.relpath(OUT)}: {len(data['cases'])} cases, "
          f"{len(data['sommerfeld_remainder'])} Sommerfeld points")


if __name__ == "__main__":
    main()
