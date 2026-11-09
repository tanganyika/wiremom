"""Cross-check wiremom against nec2c on a set of reference antennas.

Usage:  python3 tools/nec2_compare.py          (needs `nec2c` on PATH)

Each case is described once and turned into both a wiremom Model and a NEC-2
deck.  Sources and loads in these cases sit at the centre of a wire (or at a
wire's ground end), where wiremom's node and NEC-2's segment centre coincide:
wiremom uses an even segment count with the node in the middle, NEC-2 the
next odd count with the middle segment.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wiremom import C0, Model, Structure  # noqa: E402


@dataclass
class W:
    start: tuple
    end: tuple
    radius: float
    nseg: int  # wiremom segments (even when the wire carries a centre feed/load)
    centred: bool = False  # NEC-2 gets nseg+1 so it has a centre segment
    sigma: float | None = None


@dataclass
class Case:
    name: str
    freq: float
    wires: list[W]
    feed: tuple[int, str]  # (wire index from 1, "centre" | "base")
    loads: list[tuple[int, dict]] = field(default_factory=list)  # (wire, load kwargs), centre
    ground: str | None = None
    directions: list[tuple[float, float]] = field(default_factory=lambda: [(90.0, 0.0)])


def wiremom_solve(case: Case):
    st = Structure()
    for w in case.wires:
        st.add_wire(w.start, w.end, w.radius, w.nseg, conductivity=w.sigma)
    m = Model(st, ground=case.ground)
    tag, where = case.feed
    m.add_source(tag, case.wires[tag - 1].nseg // 2 if where == "centre" else 0)
    for tag, kw in case.loads:
        m.add_load(tag, case.wires[tag - 1].nseg // 2, **kw)
    sol = m.solve(case.freq)
    th = np.radians([d[0] for d in case.directions])
    ph = np.radians([d[1] for d in case.directions])
    return sol.input_impedance, sol.gain_dbi(th, ph)


def nec_deck(case: Case) -> str:
    lines = ["CM wiremom comparison: " + case.name, "CE"]
    for i, w in enumerate(case.wires, 1):
        n = w.nseg + 1 if w.centred else w.nseg
        lines.append("GW %d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g" % (i, n, *w.start, *w.end, w.radius))
    lines.append("GE %d" % (1 if case.ground == "pec" else 0))
    if case.ground == "pec":
        lines.append("GN 1")

    def seg_of(tag, where):
        w = case.wires[tag - 1]
        return 1 if where == "base" else (w.nseg + 1) // 2 + 1 if w.centred else w.nseg // 2

    for i, w in enumerate(case.wires, 1):
        if w.sigma:
            lines.append("LD 5 %d 0 0 %.6g" % (i, w.sigma))
    for tag, kw in case.loads:
        s = seg_of(tag, "centre")
        r, l, c = kw.get("r", 0.0), kw.get("l", 0.0), kw.get("c") or 0.0
        lines.append("LD %d %d %d %d %.9g %.9g %.9g" % (1 if kw.get("parallel") else 0, tag, s, s, r, l, c))
    lines.append("EX 0 %d %d 0 1 0" % (case.feed[0], seg_of(*case.feed)))
    lines.append("FR 0 1 0 0 %.9g 0" % (case.freq / 1e6))
    for th, ph in case.directions:
        lines.append("RP 0 1 1 1000 %.6g %.6g 0 0" % (th, ph))
    lines.append("EN")
    return "\n".join(lines) + "\n"


def nec2c_solve(case: Case):
    with tempfile.TemporaryDirectory() as d:
        deck, out = os.path.join(d, "c.nec"), os.path.join(d, "c.out")
        with open(deck, "w") as f:
            f.write(nec_deck(case))
        subprocess.run(["nec2c", "-i", deck, "-o", out], check=True, capture_output=True)
        text = open(out).read()
    m = re.search(r"ANTENNA INPUT PARAMETERS.*?\n(?:.*\n){2}\s*\d+\s+\d+((?:\s+\S+){6})", text)
    vals = [float(x) for x in m.group(1).split()]
    z = complex(vals[4], vals[5])
    gains = []
    for blk in re.finditer(r"RADIATION PATTERNS.*?DEGREES\s+DEGREES\s+DB\s+DB\s+DB.*?\n\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
                           text, re.S):
        gains.append(float(blk.group(5)))
    return z, np.array(gains)


def cases() -> list[Case]:
    f = 14.2e6
    lam = C0 / f
    h = lam / 4
    cu = 5.8e7
    out = [
        Case("half-wave dipole, a=lam/1000", 299.792458e6,
             [W((0, 0, -0.25), (0, 0, 0.25), 0.001, 40, True)], (1, "centre")),
        Case("20 m dipole 0.475 lam, 2 mm copper", f,
             [W((0, -0.475 * lam / 2, 10), (0, 0.475 * lam / 2, 10), 0.001, 40, True, cu)],
             (1, "centre"), directions=[(90.0, 0.0), (90.0, 45.0)]),
        Case("quarter-wave monopole on PEC ground", f,
             [W((0, 0, 0), (0, 0, 0.24 * lam), 0.005, 20)], (1, "base"), ground="pec",
             directions=[(90.0, 0.0), (60.0, 0.0)]),
        Case("short dipole with centre loading coil", f,
             [W((0, 0, -lam / 8), (0, 0, lam / 8), 0.002, 40, True)], (1, "centre"),
             loads=[], directions=[(90.0, 0.0)]),
        Case("trap dipole: parallel-LC trap at the centre of each outer section", f,
             [W((0, -0.1 * lam, 10), (0, 0.1 * lam, 10), 0.001, 20, True),
              W((0, 0.1 * lam, 10), (0, 0.2 * lam, 10), 0.001, 10, True),
              W((0, -0.1 * lam, 10), (0, -0.2 * lam, 10), 0.001, 10, True)],
             (1, "centre"),
             # trap resonant at 18 MHz; loss as a parallel R for Q = 100
             loads=[(t, dict(l=1.5e-6, c=1 / ((2 * math.pi * 18e6) ** 2 * 1.5e-6),
                             r=100 * 2 * math.pi * 18e6 * 1.5e-6, parallel=True)) for t in (2, 3)]),
        Case("3-element Yagi (forward / back)", f,
             [W((0, -0.51 * lam / 2, 0), (0, 0.51 * lam / 2, 0), 0.006, 40, True),
              W((0.2 * lam, -0.475 * lam / 2, 0), (0.2 * lam, 0.475 * lam / 2, 0), 0.006, 40, True),
              W((0.35 * lam, -0.44 * lam / 2, 0), (0.35 * lam, 0.44 * lam / 2, 0), 0.006, 40, True)],
             (2, "centre"), directions=[(90.0, 0.0), (90.0, 180.0)]),
        Case("inverted-V on PEC ground (bent junctions, short centre feed wire)", f,
             [W((0, -0.25, 10), (0, 0.25, 10), 0.001, 2, True),
              W((0, 0.25, 10), (0, 0.25 + 0.24 * lam * math.cos(0.6), 10 - 0.24 * lam * math.sin(0.6)), 0.001, 20),
              W((0, -0.25, 10), (0, -0.25 - 0.24 * lam * math.cos(0.6), 10 - 0.24 * lam * math.sin(0.6)), 0.001, 20)],
             (1, "centre"), ground="pec", directions=[(0.0, 0.0), (45.0, 0.0), (45.0, 90.0)]),
    ]
    # loading coil: series L that roughly resonates the quarter-wave dipole
    out[3].loads = [(1, dict(l=10e-6, r=1.0))]
    return out


def main() -> int:
    worst = 0.0
    for case in cases():
        zw, gw = wiremom_solve(case)
        zn, gn = nec2c_solve(case)
        dz = abs(zw - zn) / abs(zn)
        worst = max(worst, dz)
        print(f"{case.name}\n  wiremom  Z = {zw.real:9.2f} {zw.imag:+9.2f}j   gain dBi = {np.round(gw, 2)}"
              f"\n  nec2c    Z = {zn.real:9.2f} {zn.imag:+9.2f}j   gain dBi = {np.round(gn, 2)}"
              f"\n  |dZ|/|Z| = {100 * dz:.1f}%   max dGain = {np.max(np.abs(gw - gn)):.2f} dB")
    print(f"\nworst impedance difference: {100 * worst:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
