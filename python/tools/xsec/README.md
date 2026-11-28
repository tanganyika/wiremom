# Cross-section solver for wires on or near the soil (experimental)

A radial lying on the ground is partly in air and partly in soil, which no
thin-wire engine can model. These scripts solve the static field of the wire's
cross-section (conductor, optional insulation, air above, lossy soil below, the
wire sunk by any amount) and turn it into the guided wave along the wire.

Needs `gmsh` and `scikit-fem` (and numpy, scipy, mpmath), for example in a
virtual environment: `python3 -m venv --system-site-packages venv &&
venv/bin/pip install gmsh scikit-fem`.

| File | What it does |
|---|---|
| `xsec.py` | 2-D finite-element solve: potential per unit charge `p` (units 1/(2π ε0)) of a wire at height `zc` (negative = in the soil), bare or insulated |
| `check_xsec.py` | Checks `xsec.py` against image theory (two heights over soil) and the coax formula (insulation), and that results don't depend on the outer boundary |
| `guided.py` | Guided-mode equation of one wire, or two parallel wires in phase, near lossy ground (2-D spectral form, oscillatory integrals by QUADPACK's QAWF) |
| `wave_cases.py` | The guided wave along a radial above, resting on (5–50% of its circumference in the soil) and under the soil, bare and PVC, corrected from bare filaments 3 mm above and below the surface |
| `resting_antenna.py` | A 20 m vertical with 16 radials (3 m and 5 m, bare or PVC) above, resting on and in the soil: impedance and efficiency, from both references; results in `resting_antenna.json` |
| `extra_p-experiment.patch`, `eff-runner/` | Experiment: `Wire::extra_p`, a per-wire near-field correction in wiremom, and a runner for a 20 m vertical with radials |

## Method

The soil isn't magnetic, so the wire's surroundings change only its electric
near field: its potential per unit charge. The mode equation of a thin filament
(`guided.py`, which wiremom matches within about 1%) is corrected by
`j β²/ω · Δp/(2π ε0)`, with `Δp` the 2-D difference between the real
cross-section and the filament's. Corrected from 3 mm above to 3 mm below the
surface (or back), it reproduces the exact guided wave within 0.5%, for one wire
and for two parallel wires.

## Status

The guided wave along a radial is reliable. Put into wiremom (`extra_p`), the
same correction now also gives whole antennas: corrected from 3 mm above to 3 mm
below the surface (or back), a 20 m vertical with 16 radials matches wiremom's
own model of the buried radials (25.8%, and 30.2% against 30.3% the other way).
It didn't before wiremom's fix for currents through the surface (the closed-form
potentials jumped there), which this cross-check found.

`interp_check.py` and `effective_height.py` explore how strongly a radial picks
up the vertical's field at different heights; that turned out not to matter.
