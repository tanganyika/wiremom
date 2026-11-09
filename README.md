# wiremom

An open-source thin-wire **method-of-moments** antenna solver built on the
**mixed-potential integral equation (MPIE)**, the formulation NEC-5 moved to. It
is written from the published literature, not from NEC source code. It is meant
to become the engine of an antenna-modelling app, so the core is a small,
dependency-light library (NumPy, plus SciPy for Bessel functions and splines)
with a plain Python API.

## What it does today

| Feature | Status |
|---|---|
| Straight wires, any orientation, automatic junctions (any number of wires per node) | ✅ |
| Triangle (rooftop) basis functions + Galerkin testing, MPIE | ✅ |
| Delta-gap voltage sources (interior nodes, two-wire joints, ground connections) | ✅ |
| Lumped series / parallel RLC loads (coils, traps), open circuits (insulators) | ✅ |
| Skin-effect wire loss (conductivity per wire) | ✅ |
| Free space, perfect ground plane (image theory, wires may touch ground) | ✅ |
| Real (lossy) ground: full Sommerfeld solution, Fresnel far field, soil presets | ✅ |
| Feed-point impedance, VSWR, frequency sweeps | ✅ |
| Far-field gain (dBi), radiated power, efficiency | ✅ |
| Wires buried in or connected to real ground (radials on the soil, ground rods) | ⬜ roadmap |
| Near fields, current distribution export | ⬜ roadmap |
| NEC-2 deck (`GW/GA/GH/GM/EX/LD/FR/GN/RP`) import and export | ⬜ roadmap |
| Transmission-line and network elements (`TL`/`NT`) | ⬜ roadmap |
| Surface patches and wire-to-patch junctions (the NEC-5 headline feature) | ⬜ roadmap |
| Speed: frequency interpolation of the matrix, native core | ⬜ roadmap |

## How it works

1. **Mesh.** Each wire is split into equal straight segments. Segment ends closer
   than a small tolerance merge into one node. A node shared by *k* segment ends
   carries *k − 1* triangle basis functions, and free ends carry none. On a
   ground plane, each segment end at z = 0 gets a half-triangle whose other half
   is its image.
2. **Integral equation.** The tangential field on each wire must be zero:
   `jωA·ŝ + ∂Φ/∂s = E_inc·ŝ`. The vector potential **A** comes from the current
   and the scalar potential Φ from the charge (the derivative of the current).
   Keeping the two separate is what makes MPIE robust with short segments and at
   low frequency.
3. **Matrix.** With Galerkin testing,
   `Z_mn = jωμ ∬ Λm Λn (ŝm·ŝn) G + (1/jωε) ∬ Λm′ Λn′ G`, where
   `G = e^{−jkR}/4πR` and `R = √(|r−r′|² + a²)` is the reduced thin-wire kernel.
   The singular `1/R` part is integrated analytically along the source segment,
   and the smooth remainder by Gauss–Legendre quadrature. For wires of different
   radius, `a²` is the mean of both radii squared, which keeps Z symmetric
   (reciprocal).
4. **Solve** `Z I = V`. The coefficient of a node's basis function is the
   current at that node, so the feed impedance is `V / I_feed`.
5. **Far field.** Integrate the currents (plus images) with the far-field phase
   factor, then `G = 4πU / P_in`.

### Real ground

The field reflected by a lossy half-space is written as Sommerfeld integrals.
Each plane wave in the source's spectrum reflects with its own TE/TM Fresnel
coefficient, and integrating over azimuth leaves J0/J1/J2 Bessel integrals over
the radial wavenumber
([ground.py](wiremom/ground.py) has the formulas). As in NEC-2:

- The near-singular part, the quasi-static image scaled by
  `Γ0 = (ε−1)/(ε+1)`, is taken out and handled by the same analytic image
  integrals as the perfect ground.
- The smooth remainder is tabulated per frequency on an (R, θ) grid and
  interpolated with splines.
- The integrals follow two raised arcs in the complex plane around the branch
  points and the surface-wave pole. The tail then follows Hankel functions on
  vertical paths, or the real axis when the source is nearly overhead.
- Tables are stored in normalised (kR) form, so a sweep reuses one table within
  ±2% of its frequency (`Model.ground_table_reuse`), at a cost of ~0.1% in
  impedance.
- The far field adds the image's field scaled by the Fresnel coefficients at
  each elevation angle (the space wave). Efficiency therefore counts power
  absorbed by the ground as lost.

```python
m = Model(st, ground="average")       # or "poor", "very good", "sea water", ...
m = Model(st, ground=(15.0, 0.008))   # (relative permittivity, conductivity S/m)
```

Over real ground, wires must be above z = 0. Very low wires (a few cm) work,
but buried or ground-connected wires don't yet.

## Usage

```python
from wiremom import Structure, Model, C0

f = 14.2e6
lam = C0 / f
st = Structure()
st.add_wire([0, -0.24 * lam, 10], [0, 0.24 * lam, 10], radius=0.001, nseg=40,
            conductivity=5.8e7)                     # copper
m = Model(st, ground="pec")                          # or ground=None for free space
m.add_source(1, 20)                                  # wire tag 1, centre node
sol = m.solve(f)
print(sol.input_impedance, sol.vswr(50), sol.gain_dbi(theta=0.5, phi=0.0))
```

Nodes are numbered `0 … nseg` along each wire, from its start point to its end
point. Loads use the same addressing:
`m.add_load(tag, node, l=8.2e-6, c=60e-12, r=37e3, parallel=True)`.

See [examples/w3dzz_trap_dipole.py](examples/w3dzz_trap_dipole.py) for a five-band
trap dipole swept across 80–10 m.

## Validation

`tests/` checks the physics directly: thin-dipole impedance and 2.15 dBi gain,
convergence with segmentation, matrix symmetry, monopole = ½ dipole and +3 dB,
copper loss against the analytic skin-effect resistance, series loads, trap
isolation, feeds at wire joints, and agreement with NEC-2.

`python3 tools/nec2_compare.py` runs the same models through
[nec2c](https://www.qsl.net/5b4az/pages/nec2.html). Current results:

| Case | |ΔZ| / |Z| | max ΔGain |
|---|---|---|
| half-wave dipole | 3.1 % | 0.00 dB |
| 20 m copper dipole | 1.9 % | 0.00 dB |
| λ/4 monopole on PEC ground | 2.2 % | 0.01 dB |
| short dipole + loading coil | 8.5 % | 0.01 dB |
| trap dipole (parallel RLC, Q = 100) | 1.1 % | 0.01 dB |
| 3-element Yagi (fwd/back) | 1.7 % | 0.09 dB |
| inverted-V on PEC ground | 0.8 % | 0.00 dB |
| 20 m dipole at 10 m, average soil (Sommerfeld) | 0.8 % | 0.01 dB |
| 20 m dipole at 4 m, poor soil | 1.0 % | 0.00 dB |
| 40 m dipole at 6 m (0.14 λ), average soil | 1.0 % | 0.00 dB |
| vertical λ/2, base 1 m up, average soil | 1.7 % | 0.01 dB |
| 3-element Yagi at 12 m, very good soil | 2.4 % | 0.24 dB (back lobe) |

Over real ground the gains are compared at 10–90° elevation. Spot checks
outside the harness also agree within about 2%: a dipole only 0.3 m above soil,
a vertical with its base 0.1 m up, sea water, and very poor (city) ground.
[tests/test_ground.py](tests/test_ground.py) also checks the Sommerfeld
integrals against the closed-form image for a perfect conductor (to 1e-7) and
the tables against direct integration for every soil preset.

Resistance and gain agree closely. Most of the impedance difference is in the
reactance, because NEC-2 drives a whole segment while wiremom drives a node. The
loading-coil case is largest because the coil cancels most of the dipole's
−720 Ω, which magnifies a ~14 Ω difference. NEC-2's own value moves toward
wiremom's as its segmentation is refined (−743 → −732 → −725 Ω for 20 → 40 → 80
segments), while wiremom's barely changes.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy pytest
.venv/bin/python -m pytest -q
```

## References

- R. F. Harrington, *Field Computation by Moment Methods* (1968).
- S. M. Rao, D. R. Wilton, A. W. Glisson, "Electromagnetic scattering by
  surfaces of arbitrary shape," IEEE Trans. AP-30, 1982 (RWG patches, for the
  surface roadmap item).
- D. R. Wilton, C. M. Butler, "Efficient numerical techniques for solving
  Pocklington's equation and their relationships to other methods," IEEE
  Trans. AP-24, 1976.
- K. A. Michalski, J. R. Mosig, "Multilayered media Green's functions in
  integral equation formulations," IEEE Trans. AP-45, 1997 (real ground).
- G. J. Burke, E. K. Miller, "Modeling antennas near to and penetrating a lossy
  interface," IEEE Trans. AP-32, 1984 (NEC's Sommerfeld treatment).
- W. C. Chew, *Waves and Fields in Inhomogeneous Media* (1990), ch. 2
  (plane-wave expansion of the half-space dyadic Green's function).
- G. J. Burke, A. J. Poggio, *NEC — Method of Moments* (NEC-2 manual, public
  domain), for the input-deck format.
