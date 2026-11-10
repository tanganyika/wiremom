# wiremom

An open-source thin-wire **method-of-moments** antenna solver in Rust. It uses
the **mixed-potential integral equation (MPIE)**, the formulation NEC-5 moved to,
with triangle basis functions and Galerkin testing, and supports free space, a
perfect ground plane, and real (lossy) ground by the **Sommerfeld** half-space
solution. It is written from the published literature, not from NEC source code.

Everything is pure Rust: `num-complex`, `rayon`, and `complex-bessel` (a Rust
port of Amos' complex Bessel functions). Matrix fill, LU, Sommerfeld tables and
frequency sweeps run on all cores.

It is the engine behind the *wiremom* option in
[Wire Antenna Analyser](../wire-antenna-analyser).

## Features

| Feature | Status |
|---|---|
| Straight wires, any orientation, automatic junctions | ✅ |
| Equal or arbitrary segment lengths per wire (e.g. MMANA-style tapers) | ✅ |
| Voltage sources with phase, at interior nodes, wire joints, junctions, ground connections | ✅ |
| Lumped loads: series/parallel RLC, fixed impedances, open circuits | ✅ |
| Transmission lines (Z0, length, velocity factor, loss, crossed, end shunts) and general two-ports between gaps, NEC TL/NT semantics | ✅ |
| Skin-effect loss (conductivity, relative permeability) and extra series L per metre (insulation) | ✅ |
| Free space, perfect ground (wires may touch it), Sommerfeld real ground | ✅ |
| Wires buried in or crossing lossy ground (radials, ground rods, buried counterpoises) | ✅ |
| A second ground medium for the far field (sea beyond a shore, a marsh, a cliff drop), straight or circular boundary, as NEC-2's GD card, with a Fresnel-zone transition | ✅ |
| Conducting surfaces (triangular RWG patches, e.g. plates, reflectors, car bodies), coupled to wires, over any ground | ✅ |
| Wire-to-surface junctions (a wire ending on a surface vertex feeds current into it) | ✅ |
| Impedance, VSWR, gain (θ/φ parts), radiated power, efficiency, current distribution | ✅ |
| Current elements for an external far-field code | ✅ |
| Parallel frequency sweeps sharing Sommerfeld tables | ✅ |
| NEC deck import/export | ⬜ roadmap |

## Usage

```rust
use num_complex::Complex64 as C;
use wiremom::{C0, Conductor, Ground, LoadKind, Model, RealGround, Surface, Wire};

let f = 14.2e6;
let lam = C0 / f;
let dipole = Wire::new([0.0, -0.24 * lam, 10.0], [0.0, 0.24 * lam, 10.0], 0.001, 40)
    .with_conductor(Conductor::Lossy { sigma: 5.8e7, mu_r: 1.0 });
let mut m = Model::new(vec![dipole], Ground::Real(RealGround::AVERAGE));
m.add_source(0, 20, C::new(1.0, 0.0));            // wire 0, centre node
m.add_load(0, 20, LoadKind::Series { r: 0.0, l: 1e-7, c: None });
let sol = m.solve(f)?;
println!("Z = {}, SWR {:.2}", sol.input_impedance(), sol.vswr(50.0));
println!("gain at 30° elevation: {:.2} dBi", sol.gain_dbi(60f64.to_radians(), 0.0));
let sweep = m.sweep(&[14.0e6, 14.1e6, 14.2e6, 14.3e6]);   // parallel

// a quarter-wave monopole fed against a 1 m square plate (the wire's start is a
// plate vertex, so it connects)
let mut m = Model::new(vec![Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.25 * lam], 0.002, 12)], Ground::Free);
m.surfaces.push(Surface::plate([-0.5, -0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10, 10));
m.add_source(0, 0, C::new(1.0, 0.0));

// a feed line: source on a short "feed wire" (wire 1), 0.3 λ of 75 Ω line to the dipole
use wiremom::{Port, TwoPortKind};
let mut m = Model::new(vec![
    Wire::new([0.0, -0.24 * lam, 10.0], [0.0, 0.24 * lam, 10.0], 0.001, 40),
    Wire::new([0.0, 0.0, 30.0], [0.0, 0.0, 30.05], 0.001, 2),
], Ground::Free);
m.add_source(1, 1, C::new(1.0, 0.0));
m.add_network(Port { wire: 1, node: 1 }, Port { wire: 0, node: 20 }, TwoPortKind::line(75.0, Some(0.3 * lam)));
# Ok::<(), wiremom::Error>(())
```

Wires and nodes are 0-based. Node `0` is a wire's start and node `nseg` its end.
A source or load at a node sits in the *gap* there. At an interior node that is
one basis function; at a wire end that joins other wires it is the gap between
this wire and the junction.

## How it works

1. **Mesh.** Wires are cut into segments, and segment ends within the merge
   tolerance become shared nodes. A node shared by *k* ends carries *k − 1*
   triangle basis functions. On a perfect ground, an end at z = 0 gets a
   half-triangle completed by its image.
2. **Matrix.** `Z_mn = jωμ ∬ Λm Λn (ŝm·ŝn) G + (1/jωε) ∬ Λm′ Λn′ G` with the
   reduced thin-wire kernel. The 1/R singularity is integrated analytically along
   the source segment and the rest by Gauss–Legendre quadrature. Wires of
   different radius use the mean of a², which keeps Z symmetric.
3. **Ground.** Images handle perfect ground. For real ground, the reflected
   field is a Sommerfeld integral over TE/TM Fresnel-reflected plane waves (see
   [src/sommerfeld.rs](src/sommerfeld.rs)). As in NEC-2, the near-singular
   `Γ0 = (ε−1)/(ε+1)` image is handled analytically, and the remainder is
   tabulated per frequency and spline-interpolated. Tables are normalised in kR,
   so a sweep shares one table per frequency bucket (`ground_table_reuse`,
   blended between neighbouring frequencies).

   **Wires in the ground** are handled the same way, from the other side of the
   surface. A wire that crosses z = 0 gets a node on the surface (uniform wires
   are re-cut evenly on each side), and every segment lies in one medium. The
   kernel depends on the media of each pair of segments:

   | Pair | Kernel |
   |---|---|
   | air–air | free space + Γ0 image + tabulated remainder (as above) |
   | ground–ground | free space with the ground's complex wavenumber k√ε and permittivity, a −Γ0 image, and a remainder table built from inside the ground |
   | air–ground | the field transmitted through the surface. Its quasi-static part — the free field with the charge term scaled by 2/(1+ε) — is integrated analytically; the remainder is interpolated from a 3-D table over (ρ, source height, observer depth). Each table row shares one set of Bessel evaluations across all heights and depths |

   Buried currents radiate into the air through the Fresnel transmission
   coefficients, found by reciprocity. A wire that ends on the surface from
   above, without anything continuing below, carries no current into the ground:
   point contacts have a very high spreading resistance, so real installations
   use rods or radials. `Solution::warnings` reports such ends.
4. **Surfaces.** Plates are meshed into triangles. Each interior edge carries an
   RWG function, and edges shared by more than two triangles (plates meeting at
   a seam) get one per extra triangle. The 1/R part of the Green's function over
   a source triangle is integrated analytically (the Wilton–Graglia potential
   formulas), the smooth rest with a 7-point rule. The same formulas couple
   wires to plates.

   A wire ending at a surface vertex gets a **junction** function. Its wire half
   carries the current to the vertex, and on the triangles around the vertex it
   continues as `(r−v)/(Θ|r−v|²)·(1 − ((r−v)·n)²/h²)`: exactly 1 A leaving the
   vertex, no normal current on any edge, bounded divergence. Its 1/|r−v|
   singularity is resolved with subtriangles graded geometrically toward the
   vertex. Images and the Sommerfeld remainder treat surfaces like wires.
5. **Networks.** Transmission lines and other two-ports join gaps, as in NEC.
   The wire structure is solved once per port, for a unit voltage there, which
   gives its admittance seen from the ports. The networks' admittances are added,
   and the small port system is solved with the source voltages fixed. A source
   on a port drives the line and the wire in parallel, and a 0 V source is a short
   (for stubs).
6. **Solve and far field.** Z I = V by parallel LU, then the far field is
   integrated over the currents, adding the image weighted by the Fresnel
   coefficients over real ground.
7. **Second ground medium** (`Model::second_ground`). As NEC-2's GD card, it
   changes only the reflection in the far field; the currents are solved over
   the ground under the antenna. Medium 2 lies beyond a straight line (any
   bearing) or outside a circle, optionally lower by a drop (a cliff). NEC-2
   takes the medium at each ray's specular point, which makes the pattern jump
   where that point crosses the shore. By default wiremom shares the reflection
   between the media with the physical-optics integral across the edge (a
   complex Fresnel integral of the edge's distance from the specular point in
   Fresnel-zone units), per current element. This gives a smooth transition
   while the shore is inside the first Fresnel zone, and NEC-2's switch far from
   it. Low rays from low antennas have zones hundreds of metres long, so a shore
   well beyond the specular point still matters. `fresnel = false` gives NEC-2's
   behaviour.

## Validation

`cargo test --release` runs:

- **Reference values** (`tests/data/reference.json`), frozen from the
  [Python prototype](python/README.md) and from nec2c: 12 antennas in free
  space, on perfect ground and over four soils. The Rust engine matches the
  prototype to 1e-13 (free space, perfect ground) and 1e-5 (real ground), and
  nec2c within 1–3% in impedance and 0.01–0.24 dB in gain.
- **Sommerfeld integrals** at 49 points over 7 soil and water types, matching
  the prototype to 1e-9. With perfect-conductor coefficients they also reproduce
  the closed-form image field.
- **Wires in the ground** (no other code is available to compare against —
  nec2c can't model them — so the checks are exact limits and classic results):
  - with a transparent "ground" (ε = 1, σ = 0), buried, surface-crossing and
    coupled wires reproduce free space exactly;
  - the transmitted field matches the free field through a transparent ground,
    and its quasi-static part matches its closed form;
  - the reflection seen from inside the ground, with perfect-conductor
    coefficients, reproduces the image field with the complex ground wavenumber;
  - two 2 m ground rods at 10 kHz: 89.2 Ω, against 89.8 Ω from Dwight's
    formula;
  - the matrix stays symmetric (1.8e-5) with wires on both sides;
  - buried radials: more radials raise a vertical's efficiency;
  - the ground-side and air-to-ground tables match direct Sommerfeld
    integration (to under 2% and 0.5% point by point, out to grazing angles in
    sea water), and tabulated couplings change a radial system's impedance by
    about 1e-6 (`--example crosscheck`);
  - a quarter-wave vertical over 16 radials in sea water gives 38 Ω at 86%
    efficiency.
- **Surfaces:**
  - the analytic triangle integrals match brute force and a closed form;
  - a lossless dipole or monopole with a plate radiates exactly its input power;
  - the matrix stays symmetric (to 1e-6) with plates and junctions;
  - a dipole over a plate agrees with an equivalent wire-grid plate (1% in
    impedance, 0.2 dB in gain);
  - plate meshes converge (90.55 → 90.27 → 90.22 Ω);
  - a 2 λ plate approaches the infinite ground plane: within 0.4% in impedance
    for a dipole, 5% for a monopole fed against it;
  - a monopole fed against a 1 λ plate converges to 39.9 Ω (wire grids approach
    this slowly from below).
- **Transmission lines:** a λ/2 line repeats the antenna's impedance, a λ/4 line
  inverts it (Z0²/Z), and any length and velocity factor follows the textbook
  formula (to 1%). Shorted λ/4 and λ/8 stubs match theory. A 1 dB line takes 1 dB
  of efficiency. A crossed line flips a pair of dipoles from broadside to
  end-fire. The `TL` card of nec2c agrees within 2.3%.
- **Physics checks:**
  - dipole impedance and 2.15 dBi gain;
  - convergence with segmentation; tapered vs. uniform segmentation;
  - matrix symmetry;
  - monopole = ½ dipole, +3 dB;
  - skin-effect and steel loss vs. analytic values, insulation inductance;
  - loads and traps;
  - sources at wire joints; phased sources;
  - real-ground limits: the perfect-conductor limit, the null at the horizon,
    efficiency below 1;
  - sweep table sharing; current elements vs. the built-in far field;
  - second ground medium: NEC-2 mode matches nec2c's linear cliff (GD card,
    with and without a 5 m drop, horizontal and vertical dipoles) within
    0.1 dB away from the transition angles; a distant boundary changes nothing;
    radius 0 gives medium 2 everywhere; the Fresnel transition is continuous and
    matches NEC-2 far from the shore.

No Python is needed to build, run or test the crate. The prototype in `python/`
is a frozen reference: `python/tools/make_reference.py` regenerates the JSON if
the reference cases ever need to change.

## Speed

`cargo run --release --example bench` times a 3-element Yagi (70 segments). On
12 threads it gives:

| | Solve | Pattern (5,000 directions) | 24-point sweep |
|---|---|---|---|
| Free space | 2 ms | 12 ms | 12 ms |
| 15 m over average ground | 23 ms | 27 ms | 130 ms |

A vertical with a ground rod and 8 buried radials (about 120 segments) solves in
70 ms. With the air-to-ground couplings integrated directly it took 750 ms.

What keeps it fast:

- **Adaptive quadrature:** distant segment pairs get fewer Gauss points. The
  singular part is analytic anyway; this changes impedances by ~1e-9.
- **Far-field samples sized to the segments:** 2 per segment under about λ/8,
  more for longer ones.
- **Ground tables limited to the geometry:** they cover only the distances and
  angles the antenna produces.
- **Fewer points for the ground remainder between distant segments:** 2 per
  segment instead of 6 beyond about three segment lengths (the remainder is
  smooth on the scale of the distance). This halves the fill for large models
  over ground, with no change in the fourth decimal of impedance.
- **Tabulated air-to-ground couplings** for buried wires: one table row per
  horizontal distance, with its Bessel functions shared across all heights and
  depths.
- **Blended tables in sweeps:** a sweep builds tables at frequency buckets 4%
  apart and blends the two around each frequency (error under ~5e-5). A single
  solve builds one exact table.

`cargo run --release --example accuracy` compares the fast settings with full
quadrature, and `--example reuse` the blended tables with exact ones.

## Known limitations

These are the weaker spots, with how far they have been checked.

- **No independent reference for some features.** nec2c can't model wires in the
  ground or thin plates, so those are validated by exact limits, closed forms,
  energy conservation, symmetry and classic results instead:
  - **Wires in the ground:** a transparent ground reproduces free space, and
    Dwight's ground-rod formula matches within 0.7%. Not compared with NEC-4 or
    measurements.
  - **Wire-to-surface junctions:** the plate model converges as the mesh is
    refined, to 39.9 Ω for a λ/4 monopole on a 1 λ plate, and energy is
    conserved. An equivalent wire grid differs by up to ~8%, approaching the
    plate result as the grid is refined. The exact junction impedance is not
    confirmed independently.
- **Point contacts into lossy ground** (a wire stopping on the surface with
  nothing below it) carry no current into the ground. `Solution::warnings`
  reports them; model a rod or radials instead.
- **A wire end on a surface must be at a mesh vertex** to connect.
  `Surface::plate_grid` lets you place grid lines through such points.
- **Second ground medium:** far field only, as in NEC-2. The currents (and so
  the impedance) see only the ground under the antenna. The cliff face itself
  doesn't reflect or diffract. Circles smaller than the Fresnel zone are
  approximate (treated as their tangent line).
- **Not supported:** surfaces in or touching lossy ground, curved surfaces
  (other than as flat triangles), and NEC deck import/export (the application
  using this crate does that).
- **Speed:**
  - buried-wire tables are rebuilt for every frequency of a sweep (not
    blended), and strongly conducting grounds such as sea water need finer
    tables (about 0.3 s for 16 radials);
  - the matrix is dense, so memory grows as N² and the solve as N³ (about 2,000
    unknowns take a few seconds);
  - no frequency interpolation of the matrix yet.
- **Thin-wire limits as usual:** segments much shorter than about two wire
  diameters, or wires closer than a few radii, lose accuracy, as in NEC.
- **Line loss** is given as dB/100 m at the solve frequency; there's no
  frequency-dependent loss model or complex characteristic impedance.

## References

- R. F. Harrington, *Field Computation by Moment Methods* (1968).
- D. R. Wilton, C. M. Butler, IEEE Trans. AP-24, 1976 (thin-wire integral
  equations).
- S. M. Rao, D. R. Wilton, A. W. Glisson, IEEE Trans. AP-30, 1982 (RWG, for
  surfaces).
- K. A. Michalski, J. R. Mosig, IEEE Trans. AP-45, 1997 (layered-media Green's
  functions).
- G. J. Burke, E. K. Miller, IEEE Trans. AP-32, 1984 (Sommerfeld treatment in
  NEC).
- W. C. Chew, *Waves and Fields in Inhomogeneous Media* (1990), ch. 2.
- D. E. Amos, ACM TOMS 12, 1986 (complex Bessel functions).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
