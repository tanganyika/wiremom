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
| Skin-effect loss (conductivity, relative permeability) and extra series L per metre (insulation) | ✅ |
| Free space, perfect ground (wires may touch it), Sommerfeld real ground | ✅ |
| Wires buried in or crossing lossy ground (radials, ground rods, buried counterpoises) | ✅ |
| Impedance, VSWR, gain (θ/φ parts), radiated power, efficiency, current distribution | ✅ |
| Current elements for an external far-field code | ✅ |
| Parallel frequency sweeps sharing Sommerfeld tables | ✅ |
| NEC deck import/export | ⬜ roadmap |
| Transmission-line and network elements | ⬜ roadmap |
| Surface patches and wire-to-patch junctions | ⬜ roadmap |

## Usage

```rust
use num_complex::Complex64 as C;
use wiremom::{C0, Conductor, Ground, LoadKind, Model, RealGround, Wire};

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
   ±1–2%).

   **Wires in the ground** are handled the same way, from the other side of the
   surface. A wire that crosses z = 0 gets a node on the surface (uniform wires
   are re-cut evenly on each side), and every segment lies in one medium. The
   kernel depends on the media of each pair of segments:

   | Pair | Kernel |
   |---|---|
   | air–air | free space + Γ0 image + tabulated remainder (as above) |
   | ground–ground | free space with the ground's complex wavenumber k√ε and permittivity, a −Γ0 image, and a remainder table built from inside the ground |
   | air–ground | the field transmitted through the surface. Its quasi-static part — the free field with the charge term scaled by 2/(1+ε) — is integrated analytically; the remainder is evaluated directly as a Sommerfeld integral |

   Buried currents radiate into the air through the Fresnel transmission
   coefficients, found by reciprocity. A wire that ends on the surface from
   above, without anything continuing below, carries no current into the ground:
   point contacts have a very high spreading resistance, so real installations
   use rods or radials. `Solution::warnings` reports such ends.
4. **Solve and far field.** Z I = V by parallel LU, then the far field is
   integrated over the currents, adding the image weighted by the Fresnel
   coefficients over real ground.

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
  - buried radials: more radials raise a vertical's efficiency.
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
  - sweep table sharing; current elements vs. the built-in far field.

No Python is needed to build, run or test the crate. The prototype in `python/`
is a frozen reference: `python/tools/make_reference.py` regenerates the JSON if
the reference cases ever need to change.

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
