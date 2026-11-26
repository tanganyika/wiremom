//! Sloping ground: a tilted perfect ground against explicit images in free space,
//! and a tilted real ground against the same antenna turned by hand over flat
//! ground.

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::vec3::*;
use wiremom::{Ground, Model, RealGround, Slope, Wire};

const F: f64 = 14.2e6;

fn slope() -> Slope {
    Slope::new(20f64.to_radians(), 30f64.to_radians())
}

/// A slanted dipole, not lined up with anything, above the sloping plane.
fn dipole() -> Wire {
    Wire::new([1.0, -3.0, 6.0], [2.0, 3.5, 8.5], 0.002, 21)
}

fn mirror(p: V3, n: V3) -> V3 {
    sub(p, scale(n, 2.0 * dot(p, n)))
}

fn dir(theta: f64, phi: f64) -> V3 {
    [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
}

fn angles(r: V3) -> (f64, f64) {
    (r[2].clamp(-1.0, 1.0).acos(), r[1].atan2(r[0]))
}

#[test]
fn perfect_slope_matches_explicit_image() {
    let sl = slope();
    let n = sl.normal();
    let mut m = Model::new(vec![dipole()], Ground::Perfect);
    m.slope = Some(sl);
    m.add_source(0, 11, C::new(1.0, 0.0));
    let s = m.solve(F).unwrap();

    // free space: the dipole and its image (mirrored, current reversed)
    let w = dipole();
    let img = Wire::new(mirror(w.start, n), mirror(w.end, n), w.radius, 21);
    let mut fs = Model::new(vec![w, img], Ground::Free);
    fs.add_source(0, 11, C::new(1.0, 0.0));
    fs.add_source(1, 11, C::new(-1.0, 0.0));
    let f = fs.solve(F).unwrap();

    let (z, zf) = (s.input_impedance(), f.source_impedances()[0]);
    assert!((z - zf).norm() / zf.norm() < 1e-6, "{z} vs {zf}");
    // above the plane the pair's field (half the input power each) is the field
    // over ground; below it nothing
    let mut above = 0;
    for th in (0..=180).step_by(15) {
        for ph in (0..360).step_by(30) {
            let (t, p) = ((th as f64).to_radians(), (ph as f64).to_radians());
            let (g, gf) = (s.gain(t, p), f.gain(t, p));
            let up = dot(dir(t, p), n);
            if up.abs() < 1e-9 {
                continue; // along the surface
            }
            if up > 0.0 {
                above += 1;
                for (a, b) in [(g.total, gf.total), (g.theta, gf.theta), (g.phi, gf.phi)] {
                    assert!((a - 2.0 * b).abs() < 1e-5 * (1.0 + 2.0 * gf.total), "theta {th} phi {ph}: {a} vs {}", 2.0 * b);
                }
            } else {
                assert!(g.total == 0.0, "theta {th} phi {ph} is under the ground");
            }
        }
    }
    assert!(above > 70, "{above}");
    // downhill the pattern reaches below the horizon
    assert!(s.gain(100f64.to_radians(), sl.downhill).total > 0.0);
    // lossless over a perfect ground: everything is radiated
    let eff = s.efficiency();
    assert!((eff - 1.0).abs() < 2e-3, "{eff}");
    // currents come back in world coordinates, on the wire as built
    let (p0, _, _) = s.segment_currents()[0];
    assert!(norm(sub(p0, dipole().start)) < 1e-9);
}

#[test]
fn real_slope_matches_the_antenna_turned_by_hand() {
    let sl = slope();
    let fr = sl.frame();
    let mut m = Model::new(vec![dipole()], Ground::Real(RealGround::AVERAGE));
    m.slope = Some(sl);
    m.add_source(0, 11, C::new(1.0, 0.0));
    let s = m.solve(F).unwrap();

    let w = dipole();
    let turned = Wire::new(fr.to_ground(w.start), fr.to_ground(w.end), w.radius, 21);
    let mut flat = Model::new(vec![turned], Ground::Real(RealGround::AVERAGE));
    flat.add_source(0, 11, C::new(1.0, 0.0));
    let f = flat.solve(F).unwrap();

    assert!((s.input_impedance() - f.input_impedance()).norm() < 1e-9);
    for th in (0..=120).step_by(10) {
        for ph in (0..360).step_by(45) {
            let (t, p) = ((th as f64).to_radians(), (ph as f64).to_radians());
            let (tl, pl) = angles(fr.to_ground(dir(t, p)));
            let (a, b) = (s.gain(t, p).total, f.gain(tl, pl).total);
            assert!((a - b).abs() < 1e-9 * (1.0 + b), "theta {th} phi {ph}: {a} vs {b}");
        }
    }
    assert!((s.efficiency() - f.efficiency()).abs() < 1e-9);
}

#[test]
fn zero_slope_is_flat_ground() {
    let mut m = Model::new(vec![dipole()], Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 11, C::new(1.0, 0.0));
    let flat = m.solve(F).unwrap();
    m.slope = Some(Slope::new(0.0, 1.0));
    let s = m.solve(F).unwrap();
    assert!((s.input_impedance() - flat.input_impedance()).norm() < 1e-9);
    assert!((s.gain_dbi(1.0, 0.3) - flat.gain_dbi(1.0, 0.3)).abs() < 1e-9);
}

#[test]
fn wire_under_the_slope_is_in_the_ground() {
    // 2 m up at the origin, but 20 m uphill the ground is ~7 m up
    let sl = Slope::new(20f64.to_radians(), PI);
    let w = Wire::new([18.0, -1.0, 2.0], [22.0, 1.0, 2.0], 0.002, 9);
    assert!(sl.surface_z(20.0, 0.0) > 7.0);
    // under a perfect ground it is refused, in real ground it is buried
    let mut m = Model::new(vec![w], Ground::Perfect);
    m.slope = Some(sl);
    m.add_source(0, 4, C::new(1.0, 0.0));
    assert!(m.solve(F).is_err());
    m.ground = Ground::Real(RealGround::AVERAGE);
    let s = m.solve(F).unwrap();
    assert!(s.mesh.medium.iter().all(|&md| md == wiremom::mesh::GROUND));
}
