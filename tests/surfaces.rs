//! Conducting plates (RWG patches).

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use std::time::Instant;
use wiremom::{Ground, Model, RealGround, Surface, Wire};

const F: f64 = 299.792458e6; // wavelength 1 m
const ONE: C = C::new(1.0, 0.0);

fn dipole(h: f64) -> Wire {
    Wire::new([0.0, -0.235, h], [0.0, 0.235, h], 0.002, 20)
}

/// Dipole along y at height h over a square plate of side s in z = 0, n x n cells.
fn with_plate(h: f64, s: f64, n: usize) -> Model {
    let mut m = Model::new(vec![dipole(h)], Ground::Free);
    m.surfaces.push(Surface::plate([-s / 2.0, -s / 2.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0], n, n));
    m.add_source(0, 10, ONE);
    m
}

/// The same plate as a wire grid (spacing d, equal-area radius d / 2 pi).
fn with_grid(h: f64, s: f64, n: usize) -> Model {
    let d = s / n as f64;
    let a = d / (2.0 * PI);
    let mut w = vec![dipole(h)];
    for i in 0..=n {
        let c = -s / 2.0 + i as f64 * d;
        w.push(Wire::new([c, -s / 2.0, 0.0], [c, s / 2.0, 0.0], a, n));
        w.push(Wire::new([-s / 2.0, c, 0.0], [s / 2.0, c, 0.0], a, n));
    }
    let mut m = Model::new(w, Ground::Free);
    m.add_source(0, 10, ONE);
    m
}

#[test]
fn energy_is_conserved_with_a_lossless_plate() {
    let t = Instant::now();
    let s = with_plate(0.25, 1.0, 10).solve(F).unwrap();
    let e = s.efficiency();
    println!("dipole over 1x1 plate: Z = {:.2}, efficiency {e:.4}, {} unknowns, {:.0} ms", s.input_impedance(), s.mesh.n_unknowns(), t.elapsed().as_secs_f64() * 1e3);
    assert!((e - 1.0).abs() < 0.01, "{e}");
}

#[test]
fn matrix_is_symmetric_with_plates() {
    let m = with_plate(0.2, 0.6, 5);
    let mesh = m.mesh().unwrap();
    let z = m.impedance_matrix(&mesh, F).unwrap();
    let n = mesh.n_unknowns();
    let zmax = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let worst = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (z[i * n + j] - z[j * n + i]).norm()).fold(0.0, f64::max) / zmax;
    println!("max asymmetry {worst:.2e}");
    assert!(worst < 1e-3, "{worst}");
}

#[test]
fn plate_agrees_with_an_equivalent_wire_grid() {
    let sp = with_plate(0.25, 1.0, 10).solve(F).unwrap();
    let sg = with_grid(0.25, 1.0, 10).solve(F).unwrap();
    let (zp, zg) = (sp.input_impedance(), sg.input_impedance());
    let (gp, gg) = (sp.gain_dbi(0.0, 0.0), sg.gain_dbi(0.0, 0.0));
    let (bp, bg) = (sp.gain_dbi(PI, 0.0), sg.gain_dbi(PI, 0.0));
    println!("plate: Z = {zp:.2}, fwd {gp:.2} dBi, back {bp:.2}; wire grid: Z = {zg:.2}, fwd {gg:.2}, back {bg:.2}");
    assert!((zp - zg).norm() < 0.06 * zg.norm(), "{zp} vs {zg}");
    assert!((gp - gg).abs() < 0.4, "{gp} vs {gg}");
}

#[test]
fn refining_the_plate_mesh_converges() {
    let z: Vec<C> = [6, 10, 14].iter().map(|&n| with_plate(0.25, 1.0, n).solve(F).unwrap().input_impedance()).collect();
    println!("6x6 {:.2}, 10x10 {:.2}, 14x14 {:.2}", z[0], z[1], z[2]);
    assert!((z[2] - z[1]).norm() < (z[1] - z[0]).norm() + 0.5);
    assert!((z[2] - z[1]).norm() < 0.03 * z[2].norm());
}

#[test]
fn large_plate_approaches_a_ground_plane() {
    // a dipole a quarter wave over a 2 x 2 wavelength plate vs an infinite plane
    let sp = with_plate(0.25, 2.0, 16).solve(F).unwrap();
    let mut img = Model::new(vec![dipole(0.25)], Ground::Perfect);
    img.add_source(0, 10, ONE);
    let si = img.solve(F).unwrap();
    let (zp, zi) = (sp.input_impedance(), si.input_impedance());
    let (gp, gi) = (sp.gain_dbi(0.0, 0.0), si.gain_dbi(0.0, 0.0));
    println!("2x2 plate: Z = {zp:.2}, zenith {gp:.2} dBi; infinite plane: Z = {zi:.2}, {gi:.2} dBi");
    assert!((zp - zi).norm() < 0.1 * zi.norm());
    assert!((gp - gi).abs() < 1.0);
}

#[test]
fn plates_over_perfect_and_real_ground() {
    // a horizontal plate (a "ground screen") under a vertical dipole, both above ground
    let mk = |g: Ground| {
        let mut m = Model::new(vec![Wire::new([0.3, 0.0, 0.5], [0.3, 0.0, 0.97], 0.002, 16)], g);
        m.surfaces.push(Surface::plate([-0.4, -0.4, 0.3], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0], 8, 8));
        m.add_source(0, 8, ONE);
        m.solve(F).unwrap()
    };
    let sp = mk(Ground::Perfect);
    let sr = mk(Ground::Real(RealGround::AVERAGE));
    let (ep, er) = (sp.efficiency(), sr.efficiency());
    println!("perfect ground: Z = {:.2}, eff {ep:.3}; average ground: Z = {:.2}, eff {er:.3}", sp.input_impedance(), sr.input_impedance());
    assert!((ep - 1.0).abs() < 0.02, "{ep}");
    assert!(er < 0.95 && er > 0.3, "{er}");
}

/// Quarter-wave monopole at the centre of a square plate (side s, n x n cells, n
/// even so the centre is a mesh vertex), fed at its base.
fn monopole_on_plate(s: f64, n: usize) -> Model {
    let mut m = Model::new(vec![Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.24], 0.002, 12)], Ground::Free);
    m.surfaces.push(Surface::plate([-s / 2.0, -s / 2.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0], n, n));
    m.add_source(0, 0, ONE);
    m
}

fn monopole_on_grid(s: f64, n: usize) -> Model {
    let d = s / n as f64;
    let a = d / (2.0 * PI);
    let mut w = vec![Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.24], 0.002, 12)];
    for i in 0..=n {
        let c = -s / 2.0 + i as f64 * d;
        w.push(Wire::new([c, -s / 2.0, 0.0], [c, s / 2.0, 0.0], a, n));
        w.push(Wire::new([-s / 2.0, c, 0.0], [s / 2.0, c, 0.0], a, n));
    }
    let mut m = Model::new(w, Ground::Free);
    m.add_source(0, 0, ONE);
    m
}

#[test]
fn monopole_fed_against_a_plate() {
    // the plate model converges smoothly with the mesh (41.7, 40.6, 40.2, 40.0, 39.9
    // ohm for 6..26 cells a side); wire grids approach it slowly from below and
    // depend on the grid-wire radius at the feed junction, so they're compared loosely
    let z: Vec<C> = [6, 10, 14].iter().map(|&n| monopole_on_plate(1.0, n).solve(F).unwrap().input_impedance()).collect();
    assert!((z[2] - z[1]).norm() < (z[1] - z[0]).norm());
    let t = Instant::now();
    let sp = monopole_on_plate(1.0, 10).solve(F).unwrap();
    let tp = t.elapsed().as_secs_f64() * 1e3;
    let sg = monopole_on_grid(1.0, 20).solve(F).unwrap();
    let (zp, zg) = (sp.input_impedance(), sg.input_impedance());
    let (ep, eg) = (sp.efficiency(), sg.efficiency());
    let (gp, gg) = (sp.gain_dbi(PI / 3.0, 0.0), sg.gain_dbi(PI / 3.0, 0.0));
    println!("monopole on 1x1 plate: Z = {zp:.2} (eff {ep:.4}, {gp:.2} dBi at 30 deg el, {tp:.0} ms); on 20x20 wire grid: Z = {zg:.2} (eff {eg:.4}, {gg:.2} dBi)");
    assert!((ep - 1.0).abs() < 0.02, "plate efficiency {ep}");
    assert!((zp - zg).norm() < 0.10 * zg.norm(), "{zp} vs {zg}");
    assert!((gp - gg).abs() < 0.5, "{gp} vs {gg}");
}

#[test]
fn monopole_on_a_large_plate_approaches_a_ground_plane() {
    let sp = monopole_on_plate(2.0, 16).solve(F).unwrap();
    let mut g = Model::new(vec![Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.24], 0.002, 12)], Ground::Perfect);
    g.add_source(0, 0, ONE);
    let sg = g.solve(F).unwrap();
    println!("monopole on 2x2 plate: Z = {:.2}; on a ground plane: {:.2}", sp.input_impedance(), sg.input_impedance());
    assert!((sp.input_impedance() - sg.input_impedance()).norm() < 0.12 * sg.input_impedance().norm());
}

#[test]
fn junction_matrix_stays_symmetric() {
    let m = monopole_on_plate(0.6, 4);
    let mesh = m.mesh().unwrap();
    assert_eq!(mesh.junctions.len(), 1);
    let z = m.impedance_matrix(&mesh, F).unwrap();
    let n = mesh.n_unknowns();
    let zmax = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let worst = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (z[i * n + j] - z[j * n + i]).norm()).fold(0.0, f64::max) / zmax;
    println!("max asymmetry {worst:.2e}");
    assert!(worst < 2e-3, "{worst}");
}
