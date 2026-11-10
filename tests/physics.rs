//! Physics checks that need no reference data.

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::sommerfeld::{Table, table_range};
use wiremom::{C0, Conductor, Ground, LoadKind, Model, RealGround, Segments, Wire};

const F: f64 = 299.792458e6; // wavelength 1 m
const ONE: C = C::new(1.0, 0.0);

fn dipole(len: f64, a: f64, nseg: usize, ground: Ground) -> Model {
    let mut m = Model::new(vec![Wire::new([0.0, 0.0, -len / 2.0], [0.0, 0.0, len / 2.0], a, nseg)], ground);
    m.add_source(0, nseg / 2, ONE);
    m
}

fn db(x: f64) -> f64 {
    10.0 * x.log10()
}

#[test]
fn half_wave_dipole() {
    let s = dipole(0.5, 0.001, 40, Ground::Free).solve(F).unwrap();
    let z = s.input_impedance();
    assert!(z.re > 82.0 && z.re < 89.0 && z.im > 40.0 && z.im < 50.0, "{z}");
    assert!((s.gain_dbi(PI / 2.0, 0.0) - 2.15).abs() < 0.05);
    assert!((s.efficiency() - 1.0).abs() < 2e-3);
}

#[test]
fn converges_with_segmentation() {
    let z: Vec<C> = [20, 40, 80].iter().map(|&n| dipole(0.5, 0.001, n, Ground::Free).solve(F).unwrap().input_impedance()).collect();
    assert!((z[2] - z[1]).norm() < (z[1] - z[0]).norm());
    assert!((z[2].re - z[1].re).abs() < 1.0);
}

#[test]
fn tapered_segments_agree_with_uniform() {
    // MMANA-style taper: short segments at the ends, long in the middle
    let mut lens = vec![];
    for i in 0..6 {
        lens.push(2f64.powi(i));
    }
    lens.extend(vec![32.0; 8]);
    for i in (0..6).rev() {
        lens.push(2f64.powi(i));
    }
    let n = lens.len();
    let w = Wire { segments: Segments::Lengths(lens), ..Wire::new([0.0, 0.0, -0.25], [0.0, 0.0, 0.25], 0.001, 1) };
    let mut m = Model::new(vec![w], Ground::Free);
    m.add_source(0, n / 2, ONE);
    let zt = m.solve(F).unwrap().input_impedance();
    let zu = dipole(0.5, 0.001, 160, Ground::Free).solve(F).unwrap().input_impedance();
    assert!((zt - zu).norm() / zu.norm() < 0.03, "tapered {zt} uniform {zu}");
}

#[test]
fn matrix_is_symmetric() {
    let w = vec![
        Wire::new([0.0, 0.0, 0.0], [0.0, 0.3, 0.1], 0.002, 9),
        Wire::new([0.0, 0.3, 0.1], [0.2, 0.3, 0.4], 0.001, 7),
        Wire::new([0.0, 0.3, 0.1], [-0.3, 0.2, 0.1], 0.003, 5),
    ];
    for (n, tol) in [(8, 5e-4), (16, 1e-5)] {
        let mut m = Model::new(w.clone(), Ground::Free);
        m.n_outer = n;
        m.n_inner = n;
        let mesh = m.mesh().unwrap();
        let z = m.impedance_matrix(&mesh, F).unwrap();
        let nb = mesh.n_unknowns();
        let zmax = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for i in 0..nb {
            for j in 0..nb {
                assert!((z[i * nb + j] - z[j * nb + i]).norm() < tol * zmax);
            }
        }
    }
}

#[test]
fn monopole_is_half_a_dipole() {
    let mut mono = Model::new(vec![Wire::new([0.0; 3], [0.0, 0.0, 0.24], 0.005, 20)], Ground::Perfect);
    mono.add_source(0, 0, ONE);
    let sm = mono.solve(F).unwrap();
    let sd = dipole(0.48, 0.005, 40, Ground::Free).solve(F).unwrap();
    assert!((sm.input_impedance() - sd.input_impedance() / 2.0).norm() < 1e-6 * sd.input_impedance().norm());
    let dg = sm.gain_dbi(PI / 2.0, 0.0) - sd.gain_dbi(PI / 2.0, 0.0);
    assert!((dg - db(2.0)).abs() < 0.02, "{dg}");
    assert!((sm.efficiency() - 1.0).abs() < 3e-3);
}

#[test]
fn skin_effect_loss() {
    let a = 1e-4;
    let lossless = dipole(0.5, a, 40, Ground::Free).solve(F).unwrap();
    let mut m = dipole(0.5, a, 40, Ground::Free);
    m.wires[0].conductor = Conductor::Lossy { sigma: 5.8e7, mu_r: 1.0 };
    let cu = m.solve(F).unwrap();
    let rprime = (2.0 * PI * F * wiremom::MU0 / (2.0 * 5.8e7)).sqrt() / (2.0 * PI * a);
    let added = cu.input_impedance().re - lossless.input_impedance().re;
    // sinusoidal current on a half-wave dipole: R_loss ~ R' L / 2
    assert!((added / (rprime * 0.25) - 1.0).abs() < 0.1, "{added}");
    // steel: R' grows with sqrt(mu_r)
    m.wires[0].conductor = Conductor::Lossy { sigma: 5.8e7, mu_r: 100.0 };
    let fe = m.solve(F).unwrap();
    let added_fe = fe.input_impedance().re - lossless.input_impedance().re;
    assert!((added_fe / added / 10.0 - 1.0).abs() < 0.05, "{added_fe} {added}");
}

#[test]
fn series_inductance_per_metre_adds_reactance() {
    // a uniform series L' adds roughly jwL'·L/2 at a half-wave dipole's feed
    let base = dipole(0.5, 0.001, 40, Ground::Free).solve(F).unwrap().input_impedance();
    let mut m = dipole(0.5, 0.001, 40, Ground::Free);
    m.wires[0].series_l_per_m = 1e-8;
    let z = m.solve(F).unwrap().input_impedance();
    let expect = 2.0 * PI * F * 1e-8 * 0.25;
    assert!(((z - base).im / expect - 1.0).abs() < 0.25, "{}", (z - base).im);
}

#[test]
fn series_load_adds_at_the_feed() {
    let s0 = dipole(0.5, 0.001, 40, Ground::Free).solve(F).unwrap();
    let mut m = dipole(0.5, 0.001, 40, Ground::Free);
    m.add_load(0, 20, LoadKind::Series { r: 10.0, l: 1e-8, c: None });
    let s1 = m.solve(F).unwrap();
    let expect = C::new(10.0, 2.0 * PI * F * 1e-8);
    assert!((s1.input_impedance() - s0.input_impedance() - expect).norm() < 1e-6 * expect.norm());
}

#[test]
fn resonant_trap_isolates_outer_wires() {
    let f = 14.2e6;
    let lam = C0 / f;
    let l = 2e-6;
    let c = 1.0 / ((2.0 * PI * f).powi(2) * l);
    let w = vec![
        Wire::new([0.0, -0.12 * lam, 0.0], [0.0, 0.12 * lam, 0.0], 0.001, 24),
        Wire::new([0.0, 0.12 * lam, 0.0], [0.0, 0.20 * lam, 0.0], 0.001, 8),
        Wire::new([0.0, -0.12 * lam, 0.0], [0.0, -0.20 * lam, 0.0], 0.001, 8),
    ];
    let mut trap = Model::new(w, Ground::Free);
    trap.add_source(0, 12, ONE);
    for wi in [1, 2] {
        trap.add_load(wi, 4, LoadKind::Parallel { r: 0.0, l, c });
    }
    let mut alone = Model::new(vec![Wire::new([0.0, -0.16 * lam, 0.0], [0.0, 0.16 * lam, 0.0], 0.001, 32)], Ground::Free);
    alone.add_source(0, 16, ONE);
    let (zt, za) = (trap.solve(f).unwrap().input_impedance(), alone.solve(f).unwrap().input_impedance());
    assert!((zt - za).norm() / za.norm() < 0.1, "{zt} {za}");
}

#[test]
fn source_at_a_wire_joint() {
    let zref = dipole(0.5, 0.001, 40, Ground::Free).solve(F).unwrap().input_impedance();
    for reverse in [false, true] {
        let w2 = if reverse {
            Wire::new([0.0, 0.0, 0.25], [0.0, 0.0, 0.0], 0.001, 20)
        } else {
            Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.25], 0.001, 20)
        };
        let w = vec![Wire::new([0.0, 0.0, -0.25], [0.0, 0.0, 0.0], 0.001, 20), w2];
        for (wire, node) in [(0, 20), (1, if reverse { 20 } else { 0 })] {
            let mut m = Model::new(w.clone(), Ground::Free);
            m.add_source(wire, node, ONE);
            let z = m.solve(F).unwrap().input_impedance();
            assert!((z - zref).norm() < 1e-9 * zref.norm(), "reverse={reverse} wire={wire}: {z} {zref}");
        }
    }
}

#[test]
fn phased_pair_is_symmetric() {
    // two parallel dipoles fed in phase have equal impedances; in antiphase too
    let w = vec![Wire::new([0.0, 0.0, -0.25], [0.0, 0.0, 0.25], 0.001, 20), Wire::new([0.25, 0.0, -0.25], [0.25, 0.0, 0.25], 0.001, 20)];
    for v2 in [ONE, -ONE, C::from_polar(1.0, PI / 2.0)] {
        let mut m = Model::new(w.clone(), Ground::Free);
        m.add_source(0, 10, ONE);
        m.add_source(1, 10, v2);
        let z = m.solve(F).unwrap().source_impedances();
        if v2.im == 0.0 {
            assert!((z[0] - z[1]).norm() < 1e-9 * z[0].norm());
        } else {
            // 90 degrees: Z1 = Z11 + j Z12, Z2 = Z11 - j Z12 (mutual coupling)
            assert!((z[0] - z[1]).norm() > 1.0);
        }
    }
}

// ---------------------------------------------------------------- real ground

fn over(ground: Ground, height: f64, vertical: bool) -> Model {
    let lam = C0 / 14.2e6;
    let w = if vertical {
        Wire::new([0.0, 0.0, height], [0.0, 0.0, height + 0.48 * lam], 0.005, 30)
    } else {
        Wire::new([0.0, -0.24 * lam, height], [0.0, 0.24 * lam, height], 0.002, 30)
    };
    let mut m = Model::new(vec![w], ground);
    m.add_source(0, 15, ONE);
    m
}

#[test]
fn tabulated_pec_image_matches_closed_form() {
    let f = 14.2e6;
    let lam = C0 / f;
    for (vertical, h) in [(false, 0.25 * lam), (true, 0.05 * lam)] {
        let zp = over(Ground::Perfect, h, vertical).solve(f).unwrap().input_impedance();
        let m = over(Ground::Real(RealGround::AVERAGE), h, vertical);
        let mesh = m.mesh().unwrap();
        let (r0, r1) = table_range(&mesh, m.n_ground).unwrap();
        let t = Table::new(RealGround::AVERAGE, f, r0, r1, 1.0, Some((-ONE, ONE)));
        let mut z = m.impedance_matrix_with(&mesh, f, Some(&t)).unwrap();
        let nb = mesh.n_unknowns();
        let mut v = vec![C::new(0.0, 0.0); nb];
        let (b, _) = mesh.gap(0, 15).unwrap()[0];
        v[b] = ONE;
        wiremom::lu::solve(&mut z, &mut v, nb).unwrap();
        let zt = 1.0 / v[b];
        assert!((zt - zp).norm() / zp.norm() < 1e-4, "{zt} {zp}");
    }
}

#[test]
fn conducting_ground_approaches_pec() {
    let f = 14.2e6;
    let zp = over(Ground::Perfect, 1.0, true).solve(f).unwrap().input_impedance();
    let z5 = over(Ground::Real(RealGround { eps_r: 81.0, sigma: 5.0 }), 1.0, true).solve(f).unwrap().input_impedance();
    let z1k = over(Ground::Real(RealGround { eps_r: 81.0, sigma: 1e3 }), 1.0, true).solve(f).unwrap().input_impedance();
    assert!((z1k - zp).norm() < (z5 - zp).norm() + 1e-9);
    assert!((z1k - zp).norm() / zp.norm() < 2e-3);
}

#[test]
fn real_ground_far_field() {
    let f = 14.2e6;
    let s = over(Ground::Real(RealGround::AVERAGE), 1.0, true).solve(f).unwrap();
    let g: Vec<f64> = [89.99f64, 80.0].iter().map(|t| s.gain_dbi(t.to_radians(), 0.0)).collect();
    assert!(g[0] < g[1] - 20.0, "{g:?}");
    assert!(s.efficiency() < 0.9);
    let sp = over(Ground::Perfect, 1.0, true).solve(f).unwrap();
    assert!((sp.efficiency() - 1.0).abs() < 5e-3);
    let (gte, gtm) = RealGround { eps_r: 1.0, sigma: 1e12 }.reflection(f, 0.7);
    assert!((gte + ONE).norm() < 1e-4 && (gtm - ONE).norm() < 1e-4);
}

#[test]
fn sweep_shares_tables_accurately() {
    let freqs: Vec<f64> = (0..8).map(|i| 14.0e6 + i as f64 * 0.05e6).collect();
    let m = over(Ground::Real(RealGround::AVERAGE), 8.0, false);
    let shared: Vec<C> = m.sweep(&freqs).into_iter().map(|s| s.unwrap().input_impedance()).collect();
    let mut exact = m.clone();
    exact.ground_table_reuse = 0.0;
    exact.tables = Default::default();
    for (f, z) in freqs.iter().zip(&shared) {
        let ze = exact.solve(*f).unwrap().input_impedance();
        assert!((z - ze).norm() / ze.norm() < 2e-3, "{f}: {z} {ze}");
    }
}

#[test]
fn wires_must_be_above_real_ground() {
    let mut m = Model::new(vec![Wire::new([0.0; 3], [0.0, 0.0, 5.0], 0.002, 10)], Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 5, ONE);
    assert!(m.solve(14.2e6).is_err());
}

#[test]
fn current_elements_reproduce_the_far_field() {
    // what an external far-field code sees: sum the elements and their image
    let f = 14.2e6;
    let s = over(Ground::Perfect, 5.0, false).solve(f).unwrap();
    let k = 2.0 * PI * f / C0;
    let (th, ph) = (0.6f64, 0.3f64);
    let r = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
    let mut n = [C::new(0.0, 0.0); 3];
    for (p, c) in s.current_elements(6) {
        let d = C::from_polar(1.0, k * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2]));
        let i = C::from_polar(1.0, k * (r[0] * p[0] + r[1] * p[1] - r[2] * p[2]));
        n[0] += c[0] * (d - i);
        n[1] += c[1] * (d - i);
        n[2] += c[2] * (d + i);
    }
    let phat = [-ph.sin(), ph.cos(), 0.0];
    let e_ph = C::new(0.0, -2.0 * PI * f * wiremom::MU0 / (4.0 * PI)) * (n[0] * phat[0] + n[1] * phat[1]);
    let (_, ep) = s.far_field(th, ph);
    assert!((e_ph - ep).norm() < 1e-9 * ep.norm());
}
