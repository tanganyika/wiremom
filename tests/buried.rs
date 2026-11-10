//! Wires in and through lossy ground.  No external code models these (nec2c can't),
//! so the checks are exact limits, closed forms and a classic low-frequency formula.

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::sommerfeld::{Coeffs, Side, reflected_dyadic_side, transmitted_dyadic};
use wiremom::{C0, Ground, MU0, Model, RealGround, Wire};

const ONE: C = C::new(1.0, 0.0);
const J: C = C::new(0.0, 1.0);

/// Free-space field (rr, ff, rz, zr, zz in local axes) at offset (rho, 0, dz) from a
/// unit current element, for wavenumber `k` (may be complex); (A part, full).
fn free_field(rho: f64, dz: f64, freq: f64, k: C) -> ([C; 5], [C; 5]) {
    let w = 2.0 * PI * freq;
    let r = (rho * rho + dz * dz).sqrt();
    let (ux, uz) = (rho / r, dz / r);
    let kr = k * r;
    let g = (-J * kr).exp() / (4.0 * PI * r);
    let a = 1.0 - J / kr - 1.0 / (kr * kr);
    let b = -1.0 + 3.0 * J / kr + 3.0 / (kr * kr);
    let p = -J * w * MU0 * g;
    let full = [p * (a + b * ux * ux), p * a, p * b * ux * uz, p * b * uz * ux, p * (a + b * uz * uz)];
    let apart = [p, p, C::new(0.0, 0.0), C::new(0.0, 0.0), p];
    (apart, full)
}

fn close(a: &[C; 5], b: &[C; 5], tol: f64) -> bool {
    let scale = b.iter().map(|v| v.norm()).fold(0.0, f64::max);
    a.iter().zip(b).all(|(x, y)| (x - y).norm() <= tol * scale)
}

#[test]
fn transmitted_field_through_a_transparent_ground_is_the_free_field() {
    let f = 14.2e6;
    let k = C::new(2.0 * PI * f / C0, 0.0);
    for (rho, zs, d) in [(0.0, 2.0, 1.0), (3.0, 1.0, 0.5), (12.0, 0.3, 0.2), (0.5, 4.0, 0.05)] {
        let t = transmitted_dyadic(rho, zs, d, f, ONE, false);
        // observer below the source: dz = -(zs + d)
        let (_, free) = free_field(rho, -(zs + d), f, k);
        assert!(close(&t, &free, 1e-7), "rho={rho} zs={zs} d={d}: {t:?} vs {free:?}");
    }
}

#[test]
fn transmitted_quasi_static_part_matches_its_closed_form() {
    // what the solver integrates analytically: free field with the charge part scaled
    let f = 14.2e6;
    let k = C::new(2.0 * PI * f / C0, 0.0);
    let eps = RealGround::AVERAGE.eps_c(f);
    let c = 2.0 / (1.0 + eps);
    for (rho, zs, d) in [(0.4, 1.0, 0.3), (5.0, 0.5, 0.5), (1.0, 2.0, 0.1)] {
        let full = transmitted_dyadic(rho, zs, d, f, eps, false);
        let rem = transmitted_dyadic(rho, zs, d, f, eps, true);
        let sub: [C; 5] = [0, 1, 2, 3, 4].map(|i| full[i] - rem[i]);
        let (a, fr) = free_field(rho, -(zs + d), f, k);
        let want: [C; 5] = [0, 1, 2, 3, 4].map(|i| a[i] + c * (fr[i] - a[i]));
        assert!(close(&sub, &want, 1e-7), "rho={rho}: {sub:?} vs {want:?}");
    }
}

#[test]
fn ground_side_reflection_with_pec_coefficients_is_the_image() {
    // seen from inside the ground, the same formulas with the ground's wavenumber
    let f = 14.2e6;
    let eps = RealGround::AVERAGE.eps_c(f);
    let k2 = C::new(2.0 * PI * f / C0, 0.0) * eps.sqrt();
    let co = Coeffs { shift: (C::new(0.0, 0.0), C::new(0.0, 0.0)), fixed: Some((-ONE, ONE)) };
    for (rho, d, ds) in [(0.0, 0.3, 0.2), (0.8, 0.5, 0.1), (2.0, 0.2, 0.2)] {
        let got = reflected_dyadic_side(rho, d + ds, f, eps, &co, Side::Ground);
        // observer at depth d, image of a source at depth ds sits at height ds;
        // image current -P c: horizontal parts reversed
        let (_, fr) = free_field(rho, -(d + ds), f, k2);
        let want = [-fr[0], -fr[1], fr[2], -fr[3], fr[4]];
        assert!(close(&got, &want, 1e-6), "rho={rho}: {got:?} vs {want:?}");
    }
}

fn transparent() -> Ground {
    Ground::Real(RealGround { eps_r: 1.0, sigma: 0.0 })
}

#[test]
fn transparent_ground_gives_free_space() {
    let f = 299.792458e6;
    // buried, crossing and above-ground dipoles must all behave as in free space
    for (z0, label) in [(-0.6, "buried"), (0.0, "crossing"), (0.6, "above")] {
        let wire = Wire::new([0.0, 0.0, z0 - 0.25], [0.0, 0.0, z0 + 0.25], 0.001, 20);
        let mut free = Model::new(vec![wire.clone()], Ground::Free);
        free.add_source(0, 10, ONE);
        let zf = free.solve(f).unwrap().input_impedance();
        let mut m = Model::new(vec![wire], transparent());
        m.ground_table_reuse = 0.0;
        m.add_source(0, 10, ONE);
        let s = m.solve(f).unwrap();
        let z = s.input_impedance();
        println!("{label}: {z:.3} vs free {zf:.3}");
        assert!((z - zf).norm() < 2e-3 * zf.norm(), "{label}: {z} vs {zf}");
        // far field in the upper half matches free space too
        let sf = free.solve(f).unwrap();
        for th in [0.3f64, 1.0, 1.4] {
            let (a, b) = (s.gain_dbi(th, 0.4), sf.gain_dbi(th, 0.4));
            assert!((a - b).abs() < 0.02, "{label} theta {th}: {a} vs {b}");
        }
    }
    // a horizontal wire buried and an above-ground wire coupling across the surface
    let w = vec![
        Wire::new([0.0, -0.24, 0.3], [0.0, 0.24, 0.3], 0.001, 16),
        Wire::new([0.1, -0.24, -0.2], [0.1, 0.24, -0.2], 0.001, 16),
    ];
    let mut free = Model::new(w.clone(), Ground::Free);
    free.add_source(0, 8, ONE);
    let mut m = Model::new(w, transparent());
    m.ground_table_reuse = 0.0;
    m.add_source(0, 8, ONE);
    let (z, zf) = (m.solve(f).unwrap().input_impedance(), free.solve(f).unwrap().input_impedance());
    assert!((z - zf).norm() < 2e-3 * zf.norm(), "coupled: {z} vs {zf}");
}

#[test]
fn matrix_is_symmetric_with_wires_on_both_sides() {
    let f = 7.1e6;
    let w = vec![
        Wire::new([0.0, 0.0, -0.5], [0.0, 0.0, 10.0], 0.005, 22), // rod + vertical
        Wire::new([0.0, 0.0, -0.1], [5.0, 1.0, -0.15], 0.002, 6), // buried radial
        Wire::new([0.5, 0.0, 3.0], [2.0, 0.0, 2.0], 0.002, 4),    // sloping wire above
    ];
    let m = Model::new(w, Ground::Real(RealGround::AVERAGE));
    let mesh = m.mesh().unwrap();
    let z = m.impedance_matrix(&mesh, f).unwrap();
    let nb = mesh.n_unknowns();
    let zmax = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let mut worst = 0.0f64;
    for i in 0..nb {
        for j in 0..nb {
            worst = worst.max((z[i * nb + j] - z[j * nb + i]).norm() / zmax);
        }
    }
    println!("max asymmetry {worst:.2e}");
    assert!(worst < 2e-4, "{worst}");
}

#[test]
fn ground_rod_resistance_matches_dwight() {
    // Two 2 m rods 50 m apart joined by a wire 1 m up, fed in the middle, at 10 kHz
    // (skin depth 50 m): the resistance is two rods in series less their mutual
    // resistance.  Dwight: R = (ln(4L/a) - 1) / (2 pi sigma L).
    let (f, sigma, l, a, s) = (1.0e4, 0.01, 2.0, 0.01, 50.0);
    let g = RealGround { eps_r: 10.0, sigma };
    let w = vec![
        Wire::new([-s / 2.0, 0.0, -l], [-s / 2.0, 0.0, 1.0], a, 12),
        Wire::new([-s / 2.0, 0.0, 1.0], [s / 2.0, 0.0, 1.0], 0.002, 26),
        Wire::new([s / 2.0, 0.0, 1.0], [s / 2.0, 0.0, -l], a, 12),
    ];
    let mut m = Model::new(w, Ground::Real(g));
    m.ground_table_reuse = 0.0;
    m.add_source(1, 13, ONE);
    let z = m.solve(f).unwrap().input_impedance();
    let r_rod = ((4.0 * l / a).ln() - 1.0) / (2.0 * PI * sigma * l);
    let r_mutual = 1.0 / (2.0 * PI * sigma * s);
    let want = 2.0 * (r_rod - r_mutual);
    println!("two rods: Z = {z:.2}, Dwight {want:.2} ohm");
    assert!((z.re / want - 1.0).abs() < 0.1, "{z} vs {want}");
    assert!(z.im > 0.0 && z.im < 0.5 * want, "{z}");
}

#[test]
fn more_buried_radials_raise_efficiency() {
    // quarter-wave vertical on average ground, fed against a short rod and buried
    // radials 0.1 m down: more radials, less ground loss
    let f = 7.1e6;
    let lam = C0 / f;
    let eff = |n: usize| {
        let mut w = vec![Wire::new([0.0, 0.0, -0.1], [0.0, 0.0, 0.24 * lam], 0.005, 26)];
        for i in 0..n {
            let a = 2.0 * PI * i as f64 / n as f64;
            w.push(Wire::new([0.0, 0.0, -0.1], [0.2 * lam * a.cos(), 0.2 * lam * a.sin(), -0.1], 0.002, 10));
        }
        let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE));
        // feed just above the surface (node 1 of the vertical is z = 0)
        m.add_source(0, 1, ONE);
        let s = m.solve(f).unwrap();
        let e = s.efficiency();
        println!("{n} radials: Z = {:.1}, efficiency {:.1}%", s.input_impedance(), 100.0 * e);
        e
    };
    let (e4, e16) = (eff(4), eff(16));
    assert!(e16 > e4, "{e4} {e16}");
    assert!(e4 > 0.05 && e16 < 0.9);
}

#[test]
fn ground_side_table_matches_direct_integration() {
    // the ground-side reflection table against direct integration, out to grazing
    // angles, for grounds from poor to sea water (sea water used to blow up: the
    // table divided the lateral wave by the ground's attenuation along R)
    use wiremom::sommerfeld::{Table, Window};
    for (f, g) in [(28e6, RealGround::POOR), (7.1e6, RealGround::AVERAGE), (1.9e6, RealGround::VERY_GOOD), (3.6e6, RealGround::SEA_WATER)] {
        let eps = g.eps_c(f);
        let g0 = g.gamma0(f);
        let co = Coeffs { shift: (-g0, g0), fixed: None };
        let t = Table::new_window(g, f, Window { r_min: 0.05, r_max: 45.0, theta_min: 0.001 }, 1.05, None, Side::Ground);
        let peak = reflected_dyadic_side(0.05, 0.2, f, eps, &co, Side::Ground).iter().map(|v| v.norm()).fold(0.0, f64::max);
        for (rho, h) in [(0.1, 0.6), (1.0, 0.6), (3.0, 0.2), (10.0, 0.6), (16.0, 1.5), (30.0, 0.6), (40.0, 0.1)] {
            let want = reflected_dyadic_side(rho, h, f, eps, &co, Side::Ground);
            let got = t.eval(rho, h, f);
            let scale = want.iter().map(|v| v.norm()).fold(0.0, f64::max);
            let err = got.iter().zip(&want).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 0.02 * scale + 1e-9 * peak, "{g:?} rho {rho} h {h}: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn transmission_table_matches_direct_integration() {
    use wiremom::sommerfeld::TransTable;
    for (f, g) in [(28e6, RealGround::POOR), (7.1e6, RealGround::AVERAGE), (3.6e6, RealGround::SEA_WATER)] {
        let lam = C0 / f;
        let eps = g.eps_c(f);
        let t = TransTable::new(eps, f, 0.35 * lam, (0.008, 0.22 * lam), (0.004, 1.0));
        let mut worst = 0.0f64;
        for i in 0..200 {
            let u = |p: usize| ((i * p) % 199) as f64 / 199.0;
            let (rho, zs, d) = (0.34 * lam * u(37).powi(3), 0.009 + 0.21 * lam * u(53).powi(3), 0.0045 + 0.99 * u(71).powi(2));
            let want = transmitted_dyadic(rho, zs, d, f, eps, true);
            let got = t.eval(rho, zs, d);
            let scale = want.iter().map(|v| v.norm()).fold(0.0, f64::max);
            worst = worst.max(got.iter().zip(&want).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) / scale);
        }
        println!("{g:?}: worst relative error {worst:.1e}");
        assert!(worst < 5e-3, "{g:?}: {worst}");
    }
}

#[test]
fn tabulated_air_to_ground_couplings_match_direct() {
    // vertical with a rod and buried radials, in average ground and in sea water
    for (f, g, depth) in [(7.1e6, RealGround::AVERAGE, 0.1), (3.6e6, RealGround::SEA_WATER, 0.3)] {
        let lam = C0 / f;
        let mut w = vec![Wire::new([0.0, 0.0, -1.0], [0.0, 0.0, 0.24 * lam], 0.005, 26)];
        for i in 0..6 {
            let a = 2.0 * PI * i as f64 / 6.0;
            w.push(Wire::new([0.0, 0.0, -depth], [0.2 * lam * a.cos(), 0.2 * lam * a.sin(), -depth], 0.002, 8));
        }
        let mut m = Model::new(w, Ground::Real(g));
        m.add_source(0, 3, ONE);
        let st = m.solve(f).unwrap();
        m.tabulate_cross = false;
        let sd = m.solve(f).unwrap();
        let (zt, zd) = (st.input_impedance(), sd.input_impedance());
        println!("{g:?}: table {zt:.4}, direct {zd:.4}, efficiency {:.3}", st.efficiency());
        assert!((zt - zd).norm() < 1e-5 * zd.norm(), "{zt} vs {zd}");
        // a quarter-wave vertical fed against radials in the sea: low loss, ~36 ohm
        if g == RealGround::SEA_WATER {
            assert!(st.efficiency() > 0.8 && (zt.re - 37.0).abs() < 6.0, "{zt}");
        }
    }
}
