//! Wires near, on and in lossy ground against independent references
//! (`python/tools/near_ground_reference.py`): derived afresh from the
//! transmission-line analogy of the half-space, not from wiremom's Sommerfeld code.

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::sommerfeld::transmitted_dyadic;
use wiremom::{C0, Ground, Model, RealGround, Wire};

/// Propagation constant from currents sampled `dx` apart away from sources and
/// ends: any mix of exp(-j beta x) and exp(+j beta x) has I(n+1) + I(n-1) =
/// 2 cos(beta dx) I(n); least squares for cos(beta dx).
fn fit_beta(cur: &[C], dx: f64) -> C {
    let (mut num, mut den) = (C::new(0.0, 0.0), 0.0);
    for n in 1..cur.len() - 1 {
        num += cur[n].conj() * (cur[n + 1] + cur[n - 1]);
        den += cur[n].norm_sqr();
    }
    let b = (num / den / 2.0).acos() / dx;
    if b.re < 0.0 { -b } else { b }
}

#[test]
fn guided_wave_along_wires_near_and_in_the_ground_matches_the_modal_equation() {
    // A long centre-fed wire (radius 1 mm) parallel to average ground at 14.15 MHz,
    // above it or buried; beta fitted from the current between `w0` and `w1` metres
    // from the feed.  Reference: the guided-mode root of an infinite wire.
    // The 2 mm cases with segments 50-200 times their height failed before the
    // remainder integrals were graded near the surface (18% too slow a wave).
    let f = 14.15e6;
    let k0 = 2.0 * PI * f / C0;
    // (height, length, segment, w0, w1, reference beta/k0)
    let cases: [(f64, f64, f64, f64, f64, C); 5] = [
        (0.1, 60.0, 0.25, 4.5, 10.5, C::new(1.147343, -0.105661)),
        (0.002, 60.0, 0.25, 4.5, 10.5, C::new(1.847569, -0.276594)),
        (0.002, 60.0, 0.5, 4.5, 10.5, C::new(1.847569, -0.276594)),
        (-0.002, 20.0, 0.4, 3.0, 8.0, C::new(2.922490, -0.777800)),
        (-0.1, 20.0, 0.1, 3.0, 8.0, C::new(3.439680, -0.860559)),
    ];
    for (z0, len, dx, w0, w1, want) in cases {
        let n = (len / dx).round() as usize;
        let mut m = Model::new(vec![Wire::new([-len / 2.0, 0.0, z0], [len / 2.0, 0.0, z0], 0.001, n)], Ground::Real(RealGround::AVERAGE));
        m.ground_table_reuse = 0.0;
        m.add_source(0, n / 2, C::new(1.0, 0.0));
        let s = m.solve(f).unwrap();
        let cur: Vec<C> = s.segment_currents().iter().map(|c| c.2).collect();
        let got = fit_beta(&cur[n / 2 + (w0 / dx) as usize..n / 2 + (w1 / dx) as usize], dx) / k0;
        let err = (got - want).norm() / want.norm();
        println!("z0 {z0:+.3}, segments {dx} m: beta/k0 {got:.5}, reference {want:.5} ({:.2}%)", 100.0 * err);
        assert!(err < 0.02, "z0 {z0}, dx {dx}: {got} vs {want}");
    }
}

#[test]
fn transmitted_field_matches_an_independent_derivation() {
    // field in average ground from a unit element in the air: E_rr, E_ff
    // (horizontal element) and E_zz (vertical), at (rho, source height, depth)
    let f = 14.15e6;
    let eps = RealGround::AVERAGE.eps_c(f);
    let c = |re: f64, im: f64| C::new(re, im);
    let cases = [
        ((0.5, 1.0, 0.1), [c(-4.828815, -0.4134571), c(-6.438469, 3.253770), c(4.321322, -11.68978)]),
        ((3.0, 0.5, 0.1), [c(-0.7606016, -0.5169699), c(-1.214500, 1.169055), c(-0.4476586, 0.3665306)]),
        ((0.0, 2.0, 0.3), [c(-2.452942, 0.3895375), c(-2.452942, 0.3895375), c(0.4113544, -2.820949)]),
        ((1.0, 0.1, 0.1), [c(4.983681, -26.96127), c(-9.968252, 7.703024), c(-5.108793, 8.925312)]),
        ((0.2, 0.05, 0.02), [c(948.7713, -2145.894), c(-578.7175, 1225.807), c(-382.9118, 840.2946)]),
        ((8.0, 0.3, 0.5), [c(-0.03630716, -0.08488280), c(0.08330825, 0.01184678), c(0.01719202, 0.1051880)]),
    ];
    for ((rho, zs, d), want) in cases {
        let t = transmitted_dyadic(rho, zs, d, f, eps, false);
        let got = [t[0], t[1], t[4]];
        let scale = want.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-5 * scale, "({rho}, {zs}, {d}): {got:?} vs {want:?}");
        }
    }
}

#[test]
fn current_through_the_surface_is_passive() {
    // A vertical rod through a lossless dielectric ground: the basis whose current
    // passes through the surface used to have a self-resistance of
    // -(eta0/4 pi) 2(sqrt(eps)-1)/(eps+1) (-11 ohm at eps 13), from closed-form
    // potentials that jump at the surface.  Its resistance is only radiation now.
    let f = 14.15e6;
    for eps_r in [2.0, 4.0, 13.0, 81.0] {
        let m = Model::new(vec![Wire::new([0.0, 0.0, -0.2], [0.0, 0.0, 0.2], 0.0005, 8)], Ground::Real(RealGround { eps_r, sigma: 0.0 }));
        let mesh = m.mesh().unwrap();
        let z = m.impedance_matrix(&mesh, f).unwrap();
        let nb = mesh.n_unknowns();
        let b = (0..nb).find(|&b| mesh.nodes[mesh.basis[b].node][2].abs() < 1e-12).unwrap();
        let r = z[b * nb + b].re;
        println!("eps {eps_r}: crossing basis self-resistance {r:.4} ohm");
        assert!(r > 0.0 && r < 0.1, "eps {eps_r}: {r}");
    }
    // a short vertical fed just above the surface over shallow buried radials: the
    // input resistance went negative with 8 or more radials
    for n in [4usize, 16, 32] {
        let mut w = vec![Wire::new([0.0, 0.0, -0.003], [0.0, 0.0, 0.3], 0.005, 6)];
        for i in 0..n {
            let t = 2.0 * PI * i as f64 / n as f64;
            w.push(Wire::new([0.0, 0.0, -0.003], [0.5 * t.cos(), 0.5 * t.sin(), -0.003], 0.0005, 4));
        }
        let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE));
        m.add_source(0, 1, C::new(1.0, 0.0));
        let z = m.solve(f).unwrap().input_impedance();
        println!("{n} radials 3 mm deep: Z = {z:.2}");
        assert!(z.re > 1.0, "{n} radials: {z}");
    }
}
