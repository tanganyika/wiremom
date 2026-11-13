//! Insulated wires: in air the sleeve acts as the familiar extra series inductance;
//! deep in lossy ground an insulated wire guides a slow wave whose propagation
//! constant King's theory of insulated antennas gives in closed form.

use complex_bessel::{Scaling, hankel2_seq};
use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::{C0, EPS0, Ground, Insulation, MU0, Model, RealGround, Wire};

const ONE: C = C::new(1.0, 0.0);

/// Centre-fed straight dipole along x, `n` segments (even), at height `z`.
fn dipole(half: f64, z: f64, a: f64, n: usize, ins: Option<Insulation>) -> Wire {
    Wire { insulation: ins, ..Wire::new([-half, 0.0, z], [half, 0.0, z], a, n) }
}

fn z_in(w: Wire, ground: Ground, f: f64) -> C {
    let n = match w.segments {
        wiremom::Segments::Uniform(n) => n,
        _ => unreachable!(),
    };
    let mut m = Model::new(vec![w], ground);
    m.add_source(0, n / 2, ONE);
    m.solve(f).unwrap().input_impedance()
}

/// The first zero of X(x) between x0 and x1 (bisection; X rises through zero).
fn zero_of(mut x0: f64, mut x1: f64, f: impl Fn(f64) -> f64) -> f64 {
    let (mut f0, f1) = (f(x0), f(x1));
    assert!(f0.signum() != f1.signum(), "no zero between {x0} and {x1}: {f0} {f1}");
    for _ in 0..40 {
        let xm = 0.5 * (x0 + x1);
        let fm = f(xm);
        if fm.signum() == f0.signum() {
            x0 = xm;
            f0 = fm;
        } else {
            x1 = xm;
        }
    }
    0.5 * (x0 + x1)
}

#[test]
fn in_air_it_is_the_series_inductance() {
    // a 2 mm wire with 1 mm of PVC: the charge-term model and the classic L' model
    // put the half-wave resonance within a few tenths of a percent of each other
    let (a, t, er) = (0.001, 0.001, 3.5);
    let ins = Insulation { thickness: t, eps_r: er };
    let lp = MU0 / (2.0 * PI) * (1.0 - 1.0 / er) * ((a + t) / a).ln();
    let half = 0.24 * C0 / 14.2e6;
    let res = |w: &dyn Fn(f64) -> Wire| zero_of(11e6, 16e6, |f| z_in(w(f), Ground::Free, f).im);
    let bare = res(&|_| dipole(half, 0.0, a, 40, None));
    let charge = res(&|_| dipole(half, 0.0, a, 40, Some(ins)));
    let induct = res(&|_| Wire { series_l_per_m: lp, ..dipole(half, 0.0, a, 40, None) });
    println!("resonance: bare {:.4} MHz, insulated {:.4} (charge model) {:.4} (L' model)", bare / 1e6, charge / 1e6, induct / 1e6);
    assert!(charge < bare * 0.98, "insulation lowers the resonance");
    // second order apart: the L' form assumes the current varies like the free-space wave
    assert!((charge - induct).abs() / induct < 0.006);
}

/// King's insulated-antenna line constants: series z, shunt y per metre, for a wire of
/// radius a in a sleeve to radius b (εᵢ) in a medium of wavenumber k4 (e^{jωt}).
fn king(f: f64, a: f64, b: f64, er_ins: f64, eps_out: C) -> C {
    let w = 2.0 * PI * f;
    let k4 = w / C0 * eps_out.sqrt();
    let x = k4 * b;
    let h = hankel2_seq(0.0, x, 2, Scaling::Unscaled).unwrap().values;
    let z = C::new(0.0, w * MU0 / (2.0 * PI)) * ((b / a).ln() + h[0] / (x * h[1]));
    let y = C::new(0.0, 2.0 * PI * w * EPS0 * er_ins / (b / a).ln());
    let g = (z * y).sqrt();
    if g.re < 0.0 { -g } else { g }
}

/// An insulated dipole 20 m down: its first resonance against King's π/2β.
fn buried_resonance(g: RealGround, f: f64) -> (f64, f64, C) {
    let (a, t, er) = (0.001, 0.001, 3.5);
    let gamma = king(f, a, a + t, er, g.eps_c(f));
    let ins = Some(Insulation { thickness: t, eps_r: er });
    let x = |half: f64| z_in(dipole(half, -20.0, a, 40, ins), Ground::Real(g), f).im;
    let half = zero_of(3.0, 12.0, x);
    (half, PI / 2.0 / gamma.im, gamma)
}

#[test]
fn in_wet_ground_it_follows_king() {
    // King's closed form holds when the medium is far denser than the sleeve
    // (|ε̃ₘ| >> εᵢ): wet salty ground, 0.5 S/m, |ε̃ₘ| ≈ 4900 at 1.83 MHz
    let (half, half_king, gamma) = buried_resonance(RealGround { eps_r: 30.0, sigma: 0.5 }, 1.83e6);
    println!("wet ground: King β {:.4} rad/m α {:.4} Np/m → half-length {half_king:.3} m; wiremom {half:.3} m", gamma.im, gamma.re);
    // within 0.5 %: the open ends add next to nothing in so dense a medium, and the
    // soil's small remaining share lengthens it slightly
    assert!((half - half_king).abs() < 0.02 * half_king, "{half} vs {half_king}");
}

#[test]
fn in_average_ground_the_soil_adds_its_own_share() {
    // average ground (|ε̃ₘ| ≈ 51, only ~15 times the sleeve's 3.5): the soil's own
    // share of the potential, which King's closed form leaves out, is comparable with
    // the sleeve's, so the wave is faster and the resonance longer than King's
    let (half, half_king, _) = buried_resonance(RealGround { eps_r: 13.0, sigma: 0.005 }, 1.83e6);
    println!("average ground: King half-length {half_king:.3} m; wiremom {half:.3} m");
    assert!(half > half_king && half < 1.3 * half_king, "{half} vs {half_king}");
}
