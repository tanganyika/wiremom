//! Power budget: input power = radiated power + the losses in the wires, loads and
//! networks (free space and perfect ground, where nothing else can absorb power).

use num_complex::Complex64 as C;
use wiremom::{C0, Conductor, Ground, LoadKind, Model, Port, RealGround, Solution, TwoPortKind, Wire};

const F: f64 = 14.2e6;
const ONE: C = C::new(1.0, 0.0);

fn lam() -> f64 {
    C0 / F
}

fn copper(w: Wire) -> Wire {
    Wire { conductor: Conductor::Lossy { sigma: 5.8e7, mu_r: 1.0 }, ..w }
}

fn dipole(radius: f64) -> Wire {
    let l = lam();
    Wire::new([0.0, -0.24 * l, 10.0], [0.0, 0.24 * l, 10.0], radius, 21)
}

/// input − radiated − losses, as a fraction of the input
fn unaccounted(s: &Solution) -> f64 {
    let l = s.losses;
    let p = s.input_power();
    (p - s.radiated_power(91, 72) - l.wire - l.loads - l.networks) / p
}

#[test]
fn lossless_wire_loses_nothing() {
    let mut m = Model::new(vec![dipole(0.001)], Ground::Free);
    m.add_source(0, 10, ONE);
    let s = m.solve(F).unwrap();
    assert_eq!(s.losses.wire, 0.0);
    assert!(unaccounted(&s).abs() < 2e-3, "{}", unaccounted(&s));
}

#[test]
fn wire_loss_balances() {
    // thin steel-like wire (low conductivity) so the loss is large
    for (sigma, ground) in [(5.8e7, Ground::Free), (1e5, Ground::Free), (1e5, Ground::Perfect)] {
        let w = Wire { conductor: Conductor::Lossy { sigma, mu_r: 1.0 }, ..dipole(0.0005) };
        let mut m = Model::new(vec![w], ground);
        m.add_source(0, 10, ONE);
        let s = m.solve(F).unwrap();
        let frac = s.losses.wire / s.input_power();
        println!("sigma {sigma:e} {ground:?}: wire loss {:.2} %, unaccounted {:.3} %", frac * 100.0, unaccounted(&s) * 100.0);
        assert!(frac > 0.0);
        assert!(unaccounted(&s).abs() < 3e-3, "{}", unaccounted(&s));
    }
}

#[test]
fn copper_loss_is_about_a_percent() {
    // NEC-2 gives 98.87 % for this dipole in 1 mm radius copper (1.1 % loss)
    let mut m = Model::new(vec![copper(dipole(0.001))], Ground::Free);
    m.add_source(0, 10, ONE);
    let s = m.solve(F).unwrap();
    let frac = s.losses.wire / s.input_power();
    assert!(frac > 0.008 && frac < 0.015, "{frac}");
}

#[test]
fn load_loss_balances() {
    let mut m = Model::new(vec![dipole(0.001)], Ground::Free);
    m.add_source(0, 10, ONE);
    m.add_load(0, 5, LoadKind::Series { r: 20.0, l: 1e-6, c: None });
    m.add_load(0, 15, LoadKind::Parallel { r: 5e3, l: 2e-6, c: 0.0 });
    let s = m.solve(F).unwrap();
    assert!(s.losses.loads / s.input_power() > 0.05, "{:?}", s.losses);
    assert!(unaccounted(&s).abs() < 3e-3, "{}", unaccounted(&s));
}

#[test]
fn line_loss_balances() {
    let l = lam();
    let wires = vec![Wire::new([0.0, -0.24 * l, 0.0], [0.0, 0.24 * l, 0.0], 0.001, 20), Wire::new([0.0, 0.0, 30.0], [0.0, 0.0, 30.05], 0.001, 2)];
    let mut m = Model::new(wires, Ground::Free);
    m.add_source(1, 1, ONE);
    let line = TwoPortKind::Line { z0: 75.0, length: Some(50.0), velocity_factor: 0.66, loss_db_per_100m: 2.0, crossed: false, shunt: [C::new(0.0, 0.0); 2] };
    m.add_network(Port { wire: 1, node: 1 }, Port { wire: 0, node: 10 }, line);
    let s = m.solve(F).unwrap();
    let frac = s.losses.networks / s.input_power();
    // 1 dB of matched loss is 21 %; mismatch adds a little
    assert!(frac > 0.18 && frac < 0.35, "{frac}");
    assert!(unaccounted(&s).abs() < 3e-3, "{}", unaccounted(&s));
}

#[test]
fn real_ground_takes_what_is_left() {
    // a low dipole over poor ground: the remainder is the ground loss, and it is large
    let w = Wire::new([0.0, -0.24 * lam(), 3.0], [0.0, 0.24 * lam(), 3.0], 0.001, 21);
    let mut m = Model::new(vec![copper(w)], Ground::Real(RealGround { eps_r: 13.0, sigma: 0.005 }));
    m.add_source(0, 10, ONE);
    let s = m.solve(F).unwrap();
    let ground = unaccounted(&s);
    println!("ground loss {:.1} %, wire {:.2} %", ground * 100.0, s.losses.wire / s.input_power() * 100.0);
    assert!(ground > 0.1 && ground < 0.9, "{ground}");
}
