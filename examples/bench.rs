//! Speed benchmark: `cargo run --release --example bench`
//! A 3-element 20 m Yagi (about 70 segments) in free space and 15 m over average
//! ground: matrix fill + solve, a pattern scan like an application's (about 8,000
//! directions) and a 24-point sweep; then a vertical with a ground rod and eight
//! buried radials.

use num_complex::Complex64 as C;
use std::time::Instant;
use wiremom::{C0, Ground, Model, RealGround, Wire};

fn yagi(ground: Ground, h: f64) -> Model {
    let f = 14.15e6;
    let lam = C0 / f;
    let w = vec![
        Wire::new([0.0, -0.255 * lam, h], [0.0, 0.255 * lam, h], 0.006, 23),
        Wire::new([0.2 * lam, -0.2375 * lam, h], [0.2 * lam, 0.2375 * lam, h], 0.006, 24),
        Wire::new([0.35 * lam, -0.22 * lam, h], [0.35 * lam, 0.22 * lam, h], 0.006, 23),
    ];
    let mut m = Model::new(w, ground);
    m.add_source(1, 12, C::new(1.0, 0.0));
    m
}

fn main() {
    let f = 14.15e6;
    for (name, g, h) in [("free space", Ground::Free, 0.0), ("real ground", Ground::Real(RealGround::AVERAGE), 15.0)] {
        let m = yagi(g, h);
        let t = Instant::now();
        let s = m.solve(f).unwrap();
        let t_solve = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut best = 0.0f64;
        for el in 0..31 {
            for az in 0..120 {
                let g = s.gain((90.0 - 3.0 * el as f64).to_radians(), (3.0 * az as f64).to_radians()).total;
                best = best.max(g);
            }
        }
        for _ in 0..4 {
            for a in 0..361 {
                best = best.max(s.gain(1.2, (a as f64).to_radians()).total);
            }
        }
        let t_far = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let freqs: Vec<f64> = (0..24).map(|i| f * (0.97 + 0.06 * i as f64 / 23.0)).collect();
        let sw = m.sweep(&freqs);
        let t_sweep = t.elapsed().as_secs_f64() * 1e3;
        let z = s.input_impedance();
        println!(
            "{name:12}: solve {t_solve:6.1} ms (fill {:.1}), pattern {t_far:6.1} ms, sweep24 {t_sweep:6.0} ms | Z = {:.3}{:+.3}j, Gmax {:.3} dBi, Z(sweep end) {:.3}",
            s.t_fill_ms,
            z.re,
            z.im,
            10.0 * best.log10(),
            sw.last().unwrap().as_ref().unwrap().input_impedance()
        );
    }
    let f = 7.1e6;
    let lam = C0 / f;
    let mut w = vec![Wire::new([0.0, 0.0, -1.0], [0.0, 0.0, 0.24 * lam], 0.005, 26)];
    for i in 0..8 {
        let a = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
        w.push(Wire::new([0.0, 0.0, -0.1], [10.0 * a.cos(), 10.0 * a.sin(), -0.1], 0.002, 10));
    }
    let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 3, C::new(1.0, 0.0));
    let _ = m.solve(f).unwrap(); // tables
    let t = Instant::now();
    let s = m.solve(f).unwrap();
    println!("buried radials: solve {:.0} ms | Z = {:.3}", t.elapsed().as_secs_f64() * 1e3, s.input_impedance());
}
