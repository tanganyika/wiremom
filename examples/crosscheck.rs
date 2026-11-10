//! Tabulated vs direct air-to-ground couplings for buried-wire models:
//! `cargo run --release --example crosscheck`
use num_complex::Complex64 as C;
use std::f64::consts::PI;
use std::time::Instant;
use wiremom::{C0, Ground, Model, RealGround, Wire};
fn radials(f: f64, g: RealGround, n: usize, depth: f64) -> Model {
    let lam = C0 / f;
    let mut w = vec![Wire::new([0.0, 0.0, -1.0], [0.0, 0.0, 0.24 * lam], 0.005, 26)];
    for i in 0..n {
        let a = 2.0 * PI * i as f64 / n as f64;
        w.push(Wire::new([0.0, 0.0, -depth], [0.2 * lam * a.cos(), 0.2 * lam * a.sin(), -depth], 0.002, 10));
    }
    let mut m = Model::new(w, Ground::Real(g));
    m.add_source(0, 3, C::new(1.0, 0.0));
    m
}
fn main() {
    let cases: Vec<(&str, Model, f64)> = vec![
        ("8 radials avg 7.1", radials(7.1e6, RealGround::AVERAGE, 8, 0.1), 7.1e6),
        ("4 radials poor 28", radials(28e6, RealGround::POOR, 4, 0.05), 28e6),
        ("16 radials sea 3.6", radials(3.6e6, RealGround::SEA_WATER, 16, 0.3), 3.6e6),
        ("8 radials v.good 1.9", radials(1.9e6, RealGround::VERY_GOOD, 8, 0.02), 1.9e6),
    ];
    for (name, mut m, f) in cases {
        m.tabulate_cross = false;
        let t = Instant::now();
        let sd = m.solve(f).unwrap();
        let td = t.elapsed().as_secs_f64() * 1e3;
        m.tabulate_cross = true;
        let t = Instant::now();
        let st = m.solve(f).unwrap();
        let tt = t.elapsed().as_secs_f64() * 1e3;
        let (zd, zt) = (sd.input_impedance(), st.input_impedance());
        println!("{name:22}: direct {zd:.4} ({td:.0} ms), table {zt:.4} ({tt:.0} ms), rel diff {:.1e}, eff {:.4} vs {:.4}", (zd - zt).norm() / zd.norm(), sd.efficiency(), st.efficiency());
    }
}
