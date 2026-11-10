//! How much the speed settings change results: adaptive vs full quadrature.
use num_complex::Complex64 as C;
use wiremom::*;
fn main() {
    let f = 14.15e6;
    let lam = C0 / f;
    let yagi = |g: Ground, h: f64| {
        let w = vec![
            Wire::new([0.0, -0.255 * lam, h], [0.0, 0.255 * lam, h], 0.006, 23),
            Wire::new([0.2 * lam, -0.2375 * lam, h], [0.2 * lam, 0.2375 * lam, h], 0.006, 24),
            Wire::new([0.35 * lam, -0.22 * lam, h], [0.35 * lam, 0.22 * lam, h], 0.006, 23),
        ];
        let mut m = Model::new(w, g);
        m.add_source(1, 12, C::new(1.0, 0.0));
        m
    };
    for (name, g, h) in [("free", Ground::Free, 0.0), ("pec", Ground::Perfect, 15.0), ("real", Ground::Real(RealGround::AVERAGE), 15.0)] {
        let mut full = yagi(g, h);
        full.adaptive = false;
        full.ground_table_reuse = 0.0;
        let mut fast = yagi(g, h);
        fast.ground_table_reuse = 0.0;
        let (zf, zs) = (full.solve(f).unwrap(), fast.solve(f).unwrap());
        let d = (zf.input_impedance() - zs.input_impedance()).norm() / zf.input_impedance().norm();
        let dg = zf.gain_dbi(1.2, 0.0) - zs.gain_dbi(1.2, 0.0);
        println!("{name}: full {:.4} fast {:.4}  rel diff {d:.1e}, gain diff {dg:.1e} dB (fill {:.1} vs {:.1} ms)", zf.input_impedance(), zs.input_impedance(), zf.t_fill_ms, zs.t_fill_ms);
    }
}
