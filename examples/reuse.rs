use num_complex::Complex64 as C;
use wiremom::*;
fn main() {
    let f0 = 14.15e6;
    let lam = C0 / f0;
    let mk = |reuse: f64, h: f64, g: RealGround| {
        let w = vec![
            Wire::new([0.0, -0.255 * lam, h], [0.0, 0.255 * lam, h], 0.006, 23),
            Wire::new([0.2 * lam, -0.2375 * lam, h], [0.2 * lam, 0.2375 * lam, h], 0.006, 24),
            Wire::new([0.35 * lam, -0.22 * lam, h], [0.35 * lam, 0.22 * lam, h], 0.006, 23),
        ];
        let mut m = Model::new(w, Ground::Real(g));
        m.ground_table_reuse = reuse;
        m.add_source(1, 12, C::new(1.0, 0.0));
        m
    };
    for (gname, g) in [("average", RealGround::AVERAGE), ("poor", RealGround::POOR), ("sea", RealGround::SEA_WATER)] {
        for h in [3.0, 15.0] {
            let freqs: Vec<f64> = (0..25).map(|i| f0 * (0.94 + 0.12 * i as f64 / 24.0)).collect();
            let exact: Vec<C> = freqs.iter().map(|&f| mk(0.0, h, g).solve(f).unwrap().input_impedance()).collect();
            for reuse in [0.01, 0.04, 0.08, 0.12] {
                let m = mk(reuse, h, g);
                let t = std::time::Instant::now();
                let z: Vec<C> = m.sweep(&freqs).into_iter().map(|s| s.unwrap().input_impedance()).collect();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let worst = z.iter().zip(&exact).map(|(a, b)| (a - b).norm() / b.norm()).fold(0.0, f64::max);
                println!("{gname:8} h={h:4} reuse {reuse:.2}: worst rel error {worst:.1e}, sweep25 {ms:.0} ms");
            }
        }
    }
}
