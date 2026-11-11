use num_complex::Complex64 as C;
use std::time::Instant;
fn main() {
    for n in [300usize, 800, 1347, 2000] {
        let mut seed = 3u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5 };
        let a: Vec<C> = (0..n * n).map(|_| C::new(rnd(), rnd())).collect();
        let t = Instant::now();
        let _ = wiremom::lu::Lu::factor(a, n).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!("n {n}: {:.0} ms, {:.1} GFLOP/s", s * 1e3, 8.0 / 3.0 * (n as f64).powi(3) / s / 1e9);
    }
}
