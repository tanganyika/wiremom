//! Fill time and impedance of the reduced and exact kernels on a 20 m Yagi (thin
//! elements) and on a thick-element 432 MHz Yagi with segments under one diameter.
//!
//! cargo run --release --example kernel_cost
use num_complex::Complex64 as C;
use wiremom::*;

fn yagi(f: f64, a: f64, nseg: usize) -> Model {
    let lam = C0 / f;
    let w = vec![
        Wire::new([0.0, -0.255 * lam, 0.0], [0.0, 0.255 * lam, 0.0], a, nseg + 1),
        Wire::new([0.2 * lam, -0.2375 * lam, 0.0], [0.2 * lam, 0.2375 * lam, 0.0], a, nseg),
        Wire::new([0.35 * lam, -0.22 * lam, 0.0], [0.35 * lam, 0.22 * lam, 0.0], a, nseg + 1),
    ];
    let mut m = Model::new(w, Ground::Free);
    m.add_source(1, nseg / 2, C::new(1.0, 0.0));
    m
}

fn main() {
    for (name, f, a, nseg) in [("20 m, 6 mm radius, 24 seg", 14.15e6, 0.006, 24), ("20 m, 6 mm, 200 seg", 14.15e6, 0.006, 200), ("432 MHz, 3.125 mm radius, 50 seg (d/a 2)", 432e6, 0.003125, 50), ("432 MHz, 3.125 mm, 200 seg (d/a 0.5)", 432e6, 0.003125, 200)] {
        for k in [Kernel::Reduced, Kernel::Exact] {
            let mut m = yagi(f, a, nseg);
            m.kernel = k;
            let _ = m.solve(f).unwrap();
            let s = m.solve(f).unwrap();
            println!("{name:42} {k:?}: Z = {:.3}, fill {:.1} ms", s.input_impedance(), s.t_fill_ms);
        }
    }
}
