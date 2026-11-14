//! Stepped-radius dipole (centre half of the length at ka = 0.04, tips ka = 0.02)
//! against the body-of-revolution reference in tests/data/thick_dipole.json.
//!
//! cargo run --release --example stepped_dipole
use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::*;

pub fn stepped(kh: f64, ka1: f64, ka2: f64, da: f64, kernel: Kernel) -> C {
    let (h, a1, a2) = (kh / (2.0 * PI), ka1 / (2.0 * PI), ka2 / (2.0 * PI));
    let zs = 0.5 * h;
    let n = |len: f64, a: f64| ((len / (da * a)).round() as usize).max(1);
    let (nt, nc) = (n(h - zs, a2), n(zs, a1));
    let w = vec![
        Wire::new([0.0, 0.0, -h], [0.0, 0.0, -zs], a2, nt),
        Wire::new([0.0, 0.0, -zs], [0.0, 0.0, 0.0], a1, nc),
        Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, zs], a1, nc),
        Wire::new([0.0, 0.0, zs], [0.0, 0.0, h], a2, nt),
    ];
    let mut m = Model::new(w, Ground::Free);
    m.kernel = kernel;
    m.add_source(1, nc, C::new(1.0, 0.0));
    1e3 / m.solve(C0).unwrap().input_impedance()
}

fn main() {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/thick_dipole.json")).unwrap();
    let d: serde_json::Value = serde_json::from_str(&s).unwrap();
    println!("{:>5} {:>4} {:>18} | {:>18} {:>18}", "kh", "d/a", "BoR G+jB mS", "reduced", "exact");
    for r in d["stepped"].as_array().unwrap() {
        let (kh, da) = (r["kh"].as_f64().unwrap(), r["da"].as_f64().unwrap());
        let y = C::new(r["G"].as_f64().unwrap(), r["B"].as_f64().unwrap());
        print!("{kh:>5.2} {da:>4} {:>9.3}{:+8.3}j |", y.re, y.im);
        for k in [Kernel::Reduced, Kernel::Exact] {
            let yc = stepped(kh, r["ka_centre"].as_f64().unwrap(), r["ka_tips"].as_f64().unwrap(), da, k);
            print!(" {:>9.3}{:+8.3}j", yc.re, yc.im);
        }
        println!();
    }
}
