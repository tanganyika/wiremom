//! Error tables for the reduced and exact (tube) kernels on a centre-fed tube dipole,
//! against segment length / radius (tests/data/thick_dipole.json):
//!
//! 1. conductance against the published asymptotic solution (Saoudy 1987, N = 4);
//! 2. admittance against an independent exact-kernel Hallén solution with the same
//!    node spacing, so the same delta-gap feed (a delta gap's susceptance grows as
//!    the gap shrinks; comparing at equal spacing leaves only the kernel's error).
//!
//! cargo run --release --example thick_dipole
use num_complex::Complex64 as C;
use serde_json::Value;
use std::f64::consts::PI;
use wiremom::*;

fn admittance(kh: f64, ka: f64, n: usize, kernel: Kernel) -> C {
    let (h, a) = (kh / (2.0 * PI), ka / (2.0 * PI));
    let mut m = Model::new(vec![Wire::new([0.0, 0.0, -h], [0.0, 0.0, h], a, n)], Ground::Free);
    m.kernel = kernel;
    m.end_caps = false;
    m.add_source(0, n / 2, C::new(1.0, 0.0));
    1e3 / m.solve(C0).unwrap().input_impedance()
}

const DAS: [f64; 5] = [8.0, 4.0, 2.0, 1.0, 0.5];

fn header(title: &str) {
    println!("\n{title}");
    print!("{:>5} {:>6} {:>8} |", "kh", "Omega", "ref");
    for k in ["reduced", "exact"] {
        for da in DAS {
            print!(" {:>3}{da:<3}", &k[..3]);
        }
        print!(" |");
    }
    println!();
}

fn main() {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/thick_dipole.json")).unwrap();
    let d: Value = serde_json::from_str(&s).unwrap();
    let f = |v: &Value| v.as_f64().unwrap();
    for series in d["series"].as_array().unwrap() {
        let ka = f(&series["ka"]);
        header(&format!("ka = {ka}: conductance error % against Saoudy (1987) N = 4, by segment length / radius"));
        for row in series["rows"].as_array().unwrap() {
            let kh = f(&row["kh"]);
            if kh > 2.05 {
                continue;
            }
            let g = f(&row["G"][3]);
            print!("{kh:>5.2} {:>6.2} {g:>8.3} |", 2.0 * (2.0 * kh / ka).ln());
            for kernel in [Kernel::Reduced, Kernel::Exact] {
                for da in DAS {
                    let n = (2.0 * kh / (da * ka)).round() as usize;
                    let y = admittance(kh, ka, n + n % 2, kernel);
                    print!(" {:>+6.2}", 100.0 * (y.re - g) / g);
                }
                print!(" |");
            }
            println!();
        }
        header(&format!("ka = {ka}: |Y - Y_Hallén| / |Y_Hallén| % at the same node spacing"));
        let rows: Vec<&Value> = d["hallen"].as_array().unwrap().iter().filter(|r| f(&r["ka"]) == ka).collect();
        let mut khs: Vec<f64> = rows.iter().map(|r| f(&r["kh"])).collect();
        khs.dedup();
        for kh in khs {
            print!("{kh:>5.2} {:>6.2} {:>8} |", 2.0 * (2.0 * kh / ka).ln(), "");
            for kernel in [Kernel::Reduced, Kernel::Exact] {
                for da in DAS {
                    match rows.iter().find(|r| f(&r["kh"]) == kh && (f(&r["da"]) / da - 1.0).abs() < 0.1) {
                        Some(r) => {
                            let yr = C::new(f(&r["G"]), f(&r["B"]));
                            let y = admittance(kh, ka, 2 * r["M"].as_u64().unwrap() as usize, kernel);
                            print!(" {:>6.2}", 100.0 * (y - yr).norm() / yr.norm());
                        }
                        None => print!(" {:>6}", "-"),
                    }
                }
                print!(" |");
            }
            println!();
        }
    }
}
