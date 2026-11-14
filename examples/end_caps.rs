//! Flat end caps: wiremom's open and capped dipole against the body-of-revolution
//! reference (tests/data/thick_dipole.json, "caps"), then the cap's effect on a
//! thick Yagi element's resonance.
//!
//! cargo run --release --example end_caps
use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::*;

fn dipole(kh: f64, ka: f64, da: f64, caps: bool) -> C {
    let (h, a) = (kh / (2.0 * PI), ka / (2.0 * PI));
    let mut n = (2.0 * h / (da * a)).round() as usize;
    n += n % 2;
    let mut m = Model::new(vec![Wire::new([0.0, 0.0, -h], [0.0, 0.0, h], a, n)], Ground::Free);
    m.end_caps = caps;
    m.add_source(0, n / 2, C::new(1.0, 0.0));
    1e3 / m.solve(C0).unwrap().input_impedance()
}

fn main() {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/thick_dipole.json")).unwrap();
    let d: serde_json::Value = serde_json::from_str(&s).unwrap();
    println!("{:>5} {:>5} {:>4} {:>5} | {:>18} {:>18} | {:>8}", "ka", "kh", "d/a", "caps", "BoR G+jB mS", "wiremom", "|dY|/|Y|");
    for r in d["caps"].as_array().unwrap() {
        let (ka, kh, da, caps) = (r["ka"].as_f64().unwrap(), r["kh"].as_f64().unwrap(), r["da"].as_f64().unwrap(), r["caps"].as_bool().unwrap());
        let y = C::new(r["G"].as_f64().unwrap(), r["B"].as_f64().unwrap());
        let yc = dipole(kh, ka, da, caps);
        println!("{ka:>5} {kh:>5} {da:>4} {caps:>5} | {:>9.3}{:+8.3}j {:>9.3}{:+8.3}j | {:>7.2}%", y.re, y.im, yc.re, yc.im, 100.0 * (yc - y).norm() / y.norm());
    }

    // parasitic resonance (no feed gap): a tube driven by a distant probe dipole,
    // resonant where its centre current is in phase with the incident field;
    // against the body-of-revolution plane-wave resonances in tests/data
    println!("\nparasitic tube, ka = 0.04: resonant kh (BoR open {}, capped {})", d["parasitic"]["open"], d["parasitic"]["capped"]);
    println!("{:>5} {:>10} {:>10}", "d/a", "open", "capped");
    for da in [2.0, 1.0, 0.5] {
        println!("{da:>5} {:>10.5} {:>10.5}", parasitic_kh(0.04, da, false), parasitic_kh(0.04, da, true));
    }
    // a 432 MHz Yagi element of 6.25 mm rod (ka = 0.0283): its resonant length
    let ka = 2.0 * PI * 0.003125 / (C0 / 432e6);
    println!("\n6.25 mm rod at 432 MHz: resonant parasitic length (mm)");
    println!("{:>5} {:>8} {:>8} {:>8}", "d/a", "open", "capped", "change");
    for da in [2.0, 1.0, 0.5] {
        let l = |caps: bool| parasitic_kh(ka, da, caps) / PI * (C0 / 432e6) * 1e3;
        let (o, c) = (l(false), l(true));
        println!("{da:>5} {o:>8.2} {c:>8.2} {:>+7.2}%", 100.0 * (c - o) / o);
    }
}

/// Resonant kh of a parasitic tube (lambda = 1 m), by bisection on its length.
pub fn parasitic_kh(ka: f64, da: f64, caps: bool) -> f64 {
    let a = ka / (2.0 * PI);
    let k = 2.0 * PI;
    let dist = 50.0;
    let phase = |kh: f64| {
        let h = kh / k;
        let mut n = (2.0 * h / (da * a)).round() as usize;
        n += n % 2;
        let lp = 0.05;
        let w = vec![Wire::new([0.0, 0.0, -h], [0.0, 0.0, h], a, n), Wire::new([dist, 0.0, -lp / 2.0], [dist, 0.0, lp / 2.0], 1e-4, 4)];
        let mut m = Model::new(w, Ground::Free);
        m.end_caps = caps;
        m.add_source(1, 2, C::new(1.0, 0.0));
        let s = m.solve(C0).unwrap();
        let cur = s.segment_currents();
        let ie = 0.5 * (cur[n / 2 - 1].2 + cur[n / 2].2);
        let mom: C = s.current_elements(4).iter().skip(4 * n).map(|e| e.1[2]).sum();
        let kd = k * dist;
        // E_z of a short dipole broadside, to a constant factor
        let e = -C::new(0.0, 1.0) * mom * (C::new(0.0, -kd)).exp() / dist * (1.0 + 1.0 / C::new(0.0, kd) - 1.0 / (kd * kd));
        (ie / e).im
    };
    let (mut lo, mut hi) = (1.3, 1.55);
    let flo = phase(lo);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if (phase(mid) > 0.0) == (flo > 0.0) { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}
