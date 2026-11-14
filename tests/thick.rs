//! Thick wires and short segments: the tube (exact) kernel and end caps against
//! published and independently computed thick-dipole data (tests/data/thick_dipole.json).
//!
//! * `series`: Saoudy (1987), Table B-1, Wu/Hurd-type asymptotic solution of Hallén's
//!   equation for a delta-gap tube dipole, ka = 0.01 (Ω ≈ 11.4–12) and 0.04 (Ω ≈ 8.6–9.2).
//! * `hallen`: Hallén's equation with the exact kernel (python/tools/hallen_reference.py),
//!   a different equation, basis and singular treatment, at the same node spacing.
//! * `stepped`, `caps`, `parasitic`: body-of-revolution MoM of the real surfaces
//!   (python/tools/bor_reference.py): a stepped tube with its step face, a solid rod
//!   with flat end caps.
//!
//! `cargo run --release --example thick_dipole` prints the error tables.

use num_complex::Complex64 as C;
use serde_json::Value;
use std::f64::consts::PI;
use wiremom::{C0, Ground, Kernel, Model, Wire};

fn data() -> Value {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/thick_dipole.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// Admittance (mS) of a centre-fed tube dipole, lambda = 1 m, `n` segments.
fn dipole(kh: f64, ka: f64, n: usize, kernel: Kernel, caps: bool) -> C {
    let (h, a) = (kh / (2.0 * PI), ka / (2.0 * PI));
    let mut m = Model::new(vec![Wire::new([0.0, 0.0, -h], [0.0, 0.0, h], a, n)], Ground::Free);
    m.kernel = kernel;
    m.end_caps = caps;
    m.add_source(0, n / 2, C::new(1.0, 0.0));
    1e3 / m.solve(C0).unwrap().input_impedance()
}

fn even_segments(kh: f64, ka: f64, da: f64) -> usize {
    let n = (2.0 * kh / (da * ka)).round() as usize;
    n + n % 2
}

#[test]
fn exact_kernel_matches_published_conductance() {
    // ka = 0.01: the asymptotic series is converged to ~1–2 % (spread of its orders
    // N = 1..4); the tube kernel's conductance stays within that down to segments of
    // half a radius, where the reduced kernel has broken down.  (Susceptance depends
    // on the feed-gap model: a delta gap's capacitance grows as the gap shrinks.)
    let d = data();
    let s = &d["series"][0];
    assert_eq!(f(&s["ka"]), 0.01);
    for r in s["rows"].as_array().unwrap() {
        let kh = f(&r["kh"]);
        if kh > 2.05 {
            continue;
        }
        let g: Vec<f64> = r["G"].as_array().unwrap().iter().map(f).collect();
        let yref = C::new(g[3], f(&r["B"][3]));
        let spread = g.iter().cloned().fold(f64::MIN, f64::max) - g.iter().cloned().fold(f64::MAX, f64::min);
        for da in [2.0, 1.0, 0.5] {
            let n = even_segments(kh, 0.01, da);
            let y = dipole(kh, 0.01, n, Kernel::Exact, false);
            let tol = 2.0 * spread + 0.01 * g[3];
            assert!((y.re - g[3]).abs() < tol, "kh {kh} d/a {da}: G {} vs {} (tol {tol})", y.re, g[3]);
            if da == 0.5 {
                let yr = dipole(kh, 0.01, n, Kernel::Reduced, false);
                assert!((yr - yref).norm() > 0.2 * yref.norm(), "kh {kh}: the reduced kernel should fail at d/a 0.5 ({yr} vs {yref})");
            }
        }
    }
}

#[test]
fn exact_kernel_matches_hallen_solution() {
    // same node spacing: conductance and susceptance (both feed-gap models are a
    // node delta gap with linear current), for both radii
    let d = data();
    let mut worst = 0.0f64;
    for r in d["hallen"].as_array().unwrap() {
        let (ka, kh, m) = (f(&r["ka"]), f(&r["kh"]), r["M"].as_u64().unwrap() as usize);
        let yref = C::new(f(&r["G"]), f(&r["B"]));
        let y = dipole(kh, ka, 2 * m, Kernel::Exact, false);
        let e = (y - yref).norm() / yref.norm();
        worst = worst.max(e);
        // the two discretisations differ at 1–2 % with 8-radius segments and agree
        // to ~0.01 % from 1 radius down
        let tol = if f(&r["da"]) > 6.0 { 2e-2 } else if f(&r["da"]) > 3.0 { 4e-3 } else { 1e-3 };
        assert!(e < tol, "ka {ka} kh {kh} M {m}: {y} vs Hallén {yref} ({:.3} %)", 100.0 * e);
    }
    println!("worst |dY|/|Y| against Hallén: {:.3} %", 100.0 * worst);
}

#[test]
fn stepped_radius_junction_matches_bor() {
    let d = data();
    for r in d["stepped"].as_array().unwrap() {
        let (kh, da, ka1, ka2) = (f(&r["kh"]), f(&r["da"]), f(&r["ka_centre"]), f(&r["ka_tips"]));
        let yref = C::new(f(&r["G"]), f(&r["B"]));
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
        m.end_caps = false;
        m.add_source(1, nc, C::new(1.0, 0.0));
        let y = 1e3 / m.solve(C0).unwrap().input_impedance();
        let e = (y - yref).norm() / yref.norm();
        assert!(e < 1e-2, "kh {kh} d/a {da}: {y} vs BoR {yref} ({:.2} %)", 100.0 * e);
    }
}

#[test]
fn end_caps_match_solid_rod() {
    // open: the same model as the BoR open tube (to ~1e-4); capped: within 0.3 % of
    // the solid rod
    let d = data();
    for r in d["caps"].as_array().unwrap() {
        let (ka, kh, da, caps) = (f(&r["ka"]), f(&r["kh"]), f(&r["da"]), r["caps"].as_bool().unwrap());
        let yref = C::new(f(&r["G"]), f(&r["B"]));
        let y = dipole(kh, ka, even_segments(kh, ka, da), Kernel::Exact, caps);
        let e = (y - yref).norm() / yref.norm();
        let tol = if caps { 3e-3 } else { 5e-4 };
        assert!(e < tol, "ka {ka} kh {kh} d/a {da} caps {caps}: {y} vs BoR {yref} ({:.3} %)", 100.0 * e);
    }
}

#[test]
fn end_caps_are_on_by_default_and_lengthen_elements() {
    let m = Model::new(vec![], Ground::Free);
    assert!(m.end_caps && m.kernel == Kernel::Exact);
    // a capped element resonates lower (it is electrically longer)
    let (o, c) = (dipole(1.45, 0.04, 72, Kernel::Exact, false), dipole(1.45, 0.04, 72, Kernel::Exact, true));
    assert!(c.im < o.im - 0.5, "caps should add inductive susceptance: {c} vs {o}");
}
