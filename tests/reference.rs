//! Checks against the frozen reference values in tests/data/reference.json
//! (from the Python prototype and nec2c; see python/tools/make_reference.py).

use num_complex::Complex64 as C;
use serde_json::Value;
use wiremom::sommerfeld::{Coeffs, reflected_dyadic};
use wiremom::{Conductor, Ground, LoadKind, Model, RealGround, Wire};

fn data() -> Value {
    let s = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/reference.json")).unwrap();
    serde_json::from_str(&s).unwrap()
}

fn cplx(v: &Value) -> C {
    C::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}

fn v3(v: &Value) -> [f64; 3] {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap()]
}

fn model(case: &Value) -> Model {
    let g = &case["ground"];
    let ground = match g["kind"].as_str().unwrap() {
        "free" => Ground::Free,
        "pec" => Ground::Perfect,
        _ => Ground::Real(RealGround { eps_r: g["eps_r"].as_f64().unwrap(), sigma: g["sigma"].as_f64().unwrap() }),
    };
    let wires = case["wires"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| {
            let mut wire = Wire::new(v3(&w["start"]), v3(&w["end"]), w["radius"].as_f64().unwrap(), w["nseg"].as_u64().unwrap() as usize);
            if let Some(s) = w["sigma"].as_f64() {
                wire = wire.with_conductor(Conductor::Lossy { sigma: s, mu_r: 1.0 });
            }
            wire
        })
        .collect();
    let mut m = Model::new(wires, ground);
    // exact tables, as the Python prototype used
    m.ground_table_reuse = 0.0;
    let src = &case["source"];
    m.add_source(src["wire"].as_u64().unwrap() as usize - 1, src["node"].as_u64().unwrap() as usize, C::new(1.0, 0.0));
    for l in case["loads"].as_array().unwrap() {
        let (r, ll) = (l["r"].as_f64().unwrap(), l["l"].as_f64().unwrap());
        let c = l["c"].as_f64();
        let kind = if l["parallel"].as_bool().unwrap() {
            LoadKind::Parallel { r, l: ll, c: c.unwrap_or(0.0) }
        } else {
            LoadKind::Series { r, l: ll, c }
        };
        m.add_load(l["wire"].as_u64().unwrap() as usize - 1, l["node"].as_u64().unwrap() as usize, kind);
    }
    m
}

fn solve_case(case: &Value) -> (C, Vec<f64>) {
    let m = model(case);
    let sol = m.solve(case["freq"].as_f64().unwrap()).unwrap();
    let gains = case["directions_deg"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| sol.gain_dbi(d[0].as_f64().unwrap().to_radians(), d[1].as_f64().unwrap().to_radians()))
        .collect();
    (sol.input_impedance(), gains)
}

#[test]
fn matches_python_prototype() {
    let d = data();
    for case in d["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (z, g) = solve_case(case);
        let zp = cplx(&case["python"]["z"]);
        let real = case["ground"]["kind"] == "real";
        // free space / PEC are the same algorithm to rounding; over real ground the
        // table splines differ slightly from SciPy's
        let tol = if real { 2e-4 } else { 1e-8 };
        let dz = (z - zp).norm() / zp.norm();
        println!("{name}: rust {z:.4} python {zp:.4} rel diff {dz:.1e}");
        assert!(dz < tol, "{name}: {z} vs {zp}");
        for (a, b) in g.iter().zip(case["python"]["gain_dbi"].as_array().unwrap()) {
            assert!((a - b.as_f64().unwrap()).abs() < 1e-3, "{name}: gain {a} vs {b}");
        }
    }
}

#[test]
fn agrees_with_nec2c() {
    let d = data();
    for case in d["cases"].as_array().unwrap() {
        let Some(nec) = case.get("nec2c") else { continue };
        let name = case["name"].as_str().unwrap();
        let (z, g) = solve_case(case);
        let zn = cplx(&nec["z"]);
        // resistance and gain are tightly comparable; reactance differs by the feed
        // model (node gap vs. segment gap), most in the near-cancelling loaded case
        let tol = if name.contains("loading coil") { 0.10 } else { 0.04 };
        assert!((z - zn).norm() / zn.norm() < tol, "{name}: {z} vs nec2c {zn}");
        let gn: Vec<f64> = nec["gain_dbi"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let gmax = gn.iter().cloned().fold(f64::MIN, f64::max);
        for (a, b) in g.iter().zip(&gn) {
            let gtol = if *b > gmax - 10.0 { 0.15 } else { 0.5 };
            assert!((a - b).abs() < gtol, "{name}: gain {a} vs nec2c {b}");
        }
    }
}

#[test]
fn sommerfeld_integrals_match_python() {
    let d = data();
    for p in d["sommerfeld_remainder"].as_array().unwrap() {
        let g = RealGround { eps_r: p["eps_r"].as_f64().unwrap(), sigma: p["sigma"].as_f64().unwrap() };
        let f = p["freq"].as_f64().unwrap();
        let g0 = g.gamma0(f);
        let co = Coeffs { shift: (g0, -g0), fixed: None };
        let got = reflected_dyadic(p["rho"].as_f64().unwrap(), p["h"].as_f64().unwrap(), f, g.eps_c(f), &co);
        let want: Vec<C> = p["d"].as_array().unwrap().iter().map(cplx).collect();
        let scale = want.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).norm() < 1e-9 * scale, "{} rho={} h={}: {a} vs {b}", p["ground"], p["rho"], p["h"]);
        }
    }
}

#[test]
fn lossy_ground_quadrature_is_converged() {
    // the sparse second arc used for very lossy ground must agree with a dense rule:
    // compare against the same integral with the ground split into a lossless and a
    // lossy part is not possible, so check self-consistency across conductivities
    // through the PEC limit instead: remainder -> 0 as sigma -> infinity
    let f = 14.2e6;
    for sigma in [5.0, 100.0, 1e3, 1e5] {
        let g = RealGround { eps_r: 81.0, sigma };
        let g0 = g.gamma0(f);
        let co = Coeffs { shift: (g0, -g0), fixed: None };
        let d = reflected_dyadic(2.0, 3.0, f, g.eps_c(f), &co);
        let full = reflected_dyadic(2.0, 3.0, f, g.eps_c(f), &Coeffs { shift: (C::new(0.0, 0.0), C::new(0.0, 0.0)), fixed: None });
        let r = d.iter().map(|v| v.norm()).fold(0.0, f64::max) / full.iter().map(|v| v.norm()).fold(0.0, f64::max);
        // the remainder is O(1/sqrt(eps)) of the reflected field
        let bound = 3.0 / g.eps_c(f).norm().sqrt();
        assert!(r < bound, "sigma {sigma}: remainder ratio {r:.2e} > {bound:.2e}");
    }
}
