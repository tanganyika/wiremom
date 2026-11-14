//! Solve an antenna exported by wireant's `get_antenna` (JSON) with each kernel and
//! cap setting and several segment densities: how much thick conductors move it.
//!
//! cargo run --release --example wireant_json -- antenna.json [MHz ...]
use num_complex::Complex64 as C;
use serde_json::Value;
use wiremom::*;

fn model(a: &Value, seg_scale: f64, kernel: Kernel, caps: bool) -> Model {
    let v3 = |v: &Value| [v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap()];
    let wires: Vec<Wire> = a["wires"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| Wire::new(v3(&w["p1"]), v3(&w["p2"]), w["r"].as_f64().unwrap(), ((w["seg"].as_f64().unwrap() * seg_scale).round() as usize).max(1)))
        .collect();
    let mut m = Model::new(wires.clone(), Ground::Free);
    m.kernel = kernel;
    m.end_caps = caps;
    for s in a["sources"].as_array().unwrap() {
        let wi = s["wire"].as_u64().unwrap() as usize;
        let n = match &wires[wi].segments {
            Segments::Uniform(n) => *n,
            Segments::Lengths(l) => l.len(),
        };
        let node = (s["pos"].as_f64().unwrap() * n as f64).round() as usize;
        m.add_source(wi, node.clamp(1, n - 1), C::new(s["volts"].as_f64().unwrap(), 0.0));
    }
    m
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&args[0]).unwrap()).unwrap();
    let a = v.get("antenna").unwrap_or(&v);
    let freqs: Vec<f64> = if args.len() > 1 { args[1..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![a["freq_mhz"].as_f64().unwrap()] };
    for (label, kernel, caps) in [("reduced, open ends", Kernel::Reduced, false), ("exact, open ends", Kernel::Exact, false), ("exact, end caps", Kernel::Exact, true)] {
        for scale in std::env::var("SCALES").map(|s| s.split(",").map(|x| x.parse().unwrap()).collect::<Vec<f64>>()).unwrap_or(vec![0.5, 1.0, 2.0]) {
            let m = model(a, scale, kernel, caps);
            print!("{label:20} segs x{scale:<3}:");
            for f in &freqs {
                let z = m.solve(f * 1e6).unwrap().input_impedance();
                print!("  {f} MHz {:7.2}{:+7.2}j", z.re, z.im);
            }
            println!();
        }
    }
}
