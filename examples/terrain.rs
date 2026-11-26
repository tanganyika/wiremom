//! Terrain far field: checks against level ground, the slope model and the
//! second ground's cliff, and a few classic sites.
use num_complex::Complex64 as C;
use wiremom::{Boundary, Ground, Model, Profile, RealGround, SecondGround, Slope, Terrain, Wire};

const F: f64 = 14.2e6;

fn dipole(h: f64) -> Model {
    let mut m = Model::new(vec![Wire::new([0.0, -5.2, h], [0.0, 5.2, h], 0.001, 21)], Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 11, C::new(1.0, 0.0));
    m
}

fn row(name: &str, m: &Model, phi: f64, els: &[f64]) {
    let s = m.solve(F).unwrap();
    let g: Vec<String> = els.iter().map(|e| format!("{:6.1}", s.gain_dbi((90.0 - e).to_radians(), phi).max(-99.0))).collect();
    println!("{name:<34}{}", g.join(""));
}

fn main() {
    let els = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0, 70.0];
    println!("{:<34}{}", "elevation (deg)", els.iter().map(|e| format!("{e:6.0}")).collect::<String>());
    println!("-- level profile vs level ground");
    row("level ground", &dipole(10.0), 0.0, &els);
    let mut m = dipole(10.0);
    m.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [500.0, 0.0]]));
    row("level terrain", &m, 0.0, &els);

    println!("-- 10° slope down toward +x: slope model vs a long sloping profile");
    let a = 10f64.to_radians();
    let mut sl = dipole(10.0);
    sl.slope = Some(Slope::new(a, 0.0));
    row("slope model (downhill)", &sl, 0.0, &els);
    let mut t = sl.clone();
    let l = 1e6;
    t.terrain = Some(Terrain { profiles: vec![Profile::new(0.0, vec![[0.0, 0.0], [l, -a.tan() * l]]), Profile::new(std::f64::consts::PI, vec![[0.0, 0.0], [l, a.tan() * l]])] });
    row("profile (downhill)", &t, 0.0, &els);
    row("slope model (uphill)", &sl, std::f64::consts::PI, &els);
    row("profile (uphill)", &t, std::f64::consts::PI, &els);
    let mut short = sl.clone();
    short.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [300.0, -a.tan() * 300.0]]));
    row("300 m of slope, then level", &short, 0.0, &els);

    println!("-- 10 m cliff 40 m out (same soil): second ground vs profile");
    let mut sg = dipole(10.0);
    let mut g2 = SecondGround::new(RealGround::AVERAGE, Boundary::Line { distance: 40.0, azimuth: 0.0 });
    g2.drop = 10.0;
    sg.second_ground = Some(g2);
    row("second ground, Fresnel", &sg, 0.0, &els);
    let mut cl = dipole(10.0);
    cl.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [40.0, 0.0], [40.0, -10.0]]));
    row("profile with a cliff", &cl, 0.0, &els);

    println!("-- sites");
    let mut hill = dipole(10.0);
    hill.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [100.0, 0.0], [300.0, 30.0], [1000.0, 30.0]]));
    row("hill 30 m high 300 m in front", &hill, 0.0, &els);
    let mut top = dipole(10.0);
    top.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [50.0, 0.0], [1000.0, -100.0]]));
    row("hilltop, 100 m drop over 1 km", &top, 0.0, &els);
    let mut valley = dipole(10.0);
    valley.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [200.0, 0.0], [600.0, 80.0], [2000.0, 80.0]]));
    row("valley, 80 m ridge 600 m out", &valley, 0.0, &els);
}
