//! Second ground medium (sea beyond x = 30 m, with and without a 5 m drop) against
//! nec2c's GD card.  Needs the nec2c pattern dumps; kept for reference.
use num_complex::Complex64 as C;
use wiremom::{Boundary, Ground, Model, RealGround, SecondGround, Wire};
fn main() {
    let dir = std::env::var("CLIFF_DIR").unwrap_or_else(|_| "/tmp/claude-1000/-home-rbaker-GitHub-BambuStudio/042ee934-76dc-4a7a-b7e5-dc00fcb5b89f/scratchpad/cliff".into());
    for geo in ["h", "v"] {
        for drop in [0.0, 5.0] {
            let w = if geo == "h" { Wire::new([0.0, -5.25, 10.0], [0.0, 5.25, 10.0], 0.002, 21) } else { Wire::new([0.0, 0.0, 4.7], [0.0, 0.0, 15.2], 0.002, 21) };
            let mut m = Model::new(vec![w], Ground::Real(RealGround::AVERAGE));
            m.add_source(0, 11, C::new(1.0, 0.0));
            let mut sg = SecondGround::new(RealGround { eps_r: 81.0, sigma: 5.0 }, Boundary::Line { distance: 30.0, azimuth: 0.0 });
            sg.drop = drop;
            sg.fresnel = false;
            m.second_ground = Some(sg);
            let s = m.solve(14.2e6).unwrap();
            m.second_ground.as_mut().unwrap().fresnel = true;
            let fz = m.solve(14.2e6).unwrap();
            let txt = std::fs::read_to_string(format!("{dir}/{geo}{}.txt", drop as i32)).unwrap();
            let mut worst: f64 = 0.0;
            for l in txt.lines() {
                let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
                let (th, ph, g) = (v[0].to_radians(), v[1].to_radians(), v[2]);
                let gw = s.gain_dbi(th, ph);
                let gf = fz.gain_dbi(th, ph);
                if (gw - g).abs() > 0.3 || (v[0] as i32) % 10 == 0 {
                    println!("{geo} drop {drop}: theta {:.0} phi {:.0}: nec2c {g:.2} wiremom {gw:.2} fresnel {gf:.2}", v[0], v[1]);
                }
                if g > -20.0 { worst = worst.max((gw - g).abs()); }
            }
            println!("== {geo} drop {drop}: worst {worst:.2} dB");
        }
    }
}
#[allow(dead_code)]
pub fn timing() {
    let w = Wire::new([0.0, 0.0, 4.7], [0.0, 0.0, 15.2], 0.002, 21);
    let mut m = Model::new(vec![w], Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 11, C::new(1.0, 0.0));
    let s0 = m.solve(14.2e6).unwrap();
    m.second_ground = Some(SecondGround::new(RealGround { eps_r: 81.0, sigma: 5.0 }, Boundary::Line { distance: 30.0, azimuth: 0.0 }));
    let s1 = m.solve(14.2e6).unwrap();
    for (name, s) in [("plain", &s0), ("second ground", &s1)] {
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        for el in 0..31 { for az in 0..120 { acc += s.gain((90.0 - 3.0 * el as f64).to_radians(), (3.0 * az as f64).to_radians()).total; } }
        println!("{name}: 3720 directions in {:.1} ms ({acc:.3})", t.elapsed().as_secs_f64() * 1e3);
    }
}
