//! 20 m quarter-wave vertical over average ground, 16 radials (bare filament,
//! radius a, at z_ref) with an extra near-field potential per unit charge dp from a
//! 2-D cross-section.  Args: z_ref dp_re dp_im [len] [nrad].  Prints Z and efficiency.
use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::*;
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (z, dp) = (a[0], C::new(a[1], a[2]));
    let len = *a.get(3).unwrap_or(&5.0);
    let nrad = *a.get(4).unwrap_or(&16.0) as usize;
    let f = 14.15e6;
    let cu = Conductor::Lossy { sigma: 5.8e7, mu_r: 1.0 };
    let mut w = vec![];
    if std::env::var("DIPOLE").is_ok() {
        // a 10 m centre-fed wire lying along the surface
        let mut d = Wire::new([-5.0, 0.0, z], [5.0, 0.0, z], 0.0005, 40).with_conductor(cu);
        d.extra_p = dp;
        let mut m = Model::new(vec![d], Ground::Real(RealGround::AVERAGE));
        m.add_source(0, 20, C::new(1.0, 0.0));
        let s = m.solve(f).unwrap();
        let zi = s.input_impedance();
        println!("{} {} {}", zi.re, zi.im, s.efficiency());
        return;
    }
    if z > 0.0 {
        w.push(Wire::new([0.0, 0.0, z], [0.0, 0.0, 5.1], 0.005, 11).with_conductor(cu));
    } else {
        w.push(Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 5.1], 0.005, 11).with_conductor(cu));
        w.push(Wire::new([0.0, 0.0, z], [0.0, 0.0, 0.0], 0.0005, 1).with_conductor(cu));
    }
    let ns = (len / 0.25).round() as usize;
    if let Ok(zo) = std::env::var("OUTER_Z") {
        // hub in the air at z (z > 0); each radial runs out 2 cm at z, then goes to
        // z_out (a short slope) and on at z_out with the correction dp
        let zo: f64 = zo.parse().unwrap();
        let r1: f64 = std::env::var("R1").map(|v| v.parse().unwrap()).unwrap_or(0.02); let r2 = r1 + 0.01;
        for i in 0..nrad {
            let t = 2.0 * PI * i as f64 / nrad as f64;
            let (c, sn) = (t.cos(), t.sin());
            w.push(Wire::new([0.0, 0.0, z], [r1 * c, r1 * sn, z], 0.0005, 1).with_conductor(cu));
            w.push(Wire::new([r1 * c, r1 * sn, z], [r2 * c, r2 * sn, zo], 0.0005, 1).with_conductor(cu));
            let mut r = Wire::new([r2 * c, r2 * sn, zo], [len * c, len * sn, zo], 0.0005, ns).with_conductor(cu);
            r.extra_p = dp;
            w.push(r);
        }
        let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE)); if std::env::var("NOCAPS").is_ok() { m.end_caps = false; }
        m.add_source(0, 0, C::new(1.0, 0.0));
        let s = m.solve(f).unwrap();
        let zi = s.input_impedance();
        println!("{} {} {}", zi.re, zi.im, s.efficiency());
        if std::env::var("PRINTI").is_ok() {
            let i_in = s.source_currents()[0];
            for (p0, p1, i) in s.segment_currents() {
                let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
                if mid[1].abs() < 1e-9 && mid[0] > 0.0 && (p1[2] - p0[2]).abs() < 1e-6 {
                    let r = i / i_in;
                    println!("I {:.3} {:.4} {:.1}", mid[0], r.norm(), r.arg().to_degrees());
                }
            }
            println!("L {:.4} {:.4}", s.losses.wire / s.input_power(), 1.0);
        }
        return;
    }
    for i in 0..nrad {
        let t = 2.0 * PI * i as f64 / nrad as f64;
        let mut r = Wire::new([0.0, 0.0, z], [len * t.cos(), len * t.sin(), z], 0.0005, ns).with_conductor(cu);
        r.extra_p = dp;
        w.push(r);
    }
    let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE)); if std::env::var("NOCAPS").is_ok() { m.end_caps = false; }
    m.add_source(0, 0, C::new(1.0, 0.0));
    let s = m.solve(f).unwrap();
    let zi = s.input_impedance();
    println!("{} {} {}", zi.re, zi.im, s.efficiency());
}
