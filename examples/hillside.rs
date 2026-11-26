//! A 20 m dipole 10 m up on a hillside: gain toward the valley and back up the
//! hill as the slope grows (infinite slope, average ground).
use num_complex::Complex64 as C;
use wiremom::{Ground, Model, RealGround, Slope, Wire};

fn main() {
    let f = 14.2e6;
    println!("slope   Z (ohm)          downhill (phi 0): el -5  0   3   5  10  20 | peak@el     uphill: el 5  10  20 | peak@el");
    for deg in [0.0, 5.0, 10.0, 15.0, 20.0f64] {
        let w = Wire::new([0.0, -5.2, 10.0], [0.0, 5.2, 10.0], 0.001, 21);
        let mut m = Model::new(vec![w], Ground::Real(RealGround::AVERAGE));
        m.slope = Some(Slope::new(deg.to_radians(), 0.0));
        m.add_source(0, 11, C::new(1.0, 0.0));
        let s = m.solve(f).unwrap();
        let g = |el: f64, az: f64| s.gain_dbi((90.0 - el).to_radians(), az.to_radians());
        let peak = |az: f64| (-30..=900).map(|i| i as f64 / 10.0).map(|e| (g(e, az), e)).fold((-999.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        let z = s.input_impedance();
        let (pd, ed) = peak(0.0);
        let (pu, eu) = peak(180.0);
        let d: Vec<String> = [-5.0, 0.0, 3.0, 5.0, 10.0, 20.0].iter().map(|&e| format!("{:5.1}", g(e, 0.0).max(-99.0))).collect();
        let u: Vec<String> = [5.0, 10.0, 20.0].iter().map(|&e| format!("{:5.1}", g(e, 180.0).max(-99.0))).collect();
        println!("{deg:4.0}°  {:6.1}{:+7.1}j   {}  | {pd:4.1}@{ed:4.1}°   {}  | {pu:4.1}@{eu:4.1}°", z.re, z.im, d.join(" "), u.join(" "));
    }
}
