//! A second ground medium (sea beyond a shore, with or without a cliff) for the
//! far field, against nec2c's GD card and limiting cases.

use num_complex::Complex64 as C;
use wiremom::{Boundary, Ground, Model, RealGround, SecondGround, Wire};

const F: f64 = 14.2e6;
const SEA: RealGround = RealGround { eps_r: 81.0, sigma: 5.0 };

fn model(vertical: bool, drop: f64, fresnel: bool, boundary: Boundary) -> Model {
    let w = if vertical { Wire::new([0.0, 0.0, 4.7], [0.0, 0.0, 15.2], 0.002, 21) } else { Wire::new([0.0, -5.25, 10.0], [0.0, 5.25, 10.0], 0.002, 21) };
    let mut m = Model::new(vec![w], Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 11, C::new(1.0, 0.0));
    let mut sg = SecondGround::new(SEA, boundary);
    sg.drop = drop;
    sg.fresnel = fresnel;
    m.second_ground = Some(sg);
    m
}

#[test]
fn specular_mode_matches_nec2c_linear_cliff() {
    // nec2c: GN 2 (13, 0.005), GD 81 5 30 drop, RP 2 (linear cliff at x = 30 m)
    let line = Boundary::Line { distance: 30.0, azimuth: 0.0 };
    let cases: [(bool, f64, &[(f64, f64, f64)]); 4] = [
        (false, 0.0, &[(30.0, 0.0, 0.86), (60.0, 0.0, 7.13), (70.0, 0.0, 6.19), (80.0, 180.0, 1.90)]),
        (false, 5.0, &[(30.0, 0.0, 0.86), (60.0, 0.0, 7.13), (80.0, 0.0, 5.11), (80.0, 180.0, 1.90)]),
        (true, 0.0, &[(30.0, 0.0, -2.52), (60.0, 0.0, -2.97), (80.0, 0.0, 6.24), (70.0, 180.0, 0.12)]),
        (true, 5.0, &[(30.0, 0.0, -2.52), (60.0, 0.0, -2.97), (80.0, 0.0, 4.36), (80.0, 180.0, 0.45)]),
    ];
    for (vertical, drop, want) in cases {
        let s = model(vertical, drop, false, line).solve(F).unwrap();
        for &(th, ph, g) in want {
            let got = s.gain_dbi(th.to_radians(), ph.to_radians());
            assert!((got - g).abs() < 0.1, "vertical {vertical} drop {drop} theta {th} phi {ph}: {got:.2} vs nec2c {g}");
        }
    }
}

#[test]
fn far_boundary_changes_nothing_and_a_boundary_at_the_origin_gives_medium_two() {
    for fresnel in [false, true] {
        let mut m = model(true, 0.0, fresnel, Boundary::Circle { radius: 1e7 });
        let far = m.solve(F).unwrap();
        m.second_ground = None;
        let one = m.solve(F).unwrap();
        // the same currents with the sea's reflection everywhere
        let mut sea = m.solve(F).unwrap();
        sea.ground = Ground::Real(SEA);
        let all = model(true, 0.0, fresnel, Boundary::Circle { radius: 0.0 }).solve(F).unwrap();
        for th in [20.0f64, 50.0, 75.0, 85.0] {
            let t = th.to_radians();
            assert!((far.gain_dbi(t, 0.3) - one.gain_dbi(t, 0.3)).abs() < 1e-3, "{th}");
            let tol = 1e-9;
            assert!((all.gain_dbi(t, 0.3) - sea.gain_dbi(t, 0.3)).abs() < tol, "fresnel {fresnel} theta {th}: {} vs {}", all.gain_dbi(t, 0.3), sea.gain_dbi(t, 0.3));
        }
    }
}

#[test]
fn fresnel_mode_is_continuous_and_tends_to_specular_for_high_antennas() {
    // the shore in the first Fresnel zone: smooth in elevation, no step
    let line = Boundary::Line { distance: 30.0, azimuth: 0.0 };
    let s = model(false, 0.0, true, line).solve(F).unwrap();
    let g: Vec<f64> = (300..850).map(|i| s.gain_dbi((i as f64 * 0.1).to_radians(), 0.0)).collect();
    let (at, jump) = g.windows(2).map(|w| (w[1] - w[0]).abs()).enumerate().fold((0, 0.0), |a, (i, v)| if v > a.1 { (i, v) } else { a });
    assert!(jump < 0.2, "largest step between 0.1 degree samples {jump} at theta {}", 30.0 + 0.1 * at as f64);
    // a dipole 30 wavelengths up, the shore 2 km out: rays reflecting many
    // Fresnel zones from it see the same ground in both modes
    let lam = 299.792458e6 / F;
    let hi = |fresnel: bool| {
        let mut m = Model::new(vec![Wire::new([0.0, -0.25 * lam, 30.0 * lam], [0.0, 0.25 * lam, 30.0 * lam], 0.002, 21)], Ground::Real(RealGround::AVERAGE));
        m.add_source(0, 11, C::new(1.0, 0.0));
        let mut sg = SecondGround::new(SEA, Boundary::Line { distance: 2000.0, azimuth: 0.0 });
        sg.fresnel = fresnel;
        m.second_ground = Some(sg);
        m.solve(F).unwrap()
    };
    let (a, b) = (hi(true), hi(false));
    // specular points 0.63 and 0.37 km out (land), 7 km (sea)
    for el in [45.0f64, 60.0, 5.0] {
        let t = (90.0 - el).to_radians();
        let (x, y) = (a.gain_dbi(t, 0.0), b.gain_dbi(t, 0.0));
        assert!((x - y).abs() < 0.1, "el {el}: {x} vs {y}");
    }
}
