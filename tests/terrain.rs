//! Terrain for the far field: exact limits (level terrain, an endless slope), the
//! second ground's cliff, and the effects a terrain model must show (a ridge in
//! front cuts low angles, a hilltop raises them).

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::terrain::knife_edge;
use wiremom::{Boundary, Ground, Model, Profile, RealGround, SecondGround, Slope, Terrain, Wire};

const F: f64 = 14.2e6;

fn dipole(ground: Ground) -> Model {
    let mut m = Model::new(vec![Wire::new([0.0, -5.2, 10.0], [0.0, 5.2, 10.0], 0.001, 21)], ground);
    m.add_source(0, 11, C::new(1.0, 0.0));
    m
}

/// A quarter-wave vertical over four buried radials (currents in the ground too).
fn vertical_with_radials() -> Model {
    let l = 5.0;
    let mut w = vec![Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, l], 0.002, 10)];
    for i in 0..4 {
        let a = i as f64 * PI / 2.0;
        w.push(Wire::new([0.0, 0.0, -0.1], [l * a.cos(), l * a.sin(), -0.1], 0.002, 10));
    }
    w.push(Wire::new([0.0, 0.0, -0.1], [0.0, 0.0, 0.0], 0.002, 1));
    let mut m = Model::new(w, Ground::Real(RealGround::AVERAGE));
    m.add_source(0, 0, C::new(1.0, 0.0));
    m
}

fn el(deg: f64) -> f64 {
    (90.0 - deg).to_radians()
}

#[test]
fn level_terrain_is_level_ground() {
    let models = [dipole(Ground::Real(RealGround::AVERAGE)), dipole(Ground::Perfect), vertical_with_radials()];
    for m in models {
        let flat = m.solve(F).unwrap();
        let mut t = m.clone();
        t.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [80.0, 0.0], [500.0, 0.0]]));
        let s = t.solve(F).unwrap();
        for e in [0.5, 2.0, 7.0, 20.0, 45.0, 89.0] {
            for ph in [0.0, 1.0, 2.5, 4.0] {
                let (a, b) = (s.gain(el(e), ph), flat.gain(el(e), ph));
                assert!((a.total - b.total).abs() <= 1e-9 * b.total.max(1e-12), "el {e} phi {ph}: {} vs {}", a.total, b.total);
                assert!((a.theta - b.theta).abs() <= 1e-9 * b.total.max(1e-12));
            }
        }
        // nothing at or below the horizon
        assert_eq!(s.gain(el(0.0), 0.0).total, 0.0);
        assert_eq!(s.gain(el(-3.0), 0.0).total, 0.0);
    }
}

#[test]
fn endless_slope_matches_the_slope_model() {
    // currents over a 10° slope falling toward +x; the terrain is the same slope,
    // long enough that the level ground at its end doesn't reflect these rays
    let a = 10f64.to_radians();
    let mut m = dipole(Ground::Real(RealGround::AVERAGE));
    m.slope = Some(Slope::new(a, 0.0));
    let slope = m.solve(F).unwrap();
    let l = 1e6;
    m.terrain = Some(Terrain { profiles: vec![Profile::new(0.0, vec![[0.0, 0.0], [l, -a.tan() * l]]), Profile::new(PI, vec![[0.0, 0.0], [l, a.tan() * l]])] });
    let t = m.solve(F).unwrap();
    for (phi, from) in [(0.0, 13.0), (PI, 13.0)] {
        for i in 0..=38 {
            let e = from + 2.0 * i as f64;
            if e >= 89.0 {
                break;
            }
            let (x, y) = (t.gain_dbi(el(e), phi), slope.gain_dbi(el(e), phi));
            assert!((x - y).abs() < 0.05, "phi {phi} el {e}: terrain {x:.3} vs slope {y:.3}");
        }
    }
    // uphill, below the slope, almost nothing comes over the far end
    assert!(t.gain_dbi(el(5.0), PI) < -30.0);
}

#[test]
fn cliff_agrees_with_the_second_ground() {
    // a 10 m drop 40 m out, the same soil above and below; the two models treat
    // the edge differently (measured: 1.2 dB above 4° for a horizontal dipole,
    // 1.5 dB above 6° for a vertical; near grazing they part by up to 9 dB, where
    // the second ground's quadratic phase underrates the edge)
    for (vertical, from, tol) in [(false, 4.0, 1.5), (true, 6.0, 1.8)] {
        let w = if vertical { Wire::new([0.0, 0.0, 4.7], [0.0, 0.0, 15.2], 0.002, 21) } else { Wire::new([0.0, -5.25, 10.0], [0.0, 5.25, 10.0], 0.002, 21) };
        let mut sg = Model::new(vec![w], Ground::Real(RealGround::AVERAGE));
        sg.add_source(0, 11, C::new(1.0, 0.0));
        let mut tr = sg.clone();
        let mut g2 = SecondGround::new(RealGround::AVERAGE, Boundary::Line { distance: 40.0, azimuth: 0.0 });
        g2.drop = 10.0;
        sg.second_ground = Some(g2);
        tr.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [40.0, 0.0], [40.0, -10.0]]));
        let (a, b) = (sg.solve(F).unwrap(), tr.solve(F).unwrap());
        let mut e = from;
        while e < 89.0 {
            let (x, y) = (a.gain_dbi(el(e), 0.0), b.gain_dbi(el(e), 0.0));
            assert!((x - y).abs() < tol, "vertical {vertical} el {e}: second ground {x:.2} vs terrain {y:.2}");
            e += 0.5;
        }
    }
}

#[test]
fn ridge_in_front_cuts_low_angles_and_hilltop_raises_them() {
    let level = dipole(Ground::Real(RealGround::AVERAGE)).solve(F).unwrap();
    // a 400 m ridge 600 m out toward +x (34° above the antenna), level toward -x
    let mut m = dipole(Ground::Real(RealGround::AVERAGE));
    m.terrain = Some(Terrain { profiles: vec![Profile::new(0.0, vec![[0.0, 0.0], [200.0, 0.0], [600.0, 400.0], [2000.0, 400.0]]), Profile::new(PI, vec![[0.0, 0.0]])] });
    let ridge = m.solve(F).unwrap();
    // (measured 13 to 18 dB less from 3° to 20°; nearer grazing the ridge also
    // breaks up the ground reflection's cancellation, "obstacle gain", so the cut
    // is smaller there)
    for (e, cut) in [(3.0, 10.0), (5.0, 15.0), (10.0, 15.0), (20.0, 15.0)] {
        assert!(ridge.gain_dbi(el(e), 0.0) < level.gain_dbi(el(e), 0.0) - cut, "el {e}");
    }
    // the other way is level, so as level ground
    for e in [2.0, 10.0, 30.0] {
        assert!((ridge.gain_dbi(el(e), PI) - level.gain_dbi(el(e), PI)).abs() < 1e-9);
    }
    // on a hilltop (100 m drop over 1 km all round) low angles gain a lot
    // (measured 19.5, 14.4 and 9.7 dB at 1°, 2° and 3°)
    let mut top = dipole(Ground::Real(RealGround::AVERAGE));
    top.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [50.0, 0.0], [1000.0, -100.0]]));
    let top = top.solve(F).unwrap();
    for e in [1.0, 2.0, 3.0] {
        assert!(top.gain_dbi(el(e), 0.0) > level.gain_dbi(el(e), 0.0) + 8.0, "el {e}");
    }
}

#[test]
fn knife_edge_is_smooth() {
    assert!((knife_edge(0.0) - C::new(0.5, 0.0)).norm() < 1e-9);
    assert_eq!(knife_edge(-0.8), C::new(1.0, 0.0));
    assert!((knife_edge(-0.7799).norm() - 1.0).abs() < 1e-3);
    assert!((knife_edge(-1e-9).norm() - 0.5).abs() < 1e-3);
    let mut prev = 1.0;
    for i in 0..300 {
        let g = knife_edge(-0.9 + i as f64 * 0.01).norm();
        assert!(g <= prev + 1e-9, "not falling at {}", -0.9 + i as f64 * 0.01);
        prev = g;
    }
    assert!(knife_edge(3.0).norm() < 0.1);
}

#[test]
fn terrain_and_second_ground_are_refused_together() {
    let mut m = dipole(Ground::Real(RealGround::AVERAGE));
    m.terrain = Some(Terrain::uniform(vec![[0.0, 0.0]]));
    m.second_ground = Some(SecondGround::new(RealGround::SEA_WATER, Boundary::Circle { radius: 100.0 }));
    assert!(m.solve(F).is_err());
}

/// Mean and largest |second difference| of the gain (dB) toward +x over 0.25° steps.
fn roughness(m: &Model, lo: f64, hi: f64) -> (f64, f64) {
    let s = m.solve(F).unwrap();
    let g: Vec<f64> = (0..=((hi - lo) * 4.0) as usize).map(|i| s.gain_dbi(el(lo + i as f64 * 0.25), 0.0)).collect();
    let d2: Vec<f64> = g.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]).abs()).collect();
    (d2.iter().sum::<f64>() / d2.len() as f64, d2.iter().cloned().fold(0.0, f64::max))
}

/// A gently curving profile (0.1° more fall every 50 m) is nearly level: its
/// pattern stays as smooth as a single bend's.  Each facet's edge terms take the
/// exact path phase and fade with the incidence's obliquity, so neighbouring
/// facets' terms cancel at their joins (with the Fresnel integral's quadratic
/// phase alone, this pattern was jagged: 2.8 dB mean second difference).
#[test]
fn gently_curving_profile_is_smooth() {
    let mut m = dipole(Ground::Real(RealGround::AVERAGE));
    let fall = 0.1f64.to_radians().tan();
    m.terrain = Some(Terrain::uniform((0..20).map(|i| [i as f64 * 50.0, -(i * i) as f64 * 50.0 * fall / 2.0]).collect()));
    let (low, _) = roughness(&m, 1.0, 20.0);
    let (high, high_max) = roughness(&m, 20.0, 80.0);
    println!("curving: 1-20° mean {low:.3} | 20-80° mean {high:.3} max {high_max:.3}");
    assert!(low < 0.1 && high < 0.02 && high_max < 0.05);
    // the same total fall as one bend
    let total = 19.0 * 0.1f64.to_radians();
    m.terrain = Some(Terrain::uniform(vec![[0.0, 0.0], [50.0, 0.0], [950.0, -900.0 * total.tan()]]));
    let (low1, _) = roughness(&m, 1.0, 20.0);
    let (high1, _) = roughness(&m, 20.0, 80.0);
    assert!(low1 < 0.1 && high1 < 0.02);
}

