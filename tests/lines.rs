//! Transmission lines and two-ports: transmission-line theory and nec2c's TL card.

use num_complex::Complex64 as C;
use std::f64::consts::PI;
use wiremom::{C0, Ground, Model, Port, TwoPortKind, Wire};

const F: f64 = 14.2e6;
const ONE: C = C::new(1.0, 0.0);
const J: C = C::new(0.0, 1.0);

fn lam() -> f64 {
    C0 / F
}

/// A 0.48-wavelength dipole (wire 0, feed node 10) and a short feed wire far away
/// (wire 1, gap at node 1) for the source and the line's far end.
fn dipole_and_feed_wire() -> Vec<Wire> {
    let l = lam();
    vec![
        Wire::new([0.0, -0.24 * l, 0.0], [0.0, 0.24 * l, 0.0], 0.001, 20),
        Wire::new([0.0, 0.0, 30.0], [0.0, 0.0, 30.05], 0.001, 2),
    ]
}

fn dipole_alone() -> C {
    let mut m = Model::new(vec![dipole_and_feed_wire()[0].clone()], Ground::Free);
    m.add_source(0, 10, ONE);
    m.solve(F).unwrap().input_impedance()
}

fn fed_through(kind: TwoPortKind) -> C {
    let mut m = Model::new(dipole_and_feed_wire(), Ground::Free);
    m.add_source(1, 1, ONE);
    m.add_network(Port { wire: 1, node: 1 }, Port { wire: 0, node: 10 }, kind);
    m.solve(F).unwrap().input_impedance()
}

#[test]
fn half_wave_line_repeats_the_load() {
    let zd = dipole_alone();
    let z = fed_through(TwoPortKind::line(50.0, Some(0.5 * lam())));
    println!("dipole {zd:.2}, through lambda/2: {z:.2}");
    assert!((z - zd).norm() < 0.01 * zd.norm());
}

#[test]
fn quarter_wave_line_inverts_the_load() {
    let zd = dipole_alone();
    let z = fed_through(TwoPortKind::line(50.0, Some(0.25 * lam())));
    let want = 2500.0 / zd;
    println!("through lambda/4: {z:.2}, theory {want:.2}");
    assert!((z - want).norm() < 0.01 * want.norm());
}

#[test]
fn line_of_any_length_matches_the_formula() {
    let zd = dipole_alone();
    for (z0, frac, vf) in [(75.0, 0.13, 1.0), (300.0, 0.37, 0.8), (50.0, 0.71, 0.66)] {
        let l = frac * lam();
        let line = TwoPortKind::Line { z0, length: Some(l), velocity_factor: vf, loss_db_per_100m: 0.0, crossed: false, shunt: [C::new(0.0, 0.0); 2] };
        let z = fed_through(line);
        let t = (2.0 * PI * l / (vf * lam())).tan();
        let want = z0 * (zd + J * z0 * t) / (z0 + J * zd * t);
        assert!((z - want).norm() < 0.01 * want.norm(), "z0 {z0}: {z} vs {want}");
    }
}

#[test]
fn shorted_stubs_in_parallel_at_the_feed() {
    let l = lam();
    let zd = dipole_alone();
    for frac in [0.25, 0.125] {
        // dipole fed directly; a shorted stub (0 V source = short) across its feed
        let mut m = Model::new(dipole_and_feed_wire(), Ground::Free);
        m.add_source(0, 10, ONE);
        m.add_source(1, 1, C::new(0.0, 0.0));
        m.add_network(Port { wire: 0, node: 10 }, Port { wire: 1, node: 1 }, TwoPortKind::line(50.0, Some(frac * l)));
        let z = m.solve(F).unwrap().source_impedances()[0];
        let zs = J * 50.0 * (2.0 * PI * frac).tan();
        let want = if frac == 0.25 { zd } else { 1.0 / (1.0 / zd + 1.0 / zs) };
        println!("stub {frac} lambda: {z:.2}, theory {want:.2}");
        assert!((z - want).norm() < 0.01 * want.norm());
    }
}

#[test]
fn line_loss_reduces_efficiency() {
    // matched-ish line: loss in dB shows up directly in the efficiency
    let zd = dipole_alone();
    let mk = |db: f64| {
        let line = TwoPortKind::Line { z0: zd.norm(), length: Some(50.0), velocity_factor: 1.0, loss_db_per_100m: db, crossed: false, shunt: [C::new(0.0, 0.0); 2] };
        let mut m = Model::new(dipole_and_feed_wire(), Ground::Free);
        m.add_source(1, 1, ONE);
        m.add_network(Port { wire: 1, node: 1 }, Port { wire: 0, node: 10 }, line);
        m.solve(F).unwrap().efficiency()
    };
    let (e0, e2) = (mk(0.0), mk(2.0)); // 50 m at 2 dB/100 m = 1 dB
    println!("efficiency lossless {e0:.3}, 1 dB line {e2:.3}");
    assert!((e0 - 1.0).abs() < 0.01);
    assert!(((e2 / e0).log10() * -10.0 - 1.0).abs() < 0.15, "{e2}");
}

#[test]
fn crossed_line_flips_the_phase() {
    // two parallel dipoles fed through equal lines from one source: in phase vs
    // crossed (antiphase) gives a broadside vs end-fire pattern
    let l = lam();
    let w = vec![
        Wire::new([0.0, -0.24 * l, 0.0], [0.0, 0.24 * l, 0.0], 0.001, 20),
        Wire::new([0.25 * l, -0.24 * l, 0.0], [0.25 * l, 0.24 * l, 0.0], 0.001, 20),
        Wire::new([0.0, 0.0, 30.0], [0.0, 0.0, 30.05], 0.001, 2),
    ];
    let gains = |crossed: bool| {
        let mut m = Model::new(w.clone(), Ground::Free);
        m.add_source(2, 1, ONE);
        for (wi, cross) in [(0, false), (1, crossed)] {
            let line = TwoPortKind::Line { z0: 300.0, length: Some(0.5 * l), velocity_factor: 1.0, loss_db_per_100m: 0.0, crossed: cross, shunt: [C::new(0.0, 0.0); 2] };
            m.add_network(Port { wire: 2, node: 1 }, Port { wire: wi, node: 10 }, line);
        }
        let s = m.solve(F).unwrap();
        // broadside to the pair is +-y... the dipoles lie along y, so look along x (end-fire) and z (broadside)
        (s.gain_dbi(PI / 2.0, 0.0), s.gain_dbi(0.0, 0.0))
    };
    let (endfire_in, broad_in) = gains(false);
    let (endfire_x, broad_x) = gains(true);
    println!("in phase: end-fire {endfire_in:.1} broadside {broad_in:.1}; crossed: end-fire {endfire_x:.1} broadside {broad_x:.1}");
    assert!(broad_in > endfire_in + 3.0);
    assert!(broad_x < broad_in - 10.0);
}

#[test]
fn agrees_with_nec2c_tl_card() {
    let Ok(out) = std::process::Command::new("nec2c").arg("-h").output() else { return };
    let _ = out;
    let l = lam();
    let dir = std::env::temp_dir().join(format!("wiremom-tl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (z0, len) in [(50.0, 0.3 * l), (75.0, 0.62 * l), (450.0, 0.17 * l)] {
        // NEC: dipole 21 segments (feed = centre segment), 1-segment feed wire
        let deck = format!(
            "CE\nGW 1 21 0 {} 0 0 {} 0 0.001\nGW 2 1 0 0 30 0 0 30.05 0.001\nGE 0\nTL 2 1 1 11 {z0} {len}\nEX 0 2 1 0 1 0\nFR 0 1 0 0 {} 0\nXQ\nEN\n",
            -0.24 * l,
            0.24 * l,
            F / 1e6
        );
        let (i, o) = (dir.join("t.nec"), dir.join("t.out"));
        std::fs::write(&i, deck).unwrap();
        std::process::Command::new("nec2c").arg(format!("-i{}", i.display())).arg(format!("-o{}", o.display())).status().unwrap();
        let text = std::fs::read_to_string(&o).unwrap();
        let line = text.lines().skip_while(|l| !l.contains("ANTENNA INPUT PARAMETERS")).nth(3).unwrap().to_string();
        let v: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        let zn = C::new(v[6], v[7]);
        let z = fed_through(TwoPortKind::line(z0, Some(len)));
        println!("Z0 {z0} length {:.2} lambda: wiremom {z:.2} nec2c {zn:.2}", len / l);
        assert!((z - zn).norm() < 0.04 * zn.norm(), "{z} vs {zn}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
