//! Real (lossy) ground: the Sommerfeld half-space solution.
//!
//! A unit current element along c at r' radiates plane waves; each reflects from
//! the ground with the Fresnel coefficient of its TE or TM part.  Summing over the
//! spectrum and integrating out the azimuth gives the reflected field
//! `E(r) = D(rho, h) . c` (`h = z + z'`), in local axes (rho, phi, z):
//!
//! ```text
//! D_rr = P ∫ W [ gTE (J0 + J2) - gTM (kz²/k²) (J0 - J2) ]
//! D_ff = P ∫ W [ gTE (J0 - J2) - gTM (kz²/k²) (J0 + J2) ]
//! D_rz = P ∫ W [ 2j gTM (kz kρ/k²) J1 ]     (E_rho from a z source)
//! D_zr = -D_rz
//! D_zz = P ∫ W [ 2 gTM (kρ²/k²) J0 ]
//! ```
//!
//! with `P = -w mu0/(8 pi)`, `W = (kρ/kz) exp(-j kz h)`, Bessel functions of
//! `kρ rho`, `gTE = (kz - kz2)/(kz + kz2)` and `gTM = (eps kz - kz2)/(eps kz + kz2)`
//! (`(-1, +1)` for a perfect conductor, which reproduces the image field exactly).
//!
//! As in NEC-2, the `Gamma0 = (eps-1)/(eps+1)`-scaled image (near-singular, 1/R³)
//! is handled analytically by the solver's image integrals; the remainder
//! (coefficients `gTE + Gamma0`, `gTM - Gamma0`) is tabulated in normalised form
//! `R exp(jkR) D / (w k)` over (kR, theta) and spline-interpolated.
//!
//! Spectral path: two raised arcs in the first quadrant (around k and the
//! surface-wave pole, then out past Re k2), then for rho > h a Hankel split onto
//! vertical paths, otherwise the real axis.

use crate::mesh::Mesh;
use crate::quad::{composite, gauss01};
use crate::spline::Spline2;
use crate::vec3::*;
use crate::{C0, EPS0, Error, MU0};
use complex_bessel::{Scaling, besselj_seq, hankel1_seq, hankel2_seq};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::f64::consts::PI;

const J: C = C::new(0.0, 1.0);

/// Homogeneous lossy ground below z = 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealGround {
    pub eps_r: f64,
    /// conductivity, S/m
    pub sigma: f64,
}

impl RealGround {
    pub const AVERAGE: RealGround = RealGround { eps_r: 13.0, sigma: 0.005 };
    pub const POOR: RealGround = RealGround { eps_r: 5.0, sigma: 0.001 };
    pub const VERY_POOR: RealGround = RealGround { eps_r: 3.0, sigma: 0.001 };
    pub const VERY_GOOD: RealGround = RealGround { eps_r: 20.0, sigma: 0.0303 };
    pub const SEA_WATER: RealGround = RealGround { eps_r: 81.0, sigma: 5.0 };
    pub const FRESH_WATER: RealGround = RealGround { eps_r: 80.0, sigma: 0.001 };

    pub fn eps_c(&self, freq: f64) -> C {
        C::new(self.eps_r, -self.sigma / (2.0 * PI * freq * EPS0))
    }
    pub fn gamma0(&self, freq: f64) -> C {
        let e = self.eps_c(freq);
        (e - 1.0) / (e + 1.0)
    }
    /// Plane-wave coefficients (gTE, gTM) at angle `theta` from zenith; the reflected
    /// far field is `gTM * E_theta` and `-gTE * E_phi` of the PEC image.
    pub fn reflection(&self, freq: f64, theta: f64) -> (C, C) {
        let k = 2.0 * PI * freq / C0;
        coefficients(C::new(k * theta.sin(), 0.0), k, self.eps_c(freq))
    }
}

#[inline]
fn kz(kr: C, k2: C) -> C {
    -J * (kr * kr - k2).sqrt()
}

pub fn coefficients(kr: C, k: f64, eps: C) -> (C, C) {
    let kz1 = kz(kr, C::new(k * k, 0.0));
    let kz2 = kz(kr, eps * k * k);
    ((kz1 - kz2) / (kz1 + kz2), (eps * kz1 - kz2) / (eps * kz1 + kz2))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    J,
    H1,
    H2,
}

#[derive(Clone, Copy)]
pub struct Coeffs {
    /// added to (gTE, gTM): (Gamma0, -Gamma0) gives the remainder
    pub shift: (C, C),
    /// constant (gTE, gTM) instead of the Fresnel ones (testing: (-1, +1) = PEC image)
    pub fixed: Option<(C, C)>,
}

fn bessel01(kind: Kind, x: C) -> (C, C) {
    let r = match kind {
        Kind::J => besselj_seq(0.0, x, 2, Scaling::Unscaled),
        Kind::H1 => hankel1_seq(0.0, x, 2, Scaling::Unscaled),
        Kind::H2 => hankel2_seq(0.0, x, 2, Scaling::Unscaled),
    };
    match r {
        Ok(v) => (v.values[0], v.values[1]),
        // only reached deep in decaying tails, where the contribution is nil
        Err(_) => (C::new(0.0, 0.0), C::new(0.0, 0.0)),
    }
}

fn integrand(kr: C, rho: f64, h: f64, k: f64, eps: C, kind: Kind, co: &Coeffs) -> [C; 5] {
    let k2 = k * k;
    let kz1 = kz(kr, C::new(k2, 0.0));
    let (c_te, c_tm) = match co.fixed {
        Some(f) => f,
        None => {
            let kz2 = kz(kr, eps * k2);
            ((kz1 - kz2) / (kz1 + kz2) + co.shift.0, (eps * kz1 - kz2) / (eps * kz1 + kz2) + co.shift.1)
        }
    };
    let x = kr * rho;
    let (b0, b1) = bessel01(kind, x);
    let b2 = if x.norm() > 1e-12 { 2.0 * b1 / x - b0 } else { C::new(0.0, 0.0) };
    let w = kr / kz1 * (-J * kz1 * h).exp();
    let kzk = kz1 * kz1 / k2;
    let cross = kz1 * kr / k2;
    [
        w * (c_te * (b0 + b2) - c_tm * kzk * (b0 - b2)),
        w * (c_te * (b0 - b2) - c_tm * kzk * (b0 + b2)),
        w * (2.0 * J * c_tm * cross * b1),
        w * (-2.0 * J * c_tm * cross * b1),
        w * (2.0 * c_tm * (kr * kr / k2) * b0),
    ]
}

/// Spectral quadrature for one (rho, h): (kind, krho, weight incl. Jacobian).
/// `k2` is the ground's wavenumber (None for fixed coefficients).
fn nodes(rho: f64, h: f64, k: f64, k2: Option<C>, out: &mut Vec<(Kind, C, C)>) {
    let a1 = 2.0 * k;
    let a = a1.max(1.2 * k2.map_or(0.0, |v| v.re) + 0.5 * k);
    let b = (0.6 * k).min(1.0 / rho.max(1e-30));
    let oscillation = 2.0 * rho / PI;
    let mut buf = Vec::new();
    for (i, (lo, hi)) in [(0.0, a1), (a1, a)].into_iter().enumerate() {
        if hi <= lo {
            continue;
        }
        buf.clear();
        // Out on the second arc the integrand changes on the scale of the distance
        // to k2.  For lossy ground (|Im k2| > k, e.g. sea water) that is large, so
        // the arc needs far fewer points; subintervals then grow geometrically away
        // from the branch point k.
        let im2 = k2.map_or(0.0, |v| v.im.abs());
        if i == 1 && im2 > k {
            // the exp(-krho h) factor kills the integrand beyond ~40/h, so Bessel
            // oscillations only need resolving up to there
            let live = (hi - lo).min(40.0 / h.max(1e-30));
            let n_sub = 16 + (live * oscillation + (hi - lo) * 4.0 / im2) as usize;
            composite(lo, hi, n_sub, 8, Some(0.25 * k), &mut buf);
        } else {
            composite(lo, hi, 8 + ((hi - lo) * oscillation.max(4.0 / k)) as usize, 8, None, &mut buf);
        }
        for &(x, w) in &buf {
            let s = PI * (x - lo) / (hi - lo);
            let kr = C::new(x, b * s.sin());
            let dk = C::new(1.0, b * PI / (hi - lo) * s.cos());
            out.push((Kind::J, kr, dk * w));
        }
    }
    buf.clear();
    if rho > h {
        composite(0.0, 40.0 / rho, 32, 8, Some(0.25 * k), &mut buf);
        for &(t, w) in &buf {
            out.push((Kind::H1, C::new(a, t), J * (0.5 * w)));
            out.push((Kind::H2, C::new(a, -t), -J * (0.5 * w)));
        }
    } else {
        composite(a, a + 40.0 / h, 32, 8, Some(0.25 * k), &mut buf);
        for &(x, w) in &buf {
            out.push((Kind::J, C::new(x, 0.0), C::new(w, 0.0)));
        }
    }
}

/// Reflected-field components (rr, ff, rz, zr, zz) at one (rho, h).
pub fn reflected_dyadic(rho: f64, h: f64, freq: f64, eps: C, co: &Coeffs) -> [C; 5] {
    let omega = 2.0 * PI * freq;
    let k = omega / C0;
    let k2 = if co.fixed.is_some() { None } else { Some(eps.sqrt() * k) };
    let mut pts = Vec::new();
    nodes(rho, h, k, k2, &mut pts);
    let mut total = [C::new(0.0, 0.0); 5];
    for (kind, kr, w) in pts {
        let v = integrand(kr, rho, h, k, eps, kind, co);
        for c in 0..5 {
            total[c] += v[c] * w;
        }
    }
    let pref = -omega * MU0 / (8.0 * PI);
    total.map(|v| v * pref)
}

/// Interpolation table of the remainder field for one frequency.  Stored
/// normalised (`D = w k F(kR, theta)` exactly for a fixed permittivity), so it also
/// serves nearby frequencies of a sweep.
pub struct Table {
    pub ground: RealGround,
    pub freq: f64,
    pub k: f64,
    pub gamma0: C,
    fixed: bool,
    x: Vec<f64>,
    splines: Vec<Spline2>,
}

impl Table {
    pub fn new(ground: RealGround, freq: f64, r_min: f64, r_max: f64, margin: f64, fixed: Option<(C, C)>) -> Table {
        let omega = 2.0 * PI * freq;
        let k = omega / C0;
        let eps = ground.eps_c(freq);
        let gamma0 = ground.gamma0(freq);
        let lam = 2.0 * PI / k;
        let r_min = (r_min / margin).max(1e-6 * lam);
        let r_max = (r_max * margin).max(2.0 * r_min);
        // geometric to lam/4, then ~lam/30 (finer for low-loss water: lateral wave)
        let n2 = eps.sqrt();
        let fine = if n2.im.abs() < n2.re / 4.0 { (n2.re / 4.0).clamp(1.0, 4.0) } else { 1.0 };
        let step = lam / (30.0 * fine);
        let near_end = (lam / 4.0).max(step).min(r_max);
        let mut rs: Vec<f64> = if r_min < near_end {
            (0..24).map(|i| r_min * (near_end / r_min).powf(i as f64 / 23.0)).collect()
        } else {
            vec![r_min]
        };
        let mut r = *rs.last().unwrap() + step;
        while r < r_max + step {
            rs.push(r);
            r += step;
        }
        if rs.len() < 4 {
            rs = (0..4).map(|i| r_min + (r_max - r_min) * i as f64 / 3.0).collect();
        }
        let ts: Vec<f64> = (0..30).map(|i| (PI / 2.0) * (i as f64 / 29.0).powf(1.7)).collect();
        let co = Coeffs { shift: if fixed.is_some() { (C::new(0.0, 0.0), C::new(0.0, 0.0)) } else { (gamma0, -gamma0) }, fixed };
        let nt = ts.len();
        let vals: Vec<[C; 5]> = (0..rs.len() * nt)
            .into_par_iter()
            .map(|idx| {
                let (rr, th) = (rs[idx / nt], ts[idx % nt]);
                let rho = rr * th.cos();
                let h = (rr * th.sin()).max(1e-9 * lam);
                let d = reflected_dyadic(rho, h, freq, eps, &co);
                let norm = C::from_polar(rr, k * rr) / (omega * k);
                d.map(|v| v * norm)
            })
            .collect();
        let x: Vec<f64> = rs.iter().map(|r| k * r).collect();
        let splines = (0..5).map(|c| Spline2::new(x.clone(), ts.clone(), vals.iter().map(|v| v[c]).collect())).collect();
        Table { ground, freq, k, gamma0, fixed: fixed.is_some(), x, splines }
    }

    pub fn covers(&self, ground: &RealGround, freq: f64, r_min: f64, r_max: f64) -> bool {
        let k = 2.0 * PI * freq / C0;
        !self.fixed && *ground == self.ground && k * r_min >= self.x[0] * (1.0 - 1e-9) && k * r_max <= *self.x.last().unwrap()
    }

    /// Remainder components at (rho, h) for frequency `freq`.
    #[inline]
    pub fn eval(&self, rho: f64, h: f64, freq: f64) -> [C; 5] {
        let omega = 2.0 * PI * freq;
        let k = omega / C0;
        let r = (rho * rho + h * h).sqrt();
        let th = h.atan2(rho);
        let back = C::from_polar(omega * k / r, -k * r);
        let x = k * r;
        [0, 1, 2, 3, 4].map(|c| self.splines[c].eval(x, th) * back)
    }
}

/// Quadrature points on every segment.
fn quad_points(m: &Mesh, n_q: usize) -> Vec<Vec<V3>> {
    let (x, _) = gauss01(n_q);
    (0..m.n_segments()).map(|s| x.iter().map(|&xi| along(m.p0[s], m.t[s], xi * m.len[s])).collect()).collect()
}

/// (R_min, R_max) between quadrature points and image points.
pub fn table_range(m: &Mesh, n_q: usize) -> Result<(f64, f64), Error> {
    let pts: Vec<V3> = quad_points(m, n_q).into_iter().flatten().collect();
    let zmin = pts.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
    let zmax = pts.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
    if !(zmin > 0.0) {
        return Err(Error::new("over real ground all wires must be above z = 0"));
    }
    let span = |i: usize| {
        let lo = pts.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let r_max = (span(0).powi(2) + span(1).powi(2) + (2.0 * zmax).powi(2)).sqrt();
    Ok((2.0 * zmin, r_max))
}

/// Galerkin integrals of the tabulated reflected field between segment pairs,
/// `K[p*N + q][a][b] = ∫p ∫q shape_a shape_b t_p . D . t_q`.  Reciprocal, so only
/// q >= p is evaluated.
pub fn segment_integrals(table: &Table, m: &Mesh, n_q: usize, freq: f64) -> Vec<[[C; 2]; 2]> {
    let n = m.n_segments();
    let (x, w) = gauss01(n_q);
    let pts = quad_points(m, n_q);
    let ws: [Vec<f64>; 2] = [x.iter().zip(w).map(|(x, w)| x * w).collect(), x.iter().zip(w).map(|(x, w)| (1.0 - x) * w).collect()];
    let rows: Vec<Vec<[[C; 2]; 2]>> = (0..n)
        .into_par_iter()
        .map(|p| {
            let tp = m.t[p];
            (p..n)
                .map(|q| {
                    let tq = m.t[q];
                    let mut acc = [[C::new(0.0, 0.0); 2]; 2];
                    for (j, r) in pts[p].iter().enumerate() {
                        let mut inner = [C::new(0.0, 0.0); 2];
                        for (kk, s) in pts[q].iter().enumerate() {
                            let dx = r[0] - s[0];
                            let dy = r[1] - s[1];
                            let h = r[2] + s[2];
                            let rho = (dx * dx + dy * dy).sqrt();
                            let (cx, cy) = if rho > 1e-12 * (1.0 + rho) { (dx / rho, dy / rho) } else { (1.0, 0.0) };
                            let d = table.eval(rho, h, freq);
                            let tpr = tp[0] * cx + tp[1] * cy;
                            let tpf = -tp[0] * cy + tp[1] * cx;
                            let tqr = tq[0] * cx + tq[1] * cy;
                            let tqf = -tq[0] * cy + tq[1] * cx;
                            let kern = d[0] * (tpr * tqr) + d[1] * (tpf * tqf) + d[2] * (tpr * tq[2]) + d[3] * (tp[2] * tqr) + d[4] * (tp[2] * tq[2]);
                            inner[0] += kern * ws[0][kk];
                            inner[1] += kern * ws[1][kk];
                        }
                        for a in 0..2 {
                            for b in 0..2 {
                                acc[a][b] += inner[b] * ws[a][j];
                            }
                        }
                    }
                    let l = m.len[p] * m.len[q];
                    acc.map(|r| r.map(|v| v * l))
                })
                .collect()
        })
        .collect();
    let mut k = vec![[[C::new(0.0, 0.0); 2]; 2]; n * n];
    for (p, row) in rows.into_iter().enumerate() {
        for (i, v) in row.into_iter().enumerate() {
            let q = p + i;
            k[p * n + q] = v;
            k[q * n + p] = [[v[0][0], v[1][0]], [v[0][1], v[1][1]]];
        }
    }
    k
}
