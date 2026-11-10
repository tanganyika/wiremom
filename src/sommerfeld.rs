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

/// Reflected-field integrand.  `k_self` is the wavenumber of the medium the source
/// and observer are in, `eps_rel` = eps_other / eps_self (so k_other² = eps_rel k_self²).
fn integrand(kr: C, rho: f64, h: f64, k_self: C, eps_rel: C, kind: Kind, co: &Coeffs) -> [C; 5] {
    let k2 = k_self * k_self;
    let kz1 = kz(kr, k2);
    let (c_te, c_tm) = match co.fixed {
        Some(f) => f,
        None => {
            let kz2 = kz(kr, eps_rel * k2);
            ((kz1 - kz2) / (kz1 + kz2) + co.shift.0, (eps_rel * kz1 - kz2) / (eps_rel * kz1 + kz2) + co.shift.1)
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

/// Field transmitted through the surface: observer in the ground at depth `d`, unit
/// source in the air at height `zs`.  With `remainder`, the quasi-static part — the
/// free-space field (wavenumber k) with its charge (scalar-potential) part scaled by
/// 2/(1+eps) — is subtracted; that part is integrated analytically by the solver.
fn integrand_trans(kr: C, rho: f64, zs: f64, d: f64, k: f64, eps: C, kind: Kind, remainder: bool) -> [C; 5] {
    let k1s = C::new(k * k, 0.0);
    let kz1 = kz(kr, k1s);
    let kz2 = kz(kr, eps * k1s);
    let n = eps.sqrt();
    let t_te = 2.0 * kz1 / (kz1 + kz2);
    let t_tm = 2.0 * n * kz1 / (eps * kz1 + kz2);
    let x = kr * rho;
    let (b0, b1) = bessel01(kind, x);
    let b2 = if x.norm() > 1e-12 { 2.0 * b1 / x - b0 } else { C::new(0.0, 0.0) };
    let k1k2 = k1s * n;
    let wt = kr / kz1 * (-J * (kz1 * zs + kz2 * d)).exp();
    let tm = t_tm / k1k2;
    let trans = [
        wt * (t_te * (b0 + b2) + tm * kz1 * kz2 * (b0 - b2)),
        wt * (t_te * (b0 - b2) + tm * kz1 * kz2 * (b0 + b2)),
        wt * (tm * kz2 * kr * (-2.0 * J) * b1),
        wt * (tm * kr * kz1 * (-2.0 * J) * b1),
        wt * (tm * kr * kr * 2.0 * b0),
    ];
    if !remainder {
        return trans;
    }
    // free-space field (eps = 1) split into vector-potential (identity) and charge parts
    let wf = kr / kz1 * (-J * kz1 * (zs + d)).exp();
    let a = [wf * 2.0 * b0, wf * 2.0 * b0, C::new(0.0, 0.0), C::new(0.0, 0.0), wf * 2.0 * b0];
    let fr = [
        wf * ((b0 + b2) + kz1 * kz1 / k1s * (b0 - b2)),
        wf * ((b0 - b2) + kz1 * kz1 / k1s * (b0 + b2)),
        wf * (kz1 * kr / k1s * (-2.0 * J) * b1),
        wf * (kz1 * kr / k1s * (-2.0 * J) * b1),
        wf * (kr * kr / k1s * 2.0 * b0),
    ];
    let c = 2.0 / (1.0 + eps);
    [0, 1, 2, 3, 4].map(|i| trans[i] - (a[i] + c * (fr[i] - a[i])))
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
        composite(0.0, 40.0 / rho, 16, 8, Some(0.25 * k), &mut buf);
        for &(t, w) in &buf {
            out.push((Kind::H1, C::new(a, t), J * (0.5 * w)));
            out.push((Kind::H2, C::new(a, -t), -J * (0.5 * w)));
        }
    } else {
        composite(a, a + 40.0 / h, 16, 8, Some(0.25 * k), &mut buf);
        for &(x, w) in &buf {
            out.push((Kind::J, C::new(x, 0.0), C::new(w, 0.0)));
        }
    }
}

/// Which side of the surface a reflected field stays on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// source and observer in the air (z > 0)
    Air,
    /// source and observer in the ground (z < 0)
    Ground,
}

/// Reflected-field components (rr, ff, rz, zr, zz) at one (rho, h), both in the air.
pub fn reflected_dyadic(rho: f64, h: f64, freq: f64, eps: C, co: &Coeffs) -> [C; 5] {
    reflected_dyadic_side(rho, h, freq, eps, co, Side::Air)
}

/// Reflected-field components at one (rho, h = total distance to the surface and
/// back), on either side.  Components refer to the real z axis.
pub fn reflected_dyadic_side(rho: f64, h: f64, freq: f64, eps: C, co: &Coeffs, side: Side) -> [C; 5] {
    let omega = 2.0 * PI * freq;
    let k = omega / C0;
    // the path must clear the branch point of the medium the field travels in, so in
    // the ground k2 is needed even with fixed coefficients
    let k2 = if co.fixed.is_some() && side == Side::Air { None } else { Some(eps.sqrt() * k) };
    let (k_self, eps_rel) = match side {
        Side::Air => (C::new(k, 0.0), eps),
        Side::Ground => (eps.sqrt() * k, 1.0 / eps),
    };
    let mut pts = Vec::new();
    nodes(rho, h, k, k2, &mut pts);
    let mut total = [C::new(0.0, 0.0); 5];
    for (kind, kr, w) in pts {
        let v = integrand(kr, rho, h, k_self, eps_rel, kind, co);
        for c in 0..5 {
            total[c] += v[c] * w;
        }
    }
    let pref = -omega * MU0 / (8.0 * PI);
    let mut out = total.map(|v| v * pref);
    if side == Side::Ground {
        // derived for the mirrored geometry (z -> -z): the rho-z couplings change sign
        out[2] = -out[2];
        out[3] = -out[3];
    }
    out
}

/// Transmitted-field components (rr, ff, rz, zr, zz): observer in the ground at
/// depth `d`, unit source in the air at height `zs`, horizontal offset `rho` (local
/// axes from source to observer).  With `remainder` the quasi-static part is removed
/// (see `integrand_trans`).
pub fn transmitted_dyadic(rho: f64, zs: f64, d: f64, freq: f64, eps: C, remainder: bool) -> [C; 5] {
    let omega = 2.0 * PI * freq;
    let k = omega / C0;
    let mut pts = Vec::new();
    nodes(rho, (zs + d).max(1e-12), k, Some(eps.sqrt() * k), &mut pts);
    let mut total = [C::new(0.0, 0.0); 5];
    for (kind, kr, w) in pts {
        let v = integrand_trans(kr, rho, zs, d, k, eps, kind, remainder);
        for c in 0..5 {
            total[c] += v[c] * w;
        }
    }
    let pref = -omega * MU0 / (8.0 * PI);
    total.map(|v| v * pref)
}

/// Interpolation table of a reflected remainder field for one frequency and side.
/// Stored normalised (`D = w k F(kR, theta)` exactly for a fixed permittivity), so it
/// also serves nearby frequencies of a sweep.
pub struct Table {
    pub ground: RealGround,
    pub freq: f64,
    pub k: f64,
    pub gamma0: C,
    pub side: Side,
    fixed: bool,
    /// wavenumber of the table's medium / k (1, or sqrt(eps) in the ground)
    n_self: C,
    x: Vec<f64>,
    splines: Vec<Spline2>,
}

impl Table {
    pub fn new(ground: RealGround, freq: f64, r_min: f64, r_max: f64, margin: f64, fixed: Option<(C, C)>) -> Table {
        Self::new_side(ground, freq, r_min, r_max, margin, fixed, Side::Air)
    }

    pub fn new_side(ground: RealGround, freq: f64, r_min: f64, r_max: f64, margin: f64, fixed: Option<(C, C)>, side: Side) -> Table {
        Self::new_window(ground, freq, Window { r_min, r_max, theta_min: 0.0 }, margin, fixed, side)
    }

    /// A table covering only `win` (distances and the lowest angle above the
    /// surface that the geometry produces), which is far cheaper for antennas well
    /// above the ground.
    pub fn new_window(ground: RealGround, freq: f64, win: Window, margin: f64, fixed: Option<(C, C)>, side: Side) -> Table {
        let (r_min, r_max) = (win.r_min, win.r_max);
        let omega = 2.0 * PI * freq;
        let k = omega / C0;
        let eps = ground.eps_c(freq);
        let gamma0 = ground.gamma0(freq);
        let lam = 2.0 * PI / k;
        let r_min = (r_min / margin).max(1e-6 * lam);
        let r_max = (r_max * margin).max(2.0 * r_min);
        let n2 = eps.sqrt();
        let n_self = if side == Side::Ground { n2 } else { C::new(1.0, 0.0) };
        // geometric to lam/4, then ~lam/30 (finer for low-loss water: lateral wave;
        // in the ground, on the scale of the ground wavelength or skin depth)
        let fine = if side == Side::Ground {
            (n2.norm() / 2.0).clamp(1.0, 8.0)
        } else if n2.im.abs() < n2.re / 4.0 {
            (n2.re / 4.0).clamp(1.0, 4.0)
        } else {
            1.0
        };
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
        // angles from the window's lowest (with margin) to straight up, clustered
        // toward the low end where the field changes fastest
        let t0 = (win.theta_min / margin.max(1.0) - 0.02).max(0.0);
        let nt = ((30.0 * (PI / 2.0 - t0) / (PI / 2.0)).ceil() as usize).max(8);
        let ts: Vec<f64> = (0..nt).map(|i| t0 + (PI / 2.0 - t0) * (i as f64 / (nt - 1) as f64).powf(1.7)).collect();
        // remainder after the quasi-static image; in the air gTM -> +Gamma0, in the
        // ground gTM -> -Gamma0.  `fixed` tabulates constant coefficients (testing).
        let g = if side == Side::Ground { -gamma0 } else { gamma0 };
        let co = Coeffs { shift: if fixed.is_some() { (C::new(0.0, 0.0), C::new(0.0, 0.0)) } else { (g, -g) }, fixed };
        let nt = ts.len();
        let vals: Vec<[C; 5]> = (0..rs.len() * nt)
            .into_par_iter()
            .map(|idx| {
                let (rr, th) = (rs[idx / nt], ts[idx % nt]);
                let rho = rr * th.cos();
                let h = (rr * th.sin()).max(1e-9 * lam);
                let d = reflected_dyadic_side(rho, h, freq, eps, &co, side);
                let norm = (J * n_self * k * rr).exp() * rr / (omega * k);
                d.map(|v| v * norm)
            })
            .collect();
        let x: Vec<f64> = rs.iter().map(|r| k * r).collect();
        let splines = (0..5).map(|c| Spline2::new(x.clone(), ts.clone(), vals.iter().map(|v| v[c]).collect())).collect();
        Table { ground, freq, k, gamma0, side, fixed: fixed.is_some(), n_self, x, splines }
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
        let back = (-J * self.n_self * k * r).exp() * (omega * k / r);
        let x = k * r;
        [0, 1, 2, 3, 4].map(|c| self.splines[c].eval(x, th) * back)
    }
}

/// Anything that gives the remainder field at (rho, h, freq): one table, or a blend
/// of the tables at two neighbouring frequencies.
pub trait Eval: Send + Sync {
    fn eval(&self, rho: f64, h: f64, freq: f64) -> [C; 5];
}

impl Eval for Table {
    fn eval(&self, rho: f64, h: f64, freq: f64) -> [C; 5] {
        Table::eval(self, rho, h, freq)
    }
}

/// Linear blend (in log frequency) of the tables at the two frequency buckets around
/// the solve frequency.  Each table freezes the ground's permittivity at its own
/// frequency; blending makes that error second-order in the bucket width.
pub struct Blend {
    pub a: std::sync::Arc<Table>,
    pub b: std::sync::Arc<Table>,
    pub w: f64,
}

impl Eval for Blend {
    fn eval(&self, rho: f64, h: f64, freq: f64) -> [C; 5] {
        let (x, y) = (self.a.eval(rho, h, freq), self.b.eval(rho, h, freq));
        [0, 1, 2, 3, 4].map(|i| x[i] * (1.0 - self.w) + y[i] * self.w)
    }
}

/// Quadrature points on the given segments.
fn quad_points(m: &Mesh, n_q: usize, segs: &[usize]) -> Vec<Vec<V3>> {
    let (x, _) = gauss01(n_q);
    segs.iter().map(|&s| x.iter().map(|&xi| along(m.p0[s], m.t[s], xi * m.len[s])).collect()).collect()
}

/// (R_min, R_max) between quadrature points of `segs` and their mirror images, all
/// on one side of the surface.
pub fn table_range(m: &Mesh, n_q: usize) -> Result<(f64, f64), Error> {
    let all: Vec<usize> = (0..m.n_segments()).collect();
    table_range_of(m, n_q, &all)
}

pub fn table_range_of(m: &Mesh, n_q: usize, segs: &[usize]) -> Result<(f64, f64), Error> {
    let pts: Vec<V3> = quad_points(m, n_q, segs).into_iter().flatten().collect();
    let dist = |p: &V3| p[2].abs();
    let zmin = pts.iter().map(dist).fold(f64::INFINITY, f64::min);
    let zmax = pts.iter().map(dist).fold(f64::NEG_INFINITY, f64::max);
    if !(zmin > 0.0) {
        return Err(Error::new("a wire lies in the ground surface; move it slightly above or below z = 0"));
    }
    let span = |i: usize| {
        let lo = pts.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let r_max = (span(0).powi(2) + span(1).powi(2) + (2.0 * zmax).powi(2)).sqrt();
    Ok((2.0 * zmin, r_max))
}

/// Galerkin integral of a dyadic kernel between two segments:
/// `K[a][b] = ∫p ∫q shape_a(u) shape_b(u') t_p . D(r, r') . t_q`, with D given in
/// local axes (rho from source to observer, phi, z) by `field(r, r', rho, cx, cy)`.
fn galerkin(
    m: &Mesh,
    p: usize,
    q: usize,
    pts_p: &[V3],
    pts_q: &[V3],
    n_q: usize,
    mut field: impl FnMut(V3, V3, f64) -> [C; 5],
) -> [[C; 2]; 2] {
    let (x, w) = gauss01(n_q);
    let (tp, tq) = (m.t[p], m.t[q]);
    let mut acc = [[C::new(0.0, 0.0); 2]; 2];
    for (j, r) in pts_p.iter().enumerate() {
        let mut inner = [C::new(0.0, 0.0); 2];
        for (kk, s) in pts_q.iter().enumerate() {
            let dx = r[0] - s[0];
            let dy = r[1] - s[1];
            let rho = (dx * dx + dy * dy).sqrt();
            let (cx, cy) = if rho > 1e-12 * (1.0 + rho) { (dx / rho, dy / rho) } else { (1.0, 0.0) };
            let d = field(*r, *s, rho);
            let tpr = tp[0] * cx + tp[1] * cy;
            let tpf = -tp[0] * cy + tp[1] * cx;
            let tqr = tq[0] * cx + tq[1] * cy;
            let tqf = -tq[0] * cy + tq[1] * cx;
            let kern = d[0] * (tpr * tqr) + d[1] * (tpf * tqf) + d[2] * (tpr * tq[2]) + d[3] * (tp[2] * tqr) + d[4] * (tp[2] * tq[2]);
            inner[0] += kern * (x[kk] * w[kk]);
            inner[1] += kern * ((1.0 - x[kk]) * w[kk]);
        }
        for a in 0..2 {
            let wa = if a == 0 { x[j] * w[j] } else { (1.0 - x[j]) * w[j] };
            for b in 0..2 {
                acc[a][b] += inner[b] * wa;
            }
        }
    }
    let l = m.len[p] * m.len[q];
    acc.map(|r| r.map(|v| v * l))
}

/// Galerkin integrals of a tabulated reflected field between all pairs of `segs`
/// (all on the table's side): `K[i*n + j]` for segs[i] (test), segs[j] (source).
/// Reciprocal, so only j >= i is evaluated.
pub fn segment_integrals(table: &dyn Eval, m: &Mesh, n_q: usize, freq: f64) -> Vec<[[C; 2]; 2]> {
    let all: Vec<usize> = (0..m.n_segments()).collect();
    segment_integrals_of(table, m, n_q, freq, &all)
}

pub fn segment_integrals_of(table: &dyn Eval, m: &Mesh, n_q: usize, freq: f64, segs: &[usize]) -> Vec<[[C; 2]; 2]> {
    let n = segs.len();
    let pts = quad_points(m, n_q, segs);
    let rows: Vec<Vec<[[C; 2]; 2]>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (i..n)
                .map(|j| galerkin(m, segs[i], segs[j], &pts[i], &pts[j], n_q, |r, s, rho| table.eval(rho, r[2].abs() + s[2].abs(), freq)))
                .collect()
        })
        .collect();
    let mut k = vec![[[C::new(0.0, 0.0); 2]; 2]; n * n];
    for (i, row) in rows.into_iter().enumerate() {
        for (o, v) in row.into_iter().enumerate() {
            let j = i + o;
            k[i * n + j] = v;
            k[j * n + i] = [[v[0][0], v[1][0]], [v[0][1], v[1][1]]];
        }
    }
    k
}

/// Range (R_min, R_max) of a remainder table for these points (all on one side).
/// The part of (distance, angle) space a remainder table must cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub r_min: f64,
    pub r_max: f64,
    /// lowest angle above the surface (radians) between a point and an image point
    pub theta_min: f64,
}

pub fn window_of_points(pts: &[V3]) -> Result<Window, Error> {
    let (r_min, r_max) = range_of_points(pts)?;
    let dmin = pts.iter().map(|p| p[2].abs()).fold(f64::INFINITY, f64::min);
    let span = |i: usize| {
        let lo = pts.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let rho_max = (span(0).powi(2) + span(1).powi(2)).sqrt();
    Ok(Window { r_min, r_max, theta_min: (2.0 * dmin).atan2(rho_max) })
}

pub fn range_of_points(pts: &[V3]) -> Result<(f64, f64), Error> {
    let dist = |p: &V3| p[2].abs();
    let zmin = pts.iter().map(dist).fold(f64::INFINITY, f64::min);
    let zmax = pts.iter().map(dist).fold(f64::NEG_INFINITY, f64::max);
    if !(zmin > 0.0) {
        return Err(Error::new("a wire or surface lies in the ground surface; move it slightly above or below z = 0"));
    }
    let span = |i: usize| {
        let lo = pts.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min);
        let hi = pts.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    Ok((2.0 * zmin, (span(0).powi(2) + span(1).powi(2) + (2.0 * zmax).powi(2)).sqrt()))
}

/// Ground-side quadrature points of buried wires (`n_q` per segment).
pub fn ground_points(m: &Mesh, n_q: usize) -> Vec<V3> {
    let gnd: Vec<usize> = (0..m.n_segments()).filter(|&s| m.medium[s] == crate::mesh::GROUND).collect();
    quad_points(m, n_q, &gnd).into_iter().flatten().collect()
}

/// Air-side quadrature points of wires (`n_q` per segment) and surfaces.
pub fn air_points(m: &Mesh, n_q: usize) -> Vec<V3> {
    let air: Vec<usize> = (0..m.n_segments()).filter(|&s| m.medium[s] == crate::mesh::AIR).collect();
    let mut pts: Vec<V3> = quad_points(m, n_q, &air).into_iter().flatten().collect();
    for t in 0..m.surf.tris.len() {
        pts.extend(crate::surface::tri_points(&m.surf.tri_verts(t)));
    }
    pts
}

/// a·D·b for a dyadic in local axes (rho from source to observer, phi, z).
#[inline]
fn contract(d: &[C; 5], a: V3, b: V3, cx: f64, cy: f64) -> C {
    let (ar, af) = (a[0] * cx + a[1] * cy, -a[0] * cy + a[1] * cx);
    let (br, bf) = (b[0] * cx + b[1] * cy, -b[0] * cy + b[1] * cx);
    d[0] * (ar * br) + d[1] * (af * bf) + d[2] * (ar * b[2]) + d[3] * (a[2] * br) + d[4] * (a[2] * b[2])
}

#[inline]
fn axes(r: V3, s: V3) -> (f64, f64, f64) {
    let (dx, dy) = (r[0] - s[0], r[1] - s[1]);
    let rho = (dx * dx + dy * dy).sqrt();
    if rho > 1e-12 * (1.0 + rho) { (rho, dx / rho, dy / rho) } else { (rho, 1.0, 0.0) }
}

/// Remainder (tabulated reflected field) between test triangle p and source
/// triangle q: `K[i][j] = ∫p ∫q (r - v_i)·D·(r' - v_j)`.
pub fn tri_tri_remainder(table: &dyn Eval, freq: f64, vp: &[V3; 3], ap: f64, vq: &[V3; 3], aq: f64) -> [[C; 3]; 3] {
    let (pp, pq) = (crate::surface::tri_points(vp), crate::surface::tri_points(vq));
    let mut k = [[C::new(0.0, 0.0); 3]; 3];
    for (r, (_, wr)) in pp.iter().zip(crate::surface::TRI7) {
        for (s, (_, ws)) in pq.iter().zip(crate::surface::TRI7) {
            let (rho, cx, cy) = axes(*r, *s);
            let d = table.eval(rho, r[2] + s[2], freq);
            let w = wr * ap * ws * aq;
            for i in 0..3 {
                for j in 0..3 {
                    k[i][j] += contract(&d, sub(*r, vp[i]), sub(*s, vq[j]), cx, cy) * w;
                }
            }
        }
    }
    k
}

/// Remainder between a test wire segment and source triangle q:
/// `K[s][j] = ∫seg shape_s t·D·(r' - v_j)` (shape 0 rising, 1 falling).
pub fn seg_tri_remainder(table: &dyn Eval, freq: f64, p0: V3, t: V3, len: f64, n_q: usize, vq: &[V3; 3], aq: f64) -> [[C; 3]; 2] {
    let (x, w) = gauss01(n_q);
    let pq = crate::surface::tri_points(vq);
    let mut k = [[C::new(0.0, 0.0); 3]; 2];
    for (&xi, &wi) in x.iter().zip(w) {
        let r = along(p0, t, xi * len);
        for (s, (_, ws)) in pq.iter().zip(crate::surface::TRI7) {
            let (rho, cx, cy) = axes(r, *s);
            let d = table.eval(rho, r[2] + s[2], freq);
            let wt = wi * len * ws * aq;
            for j in 0..3 {
                let v = contract(&d, t, sub(*s, vq[j]), cx, cy) * wt;
                k[0][j] += v * xi;
                k[1][j] += v * (1.0 - xi);
            }
        }
    }
    k
}

/// Galerkin integrals of the transmitted remainder: `K[i*na + j]` for test segment
/// below[i] (in the ground) and source segment above[j] (in the air).  Evaluated
/// directly (no table): 6 Gauss points per segment for nearby pairs, 2 otherwise.
pub fn cross_integrals(m: &Mesh, freq: f64, eps: C, below: &[usize], above: &[usize]) -> Vec<[[C; 2]; 2]> {
    let mid = |s: usize| along(m.p0[s], m.t[s], 0.5 * m.len[s]);
    let pts6b = quad_points(m, 6, below);
    let pts6a = quad_points(m, 6, above);
    let pts2b = quad_points(m, 2, below);
    let pts2a = quad_points(m, 2, above);
    below
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, &p)| {
            let (pts6b, pts6a, pts2b, pts2a) = (&pts6b, &pts6a, &pts2b, &pts2a);
            above.iter().enumerate().map(move |(j, &q)| {
                let near = norm(sub(mid(p), mid(q))) < 3.0 * (m.len[p] + m.len[q]);
                let (pp, pq, nq) = if near { (&pts6b[i], &pts6a[j], 6) } else { (&pts2b[i], &pts2a[j], 2) };
                galerkin(m, p, q, pp, pq, nq, |r, s, rho| transmitted_dyadic(rho, s[2], -r[2], freq, eps, true))
            })
        })
        .collect()
}

/// Far-field transmission of a buried current element (by reciprocity: the field a
/// plane wave arriving from (theta, phi) sets up at the element).  Returns
/// (tTE, tTM, ê_TMt) where E_theta ∝ tTM (ê_TMt . c) and E_phi ∝ tTE (phî . c), and
/// the element's phase factor gets exp(-j kz2 |z'|) (`kz2` also returned).
pub fn transmission(ground: &RealGround, freq: f64, theta: f64, phi: f64) -> (C, C, [C; 3], C) {
    let k = 2.0 * PI * freq / C0;
    let eps = ground.eps_c(freq);
    let kr = C::new(k * theta.sin(), 0.0);
    let kz1 = kz(kr, C::new(k * k, 0.0));
    let kz2 = kz(kr, eps * k * k);
    let n = eps.sqrt();
    let t_te = 2.0 * kz1 / (kz1 + kz2);
    let t_tm = 2.0 * n * kz1 / (eps * kz1 + kz2);
    // ê_TMt = (kz2 rho_phi - krho z) / k2 with rho_phi the horizontal unit toward phi
    let k2 = n * k;
    let e = [kz2 * phi.cos() / k2, kz2 * phi.sin() / k2, -kr / k2];
    (t_te, t_tm, e, kz2)
}
