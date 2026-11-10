//! Segment-pair integrals of the free-space Green's function (reduced thin-wire
//! kernel), for the MPIE matrix.
//!
//! `G(R) = exp(-jkR)/(4 pi R)`, `R = sqrt(|r - r'|^2 + a^2)`, split into the
//! singular `1/(4 pi R)`, integrated analytically along the source segment for
//! constant and linear currents, and the smooth `(exp(-jkR) - 1)/(4 pi R)`, by
//! Gauss-Legendre quadrature.  The test (outer) integral is Gauss-Legendre.
//! Between segments of different radius `a^2` is the mean of both radii squared,
//! which keeps the matrix symmetric.

use crate::quad::gauss01;
use crate::vec3::*;
use num_complex::Complex64 as C;
use std::f64::consts::PI;

/// Integrals between a test segment and a source segment:
/// `s = ∫∫ G du du'` and `t[a][b] = ∫∫ shape_a(u) shape_b(u') G du du'`
/// (shape 0 = rising u/L, 1 = falling 1 - u/L).
#[derive(Clone, Copy, Debug, Default)]
pub struct PairIntegrals {
    pub s: C,
    pub t: [[C; 2]; 2],
}

pub struct Seg {
    pub p0: V3,
    pub t: V3,
    pub len: f64,
    pub radius: f64,
}

pub fn pair(test: &Seg, src: &Seg, k: f64, n_outer: usize, n_inner: usize) -> PairIntegrals {
    let (xo, wo) = gauss01(n_outer);
    let (xi, wi) = gauss01(n_inner);
    let a2 = 0.5 * (test.radius * test.radius + src.radius * src.radius);
    let lq = src.len;
    let four_pi = 4.0 * PI;
    let mut out = PairIntegrals::default();
    for (&x, &w) in xo.iter().zip(wo) {
        let r = along(test.p0, test.t, x * test.len);
        // analytic 1/R part along the source line
        let rel = sub(r, src.p0);
        let z0 = dot(rel, src.t);
        let d2 = dot(rel, rel) - z0 * z0 + a2;
        let d = d2.max(1e-300).sqrt();
        let i0 = ((lq - z0) / d).asinh() + (z0 / d).asinh();
        let i1 = ((lq - z0) * (lq - z0) + d2).sqrt() - (z0 * z0 + d2).sqrt() + z0 * i0;
        let mut s0 = C::new(i0 / four_pi, 0.0);
        let mut s1 = C::new(i1 / (lq * four_pi), 0.0);
        // smooth part
        for (&xs, &ws) in xi.iter().zip(wi) {
            let q = along(src.p0, src.t, xs * lq);
            let dd = sub(r, q);
            let rr = (dot(dd, dd) + a2).sqrt();
            let kr = k * rr;
            // exp(-jkR) - 1 without cancellation
            let h = (0.5 * kr).sin();
            let f = C::new(-2.0 * h * h, -kr.sin()) / (four_pi * rr);
            s0 += f * (lq * ws);
            s1 += f * (lq * ws * xs);
        }
        let src_rise = s1;
        let src_fall = s0 - s1;
        let wl = w * test.len;
        let (rise_o, fall_o) = (x, 1.0 - x);
        out.s += s0 * wl;
        out.t[0][0] += src_rise * (wl * rise_o);
        out.t[0][1] += src_fall * (wl * rise_o);
        out.t[1][0] += src_rise * (wl * fall_o);
        out.t[1][1] += src_fall * (wl * fall_o);
    }
    out
}
