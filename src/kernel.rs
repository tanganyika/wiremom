//! Segment-pair integrals of the free-space Green's function, for the MPIE matrix.
//!
//! Two kernels (see [`Kernel`]):
//!
//! * **Reduced** (thin-wire): the current on the axis, `G(R) = exp(-jkR)/(4 pi R)`,
//!   `R = sqrt(|r - r'|^2 + a^2)`, split into the singular `1/(4 pi R)`, integrated
//!   analytically along the source segment for constant and linear currents, and
//!   the smooth `(exp(-jkR) - 1)/(4 pi R)`, by Gauss-Legendre quadrature.  Between
//!   segments of different radius `a^2` is the mean of both radii squared, which
//!   keeps the matrix symmetric.
//! * **Exact** (tube): the current spread uniformly round the surface of each tube
//!   and `G` averaged over both circumferences, `R^2 = d^2 + a1^2 + a2^2 - 2 a1 a2
//!   cos(phi)` with `d` the axis-to-axis distance.  The static part is a complete
//!   elliptic integral with a logarithmic singularity (for `a1 = a2`, `d -> 0`),
//!   taken out and integrated analytically; the smooth rest is integrated by Gauss
//!   rules split at the singular point, and the dynamic part by Gauss in `phi`.
//!   It is symmetric in the two radii.  Used for segment pairs closer than a few
//!   radii ([`needs_exact`]); farther pairs use the reduced form with `a^2 = a1^2 +
//!   a2^2`, which matches the tube kernel's `1/R` expansion to `O(a^4/d^4)`.
//!
//! The test (outer) integral is Gauss-Legendre.

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

#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub p0: V3,
    pub t: V3,
    pub len: f64,
    pub radius: f64,
}

/// `k` may be complex (inside lossy ground, k = k0 sqrt(eps)).
/// Reduced kernel with the mean of both radii squared (the thin-wire kernel).
pub fn pair(test: &Seg, src: &Seg, k: C, n_outer: usize, n_inner: usize) -> PairIntegrals {
    pair_reduced(test, src, 0.5 * (test.radius * test.radius + src.radius * src.radius), k, n_outer, n_inner)
}

/// Reduced kernel with an explicit `a^2`.
pub fn pair_reduced(test: &Seg, src: &Seg, a2: f64, k: C, n_outer: usize, n_inner: usize) -> PairIntegrals {
    let (xo, wo) = gauss01(n_outer);
    let (xi, wi) = gauss01(n_inner);
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
            let f = expm1(C::new(0.0, -1.0) * k * rr) / (four_pi * rr);
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

/// Which wire kernel fills the matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kernel {
    /// Thin-wire reduced kernel (current on the axis) for every pair, with the mean
    /// of the two radii squared: the solver's original behaviour.  Good while
    /// segments are several radii long; fails as they approach the diameter.
    Reduced,
    /// Tube kernel (current on the surface) for close pairs, reduced kernel with
    /// `a1^2 + a2^2` beyond.  Accurate for thick wires and short segments.
    #[default]
    Exact,
}

/// Pair integrals with the chosen kernel: for [`Kernel::Exact`], the tube kernel
/// when [`needs_exact`], else the reduced kernel with `a1^2 + a2^2`.
pub fn pair_with(kind: Kernel, test: &Seg, src: &Seg, k: C, n_outer: usize, n_inner: usize) -> PairIntegrals {
    match kind {
        Kernel::Reduced => pair(test, src, k, n_outer, n_inner),
        Kernel::Exact if needs_exact(test, src) => pair_exact(test, src, k, n_outer, n_inner),
        Kernel::Exact => pair_reduced(test, src, test.radius * test.radius + src.radius * src.radius, k, n_outer, n_inner),
    }
}

/// Whether a pair is close enough for the tube kernel to matter: the gap between
/// the segments (centre distance less both half-lengths, a lower bound) is under
/// four times the sum of the radii.
pub fn needs_exact(test: &Seg, src: &Seg) -> bool {
    let c = |s: &Seg| along(s.p0, s.t, 0.5 * s.len);
    let gap = norm(sub(c(test), c(src))) - 0.5 * (test.len + src.len);
    gap < 4.0 * (test.radius + src.radius)
}

/// Complete elliptic integrals `(K(m), E(m))` by the AGM, given `m1 = 1 - m`
/// (accurate as `m -> 1`).
#[inline]
fn ellip_ke(m1: f64) -> (f64, f64) {
    let m1 = m1.clamp(0.0, 1.0);
    let (mut a, mut b) = (1.0f64, m1.sqrt());
    // E = K (1 - sum 2^(n-1) c_n^2), c_0^2 = m
    let mut sum = 0.5 * (1.0 - m1);
    let mut p2 = 0.5;
    for _ in 0..40 {
        let c = 0.5 * (a - b);
        if c.abs() <= 1e-16 * a {
            break;
        }
        p2 *= 2.0;
        sum += p2 * c * c;
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    let k = PI / (a + b);
    (k, k * (1.0 - sum))
}

/// `(∫_0^L ln((u - z0)^2 + d2) du, ∫_0^L u ln((u - z0)^2 + d2) du)`.
fn log_moments(z0: f64, d2: f64, l: f64) -> (f64, f64) {
    let d = d2.max(0.0).sqrt();
    let lg = |x: f64| if x * x + d2 > 0.0 { (x * x + d2).ln() } else { 0.0 };
    // antiderivatives in x = u - z0
    let f0 = |x: f64| {
        let at = if d > 0.0 { 2.0 * d * (x / d).atan() } else { 0.0 };
        (if x != 0.0 { x * lg(x) } else { 0.0 }) - 2.0 * x + at
    };
    let f1 = |x: f64| 0.5 * ((if x * x + d2 > 0.0 { (x * x + d2) * lg(x) } else { 0.0 }) - x * x);
    let (xa, xb) = (-z0, l - z0);
    let j0 = f0(xb) - f0(xa);
    let j1 = f1(xb) - f1(xa) + z0 * j0;
    (j0, j1)
}

/// Gauss points in phi (on [0, pi]) for the dynamic part of the tube kernel.
const N_PHI: usize = 8;

/// `(s0, s1) = (∫ Gt du', ∫ (u'/L) Gt du')` along `src` from the axis point `r` of
/// a tube of radius `a1`, with `Gt` the tube-to-tube kernel (both currents on their
/// surfaces, averaged over both circumferences).
fn tube_line(r: V3, a1: f64, src: &Seg, k: C, n: usize) -> (C, C) {
    let a2 = src.radius;
    let lq = src.len;
    let four_pi = 4.0 * PI;
    let rel = sub(r, src.p0);
    let z0 = dot(rel, src.t);
    let rho2 = (dot(rel, rel) - z0 * z0).max(0.0);
    let sum2 = (a1 + a2) * (a1 + a2);
    let delta2 = rho2 + (a1 - a2) * (a1 - a2);
    let aa = a1 * a1 + a2 * a2;
    let (dif2, bb) = ((a1 - a2) * (a1 - a2), 4.0 * a1 * a2);
    // singular part: -(c/pi) ln(x^2 + delta^2) for the circumferential mean of 1/R
    let c = 1.0 / (rho2 + sum2).sqrt();
    let (j0, j1) = log_moments(z0, delta2, lq);
    let mut s0 = C::new(-c / PI * j0 / four_pi, 0.0);
    let mut s1 = C::new(-c / PI * j1 / (lq * four_pi), 0.0);
    let (xi, wi) = gauss01(n);
    let (xp, wp) = gauss01(N_PHI);
    let mj = C::new(0.0, -1.0) * k;
    // split the source at the foot of the perpendicular, where the log peaks, and
    // a couple of radii either side, where the kernel changes on the radius scale
    let w = 2.0 * (a1 + a2) + rho2.sqrt();
    let mut edges = [0.0, lq, z0, z0 - w, z0 + w];
    edges.sort_by(f64::total_cmp);
    let mut last = 0.0;
    for &e in &edges {
        if !(e > last + 1e-9 * lq && e <= lq) {
            continue;
        }
        let (u0, u1) = (last, e);
        last = e;
        let span = u1 - u0;
        for (&xs, &ws) in xi.iter().zip(wi) {
            let u = u0 + xs * span;
            let x = u - z0;
            let x2r = x * x + rho2;
            let dd = x2r + sum2;
            let m1 = (x * x + delta2) / dd;
            let (ek, ee) = ellip_ke(m1);
            // static remainder: mean 1/R minus its log singularity
            let sd = dd.sqrt();
            let mean_inv_r = 2.0 / PI * ek / sd;
            let stat = mean_inv_r + c / PI * (x * x + delta2).ln();
            // dynamic part: mean of (exp(-jkR) - 1)/R
            let dynp = if x2r > 100.0 * sum2 {
                let rr = (x2r + aa).sqrt();
                expm1(mj * rr) / rr
            } else if k.norm() * sd < 0.5 {
                // series sum (-jk)^n <R^(n-1)>/n!: <R> = (2/pi) sqrt(D) E(m),
                // <R^2> = A, <R^4> = A^2 + b^2/2 (R^2 = A - b cos phi) exactly;
                // odd ones above <R> as <R> times the even below (their terms are
                // O((ka)^4) of the static part)
                let r1 = 2.0 / PI * sd * ee;
                let r2 = x2r + aa;
                let r4 = r2 * r2 + 0.125 * bb * bb;
                let moments = [1.0, r1, r2, r1 * r2, r4, r1 * r4];
                let mut acc = C::new(0.0, 0.0);
                let mut t = C::new(1.0, 0.0);
                for n in 1..=6 {
                    t = t * mj / n as f64;
                    acc += t * moments[n - 1];
                }
                acc
            } else {
                let mut acc = C::new(0.0, 0.0);
                for (&xf, &wf) in xp.iter().zip(wp) {
                    let sh = (0.5 * PI * xf).sin();
                    let rr = (x2r + dif2 + bb * sh * sh).sqrt();
                    acc += if rr > 0.0 { expm1(mj * rr) / rr * wf } else { mj * wf };
                }
                acc
            };
            let f = (dynp + stat) / four_pi * (span * ws);
            s0 += f;
            s1 += f * (u / lq);
        }
    }
    (s0, s1)
}

/// `∫ G du'` along `src` from the charge on a flat end cap (a disk of radius `a` at
/// `at`, coaxial with its wire), the disk's charge in its equilibrium distribution
/// (`σ ∝ 1/sqrt(a^2 - r^2)`, i.e. rings of radius `a sin θ` weighted `sin θ dθ`),
/// with the tube kernel for nearby segments and the reduced kernel (`a^2` -> mean
/// ring radius squared `2a^2/3` + the segment's) beyond.
pub fn cap_line(at: V3, a: f64, src: &Seg, k: C, n: usize) -> C {
    let rel = sub(at, src.p0);
    let z0 = dot(rel, src.t).clamp(0.0, src.len);
    let gap = norm(sub(at, along(src.p0, src.t, z0)));
    if gap > 4.0 * (a + src.radius) + 0.5 * src.len {
        let a2 = 2.0 / 3.0 * a * a + src.radius * src.radius;
        return line_green(at, src, a2.sqrt(), k, n).0;
    }
    let (xt, wt) = gauss01(6);
    xt.iter().zip(wt).map(|(&x, &w)| {
        let th = 0.5 * PI * x;
        tube_line(at, a * th.sin(), src, k, n).0 * (0.5 * PI * w * th.sin())
    }).sum()
}

/// `∫∫ G` between the charges of two end caps (as [`cap_line`]): a cap with
/// itself is the isolated disk, mean potential `Q/(8 eps a)`.
pub fn cap_cap(at1: V3, a1: f64, at2: V3, a2: f64, same: bool, k: C) -> C {
    let mj = C::new(0.0, -1.0) * k;
    if same {
        return C::new(1.0 / (8.0 * a1), 0.0) + mj / (4.0 * PI);
    }
    let d = sub(at1, at2);
    let r = (dot(d, d) + 2.0 / 3.0 * (a1 * a1 + a2 * a2)).sqrt();
    (mj * r).exp() / (4.0 * PI * r)
}

/// Tube-kernel pair integrals (see the module notes); symmetric in the radii.
/// The outer integrand has `x ln x` corners where the test point passes the
/// source's ends (on the radius scale), so the outer rule is split into panels
/// there: end panels of width ~2 radii when the segment is long, halves otherwise.
pub fn pair_exact(test: &Seg, src: &Seg, k: C, n_outer: usize, n_inner: usize) -> PairIntegrals {
    let (xo, wo) = gauss01(n_outer);
    let l = test.len;
    let e = 2.0 * test.radius.max(src.radius);
    let mut edges: Vec<f64> = if l > 6.0 * e { vec![0.0, e, l - e, l] } else { vec![0.0, 0.5 * l, l] };
    // and where the test axis passes the source's ends
    for end in [src.p0, along(src.p0, src.t, src.len)] {
        let x = dot(sub(end, test.p0), test.t);
        if x > 1e-6 * l && x < l * (1.0 - 1e-6) && edges.iter().all(|&y| (y - x).abs() > 0.05 * l) {
            edges.push(x);
        }
    }
    edges.sort_by(f64::total_cmp);
    let mut out = PairIntegrals::default();
    for pn in edges.windows(2) {
        let span = pn[1] - pn[0];
        for (&xg, &wg) in xo.iter().zip(wo) {
            let u = pn[0] + xg * span;
            let x = u / l;
            let r = along(test.p0, test.t, u);
            let (s0, s1) = tube_line(r, test.radius, src, k, n_inner);
            let src_rise = s1;
            let src_fall = s0 - s1;
            let wl = wg * span;
            let (rise_o, fall_o) = (x, 1.0 - x);
            out.s += s0 * wl;
            out.t[0][0] += src_rise * (wl * rise_o);
            out.t[0][1] += src_fall * (wl * rise_o);
            out.t[1][0] += src_rise * (wl * fall_o);
            out.t[1][1] += src_fall * (wl * fall_o);
        }
    }
    out
}

/// exp(z) - 1 without cancellation for small |z| (= 2 sinh(z/2) exp(z/2)).
#[inline]
pub fn expm1(z: C) -> C {
    let h = z * 0.5;
    2.0 * h.sinh() * h.exp()
}

/// `(∫ G du', ∫ (u'/L) G du')` along a source segment, seen from point `r`, with
/// the reduced kernel of radius `a` (analytic 1/R part, Gauss for the rest).
pub fn line_green(r: V3, src: &Seg, a: f64, k: C, n: usize) -> (C, C) {
    let (xi, wi) = gauss01(n);
    let a2 = a * a;
    let lq = src.len;
    let four_pi = 4.0 * PI;
    let rel = sub(r, src.p0);
    let z0 = dot(rel, src.t);
    let d2 = dot(rel, rel) - z0 * z0 + a2;
    let d = d2.max(1e-300).sqrt();
    let i0 = ((lq - z0) / d).asinh() + (z0 / d).asinh();
    let i1 = ((lq - z0) * (lq - z0) + d2).sqrt() - (z0 * z0 + d2).sqrt() + z0 * i0;
    let mut s0 = C::new(i0 / four_pi, 0.0);
    let mut s1 = C::new(i1 / (lq * four_pi), 0.0);
    for (&xs, &ws) in xi.iter().zip(wi) {
        let q = along(src.p0, src.t, xs * lq);
        let dd = sub(r, q);
        let rr = (dot(dd, dd) + a2).sqrt();
        let f = expm1(C::new(0.0, -1.0) * k * rr) / (four_pi * rr);
        s0 += f * (lq * ws);
        s1 += f * (lq * ws * xs);
    }
    (s0, s1)
}

/// Gauss order for a segment pair: the full `n_max` for neighbours, fewer as the
/// pair separates (the singular part is analytic, so only the smooth remainder
/// and the outer integral need points).  Distance between centres relative to the
/// longer segment: < 1.5 -> n_max, < 3 -> 6, < 6 -> 4, beyond -> 3.
pub fn order(test: &Seg, src: &Seg, n_max: usize) -> usize {
    let c = |s: &Seg| along(s.p0, s.t, 0.5 * s.len);
    let d = norm(sub(c(test), c(src)));
    let l = test.len.max(src.len);
    let n = if d < 1.5 * l {
        n_max
    } else if d < 3.0 * l {
        6
    } else if d < 6.0 * l {
        4
    } else {
        3
    };
    n.min(n_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quad::composite;

    /// Brute-force `(∫ Gt du', ∫ (u'/L) Gt du')`: graded composite Gauss in u' (toward
    /// the foot of the perpendicular) and in phi (toward phi = 0), on the full
    /// `exp(-jkR)/R`, no singularity subtraction.
    fn brute(r: V3, a1: f64, src: &Seg, k: C) -> (C, C) {
        let rel = sub(r, src.p0);
        let z0 = dot(rel, src.t).clamp(0.0, src.len);
        let rho2 = (dot(rel, rel) - dot(rel, src.t).powi(2)).max(0.0);
        let a2 = src.radius;
        let mut us = vec![];
        if z0 > 0.0 {
            // graded toward z0: v = z0 - u from 0
            let mut vs = vec![];
            composite(0.0, z0, 40, 12, Some(1e-9 * src.len), &mut vs);
            us.extend(vs.into_iter().map(|(v, w)| (z0 - v, w)));
        }
        if z0 < src.len {
            composite(z0, src.len, 40, 12, Some(1e-9 * src.len), &mut us);
        }
        let mut phis = vec![];
        composite(0.0, PI, 40, 12, Some(1e-9), &mut phis);
        let (mut s0, mut s1) = (C::new(0.0, 0.0), C::new(0.0, 0.0));
        for &(u, wu) in &us {
            let x = u - dot(rel, src.t);
            let mut g = C::new(0.0, 0.0);
            for &(p, wp) in &phis {
                let sh = (0.5 * p).sin();
                let rr = (x * x + rho2 + (a1 - a2) * (a1 - a2) + 4.0 * a1 * a2 * sh * sh).sqrt();
                g += (C::new(0.0, -1.0) * k * rr).exp() / rr * wp;
            }
            let g = g / PI / (4.0 * PI);
            s0 += g * wu;
            s1 += g * wu * (u / src.len);
        }
        (s0, s1)
    }

    #[test]
    fn tube_kernel_matches_brute_force() {
        let k = C::new(2.0 * PI, 0.0);
        let src = Seg { p0: [0.0, 0.0, 0.0], t: [0.0, 0.0, 1.0], len: 0.01, radius: 0.004 };
        // (observation axis point, its radius): on the source (singular), beyond its
        // end, beside it, a different radius, a bend
        let cases: [(V3, f64); 6] = [
            ([0.0, 0.0, 0.0037], 0.004),
            ([0.0, 0.0, 0.0123], 0.004),
            ([0.0, 0.0, -0.002], 0.004),
            ([0.0, 0.0, 0.006], 0.0025),
            ([0.003, 0.0, 0.005], 0.004),
            ([0.02, 0.0, 0.004], 0.004),
        ];
        for (r, a1) in cases {
            let (s0, s1) = tube_line(r, a1, &src, k, 8);
            let (b0, b1) = brute(r, a1, &src, k);
            let e0 = (s0 - b0).norm() / b0.norm();
            let e1 = (s1 - b1).norm() / b1.norm();
            println!("{r:?} a1={a1}: {s0:.6e} vs {b0:.6e} ({e0:.1e}), {e1:.1e}");
            assert!(e0 < 2e-5 && e1 < 2e-5, "{r:?}: {e0:.1e} {e1:.1e}");
        }
    }

    #[test]
    fn tube_kernel_is_reciprocal() {
        let k = C::new(2.0 * PI, 0.0);
        let p = Seg { p0: [0.0, 0.0, 0.0], t: [0.0, 0.0, 1.0], len: 0.01, radius: 0.004 };
        let s = (0.5f64).sqrt();
        let q = Seg { p0: [0.0, 0.0, 0.01], t: [s, 0.0, s], len: 0.006, radius: 0.0025 };
        let a = pair_exact(&p, &q, k, 16, 16);
        let b = pair_exact(&q, &p, k, 16, 16);
        assert!((a.s - b.s).norm() < 1e-4 * a.s.norm(), "{} {}", a.s, b.s);
        for i in 0..2 {
            for j in 0..2 {
                assert!((a.t[i][j] - b.t[j][i]).norm() < 1e-4 * a.s.norm());
            }
        }
    }

    #[test]
    fn far_pairs_agree_between_kernels() {
        // beyond a few radii the tube kernel and the reduced one with a1^2 + a2^2 agree
        let k = C::new(2.0 * PI, 0.0);
        let p = Seg { p0: [0.0, 0.0, 0.0], t: [0.0, 0.0, 1.0], len: 0.01, radius: 0.001 };
        let q = Seg { p0: [0.0, 0.0, 0.0182], ..p };
        let e = pair_exact(&p, &q, k, 8, 8);
        let r = pair_reduced(&p, &q, 2e-6, k, 8, 8);
        assert!((e.s - r.s).norm() < 3e-4 * e.s.norm(), "{} {}", e.s, r.s);
    }
}

#[cfg(test)]
mod convergence {
    use super::*;
    #[test]
    #[ignore]
    fn outer_order() {
        let k = C::new(2.0 * PI, 0.0);
        for &(l, a) in &[(0.01, 0.004), (0.002, 0.004), (0.02, 0.0005)] {
            let p = Seg { p0: [0.0, 0.0, 0.0], t: [0.0, 0.0, 1.0], len: l, radius: a };
            let q = Seg { p0: [0.0, 0.0, l], ..p };
            let refs = (pair_exact(&p, &p, k, 64, 16), pair_exact(&p, &q, k, 64, 16));
            for n in [4, 6, 8, 12, 16] {
                let (s, t) = (pair_exact(&p, &p, k, n, n), pair_exact(&p, &q, k, n, n));
                println!("L/a={} n={n}: self s {:.1e} t00 {:.1e} t01 {:.1e} | adj s {:.1e} t01 {:.1e} t10 {:.1e}", l / a,
                    (s.s - refs.0.s).norm() / refs.0.s.norm(), (s.t[0][0] - refs.0.t[0][0]).norm() / refs.0.t[0][0].norm(), (s.t[0][1] - refs.0.t[0][1]).norm() / refs.0.t[0][1].norm(),
                    (t.s - refs.1.s).norm() / refs.1.s.norm(), (t.t[0][1] - refs.1.t[0][1]).norm() / refs.1.t[0][1].norm(), (t.t[1][0] - refs.1.t[1][0]).norm() / refs.1.t[1][0].norm());
            }
        }
    }
}
