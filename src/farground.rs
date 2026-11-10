//! A second ground medium for the far field: sea beyond a shore line, a salt marsh
//! around a site, a drop to a beach.
//!
//! As NEC-2's GD card, the second medium changes only the ground reflection of
//! the radiated field; the currents (impedance) are solved over the primary ground
//! under the antenna.  Medium 2 may lie `drop` metres below medium 1 (a cliff).
//!
//! Each current element's ray reflects from the ground around its specular point.
//! NEC-2 takes the medium at that point, so the pattern jumps where the point
//! crosses the boundary.  Here, by default (`fresnel`), each medium's reflection
//! is weighted by the physical-optics integral over its side of the boundary:
//! expanding the reflected path to second order about the specular point, the
//! share of the half-plane beyond a straight edge is a complex Fresnel integral
//! `F(nu) = ∫_nu^∞ exp(-j pi v²/2) dv / (1 - j)`, with `nu` the edge's distance
//! from the specular point in units of the ray's Fresnel-zone size.  This gives
//! the smooth (slightly rippled) transition seen within the first Fresnel zone of
//! the shore, and the hard switch for rays reflecting far from it.  A circular
//! boundary is treated as its tangent line nearest the specular point, which is
//! approximate for circles smaller than the Fresnel zone (radius 0 means medium 2
//! everywhere, as in NEC-2).  Low-angle rays from low antennas have zones hundreds
//! of metres long, so a shore well beyond the specular point still matters.

use crate::sommerfeld::RealGround;
use num_complex::Complex64 as C;
use std::f64::consts::PI;

/// Where the second medium is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    /// A straight join `distance` metres from the origin, square to the bearing
    /// `azimuth` (radians from +x toward +y); medium 2 lies beyond it.  NEC-2's
    /// linear cliff is azimuth 0.
    Line { distance: f64, azimuth: f64 },
    /// A circle of `radius` metres about the origin; medium 2 lies outside it.
    Circle { radius: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SecondGround {
    pub ground: RealGround,
    pub boundary: Boundary,
    /// how far the surface of medium 2 lies below medium 1 (m, >= 0)
    pub drop: f64,
    /// physical-optics transition across the boundary (default) instead of the
    /// medium at the specular point (NEC-2)
    pub fresnel: bool,
}

/// What a ray needs, computed once per direction.
pub struct Ray {
    s: f64,
    cot: f64,
    e: [f64; 2],
    k: f64,
}

impl Ray {
    /// Ray toward elevation `psi`, azimuth `phi` (radians), wavenumber `k`.
    pub fn new(psi: f64, phi: f64, k: f64) -> Ray {
        let s = psi.sin().max(1e-9);
        Ray { s, cot: psi.cos() / s, e: [phi.cos(), phi.sin()], k }
    }
}

impl SecondGround {
    pub fn new(ground: RealGround, boundary: Boundary) -> SecondGround {
        SecondGround { ground, boundary, drop: 0.0, fresnel: true }
    }

    /// Is the ground point (x, y) in medium 2?
    pub fn in_second(&self, x: f64, y: f64) -> bool {
        match self.boundary {
            Boundary::Line { distance, azimuth } => x * azimuth.cos() + y * azimuth.sin() > distance,
            Boundary::Circle { radius } => x * x + y * y > radius * radius,
        }
    }

    /// Share of medium 2's half-plane in the reflection of a ray from (x, y) at
    /// height `h` above the reflecting plane.
    fn share(&self, x: f64, y: f64, h: f64, ray: &Ray) -> C {
        let h = h.max(0.0);
        let d = h * ray.cot;
        let (sx, sy) = (x + d * ray.e[0], y + d * ray.e[1]);
        if !self.fresnel || h == 0.0 {
            return C::new(if self.in_second(sx, sy) { 1.0 } else { 0.0 }, 0.0);
        }
        // the boundary as a line n . p = dist near the specular point
        let (n, dist) = match self.boundary {
            Boundary::Line { distance, azimuth } => ([azimuth.cos(), azimuth.sin()], distance),
            // radius 0: medium 2 everywhere (as NEC-2)
            Boundary::Circle { radius } if radius <= 0.0 => return C::new(1.0, 0.0),
            Boundary::Circle { radius } => {
                let r = sx.hypot(sy);
                let n = if r > 1e-12 { [sx / r, sy / r] } else { ray.e };
                (n, radius)
            }
        };
        let t = dist - (n[0] * sx + n[1] * sy);
        // path curvature at the specular point: s³/h along the ray, s/h across
        let s = ray.s;
        let (a, b) = (s * s * s / h, s / h);
        let ne = n[0] * ray.e[0] + n[1] * ray.e[1];
        let np = -n[0] * ray.e[1] + n[1] * ray.e[0];
        let c = 1.0 / (ne * ne / a + np * np / b);
        fresnel_tail(t * (ray.k * c / PI).sqrt())
    }

    /// Complex weights (medium 1, medium 2) of the two media's reflections for the
    /// ray from a current element at `p`.  With a drop, the ray to medium 2's
    /// lower surface clears the cliff edge exactly when its specular point on the
    /// upper surface lies beyond the edge, so the upper surface decides the shares
    /// (as in NEC-2); medium 2's reflection then comes from its own, lower image.
    pub fn weights(&self, p: [f64; 3], ray: &Ray) -> (C, C) {
        let w2 = self.share(p[0], p[1], p[2], ray);
        (C::new(1.0, 0.0) - w2, w2)
    }
}

/// `F(nu) = ∫_nu^∞ exp(-j pi v²/2) dv / (1 - j)`: 1 for nu -> -inf, 1/2 at 0,
/// -> 0 (oscillating) for nu -> +inf.  Cubic Hermite interpolation (with the exact
/// slope) of a table on [0, 6], error ~1e-7; asymptotic series beyond.
pub fn fresnel_tail(nu: f64) -> C {
    if nu < 0.0 {
        return C::new(1.0, 0.0) - fresnel_tail(-nu);
    }
    if nu > TAB_END {
        return fresnel_asymptotic(nu);
    }
    let tab = TABLE.get_or_init(|| (0..=TAB_N).map(|i| fresnel_direct(i as f64 * TAB_STEP)).collect());
    let x = nu / TAB_STEP;
    let i = (x as usize).min(TAB_N - 1);
    let t = x - i as f64;
    let slope = |v: f64| -C::from_polar(1.0, -PI * v * v / 2.0) / C::new(1.0, -1.0) * TAB_STEP;
    let (t2, t3) = (t * t, t * t * t);
    tab[i] * (2.0 * t3 - 3.0 * t2 + 1.0)
        + slope(i as f64 * TAB_STEP) * (t3 - 2.0 * t2 + t)
        + tab[i + 1] * (-2.0 * t3 + 3.0 * t2)
        + slope((i + 1) as f64 * TAB_STEP) * (t3 - t2)
}

const TAB_END: f64 = 6.0;
const TAB_STEP: f64 = 0.01;
const TAB_N: usize = 600;
static TABLE: std::sync::OnceLock<Vec<C>> = std::sync::OnceLock::new();

/// exp(-j pi nu²/2) / (j pi nu) (1 + sum (2m-1)!! / (-j pi nu²)^m), for nu > 6.
fn fresnel_asymptotic(nu: f64) -> C {
    let z = C::new(0.0, PI * nu * nu);
    let mut term = C::new(1.0, 0.0);
    let mut sum = term;
    for m in 1..6 {
        term *= -((2 * m - 1) as f64) / z;
        sum += term;
    }
    let e = C::from_polar(1.0, -PI * nu * nu / 2.0);
    e / C::new(0.0, PI * nu) * sum / C::new(1.0, -1.0)
}

/// By quadrature ((1 - j)/2 minus the integral from 0), for 0 <= nu <= 6.
fn fresnel_direct(nu: f64) -> C {
    let norm = C::new(1.0, -1.0);
    let (x, w) = crate::quad::gauss01(64);
    let mut head = C::new(0.0, 0.0);
    for (&xi, &wi) in x.iter().zip(w) {
        let v = nu * xi;
        head += C::from_polar(wi * nu, -PI * v * v / 2.0);
    }
    (norm * 0.5 - head) / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_tail_values() {
        assert!((fresnel_tail(0.0) - C::new(0.5, 0.0)).norm() < 1e-12);
        assert!((fresnel_tail(-30.0) - C::new(1.0, 0.0)).norm() < 1e-2);
        assert!(fresnel_tail(30.0).norm() < 1e-2);
        // continuous where the asymptotic form takes over
        assert!((fresnel_tail(6.0 - 1e-9) - fresnel_tail(6.0 + 1e-9)).norm() < 1e-7);
        // the table between its points
        for i in 0..500 {
            let nu = 0.0123 * i as f64;
            assert!((fresnel_tail(nu) - fresnel_direct(nu)).norm() < 1e-6, "{nu}");
        }
        // C(1) = 0.779893, S(1) = 0.438259
        let want = (C::new(0.5 - 0.779893, -(0.5 - 0.438259))) / C::new(1.0, -1.0);
        assert!((fresnel_tail(1.0) - want).norm() < 1e-6);
    }

    #[test]
    fn transition_across_the_shore() {
        let k = 2.0 * PI / 20.0;
        let sea = SecondGround::new(RealGround::SEA_WATER, Boundary::Line { distance: 0.0, azimuth: 0.0 });
        // specular point on the shore line: half each
        let (w1, w2) = sea.weights([-50.0, 0.0, 5.0], &Ray::new((0.1f64).atan(), 0.0, k));
        assert!((w2 - C::new(0.5, 0.0)).norm() < 1e-9 && (w1 + w2 - 1.0).norm() < 1e-12);
        // rays reflecting far out on the sea / far back on land (many Fresnel
        // zones from the shore: low rays from low antennas have zones hundreds of
        // metres long)
        assert!((sea.weights([100.0, 0.0, 5.0], &Ray::new(0.3, 0.0, k)).1 - 1.0).norm() < 0.15);
        assert!(sea.weights([-100.0, 0.0, 5.0], &Ray::new(0.3, PI, k)).1.norm() < 0.15);
        // the specular mode switches at the specular point
        let mut hard = sea;
        hard.fresnel = false;
        assert_eq!(hard.weights([-30.0, 0.0, 10.0], &Ray::new(0.5, 0.0, k)).1, C::new(0.0, 0.0)); // 18 m out: land
        assert_eq!(hard.weights([-30.0, 0.0, 10.0], &Ray::new(0.1, 0.0, k)).1, C::new(1.0, 0.0)); // 100 m: sea
        // a large circle looks like its tangent line
        let c = SecondGround::new(RealGround::SEA_WATER, Boundary::Circle { radius: 1000.0 });
        let l = SecondGround::new(RealGround::SEA_WATER, Boundary::Line { distance: 1000.0, azimuth: 0.0 });
        for psi in [0.05, 0.2, 0.6] {
            let r = Ray::new(psi, 0.0, k);
            assert!((c.weights([950.0, 0.0, 4.0], &r).1 - l.weights([950.0, 0.0, 4.0], &r).1).norm() < 1e-9);
        }
    }
}
