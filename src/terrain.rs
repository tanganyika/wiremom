//! Terrain for the far field: the ground's height along each azimuth (a profile),
//! as HFTA and similar terrain-analysis programs use.
//!
//! Like the second ground medium ([`crate::SecondGround`]), terrain changes only
//! the far field: the currents (impedance) are solved over the model's own ground
//! under the antenna (level or sloping).  Low rays reflect hundreds of metres to
//! kilometres away, so it is the terrain there that sets a hill site's low-angle
//! pattern, not the ground under the mast.
//!
//! Toward azimuth phi the terrain is taken as two-dimensional: the profile's
//! heights along that bearing, the same across it.  The profile is a polyline of
//! straight facets; the first runs on backward behind the antenna, and the last
//! point continues level for ever.  For each current element and ray:
//!
//! * every facet reflects with its own image (mirrored in the facet's plane) and
//!   Fresnel coefficients at the ray's grazing angle to the facet.  As in
//!   [`crate::farground`], the facet's share of the reflection is the
//!   physical-optics integral over its lit part: `F(nu_a) - F(nu_b)`, with
//!   `nu` the lit part's ends measured from the specular point in Fresnel-zone
//!   units.  The lit part is the facet less what ridges between the element and
//!   the facet hide from it.  Ridges beyond the facet cut the reflected ray as a
//!   knife edge, seen from the image (the two-ray model of diffraction over
//!   ground).  A level profile gives exactly the level-ground reflection;
//! * the direct ray is cut by the most obstructing ridge in front of the element,
//!   as a knife edge with `nu = H sqrt(2 / (lambda D))` (H the ridge's height
//!   above the ray, D the distance to it): see [`knife_edge`].
//!
//! Every facet has the model's ground constants.  Faces steeper than vertical
//! (zero length along the bearing, a cliff face) don't reflect.

use crate::farground::fresnel_tail;
use num_complex::Complex64 as C;
use std::f64::consts::PI;

/// The knife-edge factor of a ray passing an edge at Fresnel parameter `nu`
/// (positive: the edge stands above the ray).  Where the edge blocks the ray
/// (`nu >= 0`) it is the exact `F(nu)`.  Where the ray clears it, the exact
/// factor ripples about 1 (the edge's diffracted wave beating with the ray); over
/// a ground ridge that wave is already in the facets' reflections, cut off at the
/// ridge, so here it falls smoothly to 1 instead, with ITU-R P.526's loss J(nu)
/// (6 dB at nu = 0, none below nu = -0.78).
pub fn knife_edge(nu: f64) -> C {
    if !nu.is_finite() || nu <= -0.78 {
        return if nu == f64::INFINITY { C::new(0.0, 0.0) } else { C::new(1.0, 0.0) };
    }
    if nu >= 0.0 {
        return fresnel_tail(nu);
    }
    let j_db = 6.9 + 20.0 * (((nu - 0.1) * (nu - 0.1) + 1.0).sqrt() + nu - 0.1).log10();
    C::new(10f64.powf(-j_db.max(0.0) / 20.0), 0.0)
}

/// The ground's height along one bearing.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// bearing (radians from +x toward +y)
    pub azimuth: f64,
    /// (distance from the origin along the bearing, height) in metres, distance
    /// increasing from 0.  Heights are relative to the ground at the origin, which
    /// is (0, 0) when the profile doesn't start there.  Beyond the last point the
    /// ground stays at its height.
    pub points: Vec<[f64; 2]>,
}

impl Profile {
    pub fn new(azimuth: f64, points: Vec<[f64; 2]>) -> Profile {
        Profile { azimuth, points }
    }

    /// Height at distance `d` (linear between points, level beyond the last; before
    /// the first, along the first facet).
    pub fn height(&self, d: f64) -> f64 {
        let p = self.cleaned();
        let i = p.partition_point(|q| q[0] <= d);
        if i == 0 || p.len() == 1 {
            return if p.len() > 1 && i == 0 { lerp(p[0], p[1], d) } else { p[0][1] };
        }
        if i == p.len() {
            return p[p.len() - 1][1];
        }
        lerp(p[i - 1], p[i], d)
    }

    /// The points sorted, starting at distance 0, without repeats.
    fn cleaned(&self) -> Vec<[f64; 2]> {
        let mut p: Vec<[f64; 2]> = self.points.iter().copied().filter(|q| q[0] >= 0.0 && q[0].is_finite() && q[1].is_finite()).collect();
        p.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        if p.first().is_none_or(|q| q[0] > 0.0) {
            p.insert(0, [0.0, 0.0]);
        }
        p
    }
}

fn lerp(a: [f64; 2], b: [f64; 2], d: f64) -> f64 {
    if b[0] - a[0] <= 0.0 { b[1] } else { a[1] + (b[1] - a[1]) * (d - a[0]) / (b[0] - a[0]) }
}

/// Terrain around the antenna: one or more profiles.  Between two profiles'
/// bearings the heights are blended linearly; one profile applies all round.
#[derive(Clone, Debug, PartialEq)]
pub struct Terrain {
    pub profiles: Vec<Profile>,
}

impl Terrain {
    /// The same profile toward every bearing.
    pub fn uniform(points: Vec<[f64; 2]>) -> Terrain {
        Terrain { profiles: vec![Profile::new(0.0, points)] }
    }

    /// The profile toward `phi`.
    pub fn profile_at(&self, phi: f64) -> Vec<[f64; 2]> {
        let n = self.profiles.len();
        if n == 0 {
            return vec![[0.0, 0.0]];
        }
        if n == 1 {
            return self.profiles[0].cleaned();
        }
        let tau = 2.0 * PI;
        let az = |p: &Profile| p.azimuth.rem_euclid(tau);
        let mut order: Vec<&Profile> = self.profiles.iter().collect();
        order.sort_by(|a, b| az(a).partial_cmp(&az(b)).unwrap());
        let phi = phi.rem_euclid(tau);
        // the profiles either side (circularly)
        let j = order.partition_point(|p| az(p) <= phi);
        let (a, b) = (order[(j + n - 1) % n], order[j % n]);
        let span = (az(b) - az(a)).rem_euclid(tau);
        let t = if span > 0.0 { (phi - az(a)).rem_euclid(tau) / span } else { 0.0 };
        if t <= 1e-12 {
            return a.cleaned();
        }
        let mut ds: Vec<f64> = a.cleaned().iter().chain(b.cleaned().iter()).map(|q| q[0]).collect();
        ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
        ds.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        ds.iter().map(|&d| [d, (1.0 - t) * a.height(d) + t * b.height(d)]).collect()
    }

    /// The terrain's cut toward `phi`, ready for far-field sums.
    pub fn cut(&self, phi: f64) -> Cut {
        Cut::new(&self.profile_at(phi))
    }
}

/// A straight piece of the profile, from `u` = 0 at `q` along `t`.
#[derive(Clone, Copy, Debug)]
pub struct Facet {
    /// a point on it (distance, height)
    pub q: [f64; 2],
    /// unit vector along it, outward (cos a, sin a)
    pub t: [f64; 2],
    /// its rise angle (radians, positive rising away from the antenna)
    pub alpha: f64,
    /// its extent in `u` (may be infinite)
    pub u0: f64,
    pub u1: f64,
}

impl Facet {
    /// Unit normal, upward.
    pub fn normal(&self) -> [f64; 2] {
        [-self.t[1], self.t[0]]
    }
}

/// The profile along one bearing: facets and the ridges (convex corners) that can
/// hide things.
#[derive(Clone, Debug)]
pub struct Cut {
    pub facets: Vec<Facet>,
    /// convex corners (distance, height)
    pub ridges: Vec<[f64; 2]>,
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

impl Cut {
    pub fn new(points: &[[f64; 2]]) -> Cut {
        // straight pieces; zero-length ones (vertical faces) are dropped, and the
        // last point continues level
        let mut pts: Vec<[f64; 2]> = points.to_vec();
        let last = *pts.last().unwrap();
        pts.push([f64::INFINITY, last[1]]);
        let mut pieces: Vec<([f64; 2], [f64; 2], f64)> = vec![]; // start, unit dir, length
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b[0] - a[0] <= 1e-9 {
                continue;
            }
            if b[0].is_infinite() {
                pieces.push((a, [1.0, 0.0], f64::INFINITY));
            } else {
                let d = sub2(b, a);
                let l = d[0].hypot(d[1]);
                pieces.push((a, [d[0] / l, d[1] / l], l));
            }
        }
        let mut facets: Vec<Facet> = pieces
            .iter()
            .enumerate()
            .map(|(i, &(a, t, l))| Facet { q: a, t, alpha: t[1].atan2(t[0]), u0: if i == 0 { f64::NEG_INFINITY } else { 0.0 }, u1: l })
            .collect();
        // join pieces that continue straight on (a point in the middle of a slope)
        let mut merged: Vec<Facet> = vec![];
        for f in facets.drain(..) {
            if let Some(m) = merged.last_mut() {
                let end = [m.q[0] + m.u1 * m.t[0], m.q[1] + m.u1 * m.t[1]];
                if (f.alpha - m.alpha).abs() < 1e-12 && dot2(sub2(f.q, end), sub2(f.q, end)) < 1e-18 {
                    m.u1 += f.u1;
                    continue;
                }
            }
            merged.push(f);
        }
        // ridges: where the ground turns downward (a vertical face between two
        // facets counts by the facets either side)
        let mut ridges = vec![];
        for w in merged.windows(2) {
            let (a, b) = (w[0], w[1]);
            let end = [a.q[0] + a.u1 * a.t[0], a.q[1] + a.u1 * a.t[1]];
            if b.q[1] > end[1] + 1e-9 {
                // a face rising to the next facet: its top
                ridges.push(b.q);
            } else if b.alpha < a.alpha - 1e-12 || b.q[1] < end[1] - 1e-9 {
                // turning down, or a face falling away: the end of the first facet
                ridges.push(end);
            }
        }
        Cut { facets: merged, ridges }
    }

    /// The knife-edge factor of the direct ray from (s, z) at elevation `psi`
    /// (1 when nothing is in front).
    pub fn direct(&self, s: f64, z: f64, psi: f64, k: f64) -> C {
        self.over_ridges(s, z, psi, k, |_| true)
    }

    /// Knife-edge factor of the ray from (s, z) at elevation `psi` over the most
    /// obstructing ridge in front of it that `keep` accepts.
    fn over_ridges(&self, s: f64, z: f64, psi: f64, k: f64, keep: impl Fn(&[f64; 2]) -> bool) -> C {
        let (sp, cp) = psi.sin_cos();
        let mut nu_max = f64::NEG_INFINITY;
        for r in &self.ridges {
            let dx = r[0] - s;
            if dx <= 1e-9 || !keep(r) {
                continue;
            }
            // height above the ray, square to it, and the distance along it
            let h = (r[1] - z) * cp - dx * sp;
            let dist = dx * cp + (r[1] - z) * sp;
            if dist <= 0.0 {
                continue;
            }
            nu_max = nu_max.max(h * (k / (PI * dist)).sqrt());
        }
        knife_edge(nu_max)
    }

    /// Facet `i`'s share of the reflection of the ray from (s, z) leaving at
    /// elevation `psi`, and the grazing angle to the facet (None when it can't
    /// reflect that ray).
    pub fn share(&self, i: usize, s: f64, z: f64, psi: f64, k: f64) -> Option<(C, f64)> {
        let f = &self.facets[i];
        let graze = psi - f.alpha;
        if graze <= 1e-9 {
            return None;
        }
        let e = [s, z];
        let h = dot2(sub2(e, f.q), f.normal());
        if h <= 1e-12 {
            return None;
        }
        let ue = dot2(sub2(e, f.q), f.t);
        let u_spec = ue + h / graze.tan();
        let (mut a, mut b) = (f.u0, f.u1);
        let start_d = if a.is_finite() { f.q[0] } else { f64::NEG_INFINITY };
        let end_d = if b.is_finite() { f.q[0] + b * f.t[0] } else { f64::INFINITY };
        for r in &self.ridges {
            if r[0] > s + 1e-9 && r[0] <= start_d + 1e-9 {
                // incoming rays from the element over the ridge
                let pe = sub2(*r, e);
                let c0 = cross(pe, sub2(f.q, e));
                let c1 = cross(pe, f.t);
                if c1.abs() < 1e-15 {
                    if c0 < 0.0 {
                        return None;
                    }
                } else if c1 > 0.0 {
                    a = a.max(-c0 / c1);
                } else {
                    b = b.min(-c0 / c1);
                }
            }
        }
        if !(a < b) {
            return None;
        }
        // Fresnel-zone scale along the facet at the specular point (curvature
        // s³/h of the path difference, as in farground)
        let sg = graze.sin();
        let scale = (k * sg * sg * sg / (h * PI)).sqrt();
        let tail = |u: f64| {
            if u == f64::NEG_INFINITY {
                C::new(1.0, 0.0)
            } else if u == f64::INFINITY {
                C::new(0.0, 0.0)
            } else {
                fresnel_tail((u - u_spec) * scale)
            }
        };
        // ridges beyond the facet: the reflected ray leaves as from the image (the
        // element mirrored in the facet) and passes them as a knife edge (the
        // facet's own far end is in the cut-off above)
        let img = sub2(e, [2.0 * h * f.normal()[0], 2.0 * h * f.normal()[1]]);
        let end_h = if f.u1.is_finite() { f.q[1] + f.u1 * f.t[1] } else { f64::NAN };
        let own_end = |r: &[f64; 2]| (r[0] - end_d).abs() < 1e-9 && (r[1] - end_h).abs() < 1e-9;
        let over = self.over_ridges(img[0], img[1], psi, k, |r| r[0] >= end_d - 1e-9 && !own_end(r));
        Some(((tail(a) - tail(b)) * over, graze))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_profile_is_one_facet_without_ridges() {
        let c = Terrain::uniform(vec![[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]]).cut(0.3);
        assert_eq!(c.facets.len(), 1);
        assert!(c.ridges.is_empty());
        let (w, g) = c.share(0, 3.0, 10.0, 0.2, 0.3).unwrap();
        assert_eq!(w, C::new(1.0, 0.0));
        assert!((g - 0.2).abs() < 1e-15);
        assert_eq!(c.direct(3.0, 10.0, 0.1, 0.3), C::new(1.0, 0.0));
    }

    /// A ridge beyond a level facet cuts the reflected ray as a knife edge seen
    /// from the image: with the reflection's Fresnel zone well inside the facet,
    /// its share is the image's knife-edge factor (the two-ray model of
    /// diffraction over ground), give or take the facet end's ripple.
    #[test]
    fn shadowed_reflection_is_the_image_over_the_ridge() {
        // level for 3 km, then a 230 m wall
        let c = Cut::new(&[[0.0, 0.0], [3000.0, 0.0], [3000.0, 230.0], [5000.0, 230.0]]);
        assert_eq!(c.ridges, vec![[3000.0, 230.0]]);
        let k = 2.0 * PI / 21.1;
        for psi in [3.0f64, 4.0, 5.0, 8.0].map(f64::to_radians) {
            let (w, _) = c.share(0, 0.0, 10.0, psi, k).unwrap();
            // the image at z = -10 seen past the wall's top
            let (sp, cp) = psi.sin_cos();
            let (dx, dz) = (3000.0, 240.0);
            let nu = (dz * cp - dx * sp) * (k / (PI * (dx * cp + dz * sp))).sqrt();
            let f = knife_edge(nu);
            assert!((w - f).norm() < 0.12 * f.norm() + 0.01, "psi {psi}: {w} vs {f}");
        }
    }

    #[test]
    fn hilltop_and_blending() {
        // up 20 m over 200 m, then down 50 m over 100 m, then level
        let p = vec![[0.0, 0.0], [200.0, 20.0], [300.0, -30.0]];
        let c = Terrain::uniform(p.clone()).cut(0.0);
        assert_eq!(c.facets.len(), 3);
        assert_eq!(c.ridges, vec![[200.0, 20.0]]);
        // a low ray from 10 m up is cut by the hill; a high one isn't
        let k = 2.0 * PI / 20.0;
        // (8 m above the ray 200 m out: nu = 0.18, about 0.4 of the field)
        let low = c.direct(0.0, 10.0, 0.01, k).norm();
        assert!(low > 0.3 && low < 0.45, "{low}");
        assert!(c.direct(0.0, 2.0, 0.01, k).norm() < low);
        assert!((c.direct(0.0, 10.0, 0.5, k) - 1.0).norm() < 0.1);
        // two profiles blend halfway between their bearings
        let t = Terrain { profiles: vec![Profile::new(0.0, p), Profile::new(PI / 2.0, vec![[0.0, 0.0], [200.0, 0.0]])] };
        let mid = t.profile_at(PI / 4.0);
        let h = |d: f64| Profile::new(0.0, mid.clone()).height(d);
        assert!((h(200.0) - 10.0).abs() < 1e-9 && (h(300.0) + 15.0).abs() < 1e-9);
        assert!((Profile::new(0.0, t.profile_at(0.0)).height(100.0) - 10.0).abs() < 1e-9);
    }
}
