//! Conducting surfaces: flat triangular patches with RWG (Rao–Wilton–Glisson) basis
//! functions, in the same mixed-potential formulation as the wires.
//!
//! An interior edge shared by triangles T+ and T- carries
//! `f(r) = ± l/(2A±) (r - v±)` (v± the vertex opposite the edge), whose divergence
//! is `± l/A±`.  An edge shared by more than two triangles (plates meeting at a
//! seam) gets one function per extra triangle, paired with the first; boundary
//! edges carry none (no current leaves a plate's edge).
//!
//! Over each source triangle the singular part of the Green's function is
//! integrated analytically (Wilton et al. 1984, Graglia 1993):
//! `∫ 1/R dS'` and `∫ (r' - r)/R dS'`; the smooth rest by a 7-point rule.

use crate::Error;
use crate::kernel::expm1;
use crate::vec3::*;
use num_complex::Complex64 as C;
use std::collections::HashMap;
use std::f64::consts::PI;

/// A surface given as triangles over shared vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    pub vertices: Vec<V3>,
    pub triangles: Vec<[usize; 3]>,
}

impl Surface {
    /// A flat parallelogram plate from `corner` along `edge1` and `edge2`, cut into
    /// n1 x n2 cells of two triangles each.
    pub fn plate(corner: V3, edge1: V3, edge2: V3, n1: usize, n2: usize) -> Surface {
        let (n1, n2) = (n1.max(1), n2.max(1));
        let mut vertices = vec![];
        for j in 0..=n2 {
            for i in 0..=n1 {
                vertices.push(add(corner, add(scale(edge1, i as f64 / n1 as f64), scale(edge2, j as f64 / n2 as f64))));
            }
        }
        let id = |i: usize, j: usize| j * (n1 + 1) + i;
        let mut triangles = vec![];
        for j in 0..n2 {
            for i in 0..n1 {
                // alternate the diagonal so the mesh has no preferred direction
                if (i + j) % 2 == 0 {
                    triangles.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                    triangles.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
                } else {
                    triangles.push([id(i, j), id(i + 1, j), id(i, j + 1)]);
                    triangles.push([id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
                }
            }
        }
        Surface { vertices, triangles }
    }
}

/// One half of an RWG function: on triangle `tri`, ±(l/2A)(r - v) with v the
/// triangle's local vertex `free`.
#[derive(Clone, Copy, Debug)]
pub struct RwgHalf {
    pub tri: usize,
    pub free: usize,
    /// +1 on T+, -1 on T-
    pub sign: f64,
}

#[derive(Clone, Debug)]
pub struct Rwg {
    pub len: f64,
    pub halves: [RwgHalf; 2],
}

/// All surfaces merged into one triangle mesh.
#[derive(Clone, Debug, Default)]
pub struct SurfMesh {
    pub verts: Vec<V3>,
    pub tris: Vec<[usize; 3]>,
    pub area: Vec<f64>,
    pub normal: Vec<V3>,
    pub rwg: Vec<Rwg>,
}

impl SurfMesh {
    pub fn build(surfaces: &[Surface], tol: f64) -> Result<SurfMesh, Error> {
        let mut m = SurfMesh::default();
        let cell = |p: V3| [(p[0] / tol).floor() as i64, (p[1] / tol).floor() as i64, (p[2] / tol).floor() as i64];
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for s in surfaces {
            let mut map = vec![0; s.vertices.len()];
            for (i, &p) in s.vertices.iter().enumerate() {
                let c = cell(p);
                let mut found = None;
                'f: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            for &v in grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]).into_iter().flatten() {
                                if norm(sub(m.verts[v], p)) <= tol {
                                    found = Some(v);
                                    break 'f;
                                }
                            }
                        }
                    }
                }
                map[i] = found.unwrap_or_else(|| {
                    m.verts.push(p);
                    grid.entry(c).or_default().push(m.verts.len() - 1);
                    m.verts.len() - 1
                });
            }
            for t in &s.triangles {
                let t = [map[t[0]], map[t[1]], map[t[2]]];
                let n = cross(sub(m.verts[t[1]], m.verts[t[0]]), sub(m.verts[t[2]], m.verts[t[0]]));
                let a2 = norm(n);
                if !(a2 > 0.0) {
                    return Err(Error::new("a surface has a degenerate (zero-area) triangle"));
                }
                m.tris.push(t);
                m.area.push(0.5 * a2);
                m.normal.push(scale(n, 1.0 / a2));
            }
        }
        // edges -> (triangle, local index of the opposite vertex)
        let mut edges: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        for (ti, t) in m.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                edges.entry((a.min(b), a.max(b))).or_default().push((ti, k));
            }
        }
        let mut keys: Vec<_> = edges.keys().copied().collect();
        keys.sort();
        for key in keys {
            let list = &edges[&key];
            let len = norm(sub(m.verts[key.0], m.verts[key.1]));
            for &(tj, kj) in &list[1..] {
                let (ti, ki) = list[0];
                m.rwg.push(Rwg { len, halves: [RwgHalf { tri: ti, free: ki, sign: 1.0 }, RwgHalf { tri: tj, free: kj, sign: -1.0 }] });
            }
        }
        Ok(m)
    }

    pub fn vertex(&self, tri: usize, local: usize) -> V3 {
        self.verts[self.tris[tri][local]]
    }

    pub fn tri_verts(&self, tri: usize) -> [V3; 3] {
        let t = self.tris[tri];
        [self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]]
    }
}

pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// 7-point degree-5 rule on a triangle: barycentric points and weights (sum 1).
pub const TRI7: [([f64; 3], f64); 7] = [
    ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
    ([0.059715871789770, 0.470142064105115, 0.470142064105115], 0.132394152788506),
    ([0.470142064105115, 0.059715871789770, 0.470142064105115], 0.132394152788506),
    ([0.470142064105115, 0.470142064105115, 0.059715871789770], 0.132394152788506),
    ([0.797426985353087, 0.101286507323456, 0.101286507323456], 0.125939180544827),
    ([0.101286507323456, 0.797426985353087, 0.101286507323456], 0.125939180544827),
    ([0.101286507323456, 0.101286507323456, 0.797426985353087], 0.125939180544827),
];

pub fn tri_points(v: &[V3; 3]) -> [V3; 7] {
    TRI7.map(|(b, _)| [0, 1, 2].map(|i| b[0] * v[0][i] + b[1] * v[1][i] + b[2] * v[2][i]))
}

/// Analytic `∫_T 1/R dS'` and `∫_T (r' - r)/R dS'` over triangle `v` for
/// observation point `r`.
pub fn potentials(r: V3, v: &[V3; 3]) -> (f64, V3) {
    let n = {
        let c = cross(sub(v[1], v[0]), sub(v[2], v[0]));
        scale(c, 1.0 / norm(c))
    };
    let d = dot(n, sub(r, v[0]));
    let rho = sub(r, scale(n, d));
    let ad = d.abs();
    let mut i0 = 0.0;
    let mut irho = [0.0; 3];
    for k in 0..3 {
        let (a, b) = (v[k], v[(k + 1) % 3]);
        let el = norm(sub(b, a));
        let lh = scale(sub(b, a), 1.0 / el);
        let u = cross(lh, n); // outward in-plane edge normal
        let lp = dot(sub(b, rho), lh);
        let lm = dot(sub(a, rho), lh);
        let t0 = dot(sub(a, rho), u);
        let r02 = t0 * t0 + d * d;
        let rp = (r02 + lp * lp).sqrt();
        let rm = (r02 + lm * lm).sqrt();
        // ln((R+ + l+)/(R- + l-)), in the form that avoids cancellation
        let lnf = if r02 < 1e-30 * (lp * lp + lm * lm) {
            // projected point on the edge line, no height: only the edge-length terms count
            0.0
        } else if lm >= 0.0 || lp > 0.0 && (rm + lm) > 1e-12 * rm {
            ((rp + lp) / (rm + lm)).ln()
        } else {
            ((rm - lm) / (rp - lp)).ln()
        };
        let atn = if ad > 0.0 || t0 != 0.0 {
            (t0 * lp).atan2(r02 + ad * rp) - (t0 * lm).atan2(r02 + ad * rm)
        } else {
            0.0
        };
        i0 += t0 * lnf - ad * atn;
        let g = 0.5 * (r02 * lnf + lp * rp - lm * rm);
        for i in 0..3 {
            irho[i] += g * u[i];
        }
    }
    // in-plane part ∫(ρ'-ρ)/R plus the normal part -d n ∫1/R
    (i0, sub(irho, scale(n, d * i0)))
}

/// `S0 = ∫_T G dS'` and `V = ∫_T G (r' - r) dS'` for wavenumber k (complex
/// allowed): analytic 1/R part plus the smooth remainder by the 7-point rule.
pub fn green_integrals(r: V3, v: &[V3; 3], area: f64, k: C) -> (C, [C; 3]) {
    let (i0, iv) = potentials(r, v);
    let four_pi = 4.0 * PI;
    let mut s0 = C::new(i0 / four_pi, 0.0);
    let mut sv = iv.map(|x| C::new(x / four_pi, 0.0));
    for (q, (_, w)) in tri_points(v).iter().zip(TRI7) {
        let dd = sub(*q, r);
        let rr = norm(dd);
        if rr == 0.0 {
            // the smooth part's limit is -jk/(4 pi)
            s0 += C::new(0.0, -1.0) * k / four_pi * (w * area);
            continue;
        }
        let f = expm1(C::new(0.0, -1.0) * k * rr) / (four_pi * rr) * (w * area);
        s0 += f;
        for i in 0..3 {
            sv[i] += f * dd[i];
        }
    }
    (s0, sv)
}

/// Integrals between test triangle `p` and source triangle `q`:
/// `a[i][j] = ∫p ∫q (r - v_i)·(r' - v_j) G` and `s = ∫p ∫q G`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TriPair {
    pub a: [[C; 3]; 3],
    pub s: C,
}

pub fn tri_pair(vp: &[V3; 3], area_p: f64, vq: &[V3; 3], area_q: f64, k: C) -> TriPair {
    let mut out = TriPair::default();
    for (r, (_, w)) in tri_points(vp).iter().zip(TRI7) {
        let (s0, sv) = green_integrals(*r, vq, area_q, k);
        let wa = w * area_p;
        out.s += s0 * wa;
        for i in 0..3 {
            let ri = sub(*r, vp[i]);
            for j in 0..3 {
                // (r - v_i)·(r' - v_j) = (r - v_i)·[(r' - r) + (r - v_j)]
                let rj = sub(*r, vq[j]);
                let val = ri[0] * (sv[0] + s0 * rj[0]) + ri[1] * (sv[1] + s0 * rj[1]) + ri[2] * (sv[2] + s0 * rj[2]);
                out.a[i][j] += val * wa;
            }
        }
    }
    out
}

/// Integrals between test wire segment (start p0, unit t, length len) and source
/// triangle q: `a[s][j] = ∫seg shape_s(u) t·∫q G (r' - v_j)` and `s = ∫seg ∫q G`
/// (shape 0 rising, 1 falling), with `n` Gauss points on the segment axis.
#[derive(Clone, Copy, Debug, Default)]
pub struct WireTri {
    pub a: [[C; 3]; 2],
    pub s: C,
}

pub fn wire_tri(p0: V3, t: V3, len: f64, vq: &[V3; 3], area_q: f64, k: C, n: usize) -> WireTri {
    let (x, w) = crate::quad::gauss01(n);
    let mut out = WireTri::default();
    for (&xi, &wi) in x.iter().zip(w) {
        let r = along(p0, t, xi * len);
        let (s0, sv) = green_integrals(r, vq, area_q, k);
        let wl = wi * len;
        out.s += s0 * wl;
        let tv = t[0] * sv[0] + t[1] * sv[1] + t[2] * sv[2];
        for j in 0..3 {
            let rj = sub(r, vq[j]);
            let val = tv + s0 * dot(t, rj);
            out.a[0][j] += val * (wl * xi);
            out.a[1][j] += val * (wl * (1.0 - xi));
        }
    }
    out
}

/// 3-point degree-2 rule on a triangle (weights sum 1).
pub const TRI3: [([f64; 3], f64); 3] = [
    ([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], 1.0 / 3.0),
    ([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], 1.0 / 3.0),
    ([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], 1.0 / 3.0),
];

/// One small triangle of a junction's surface part, with the (constant) current
/// density `f` and divergence `div` of the junction function at its centroid.
#[derive(Clone, Copy, Debug)]
pub struct Sub {
    pub v: [V3; 3],
    pub area: f64,
    pub f: V3,
    pub div: f64,
}

impl Sub {
    pub fn points(&self) -> [(V3, f64); 3] {
        TRI3.map(|(b, w)| ([0, 1, 2].map(|i| b[0] * self.v[0][i] + b[1] * self.v[1][i] + b[2] * self.v[2][i]), w * self.area))
    }
    pub fn centroid(&self) -> V3 {
        [0, 1, 2].map(|i| (self.v[0][i] + self.v[1][i] + self.v[2][i]) / 3.0)
    }
}

/// The surface part of a wire-to-surface junction at mesh vertex `vertex`: 1 A
/// leaves the vertex and spreads over the triangles around it as
/// `f = (r-v)/(Θ|r-v|²)·(1 - ((r-v)·n)²/h²)` (n toward the opposite edge, h its
/// distance, Θ the total angle at v).  Its normal component vanishes on every edge;
/// `div f = -2((r-v)·n)²/(Θ h² |r-v|²)`, integrating to -1 (the charge balancing
/// the wire's).  The 1/|r-v| singularity is resolved by subtriangles graded
/// geometrically toward v.
pub fn junction_part(sm: &SurfMesh, vertex: usize) -> Vec<Sub> {
    let v = sm.verts[vertex];
    let attached: Vec<usize> = (0..sm.tris.len()).filter(|&t| sm.tris[t].contains(&vertex)).collect();
    let mut theta = 0.0;
    let mut tris = vec![];
    for &t in &attached {
        let k = sm.tris[t].iter().position(|&x| x == vertex).unwrap();
        let (a, b) = (sm.vertex(t, (k + 1) % 3), sm.vertex(t, (k + 2) % 3));
        let (ea, eb) = (sub(a, v), sub(b, v));
        theta += (dot(ea, eb) / (norm(ea) * norm(eb))).clamp(-1.0, 1.0).acos();
        tris.push((a, b));
    }
    let mut subs = vec![];
    let (q, layers, across) = (0.5f64, 10usize, 2usize);
    for (a, b) in tris {
        let e = scale(sub(b, a), 1.0 / norm(sub(b, a)));
        let n_raw = sub(sub(a, v), scale(e, dot(sub(a, v), e)));
        let h = norm(n_raw);
        let n = scale(n_raw, 1.0 / h);
        let at = |z: f64, w: f64| add(v, scale(add(sub(a, v), scale(sub(b, a), w)), z));
        let mut push = |p: [V3; 3]| {
            let area = 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            let c = [0, 1, 2].map(|i| (p[0][i] + p[1][i] + p[2][i]) / 3.0);
            let d = sub(c, v);
            let r2 = dot(d, d);
            let dn = dot(d, n);
            let f = scale(d, (1.0 - dn * dn / (h * h)) / (theta * r2));
            let div = -2.0 * dn * dn / (theta * h * h * r2);
            subs.push(Sub { v: p, area, f, div });
        };
        for l in 0..layers {
            let (z_out, z_in) = (q.powi(l as i32), q.powi(l as i32 + 1));
            for j in 0..across {
                let (w0, w1) = (j as f64 / across as f64, (j + 1) as f64 / across as f64);
                let (p00, p01, p10, p11) = (at(z_in, w0), at(z_in, w1), at(z_out, w0), at(z_out, w1));
                push([p00, p10, p11]);
                push([p00, p11, p01]);
            }
        }
        let z0 = q.powi(layers as i32);
        push([v, at(z0, 0.0), at(z0, 1.0)]);
    }
    subs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// brute-force ∫ 1/R and ∫(r'-r)/R over a triangle by fine subdivision
    fn brute(r: V3, v: &[V3; 3]) -> (f64, V3) {
        let n = 400;
        let (mut s, mut sv) = (0.0, [0.0; 3]);
        let area = 0.5 * norm(cross(sub(v[1], v[0]), sub(v[2], v[0])));
        let da = area / (n * n) as f64;
        for i in 0..n {
            for j in 0..n - i {
                for (a, b) in [(i as f64 + 0.31, j as f64 + 0.29), (i as f64 + 0.69, j as f64 + 0.71)] {
                    if a + b > n as f64 {
                        continue;
                    }
                    let (u, w) = (a / n as f64, b / n as f64);
                    let p = [0, 1, 2].map(|k| v[0][k] + u * (v[1][k] - v[0][k]) + w * (v[2][k] - v[0][k]));
                    let d = sub(p, r);
                    let rr = norm(d);
                    s += da / rr;
                    for k in 0..3 {
                        sv[k] += da * d[k] / rr;
                    }
                }
            }
        }
        (s, sv)
    }

    #[test]
    fn analytic_potentials_match_brute_force() {
        let v = [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.3, 0.9, 0.2]];
        for r in [[2.0, 1.5, 1.0], [0.4, 0.3, 0.3], [0.5, 0.5, -0.05], [3.0, -1.0, 0.0], [-0.5, 0.2, 0.1]] {
            let (a, av) = potentials(r, &v);
            let (b, bv) = brute(r, &v);
            assert!((a - b).abs() < 2e-3 * b.abs(), "{r:?}: {a} vs {b}");
            for k in 0..3 {
                assert!((av[k] - bv[k]).abs() < 3e-3 * norm(bv), "{r:?}: {av:?} vs {bv:?}");
            }
        }
    }

    #[test]
    fn observation_point_in_the_triangle() {
        // r on the triangle itself (the singularity is integrable): centroid of a unit
        // equilateral triangle
        let s = 1.0;
        let v = [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.5 * s, 0.75f64.sqrt() * s, 0.0]];
        let c = [0.5, 0.75f64.sqrt() / 3.0, 0.0];
        let (i0, iv) = potentials(c, &v);
        // closed form: each edge contributes h ln((R + 1/2)/(R - 1/2)), h = 1/(2 sqrt 3),
        // R = sqrt(h^2 + 1/4)
        let h = 1.0 / (2.0 * 3f64.sqrt());
        let rr = (h * h + 0.25).sqrt();
        let exact = 3.0 * h * ((rr + 0.5) / (rr - 0.5)).ln();
        assert!((i0 - exact).abs() < 1e-12, "{i0} vs {exact}");
        assert!(norm(iv) < 1e-12, "{iv:?}");
        // vertex of the triangle
        let (i1, _) = potentials(v[0], &v);
        assert!(i1.is_finite() && i1 > 0.0);
    }

    #[test]
    fn plate_mesh_has_interior_edges_only() {
        let p = Surface::plate([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3, 2);
        let m = SurfMesh::build(&[p], 1e-9).unwrap();
        assert_eq!(m.tris.len(), 12);
        // interior edges: 3x2 cells -> horizontal (3*1) + vertical (2*2) + diagonals 6
        assert_eq!(m.rwg.len(), 3 + 4 + 6);
        let total: f64 = m.area.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn junction_part_carries_one_ampere() {
        let p = Surface::plate([-0.5, -0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 4, 4);
        let m = SurfMesh::build(&[p], 1e-9).unwrap();
        let centre = m.verts.iter().position(|v| norm(*v) < 1e-12).unwrap();
        let subs = junction_part(&m, centre);
        let charge: f64 = subs.iter().map(|s| s.div * s.area).sum();
        assert!((charge + 1.0).abs() < 0.02, "{charge}");
        // corner vertex (a quarter of the angle) as well
        let corner = m.verts.iter().position(|v| norm(sub(*v, [-0.5, -0.5, 0.0])) < 1e-12).unwrap();
        let charge: f64 = junction_part(&m, corner).iter().map(|s| s.div * s.area).sum();
        assert!((charge + 1.0).abs() < 0.02, "{charge}");
    }
}
