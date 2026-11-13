//! Wires -> segments -> nodes -> triangle (rooftop) basis functions.
//!
//! Each wire is a straight conductor cut into segments (equal, or any lengths you
//! give).  Segment ends that coincide within the merge tolerance become one node,
//! so touching wires are connected.  A node shared by k >= 2 segment ends carries
//! k - 1 basis functions; a free end carries none (zero current).  On a perfect
//! ground plane every segment end at z = 0 gets a half basis whose other half is
//! its image.
//!
//! A basis function has one or two *halves*.  On a half the current is
//! `c * shape(u) * t` with `t` the segment's unit vector, `u` the distance from the
//! segment start, shape rising (u/L) or falling (1 - u/L), and `c = ±1`.  It is 1 A
//! at its node.

use crate::Error;
use crate::surface::SurfMesh;
use crate::vec3::*;
use std::collections::HashMap;

/// How a wire is cut into segments.
#[derive(Clone, Debug, PartialEq)]
pub enum Segments {
    /// n equal segments.
    Uniform(usize),
    /// Segment lengths from the start to the end, in any unit: they are scaled to
    /// the wire's length.
    Lengths(Vec<f64>),
}

/// A dielectric sleeve on a wire: its thickness (m) and relative permittivity.
///
/// Thin-coating model: across the sleeve the field is that of a coaxial line, so the
/// sleeve changes the potential between the conductor and the medium outside by
/// `q ln(b/a) / 2π · (1/εᵢ − 1/ε̃ₘ)` per unit charge (a = wire radius, b = a +
/// thickness, ε̃ₘ = the medium around the sleeve: air, or the ground's complex
/// permittivity for a wire in the soil). The field outside is that of the bare wire.
/// In air this is the familiar extra series inductance μ0/2π (1 − 1/εᵢ) ln(b/a) for
/// a current varying like the free-space wave; in the soil it also takes the wire
/// out of conductive contact with the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Insulation {
    pub thickness: f64,
    pub eps_r: f64,
}

/// Wire conductor for ohmic (skin-effect) loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Conductor {
    Perfect,
    /// Conductivity (S/m) and relative permeability.
    Lossy { sigma: f64, mu_r: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wire {
    pub start: V3,
    pub end: V3,
    pub radius: f64,
    pub segments: Segments,
    pub conductor: Conductor,
    /// Extra series inductance per metre (H/m).
    pub series_l_per_m: f64,
    /// A dielectric sleeve (see [`Insulation`]).
    pub insulation: Option<Insulation>,
}

impl Wire {
    pub fn new(start: V3, end: V3, radius: f64, nseg: usize) -> Wire {
        Wire { start, end, radius, segments: Segments::Uniform(nseg), conductor: Conductor::Perfect, series_l_per_m: 0.0, insulation: None }
    }
    pub fn with_conductor(mut self, c: Conductor) -> Wire {
        self.conductor = c;
        self
    }
    pub fn length(&self) -> f64 {
        norm(sub(self.end, self.start))
    }
}

pub const RISE: usize = 0;
pub const FALL: usize = 1;

#[derive(Clone, Copy, Debug)]
pub struct Half {
    pub seg: usize,
    /// RISE (u/L) or FALL (1 - u/L)
    pub shape: usize,
    /// ±1: current direction relative to the segment vector
    pub c: f64,
    /// sign of d(current)/du (the charge), in units of 1/L
    pub div: f64,
}

#[derive(Clone, Debug)]
pub struct Basis {
    pub node: usize,
    pub halves: Vec<Half>,
}

/// One end of a segment, as seen from a node: (segment, 0 = start / 1 = end).
type End = (usize, usize);

#[derive(Clone, Debug)]
pub struct Mesh {
    pub p0: Vec<V3>,
    pub p1: Vec<V3>,
    pub t: Vec<V3>,
    pub len: Vec<f64>,
    pub radius: Vec<f64>,
    pub wire_of_seg: Vec<usize>,
    /// Global segment indices of each wire, in order.
    pub wire_segs: Vec<Vec<usize>>,
    /// Node index at each segment's (start, end).
    pub seg_nodes: Vec<[usize; 2]>,
    pub nodes: Vec<V3>,
    pub basis: Vec<Basis>,
    pub ground_plane: bool,
    /// Medium of each segment: AIR (z > 0) or GROUND (z < 0).  Without an
    /// interface (free space, perfect ground) every segment is AIR.
    pub medium: Vec<u8>,
    /// Notes about the geometry (e.g. wire ends left unconnected at ground level).
    pub warnings: Vec<String>,
    /// Conducting surfaces (triangles with RWG functions); their unknowns follow
    /// the wires' in the system.
    pub surf: SurfMesh,
    /// Wire-to-surface junctions: (wire basis with the wire's half, surface vertex).
    /// The basis continues onto the surface around the vertex.
    pub junctions: Vec<(usize, usize)>,
}

pub const AIR: u8 = 0;
pub const GROUND: u8 = 1;

/// How the ground surface z = 0 is treated while meshing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surface {
    /// No interface: free space.
    None,
    /// Perfect ground plane: wires must be at z >= 0; ends on it connect to it.
    Perfect,
    /// Lossy ground: wires may be on either side; a wire crossing z = 0 is cut there,
    /// and every segment lies wholly in one medium.
    Interface,
}

/// Build the mesh.  `merge_tol` (m) joins segment ends; `None` uses the same rule as
/// most wire codes: 1e-4 of the shortest segment, at least 1e-7 m.
pub fn build(wires: &[Wire], surface: Surface, merge_tol: Option<f64>) -> Result<Mesh, Error> {
    build_with(wires, &[], surface, merge_tol)
}

/// As [`build`], with conducting surfaces.
pub fn build_with(wires: &[Wire], surfaces: &[crate::surface::Surface], surface: Surface, merge_tol: Option<f64>) -> Result<Mesh, Error> {
    let ground_plane = surface == Surface::Perfect;
    let mut m = Mesh {
        p0: vec![],
        p1: vec![],
        t: vec![],
        len: vec![],
        radius: vec![],
        wire_of_seg: vec![],
        wire_segs: vec![],
        seg_nodes: vec![],
        nodes: vec![],
        basis: vec![],
        ground_plane,
        medium: vec![],
        warnings: vec![],
        surf: SurfMesh::default(),
        junctions: vec![],
    };
    for (wi, w) in wires.iter().enumerate() {
        let total = w.length();
        if !(total > 0.0) {
            return Err(Error::new(format!("wire {} has zero length", wi + 1)));
        }
        if !(w.radius > 0.0) {
            return Err(Error::new(format!("wire {} has zero radius", wi + 1)));
        }
        let fracs: Vec<f64> = match &w.segments {
            Segments::Uniform(n) => {
                if *n == 0 {
                    return Err(Error::new(format!("wire {} has no segments", wi + 1)));
                }
                vec![1.0 / *n as f64; *n]
            }
            Segments::Lengths(l) => {
                let s: f64 = l.iter().sum();
                if l.is_empty() || l.iter().any(|x| !(*x > 0.0)) || !(s > 0.0) {
                    return Err(Error::new(format!("wire {} has invalid segment lengths", wi + 1)));
                }
                l.iter().map(|x| x / s).collect()
            }
        };
        let mut fracs = fracs;
        if surface == Surface::Interface && w.start[2] * w.end[2] < 0.0 {
            fracs = put_node_at(fracs, w.start[2] / (w.start[2] - w.end[2]), matches!(w.segments, Segments::Uniform(_)));
        }
        let dir = scale(sub(w.end, w.start), 1.0 / total);
        let mut segs = vec![];
        let mut acc = 0.0;
        for (i, f) in fracs.iter().enumerate() {
            let a = add(w.start, scale(dir, acc * total));
            acc += f;
            let b = if i + 1 == fracs.len() { w.end } else { add(w.start, scale(dir, acc * total)) };
            segs.push(m.p0.len());
            m.p0.push(a);
            m.p1.push(b);
            m.t.push(dir);
            m.len.push(norm(sub(b, a)));
            m.radius.push(w.radius);
            m.wire_of_seg.push(wi);
        }
        m.wire_segs.push(segs);
    }
    m.medium = (0..m.p0.len())
        .map(|s| if surface == Surface::Interface && m.p0[s][2] + m.p1[s][2] < 0.0 { GROUND } else { AIR })
        .collect();
    let min_len = m.len.iter().cloned().fold(f64::INFINITY, f64::min);
    let tol = merge_tol.unwrap_or((min_len * 1e-4).max(1e-7));

    // merge segment ends into nodes (grid hash: neighbours within tol share a cell
    // or an adjacent one)
    let cell = |p: V3| [(p[0] / tol).floor() as i64, (p[1] / tol).floor() as i64, (p[2] / tol).floor() as i64];
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut ends_at: Vec<Vec<End>> = vec![];
    m.seg_nodes = vec![[0, 0]; m.p0.len()];
    for s in 0..m.p0.len() {
        for (e, p) in [(0usize, m.p0[s]), (1usize, m.p1[s])] {
            let c = cell(p);
            let mut found = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for &n in list {
                                if norm(sub(m.nodes[n], p)) <= tol {
                                    found = Some(n);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            let n = found.unwrap_or_else(|| {
                m.nodes.push(p);
                ends_at.push(vec![]);
                grid.entry(c).or_default().push(m.nodes.len() - 1);
                m.nodes.len() - 1
            });
            ends_at[n].push((s, e));
            m.seg_nodes[s][e] = n;
        }
    }

    if !surfaces.is_empty() {
        m.surf = SurfMesh::build(surfaces, tol)?;
    }
    let surf_vertex = |p: V3| m.surf.verts.iter().position(|v| norm(sub(*v, p)) <= tol);
    for (n, ends) in ends_at.iter().enumerate() {
        if let Some(vi) = surf_vertex(m.nodes[n]) {
            // wire ends on a surface vertex: each end's current flows into the surface
            for &(s, e) in ends {
                m.junctions.push((m.basis.len(), vi));
                m.basis.push(Basis { node: n, halves: vec![half(s, e, true)] });
            }
            continue;
        }
        if ground_plane && m.nodes[n][2].abs() <= tol {
            for &(s, e) in ends {
                // current flows up out of the ground into the segment
                m.basis.push(Basis { node: n, halves: vec![half(s, e, false)] });
            }
            continue;
        }
        if ends.len() < 2 {
            if surface == Surface::Interface && m.nodes[n][2].abs() <= tol && m.p0[ends[0].0][2] + m.p1[ends[0].0][2] > 0.0 {
                m.warnings.push(format!(
                    "wire {} ends at ground level without continuing into the ground, so no current flows into the ground there; add a ground rod or radials below the surface to connect it",
                    m.wire_of_seg[ends[0].0] + 1
                ));
            }
            continue;
        }
        let (sa, ea) = ends[0];
        for &(sb, eb) in &ends[1..] {
            m.basis.push(Basis { node: n, halves: vec![half(sa, ea, true), half(sb, eb, false)] });
        }
    }
    if !surfaces.is_empty() {
        let zmin = m.surf.verts.iter().map(|v| v[2]).fold(f64::INFINITY, f64::min);
        match surface {
            Surface::Perfect if zmin < -tol => return Err(Error::new("with a ground plane every surface must be at z >= 0")),
            Surface::Interface if zmin <= tol => {
                return Err(Error::new("over real ground, surfaces must be above the ground (z > 0); surfaces in the ground are not supported"));
            }
            _ => {}
        }
    }
    if m.basis.is_empty() {
        return Err(Error::new("the structure has no current unknowns (every wire needs at least 2 segments or a connection)"));
    }
    if ground_plane && m.p0.iter().chain(m.p1.iter()).any(|p| p[2] < -tol) {
        return Err(Error::new("with a ground plane every wire must be at z >= 0"));
    }
    Ok(m)
}

/// Segment fractions with a node at fraction `f0` (where a wire crosses the ground
/// surface).  Uniform wires become uniform on each side; otherwise the nearest
/// interior node moves to `f0` when that keeps both neighbours at least half their
/// length, and only failing that is the straddling segment cut.  (A sliver next to
/// a long segment would hurt the thin-wire integrals.)
fn put_node_at(fracs: Vec<f64>, f0: f64, uniform: bool) -> Vec<f64> {
    let n = fracs.len();
    if uniform && n >= 2 {
        let nb = ((f0 * n as f64).round() as usize).clamp(1, n - 1);
        let mut out = vec![f0 / nb as f64; nb];
        out.extend(vec![(1.0 - f0) / (n - nb) as f64; n - nb]);
        return out;
    }
    let mut edges = vec![0.0];
    for f in &fracs {
        edges.push(edges.last().unwrap() + f);
    }
    // nearest interior node
    if let Some((i, _)) = (1..n).map(|i| (i, (edges[i] - f0).abs())).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        let (left, right) = (f0 - edges[i - 1], edges[i + 1] - f0);
        if left >= 0.5 * fracs[i - 1] && right >= 0.5 * fracs[i] {
            let mut out = fracs.clone();
            out[i - 1] = left;
            out[i] = right;
            return out;
        }
    }
    // cut the straddling segment
    let mut out = fracs.clone();
    if let Some(i) = (0..n).find(|&i| f0 > edges[i] && f0 < edges[i + 1]) {
        out[i] = f0 - edges[i];
        out.insert(i + 1, edges[i + 1] - f0);
    }
    out
}

/// Half of a basis on `seg` whose node is at segment end `end` (0 start, 1 end);
/// `inward` = the current flows toward the node.
fn half(seg: usize, end: usize, inward: bool) -> Half {
    let shape = if end == 1 { RISE } else { FALL };
    let toward = if end == 1 { 1.0 } else { -1.0 };
    let c = if inward { toward } else { -toward };
    let div = c * if shape == RISE { 1.0 } else { -1.0 };
    Half { seg, shape, c, div }
}

impl Mesh {
    pub fn n_segments(&self) -> usize {
        self.len.len()
    }
    pub fn n_unknowns(&self) -> usize {
        self.basis.len() + self.surf.rwg.len()
    }
    /// Number of nodes along wire `w` (segments + 1); node 0 is the wire's start.
    pub fn wire_nodes(&self, w: usize) -> usize {
        self.wire_segs[w].len() + 1
    }
    /// Position of `node` along wire `w`, as a fraction of its length.
    pub fn node_fraction(&self, w: usize, node: usize) -> f64 {
        let segs = &self.wire_segs[w];
        let total: f64 = segs.iter().map(|&s| self.len[s]).sum();
        segs.iter().take(node).map(|&s| self.len[s]).sum::<f64>() / total
    }
    /// Node of wire `w` nearest to fraction `pos` (0 = start, 1 = end) of its length.
    pub fn nearest_node(&self, w: usize, pos: f64) -> usize {
        (0..self.wire_nodes(w))
            .min_by(|&a, &b| {
                (self.node_fraction(w, a) - pos).abs().partial_cmp(&(self.node_fraction(w, b) - pos).abs()).unwrap()
            })
            .unwrap()
    }

    /// The *gap* at `node` of wire `w`: every basis function whose current crosses
    /// that point of the wire, with the sign of its current along the wire
    /// (start -> end).  A source drives all of them; the gap current is their signed
    /// sum.  At a wire end this is the gap between the wire and whatever it joins.
    pub fn gap(&self, w: usize, node: usize) -> Result<Vec<(usize, f64)>, Error> {
        let segs = self.wire_segs.get(w).ok_or_else(|| Error::new(format!("no wire {}", w + 1)))?;
        if node > segs.len() {
            return Err(Error::new(format!("wire {}: node must be 0..={}", w + 1, segs.len())));
        }
        // the segment just before the node (or the first one at node 0), and the end
        let (seg, end) = if node == 0 { (segs[0], 0) } else { (segs[node - 1], 1) };
        let n = self.seg_nodes[seg][end];
        let mut out = vec![];
        for (bi, b) in self.basis.iter().enumerate() {
            if b.node != n {
                continue;
            }
            for h in &b.halves {
                let at_this_end = (h.shape == RISE) == (end == 1);
                if h.seg == seg && at_this_end {
                    out.push((bi, h.c));
                }
            }
        }
        if out.is_empty() {
            return Err(Error::new(format!(
                "wire {} node {} is a free wire end, so it carries no current; put the source or load on another node",
                w + 1,
                node
            )));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dipole_mesh() {
        let m = build(&[Wire::new([0.0, 0.0, -0.25], [0.0, 0.0, 0.25], 1e-3, 10)], Surface::None, None).unwrap();
        assert_eq!(m.n_segments(), 10);
        assert_eq!(m.n_unknowns(), 9);
        assert_eq!(m.gap(0, 5).unwrap(), vec![(4, 1.0)]);
        assert!(m.gap(0, 0).is_err());
        assert_eq!(m.nearest_node(0, 0.5), 5);
    }

    #[test]
    fn junction_gaps() {
        // three wires meeting at the origin
        let w = [
            Wire::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3, 4),
            Wire::new([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1e-3, 4),
            Wire::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1e-3, 4),
        ];
        let m = build(&w, Surface::None, None).unwrap();
        assert_eq!(m.n_unknowns(), 3 * 3 + 2);
        // every wire end at the junction is crossed by at least one basis
        for (wi, node) in [(0, 0), (1, 4), (2, 0)] {
            assert!(!m.gap(wi, node).unwrap().is_empty());
        }
    }

    #[test]
    fn tapered_segments() {
        let w = Wire { segments: Segments::Lengths(vec![1.0, 2.0, 4.0, 2.0, 1.0]), ..Wire::new([0.0; 3], [1.0, 0.0, 0.0], 1e-3, 1) };
        let m = build(&[w], Surface::None, None).unwrap();
        assert!((m.len[2] - 0.4).abs() < 1e-12);
        assert!((m.node_fraction(0, 3) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn monopole_on_ground() {
        let m = build(&[Wire::new([0.0; 3], [0.0, 0.0, 1.0], 1e-3, 5)], Surface::Perfect, None).unwrap();
        assert_eq!(m.n_unknowns(), 5);
        assert_eq!(m.gap(0, 0).unwrap().len(), 1);
    }

    #[test]
    fn crossing_wire_is_cut_at_the_surface() {
        // uniform wire from z=-0.3 to z=0.7: uniform on each side, a node on the surface
        let m = build(&[Wire::new([0.0, 0.0, -0.3], [0.0, 0.0, 0.7], 1e-3, 5)], Surface::Interface, None).unwrap();
        assert_eq!(m.n_segments(), 5);
        assert!(m.p1[1][2].abs() < 1e-12 || m.p1[0][2].abs() < 1e-12);
        assert_eq!(m.medium.iter().filter(|&&x| x == GROUND).count() + m.medium.iter().filter(|&&x| x == AIR).count(), 5);
        // tapered wire: the nearest node moves to the surface
        let w = Wire { segments: Segments::Lengths(vec![1.0, 2.0, 4.0, 2.0, 1.0]), ..Wire::new([0.0, 0.0, -0.32], [0.0, 0.0, 0.68], 1e-3, 1) };
        let m = build(&[w], Surface::Interface, None).unwrap();
        assert_eq!(m.n_segments(), 5);
        assert!(m.p1[1][2].abs() < 1e-12, "{:?}", m.p1);
        // a wire ending on the surface from above alone gets a warning
        let m = build(&[Wire::new([0.0; 3], [0.0, 0.0, 1.0], 1e-3, 4)], Surface::Interface, None).unwrap();
        assert_eq!(m.warnings.len(), 1);
    }
}
