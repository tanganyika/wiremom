//! Antenna model, impedance-matrix assembly, solution and far field.
//!
//! With triangle basis/test functions Λm (Galerkin) the MPIE matrix is
//!
//! ```text
//! Z_mn = jωμ ∫∫ Λm Λn (ŝm·ŝn) G + 1/(jωε) ∫∫ Λm' Λn' G  (+ ground, loss, loads)
//! ```
//!
//! and Z I = V with V the delta-gap voltages.

use crate::kernel::{self, PairIntegrals};
use crate::lu;
use crate::mesh::{self, AIR, Conductor, FALL, GROUND, Mesh, RISE, Surface, Wire};
use crate::quad::gauss01;
use crate::sommerfeld::{self, Blend, Eval, RealGround, Side, Table, Window};
use crate::surface;
use crate::vec3::*;
use crate::{C0, EPS0, ETA0, Error, MU0};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ground {
    Free,
    /// Perfectly conducting plane at z = 0 (wires may touch it).
    Perfect,
    /// Lossy half-space below z = 0, Sommerfeld solution (wires must be above it).
    Real(RealGround),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadKind {
    /// R + jwL + 1/(jwC); `c: None` = no capacitor.
    Series { r: f64, l: f64, c: Option<f64> },
    /// R || L || C; a zero value means that element is absent.  (A trap's loss is
    /// a *large* parallel R, about Q·wL.)
    Parallel { r: f64, l: f64, c: f64 },
    /// A fixed impedance (ohms) at the solve frequency.
    Impedance(C),
}

impl LoadKind {
    /// Impedance at angular frequency `omega`; `None` = open circuit.
    pub fn impedance(&self, omega: f64) -> Option<C> {
        let jw = C::new(0.0, omega);
        match *self {
            LoadKind::Series { r, l, c } => Some(C::new(r, 0.0) + jw * l + c.map_or(C::new(0.0, 0.0), |c| 1.0 / (jw * c))),
            LoadKind::Parallel { r, l, c } => {
                let mut y = C::new(0.0, 0.0);
                if r != 0.0 {
                    y += 1.0 / r;
                }
                if l != 0.0 {
                    y += 1.0 / (jw * l);
                }
                if c != 0.0 {
                    y += jw * c;
                }
                if y.norm() == 0.0 { None } else { Some(1.0 / y) }
            }
            LoadKind::Impedance(z) => z.is_finite().then_some(z),
        }
    }
}

/// Voltage source across the gap at `node` (0..=nseg) of wire `wire` (0-based).
/// Positive voltage drives current from the wire's start toward its end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Source {
    pub wire: usize,
    pub node: usize,
    pub voltage: C,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Load {
    pub wire: usize,
    pub node: usize,
    pub kind: LoadKind,
}

/// A network port: the gap at `node` of wire `wire` (see [`Mesh::gap`]); its "+"
/// side is toward the wire's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Port {
    pub wire: usize,
    pub node: usize,
}

/// A non-radiating two-port connected between two gaps, as NEC's TL and NT cards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TwoPortKind {
    /// Transmission line.  `length` None = the straight distance between the ports;
    /// `crossed` reverses the conductors at the second port (NEC's negative Z0);
    /// `shunt` admittances (S) sit across each end.
    Line { z0: f64, length: Option<f64>, velocity_factor: f64, loss_db_per_100m: f64, crossed: bool, shunt: [C; 2] },
    /// General network by its short-circuit admittance parameters (S).
    Admittance { y11: C, y12: C, y22: C },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoPort {
    pub a: Port,
    pub b: Port,
    pub kind: TwoPortKind,
}

impl TwoPortKind {
    /// A lossless line with velocity factor 1 and no shunts.
    pub fn line(z0: f64, length: Option<f64>) -> TwoPortKind {
        TwoPortKind::Line { z0, length, velocity_factor: 1.0, loss_db_per_100m: 0.0, crossed: false, shunt: [C::new(0.0, 0.0); 2] }
    }

    /// Short-circuit admittance matrix (y11, y12, y22) at angular frequency `omega`,
    /// with port currents flowing into the network; `dist` is the port spacing.
    pub fn admittance(&self, omega: f64, dist: f64) -> Result<(C, C, C), Error> {
        match *self {
            TwoPortKind::Admittance { y11, y12, y22 } => Ok((y11, y12, y22)),
            TwoPortKind::Line { z0, length, velocity_factor, loss_db_per_100m, crossed, shunt } => {
                let l = length.unwrap_or(dist);
                if !(l > 0.0) || !(z0.abs() > 0.0) || !(velocity_factor > 0.0) {
                    return Err(Error::new("a transmission line needs a length, impedance and velocity factor above zero"));
                }
                let alpha = loss_db_per_100m.max(0.0) * std::f64::consts::LN_10 / 20.0 / 100.0;
                let beta = omega / (velocity_factor * C0);
                // a lossless line a whole number of half wavelengths long has no
                // admittance matrix (sinh(γl) = 0); 1e-9 Np of loss keeps it finite
                // and leaves the results accurate to ~1e-7
                let gl = C::new((alpha * l).max(1e-9), beta * l);
                let (sh, ch) = (gl.sinh(), gl.cosh());
                let y11 = ch / sh / z0;
                let mut y12 = -1.0 / (sh * z0);
                if crossed {
                    y12 = -y12;
                }
                Ok((y11 + shunt[0], y12, y11 + shunt[1]))
            }
        }
    }
}

/// Shared cache of Sommerfeld tables, for sweeps and repeated solves.  Cloning
/// shares the cache.  Frequencies are grouped in logarithmic buckets of width
/// `reuse` and one table serves a bucket.
///
/// The cache never makes a thread wait for another thread's build: waiting inside
/// rayon work can deadlock (a waiting worker may pick up a nested job that needs
/// the same table).  A miss simply builds the table; [`prepare`](Self::prepare)
/// builds the tables a sweep needs up front, so its frequencies don't all miss at once.
#[derive(Clone, Default)]
pub struct TableCache {
    inner: Arc<Mutex<Vec<(TableKey, Arc<Table>)>>>,
}

#[derive(Clone, Copy, PartialEq)]
struct TableKey {
    side: u8,
    theta: u64,
    /// frequency bits of a table built exactly at one frequency (0 for buckets)
    exact: u64,
    eps_r: u64,
    sigma: u64,
    bucket: i64,
    reuse: u64,
    r_min: u64,
    r_max: u64,
}

impl TableKey {
    /// Key and centre frequency of frequency bucket `bucket` (width ln(1+reuse)).
    fn new(g: RealGround, side: Side, bucket: i64, win: Window, reuse: f64) -> (TableKey, f64) {
        let (r_min, r_max) = (win.r_min, win.r_max);
        let step = (1.0 + reuse).ln();
        let key = TableKey {
            side: side as u8,
            theta: win.theta_min.to_bits(),
            exact: 0,
            eps_r: g.eps_r.to_bits(),
            sigma: g.sigma.to_bits(),
            bucket,
            reuse: reuse.to_bits(),
            r_min: r_min.to_bits(),
            r_max: r_max.to_bits(),
        };
        (key, (bucket as f64 * step).exp())
    }
}

impl TableCache {
    const CAPACITY: usize = 64;

    fn lookup(&self, key: &TableKey) -> Option<Arc<Table>> {
        let mut v = self.inner.lock().unwrap();
        let i = v.iter().position(|(k, _)| k == key)?;
        let e = v.remove(i);
        v.push(e.clone());
        Some(e.1)
    }

    fn insert(&self, key: TableKey, t: Arc<Table>) -> Arc<Table> {
        let mut v = self.inner.lock().unwrap();
        if let Some((_, existing)) = v.iter().find(|(k, _)| *k == key) {
            return existing.clone(); // another thread finished first
        }
        v.push((key, t.clone()));
        if v.len() > Self::CAPACITY {
            v.remove(0);
        }
        t
    }

    /// The two buckets around `freq` and the blend weight toward the upper one.
    fn buckets(freq: f64, reuse: f64) -> (i64, i64, f64) {
        let x = freq.ln() / (1.0 + reuse).ln();
        let lo = x.floor();
        (lo as i64, lo as i64 + 1, x - lo)
    }

    fn get(&self, g: RealGround, side: Side, freq: f64, win: Window, reuse: f64) -> Arc<dyn Eval> {
        if reuse <= 0.0 {
            return Arc::new(Table::new_window(g, freq, win, 1.0, None, side));
        }
        let (lo, hi, w) = Self::buckets(freq, reuse);
        let (klo, _) = TableKey::new(g, side, lo, win, reuse);
        let (khi, _) = TableKey::new(g, side, hi, win, reuse);
        if let (Some(a), Some(b)) = (self.lookup(&klo), self.lookup(&khi)) {
            // a sweep prepared the buckets: blend them
            return Arc::new(Blend { a, b, w });
        }
        // a single frequency: one exact table (cached for repeat calculations)
        let (mut key, _) = TableKey::new(g, side, 0, win, reuse);
        key.exact = freq.to_bits();
        if let Some(t) = self.lookup(&key) {
            return t;
        }
        let t = Arc::new(Table::new_window(g, freq, win, 1.0, None, side));
        self.insert(key, t)
    }

    /// Build (in parallel) the frequency-bucket tables a sweep over `freqs` needs;
    /// solves then blend the two around each frequency.  Without this, each
    /// frequency gets its own exact table.
    pub fn prepare(&self, model: &Model, freqs: &[f64]) -> Result<(), Error> {
        let Ground::Real(g) = model.ground else { return Ok(()) };
        if model.ground_table_reuse <= 0.0 {
            return Ok(());
        }
        let m = model.mesh()?;
        let mut need: Vec<(TableKey, f64, Side, Window)> = vec![];
        for side in [Side::Air, Side::Ground] {
            let pts = if side == Side::Air { sommerfeld::air_points(&m, model.n_ground) } else { sommerfeld::ground_points(&m, model.n_ground) };
            if pts.is_empty() {
                continue;
            }
            let win = sommerfeld::window_of_points(&pts)?;
            for &f in freqs {
                let (lo, hi, _) = Self::buckets(f, model.ground_table_reuse);
                for b in [lo, hi] {
                    let (key, f_b) = TableKey::new(g, side, b, win, model.ground_table_reuse);
                    if !need.iter().any(|(k, ..)| *k == key) && self.lookup(&key).is_none() {
                        need.push((key, f_b, side, win));
                    }
                }
            }
        }
        let reuse = model.ground_table_reuse;
        let built: Vec<(TableKey, Arc<Table>)> =
            need.into_par_iter().map(|(key, f_b, side, win)| (key, Arc::new(Table::new_window(g, f_b, win, 1.0 + reuse, None, side)))).collect();
        for (key, t) in built {
            self.insert(key, t);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Model {
    pub wires: Vec<Wire>,
    pub sources: Vec<Source>,
    pub loads: Vec<Load>,
    pub ground: Ground,
    /// Gauss points per segment: outer (test) and inner (source) integrals.
    pub n_outer: usize,
    pub n_inner: usize,
    /// Gauss points per segment for the Sommerfeld remainder.
    pub n_ground: usize,
    /// Fewer quadrature points for well-separated pairs of segments/triangles
    /// (default on; changes results by ~1e-5 relative, much faster).
    pub adaptive: bool,
    /// Segment ends closer than this (m) are joined; None = 1e-4 of the shortest
    /// segment, at least 1e-7 m.
    pub merge_tol: Option<f64>,
    /// Reuse a Sommerfeld table within this relative frequency distance; 0 rebuilds
    /// it for every frequency.  The two tables around each frequency are blended,
    /// so 0.04 (the default) costs under ~5e-5 in impedance.
    pub ground_table_reuse: f64,
    pub tables: TableCache,
    /// Transmission lines and other two-ports between gaps.
    pub networks: Vec<TwoPort>,
    /// Conducting surfaces (plates); currently not connected to wires.
    pub surfaces: Vec<surface::Surface>,
}

impl Model {
    pub fn new(wires: Vec<Wire>, ground: Ground) -> Model {
        Model {
            wires,
            sources: vec![],
            loads: vec![],
            ground,
            n_outer: 8,
            n_inner: 8,
            n_ground: 6,
            adaptive: true,
            merge_tol: None,
            ground_table_reuse: 0.04,
            tables: TableCache::default(),
            networks: vec![],
            surfaces: vec![],
        }
    }

    pub fn add_source(&mut self, wire: usize, node: usize, voltage: C) {
        self.sources.push(Source { wire, node, voltage });
    }

    /// Connect a two-port (e.g. [`TwoPortKind::line`]) between two gaps.  A source on
    /// one of its ports drives the network and the wire in parallel (as in NEC), so a
    /// short "feed wire" with a source and a line to the antenna models a feed line;
    /// a 0 V source is a short circuit (e.g. a shorted stub).
    pub fn add_network(&mut self, a: Port, b: Port, kind: TwoPortKind) {
        self.networks.push(TwoPort { a, b, kind });
    }

    pub fn add_load(&mut self, wire: usize, node: usize, kind: LoadKind) {
        self.loads.push(Load { wire, node, kind });
    }

    pub fn mesh(&self) -> Result<Mesh, Error> {
        let surface = match self.ground {
            Ground::Free => Surface::None,
            Ground::Perfect => Surface::Perfect,
            Ground::Real(_) => Surface::Interface,
        };
        mesh::build_with(&self.wires, &self.surfaces, surface, self.merge_tol)
    }

    /// The air-side Sommerfeld table for this mesh (wires and surfaces).
    fn air_table(&self, m: &Mesh, g: RealGround, freq: f64) -> Result<Arc<dyn Eval>, Error> {
        let win = sommerfeld::window_of_points(&sommerfeld::air_points(m, self.n_ground))?;
        Ok(self.tables.get(g, Side::Air, freq, win, self.ground_table_reuse))
    }

    /// Impedance matrix (row-major, n_unknowns²) for `mesh` at `freq` (Hz).
    pub fn impedance_matrix(&self, m: &Mesh, freq: f64) -> Result<Vec<C>, Error> {
        self.impedance_matrix_with(m, freq, None)
    }

    /// As [`impedance_matrix`](Self::impedance_matrix); `table` replaces the
    /// Sommerfeld remainder table (testing) and then no closed-form image is added.
    pub fn impedance_matrix_with(&self, m: &Mesh, freq: f64, table: Option<&dyn Eval>) -> Result<Vec<C>, Error> {
        let omega = 2.0 * PI * freq;
        let k = C::new(omega / C0, 0.0);
        let n = m.n_segments();
        let jwmu = C::new(0.0, omega * MU0);
        let inv_jweps = C::new(0.0, -1.0 / (omega * EPS0));
        let zero = C::new(0.0, 0.0);

        // segments on each side of the surface, and their index within that side
        let air: Vec<usize> = (0..n).filter(|&s| m.medium[s] == AIR).collect();
        let gnd: Vec<usize> = (0..n).filter(|&s| m.medium[s] == GROUND).collect();
        let mut loc = vec![0usize; n];
        for (i, &s) in air.iter().enumerate() {
            loc[s] = i;
        }
        for (i, &s) in gnd.iter().enumerate() {
            loc[s] = i;
        }

        // ground: image scale in the air, and the tabulated / direct remainders
        let mut img_air = zero;
        let mut eps = C::new(1.0, 0.0);
        let mut g0 = zero;
        let (mut k_air, mut k_gnd, mut k_cross) = (None, None, None);
        // the air-side table (built here unless `table` overrides it), also used by
        // the surfaces
        let mut air_owned: Option<Arc<dyn Eval>> = None;
        match self.ground {
            Ground::Free => {}
            Ground::Perfect => img_air = C::new(1.0, 0.0),
            Ground::Real(g) => {
                eps = g.eps_c(freq);
                g0 = g.gamma0(freq);
                img_air = if table.is_some() { zero } else { g0 };
                if (!air.is_empty() || !m.surf.tris.is_empty()) && table.is_none() {
                    air_owned = Some(self.air_table(m, g, freq)?);
                }
                if !air.is_empty() {
                    let t = table.or(air_owned.as_deref()).unwrap();
                    k_air = Some(sommerfeld::segment_integrals_of(t, m, self.n_ground, freq, &air));
                }
                if !gnd.is_empty() {
                    let win = sommerfeld::window_of_points(&sommerfeld::ground_points(m, self.n_ground))?;
                    let t = self.tables.get(g, Side::Ground, freq, win, self.ground_table_reuse);
                    k_gnd = Some(sommerfeld::segment_integrals_of(&*t, m, self.n_ground, freq, &gnd));
                    if !air.is_empty() {
                        k_cross = Some(sommerfeld::cross_integrals(m, freq, eps, &gnd, &air));
                    }
                }
            }
        }
        let k_ground = k * eps.sqrt();
        let inv_jweps_g = inv_jweps / eps;
        // charge seen across the surface: quasi-static transmission 2/(1+eps)
        let c_cross = 2.0 / (1.0 + eps);
        let (na, ng) = (air.len(), gnd.len());

        // per segment pair: vector-potential block v[a][b] and scalar term s
        let seg = |s: usize| kernel::Seg { p0: m.p0[s], t: m.t[s], len: m.len[s], radius: m.radius[s] };
        let pairs: Vec<Vec<([[C; 2]; 2], C)>> = (0..n)
            .into_par_iter()
            .map(|p| {
                let sp = seg(p);
                (0..n)
                    .map(|q| {
                        let sq = seg(q);
                        let tt = dot(m.t[p], m.t[q]);
                        let (mp, mq) = (m.medium[p], m.medium[q]);
                        let (kk, s_scale, img, inv_e) = match (mp, mq) {
                            (AIR, AIR) => (k, C::new(1.0, 0.0), img_air, inv_jweps),
                            (GROUND, GROUND) => (k_ground, C::new(1.0, 0.0), -g0, inv_jweps_g),
                            _ => (k, c_cross, zero, inv_jweps),
                        };
                        let (no, ni) = if self.adaptive {
                            let n = kernel::order(&sp, &sq, self.n_outer.max(self.n_inner));
                            (n.min(self.n_outer), n.min(self.n_inner))
                        } else {
                            (self.n_outer, self.n_inner)
                        };
                        let f: PairIntegrals = kernel::pair(&sp, &sq, kk, no, ni);
                        let mut v = f.t.map(|r| r.map(|x| jwmu * tt * x));
                        let mut s = s_scale * inv_e * f.s;
                        if img != zero {
                            // image: reflected segment with negated current and charge
                            let si = kernel::Seg { p0: reflect(m.p0[q]), t: reflect(m.t[q]), ..sq };
                            let (no, ni) = if self.adaptive {
                                let n = kernel::order(&sp, &si, self.n_outer.max(self.n_inner));
                                (n.min(self.n_outer), n.min(self.n_inner))
                            } else {
                                (self.n_outer, self.n_inner)
                            };
                            let fi = kernel::pair(&sp, &si, kk, no, ni);
                            let tti = dot(m.t[p], reflect(m.t[q]));
                            for a in 0..2 {
                                for b in 0..2 {
                                    v[a][b] -= img * jwmu * tti * fi.t[a][b];
                                }
                            }
                            s -= img * inv_e * fi.s;
                        }
                        // remainder fields E = D . J tested: Z = -<t, E>
                        let rem: Option<[[C; 2]; 2]> = match (mp, mq) {
                            (AIR, AIR) => k_air.as_ref().map(|kk| kk[loc[p] * na + loc[q]]),
                            (GROUND, GROUND) => k_gnd.as_ref().map(|kk| kk[loc[p] * ng + loc[q]]),
                            (GROUND, _) => k_cross.as_ref().map(|kk| kk[loc[p] * na + loc[q]]),
                            _ => k_cross.as_ref().map(|kk| {
                                let x = kk[loc[q] * na + loc[p]];
                                [[x[0][0], x[1][0]], [x[0][1], x[1][1]]]
                            }),
                        };
                        if let Some(kp) = rem {
                            for a in 0..2 {
                                for b in 0..2 {
                                    v[a][b] -= kp[a][b];
                                }
                            }
                        }
                        (v, s)
                    })
                    .collect()
            })
            .collect();

        let nb = m.basis.len();
        let mut z = vec![C::new(0.0, 0.0); nb * nb];
        z.par_chunks_mut(nb).enumerate().for_each(|(mi, row)| {
            for ha in &m.basis[mi].halves {
                let da = ha.div / m.len[ha.seg];
                let prow = &pairs[ha.seg];
                for (ni, bn) in m.basis.iter().enumerate() {
                    let mut acc = C::new(0.0, 0.0);
                    for hb in &bn.halves {
                        let (v, s) = &prow[hb.seg];
                        acc += v[ha.shape][hb.shape] * (ha.c * hb.c) + s * (da * hb.div / m.len[hb.seg]);
                    }
                    row[ni] += acc;
                }
            }
        });
        drop(pairs);

        // distributed series impedance: skin-effect R' and extra L', times the Gram
        // integral of two linear shapes on one segment (L/3 same shape, L/6 opposite)
        let mut on_seg: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
        for (bi, b) in m.basis.iter().enumerate() {
            for (hi, h) in b.halves.iter().enumerate() {
                on_seg[h.seg].push((bi, hi));
            }
        }
        for s in 0..n {
            let w = &self.wires[m.wire_of_seg[s]];
            let mut zp = C::new(0.0, omega * w.series_l_per_m);
            if let Conductor::Lossy { sigma, mu_r } = w.conductor {
                zp += (omega * MU0 * mu_r / (2.0 * sigma)).sqrt() / (2.0 * PI * m.radius[s]);
            }
            if zp == C::new(0.0, 0.0) {
                continue;
            }
            for &(ma, ia) in &on_seg[s] {
                for &(nb_, ib) in &on_seg[s] {
                    let (ha, hb) = (m.basis[ma].halves[ia], m.basis[nb_].halves[ib]);
                    let gram = m.len[s] / if ha.shape == hb.shape { 3.0 } else { 6.0 };
                    z[ma * nb + nb_] += zp * (ha.c * hb.c * gram);
                }
            }
        }

        // lumped loads across gaps; open circuits as a penalty forcing the gap current to 0
        let zmax = z.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for ld in &self.loads {
            let gap = m.gap(ld.wire, ld.node)?;
            let zl = ld.kind.impedance(omega).unwrap_or(C::new(1e12 * zmax, 0.0));
            for &(a, sa) in &gap {
                for &(b, sb) in &gap {
                    z[a * nb + b] += zl * (sa * sb);
                }
            }
        }
        self.add_surfaces(m, z, freq, img_air, table.or(air_owned.as_deref()))
    }

    /// Extend the wire matrix with the surfaces' RWG unknowns: plate–plate and
    /// wire–plate interactions, their ground images and the real-ground remainder.
    fn add_surfaces(&self, m: &Mesh, zw: Vec<C>, freq: f64, img: C, table: Option<&dyn Eval>) -> Result<Vec<C>, Error> {
        let sm = &m.surf;
        let nw = m.basis.len();
        let ns = sm.rwg.len();
        if ns == 0 && m.junctions.is_empty() {
            return Ok(zw);
        }
        let n = nw + ns;
        let omega = 2.0 * PI * freq;
        let k = C::new(omega / C0, 0.0);
        let jwmu = C::new(0.0, omega * MU0);
        let inv_jweps = C::new(0.0, -1.0 / (omega * EPS0));
        let zero = C::new(0.0, 0.0);
        let nt = sm.tris.len();
        let verts: Vec<[V3; 3]> = (0..nt).map(|t| sm.tri_verts(t)).collect();
        let iverts: Vec<[V3; 3]> = verts.iter().map(|v| v.map(reflect)).collect();
        let real = matches!(self.ground, Ground::Real(_));
        // triangle-pair integrals, combined with the image (-img x reflected source)
        // and minus the remainder (Z = -<f, D f>)
        let tt: Vec<Vec<([[C; 3]; 3], C)>> = (0..nt)
            .into_par_iter()
            .map(|p| {
                (0..nt)
                    .map(|q| {
                        let f = surface::tri_pair(&verts[p], sm.area[p], &verts[q], sm.area[q], k);
                        let mut a = f.a.map(|r| r.map(|x| jwmu * x));
                        let mut s = inv_jweps * f.s;
                        if img != zero {
                            let fi = surface::tri_pair(&verts[p], sm.area[p], &iverts[q], sm.area[q], k);
                            for i in 0..3 {
                                for j in 0..3 {
                                    a[i][j] -= img * jwmu * fi.a[i][j];
                                }
                            }
                            s -= img * inv_jweps * fi.s;
                        }
                        if let (true, Some(t)) = (real, table) {
                            let kr = sommerfeld::tri_tri_remainder(t, freq, &verts[p], sm.area[p], &verts[q], sm.area[q]);
                            for i in 0..3 {
                                for j in 0..3 {
                                    a[i][j] -= kr[i][j];
                                }
                            }
                        }
                        (a, s)
                    })
                    .collect()
            })
            .collect();
        // wire segment (test) - triangle (source) integrals, likewise
        let nseg = m.n_segments();
        let wt: Vec<Vec<([[C; 3]; 2], C)>> = (0..nseg)
            .into_par_iter()
            .map(|p| {
                (0..nt)
                    .map(|q| {
                        let f = surface::wire_tri(m.p0[p], m.t[p], m.len[p], &verts[q], sm.area[q], k, self.n_outer);
                        let mut a = f.a.map(|r| r.map(|x| jwmu * x));
                        let mut s = inv_jweps * f.s;
                        if img != zero {
                            let fi = surface::wire_tri(m.p0[p], m.t[p], m.len[p], &iverts[q], sm.area[q], k, self.n_outer);
                            for i in 0..2 {
                                for j in 0..3 {
                                    a[i][j] -= img * jwmu * fi.a[i][j];
                                }
                            }
                            s -= img * inv_jweps * fi.s;
                        }
                        if let (true, Some(t)) = (real, table) {
                            let kr = sommerfeld::seg_tri_remainder(t, freq, m.p0[p], m.t[p], m.len[p], self.n_ground, &verts[q], sm.area[q]);
                            for i in 0..2 {
                                for j in 0..3 {
                                    a[i][j] -= kr[i][j];
                                }
                            }
                        }
                        (a, s)
                    })
                    .collect()
            })
            .collect();

        // RWG half: (triangle, free vertex, sign * l/(2A)) and its divergence sign * l/A
        let coef = |h: &surface::RwgHalf, len: f64| (h.sign * len / (2.0 * sm.area[h.tri]), h.sign * len / sm.area[h.tri]);
        let mut z = vec![zero; n * n];
        for i in 0..nw {
            z[i * n..i * n + nw].copy_from_slice(&zw[i * nw..(i + 1) * nw]);
        }
        // surface rows: plate-plate and plate-wire (by reciprocity, = wire-plate)
        let rows: Vec<Vec<C>> = (0..ns)
            .into_par_iter()
            .map(|mi| {
                let bm = &sm.rwg[mi];
                let mut row = vec![zero; n];
                for hm in &bm.halves {
                    let (fm, dm) = coef(hm, bm.len);
                    for (ni, bn) in sm.rwg.iter().enumerate() {
                        let mut acc = zero;
                        for hn in &bn.halves {
                            let (fn_, dn) = coef(hn, bn.len);
                            let (a, s) = &tt[hm.tri][hn.tri];
                            acc += a[hm.free][hn.free] * (fm * fn_) + s * (dm * dn);
                        }
                        row[nw + ni] += acc;
                    }
                    for (wi, bw) in m.basis.iter().enumerate() {
                        let mut acc = zero;
                        for hw in &bw.halves {
                            let (a, s) = &wt[hw.seg][hm.tri];
                            let dw = hw.div / m.len[hw.seg];
                            acc += a[hw.shape][hm.free] * (hw.c * fm) + s * (dw * dm);
                        }
                        row[wi] += acc;
                    }
                }
                row
            })
            .collect();
        for (mi, row) in rows.into_iter().enumerate() {
            let r = nw + mi;
            z[r * n..(r + 1) * n].copy_from_slice(&row);
            for wi in 0..nw {
                z[wi * n + r] = row[wi];
            }
        }
        if !m.junctions.is_empty() {
            self.add_junctions(m, &mut z, n, freq, img, table, &verts, &iverts);
        }
        Ok(z)
    }

    /// Wire-to-surface junctions: each junction basis (a wire half, already in the
    /// matrix) continues as a surface part S_J around its vertex.  Adds ⟨S_J, L y⟩ to
    /// row and column J for every unknown y, and ⟨S_J, L S_K⟩ between junctions.
    #[allow(clippy::too_many_arguments)]
    fn add_junctions(&self, m: &Mesh, z: &mut [C], n: usize, freq: f64, img: C, table: Option<&dyn Eval>, verts: &[[V3; 3]], iverts: &[[V3; 3]]) {
        let sm = &m.surf;
        let nw = m.basis.len();
        let omega = 2.0 * PI * freq;
        let k = C::new(omega / C0, 0.0);
        let jwmu = C::new(0.0, omega * MU0);
        let inv_jweps = C::new(0.0, -1.0 / (omega * EPS0));
        let zero = C::new(0.0, 0.0);
        let real = matches!(self.ground, Ground::Real(_));
        let parts: Vec<Vec<surface::Sub>> = m.junctions.iter().map(|&(_, v)| surface::junction_part(sm, v)).collect();
        let segs: Vec<kernel::Seg> = (0..m.n_segments()).map(|s| kernel::Seg { p0: m.p0[s], t: m.t[s], len: m.len[s], radius: m.radius[s] }).collect();
        let isegs: Vec<kernel::Seg> = segs.iter().map(|g| kernel::Seg { p0: reflect(g.p0), t: reflect(g.t), ..*g }).collect();
        let nseg = segs.len();
        // quadrature samples of every unknown for the real-ground remainder:
        // (point, current-density weight vector)
        let samples: Vec<Vec<(V3, V3)>> = if real {
            let (x, w) = gauss01(self.n_ground);
            let mut out: Vec<Vec<(V3, V3)>> = m
                .basis
                .iter()
                .map(|b| {
                    b.halves
                        .iter()
                        .flat_map(|h| {
                            let sh = |xi: f64| if h.shape == RISE { xi } else { 1.0 - xi };
                            x.iter().zip(w).map(move |(&xi, &wi)| (along(m.p0[h.seg], m.t[h.seg], xi * m.len[h.seg]), scale(m.t[h.seg], h.c * sh(xi) * wi * m.len[h.seg])))
                        })
                        .collect()
                })
                .collect();
            for b in &sm.rwg {
                let mut v = vec![];
                for h in &b.halves {
                    let f = h.sign * b.len / (2.0 * sm.area[h.tri]);
                    for (r, (_, w)) in surface::tri_points(&verts[h.tri]).iter().zip(surface::TRI7) {
                        v.push((*r, scale(sub(*r, verts[h.tri][h.free]), f * w * sm.area[h.tri])));
                    }
                }
                out.push(v);
            }
            out
        } else {
            vec![]
        };
        let remainder = |r: V3, a: V3, pts: &[(V3, V3)]| -> C {
            let t = table.unwrap();
            pts.iter()
                .map(|(s, b)| {
                    let (dx, dy) = (r[0] - s[0], r[1] - s[1]);
                    let rho = (dx * dx + dy * dy).sqrt();
                    let (cx, cy) = if rho > 1e-12 * (1.0 + rho) { (dx / rho, dy / rho) } else { (1.0, 0.0) };
                    let d = t.eval(rho, r[2] + s[2], freq);
                    let (ar, af) = (a[0] * cx + a[1] * cy, -a[0] * cy + a[1] * cx);
                    let (br, bf) = (b[0] * cx + b[1] * cy, -b[0] * cy + b[1] * cx);
                    d[0] * (ar * br) + d[1] * (af * bf) + d[2] * (ar * b[2]) + d[3] * (a[2] * br) + d[4] * (a[2] * b[2])
                })
                .sum()
        };
        let pf = |v: V3| [v[0], v[1], -v[2]];
        for (ji, (&(jb, _), subs)) in m.junctions.iter().zip(&parts).enumerate() {
            // test points of S_J: (point, weight, f, div)
            let tests: Vec<(V3, f64, V3, f64)> = subs.iter().flat_map(|s| s.points().map(|(r, w)| (r, w, s.f, s.div))).collect();
            let g: Vec<C> = tests
                .par_iter()
                .map(|&(r, w, f, dv)| {
                    let mut g = vec![zero; n];
                    // wire unknowns
                    let lg: Vec<((C, C), (C, C))> = (0..nseg)
                        .map(|q| {
                            let fr = kernel::line_green(r, &segs[q], segs[q].radius, k, self.n_inner);
                            let im = if img != zero { kernel::line_green(r, &isegs[q], segs[q].radius, k, self.n_inner) } else { (zero, zero) };
                            (fr, im)
                        })
                        .collect();
                    for (yi, b) in m.basis.iter().enumerate() {
                        let mut acc = zero;
                        for h in &b.halves {
                            let ((s0, s1), (s0i, s1i)) = lg[h.seg];
                            let (sh, shi) = if h.shape == RISE { (s1, s1i) } else { (s0 - s1, s0i - s1i) };
                            let (ft, fti) = (dot(f, segs[h.seg].t), dot(f, isegs[h.seg].t));
                            acc += jwmu * h.c * (ft * sh - img * fti * shi) + inv_jweps * (h.div / m.len[h.seg]) * dv * (s0 - img * s0i);
                        }
                        g[yi] = acc;
                    }
                    // RWG unknowns
                    let tg: Vec<((C, [C; 3]), (C, [C; 3]))> = (0..sm.tris.len())
                        .map(|q| {
                            let fr = surface::green_integrals(r, &verts[q], sm.area[q], k);
                            let im = if img != zero { surface::green_integrals(r, &iverts[q], sm.area[q], k) } else { (zero, [zero; 3]) };
                            (fr, im)
                        })
                        .collect();
                    for (bi, b) in sm.rwg.iter().enumerate() {
                        let mut acc = zero;
                        for h in &b.halves {
                            let ((s0, sv), (s0i, svi)) = tg[h.tri];
                            let (fc, dc) = (h.sign * b.len / (2.0 * sm.area[h.tri]), h.sign * b.len / sm.area[h.tri]);
                            let (d, di) = (sub(r, verts[h.tri][h.free]), sub(r, pf(verts[h.tri][h.free])));
                            let vec: C = (0..3).map(|i| f[i] * (sv[i] + d[i] * s0)).sum();
                            let veci: C = (0..3).map(|i| f[i] * (svi[i] + di[i] * s0i)).sum();
                            acc += jwmu * fc * (vec - img * veci) + inv_jweps * dc * dv * (s0 - img * s0i);
                        }
                        g[nw + bi] = acc;
                    }
                    if real {
                        for (yi, pts) in samples.iter().enumerate() {
                            g[yi] -= remainder(r, f, pts);
                        }
                    }
                    g.into_iter().map(|x| x * w).collect::<Vec<C>>()
                })
                .reduce(|| vec![zero; n], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
            for y in 0..n {
                z[jb * n + y] += g[y];
                z[y * n + jb] += g[y];
            }
            // surface part against every junction's surface part
            for (ki, (&(kb, _), ksubs)) in m.junctions.iter().zip(&parts).enumerate() {
                let ksamp: Vec<(V3, V3)> = ksubs.iter().map(|s| (s.centroid(), scale(s.f, s.area))).collect();
                let h: C = tests
                    .par_iter()
                    .map(|&(r, w, f, dv)| {
                        let mut acc = zero;
                        for s in ksubs {
                            let (s0, _) = surface::green_integrals(r, &s.v, s.area, k);
                            let (s0i, _) = if img != zero { surface::green_integrals(r, &s.v.map(pf), s.area, k) } else { (zero, [zero; 3]) };
                            acc += jwmu * (dot(f, s.f) * s0 - img * dot(f, pf(s.f)) * s0i) + inv_jweps * dv * s.div * (s0 - img * s0i);
                        }
                        if real {
                            acc -= remainder(r, f, &ksamp);
                        }
                        acc * w
                    })
                    .sum();
                let _ = (ji, ki);
                z[jb * n + kb] += h;
            }
        }
    }

    /// Solve at `freq` (Hz).
    pub fn solve(&self, freq: f64) -> Result<Solution, Error> {
        if self.sources.is_empty() {
            return Err(Error::new("the model has no source"));
        }
        let t0 = Instant::now();
        let m = Arc::new(self.mesh()?);
        let z = self.impedance_matrix(&m, freq)?;
        let t1 = Instant::now();
        let nb = m.n_unknowns();
        let lu = lu::Lu::factor(z, nb).map_err(Error::new)?;
        let (currents, src_current) = if self.networks.is_empty() {
            // every port is a source: superpose directly
            let mut v = vec![C::new(0.0, 0.0); nb];
            let mut gaps = vec![];
            for s in &self.sources {
                let g = m.gap(s.wire, s.node)?;
                for &(b, sign) in &g {
                    v[b] += s.voltage * sign;
                }
                gaps.push(g);
            }
            lu.solve(&mut v);
            let ic = gaps.iter().map(|g| g.iter().map(|&(b, s)| v[b] * s).sum()).collect();
            (v, ic)
        } else {
            self.solve_ports(&m, &lu, freq)?
        };
        let t_fill_ms = (t1 - t0).as_secs_f64() * 1e3;
        let t_solve_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok(Solution {
            warnings: m.warnings.clone(),
            mesh: m,
            freq,
            currents,
            sources: self.sources.clone(),
            src_current,
            ground: self.ground,
            elements: OnceLock::new(),
            t_fill_ms,
            t_solve_ms,
        })
    }

    /// Port analysis for networks: the wire structure's admittance seen from every
    /// port (one solve per port), plus the networks' admittances, gives the port
    /// equations (Y_wire + Y_net) V = I_source.  Source ports have known V; the
    /// others carry no external current.  Returns basis currents and the current each
    /// source delivers (into the wire and the networks at its port).
    fn solve_ports(&self, m: &Mesh, lu: &lu::Lu, freq: f64) -> Result<(Vec<C>, Vec<C>), Error> {
        let omega = 2.0 * PI * freq;
        let nb = m.n_unknowns();
        let mut ports: Vec<Port> = vec![];
        let index = |p: Port, ports: &mut Vec<Port>| match ports.iter().position(|q| *q == p) {
            Some(i) => i,
            None => {
                ports.push(p);
                ports.len() - 1
            }
        };
        let src_port: Vec<usize> = self.sources.iter().map(|s| index(Port { wire: s.wire, node: s.node }, &mut ports)).collect();
        let net_ports: Vec<(usize, usize)> = self.networks.iter().map(|n| (index(n.a, &mut ports), index(n.b, &mut ports))).collect();
        let np = ports.len();
        let gaps: Vec<Vec<(usize, f64)>> = ports.iter().map(|p| m.gap(p.wire, p.node)).collect::<Result<_, _>>()?;
        // unit-voltage solutions, in parallel
        let x: Vec<Vec<C>> = gaps
            .par_iter()
            .map(|g| {
                let mut v = vec![C::new(0.0, 0.0); nb];
                for &(b, s) in g {
                    v[b] += s;
                }
                lu.solve(&mut v);
                v
            })
            .collect();
        let mut y = vec![C::new(0.0, 0.0); np * np];
        for h in 0..np {
            for g in 0..np {
                y[h * np + g] = gaps[h].iter().map(|&(b, s)| x[g][b] * s).sum();
            }
        }
        let node_pos = |p: &Port| -> V3 {
            let segs = &m.wire_segs[p.wire];
            if p.node == 0 { m.p0[segs[0]] } else { m.p1[segs[p.node - 1]] }
        };
        for (n, &(a, b)) in self.networks.iter().zip(&net_ports) {
            let dist = norm(sub(node_pos(&n.a), node_pos(&n.b)));
            let (y11, y12, y22) = n.kind.admittance(omega, dist)?;
            y[a * np + a] += y11;
            y[a * np + b] += y12;
            y[b * np + a] += y12;
            y[b * np + b] += y22;
        }
        // known port voltages from the sources
        let mut v = vec![C::new(0.0, 0.0); np];
        let mut known = vec![false; np];
        for (s, &p) in self.sources.iter().zip(&src_port) {
            if known[p] {
                return Err(Error::new("two sources share one gap"));
            }
            known[p] = true;
            v[p] = s.voltage;
        }
        let unk: Vec<usize> = (0..np).filter(|&i| !known[i]).collect();
        if !unk.is_empty() {
            let nu = unk.len();
            let mut a = vec![C::new(0.0, 0.0); nu * nu];
            let mut r = vec![C::new(0.0, 0.0); nu];
            for (i, &pi) in unk.iter().enumerate() {
                for (j, &pj) in unk.iter().enumerate() {
                    a[i * nu + j] = y[pi * np + pj];
                }
                r[i] = -(0..np).filter(|&k| known[k]).map(|k| y[pi * np + k] * v[k]).sum::<C>();
            }
            lu::solve(&mut a, &mut r, nu).map_err(|_| Error::new("the network has no solution (e.g. a line of exactly half a wavelength between two open ends)"))?;
            for (i, &pi) in unk.iter().enumerate() {
                v[pi] = r[i];
            }
        }
        let mut currents = vec![C::new(0.0, 0.0); nb];
        for (g, vg) in v.iter().enumerate() {
            if *vg != C::new(0.0, 0.0) {
                for (c, xg) in currents.iter_mut().zip(&x[g]) {
                    *c += vg * xg;
                }
            }
        }
        let src_current = src_port.iter().map(|&p| (0..np).map(|g| y[p * np + g] * v[g]).sum()).collect();
        Ok((currents, src_current))
    }

    /// Solve at many frequencies in parallel (sharing Sommerfeld tables).
    pub fn sweep(&self, freqs: &[f64]) -> Vec<Result<Solution, Error>> {
        // a failure here (e.g. wires under ground) is reported by every solve below
        let _ = self.tables.prepare(self, freqs);
        freqs.par_iter().map(|&f| self.solve(f)).collect()
    }
}

/// Far-field gain components (linear, relative to input power).
#[derive(Clone, Copy, Debug, Default)]
pub struct Gain {
    pub total: f64,
    /// E_theta (vertical) part
    pub theta: f64,
    /// E_phi (horizontal) part
    pub phi: f64,
}

pub struct Solution {
    pub mesh: Arc<Mesh>,
    pub freq: f64,
    /// Basis coefficients = node currents (A).
    pub currents: Vec<C>,
    pub sources: Vec<Source>,
    /// current each source delivers (into its gap, and any networks on it)
    src_current: Vec<C>,
    pub ground: Ground,
    /// far-field current elements (6 Gauss points per segment) and their media,
    /// built on first use
    elements: OnceLock<(Vec<(V3, [C; 3])>, Vec<u8>)>,
    /// notes about the geometry (see `Mesh::warnings`)
    pub warnings: Vec<String>,
    /// matrix fill (incl. meshing and Sommerfeld tables) and LU solve times
    pub t_fill_ms: f64,
    pub t_solve_ms: f64,
}

impl Solution {
    /// Current each source delivers: through its gap along the wire, plus into any
    /// network connected at the same gap.
    pub fn source_currents(&self) -> Vec<C> {
        self.src_current.clone()
    }
    /// Impedance at every source (ohms).
    pub fn source_impedances(&self) -> Vec<C> {
        self.sources.iter().zip(self.source_currents()).map(|(s, i)| s.voltage / i).collect()
    }
    pub fn input_impedance(&self) -> C {
        self.source_impedances()[0]
    }
    /// Total input power ½ Re Σ V I* (W).
    pub fn input_power(&self) -> f64 {
        self.sources.iter().zip(self.source_currents()).map(|(s, i)| 0.5 * (s.voltage * i.conj()).re).sum()
    }
    pub fn vswr(&self, z0: f64) -> f64 {
        let z = self.input_impedance();
        let g = ((z - z0) / (z + z0)).norm();
        if g >= 1.0 { f64::INFINITY } else { (1.0 + g) / (1.0 - g) }
    }

    /// Per segment: coefficients of the rising and falling current shapes.
    fn segment_amplitudes(&self) -> Vec<[C; 2]> {
        let m = &self.mesh;
        let mut a = vec![[C::new(0.0, 0.0); 2]; m.n_segments()];
        for (b, &i) in m.basis.iter().zip(&self.currents) {
            for h in &b.halves {
                a[h.seg][h.shape] += i * h.c;
            }
        }
        a
    }

    /// Current at the middle of every segment: (start, end, current along the wire).
    pub fn segment_currents(&self) -> Vec<(V3, V3, C)> {
        let m = &self.mesh;
        self.segment_amplitudes().iter().enumerate().map(|(s, a)| (m.p0[s], m.p1[s], 0.5 * (a[RISE] + a[FALL]))).collect()
    }

    /// Current elements for far-field sums: `n_q` Gauss points per segment, each
    /// (position, current moment I·dl as a vector, A·m).  Images are not included.
    pub fn current_elements(&self, n_q: usize) -> Vec<(V3, [C; 3])> {
        let m = &self.mesh;
        let (x, w) = gauss01(n_q);
        let amps = self.segment_amplitudes();
        let mut out = Vec::with_capacity(m.n_segments() * n_q);
        for s in 0..m.n_segments() {
            for (&xi, &wi) in x.iter().zip(w) {
                let i = amps[s][RISE] * xi + amps[s][FALL] * (1.0 - xi);
                let mom = i * (wi * m.len[s]);
                out.push((along(m.p0[s], m.t[s], xi * m.len[s]), [mom * m.t[s][0], mom * m.t[s][1], mom * m.t[s][2]]));
            }
        }
        // surfaces: the 7-point rule on every triangle
        out.extend(self.surface_elements());
        out
    }

    /// Current elements for the far field, with their media: Gauss points along
    /// each segment (2 while the segment is under ~λ/8, more for longer ones) and
    /// the 3- or 7-point rule on triangles (by their electrical size).
    fn far_elements(&self) -> (Vec<(V3, [C; 3])>, Vec<u8>) {
        let m = &self.mesh;
        let k = 2.0 * PI * self.freq / C0;
        let amps = self.segment_amplitudes();
        let mut out = vec![];
        let mut med = vec![];
        for s in 0..m.n_segments() {
            let n = ((2.5 * k * m.len[s]).ceil() as usize).clamp(2, 16);
            let (x, w) = gauss01(n);
            for (&xi, &wi) in x.iter().zip(w) {
                let i = amps[s][RISE] * xi + amps[s][FALL] * (1.0 - xi);
                let mom = i * (wi * m.len[s]);
                out.push((along(m.p0[s], m.t[s], xi * m.len[s]), [mom * m.t[s][0], mom * m.t[s][1], mom * m.t[s][2]]));
                med.push(m.medium[s]);
            }
        }
        let sm = &m.surf;
        // small triangles: the 3-point rule (exact for their linear currents' moments
        // up to the phase variation across them)
        let small = sm.area.iter().all(|a| k * a.sqrt() < 0.8);
        out.extend(self.surface_elements_with(if small { &surface::TRI3[..] } else { &surface::TRI7[..] }));
        med.resize(out.len(), AIR);
        (out, med)
    }

    /// Surface current elements: (point, J·dA) at the 7-point rule of each triangle.
    pub fn surface_elements(&self) -> Vec<(V3, [C; 3])> {
        self.surface_elements_with(&surface::TRI7[..])
    }

    /// Surface current elements on the given triangle rule (barycentric points,
    /// weights summing to 1), plus the junctions' surface parts.
    fn surface_elements_with(&self, rule: &[([f64; 3], f64)]) -> Vec<(V3, [C; 3])> {
        let m = &self.mesh;
        let sm = &m.surf;
        let nw = m.basis.len();
        let mut on_tri: Vec<Vec<(usize, &surface::RwgHalf, f64)>> = vec![vec![]; sm.tris.len()];
        for (bi, b) in sm.rwg.iter().enumerate() {
            for h in &b.halves {
                on_tri[h.tri].push((bi, h, b.len));
            }
        }
        let mut out = Vec::with_capacity(sm.tris.len() * 7);
        for (t, list) in on_tri.iter().enumerate() {
            let v = sm.tri_verts(t);
            for &(bc, w) in rule {
                let r = &[0, 1, 2].map(|i| bc[0] * v[0][i] + bc[1] * v[1][i] + bc[2] * v[2][i]);
                let mut j = [C::new(0.0, 0.0); 3];
                for &(bi, h, len) in list {
                    let f = h.sign * len / (2.0 * sm.area[t]);
                    let d = sub(*r, v[h.free]);
                    for i in 0..3 {
                        j[i] += self.currents[nw + bi] * (f * d[i]);
                    }
                }
                out.push((*r, j.map(|x| x * (w * sm.area[t]))));
            }
        }
        // junction surface parts
        for &(jb, v) in &m.junctions {
            for sb in surface::junction_part(sm, v) {
                let a = self.currents[jb] * sb.area;
                out.push((sb.centroid(), sb.f.map(|x| a * x)));
            }
        }
        out
    }

    /// Far-field factors (E_theta, E_phi)·r·exp(jkr) toward (theta from zenith, phi).
    pub fn far_field(&self, theta: f64, phi: f64) -> (C, C) {
        let k = 2.0 * PI * self.freq / C0;
        let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        let rhat = [st * cp, st * sp, ct];
        let th = [ct * cp, ct * sp, -st];
        let ph = [-sp, cp, 0.0];
        let (els, med) = self.elements.get_or_init(|| {
            let (e, med) = self.far_elements();
            (e, med)
        });
        if self.ground != Ground::Free && theta > PI / 2.0 + 1e-12 {
            return (C::new(0.0, 0.0), C::new(0.0, 0.0));
        }
        let trans = match self.ground {
            Ground::Real(g) if med.contains(&GROUND) => Some(sommerfeld::transmission(&g, self.freq, theta.min(PI / 2.0), phi)),
            _ => None,
        };
        let mut nd = [C::new(0.0, 0.0); 3];
        let mut ni = [C::new(0.0, 0.0); 3];
        let (mut et_b, mut ep_b) = (C::new(0.0, 0.0), C::new(0.0, 0.0));
        for ((p, c), &md) in els.iter().zip(med) {
            if md == GROUND {
                // buried current: transmitted through the surface (by reciprocity)
                let (t_te, t_tm, e, kz2) = trans.unwrap();
                let phase = C::from_polar(1.0, k * st * (cp * p[0] + sp * p[1])) * (C::new(0.0, -1.0) * kz2 * (-p[2])).exp();
                et_b += t_tm * (e[0] * c[0] + e[1] * c[1] + e[2] * c[2]) * phase;
                ep_b += t_te * (ph[0] * c[0] + ph[1] * c[1]) * phase;
                continue;
            }
            if self.ground == Ground::Free {
                let ph_d = C::from_polar(1.0, k * dot(rhat, *p));
                for i in 0..3 {
                    nd[i] += c[i] * ph_d;
                }
                continue;
            }
            // direct and image phases share the horizontal part
            let ph_h = C::from_polar(1.0, k * (rhat[0] * p[0] + rhat[1] * p[1]));
            let ph_v = C::from_polar(1.0, k * rhat[2] * p[2]);
            let ph_d = ph_h * ph_v;
            for i in 0..3 {
                nd[i] += c[i] * ph_d;
            }
            {
                // image current -P J at the mirror point
                let ph_i = ph_h * ph_v.conj();
                ni[0] -= c[0] * ph_i;
                ni[1] -= c[1] * ph_i;
                ni[2] += c[2] * ph_i;
            }
        }
        let pref = C::new(0.0, -2.0 * PI * self.freq * MU0 / (4.0 * PI));
        let proj = |v: &[C; 3], u: V3| v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
        let mut e_th = pref * proj(&nd, th);
        let mut e_ph = pref * proj(&nd, ph);
        match self.ground {
            Ground::Free => {}
            Ground::Perfect => {
                e_th += pref * proj(&ni, th);
                e_ph += pref * proj(&ni, ph);
            }
            Ground::Real(g) => {
                let (g_te, g_tm) = g.reflection(self.freq, theta.min(PI / 2.0));
                e_th += g_tm * pref * proj(&ni, th) + pref * et_b;
                e_ph += -g_te * pref * proj(&ni, ph) + pref * ep_b;
            }
        }
        (e_th, e_ph)
    }

    /// Power gain toward (theta from zenith, phi), linear; includes wire, load and
    /// ground losses.
    pub fn gain(&self, theta: f64, phi: f64) -> Gain {
        let (et, ep) = self.far_field(theta, phi);
        let c = 4.0 * PI / (2.0 * ETA0 * self.input_power());
        Gain { total: c * (et.norm_sqr() + ep.norm_sqr()), theta: c * et.norm_sqr(), phi: c * ep.norm_sqr() }
    }

    pub fn gain_dbi(&self, theta: f64, phi: f64) -> f64 {
        10.0 * self.gain(theta, phi).total.max(1e-30).log10()
    }

    /// Radiated (space-wave) power by integrating the pattern (W).
    pub fn radiated_power(&self, n_theta: usize, n_phi: usize) -> f64 {
        let th_max = if self.ground == Ground::Free { PI } else { PI / 2.0 };
        let (x, w) = gauss01(n_theta);
        let n_theta = x.len();
        let p_in = self.input_power();
        let s: f64 = (0..n_theta * n_phi)
            .into_par_iter()
            .map(|i| {
                let (it, ip) = (i / n_phi, i % n_phi);
                let th = x[it] * th_max;
                let ph = ip as f64 * 2.0 * PI / n_phi as f64;
                self.gain(th, ph).total * p_in / (4.0 * PI) * th.sin() * w[it]
            })
            .sum();
        s * th_max * 2.0 * PI / n_phi as f64
    }

    /// Radiated / input power.  Over real ground the power the ground absorbs counts
    /// as lost.
    pub fn efficiency(&self) -> f64 {
        self.radiated_power(91, 72) / self.input_power()
    }
}
