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
use crate::mesh::{self, Conductor, FALL, Mesh, RISE, Wire};
use crate::quad::gauss01;
use crate::sommerfeld::{self, RealGround, Table};
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
    eps_r: u64,
    sigma: u64,
    bucket: i64,
    reuse: u64,
    r_min: u64,
    r_max: u64,
}

impl TableKey {
    fn new(g: RealGround, freq: f64, r_min: f64, r_max: f64, reuse: f64) -> (TableKey, f64) {
        let step = (1.0 + reuse).ln();
        let bucket = (freq.ln() / step).round() as i64;
        let key = TableKey {
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

    fn get(&self, g: RealGround, freq: f64, r_min: f64, r_max: f64, reuse: f64) -> Arc<Table> {
        if reuse <= 0.0 {
            return Arc::new(Table::new(g, freq, r_min, r_max, 1.0, None));
        }
        let (key, f_b) = TableKey::new(g, freq, r_min, r_max, reuse);
        if let Some(t) = self.lookup(&key) {
            return t;
        }
        let t = Arc::new(Table::new(g, f_b, r_min, r_max, 1.0 + reuse, None));
        self.insert(key, t)
    }

    /// Build (in parallel) every table that solving `model` at `freqs` will need.
    pub fn prepare(&self, model: &Model, freqs: &[f64]) -> Result<(), Error> {
        let Ground::Real(g) = model.ground else { return Ok(()) };
        if model.ground_table_reuse <= 0.0 {
            return Ok(());
        }
        let m = model.mesh()?;
        let (r0, r1) = sommerfeld::table_range(&m, model.n_ground)?;
        let mut need: Vec<(TableKey, f64)> = vec![];
        for &f in freqs {
            let (key, f_b) = TableKey::new(g, f, r0, r1, model.ground_table_reuse);
            if !need.iter().any(|(k, _)| *k == key) && self.lookup(&key).is_none() {
                need.push((key, f_b));
            }
        }
        let reuse = model.ground_table_reuse;
        let built: Vec<(TableKey, Arc<Table>)> =
            need.into_par_iter().map(|(key, f_b)| (key, Arc::new(Table::new(g, f_b, r0, r1, 1.0 + reuse, None)))).collect();
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
    /// Segment ends closer than this (m) are joined; None = 1e-4 of the shortest
    /// segment, at least 1e-7 m.
    pub merge_tol: Option<f64>,
    /// Reuse a Sommerfeld table within this relative frequency distance; 0 rebuilds
    /// it for every frequency.  Reuse within ±1% costs ~0.1% in impedance.
    pub ground_table_reuse: f64,
    pub tables: TableCache,
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
            merge_tol: None,
            ground_table_reuse: 0.02,
            tables: TableCache::default(),
        }
    }

    pub fn add_source(&mut self, wire: usize, node: usize, voltage: C) {
        self.sources.push(Source { wire, node, voltage });
    }

    pub fn add_load(&mut self, wire: usize, node: usize, kind: LoadKind) {
        self.loads.push(Load { wire, node, kind });
    }

    pub fn mesh(&self) -> Result<Mesh, Error> {
        mesh::build(&self.wires, self.ground == Ground::Perfect, self.merge_tol)
    }

    /// Impedance matrix (row-major, n_unknowns²) for `mesh` at `freq` (Hz).
    pub fn impedance_matrix(&self, m: &Mesh, freq: f64) -> Result<Vec<C>, Error> {
        self.impedance_matrix_with(m, freq, None)
    }

    /// As [`impedance_matrix`](Self::impedance_matrix); `table` replaces the
    /// Sommerfeld remainder table (testing) and then no closed-form image is added.
    pub fn impedance_matrix_with(&self, m: &Mesh, freq: f64, table: Option<&Table>) -> Result<Vec<C>, Error> {
        let omega = 2.0 * PI * freq;
        let k = omega / C0;
        let n = m.n_segments();
        let jwmu = C::new(0.0, omega * MU0);
        let inv_jweps = C::new(0.0, -1.0 / (omega * EPS0));

        let (img, real_k) = match self.ground {
            Ground::Free => (None, None),
            Ground::Perfect => (Some(C::new(1.0, 0.0)), None),
            Ground::Real(g) => {
                if m.p0.iter().chain(m.p1.iter()).any(|p| p[2] <= 0.0) {
                    return Err(Error::new(
                        "over real ground all wires must be above z = 0 (wires buried in or touching real ground are not supported yet)",
                    ));
                }
                let k_int = match table {
                    Some(t) => sommerfeld::segment_integrals(t, m, self.n_ground, freq),
                    None => {
                        let (r0, r1) = sommerfeld::table_range(m, self.n_ground)?;
                        let t = self.tables.get(g, freq, r0, r1, self.ground_table_reuse);
                        sommerfeld::segment_integrals(&t, m, self.n_ground, freq)
                    }
                };
                (Some(if table.is_some() { C::new(0.0, 0.0) } else { g.gamma0(freq) }), Some(k_int))
            }
        };

        // per segment pair: vector-potential block v[a][b] and scalar term s
        let seg = |s: usize| kernel::Seg { p0: m.p0[s], t: m.t[s], len: m.len[s], radius: m.radius[s] };
        let pairs: Vec<Vec<([[C; 2]; 2], C)>> = (0..n)
            .into_par_iter()
            .map(|p| {
                let sp = seg(p);
                (0..n)
                    .map(|q| {
                        let sq = seg(q);
                        let f: PairIntegrals = kernel::pair(&sp, &sq, k, self.n_outer, self.n_inner);
                        let tt = dot(m.t[p], m.t[q]);
                        let mut v = f.t.map(|r| r.map(|x| jwmu * tt * x));
                        let mut s = inv_jweps * f.s;
                        if let Some(g) = img {
                            if g != C::new(0.0, 0.0) {
                                // image: reflected segment with negated current and charge
                                let si = kernel::Seg { p0: reflect(m.p0[q]), t: reflect(m.t[q]), ..sq };
                                let fi = kernel::pair(&sp, &si, k, self.n_outer, self.n_inner);
                                let tti = dot(m.t[p], reflect(m.t[q]));
                                for a in 0..2 {
                                    for b in 0..2 {
                                        v[a][b] -= g * jwmu * tti * fi.t[a][b];
                                    }
                                }
                                s -= g * inv_jweps * fi.s;
                            }
                        }
                        if let Some(kk) = &real_k {
                            // reflected field E = D . J tested: Z = -<t, E>
                            let kp = kk[p * n + q];
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

        let nb = m.n_unknowns();
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
        Ok(z)
    }

    /// Solve at `freq` (Hz).
    pub fn solve(&self, freq: f64) -> Result<Solution, Error> {
        if self.sources.is_empty() {
            return Err(Error::new("the model has no source"));
        }
        let t0 = Instant::now();
        let m = Arc::new(self.mesh()?);
        let mut z = self.impedance_matrix(&m, freq)?;
        let t1 = Instant::now();
        let nb = m.n_unknowns();
        let mut v = vec![C::new(0.0, 0.0); nb];
        let mut gaps = vec![];
        for s in &self.sources {
            let g = m.gap(s.wire, s.node)?;
            for &(b, sign) in &g {
                v[b] += s.voltage * sign;
            }
            gaps.push(g);
        }
        lu::solve(&mut z, &mut v, nb).map_err(Error::new)?;
        let t_fill_ms = (t1 - t0).as_secs_f64() * 1e3;
        let t_solve_ms = t1.elapsed().as_secs_f64() * 1e3;
        Ok(Solution {
            mesh: m,
            freq,
            currents: v,
            sources: self.sources.clone(),
            gaps,
            ground: self.ground,
            elements: OnceLock::new(),
            t_fill_ms,
            t_solve_ms,
        })
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
    gaps: Vec<Vec<(usize, f64)>>,
    pub ground: Ground,
    /// far-field current elements (6 Gauss points per segment), built on first use
    elements: OnceLock<Vec<(V3, [C; 3])>>,
    /// matrix fill (incl. meshing and Sommerfeld tables) and LU solve times
    pub t_fill_ms: f64,
    pub t_solve_ms: f64,
}

impl Solution {
    /// Current through each source's gap (along its wire).
    pub fn source_currents(&self) -> Vec<C> {
        self.gaps.iter().map(|g| g.iter().map(|&(b, s)| self.currents[b] * s).sum()).collect()
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
        out
    }

    /// Far-field factors (E_theta, E_phi)·r·exp(jkr) toward (theta from zenith, phi).
    pub fn far_field(&self, theta: f64, phi: f64) -> (C, C) {
        let k = 2.0 * PI * self.freq / C0;
        let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        let rhat = [st * cp, st * sp, ct];
        let th = [ct * cp, ct * sp, -st];
        let ph = [-sp, cp, 0.0];
        let els = self.elements.get_or_init(|| self.current_elements(6));
        let mut nd = [C::new(0.0, 0.0); 3];
        let mut ni = [C::new(0.0, 0.0); 3];
        for (p, c) in els {
            let ph_d = C::from_polar(1.0, k * dot(rhat, *p));
            for i in 0..3 {
                nd[i] += c[i] * ph_d;
            }
            if self.ground != Ground::Free {
                // image current -P J at the mirror point
                let ph_i = C::from_polar(1.0, k * dot(rhat, reflect(*p)));
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
            _ if theta > PI / 2.0 + 1e-12 => return (C::new(0.0, 0.0), C::new(0.0, 0.0)),
            Ground::Perfect => {
                e_th += pref * proj(&ni, th);
                e_ph += pref * proj(&ni, ph);
            }
            Ground::Real(g) => {
                let (g_te, g_tm) = g.reflection(self.freq, theta.min(PI / 2.0));
                e_th += g_tm * pref * proj(&ni, th);
                e_ph -= g_te * pref * proj(&ni, ph);
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
