//! Tensor-product cubic spline interpolation on a rectangular (non-uniform) grid,
//! evaluated as bicubic Hermite patches with spline slopes and cross derivatives.

use num_complex::Complex64 as C;

/// Slopes of the C2 cubic spline through (x, y); end slopes come from the parabola
/// through the three end points.
pub fn spline_slopes(x: &[f64], y: &[C]) -> Vec<C> {
    let n = x.len();
    if n < 2 {
        return vec![C::new(0.0, 0.0); n];
    }
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let d: Vec<C> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
    if n == 2 {
        return vec![d[0], d[0]];
    }
    let m0 = d[0] - (d[1] - d[0]) / (h[0] + h[1]) * h[0];
    let mn = d[n - 2] + (d[n - 2] - d[n - 3]) / (h[n - 3] + h[n - 2]) * h[n - 2];
    let mut m = vec![C::new(0.0, 0.0); n];
    m[0] = m0;
    m[n - 1] = mn;
    if n == 3 {
        m[1] = d[0] + (d[1] - d[0]) / (h[0] + h[1]) * h[0];
        return m;
    }
    // interior: (1/h_{i-1}) m_{i-1} + 2 (1/h_{i-1} + 1/h_i) m_i + (1/h_i) m_{i+1}
    //         = 3 (d_{i-1}/h_{i-1} + d_i/h_i)
    let k = n - 2;
    let mut diag = vec![0.0; k];
    let mut lower = vec![0.0; k];
    let mut upper = vec![0.0; k];
    let mut rhs = vec![C::new(0.0, 0.0); k];
    for j in 0..k {
        let i = j + 1;
        lower[j] = 1.0 / h[i - 1];
        upper[j] = 1.0 / h[i];
        diag[j] = 2.0 * (lower[j] + upper[j]);
        rhs[j] = 3.0 * (d[i - 1] / h[i - 1] + d[i] / h[i]);
    }
    rhs[0] -= lower[0] * m0;
    rhs[k - 1] -= upper[k - 1] * mn;
    // Thomas algorithm
    for j in 1..k {
        let f = lower[j] / diag[j - 1];
        diag[j] -= f * upper[j - 1];
        let prev = rhs[j - 1];
        rhs[j] -= prev * f;
    }
    rhs[k - 1] /= diag[k - 1];
    for j in (0..k - 1).rev() {
        let next = rhs[j + 1];
        rhs[j] = (rhs[j] - next * upper[j]) / diag[j];
    }
    m[1..n - 1].copy_from_slice(&rhs);
    m
}

pub struct Spline2 {
    x: Vec<f64>,
    y: Vec<f64>,
    f: Vec<C>,
    fx: Vec<C>,
    fy: Vec<C>,
    fxy: Vec<C>,
}

impl Spline2 {
    /// `values[i * y.len() + j]` is the value at (x[i], y[j]).
    pub fn new(x: Vec<f64>, y: Vec<f64>, values: Vec<C>) -> Spline2 {
        let (nx, ny) = (x.len(), y.len());
        assert_eq!(values.len(), nx * ny);
        let mut fx = vec![C::new(0.0, 0.0); nx * ny];
        let mut fy = vec![C::new(0.0, 0.0); nx * ny];
        let mut fxy = vec![C::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            let col: Vec<C> = (0..nx).map(|i| values[i * ny + j]).collect();
            for (i, s) in spline_slopes(&x, &col).into_iter().enumerate() {
                fx[i * ny + j] = s;
            }
        }
        for i in 0..nx {
            let row = &values[i * ny..(i + 1) * ny];
            fy[i * ny..(i + 1) * ny].copy_from_slice(&spline_slopes(&y, row));
            let rowx = fx[i * ny..(i + 1) * ny].to_vec();
            fxy[i * ny..(i + 1) * ny].copy_from_slice(&spline_slopes(&y, &rowx));
        }
        Spline2 { x, y, f: values, fx, fy, fxy }
    }

    pub(crate) fn cell(grid: &[f64], v: f64) -> usize {
        match grid.binary_search_by(|g| g.partial_cmp(&v).unwrap()) {
            Ok(i) => i.min(grid.len() - 2),
            Err(i) => i.saturating_sub(1).min(grid.len() - 2),
        }
    }

    /// Value at (u, v); points outside the grid are clamped to it.
    pub fn eval(&self, u: f64, v: f64) -> C {
        let ny = self.y.len();
        let u = u.clamp(self.x[0], *self.x.last().unwrap());
        let v = v.clamp(self.y[0], *self.y.last().unwrap());
        let i = Self::cell(&self.x, u);
        let j = Self::cell(&self.y, v);
        let hx = self.x[i + 1] - self.x[i];
        let hy = self.y[j + 1] - self.y[j];
        let s = (u - self.x[i]) / hx;
        let t = (v - self.y[j]) / hy;
        let hs = hermite(s, hx);
        let ht = hermite(t, hy);
        let idx = |a: usize, b: usize| (i + a) * ny + (j + b);
        let mut out = C::new(0.0, 0.0);
        for a in 0..2 {
            for b in 0..2 {
                let k = idx(a, b);
                out += self.f[k] * hs[2 * a] * ht[2 * b]
                    + self.fx[k] * hs[2 * a + 1] * ht[2 * b]
                    + self.fy[k] * hs[2 * a] * ht[2 * b + 1]
                    + self.fxy[k] * hs[2 * a + 1] * ht[2 * b + 1];
            }
        }
        out
    }
}

/// Tricubic Hermite interpolation of `nc` complex components on a rectangular
/// (non-uniform) 3-D grid, with spline slopes and mixed derivatives.
pub struct Spline3 {
    g: [Vec<f64>; 3],
    nc: usize,
    /// per grid point: [f, fx, fy, fz, fxy, fxz, fyz, fxyz] x nc
    d: Vec<C>,
}

impl Spline3 {
    /// `values[((i * ny + j) * nz + l) * nc + c]` is component c at (x[i], y[j], z[l]).
    pub fn new(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>, nc: usize, values: &[C]) -> Spline3 {
        let dims = [x.len(), y.len(), z.len()];
        let np = dims[0] * dims[1] * dims[2];
        assert_eq!(values.len(), np * nc);
        let g = [x, y, z];
        let stride = [dims[1] * dims[2], dims[2], 1];
        // slopes of `f` (one component, flattened) along `axis`
        let along = |f: &[C], axis: usize| -> Vec<C> {
            let mut out = vec![C::new(0.0, 0.0); np];
            let (a1, a2) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let mut line = vec![C::new(0.0, 0.0); dims[axis]];
            for u in 0..dims[a1] {
                for v in 0..dims[a2] {
                    let base = u * stride[a1] + v * stride[a2];
                    for (t, l) in line.iter_mut().enumerate() {
                        *l = f[base + t * stride[axis]];
                    }
                    for (t, s) in spline_slopes(&g[axis], &line).into_iter().enumerate() {
                        out[base + t * stride[axis]] = s;
                    }
                }
            }
            out
        };
        let mut d = vec![C::new(0.0, 0.0); np * 8 * nc];
        for c in 0..nc {
            let f: Vec<C> = (0..np).map(|p| values[p * nc + c]).collect();
            let fx = along(&f, 0);
            let fy = along(&f, 1);
            let fz = along(&f, 2);
            let fxy = along(&fx, 1);
            let fxz = along(&fx, 2);
            let fyz = along(&fy, 2);
            let fxyz = along(&fxy, 2);
            for (k, arr) in [&f, &fx, &fy, &fz, &fxy, &fxz, &fyz, &fxyz].into_iter().enumerate() {
                for p in 0..np {
                    d[(p * 8 + k) * nc + c] = arr[p];
                }
            }
        }
        Spline3 { g, nc, d }
    }

    /// All components at (u, v, w); points outside the grid are clamped to it.
    pub fn eval(&self, u: f64, v: f64, w: f64, out: &mut [C]) {
        let p = [u, v, w];
        let mut cell = [0usize; 3];
        let mut h = [[0.0; 4]; 3];
        for a in 0..3 {
            let g = &self.g[a];
            let q = p[a].clamp(g[0], *g.last().unwrap());
            let i = Spline2::cell(g, q);
            let hx = g[i + 1] - g[i];
            cell[a] = i;
            h[a] = hermite((q - g[i]) / hx, hx);
        }
        let (ny, nz) = (self.g[1].len(), self.g[2].len());
        let nc = self.nc;
        out[..nc].fill(C::new(0.0, 0.0));
        for a in 0..2 {
            for b in 0..2 {
                for e in 0..2 {
                    let pt = ((cell[0] + a) * ny + cell[1] + b) * nz + cell[2] + e;
                    let (x0, x1) = (h[0][2 * a], h[0][2 * a + 1]);
                    let (y0, y1) = (h[1][2 * b], h[1][2 * b + 1]);
                    let (z0, z1) = (h[2][2 * e], h[2][2 * e + 1]);
                    let wts = [x0 * y0 * z0, x1 * y0 * z0, x0 * y1 * z0, x0 * y0 * z1, x1 * y1 * z0, x1 * y0 * z1, x0 * y1 * z1, x1 * y1 * z1];
                    let base = pt * 8 * nc;
                    for (k, &wk) in wts.iter().enumerate() {
                        let row = &self.d[base + k * nc..base + (k + 1) * nc];
                        for c in 0..nc {
                            out[c] += row[c] * wk;
                        }
                    }
                }
            }
        }
    }
}

/// Hermite basis [value@0, slope@0, value@1, slope@1] at s, slopes scaled by h.
#[inline]
fn hermite(s: f64, h: f64) -> [f64; 4] {
    let s2 = s * s;
    let s3 = s2 * s;
    [2.0 * s3 - 3.0 * s2 + 1.0, (s3 - 2.0 * s2 + s) * h, -2.0 * s3 + 3.0 * s2, (s3 - s2) * h]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reproduces_smooth_function() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 / 39.0).powf(1.5) * 3.0).collect();
        let y: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let f = |a: f64, b: f64| C::new((a * 1.3).sin() * (b * 0.7).cos(), a * b * 0.1);
        let vals = x.iter().flat_map(|&a| y.iter().map(move |&b| f(a, b))).collect();
        let s = Spline2::new(x, y, vals);
        for k in 0..200 {
            let a = 0.01 + 2.98 * ((k * 37) % 200) as f64 / 200.0;
            let b = 0.01 + 2.88 * ((k * 53) % 200) as f64 / 200.0;
            assert!((s.eval(a, b) - f(a, b)).norm() < 2e-4, "{a} {b}");
        }
    }

    #[test]
    fn tricubic_reproduces_smooth_function() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64 / 19.0).powf(1.5) * 3.0).collect();
        let y: Vec<f64> = (0..18).map(|i| i as f64 * 0.15).collect();
        let z: Vec<f64> = (0..12).map(|i| 0.1 * 1.25f64.powi(i)).collect();
        let f = |a: f64, b: f64, c: f64| [C::new((a * 1.3).sin() * (b * 0.7).cos(), a * b * 0.1 * c), C::new(c.ln(), (a + c).cos())];
        let mut vals = vec![];
        for &a in &x {
            for &b in &y {
                for &c in &z {
                    vals.extend(f(a, b, c));
                }
            }
        }
        let s = Spline3::new(x, y, z, 2, &vals);
        let mut out = [C::new(0.0, 0.0); 2];
        for k in 0..300 {
            let a = 0.01 + 2.98 * ((k * 37) % 300) as f64 / 300.0;
            let b = 0.01 + 2.5 * ((k * 53) % 300) as f64 / 300.0;
            let c = 0.11 + 1.0 * ((k * 71) % 300) as f64 / 300.0;
            s.eval(a, b, c, &mut out);
            let want = f(a, b, c);
            for i in 0..2 {
                assert!((out[i] - want[i]).norm() < 2e-3, "{a} {b} {c}: {} vs {}", out[i], want[i]);
            }
        }
    }
}
