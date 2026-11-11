//! Dense complex LU solve with partial pivoting; the trailing update runs on rayon.

use num_complex::Complex64 as C;
use rayon::prelude::*;

/// Solve A x = b in place (A row-major n x n, destroyed; b becomes x).
pub fn solve(a: &mut [C], b: &mut [C], n: usize) -> Result<(), String> {
    let lu = Lu::factor(a.to_vec(), n)?;
    lu.solve(b);
    Ok(())
}

/// LU factorisation with partial pivoting, for solving several right-hand sides.
pub struct Lu {
    a: Vec<C>,
    piv: Vec<usize>,
    n: usize,
}

impl Lu {
    /// Factor A (row-major n x n): blocked right-looking LU with partial pivoting
    /// (as LAPACK's getrf).  Panels of `NB` columns are factored, then the trailing
    /// matrix is updated by a cache-tiled product on rayon; that update is nearly
    /// all of the work and runs from cache instead of streaming the matrix through
    /// memory once per column.
    pub fn factor(mut a: Vec<C>, n: usize) -> Result<Lu, String> {
        assert_eq!(a.len(), n * n);
        let mut piv = vec![0; n];
        const NB: usize = 64;
        const JT: usize = 192;
        let mut k0 = 0;
        while k0 < n {
            let kb = NB.min(n - k0);
            let k1 = k0 + kb;
            // ---- panel: columns k0..k1, rows k0..n
            for k in k0..k1 {
                let (p, best) = (k..n).map(|i| (i, a[i * n + k].norm())).fold((k, -1.0), |m, x| if x.1 > m.1 { x } else { m });
                if !(best > 0.0) || !best.is_finite() {
                    return Err("singular impedance matrix (check the geometry for overlapping wires)".into());
                }
                piv[k] = p;
                if p != k {
                    let (lo, hi) = a.split_at_mut(p * n);
                    lo[k * n..(k + 1) * n].swap_with_slice(&mut hi[..n]);
                }
                let (top, rest) = a.split_at_mut((k + 1) * n);
                let prow = &top[k * n..(k + 1) * n];
                let inv = 1.0 / prow[k];
                let upd = |row: &mut [C]| {
                    let f = row[k] * inv;
                    row[k] = f;
                    if f != C::new(0.0, 0.0) {
                        for j in k + 1..k1 {
                            row[j] -= f * prow[j];
                        }
                    }
                };
                if (n - k) * (k1 - k) > 32 * 1024 {
                    rest.par_chunks_mut(n).for_each(upd);
                } else {
                    rest.chunks_mut(n).for_each(upd);
                }
            }
            if k1 == n {
                break;
            }
            // ---- U12 = L11^-1 A12 (rows k0..k1, columns k1..n)
            for k in k0..k1 {
                let (top, rest) = a.split_at_mut((k + 1) * n);
                let prow = &top[k * n + k1..(k + 1) * n];
                for i in k + 1..k1 {
                    let row = &mut rest[(i - k - 1) * n..(i - k) * n];
                    let f = row[k];
                    if f != C::new(0.0, 0.0) {
                        for (x, &u) in row[k1..].iter_mut().zip(prow) {
                            *x -= f * u;
                        }
                    }
                }
            }
            // ---- A22 -= L21 U12, rows k1..n in parallel, columns in tiles of JT
            let (top, rest) = a.split_at_mut(k1 * n);
            let u = &top[k0 * n..];
            rest.par_chunks_mut(n).for_each(|row| {
                let (l, r) = row.split_at_mut(k1);
                let l = &l[k0..k1];
                let mut j0 = 0;
                let w = n - k1;
                while j0 < w {
                    let j1 = (j0 + JT).min(w);
                    let tile = &mut r[j0..j1];
                    let ur = |p: usize| &u[p * n + k1 + j0..p * n + k1 + j1];
                    // four panel columns per pass: a quarter of the tile traffic
                    let mut p = 0;
                    while p + 4 <= l.len() {
                        let (f0, f1, f2, f3) = (l[p], l[p + 1], l[p + 2], l[p + 3]);
                        let (u0, u1, u2, u3) = (ur(p), ur(p + 1), ur(p + 2), ur(p + 3));
                        for (j, x) in tile.iter_mut().enumerate() {
                            *x -= f0 * u0[j] + f1 * u1[j] + f2 * u2[j] + f3 * u3[j];
                        }
                        p += 4;
                    }
                    for q in p..l.len() {
                        let f = l[q];
                        for (x, &v) in tile.iter_mut().zip(ur(q)) {
                            *x -= f * v;
                        }
                    }
                    j0 = j1;
                }
            });
            k0 = k1;
        }
        Ok(Lu { a, piv, n })
    }

    /// Solve in place for one right-hand side.
    pub fn solve(&self, b: &mut [C]) {
        let (a, n) = (&self.a, self.n);
        for k in 0..n {
            b.swap(k, self.piv[k]);
        }
        for i in 1..n {
            let mut s = b[i];
            for j in 0..i {
                s -= a[i * n + j] * b[j];
            }
            b[i] = s;
        }
        for k in (0..n).rev() {
            let mut s = b[k];
            for j in k + 1..n {
                s -= a[k * n + j] * b[j];
            }
            b[k] = s / a[k * n + k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn random(n: usize, seed0: u64) -> (Vec<C>, Vec<C>, Vec<C>) {
        let mut seed = seed0;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a0: Vec<C> = (0..n * n).map(|_| C::new(rnd(), rnd())).collect();
        let x0: Vec<C> = (0..n).map(|_| C::new(rnd(), rnd())).collect();
        let b: Vec<C> = (0..n).map(|i| (0..n).map(|j| a0[i * n + j] * x0[j]).sum()).collect();
        (a0, x0, b)
    }

    #[test]
    fn blocked_factor_solves_sizes_around_the_block() {
        for n in [1, 2, 63, 64, 65, 129, 200, 517] {
            let (a0, x0, mut b) = random(n, n as u64 + 7);
            let lu = Lu::factor(a0, n).unwrap();
            lu.solve(&mut b);
            let err = b.iter().zip(&x0).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-8, "n {n}: {err}");
        }
    }

    #[test]
    fn solves_random_system() {
        let n = 150;
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a0: Vec<C> = (0..n * n).map(|_| C::new(rnd(), rnd())).collect();
        let x0: Vec<C> = (0..n).map(|_| C::new(rnd(), rnd())).collect();
        let mut b: Vec<C> = (0..n).map(|i| (0..n).map(|j| a0[i * n + j] * x0[j]).sum()).collect();
        let mut a = a0.clone();
        solve(&mut a, &mut b, n).unwrap();
        for i in 0..n {
            assert!((b[i] - x0[i]).norm() < 1e-9);
        }
    }
}
