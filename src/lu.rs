//! Dense complex LU solve with partial pivoting; the trailing update runs on rayon.

use num_complex::Complex64 as C;
use rayon::prelude::*;

/// Solve A x = b in place (A row-major n x n, destroyed; b becomes x).
pub fn solve(a: &mut [C], b: &mut [C], n: usize) -> Result<(), String> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n);
    for k in 0..n {
        // pivot
        let (p, best) = (k..n).map(|i| (i, a[i * n + k].norm())).fold((k, -1.0), |m, x| if x.1 > m.1 { x } else { m });
        if !(best > 0.0) || !best.is_finite() {
            return Err("singular impedance matrix (check the geometry for overlapping wires)".into());
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            b.swap(k, p);
        }
        let (top, rest) = a.split_at_mut((k + 1) * n);
        let pivot_row = &top[k * n..(k + 1) * n];
        let inv = 1.0 / pivot_row[k];
        let bk = b[k];
        let (_, b_rest) = b.split_at_mut(k + 1);
        let update = |(row, bi): (&mut [C], &mut C)| {
            let f = row[k] * inv;
            if f != C::new(0.0, 0.0) {
                row[k] = f;
                for j in k + 1..n {
                    row[j] -= f * pivot_row[j];
                }
                *bi -= f * bk;
            }
        };
        if (n - k) * (n - k) > 64 * 64 {
            rest.par_chunks_mut(n).zip(b_rest.par_iter_mut()).for_each(update);
        } else {
            rest.chunks_mut(n).zip(b_rest.iter_mut()).for_each(update);
        }
    }
    for k in (0..n).rev() {
        let mut s = b[k];
        for j in k + 1..n {
            s -= a[k * n + j] * b[j];
        }
        b[k] = s / a[k * n + k];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
