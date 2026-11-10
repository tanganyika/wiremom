//! Gauss-Legendre quadrature.

use std::sync::OnceLock;

/// Gauss-Legendre nodes and weights on [0, 1], ascending (n is clamped to 1..=128).
pub fn gauss01(n: usize) -> &'static (Vec<f64>, Vec<f64>) {
    static CACHE: OnceLock<Vec<(Vec<f64>, Vec<f64>)>> = OnceLock::new();
    let table = CACHE.get_or_init(|| (0..=128).map(compute).collect());
    &table[n.clamp(1, 128)]
}

fn compute(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        // Newton iteration on P_n from the usual cosine starting guess
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for j in 2..=n {
                let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
                p0 = p1;
                p1 = p2;
            }
            let pn = if n == 1 { z } else { p1 };
            let pm = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (z * pn - pm) / (z * z - 1.0);
            let dz = pn / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x[i] = 0.5 * (1.0 - z); // ascending on [0, 1]
        w[i] = 1.0 / ((1.0 - z * z) * dp * dp); // = (2 / ((1-z^2) P'^2)) / 2
    }
    (x, w)
}

/// Composite n_pts-point Gauss rule on [a, b] with n_sub subintervals.  With `first`
/// the subintervals grow geometrically from width ~`first` (integrands that decay away
/// from `a`).  Appends (x, w) to `out`.
pub fn composite(a: f64, b: f64, n_sub: usize, n_pts: usize, first: Option<f64>, out: &mut Vec<(f64, f64)>) {
    let (x, w) = gauss01(n_pts);
    let n_sub = n_sub.max(1);
    let len = b - a;
    let mut edges = Vec::with_capacity(n_sub + 1);
    match first {
        Some(f) if f < len / n_sub as f64 && f > 0.0 && n_sub > 1 => {
            edges.push(a);
            let r = (len / f).ln() / (n_sub - 1) as f64;
            for i in 0..n_sub {
                edges.push(a + f * (r * i as f64).exp());
            }
            *edges.last_mut().unwrap() = b;
        }
        _ => {
            for i in 0..=n_sub {
                edges.push(a + len * i as f64 / n_sub as f64);
            }
        }
    }
    for e in edges.windows(2) {
        let l = e[1] - e[0];
        for (xi, wi) in x.iter().zip(w) {
            out.push((e[0] + l * xi, l * wi));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integrates_polynomials_exactly() {
        for n in 1..12 {
            let (x, w) = gauss01(n);
            for p in 0..(2 * n) {
                let s: f64 = x.iter().zip(w).map(|(x, w)| w * x.powi(p as i32)).sum();
                assert!((s - 1.0 / (p as f64 + 1.0)).abs() < 1e-13, "n={n} p={p} s={s}");
            }
        }
    }
}
