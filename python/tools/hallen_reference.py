"""Reference values for tests/thick.rs ("hallen" in tests/data/thick_dipole.json).

Hallen's equation with the exact (tube) kernel, delta gap,
piecewise-linear current, point matching at the nodes.  Separate from wiremom's
MPIE/Galerkin code: different equation, basis, testing and singular treatment
(scipy quad with breakpoints on the full kernel via ellipkm1)."""
import sys
import numpy as np
from scipy.special import ellipkm1
from scipy.integrate import quad

ETA = 376.730313668
gx, gw = np.polynomial.legendre.leggauss(24)
phi = 0.5 * np.pi * (gx + 1.0)
wphi = 0.5 * np.pi * gw


def kern(z, a, k):
    """(1/2pi) int exp(-jkR)/(4 pi R) dphi, R^2 = z^2 + 4a^2 sin^2(phi/2)."""
    if z == 0.0:
        return 0.0
    z2 = z * z
    d = z2 + 4 * a * a
    stat = 2 / np.pi * ellipkm1(z2 / d) / np.sqrt(d)
    r = np.sqrt(z2 + 4 * a * a * np.sin(phi / 2) ** 2)
    dyn = np.sum(wphi * np.expm1(-1j * k * r) / r) / np.pi
    return (stat + dyn) / (4 * np.pi)


def cquad(f, lo, hi, pts):
    pts = [p for p in pts if lo < p < hi]
    o = dict(points=pts or None, limit=200, epsabs=0, epsrel=1e-10)
    re = quad(lambda x: f(x).real, lo, hi, **o)[0]
    im = quad(lambda x: f(x).imag, lo, hi, **o)[0]
    return re + 1j * im


def admittance(kh, ka, M):
    k = 1.0
    h, a = kh, ka
    dz = h / M
    nodes = np.arange(M + 1) * dz
    # basis j = 0..M-1: triangle at node j (j=0 is the symmetric centre triangle)
    A = np.zeros((M + 1, M + 1), complex)
    for i, z in enumerate(nodes):
        for j in range(M):
            zj = nodes[j]
            lo, hi = max(zj - dz, 0.0), zj + dz
            tri = lambda zp: 1 - abs(zp - zj) / dz
            # current on z' in [0,h] and its mirror -z'
            f = lambda zp: tri(zp) * (kern(z - zp, a, k) + kern(z + zp, a, k))
            A[i, j] = cquad(f, lo, hi, [z, zj])
        A[i, M] = 1j / ETA * np.cos(k * z)
    rhs = -1j / ETA * 0.5 * np.sin(k * nodes)  # V = 1
    x = np.linalg.solve(A, rhs)
    return x[0]


if __name__ == "__main__":
    # python3 hallen_reference.py KA KH,KH,... DA,DA,...  -> JSON lines
    import json
    ka = float(sys.argv[1])
    for kh in [float(v) for v in sys.argv[2].split(",")]:
        for da in [float(v) for v in sys.argv[3].split(",")]:
            M = int(round(kh / (da * ka)))
            y = admittance(kh, ka, M) * 1e3
            print(json.dumps({"ka": ka, "kh": kh, "da": kh / (M * ka), "M": M, "G": round(y.real, 5), "B": round(y.imag, 5)}), flush=True)
