"""Segment-pair integrals of the free-space Green's function.

We use the *reduced* thin-wire kernel: the source current sits on the wire
axis, the field is tested on the axis too, and the distance is
``R = sqrt(|r - r'|^2 + a^2)`` with ``a`` the source wire radius.  Then

    G(R) = exp(-jkR) / (4 pi R) = 1 / (4 pi R)  +  (exp(-jkR) - 1) / (4 pi R)
                                  ^ singular      ^ smooth (-> -jk/4pi)

For segments of different radius ``a^2`` is the mean of the two radii squared,
so the matrix stays symmetric.  The 1/R part is integrated *analytically* over the source segment (for a
constant or linear current), the smooth part with Gauss-Legendre quadrature.
The outer (test) integral is always Gauss-Legendre.

For every test segment p and source segment q the function returns

    S[p, q]          = int_p int_q G du du'
    T[p, q, a, b]    = int_p int_q shape_a(u) shape_b(u') G du du'

with shapes 0 = rising (u/L) and 1 = falling (1 - u/L).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _gauss01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    x.flags.writeable = False
    w.flags.writeable = False
    return x, w


def gauss01(n: int):
    """Gauss-Legendre nodes and weights on [0, 1] (cached, read-only)."""
    return _gauss01(int(n))


def segment_integrals(p0, t, length, radius, k: float, n_outer: int = 8, n_inner: int = 8,
                      src_p0=None, src_t=None):
    """Integrals with test segments (p0, t, length) and source segments of the same
    lengths/radii but optionally different position/direction (``src_p0``, ``src_t``),
    which is how ground-plane images are handled."""
    N = len(length)
    xo, wo = gauss01(n_outer)
    xi, wi = gauss01(n_inner)
    q0 = p0 if src_p0 is None else src_p0
    tq = t if src_t is None else src_t

    # source quadrature points: (N, n_inner, 3)
    src = q0[:, None, :] + (xi[None, :, None] * length[:, None, None]) * tq[:, None, :]
    # Reduced-kernel radius.  On a segment's own wire this is just its radius; between
    # wires of different radius use the mean of a^2, which keeps the operator symmetric
    # (reciprocal) instead of favouring whichever segment is the source.
    a2 = 0.5 * (radius[:, None] ** 2 + radius[None, :] ** 2)  # (N_p, N_q)

    S = np.zeros((N, N), complex)
    T = np.zeros((N, N, 2, 2), complex)
    four_pi = 4.0 * np.pi

    for i in range(n_outer):
        # observation point i on every test segment: (N, 3)
        r = p0 + (xo[i] * length)[:, None] * t
        rise_o, fall_o = xo[i], 1.0 - xo[i]

        # ---- analytic 1/R part over each source segment (line from p0_q along t_q)
        rel = r[:, None, :] - q0[None, :, :]                   # (N_p, N_q, 3)
        z0 = np.einsum("pqk,qk->pq", rel, tq)                   # axial offset
        d2 = np.einsum("pqk,pqk->pq", rel, rel) - z0 ** 2 + a2
        d = np.sqrt(np.maximum(d2, 1e-300))
        L = length[None, :]
        I0 = np.arcsinh((L - z0) / d) + np.arcsinh(z0 / d)      # int 1/R du'
        I1 = np.sqrt((L - z0) ** 2 + d2) - np.sqrt(z0 ** 2 + d2) + z0 * I0  # int u'/R du'
        s0 = I0 / four_pi
        s1 = I1 / (L * four_pi)                                 # rising shape u'/L

        # ---- smooth part by quadrature
        diff = r[:, None, None, :] - src[None, :, :, :]          # (N_p, N_q, n_inner, 3)
        R = np.sqrt(np.einsum("pqjk,pqjk->pqj", diff, diff) + a2[:, :, None])
        f = np.expm1(-1j * k * R) / (four_pi * R)
        s0 = s0 + L * np.einsum("pqj,j->pq", f, wi)
        s1 = s1 + L * np.einsum("pqj,j->pq", f, wi * xi)

        src_rise, src_fall = s1, s0 - s1
        w = wo[i] * length[:, None]                            # outer weight * L_p
        S += w * s0
        T[:, :, 0, 0] += w * rise_o * src_rise
        T[:, :, 0, 1] += w * rise_o * src_fall
        T[:, :, 1, 0] += w * fall_o * src_rise
        T[:, :, 1, 1] += w * fall_o * src_fall
    return S, T
