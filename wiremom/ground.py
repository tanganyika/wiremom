"""Lossy (real) ground: the Sommerfeld half-space solution.

Field reflected by the ground
-----------------------------
A unit current element along c at r' radiates plane waves.  Each down-going
wave hits the ground and reflects with the Fresnel coefficient of its TE or TM
part.  Summing the reflected waves over the spectrum and integrating out the
azimuth gives the reflected field E(r) = D(rho, h) . c with h = z + z' and
rho the horizontal offset, in local axes (rho_hat, phi_hat, z_hat):

    D_rr = P int W [ gTE (J0 + J2) - gTM (kz^2/k^2) (J0 - J2) ]
    D_ff = P int W [ gTE (J0 - J2) - gTM (kz^2/k^2) (J0 + J2) ]
    D_rz = P int W [ 2j gTM (kz krho / k^2) J1 ]           (E_rho from a z source)
    D_zr = -D_rz
    D_zz = P int W [ 2 gTM (krho^2 / k^2) J0 ]

with P = -w mu / (8 pi), W = (krho / kz) exp(-j kz h), Bessel functions of
krho*rho, kz = sqrt(k^2 - krho^2) (Im kz <= 0), and coefficients

    gTE = (kz - kz2) / (kz + kz2)               (-> -1 for a perfect conductor)
    gTM = (eps kz - kz2) / (eps kz + kz2)       (-> +1 for a perfect conductor)

where kz2 = sqrt(eps k^2 - krho^2) and eps the complex relative permittivity.
With (gTE, gTM) = (-1, +1) these integrals are exactly the PEC image field
(image current -P c at the mirror point, P = diag(1, 1, -1)); the tests check
that against the closed form to ~1e-12.

Singularity subtraction (as in NEC-2)
-------------------------------------
For large krho, gTM -> Gamma0 with Gamma0 = (eps - 1)/(eps + 1), which makes
the reflected field near the ground behave like a Gamma0-scaled image (1/R^3).
That part is taken out analytically: Gamma0 times the PEC image, which the
solver's image integrals already handle.  The remainder uses coefficients
(gTE + Gamma0, gTM - Gamma0), is only ~1/R singular, and is tabulated as
R exp(jkR) D on an (R, theta) grid and interpolated, like NEC-2's SOMNEC.

Spectral path: two raised arcs in the first quadrant, one around the branch
point k and the surface-wave pole, one out past Re(k2) (the ground's branch
point, far out for sea water), then for rho > h the Bessel tail is split into
Hankel functions on vertical paths (exponential decay), otherwise the real axis
(the exp(-krho h) factor decays).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import hankel1, hankel2, jv

from .integrals import gauss01

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
COMPONENTS = ("rr", "ff", "rz", "zr", "zz")


@dataclass(frozen=True)
class RealGround:
    """Homogeneous lossy ground below z = 0."""

    eps_r: float
    sigma: float  # S/m

    def eps_c(self, freq: float) -> complex:
        return complex(self.eps_r, -self.sigma / (2 * np.pi * freq * EPS0))

    def gamma0(self, freq: float) -> complex:
        e = self.eps_c(freq)
        return (e - 1) / (e + 1)

    def reflection(self, freq: float, theta):
        """Plane-wave coefficients (gTE, gTM) for a wave arriving at angle theta from
        zenith, in this module's convention ((-1, +1) for a perfect conductor).
        The reflected far field is gTM * E_theta and -gTE * E_phi of the PEC image."""
        k = 2 * np.pi * freq / 299_792_458.0
        return coefficients(k * np.sin(theta) + 0j, k, self.eps_c(freq))


PRESETS = {
    # commonly used values (e.g. in amateur antenna software) at HF
    "very good": RealGround(20.0, 0.0303),
    "good": RealGround(20.0, 0.0303),
    "average": RealGround(13.0, 0.005),
    "poor": RealGround(5.0, 0.001),
    "very poor": RealGround(3.0, 0.001),  # city / industrial
    "sea water": RealGround(81.0, 5.0),
    "fresh water": RealGround(80.0, 0.001),
}


def make_ground(spec):
    if spec is None or spec == "pec" or isinstance(spec, RealGround):
        return spec
    if isinstance(spec, str) and spec in PRESETS:
        return PRESETS[spec]
    if isinstance(spec, tuple) and len(spec) == 2:
        return RealGround(float(spec[0]), float(spec[1]))
    raise ValueError(f"unknown ground {spec!r}: use None, 'pec', a preset "
                     f"({', '.join(PRESETS)}), (eps_r, sigma) or RealGround")


# --------------------------------------------------------------------- spectra
def _kz(kr, k2):
    # Im(kz) <= 0 on the paths used (first quadrant / just above the real axis)
    return -1j * np.sqrt(kr * kr - k2)


def coefficients(kr, k, eps):
    kz = _kz(kr, k * k)
    kz2 = _kz(kr, eps * k * k)
    g_te = (kz - kz2) / (kz + kz2)
    g_tm = (eps * kz - kz2) / (eps * kz + kz2)
    return g_te, g_tm


def _integrand(kr, rho, h, k, eps, kind, shift, fixed=None):
    """Spectral integrand for all five components, shape (5, len(kr)).

    ``kind`` selects the Bessel family ('J', 'H1', 'H2'); ``shift`` is added to
    both coefficients (Gamma0 for the remainder, 0 for the full field)."""
    kz = _kz(kr, k * k)
    kz2 = _kz(kr, eps * k * k)
    if fixed is None:
        c_te = (kz - kz2) / (kz + kz2) + shift[0]
        c_tm = (eps * kz - kz2) / (eps * kz + kz2) + shift[1]
    else:  # constant coefficients (testing: (-1, +1) is the PEC image)
        c_te, c_tm = fixed
    x = kr * rho
    f = {"J": jv, "H1": hankel1, "H2": hankel2}[kind]
    b0, b1 = f(0, x), f(1, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2 = np.where(np.abs(x) > 1e-12, 2 * b1 / x - b0, 0.0)  # recurrence (J and H)
    w = kr / kz * np.exp(-1j * kz * h)
    k2 = k * k
    return w * np.array([
        c_te * (b0 + b2) - c_tm * (kz * kz / k2) * (b0 - b2),
        c_te * (b0 - b2) - c_tm * (kz * kz / k2) * (b0 + b2),
        2j * c_tm * (kz * kr / k2) * b1,
        -2j * c_tm * (kz * kr / k2) * b1,
        2 * c_tm * (kr * kr / k2) * b0,
    ])


def _composite(a, b, n_sub, n_pts, first=None):
    """Composite Gauss rule on [a, b]; with ``first`` the subintervals grow
    geometrically from width ~``first`` (for integrands that decay away from a)."""
    x, w = gauss01(n_pts)
    if first is None or first >= (b - a) / n_sub:
        edges = np.linspace(a, b, n_sub + 1)
    else:
        edges = np.concatenate([[a], a + np.geomspace(first, b - a, n_sub)])
    L = np.diff(edges)
    return (edges[:-1, None] + L[:, None] * x).ravel(), (L[:, None] * w).ravel()


def _nodes(rho, h, k, k2r):
    """Spectral quadrature for one (rho, h): {kind: (krho nodes, weights)}, weights
    including the path Jacobian and the 1/2 of the Hankel split."""
    a1 = 2.0 * k                                    # first arc: around k and the pole
    a = max(a1, 1.2 * k2r + 0.5 * k)                # second arc: out past Re(k2)
    b = min(0.6 * k, 1.0 / max(rho, 1e-30))         # arc height (keeps J bounded)
    density = max(2 * rho / np.pi, 4.0 / k)         # subintervals per unit krho
    kr_j, w_j = [], []
    for lo, hi in ((0.0, a1), (a1, a)):
        if hi <= lo:
            continue
        x, w = _composite(lo, hi, 8 + int((hi - lo) * density), 8)
        s = np.pi * (x - lo) / (hi - lo)
        kr_j.append(x + 1j * b * np.sin(s))
        w_j.append(w * (1 + 1j * b * np.pi / (hi - lo) * np.cos(s)))
    out = {}
    if rho > h:
        # J = (H1 + H2)/2 on vertical paths a +- j t, decaying like exp(-t rho)
        t, wt = _composite(0.0, 40.0 / rho, 32, 8, first=0.25 * k)
        out["H1"] = (a + 1j * t, 0.5j * wt)
        out["H2"] = (a - 1j * t, -0.5j * wt)
    else:
        x, w = _composite(a, a + 40.0 / h, 32, 8, first=0.25 * k)
        kr_j.append(x + 0j)
        w_j.append(w + 0j)
    out["J"] = (np.concatenate(kr_j), np.concatenate(w_j))
    return out


def reflected_dyadic_many(rho, h, freq, eps, shift=(0.0, 0.0), fixed=None, threads=None):
    """Reflected-field components (5, n) at many (rho, h) points at once: every
    point's spectral nodes are gathered and evaluated in large vectorised chunks,
    spread over threads (the Bessel ufuncs release the GIL)."""
    rho = np.atleast_1d(np.asarray(rho, float))
    h = np.atleast_1d(np.asarray(h, float))
    omega = 2 * np.pi * freq
    k = omega / 299_792_458.0
    pref = -omega * MU0 / (8 * np.pi)
    k2r = 0.0 if fixed is not None else (np.sqrt(eps) * k).real
    per_kind = {"J": ([], [], []), "H1": ([], [], []), "H2": ([], [], [])}
    for i, (r, z) in enumerate(zip(rho, h)):
        for kind, (kr, w) in _nodes(r, z, k, k2r).items():
            lst = per_kind[kind]
            lst[0].append(kr)
            lst[1].append(w)
            lst[2].append(np.full(len(kr), i))
    total = np.zeros((5, len(rho)), complex)
    jobs = []
    chunk = 1 << 15
    for kind, (krs, ws, idx) in per_kind.items():
        if not krs:
            continue
        kr, w, ix = np.concatenate(krs), np.concatenate(ws), np.concatenate(idx)
        for lo in range(0, len(kr), chunk):
            sl = slice(lo, lo + chunk)
            jobs.append((kind, kr[sl], w[sl], ix[sl]))

    def run(job):
        kind, kr, w, ix = job
        vals = _integrand(kr, rho[ix], h[ix], k, eps, kind, shift, fixed) * w
        n = len(rho)
        return np.array([np.bincount(ix, v.real, n) + 1j * np.bincount(ix, v.imag, n) for v in vals])

    workers = threads or min(len(jobs), os.cpu_count() or 1)
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for part in ex.map(run, jobs):
                total += part
    else:
        for job in jobs:
            total += run(job)
    return pref * total


def reflected_dyadic(rho, h, freq, eps, shift=(0.0, 0.0), fixed=None):
    """D components (rr, ff, rz, zr, zz) of the reflected field at one (rho, h)."""
    return reflected_dyadic_many([rho], [h], freq, eps, shift, fixed, threads=1)[:, 0]


class SommerfeldTable:
    """Interpolation table of the remainder field, in normalised form.

    For a fixed ground permittivity the reflected field scales exactly as
    D(rho, h) = w k F(k rho, k h), so the table stores w^-1 k^-1 R exp(jkR) D on a
    grid of (kR, theta), theta = atan2(h, rho) measured up from the ground plane.
    That lets one table serve nearby frequencies of a sweep: only the ground's
    eps(f) is frozen at the build frequency."""

    def __init__(self, ground: RealGround, freq: float, r_min: float, r_max: float,
                 n_theta: int = 30, fixed=None, margin: float = 1.0):
        self.ground = ground
        self.freq = freq
        self.k = 2 * np.pi * freq / 299_792_458.0
        self.eps = ground.eps_c(freq)
        self.gamma0 = ground.gamma0(freq)
        self.fixed = fixed
        lam = 2 * np.pi / self.k
        r_min = max(r_min / margin, 1e-6 * lam)
        r_max = max(r_max * margin, 2 * r_min)
        # geometric near the ground-image singularity, then ~lam/30 spacing, finer for
        # low-loss high-permittivity ground (water), whose lateral wave oscillates at the
        # ground wavelength without decaying much
        n2 = np.sqrt(self.eps)
        fine = np.clip(n2.real / 4, 1.0, 4.0) if abs(n2.imag) < n2.real / 4 else 1.0
        step = lam / (30 * fine)
        # geometric out to lam/4: near the ground the remainder varies on the scale of
        # the distance itself (and of the ground wavelength, tiny for sea water)
        near_end = min(max(lam / 4, step), r_max)
        near = np.geomspace(r_min, near_end, 24) if r_min < near_end else np.array([r_min])
        far = np.arange(near[-1] + step, r_max + step, step)
        R = np.concatenate([near, far])
        if len(R) < 4:
            R = np.linspace(r_min, r_max, 4)
        self.X = self.k * R
        # clustered toward grazing (theta = 0), where the field changes fastest
        self.T = (np.pi / 2) * np.linspace(0.0, 1.0, n_theta) ** 1.7
        # remainder after the Gamma0-scaled image; ``fixed`` tabulates constant
        # coefficients instead (testing: (-1, +1) is the whole PEC image)
        shift = (0.0, 0.0) if fixed is not None else (self.gamma0, -self.gamma0)
        RR, TT = np.meshgrid(R, self.T, indexing="ij")
        rho = (RR * np.cos(TT)).ravel()
        h = np.maximum((RR * np.sin(TT)).ravel(), 1e-9 * lam)
        d = reflected_dyadic_many(rho, h, freq, self.eps, shift, fixed)
        omega = 2 * np.pi * freq
        norm = (RR * np.exp(1j * self.k * RR)).ravel() / (omega * self.k)
        vals = (d * norm).reshape(5, *RR.shape)
        self._splines = [(RectBivariateSpline(self.X, self.T, v.real),
                          RectBivariateSpline(self.X, self.T, v.imag)) for v in vals]

    def covers(self, ground, freq: float, r_min: float, r_max: float, tol: float) -> bool:
        k = 2 * np.pi * freq / 299_792_458.0
        return (ground == self.ground and self.fixed is None
                and abs(freq / self.freq - 1) <= tol
                and k * r_min >= self.X[0] * (1 - 1e-9) and k * r_max <= self.X[-1])

    def __call__(self, rho, h, freq: float | None = None):
        """Remainder D components at arrays of (rho, h): shape (5, ...)."""
        freq = self.freq if freq is None else freq
        omega = 2 * np.pi * freq
        k = omega / 299_792_458.0
        R = np.sqrt(rho * rho + h * h)
        th = np.arctan2(h, rho)
        Xc = np.clip(k * R, self.X[0], self.X[-1])
        out = np.empty((5,) + np.shape(R), complex)
        back = omega * k * np.exp(-1j * k * R) / R
        for c, (sr, si) in enumerate(self._splines):
            out[c] = (sr.ev(Xc, th) + 1j * si.ev(Xc, th)) * back
        return out


def quad_points(p0, t, length, n_q):
    x, w = gauss01(n_q)
    pts = p0[:, None, :] + (x[None, :, None] * length[:, None, None]) * t[:, None, :]
    return x, w, pts


def table_range(p0, t, length, n_q):
    """(R_min, R_max) between every quadrature point and every image point."""
    _, _, pts = quad_points(p0, t, length, n_q)
    P = pts.reshape(-1, 3)
    zmin = P[:, 2].min()
    if zmin <= 0:
        raise ValueError("over real ground all wires must be above z = 0")
    span = np.ptp(P, axis=0)
    r_max = np.sqrt(span[0] ** 2 + span[1] ** 2 + (2 * P[:, 2].max()) ** 2)
    return 2 * zmin, r_max


def segment_integrals(table, p0, t, length, n_q: int = 6, chunk: int = 64, freq=None):
    """Galerkin integrals of the tabulated reflected field between segment pairs:

        K[p, q, a, b] = int_p int_q shape_a(u) shape_b(u') t_p . D(r, r') . t_q du du'

    The kernel is reciprocal (K[q, p, b, a] = K[p, q, a, b]), so only source
    segments q >= the current block are evaluated and the rest is mirrored."""
    N = len(length)
    x, w, pts = quad_points(p0, t, length, n_q)
    shapes = np.stack([x, 1 - x])                       # (2, q): rise, fall
    ws = shapes * w                                     # quadrature-weighted shapes
    K = np.zeros((N, N, 2, 2), complex)
    for p_lo in range(0, N, chunk):
        p_hi = min(N, p_lo + chunk)
        # mirror the part already computed by earlier blocks
        K[p_lo:p_hi, :p_lo] = K[:p_lo, p_lo:p_hi].transpose(1, 0, 3, 2)
        src = pts[p_lo:].reshape(-1, 3)                 # sources q >= p_lo
        tq = np.repeat(t[p_lo:], n_q, axis=0)
        Nq = N - p_lo
        obs = pts[p_lo:p_hi].reshape(-1, 3)             # (n q, 3)
        tp = np.repeat(t[p_lo:p_hi], n_q, axis=0)
        dx = obs[:, None, 0] - src[None, :, 0]
        dy = obs[:, None, 1] - src[None, :, 1]
        h = obs[:, None, 2] + src[None, :, 2]
        rho = np.hypot(dx, dy)
        safe = rho > 1e-12 * (rho.max() + 1.0)
        cx = np.where(safe, dx / np.where(safe, rho, 1.0), 1.0)
        cy = np.where(safe, dy / np.where(safe, rho, 1.0), 0.0)
        D = table(rho, h, freq)                         # (5, n q, Nq q)
        tp_r = tp[:, None, 0] * cx + tp[:, None, 1] * cy
        tp_f = -tp[:, None, 0] * cy + tp[:, None, 1] * cx
        tq_r = tq[None, :, 0] * cx + tq[None, :, 1] * cy
        tq_f = -tq[None, :, 0] * cy + tq[None, :, 1] * cx
        tp_z, tq_z = tp[:, None, 2], tq[None, :, 2]
        kern = (D[0] * tp_r * tq_r + D[1] * tp_f * tq_f + D[2] * tp_r * tq_z
                + D[3] * tp_z * tq_r + D[4] * tp_z * tq_z)
        kern = kern.reshape(p_hi - p_lo, n_q, Nq, n_q)
        Kc = np.einsum("aj,pjqk,bk->pqab", ws, kern, ws)
        K[p_lo:p_hi, p_lo:] = (Kc * length[p_lo:p_hi, None, None, None]
                               * length[None, p_lo:, None, None])
    return K
