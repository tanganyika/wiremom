"""Reference values for tests/thick.rs ("stepped", "caps", "parasitic" in
tests/data/thick_dipole.json).

Independent body-of-revolution MoM (m = 0, MPIE, Galerkin, triangle basis on the
generating curve) for a centre-fed cylinder, open tube or solid rod with flat end
caps.  Ring-to-ring kernels by elliptic integrals (static) and Gauss in phi
(dynamic); log singularities by graded quadrature.  Units: lambda = 1."""
import sys
import numpy as np
from scipy.special import ellipkm1, ellipe

K0 = 2 * np.pi
MU0 = 4e-7 * np.pi
C0 = 299792458.0
EPS0 = 1 / (MU0 * C0 * C0)
OMEGA = K0 * C0
gxp, gwp = np.polynomial.legendre.leggauss(32)
PHI = 0.5 * np.pi * (gxp + 1.0)
WPHI = 0.5 * np.pi * gwp
COSP = np.cos(PHI)
SIN2 = np.sin(PHI / 2) ** 2


def g01(rho, rhop, dz):
    """(1/2pi) int G dphi', (1/2pi) int G cos(phi') dphi', G = exp(-jkR)/(4 pi R)."""
    D = (rho + rhop) ** 2 + dz ** 2
    p = ((rho - rhop) ** 2 + dz ** 2) / D  # 1 - m
    m = 1 - p
    sq = np.sqrt(D)
    K = ellipkm1(p)
    E = ellipe(m)
    s0 = 2 / np.pi * K / sq
    small = m < 1e-4
    mm = np.where(small, 1.0, m)
    s1 = np.where(small, m / (8 * sq), 2 / np.pi * ((2 - m) * K - 2 * E) / (mm * sq))
    R = np.sqrt((rho - rhop)[..., None] ** 2 + dz[..., None] ** 2 + 4 * (rho * rhop)[..., None] * SIN2)
    f = np.expm1(-1j * K0 * R) / R
    d0 = (f * WPHI).sum(-1) / np.pi
    d1 = (f * (WPHI * COSP)).sum(-1) / np.pi
    return (s0 + d0) / (4 * np.pi), (s1 + d1) / (4 * np.pi)


def graded(n_sub=8, first=1e-6):
    """points on [0,1] graded toward 0 (geometric), 8-point Gauss each."""
    gx, gw = np.polynomial.legendre.leggauss(8)
    gx, gw = 0.5 * (gx + 1), 0.5 * gw
    e = np.concatenate([[0.0], first * np.exp(np.log(1 / first) * np.arange(n_sub) / (n_sub - 1))])
    e[-1] = 1.0
    xs, ws = [], []
    for a, b in zip(e[:-1], e[1:]):
        xs.append(a + (b - a) * gx)
        ws.append((b - a) * gw)
    return np.concatenate(xs), np.concatenate(ws)


GX, GW = graded()
SX, SW = np.polynomial.legendre.leggauss(10)
SX, SW = 0.5 * (SX + 1), 0.5 * SW


def geometry(h, a, da, caps, ncap):
    """panels along the generating curve: list of (p0, p1)."""
    pts = []
    if caps:
        for i in range(ncap + 1):
            pts.append((a * i / ncap, -h))
    nside = int(round(2 * h / (da * a)))
    nside += nside % 2
    for i in range(1, nside + 1) if caps else range(nside + 1):
        pts.append((a, -h + 2 * h * i / nside))
    if caps:
        for i in range(ncap - 1, -1, -1):
            pts.append((a * i / ncap, h))
    pts = np.array(pts)
    gap = int(np.argmin(np.abs(pts[:, 1]) + 10 * (pts[:, 0] < a * 0.999)))
    return pts, gap


def profile_points(sections, da, caps, ncap=6, nstep=4):
    """generating curve for a stepped body: sections [(z0, z1, radius)] bottom to
    top; side panels ~da * radius long; radial panels at steps and caps."""
    pts = []
    z0, _, a0 = sections[0]
    if caps:
        pts += [(a0 * i / ncap, z0) for i in range(ncap)]
    pts.append((a0, z0))
    for i, (za, zb, a) in enumerate(sections):
        if i > 0 and sections[i - 1][2] != a:
            ap = sections[i - 1][2]
            pts += [(ap + (a - ap) * j / nstep, za) for j in range(1, nstep + 1)]
        n = max(1, int(round((zb - za) / (da * a))))
        pts += [(a, za + (zb - za) * j / n) for j in range(1, n + 1)]
    zl, al = sections[-1][1], sections[-1][2]
    if caps:
        pts += [(al * i / ncap, zl) for i in range(ncap - 1, -1, -1)]
    pts = np.array(pts)
    on_side = np.array([any(abs(p[0] - s[2]) < 1e-12 for s in sections) for p in pts])
    gap = int(np.argmin(np.abs(pts[:, 1]) + 10 * (~on_side)))
    assert abs(pts[gap, 1]) < 1e-9, "no node at z = 0"
    return pts, gap


def solve(kh, ka, da, caps, ncap=6, plane=False, sections=None):
    h, a = kh / K0, ka / K0
    if sections is None:
        pts, gap = geometry(h, a, da, caps, ncap)
    else:
        pts, gap = profile_points(sections, da, caps, ncap)
    P0, P1 = pts[:-1], pts[1:]
    L = np.linalg.norm(P1 - P0, axis=1)
    T = (P1 - P0) / L[:, None]
    npnl = len(L)
    # test points
    tv = np.zeros((npnl, npnl, 2, 2), complex)
    sv = np.zeros((npnl, npnl), complex)
    for p in range(npnl):
        for xo, wo in zip(SX, SW):
            r = P0[p] + xo * L[p] * T[p]
            wl = wo * L[p]
            for q in range(npnl):
                # inner points on q, graded toward the foot of r if near
                foot = np.clip(np.dot(r - P0[q], T[q]), 0, L[q])
                dist = np.linalg.norm(P0[q] + foot * T[q] - r)
                if dist > 2 * L[q]:
                    u, w = SX * L[q], SW * L[q]
                else:
                    us, ws = [], []
                    if foot > 0:
                        us.append(foot - GX * foot); ws.append(GW * foot)
                    if foot < L[q]:
                        us.append(foot + GX * (L[q] - foot)); ws.append(GW * (L[q] - foot))
                    u, w = np.concatenate(us), np.concatenate(ws)
                qp = P0[q][None, :] + u[:, None] * T[q][None, :]
                G0, G1 = g01(np.full(len(u), r[0]), qp[:, 0], r[1] - qp[:, 1])
                vec = T[p][0] * T[q][0] * G1 + T[p][1] * T[q][1] * G0
                x = u / L[q]
                s0 = (G0 * w).sum()
                vr, vf = (vec * w * x).sum(), (vec * w * (1 - x)).sum()
                sv[p, q] += s0 * wl
                tv[p, q, 0, 0] += vr * wl * xo
                tv[p, q, 0, 1] += vf * wl * xo
                tv[p, q, 1, 0] += vr * wl * (1 - xo)
                tv[p, q, 1, 1] += vf * wl * (1 - xo)
    # basis n: rising on panel n, falling on panel n+1 (interior nodes 1..npnl-1)
    nb = npnl - 1
    Z = np.zeros((nb, nb), complex)
    jwmu = 1j * OMEGA * MU0
    inv = 1 / (1j * OMEGA * EPS0)
    halves = [[(n, 0, 1 / L[n]), (n + 1, 1, -1 / L[n + 1])] for n in range(nb)]
    for m_, hm in enumerate(halves):
        for n_, hn in enumerate(halves):
            acc = 0
            for (pa, sa, da_) in hm:
                for (pb, sb, db) in hn:
                    acc += jwmu * tv[pa, pb, sa, sb] + inv * da_ * db * sv[pa, pb]
            Z[m_, n_] = acc
    V = np.zeros(nb, complex)
    if plane:
        # broadside plane wave, E_z = 1: V_m = int f_m t_z dt (each half integrates to L/2)
        for n in range(nb):
            V[n] = 0.5 * (L[n] * T[n][1] + L[n + 1] * T[n + 1][1])
        I = np.linalg.solve(Z, V)
        return I[gap - 1]
    V[gap - 1] = 1.0
    I = np.linalg.solve(Z, V)
    return 1 / I[gap - 1]


def resonance(ka, da, caps, ncap=6):
    """kh where a broadside plane wave's induced centre current is in phase."""
    from scipy.optimize import brentq
    return brentq(lambda kh: solve(kh, ka, da, caps, ncap=ncap, plane=True).imag, 1.3, 1.55, xtol=1e-6)


if __name__ == "__main__":
    import json
    what = sys.argv[1]
    if what == "caps":
        # fed open tube / capped rod admittance
        for ka in (0.04, 0.02):
            for kh in (1.45, 1.5, 1.57):
                for da in (4.0, 2.0, 1.0):
                    for caps in (False, True):
                        y = 1e3 / solve(kh, ka, da, caps, ncap=8)
                        print(json.dumps({"ka": ka, "kh": kh, "da": da, "caps": caps, "G": round(y.real, 5), "B": round(y.imag, 5)}), flush=True)
    elif what == "stepped":
        ka1, ka2 = 0.04, 0.02
        for da in (2.0, 1.0):
            for kh in (1.5, 1.55, 1.6, 1.65):
                h, a1, a2 = kh / K0, ka1 / K0, ka2 / K0
                secs = [(-h, -h / 2, a2), (-h / 2, 0.0, a1), (0.0, h / 2, a1), (h / 2, h, a2)]
                y = 1e3 / solve(kh, ka1, da, False, sections=secs)
                print(json.dumps({"kh": kh, "ka_centre": ka1, "ka_tips": ka2, "step_at": 0.5, "da": da, "G": round(y.real, 5), "B": round(y.imag, 5)}), flush=True)
    elif what == "parasitic":
        # python3 bor_reference.py parasitic KA DA open|caps NCAP
        ka, da, caps, ncap = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "caps", int(sys.argv[5])
        print(f"ka={ka} da={da} caps={caps} ncap={ncap}: resonance kh {resonance(ka, da, caps, ncap):.6f}")
