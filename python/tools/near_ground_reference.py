"""Independent references for wires near, on and in lossy ground (tests/near_ground.rs).

Neither uses wiremom's Sommerfeld code: both are derived afresh from the
transmission-line analogy of a layered half-space (TE and TM parts of each plane
wave, voltage = transverse E), with mpmath quadrature.

    python3 near_ground_reference.py mode 0.1 0.002 -0.002 -0.1
        guided mode of an infinite thin PEC wire (radius 1 mm) parallel to average
        ground at 14.15 MHz, at height z0 (> 0) or depth -z0 (< 0): the root of
        E_x(beta) = 0 on the wire, with the reflected field a k_y integral
        (2-D spectral form, unlike wiremom's 3-D Bessel one)

    python3 near_ground_reference.py trans 0.5,1.0,0.1 3.0,0.5,0.1 ...
        field in the ground (depth d) of a unit current element in the air (height
        zs), offset rho: E_rr, E_ff (horizontal element along rho) and E_zz
        (vertical element), from Fresnel transmission coefficients
"""
import sys
import mpmath as mp

mp.mp.dps = 20
C0 = 299792458.0
MU0 = 4e-7 * mp.pi
EPS0 = 1 / (MU0 * C0**2)
F, EPSR, SIGMA, A = 14.15e6, 13.0, 0.005, 0.001
W = 2 * mp.pi * F
K0 = W / C0
EG = mp.mpc(EPSR, -SIGMA / (W * EPS0))
KG = K0 * mp.sqrt(EG)


def kz(k, kr2):
    return -1j * mp.sqrt(kr2 - k**2)


def ex_per_amp(beta, z0):
    """Axial field on the wire per amp of a current I exp(-j beta x)."""
    if z0 > 0:
        k1, k2, e1, e2, h = K0, KG, 1, EG, z0
    else:
        k1, k2, e1, e2, h = KG, K0, EG, 1, -z0
    u = mp.sqrt(beta**2 - k1**2)
    direct = -1j * W * MU0 * (1 - beta**2 / k1**2) * mp.besselk(0, u * A) / (2 * mp.pi)

    def integrand(ky):
        kr2 = beta**2 + ky**2
        z1, z2 = kz(k1, kr2), kz(k2, kr2)
        gte = (z1 - z2) / (z1 + z2)
        gtm = (z2 / e2 - z1 / e1) / (z2 / e2 + z1 / e1)
        zte, ztm = W * MU0 / (2 * z1), z1 / (2 * W * EPS0 * e1)
        return (zte * ky**2 / kr2 * gte + ztm * beta**2 / kr2 * gtm) * mp.exp(-2j * z1 * h) * mp.cos(ky * A)

    s1, s2 = abs(mp.sqrt(beta**2 - k1**2)), abs(mp.sqrt(beta**2 - k2**2))
    pts = sorted(set([0, s1 / 10, s1, 10 * s1, s2, 10 * s2, 1 / h, 10 / h, 40 / h]))
    return direct - (1 / mp.pi) * mp.quad(integrand, pts + [mp.inf])


def mode(z0):
    guess = K0 * (1.02 - 0.02j) if z0 > 0 else K0 * mp.sqrt((EG + 1) / 2)
    f = lambda b: ex_per_amp(b, z0)
    return mp.findroot(f, (guess, guess * (1.001 - 0.001j)), solver="secant", tol=1e-24, maxsteps=200)


def trans(rho, zs, d):
    def comps(kr, z1):
        # each term carries a factor z1, cancelled by the Jacobian below
        z2 = kz(KG, kr**2)
        gte = (z1 - z2) / (z1 + z2)
        gtm = (z2 / EG - z1) / (z2 / EG + z1)  # transverse E
        gv = (EG * z1 - z2) / (EG * z1 + z2)  # normal E (= -gtm)
        e = mp.exp(-1j * z1 * zs - 1j * z2 * d)
        j0, j2 = mp.besselj(0, kr * rho), mp.besselj(2, kr * rho)
        zte, ztm = W * MU0 / 2, z1 * z1 / (2 * W * EPS0)
        rr = zte * (1 + gte) * (j0 + j2) / 2 + ztm * (1 + gtm) * (j0 - j2) / 2
        ff = zte * (1 + gte) * (j0 - j2) / 2 + ztm * (1 + gtm) * (j0 + j2) / 2
        zz = (W * MU0 / 2) * (kr**2 / K0**2) / EG * (1 + gv) * j0
        return [-kr * v * e / (2 * mp.pi) for v in (rr, ff, zz)]

    # kr = k0 sin t (smooth through the branch point), then kr = k0 cosh s
    a = lambda t: comps(K0 * mp.sin(t), K0 * mp.cos(t))
    b = lambda s: [1j * v for v in comps(K0 * mp.cosh(s), -1j * K0 * mp.sinh(s))]
    smax = mp.acosh(1 + 40 / (K0 * (zs + d)))
    return [
        complex(mp.quad(lambda t: a(t)[i], [0, mp.pi / 4, mp.pi / 2]) + mp.quad(lambda s: b(s)[i], mp.linspace(0, smax, 12)))
        for i in range(3)
    ]


if __name__ == "__main__":
    what, args = sys.argv[1], sys.argv[2:]
    for arg in args:
        if what == "mode":
            z0 = float(arg)
            b = mode(z0) / K0
            print(f"z0 {z0:+.4f}: beta/k0 = {complex(b):.6f}")
        else:
            rho, zs, d = map(float, arg.split(","))
            rr, ff, zz = trans(rho, zs, d)
            print(f"({rho}, {zs}, {d}): rr {rr:.6e}  ff {ff:.6e}  zz {zz:.6e}")
