"""Guided modes of one wire and of two parallel wires (in phase) near lossy ground:
2-D spectral form, with the oscillatory k_y integrals done by QUADPACK's QAWF."""
import numpy as np, mpmath as mp, sys
from scipy.integrate import quad
from scipy.special import kv
C0 = 299792458.0; MU0 = 4e-7*np.pi; EPS0 = 1/(MU0*C0**2)
f = 14.15e6; W = 2*np.pi*f; K0 = W/C0
EG = 13.0 - 1j*0.005/(W*EPS0); KG = K0*np.sqrt(EG)
def kz(k, kr2): return -1j*np.sqrt(kr2 - k*k)
def ex_at(beta, z0, y, a):
    k1, k2, e1, e2, h = (K0, KG, 1.0, EG, z0) if z0 > 0 else (KG, K0, EG, 1.0, -z0)
    u = np.sqrt(beta**2 - k1**2)
    direct = -1j*W*MU0*(1 - beta**2/k1**2)*kv(0, u*max(y, a) if y > a else u*a)/(2*np.pi)
    def g(ky):
        kr2 = beta**2 + ky**2
        z1, z2 = kz(k1, kr2), kz(k2, kr2)
        gte = (z1 - z2)/(z1 + z2); gtm = (z2/e2 - z1/e1)/(z2/e2 + z1/e1)
        return (W*MU0/(2*z1)*ky**2/kr2*gte + z1/(2*W*EPS0*e1)*beta**2/kr2*gtm)*np.exp(-2j*z1*h)
    yy = y
    K = 20*max(abs(np.sqrt(beta**2 - k1**2)), abs(np.sqrt(beta**2 - k2**2)))
    part = lambda fn, lo, hi: quad(lambda t: fn(t).real*np.cos(t*yy), lo, hi, limit=500)[0] + 1j*quad(lambda t: fn(t).imag*np.cos(t*yy), lo, hi, limit=500)[0]
    I = part(g, 0, K)
    I += quad(lambda t: g(t).real, K, np.inf, weight="cos", wvar=yy, limlst=200)[0] + 1j*quad(lambda t: g(t).imag, K, np.inf, weight="cos", wvar=yy, limlst=200)[0]
    return direct - I/np.pi
def root(F, guess):
    return complex(mp.findroot(lambda b: mp.mpc(F(complex(b))), (mp.mpc(guess), mp.mpc(guess*(1.001-0.001j))), solver="secant", tol=1e-18, maxsteps=100))
if __name__ == "__main__":
    a, s = 1e-3, 0.3
    for z0, g in ((2e-3, 1.85 - 0.28j), (-2e-3, 2.92 - 0.78j)):
        b1 = root(lambda b: ex_at(b, z0, a, a), K0*g) / K0
        b2 = root(lambda b: ex_at(b, z0, a, a) + ex_at(b, z0, s, a), K0*g) / K0
        print(f"z {z0*1e3:+.0f} mm: one wire {b1:.4f}   two wires {s} m apart in phase {b2:.4f}", flush=True)
