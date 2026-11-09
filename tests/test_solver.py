import math
import shutil

import numpy as np
import pytest

from wiremom import C0, Model, Structure

F = 299.792458e6  # wavelength = 1 m


def dipole(length=0.5, radius=0.001, nseg=40, sigma=None, freq=F):
    st = Structure()
    st.add_wire([0, 0, -length / 2], [0, 0, length / 2], radius, nseg, conductivity=sigma)
    m = Model(st)
    m.add_source(1, nseg // 2)
    return m, m.solve(freq)


def test_half_wave_dipole_impedance_and_gain():
    _, s = dipole()
    z = s.input_impedance
    # thin half-wave dipole, delta-gap feed: ~ 85 + j46 ohm for a = lam/1000
    assert 82 < z.real < 89
    assert 40 < z.imag < 50
    assert s.gain_dbi(np.pi / 2, 0)[0] == pytest.approx(2.15, abs=0.05)
    assert s.efficiency == pytest.approx(1.0, abs=2e-3)


def test_impedance_converges_with_segmentation():
    z = [dipole(nseg=n)[1].input_impedance for n in (20, 40, 80)]
    assert abs(z[2] - z[1]) < abs(z[1] - z[0])
    assert abs(z[2].real - z[1].real) < 1.0


def test_matrix_is_symmetric():
    # Galerkin testing of a reciprocal operator gives a symmetric matrix
    st = Structure()
    st.add_wire([0, 0, 0], [0, 0.3, 0.1], 0.002, 9)
    st.add_wire([0, 0.3, 0.1], [0.2, 0.3, 0.4], 0.001, 7)
    st.add_wire([0, 0.3, 0.1], [-0.3, 0.2, 0.1], 0.003, 5)  # 3-wire junction
    m = Model(st)
    Z = m.impedance_matrix(F)
    # residual asymmetry is quadrature error only (it drops with more points)
    assert np.abs(Z - Z.T).max() < 5e-4 * np.abs(Z).max()
    Z16 = Model(st, n_outer=16, n_inner=16).impedance_matrix(F)
    assert np.abs(Z16 - Z16.T).max() < 1e-5 * np.abs(Z16).max()


def test_monopole_is_half_a_dipole():
    h, a = 0.24, 0.005
    st = Structure()
    st.add_wire([0, 0, 0], [0, 0, h], a, 20)
    mono = Model(st, ground="pec")
    mono.add_source(1, 0)
    sm = mono.solve(F)
    _, sd = dipole(length=2 * h, radius=a, nseg=40)
    assert sm.input_impedance == pytest.approx(sd.input_impedance / 2, rel=1e-6)
    g_m = sm.gain_dbi(np.pi / 2, 0)[0]
    g_d = sd.gain_dbi(np.pi / 2, 0)[0]
    assert g_m - g_d == pytest.approx(10 * math.log10(2), abs=0.02)
    assert sm.efficiency == pytest.approx(1.0, abs=3e-3)


def test_copper_loss():
    _, lossless = dipole(radius=1e-4)
    _, cu = dipole(radius=1e-4, sigma=5.8e7)
    # skin-effect resistance per metre at 300 MHz for a 0.1 mm wire
    rprime = math.sqrt(2 * math.pi * F * 4e-7 * math.pi / (2 * 5.8e7)) / (2 * math.pi * 1e-4)
    added = cu.input_impedance.real - lossless.input_impedance.real
    # sinusoidal current on a half-wave dipole: R_loss ~ R' * L / 2
    assert added == pytest.approx(rprime * 0.5 / 2, rel=0.1)
    assert cu.efficiency < 1.0
    assert cu.efficiency == pytest.approx(lossless.input_impedance.real / cu.input_impedance.real, abs=0.01)


def test_series_load_adds_directly_at_the_feed():
    m, s0 = dipole()
    m.add_load(1, 20, r=10.0, l=1e-8)
    s1 = m.solve(F)
    expected = 10.0 + 1j * 2 * math.pi * F * 1e-8
    assert s1.input_impedance - s0.input_impedance == pytest.approx(expected, rel=1e-6)


def test_parallel_trap_at_resonance_isolates_outer_wire():
    # a high-Q parallel LC at its resonance acts like an open circuit, so the inner
    # section behaves as if the outer wires were (nearly) not there
    f = 14.2e6
    lam = C0 / f
    L = 2e-6
    Cap = 1 / ((2 * math.pi * f) ** 2 * L)
    st = Structure()
    st.add_wire([0, -0.12 * lam, 0], [0, 0.12 * lam, 0], 0.001, 24)
    st.add_wire([0, 0.12 * lam, 0], [0, 0.20 * lam, 0], 0.001, 8)
    st.add_wire([0, -0.12 * lam, 0], [0, -0.20 * lam, 0], 0.001, 8)
    trap = Model(st)
    trap.add_source(1, 12)
    for tag in (2, 3):
        trap.add_load(tag, 4, l=L, c=Cap, parallel=True)  # lossless trap
    # traps sit 0.04 lam out along each outer wire, so compare with a plain wire of
    # +-0.16 lam (the dangling 0.04 lam stubs beyond the traps couple only weakly)
    st2 = Structure()
    st2.add_wire([0, -0.16 * lam, 0], [0, 0.16 * lam, 0], 0.001, 32)
    alone = Model(st2)
    alone.add_source(1, 16)
    zt, za = trap.solve(f).input_impedance, alone.solve(f).input_impedance
    assert abs(zt - za) / abs(za) < 0.1


@pytest.mark.skipif(shutil.which("nec2c") is None, reason="nec2c not installed")
def test_agrees_with_nec2():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import nec2_compare as nc

    for case in nc.cases():
        zw, gw = nc.wiremom_solve(case)
        zn, gn = nc.nec2c_solve(case)
        # resistance and gain are tightly comparable; reactance differs by the feed
        # gap model (node vs. segment), worst for the near-cancelling loaded case
        tol = 0.10 if "loading coil" in case.name else 0.04
        assert abs(zw - zn) / abs(zn) < tol, case.name
        assert np.max(np.abs(gw - gn)) < 0.15, case.name


@pytest.mark.parametrize("reverse", [False, True])
def test_source_at_two_wire_joint_matches_single_wire(reverse):
    _, ref = dipole(nseg=40)
    st = Structure()
    if reverse:  # second wire runs the other way: orientation must be handled
        st.add_wire([0, 0, -0.25], [0, 0, 0], 0.001, 20)
        st.add_wire([0, 0, 0.25], [0, 0, 0], 0.001, 20)
    else:
        st.add_wire([0, 0, -0.25], [0, 0, 0], 0.001, 20)
        st.add_wire([0, 0, 0], [0, 0, 0.25], 0.001, 20)
    m = Model(st)
    m.add_source(1, 20)
    s = m.solve(F)
    assert s.input_impedance == pytest.approx(ref.input_impedance, rel=1e-9)
    m2 = Model(st)
    m2.add_source(2, 20 if reverse else 0)
    z2 = m2.solve(F).input_impedance
    # wire 2 runs downward when reversed, so the same physical source has opposite sign
    assert z2 == pytest.approx(ref.input_impedance, rel=1e-9)
