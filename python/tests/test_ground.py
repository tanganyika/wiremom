import numpy as np
import pytest

from wiremom import C0, Model, Structure
from wiremom import ground as gnd
from wiremom.ground import MU0, PRESETS, RealGround, reflected_dyadic

F = 14.2e6
K = 2 * np.pi * F / C0
LAM = C0 / F


def closed_form_image_field(rho, h, c):
    """E at (rho, 0, h) above the image point of a unit element c (image current -P c)."""
    Rv = np.array([rho, 0.0, h])
    R = np.linalg.norm(Rv)
    Rh = Rv / R
    kr = K * R
    g = np.exp(-1j * kr) / (4 * np.pi * R)
    A = 1 - 1j / kr - 1 / kr ** 2
    B = -1 + 3j / kr + 3 / kr ** 2
    cim = -np.array([1.0, 1.0, -1.0]) * c
    return -1j * 2 * np.pi * F * MU0 * g * (A * cim + B * Rh * (Rh @ cim))


def local_field(D, c):
    rr, ff, rz, zr, zz = D
    rh, fh, zh = np.eye(3)
    return rr * (rh @ c) * rh + ff * (fh @ c) * fh + rz * c[2] * rh + zr * (rh @ c) * zh + zz * c[2] * zh


@pytest.mark.parametrize("rho,h", [(0.0, 3.0), (2.0, 3.0), (10.0, 1.0), (25.0, 0.2), (1.0, 0.05), (0.3, 8.0)])
def test_spectral_integrals_reproduce_pec_image(rho, h):
    # with the perfect-conductor coefficients the Sommerfeld integrals must equal
    # the closed-form image field: this pins down every sign and the integration path
    D = reflected_dyadic(rho, h, F, 1.0, fixed=(-1.0, 1.0))
    for c in np.eye(3).tolist() + [[0.6, 0.3, 0.74]]:
        c = np.array(c)
        exact = closed_form_image_field(rho, h, c)
        assert np.linalg.norm(local_field(D, c) - exact) < 1e-7 * np.linalg.norm(exact)


@pytest.mark.parametrize("name", list(PRESETS))
def test_table_interpolation(name):
    gr = PRESETS[name]
    eps, g0 = gr.eps_c(F), gr.gamma0(F)
    tab = gnd.SommerfeldTable(gr, F, 0.1, 2 * LAM)
    rng = np.random.default_rng(0)
    for _ in range(12):
        R, th = rng.uniform(0.1, 2 * LAM), rng.uniform(0.01, np.pi / 2)
        rho, h = R * np.cos(th), R * np.sin(th)
        d = reflected_dyadic(rho, h, F, eps, (g0, -g0))
        t = tab(np.array(rho), np.array(h))
        assert np.abs(d - t).max() < 2e-3 * np.abs(d).max()


def _dipole(height, ground, vertical=False, nseg=30):
    st = Structure()
    if vertical:
        st.add_wire([0, 0, height], [0, 0, height + 0.48 * LAM], 0.005, nseg)
    else:
        st.add_wire([0, -0.24 * LAM, height], [0, 0.24 * LAM, height], 0.002, nseg)
    m = Model(st, ground=ground)
    m.add_source(1, nseg // 2)
    return m


def test_tabulated_pec_image_matches_closed_form_image():
    # whole pipeline (table, interpolation, Galerkin geometry, matrix sign): a table
    # of the PEC coefficients, with no closed-form image, must give the PEC answer
    for vertical, height in ((False, 0.25 * LAM), (True, 0.05 * LAM)):
        zp = _dipole(height, "pec", vertical).solve(F).input_impedance
        m = _dipole(height, "average", vertical)
        st = m.structure
        st.build(False)
        r0, r1 = gnd.table_range(st.p0, st.t, st.length, m.n_ground)
        tab = gnd.SommerfeldTable(PRESETS["average"], F, r0, r1, fixed=(-1.0, 1.0))
        Z = m.impedance_matrix(F, table=tab)
        i = st.basis_at(1, 15)
        V = np.zeros(len(Z), complex)
        V[i] = 1
        z = 1 / np.linalg.solve(Z, V)[i]
        assert z == pytest.approx(zp, rel=1e-4)


def test_highly_conducting_ground_approaches_pec():
    zp = _dipole(1.0, "pec", vertical=True).solve(F).input_impedance
    z = [_dipole(1.0, (81.0, s), vertical=True).solve(F).input_impedance for s in (5.0, 1e3)]
    assert abs(z[1] - zp) < abs(z[0] - zp) + 1e-9
    assert abs(z[1] - zp) / abs(zp) < 2e-3


def test_real_ground_matrix_is_symmetric():
    st = Structure()
    st.add_wire([0, 0, 2.0], [3.0, 1.0, 6.0], 0.002, 9)
    st.add_wire([3.0, 1.0, 6.0], [3.0, 5.0, 7.0], 0.002, 7)
    Z = Model(st, ground="poor").impedance_matrix(F)
    assert np.abs(Z - Z.T).max() < 1e-3 * np.abs(Z).max()


def test_far_field_over_real_ground():
    # vertical over real ground: the reflected TM wave cancels the direct wave at
    # the horizon (gTM -> -1 at grazing), unlike perfect ground
    s = _dipole(1.0, "average", vertical=True).solve(F)
    g = s.gain_dbi(np.radians([89.99, 80.0, 70.0]), 0.0)
    assert g[0] < g[1] - 20
    # lossy ground absorbs power, a perfect one does not
    assert s.efficiency < 0.9
    sp = _dipole(1.0, "pec", vertical=True).solve(F)
    assert sp.efficiency == pytest.approx(1.0, abs=5e-3)
    # PEC limit of the Fresnel coefficients is (-1, +1)
    g_te, g_tm = RealGround(1.0, 1e12).reflection(F, np.radians([0.0, 45.0, 80.0]))
    assert np.allclose(g_te, -1, atol=1e-4) and np.allclose(g_tm, 1, atol=1e-4)


def test_sweep_table_reuse_is_accurate():
    fs = np.linspace(14.0e6, 14.3e6, 4)
    m_fresh = _dipole(8.0, "average")
    m_fresh.ground_table_reuse = 0.0
    m_reuse = _dipole(8.0, "average")
    z0 = np.array([m_fresh.solve(f).input_impedance for f in fs])
    z1 = np.array([m_reuse.solve(f).input_impedance for f in fs])
    assert len(m_reuse._tables) < len(m_fresh._tables)
    assert np.max(np.abs(z1 - z0) / np.abs(z0)) < 2e-3


def test_wires_must_be_above_real_ground():
    st = Structure()
    st.add_wire([0, 0, 0], [0, 0, 5], 0.002, 10)
    m = Model(st, ground="average")
    m.add_source(1, 5)
    with pytest.raises(ValueError):
        m.solve(F)


def test_ground_specs():
    assert gnd.make_ground("average") == RealGround(13.0, 0.005)
    assert gnd.make_ground((10, 0.01)) == RealGround(10.0, 0.01)
    with pytest.raises(ValueError):
        gnd.make_ground("swamp")
