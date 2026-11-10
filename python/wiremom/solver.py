"""Mixed-potential integral equation (MPIE) solver for thin wires.

With triangle basis/test functions Lambda_m (Galerkin), the impedance matrix is

    Z_mn = j w mu  int int Lambda_m Lambda_n (s_m . s_n) G
         + 1/(j w eps) int int Lambda_m' Lambda_n' G
         + (wire loss and lumped loads on the diagonal blocks)

and Z I = V, where V_m is the delta-gap voltage at basis m's node.  The vector
potential term carries the current, the scalar potential term the charge
(Lambda' is piecewise constant), which is what keeps the formulation well
behaved for short segments and low frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ground as gnd
from .geometry import FALL, RISE, Structure
from .integrals import gauss01, segment_integrals

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)
ETA0 = MU0 * C0
REFLECT = np.array([1.0, 1.0, -1.0])


@dataclass
class Load:
    tag: int
    node: int
    r: float = 0.0
    l: float = 0.0
    c: float | None = None  # None = no capacitor (short)
    parallel: bool = False

    def impedance(self, omega: float) -> complex:
        jw = 1j * omega
        if self.parallel:
            # R || L || C ; zero-valued R/L mean "absent"
            y = 0j
            if self.r:
                y += 1.0 / self.r
            if self.l:
                y += 1.0 / (jw * self.l)
            if self.c:
                y += jw * self.c
            return complex(np.inf) if y == 0 else 1.0 / y
        z = self.r + jw * self.l
        if self.c:
            z += 1.0 / (jw * self.c)
        return z


@dataclass
class Source:
    tag: int
    node: int
    voltage: complex = 1.0


@dataclass
class Model:
    """Structure plus sources, loads and environment."""

    structure: Structure
    sources: list[Source] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    # None = free space; "pec" = perfect ground plane at z = 0; real ground below
    # z = 0 as a preset name ("average", "poor", "very good", "sea water", ...),
    # (eps_r, sigma) or ground.RealGround
    ground: object = None
    n_outer: int = 8
    n_inner: int = 8
    n_ground: int = 6  # quadrature points per segment for the Sommerfeld remainder
    # reuse a Sommerfeld table within this relative frequency distance (sweeps);
    # 0 rebuilds the table at every frequency
    ground_table_reuse: float = 0.02
    _tables: list = field(default_factory=list, repr=False, compare=False)

    def add_source(self, tag: int, node: int, voltage: complex = 1.0) -> None:
        """Delta-gap voltage source at ``node`` (0..nseg) of wire ``tag``.

        Positive voltage drives current from the wire's start toward its end.  End
        nodes are allowed where the wire joins one other wire or meets a ground plane.
        """
        self.sources.append(Source(tag, node, voltage))

    def add_load(self, tag: int, node: int, r=0.0, l=0.0, c=None, parallel=False) -> None:
        """Lumped series (default) or parallel RLC at ``node`` of wire ``tag``.

        Series: Z = R + jwL + 1/(jwC), with ``c=None`` meaning no capacitor.
        Parallel: R || L || C, where a zero or None value means that element is absent.
        Note that in a parallel trap the loss is a *large* R (about Q * w L); a small R
        shorts the trap out.
        """
        self.loads.append(Load(tag, node, r, l, c, parallel))

    # ------------------------------------------------------------------ solve
    @property
    def ground_model(self):
        return gnd.make_ground(self.ground)

    def impedance_matrix(self, freq: float, table=None) -> np.ndarray:
        """Impedance matrix.  ``table`` overrides the Sommerfeld table (testing)."""
        st = self.structure
        g = self.ground_model
        st.build(ground_plane=g == "pec")
        omega = 2 * np.pi * freq
        k = omega / C0

        S, T = segment_integrals(st.p0, st.t, st.length, st.radius, k, self.n_outer, self.n_inner)
        tt = st.t @ st.t.T
        # image scale: 1 for PEC; Gamma0 = (eps-1)/(eps+1) for real ground, whose
        # remaining reflected field comes from the Sommerfeld table below
        img = 0.0
        if g == "pec":
            if np.any(np.minimum(st.p0[:, 2], st.p1[:, 2]) < -1e-9):
                raise ValueError("with a ground plane, all wires must be at z >= 0")
            img = 1.0
        elif g is not None:
            if np.minimum(st.p0[:, 2], st.p1[:, 2]).min() <= 0:
                raise ValueError("over real ground all wires must be above z = 0 (buried or "
                                 "ground-connected wires are not supported yet; use radials "
                                 "a little above the ground, or ground='pec')")
            img = 0.0 if table is not None else g.gamma0(freq)
        if g is not None:
            # image of a current element: horizontal part reversed, charge reversed.
            # = reflected segment carrying the negated current.
            Si, Ti = segment_integrals(st.p0, st.t, st.length, st.radius, k, self.n_outer,
                                       self.n_inner, src_p0=st.p0 * REFLECT, src_t=st.t * REFLECT)
            S = S - img * Si
            T_img = img * Ti
            tt_img = st.t @ (st.t * REFLECT).T
        K = None
        if isinstance(g, gnd.RealGround):
            if table is None:
                table = self._sommerfeld_table(g, freq)
            K = gnd.segment_integrals(table, st.p0, st.t, st.length, self.n_ground, freq=freq)

        # (M, 2) half arrays; a single-half (ground) basis is padded with a null half
        halves = [b.halves if len(b.halves) == 2 else (b.halves[0], None) for b in st.basis]
        seg = np.array([[h.seg if h else hs[0].seg for h in hs] for hs in halves])
        shp = np.array([[h.shape if h else 0 for h in hs] for hs in halves])
        cc = np.array([[h.c if h else 0.0 for h in hs] for hs in halves])
        dv = np.array([[h.div if h else 0.0 for h in hs] for hs in halves]) / st.length[seg]

        M = len(st.basis)
        Z = np.zeros((M, M), complex)
        jwmu = 1j * omega * MU0
        inv_jweps = 1.0 / (1j * omega * EPS0)
        for a in range(2):
            for b in range(2):
                sp, sq = seg[:, a][:, None], seg[:, b][None, :]
                cap = cc[:, a][:, None] * cc[:, b][None, :]
                vec = T[sp, sq, shp[:, a][:, None], shp[:, b][None, :]] * tt[sp, sq]
                if g is not None:
                    vec = vec - T_img[sp, sq, shp[:, a][:, None], shp[:, b][None, :]] * tt_img[sp, sq]
                Z += jwmu * cap * vec
                Z += inv_jweps * dv[:, a][:, None] * dv[:, b][None, :] * S[sp, sq]
                if K is not None:
                    # reflected field E = D . J tested with the basis: Z = -<t, E>
                    Z -= cap * K[sp, sq, shp[:, a][:, None], shp[:, b][None, :]]

        # skin-effect loss: R' = Rs / (2 pi a), Rs = sqrt(w mu / 2 sigma), times the
        # Gram integral of two linear shapes on the same segment (L/3 or L/6)
        on_seg: dict[int, list[tuple[int, int]]] = {}
        for m in range(M):
            for a in range(2):
                if cc[m, a]:
                    on_seg.setdefault(seg[m, a], []).append((m, a))
        for s, users in on_seg.items():
            if st.sigma[s] is None:
                continue
            rprime = np.sqrt(omega * MU0 / (2 * st.sigma[s])) / (2 * np.pi * st.radius[s])
            for m, a in users:
                for n, b in users:
                    gram = st.length[s] / (3.0 if shp[m, a] == shp[n, b] else 6.0)
                    Z[m, n] += rprime * cc[m, a] * cc[n, b] * gram

        opens = []
        for ld in self.loads:
            m = st.basis_at(ld.tag, ld.node)
            zl = ld.impedance(omega)
            if not np.isfinite(zl):
                opens.append(m)
            else:
                Z[m, m] += zl
        # open circuit (insulator, or lossless trap at resonance): force that node's
        # current to zero with an overwhelming diagonal entry
        big = 1e12 * np.abs(Z).max()
        for m in opens:
            Z[m, m] = big
        return Z

    def _sommerfeld_table(self, g, freq: float):
        st = self.structure
        r_min, r_max = gnd.table_range(st.p0, st.t, st.length, self.n_ground)
        best = None
        for tab in self._tables:
            if tab.covers(g, freq, r_min, r_max, self.ground_table_reuse):
                if best is None or abs(tab.freq - freq) < abs(best.freq - freq):
                    best = tab
        if best is not None:
            return best
        margin = 1.0 + 2 * self.ground_table_reuse
        tab = gnd.SommerfeldTable(g, freq, r_min, r_max, margin=margin)
        self._tables = (self._tables + [tab])[-16:]
        return tab

    def solve(self, freq: float) -> "Solution":
        if not self.sources:
            raise ValueError("model has no sources")
        st = self.structure
        Z = self.impedance_matrix(freq)
        V = np.zeros(len(st.basis), complex)
        idx = []
        for src in self.sources:
            m, sign = st.oriented_basis(src.tag, src.node)
            V[m] += sign * src.voltage
            idx.append((m, sign))
        I = np.linalg.solve(Z, V)
        return Solution(self, freq, I, idx)

    def sweep(self, freqs) -> list["Solution"]:
        return [self.solve(f) for f in freqs]


class Solution:
    def __init__(self, model: Model, freq: float, currents: np.ndarray,
                 source_basis: list[tuple[int, float]]):
        self.model = model
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.k = self.omega / C0
        self.currents = currents  # basis coefficients = node currents (A)
        self._src = source_basis

    # ------------------------------------------------------------- feed point
    @property
    def input_impedance(self) -> complex:
        """Impedance at the first source (ohms)."""
        return self.source_impedances[0]

    @property
    def source_impedances(self) -> list[complex]:
        return [s.voltage / (sign * self.currents[m])
                for s, (m, sign) in zip(self.model.sources, self._src)]

    def vswr(self, z0: float = 50.0) -> float:
        g = abs((self.input_impedance - z0) / (self.input_impedance + z0))
        return (1 + g) / (1 - g)

    @property
    def input_power(self) -> float:
        return 0.5 * sum((s.voltage * np.conj(sign * self.currents[m])).real
                         for s, (m, sign) in zip(self.model.sources, self._src))

    # ------------------------------------------------------------- far field
    def _segment_amplitudes(self):
        """Per segment, coefficients of the rising and falling current shapes."""
        st = self.model.structure
        A = np.zeros((st.nsegments, 2), complex)
        for b, coef in zip(st.basis, self.currents):
            for h in b.halves:
                A[h.seg, h.shape] += h.c * coef
        return A

    def radiation_vector(self, theta, phi, n_quad: int = 6, image: bool = False):
        """Radiation vector N(theta, phi) = int J exp(jk r_hat . r') dl' (cartesian, A m)
        of the currents, or with ``image`` of their PEC image (current -P J at P r')."""
        st = self.model.structure
        theta = np.atleast_1d(np.asarray(theta, float))
        phi = np.atleast_1d(np.asarray(phi, float))
        rhat = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=-1)                       # (D, 3)
        x, w = gauss01(n_quad)
        A = self._segment_amplitudes()
        sign, refl = (-1.0, REFLECT) if image else (1.0, np.ones(3))
        p0, tv = st.p0 * refl, st.t * refl
        pts = p0[:, None, :] + (x[None, :, None] * st.length[:, None, None]) * tv[:, None, :]
        amp = A[:, RISE, None] * x[None, :] + A[:, FALL, None] * (1 - x[None, :])  # (N, q)
        phase = np.exp(1j * self.k * np.einsum("dk,nqk->dnq", rhat, pts))
        s = np.einsum("dnq,nq,q,n->dn", phase, amp, w, st.length)           # (D, N)
        return sign * s @ tv, theta, phi

    def radiation_intensity(self, theta, phi):
        """U(theta, phi) in W/sr, with the far-field factors E_theta, E_phi (r e^{jkr} E).

        Over ground the reflected wave is the PEC image's field scaled by the plane-wave
        reflection coefficients at that elevation: gTM for E_theta, -gTE for E_phi
        (both 1 for a perfect ground).  This is the space wave only; the surface wave,
        which vanishes in the far field, is not included."""
        th = np.atleast_1d(np.asarray(theta, float))
        ph = np.atleast_1d(np.asarray(phi, float))
        th_hat = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
        ph_hat = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
        pref = -1j * self.omega * MU0 / (4 * np.pi)
        N, _, _ = self.radiation_vector(th, ph)
        e_th = pref * np.einsum("dk,dk->d", N, th_hat)
        e_ph = pref * np.einsum("dk,dk->d", N, ph_hat)
        g = self.model.ground_model
        if g is not None:
            Ni, _, _ = self.radiation_vector(th, ph, image=True)
            r_th, r_ph = (1.0, 1.0)
            if isinstance(g, gnd.RealGround):
                g_te, g_tm = g.reflection(self.freq, np.minimum(th, np.pi / 2))
                r_th, r_ph = g_tm, -g_te
            e_th = e_th + r_th * pref * np.einsum("dk,dk->d", Ni, th_hat)
            e_ph = e_ph + r_ph * pref * np.einsum("dk,dk->d", Ni, ph_hat)
        U = (np.abs(e_th) ** 2 + np.abs(e_ph) ** 2) / (2 * ETA0)
        if g is not None:
            U = np.where(th > np.pi / 2 + 1e-12, 0.0, U)
        return U, e_th, e_ph

    def gain(self, theta, phi):
        """Power gain (linear), relative to input power: includes wire and load loss."""
        U, _, _ = self.radiation_intensity(theta, phi)
        return 4 * np.pi * U / self.input_power

    def gain_dbi(self, theta, phi):
        return 10 * np.log10(np.maximum(self.gain(theta, phi), 1e-30))

    def radiated_power(self, n_theta: int = 91, n_phi: int = 72) -> float:
        th_max = np.pi / 2 if self.model.ground_model is not None else np.pi
        xt, wt = gauss01(n_theta)
        th = xt * th_max
        ph = np.arange(n_phi) * 2 * np.pi / n_phi
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        U, _, _ = self.radiation_intensity(TH.ravel(), PH.ravel())
        U = U.reshape(TH.shape)
        return float(np.sum(U * np.sin(TH) * wt[:, None]) * th_max * 2 * np.pi / n_phi)

    @property
    def efficiency(self) -> float:
        """Radiated (space-wave) power / input power.  Over real ground this also
        counts power absorbed by the ground as lost."""
        return self.radiated_power() / self.input_power
