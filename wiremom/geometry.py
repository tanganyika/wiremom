"""Wire geometry: wires -> segments -> nodes -> triangle (rooftop) basis functions.

A wire is a straight conductor split into equal segments.  Segment ends that
coincide (within a tolerance) are merged into one node, so wires that touch
are electrically connected.  Every node shared by k >= 2 segment ends carries
k - 1 triangle basis functions; free wire ends carry none (current is zero there).

Each basis function is made of two "halves", one on each of two segments that
meet at the node.  On a half the current is ``c * shape(u) * t`` where ``t`` is
the segment's unit vector, ``u`` the distance from the segment start, ``shape``
either rising (u/L) or falling (1 - u/L), and ``c`` = +-1 picks the flow direction.
The basis current flows into the node along its first half and out along its
second, and equals 1 A at the node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RISE, FALL = 0, 1


@dataclass
class Wire:
    tag: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    nseg: int
    conductivity: float | None = None  # S/m; None = perfect conductor
    segments: list[int] = field(default_factory=list)  # global segment indices


@dataclass
class Half:
    seg: int
    shape: int  # RISE or FALL
    c: float  # +1 / -1 current direction relative to the segment vector
    div: float  # d(current)/ds along the segment, in units of 1/L (sign only)


@dataclass
class Basis:
    node: int
    halves: tuple[Half, ...]  # two halves, or one at a node on a ground plane


class Structure:
    """A collection of straight wires.  Call :meth:`add_wire`, then solve."""

    def __init__(self, merge_tol: float = 1e-6):
        self.wires: list[Wire] = []
        self.merge_tol = merge_tol
        self._built = False
        self._ground_plane = False

    # ------------------------------------------------------------------ input
    def add_wire(self, start, end, radius: float, nseg: int, tag: int | None = None,
                 conductivity: float | None = None) -> int:
        """Add a straight wire from ``start`` to ``end`` (metres).

        Returns the wire's tag.  ``conductivity`` (S/m) adds skin-effect loss;
        e.g. 5.8e7 for copper.
        """
        if nseg < 1:
            raise ValueError("nseg must be >= 1")
        if radius <= 0:
            raise ValueError("radius must be > 0")
        tag = len(self.wires) + 1 if tag is None else tag
        if any(w.tag == tag for w in self.wires):
            raise ValueError(f"duplicate wire tag {tag}")
        self.wires.append(Wire(tag, np.asarray(start, float), np.asarray(end, float),
                               float(radius), int(nseg), conductivity))
        self._built = False
        return tag

    def wire(self, tag: int) -> Wire:
        for w in self.wires:
            if w.tag == tag:
                return w
        raise KeyError(f"no wire with tag {tag}")

    # ------------------------------------------------------------------ build
    def build(self, ground_plane: bool | None = None) -> None:
        """Mesh the wires.  With ``ground_plane`` a node lying on z = 0 connects each
        segment end to its own image, via single-half basis functions.  ``None`` keeps
        the setting of the previous build (default: no ground plane)."""
        if ground_plane is None:
            ground_plane = self._ground_plane
        if self._built and self._ground_plane == ground_plane:
            return
        self._ground_plane = ground_plane
        p0, p1, rad, sigma = [], [], [], []
        for w in self.wires:
            w.segments = []
            pts = w.start + np.linspace(0, 1, w.nseg + 1)[:, None] * (w.end - w.start)
            for i in range(w.nseg):
                w.segments.append(len(p0))
                p0.append(pts[i])
                p1.append(pts[i + 1])
                rad.append(w.radius)
                sigma.append(w.conductivity)
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        d = self.p1 - self.p0
        self.length = np.linalg.norm(d, axis=1)
        if np.any(self.length <= 0):
            raise ValueError("zero-length segment")
        self.t = d / self.length[:, None]
        self.radius = np.array(rad)
        self.sigma = sigma
        self.wire_of_seg = np.concatenate([[w.tag] * w.nseg for w in self.wires])

        # merge segment ends into nodes
        tol = self.merge_tol * self.length.min()
        nodes: list[np.ndarray] = []
        ends_at: list[list[tuple[int, int]]] = []  # node -> [(seg, which_end)]
        self.seg_nodes = np.zeros((len(p0), 2), int)
        for s in range(len(p0)):
            for e, p in ((0, self.p0[s]), (1, self.p1[s])):
                for n, q in enumerate(nodes):
                    if np.linalg.norm(p - q) <= tol:
                        break
                else:
                    n = len(nodes)
                    nodes.append(p)
                    ends_at.append([])
                ends_at[n].append((s, e))
                self.seg_nodes[s, e] = n
        self.nodes = np.array(nodes)

        # basis functions: pair the first end at each node with every other one
        self.basis: list[Basis] = []
        self._basis_by_pair: dict[tuple[int, int], int] = {}
        self._ground_basis: dict[int, int] = {}  # segment -> basis on the ground plane
        for n, ends in enumerate(ends_at):
            if ground_plane and abs(nodes[n][2]) <= tol:
                for s, e in ends:
                    # current flows out of the ground into the segment
                    self._ground_basis[s] = len(self.basis)
                    self.basis.append(Basis(n, (_half(s, e, inward=False),)))
                continue
            (sa, ea), rest = ends[0], ends[1:]
            for sb, eb in rest:
                into = _half(sa, ea, inward=True)
                out = _half(sb, eb, inward=False)
                self._basis_by_pair[(sa, sb)] = len(self.basis)
                self.basis.append(Basis(n, (into, out)))
        if not self.basis:
            raise ValueError("structure has no basis functions (every wire needs >= 2 segments "
                             "or a connection to another wire)")
        self._built = True

    # ------------------------------------------------------------------ lookup
    def basis_at(self, tag: int, node: int) -> int:
        """Index of the basis function at ``node`` (0..nseg) of wire ``tag``."""
        return self.oriented_basis(tag, node)[0]

    def oriented_basis(self, tag: int, node: int) -> tuple[int, float]:
        """Basis function at ``node`` (0..nseg) of wire ``tag``, and its orientation.

        The sign is +1 when the basis current flows along the wire (start -> end).
        Valid nodes: interior nodes; a wire end joined to exactly one other wire end
        (e.g. where a trap sits between two wires); a wire end on a ground plane
        (where the current flows up out of the ground, i.e. "along" the wire when the
        wire starts on the ground)."""
        if not self._built:
            raise RuntimeError("structure not built; solve the model first")
        w = self.wire(tag)
        if not 0 <= node <= w.nseg:
            raise ValueError(f"wire {tag}: node must be in 0..{w.nseg}")
        if 1 <= node <= w.nseg - 1:
            sa, sb = w.segments[node - 1], w.segments[node]
            if (sa, sb) in self._basis_by_pair:
                return self._basis_by_pair[(sa, sb)], 1.0
            raise ValueError(f"wire {tag} node {node} is a junction without a basis along the wire")

        seg = w.segments[0] if node == 0 else w.segments[-1]
        at_start = node == 0
        if seg in self._ground_basis:
            # the ground basis flows out of the ground into the segment
            return self._ground_basis[seg], 1.0 if at_start else -1.0
        n = self.seg_nodes[seg, 0 if at_start else 1]
        here = [i for i, b in enumerate(self.basis) if b.node == n]
        if not here:
            raise ValueError(f"wire {tag} node {node} is a free end; it carries no current")
        if len(here) > 1:
            raise ValueError(f"wire {tag} node {node} joins more than two wires; put the "
                             "source or load on an interior node instead")
        b = self.basis[here[0]]
        into = b.halves[0].seg == seg  # basis flows toward the node along this segment
        along = (not into) if at_start else into
        return here[0], 1.0 if along else -1.0

    @property
    def nsegments(self) -> int:
        return sum(w.nseg for w in self.wires)

    @property
    def nunknowns(self) -> int:
        self.build()
        return len(self.basis)


def _half(seg: int, end: int, inward: bool) -> Half:
    """Half of a basis on ``seg`` whose node is at segment end ``end`` (0=start, 1=end).

    Value is 1 at the node and 0 at the far end of the segment.  ``inward`` means the
    current flows toward the node.
    """
    shape = RISE if end == 1 else FALL
    toward_node = 1.0 if end == 1 else -1.0  # direction along t that points at the node
    c = toward_node if inward else -toward_node
    # d/du of c*shape: rise -> +1/L, fall -> -1/L
    div = c * (1.0 if shape == RISE else -1.0)
    return Half(seg, shape, c, div)
