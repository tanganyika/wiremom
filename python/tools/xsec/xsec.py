"""2-D electrostatic cross-section of a wire near (on, in) lossy soil.
import logging; logging.disable(logging.WARNING)

Coordinates: y horizontal, z vertical, soil below z = 0.  A conductor of radius a,
optionally in an insulating sleeve (outer radius b = a + t, permittivity eps_i),
with its axis at height zc (negative = below the surface).  Air above, soil with
complex relative permittivity eps_s below.  Conductor at 1 V, an outer circle of
radius R at 0 V.  Returns the potential per unit charge in units of 1/(2 pi eps0):
p = 2 pi eps0 V / q (dimensionless, complex).  Differences of p between
geometries are independent of R once R is large.
"""
import math
import numpy as np
import gmsh
import skfem
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

AIR, SOIL, INS = 1, 2, 3


def mesh(a, t, zc, R=1.0, h_fine=None, h_max=None, order=1):
    b = a + t
    ro = b if t > 0 else a
    gap = abs(abs(zc) - ro)  # distance from outer surface to the soil line (0 = crosses)
    crosses = abs(zc) < ro
    h_fine = h_fine or a / 25
    if not crosses:
        h_fine = min(h_fine, max(gap, 1e-6) / 4)
    h_max = h_max or R / 15
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("x")
    occ = gmsh.model.occ
    outer = occ.addDisk(0, 0, 0, R, R)
    cond = occ.addDisk(0, zc, 0, a, a)
    dom, _ = occ.cut([(2, outer)], [(2, cond)])
    rect, _ = occ.cut([(2, occ.addRectangle(-R, -R, 0, 2 * R, R))], [(2, occ.addDisk(0, zc, 0, a, a))])
    tools = list(rect)
    if t > 0:
        ring, _ = occ.cut([(2, occ.addDisk(0, zc, 0, b, b))], [(2, occ.addDisk(0, zc, 0, a, a))])
        tools += ring
    occ.fragment(dom, tools)
    occ.synchronize()
    # the rectangle's corners outside the outer disk
    for dim, tag in gmsh.model.getEntities(2):
        x, y, _ = occ.getCenterOfMass(dim, tag)
        if math.hypot(x, y) > 0.6 * R:
            occ.remove([(dim, tag)], recursive=True)
    occ.synchronize()
    surfs = [tag for _, tag in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, 1)
    # refine toward the wire's circles and the points where they meet the soil line
    curves = [c for _, c in gmsh.model.getEntities(1)
              if _curve_near_wire(c, zc, b if t > 0 else a)]
    pts = [p for _, p in gmsh.model.getEntities(0)
           if abs(gmsh.model.getValue(0, p, [])[1]) < 1e-9
           and abs(gmsh.model.getValue(0, p, [])[0]) < 2 * (b if t > 0 else a)]
    # element size grows geometrically away from the wire, the contact edges and
    # the soil line: size = max(floor, g * distance)
    g = 0.25
    def graded(kind, ents, floor):
        d = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d, kind, ents)
        if kind == "CurvesList":
            gmsh.model.mesh.field.setNumber(d, "Sampling", 2000)
        e = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(e, "F", f"Min({h_max}, Max({floor}, {g}*F{d}))")
        return e
    fields = [graded("CurvesList", curves, h_fine)]
    if pts:
        fields.append(graded("PointsList", pts, h_fine / 20))
    fmin = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    idx = {n: i for i, n in enumerate(nodes)}
    xy = coords.reshape(-1, 3)[:, :2]
    tris = []
    for tag in surfs:
        types, _, enodes = gmsh.model.mesh.getElements(2, tag)
        for ty, en in zip(types, enodes):
            if ty == 2:
                tris.append(np.array([idx[n] for n in en]).reshape(-1, 3))
    gmsh.finalize()
    tris = np.vstack(tris)
    # each triangle's medium from its centroid
    c = xy[tris].mean(axis=1)
    rc = np.hypot(c[:, 0], c[:, 1] - zc)
    region = np.where(c[:, 1] > 0, AIR, SOIL)
    if t > 0:
        region = np.where(rc < b, INS, region)
    used = np.unique(tris)
    remap = -np.ones(len(xy), int)
    remap[used] = np.arange(len(used))
    m = skfem.MeshTri(xy[used].T, remap[tris].T)
    return m, region


def _inside_disk(tag, zc, r):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
    tol = r * 1e-3 + 3e-7  # gmsh pads boxes by 1e-7 m
    return max(abs(xmin), abs(xmax)) <= r + tol and ymin >= zc - r - tol and ymax <= zc + r + tol


def _curve_near_wire(c, zc, r):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
    tol = r * 1e-3 + 3e-7
    return max(abs(xmin), abs(xmax)) <= r + tol and ymin >= zc - r - tol and ymax <= zc + r + tol


def _curve_on_soil_line(c):
    _, ymin, _, _, ymax, _ = gmsh.model.getBoundingBox(1, c)
    return abs(ymin) < 3e-7 and abs(ymax) < 3e-7


def solve(a, t, eps_i, zc, eps_s, R=1.0, order=2, kw_out=None, **kw):
    m, region = mesh(a, t, zc, R, **kw)
    e = skfem.ElementTriP2() if order == 2 else skfem.ElementTriP1()
    lap = skfem.BilinearForm(lambda u, v, w: dot(grad(u), grad(v)))
    A = None
    for k, eps in ((AIR, 1.0), (SOIL, eps_s), (INS, eps_i)):
        el = np.nonzero(region == k)[0]
        if len(el) == 0:
            continue
        Kk = skfem.asm(lap, skfem.Basis(m, e, elements=el)).astype(complex) * eps
        A = Kk if A is None else A + Kk
    basis = skfem.Basis(m, e)
    p = m.p
    bf = m.boundary_facets()
    mid = m.p[:, m.facets[:, bf]].mean(axis=1)
    near = np.hypot(mid[0], mid[1] - zc) < 0.5 * R
    fc, fo = bf[near], bf[~near]
    dc = basis.get_dofs(fc).all()
    do = basis.get_dofs(fo).all()
    u = np.zeros(basis.N, complex)
    u[dc] = 1.0
    D = np.union1d(dc, do)
    I = np.setdiff1d(np.arange(basis.N), D)
    A = A.tocsr()
    u[I] = spsolve(A[I][:, I].tocsc(), -A[I][:, D] @ u[D])
    q = (A @ u)[dc].sum()  # charge / eps0 per unit length
    if kw_out is not None:
        # potential per unit charge on a semicircle in the air, r0 from the wire: its
        # sin(theta) part is the vertical dipole, which sets how strongly the wire
        # picks up an outside vertical field (see effective_height)
        r0 = kw_out.get("r0", 0.03)
        th = np.linspace(0.05, math.pi - 0.05, 61)
        pts = np.vstack([r0 * np.cos(th), r0 * np.sin(th)])
        phi = basis.probes(pts) @ u / q * 2 * math.pi
        M = np.vstack([np.ones_like(th), np.sin(th), np.cos(2 * th), np.sin(3 * th)]).T
        coef = np.linalg.lstsq(M.astype(complex), phi, rcond=None)[0]
        kw_out["dipole"] = coef[1] * r0  # (2 pi eps0 / q) * dipole-term coefficient * r0
    return 2 * math.pi / q, m.t.shape[1]
