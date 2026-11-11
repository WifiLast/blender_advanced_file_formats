"""Stage 4: B-rep geometry from decoded GElement records -> triangle meshes.

Model (RVT_FORMAT.md §12):
  Geometry (GBRep) .m_pFaces -> Face { m_pSurf: Plane | CylSurf | ..., m_pFirstLoop: EdgeLoop }
  EdgeLoop.m_next -> first Edge (ref); Edge.m_next[side] -> next Edge or back to the loop;
  EdgeLoop.m_nextLoop -> further loops (holes) of the same face.
  Edge shape = UV points on each adjacent face: m_firstAndLastEdgePnts / m_interiorEdgePnts,
  EdgePnt.uv[side] = (u, v) on face m_pFace[side].
Surfaces are evaluated at the UV points; faces are triangulated in UV space.
Units: Revit feet, Z up.
"""
from __future__ import annotations

import math

import numpy as np



def _ref(x):
    return x.get("__ref__") if isinstance(x, dict) else None


DETAIL_FINE = 3
FILTER_3D = False       # visible_3d() is not validated yet (removed real geometry); see RVT_FORMAT.md §12
FINE_FILTER = True      # hidden_fine_3d(): the validated subset (detail level, plan-only direction faces)


def visible_3d(gfilter: dict) -> bool:
    """Evaluate a GFilter for a 3D view at detail level Fine (RVT_FORMAT.md §12):
    GConditionInt m_param 3 = detail level (1 coarse, 2 medium, 3 fine; m_comp 0 '!=',
    3 '==', see hidden_fine_3d); GConditionDir = view-direction (2D symbolic) -> hidden; GConditionSelected ->
    hidden; a filter with only GConditionCut conditions is plan-cut representation -> hidden."""
    conds = [c for c in gfilter.get("m_oConditions") or [] if isinstance(c, dict)]
    kinds = {c.get("__class__") for c in conds}
    if "GConditionDir" in kinds or "GConditionSelected" in kinds:
        return False
    if kinds and kinds <= {"GConditionCut"}:
        return False
    for c in conds:
        if c.get("__class__") == "GConditionInt" and c.get("m_param") == 3:
            comp, val = c.get("m_comp"), c.get("m_value")
            if comp == 0 and val == DETAIL_FINE:
                return False
            if comp == 3 and val != DETAIL_FINE:
                return False
    return True


DIR_PLAN_ONLY = (6, 7)      # GConditionDir m_comp of plan-only symbolic faces (view direction along the family axis)


def hidden_fine_3d(gfilter: dict) -> bool:
    """GFilter whose subtree a 3D view at detail level Fine does not show (verified per element against the surface
    area of Revit's own GLB export of Golden Nugget, 2026-11-11):
    - GConditionInt m_param 3 (detail level 1/2/3 = coarse/medium/fine): m_comp 0 is '!=' (value 1 = everything but
      coarse: outlets, data devices vanish if read as '=='), m_comp 3 is '==' (cable tray fittings carry a medium
      (value 2) and a fine (value 3) body; both together gave twice Revit's area);
    - GConditionDir m_comp 6 / 7: faces only drawn in views along the family's axis (plan symbols of ceiling lamps,
      sanitary fixtures, electrical equipment); dropping them gave Revit's area exactly."""
    for c in gfilter.get("m_oConditions") or []:
        if not isinstance(c, dict):
            continue
        k = c.get("__class__")
        if k == "GConditionDir" and c.get("m_comp") in DIR_PLAN_ONLY:
            return True
        if k == "GConditionInt" and c.get("m_param") == 3:
            comp, val = c.get("m_comp"), c.get("m_value")
            if (comp == 0 and val == DETAIL_FINE) or (comp == 3 and val != DETAIL_FINE):
                return True
    return False


def find_nodes(o, names, out=None, stop=("GInstance",)):
    """All dicts whose __class__ is in names; does not descend into GInstance subtrees."""
    if out is None:
        out = []
    # iterative pre-order walk (same visiting order as the recursive form; 65M recursive calls per Golden Nugget run)
    stack = [o]
    pop, push = stack.pop, stack.extend
    while stack:
        o = pop()
        if isinstance(o, dict):
            c = o.get("__class__")
            if c in names:
                out.append(o)
            if c in stop:
                continue
            if c == "GFilter" and ((FINE_FILTER and hidden_fine_3d(o)) or (FILTER_3D and not visible_3d(o))):
                continue
            kids = [v for k, v in o.items() if not k.startswith("__") and isinstance(v, (dict, list))]
            kids.reverse()
            push(kids)
        elif isinstance(o, list):
            kids = [v for v in o if isinstance(v, (dict, list))]
            kids.reverse()
            push(kids)
    return out


# -------------------------------------------------------------------------------------------------- free lines --

LINE_CURVES = ("GLine", "GArc", "GEllipse")
LINE_STOP = ("GInstance", "Geometry", "GBRep", "Edge")      # curves inside B-rep edges are not free lines
LINE_ARC_STEP = math.pi / 16                                 # polyline step along arcs / ellipses


def curve_points(c: dict):
    """Free curve node -> polyline vertices (K x 3) over its parameter range m_endParams, or None."""
    t = c.get("m_endParams")
    if not (isinstance(t, (list, tuple)) and len(t) == 2):
        return None
    try:
        t0, t1 = float(t[0]), float(t[1])
        f = curve_eval(c)
        if f is None or not (math.isfinite(t0) and math.isfinite(t1)):
            return None
        n = 1 if c.get("__class__") == "GLine" else max(2, int(math.ceil(abs(t1 - t0) / LINE_ARC_STEP)))
        pts = np.array([f(x) for x in np.linspace(t0, t1, n + 1)], float)
    except (KeyError, TypeError, ValueError):
        return None
    return pts if pts.shape == (n + 1, 3) and np.isfinite(pts).all() and np.ptp(pts, axis=0).max() > 1e-9 else None


def _visible_curves(node: dict) -> list:
    """Free curve nodes a 3D view shows. Lines are stricter than solids: ANY view-direction condition (GConditionDir,
    every m_comp) marks a plan/elevation symbol, 'selected' and cut-only filters are not drawn, and the detail-level
    rules of visible_3d apply. Checked on the grille 3104847 against Navisworks' own nugget.nwd: of its 54 free lines
    exactly the 11 under 'detail != coarse' are exported (0.075 m each); the 12 under direction filters are not."""
    out, stack = [], [node]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            c = o.get("__class__")
            if c in LINE_CURVES:
                out.append(o)
            if c in LINE_STOP and o is not node:
                continue
            if c == "GFilter" and (hidden_fine_3d(o) or not visible_3d(o)):
                continue
            kids = [v for k, v in o.items() if not k.startswith("__") and isinstance(v, (dict, list))]
            kids.reverse()
            stack.extend(kids)
        elif isinstance(o, list):
            kids = [v for v in o if isinstance(v, (dict, list))]
            kids.reverse()
            stack.extend(kids)
    return out


def point_lines(node: dict):
    """Pipe / duct / flex centre lines (SegmentCenterLine, PipeFittingCenterLine, ...) are stored as GPoint nodes only:
    the consecutive points under one parent are one polyline (an axis = two points). Revit shows them at coarse and
    medium detail, Navisworks exports them anyway (4,565 'Achse' elements in nugget.nwd), so detail filters are ignored.
    Only for elements without any solid, mesh or instance. -> (vertices N x 3, segments M x 2) or None"""
    verts, segs, base = [], [], 0
    stack = [node]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            c = o.get("__class__")
            if c in ("Geometry", "GBRep", "GPolyMesh", "GInstance"):
                return None
            stack.extend(v for k, v in o.items() if not k.startswith("__") and isinstance(v, (dict, list)))
            subs = o.get("m_subNodes")
            if isinstance(subs, list):
                pts = [p.get("m_coord") for p in subs if isinstance(p, dict) and p.get("__class__") == "GPoint"]
                pts = [q for q in pts if isinstance(q, (list, tuple)) and len(q) == 3]
                if len(pts) >= 2:
                    a = np.array(pts, float)
                    if np.isfinite(a).all():
                        verts.append(a)
                        segs.append(np.column_stack([np.arange(len(a) - 1), np.arange(1, len(a))]) + base)
                        base += len(a)
        elif isinstance(o, list):
            stack.extend(v for v in o if isinstance(v, (dict, list)))
    if not verts:
        return None
    return np.vstack(verts), np.vstack(segs).astype(np.int64)


def free_lines(node: dict):
    """Free model/symbol lines of a GRep subtree (outside B-rep solids and GInstance subtrees), those a 3D view at fine
    detail shows (the same GFilter rules as find_nodes). -> (vertices N x 3, segments M x 2 vertex indices) or None.
    Navisworks exports them as line geometry (record type 95); Revit draws grille louvres, family symbol lines and
    the axis of pipes/ducts this way."""
    verts, segs, base = [], [], 0
    for c in _visible_curves(node):
        pts = curve_points(c)
        if pts is None:
            continue
        verts.append(pts)
        k = len(pts)
        segs.append(np.column_stack([np.arange(k - 1), np.arange(1, k)]) + base)
        base += k
    if not verts:
        return point_lines(node)
    return np.vstack(verts), np.vstack(segs).astype(np.int64)


# ---------------------------------------------------------------- surfaces --

def _a(d, k):
    return np.array(d[k], float)


def curve_eval(c: dict):
    """Curve -> f(t) -> xyz, or None if the curve type is not supported."""
    k = c.get("__class__") if isinstance(c, dict) else None
    if k == "GLine":
        o, d = _a(c, "m_origin"), _a(c, "m_dirVec")
        return lambda t: o + t * d
    if k == "GArc":
        o, x, y, r = _a(c, "m_center"), _a(c, "m_xVec"), _a(c, "m_yVec"), float(c["m_radius"])
        return lambda t: o + r * (math.cos(t) * x + math.sin(t) * y)
    if k == "GEllipse":
        o, x, y = _a(c, "m_center"), _a(c, "m_xVec"), _a(c, "m_yVec")
        a, b = float(c["m_xLen"]), float(c["m_yLen"])
        return lambda t: o + a * math.cos(t) * x + b * math.sin(t) * y
    if k == "GHermiteSpline":       # cubic Hermite segments: SplineNode m_iParametr / m_iPoint / m_iTangent (d/dt)
        nodes = [n for n in c.get("m_NodeArray") or [] if isinstance(n, dict)]
        if len(nodes) < 2:
            return None
        ts = [float(n["m_iParametr"]) for n in nodes]
        ps = [_a(n, "m_iPoint") for n in nodes]
        ms = [_a(n, "m_iTangent") for n in nodes]

        def herm(t):
            i = max(0, min(len(ts) - 2, next((j for j in range(len(ts) - 1) if t <= ts[j + 1]), len(ts) - 2)))
            h = ts[i + 1] - ts[i]
            s = (t - ts[i]) / h if h else 0.0
            s2, s3 = s * s, s * s * s
            return ((2 * s3 - 3 * s2 + 1) * ps[i] + (s3 - 2 * s2 + s) * h * ms[i]
                    + (-2 * s3 + 3 * s2) * ps[i + 1] + (s3 - s2) * h * ms[i + 1])
        return herm
    return None


def curve_step(c: dict):
    """Parameter step for one straight segment of a curve: angle curves pi/16, splines 1/16 of their range,
    lines None (straight)."""
    k = c.get("__class__") if isinstance(c, dict) else None
    if k in ("GArc", "GEllipse"):
        return ANGLE_STRIP
    if k == "GHermiteSpline":
        e = c.get("m_endParams") or [0.0, 1.0]
        return abs(float(e[1]) - float(e[0])) / 16 or None
    return None


ANGLE_STRIP = math.pi / 16      # u = angle
UNIT_STRIP = 1.0 / 16           # u = normalised curve parameter


TWO_PI = 2 * math.pi
CHORD_TOL = 0.0005 / 0.3048     # feet: largest gap between a facet and the true curved surface (0.5 mm)
MAX_ANGLE_STEP = math.pi / 4


def angle_step(radius: float) -> float:
    """Angle step for a circle of this radius (feet): chord deviation <= CHORD_TOL, between pi/16 (large radii, the
    former fixed step) and pi/4 (tiny radii: a 9 mm pipe bend gets 8-10 segments instead of 32)."""
    r = abs(float(radius))
    if r <= CHORD_TOL:
        return MAX_ANGLE_STEP
    a = 2 * math.acos(max(-1.0, 1 - CHORD_TOL / r))
    return min(MAX_ANGLE_STEP, max(ANGLE_STRIP, a))


def surface_info(s: dict):
    """-> (f(u, v) -> xyz ndarray, (u step, v step), (u period, v period)) or None if unsupported.
    A step is the largest parameter interval one triangle edge may span in that direction (None = the surface is
    straight in that direction); a period is 2 pi for angle parameters (0 = not periodic).
    Parametrisations verified by edge consistency (RVT_FORMAT.md §12)."""
    c = s.get("__class__")
    if c == "Plane":
        o, x, y = _a(s, "m_origin"), _a(s, "m_xVec"), _a(s, "m_yVec")
        return (lambda u, v: o + u * x + v * y), (None, None), (0.0, 0.0)
    if c == "CylSurf":
        o, x, y, z = _a(s, "m_center"), _a(s, "m_xVec"), _a(s, "m_yVec"), _a(s, "m_zVec")
        r = float(s["m_radius"])
        return ((lambda u, v: o + r * (math.cos(u) * x + math.sin(u) * y) + v * z),
                (angle_step(r), None), (TWO_PI, 0.0))
    if c == "ConeSurf":          # v = distance along the generator line
        o, x, y, z = _a(s, "m_center"), _a(s, "m_xVec"), _a(s, "m_yVec"), _a(s, "m_zVec")
        sa, ca = math.sin(s["m_halfAngle"]), math.cos(s["m_halfAngle"])
        return ((lambda u, v: o + v * (sa * (math.cos(u) * x + math.sin(u) * y) + ca * z)),
                (ANGLE_STRIP, None), (TWO_PI, 0.0))
    if c == "SurfRev":           # u = angle, v = profile parameter; profile in the local frame
        pc = s.get("m_pProfileCurve")
        prof = curve_eval(pc)
        if prof is None:
            return None
        o, x, y, z = _a(s, "m_center"), _a(s, "m_xVec"), _a(s, "m_yVec"), _a(s, "m_zVec")

        def rev(u, v):
            p = prof(v)
            r, a0 = math.hypot(p[0], p[1]), math.atan2(p[1], p[0])
            return o + r * (math.cos(u + a0) * x + math.sin(u + a0) * y) + p[2] * z
        # arc / ellipse profile (pipe bends = tori): v is an angle too (periodic); spline profile: curved, not periodic
        angle = pc.get("__class__") in ("GArc", "GEllipse")
        # u step from the largest distance of the profile from the axis, v step from the profile arc's own radius
        e0, e1 = (pc.get("m_endParams") or [0.0, 1.0])[:2]
        reach = max(math.hypot(*prof(e0 + (e1 - e0) * k / 4)[:2]) for k in range(5))
        sv = curve_step(pc)
        if pc.get("__class__") == "GArc":
            sv = angle_step(float(pc.get("m_radius", 1.0)))
        elif pc.get("__class__") == "GEllipse":
            sv = angle_step(max(float(pc.get("m_xLen", 1.0)), float(pc.get("m_yLen", 1.0))))
        return rev, (angle_step(reach), sv), (TWO_PI, TWO_PI if angle else 0.0)
    if c == "RuledSurf":         # (1-v) C1(u) + v C2(u), u normalised to each curve's range
        ends, curved = [], False
        for i in (1, 2):
            cv = s.get(f"m_pProfileCurve{i}")
            f = curve_eval(cv) if isinstance(cv, dict) and cv.get("__class__") else None
            if f is not None:
                e0, e1 = cv["m_endParams"]
                ends.append(lambda u, f=f, e0=e0, e1=e1: f(e0 + u * (e1 - e0)))
                curved = curved or cv.get("__class__") != "GLine"
            elif isinstance(cv, dict) and cv.get("__class__"):
                return None                      # unsupported curve type
            else:
                pt = _a(s, f"m_Point{i}")
                ends.append(lambda u, pt=pt: pt)
        c1, c2 = ends
        return (lambda u, v: (1 - v) * c1(u) + v * c2(u)), (UNIT_STRIP if curved else None, None), (0.0, 0.0)
    return None


def surface_eval(s: dict):
    """-> (f(u, v) -> xyz ndarray, step in u or None) or None if unsupported (older interface of surface_info)."""
    info = surface_info(s)
    return (info[0], info[1][0]) if info else None


# ------------------------------------------------------------------ loops --

def _edge_points(e: dict, side: int):
    fl = e["m_firstAndLastEdgePnts"]
    return ([tuple(fl[0]["uv"][side])] + [tuple(p["uv"][side]) for p in e["m_interiorEdgePnts"]]
            + [tuple(fl[1]["uv"][side])])


def _close(a, b, tol=1e-7):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


CHAIN_TOL = 1e-6


def _shift_to(p, q, period):
    """Offset (du, dv) - multiples of the periods - that moves point p onto q, or None if they differ."""
    off = []
    for i in (0, 1):
        d = q[i] - p[i]
        if period[i]:
            d = period[i] * round(d / period[i])
        else:
            d = 0.0
        if abs(p[i] + d - q[i]) >= CHAIN_TOL:
            return None
        off.append(d)
    return off


def _chain(segs, period):
    """Chain a loop's edge point lists into one polygon. The edge walk gives the edges of a loop, but not always in
    traversal order or direction, and on periodic surfaces (u = angle) the same point may be given as u and u + 2 pi
    on different edges (seam). Segments are matched by endpoints in any order and direction, modulo the periods, and
    shifted so that the polygon is continuous in UV. -> (points, closed cleanly)"""
    segs = [list(s) for s in segs if len(s) >= 2]
    if not segs:
        return [], False
    ring = segs.pop(0)
    # orient the first segment so that its end connects to another segment
    if segs and not any(_shift_to(s[0], ring[-1], period) or _shift_to(s[-1], ring[-1], period) for s in segs) \
            and any(_shift_to(s[0], ring[0], period) or _shift_to(s[-1], ring[0], period) for s in segs):
        ring.reverse()
    ok = True
    while segs:
        end = ring[-1]
        hit = None
        for i, s in enumerate(segs):                    # first match in walk order
            off = _shift_to(s[0], end, period)
            if off is not None:
                hit = (i, s, off)
                break
            off = _shift_to(s[-1], end, period)
            if off is not None:
                hit = (i, s[::-1], off)
                break
        if hit is None:                                 # gap: continue with the next edge (nearest end first)
            ok = False
            s = segs[0]
            d0 = abs(s[0][0] - end[0]) + abs(s[0][1] - end[1])
            d1 = abs(s[-1][0] - end[0]) + abs(s[-1][1] - end[1])
            hit = (0, s if d0 <= d1 else s[::-1], (0.0, 0.0))
        i, s, (du, dv) = hit
        segs.pop(i)
        ring.extend((p[0] + du, p[1] + dv) for p in s[1:])
    if len(ring) > 1 and _close(ring[0], ring[-1], CHAIN_TOL):
        ring.pop()
    elif len(ring) > 1 and _shift_to(ring[-1], ring[0], period) is not None:
        ring.pop()                                      # closes only modulo a period: a ring around the surface
    else:
        ok = False
    return ring, ok


def loop_segments(loop: dict, face_tag: int, objs: dict):
    """UV point lists of the edges of one loop, in walk order, or None. A seam edge (both sides on this face) is
    walked twice: once per side, each time with that side's UV points and next pointer."""
    first = objs.get(_ref(loop.get("m_next")))
    if first is None:
        return None
    segs: list = []
    e, guard = first, 0
    loop_tag = loop.get("__tag__")
    seen: dict = {}
    while e is not None and guard < 100000:
        guard += 1
        faces = [_ref(x) for x in e.get("m_pFace", [])]
        if face_tag not in faces or "m_firstAndLastEdgePnts" not in e:
            return None
        sides = [i for i, f in enumerate(faces) if f == face_tag]
        k = seen.get(id(e), 0)
        if k >= len(sides):
            break                                       # every side of this edge walked: the loop is closed
        seen[id(e)] = k + 1
        side = sides[k]
        segs.append(_edge_points(e, side))
        nxt = _ref(e["m_next"][side])
        if nxt is None or nxt == loop_tag:
            break
        e = objs.get(nxt)
        if e is first and seen.get(id(e), 0) >= len([f for f in e.get("m_pFace", []) if _ref(f) == face_tag]):
            break
    return segs


def loop_polygon(loop: dict, face_tag: int, objs: dict, period=(0.0, 0.0)):
    """UV polygon of one loop (list of (u, v), not closed) or None."""
    segs = loop_segments(loop, face_tag, objs)
    if not segs:
        return None
    pts, _ok = _chain(segs, period)
    return pts if len(pts) >= 3 else None


def loops_from_edges(face: dict, edges: list):
    """Fallback for faces without a loop pointer: chain the face's edges by matching
    UV endpoints into closed rings."""
    tag = face.get("__tag__")
    segs = []
    for e in edges:
        faces = [_ref(x) for x in e.get("m_pFace", [])]
        if tag in faces and "m_firstAndLastEdgePnts" in e:
            segs.append(_edge_points(e, faces.index(tag)))
    rings = []
    while segs:
        ring = list(segs.pop())
        changed = True
        while changed and not _close(ring[0], ring[-1]):
            changed = False
            for i, sg in enumerate(segs):
                if _close(ring[-1], sg[0]):
                    ring.extend(sg[1:]); segs.pop(i); changed = True; break
                if _close(ring[-1], sg[-1]):
                    ring.extend(list(reversed(sg))[1:]); segs.pop(i); changed = True; break
        if len(ring) > 3 and _close(ring[0], ring[-1]):
            ring.pop()
            rings.append(ring)
    rings.sort(key=lambda r: -abs(_area(r)))      # outer ring first
    return rings


def face_loops(face: dict, objs: dict, edges=None, period=(0.0, 0.0)):
    # Faces without m_pFirstLoop are auxiliary (their edges never chain on that side).
    # Rebuilding them from edge UVs put geometry outside the stored element bounding
    # boxes (3 walls), so they are skipped; loops_from_edges() is kept for analysis.
    loops, lp, guard = [], face.get("m_pFirstLoop"), 0
    while isinstance(lp, dict) and guard < 1000:
        guard += 1
        poly = loop_polygon(lp, face["__tag__"], objs, period)
        if poly:
            loops.append(poly)
        lp = lp.get("m_nextLoop")
    return loops


def _winding(loop, period):
    """How many times a closed UV loop runs around the periodic directions: [wu, wv]."""
    out = []
    for i in (0, 1):
        if not period[i]:
            out.append(0)
            continue
        d = sum(loop[k][i] - loop[k - 1][i] for k in range(1, len(loop)))
        last = loop[0][i] - loop[-1][i]          # closing step, modulo the period like the chaining
        last -= period[i] * round(last / period[i])
        out.append(round((d + last) / period[i]))
    return out


def _open_ring(ring, axis, period):
    """A loop running once around the surface in direction axis -> open polyline from its first point to the same
    point one period further on (positive direction)."""
    per = (period, 0.0) if axis == 0 else (0.0, period)
    if _winding(ring, per)[axis] < 0:
        ring = ring[::-1]
    end = list(ring[0])
    end[axis] += period
    return list(ring) + [tuple(end)]


def annulus(rings, axis, period):
    """Two loops that each run once around a periodic surface (e.g. a cylinder side between two circles without a
    seam edge) -> one simple UV polygon, cut open along a line from the first ring's start."""
    a = _open_ring(rings[0], axis, period)
    b = _open_ring(rings[1], axis, period)[:-1]
    a0 = a[0][axis]
    # start ring b at its point nearest to a's start (modulo the period), keep it continuous, shift it next to a
    k = min(range(len(b)), key=lambda i: abs(((b[i][axis] - a0 + period / 2) % period) - period / 2))
    b = b[k:] + [tuple(p[j] + (period if j == axis else 0.0) for j in (0, 1)) for p in b[:k]]
    sh = period * round((a0 - b[0][axis]) / period)
    b = [tuple(p[j] + (sh if j == axis else 0.0) for j in (0, 1)) for p in b]
    end = list(b[0])
    end[axis] += period
    b.append(tuple(end))
    return a + b[::-1]


# ---------------------------------------------------------- triangulation --

def _area(p):
    """Signed shoelace area: positive for counter-clockwise polygons."""
    return 0.5 * sum(p[i - 1][0] * p[i][1] - p[i][0] * p[i - 1][1] for i in range(len(p)))


def _in_tri(p, a, b, c):
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def triangulate(loops):
    """Triangulate a face in UV. loops[0] = outer, rest = holes. -> (points, [(i,j,k)])
    With holes: mapbox earcut port (earcut.py). Without: linked-list ear clipping, ears tested
    only against reflex vertices (numpy), scan continues after the last ear (fast for the
    17,554-point faces of the example project). Self-test: analysis/tri_test.py."""
    if not loops:
        return [], []
    if len(loops) > 1:                       # holes: mapbox earcut port (robust bridging)
        from .earcut import earcut       # noqa: PLC0415
        pts = [tuple(p) for lp in loops for p in lp]
        return pts, earcut(loops)
    pts = list(loops[0] if _area(loops[0]) > 0 else loops[0][::-1])
    n = len(pts)
    if n < 3:
        return pts, []
    P = np.asarray(pts, float)
    prv = list(range(-1, n - 1)); prv[0] = n - 1
    nxt = list(range(1, n + 1)); nxt[-1] = 0

    def cross(i, j, k):
        return (P[j, 0] - P[i, 0]) * (P[k, 1] - P[i, 1]) - (P[j, 1] - P[i, 1]) * (P[k, 0] - P[i, 0])

    reflex = {i for i in range(n) if cross(prv[i], i, nxt[i]) <= 0}
    tris = []
    remaining, i, stall = n, 0, 0

    while remaining > 3 and stall <= remaining:
        a_, b_, c_ = prv[i], i, nxt[i]
        ear = False
        if b_ not in reflex:
            ear = True
            R = np.fromiter((r for r in reflex if r not in (a_, b_, c_)), dtype=np.int64)
            if len(R):
                Q = P[R]
                A, B, C = P[a_], P[b_], P[c_]
                d1 = (Q[:, 0] - B[0]) * (A[1] - B[1]) - (A[0] - B[0]) * (Q[:, 1] - B[1])
                d2 = (Q[:, 0] - C[0]) * (B[1] - C[1]) - (B[0] - C[0]) * (Q[:, 1] - C[1])
                d3 = (Q[:, 0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (Q[:, 1] - A[1])
                inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
                same = ((np.abs(Q - A).sum(1) < 1e-12) | (np.abs(Q - B).sum(1) < 1e-12)
                        | (np.abs(Q - C).sum(1) < 1e-12))
                ear = not bool((inside & ~same).any())
        if ear:
            tris.append((a_, b_, c_))
            nxt[a_], prv[c_] = c_, a_
            reflex.discard(b_)
            remaining -= 1
            for v in (a_, c_):
                if cross(prv[v], v, nxt[v]) > 0:
                    reflex.discard(v)
                else:
                    reflex.add(v)
            i, stall = c_, 0
        else:
            i, stall = nxt[i], stall + 1
    if remaining > 3:                   # degenerate remainder: fan it
        ring, v = [], i
        for _ in range(remaining):
            ring.append(v)
            v = nxt[v]
        tris.extend((ring[0], ring[k], ring[k + 1]) for k in range(1, len(ring) - 1))
    elif remaining == 3:
        tris.append((prv[i], i, nxt[i]))
    return pts, tris


def _clip(poly, lo, hi, axis=0):
    """Sutherland-Hodgman clip of a UV polygon to lo <= p[axis] <= hi."""
    other = 1 - axis

    def clip(p, keep, cut):
        out = []
        for i in range(len(p)):
            cur, prev = p[i], p[i - 1]
            ci, pi = keep(cur), keep(prev)
            if ci:
                if not pi:
                    out.append(cut(prev, cur))
                out.append(cur)
            elif pi:
                out.append(cut(prev, cur))
        return out

    def at(c0):
        def cut(a, b):
            t = (c0 - a[axis]) / (b[axis] - a[axis]) if b[axis] != a[axis] else 0.0
            q = [0.0, 0.0]
            q[axis], q[other] = c0, a[other] + t * (b[other] - a[other])
            return tuple(q)
        return cut
    p = clip(poly, lambda q: q[axis] >= lo, at(lo))
    return clip(p, lambda q: q[axis] <= hi, at(hi)) if p else []


def _clip_u(poly, lo, hi):
    return _clip(poly, lo, hi, 0)


def grid_pieces(poly, su, sv):
    """Cut a hole-free UV polygon into cells at most su x sv in size (None = no cut in that direction): on a curved
    surface the straight triangle edges of each cell then stay within one step, so the face comes out round."""
    pieces = [poly]
    for axis, step in ((0, su), (1, sv)):
        if not step:
            continue
        out = []
        for p in pieces:
            cs = [q[axis] for q in p]
            lo, hi = min(cs), max(cs)
            n = max(1, math.ceil((hi - lo) / step - 1e-9))
            for k in range(n):
                part = _clip(p, lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n, axis)
                if len(part) >= 3 and abs(_area(part)) > 1e-14:
                    out.append(part)
        pieces = out
    return pieces


# ------------------------------------------------------------------ faces --

class Stats:
    def __init__(self):
        self.faces = self.tris = 0
        self.skipped = {}

    def skip(self, why):
        self.skipped[why] = self.skipped.get(why, 0) + 1


def face_mesh(face: dict, objs: dict, stats: Stats, edges=None):
    """-> (vertices Nx3, triangles Mx3) in feet, or None.
    The face is triangulated in UV (holes included), then triangle edges longer than the surface's step in a curved
    direction are split (refine()), so cylinders, cones and bends (tori) come out round in both directions."""
    surf = face.get("m_pSurf")
    info = surface_info(surf) if isinstance(surf, dict) else None
    if info is None:
        sc = surf.get("__class__") if isinstance(surf, dict) else None
        if sc == "SurfRev":
            sc += " profile " + str((surf.get("m_pProfileCurve") or {}).get("__class__"))
        stats.skip(f"surface {sc}")
        return None
    f, (su, sv), period = info
    if not face.get("m_pFirstLoop"):
        stats.skip("auxiliary face without loop")
        return None
    loops = face_loops(face, objs, edges, period)
    if not loops:
        stats.skip("loop walk failed")
        return None
    if period != (0.0, 0.0):
        wind = [_winding(lp, period) for lp in loops]
        rings = [i for i, w in enumerate(wind) if w != [0, 0]]
        if rings:
            axes = {0 if wind[i][0] else 1 for i in rings}
            if len(rings) == 2 and len(axes) == 1 and all(abs(sum(wind[i])) == 1 for i in rings):
                axis = axes.pop()
                outer = annulus([loops[i] for i in rings], axis, period[axis])
                loops = [outer] + [lp for i, lp in enumerate(loops) if i not in rings]
            else:
                stats.skip("loops around a periodic surface")
                return None
    P, T = [], []
    for region in loop_regions(loops):
        if len(region) == 1:
            # no holes: cut into grid cells (u and v steps), ear-clip each cell - few, well-shaped triangles
            pts, t = [], []
            for piece in grid_pieces(region[0], su, sv) if (su or sv) else [region[0]]:
                pp, tt = triangulate([piece])
                b0 = len(pts)
                pts.extend(pp)
                t.extend((a + b0, b + b0, c + b0) for a, b, c in tt)
        else:
            # holes: triangulate the region (earcut bridges the holes), then split the edges that are too long
            pts, t = triangulate(region)
            if t:
                pts, t = refine(pts, t, su, sv)
                pts, t = [tuple(p) for p in pts.tolist()], t.tolist()
        base = len(P)
        P.extend(pts)
        T.extend((a + base, b + base, c + base) for a, b, c in t)
    if not T:
        stats.skip("triangulation empty")
        return None
    verts = np.array([f(u, v) for u, v in P], float)
    stats.faces += 1
    stats.tris += len(T)
    return verts, np.asarray(T, np.uint32).reshape(-1, 3)


def _point_in(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i - 1]
        x2, y2 = poly[i]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _loop_inside(a, b):
    """Is loop a inside loop b? Majority vote of up to 9 of a's points (loops may touch b's boundary)."""
    step = max(1, len(a) // 9)
    pts = a[::step][:9]
    return sum(_point_in(p, b) for p in pts) * 2 > len(pts)


def loop_regions(loops):
    """Face loops -> regions [outer, hole, ...]. A Revit face may have several outer loops (separate regions of the
    same surface), so loops are classified by containment, not by their order: a loop inside an odd number of other
    loops is a hole of the smallest loop that contains it, every other loop is an outer loop of its own."""
    if len(loops) == 1:
        return [loops]
    n = len(loops)
    areas = [abs(_area(lp)) for lp in loops]
    contains = [[j for j in range(n) if j != i and areas[j] > areas[i] and _loop_inside(loops[i], loops[j])]
                for i in range(n)]
    regions, holes = {}, []
    for i in range(n):
        if len(contains[i]) % 2 == 0:
            regions[i] = [loops[i]]
        else:
            holes.append(i)
    for i in holes:
        owners = [j for j in contains[i] if j in regions]
        if owners:
            regions[min(owners, key=lambda j: areas[j])].append(loops[i])
    return list(regions.values())


REFINE_PASSES = 14


def refine(P, T, su, sv):
    """Conforming refinement of a UV triangulation: every edge with |du| > su or |dv| > sv (None = no limit in that
    direction) is split at its UV midpoint, for all triangles sharing it (marked-edge red/green refinement, so the
    mesh stays crack-free). Repeated until no edge is too long. -> (points Nx2, triangles Mx3)"""
    P = np.asarray(P, float).reshape(-1, 2)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    iu = 1.0 / su if su else 0.0
    iv = 1.0 / sv if sv else 0.0
    if not (iu or iv) or not len(T):
        return P, T
    for _ in range(REFINE_PASSES):
        M = len(T)
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])      # edge k of triangle i at row k*M+i
        Es = np.sort(E, axis=1)
        d = np.abs(P[Es[:, 0]] - P[Es[:, 1]])
        long_ = np.maximum(d[:, 0] * iu, d[:, 1] * iv) > 1.0 + 1e-9
        if not long_.any():
            break
        n = len(P) + 1
        key = Es[:, 0] * n + Es[:, 1]
        uk, inv = np.unique(key[long_], return_inverse=True)
        a, b = uk // n, uk % n
        mid_of = len(P) + np.arange(len(uk))
        P = np.vstack([P, 0.5 * (P[a] + P[b])])
        mid = np.full(3 * M, -1, np.int64)
        mid[np.nonzero(long_)[0]] = mid_of[inv.ravel()]
        m01, m12, m20 = mid[:M], mid[M:2 * M], mid[2 * M:]
        aff = (m01 >= 0) | (m12 >= 0) | (m20 >= 0)
        new = []
        for (v0, v1, v2), x01, x12, x20 in zip(T[aff].tolist(), m01[aff].tolist(), m12[aff].tolist(),
                                               m20[aff].tolist()):
            marks = (x01 >= 0) + (x12 >= 0) + (x20 >= 0)
            if marks == 3:
                new += [(v0, x01, x20), (x01, v1, x12), (x20, x12, v2), (x01, x12, x20)]
                continue
            tri, mids = [v0, v1, v2], [x01, x12, x20]
            for _r in range(3):              # rotate: edge 0-1 marked (and 1-2 as well when two are marked)
                if mids[0] >= 0 and (marks == 1 or mids[1] >= 0):
                    break
                tri, mids = tri[1:] + tri[:1], mids[1:] + mids[:1]
            p0, p1, p2 = tri
            q01, q12 = mids[0], mids[1]
            if marks == 1:
                new += [(p0, q01, p2), (q01, p1, p2)]
            else:
                new += [(p0, q01, p2), (q01, p1, q12), (q01, q12, p2)]
        T = np.vstack([T[~aff], np.asarray(new, np.int64).reshape(-1, 3)])
    return P, T


INVISIBLE_SOLID = 0xE0      # GInfo.m_flags bits of a B-rep solid Revit does not draw


def solid_invisible(g: dict) -> bool:
    """B-rep solid (Geometry node) that Revit does not show: GInfo.m_flags has all of 0xE0 set (0x8c0e4 / 0x880e4
    against 0x8c004 / 0x88004 of visible solids). Verified against Revit's GLB export of Golden Nugget (view
    '3D Gebaeudetechnik'): the placeholder boxes of pipe fittings, the extra bodies of light switches, outlets and
    HVAC equipment carry them; leaving them out gives Revit's surface area exactly (2026-11-11). Only Geometry
    nodes: a GGroup with these bits can hold visible solids."""
    return ((g.get("m_GInfo") or {}).get("m_flags", 0) & INVISIBLE_SOLID) == INVISIBLE_SOLID


def solid_parts(gelem: dict, objs: dict, stats: Stats) -> list:
    """The geometry of one decoded GElement (outside GInstance subtrees), one entry per B-rep solid (Geometry node)
    and material key (face m_renderStyleId, geometry-node subcategory id), plus one per GPolyMesh.
    -> [(key, vertices Nx3, triangles Mx3, is_mesh)]"""
    out = []
    for g in find_nodes(gelem, ("Geometry", "GBRep")):
        if solid_invisible(g):
            stats.skip("invisible solid (GInfo flags 0xE0)")
            continue
        faces = [f for f in g.get("m_pFaces") or [] if isinstance(f, dict) and "__tag__" in f]
        if len(faces) == 1 and (faces[0].get("m_pSurf") or {}).get("__class__") == "Plane":
            # a single planar face is a sheet, not a body: the layer boundaries of compound walls (Revit draws only
            # the wall's box; verified on Golden Nugget against the '3D Gebaeudetechnik' GLB export)
            stats.skip("single-face sheet (layer boundary)")
            continue
        node_cat = (g.get("m_GInfo") or {}).get("m_categoryId", -1)
        edges = [e for e in g.get("m_pEdges") or [] if isinstance(e, dict)]
        groups: dict = {}
        for face in g.get("m_pFaces") or []:
            if not isinstance(face, dict) or "__tag__" not in face:
                continue
            m = face_mesh(face, objs, stats, edges)
            if m is None:
                continue
            key = (face.get("m_renderStyleId", -1), node_cat)
            groups.setdefault(key, []).append(m)
        out.extend((k, *_merge_parts(v), False) for k, v in groups.items())
    for pm in find_nodes(gelem, ("GPolyMesh",)):
        m = polymesh_part(pm)
        if m is not None:
            key = (pm.get("m_materialID", -1), (pm.get("m_GInfo") or {}).get("m_categoryId", -1))
            out.append((key, m[0], m[1], True))
    return out


def element_parts(gelem: dict, objs: dict, stats: Stats) -> dict:
    """All B-rep faces of one decoded GElement (outside GInstance subtrees), grouped by
    material key (face m_renderStyleId, geometry-node subcategory id).
    -> {key: (vertices Nx3, triangles Mx3)}"""
    groups: dict = {}
    for key, v, t, _mesh in solid_parts(gelem, objs, stats):
        groups.setdefault(key, []).append((v, t))
    return {k: _merge_parts(v) for k, v in groups.items()}


def polymesh_part(pm: dict):
    """GPolyMesh (mesh geometry, e.g. imported parts of a family): m_pFacetedTopology.m_pointsArr (family/model
    coordinates, feet) + m_facetsArr triangles with 1-based point indices (Golden Nugget lighting fixtures)."""
    ft = pm.get("m_pFacetedTopology") or {}
    pts, fac = ft.get("m_pointsArr"), ft.get("m_facetsArr")
    if not pts or not fac:
        return None
    try:
        v = np.asarray(pts, float).reshape(-1, 3)
        t = np.asarray([f[:3] for f in fac if isinstance(f, (list, tuple)) and len(f) >= 3], np.int64)
    except (TypeError, ValueError):
        return None
    if not len(t):
        return None
    if t.min() >= 1 and t.max() == len(v):          # 1-based (the form seen in the samples)
        t = t - 1
    if t.min() < 0 or t.max() >= len(v):
        return None
    return v, t


def _merge_parts(parts):
    vs, ts, base = [], [], 0
    for v, t in parts:
        vs.append(v)
        ts.append(t + base)
        base += len(v)
    return np.vstack(vs), np.vstack(ts)


def element_mesh(gelem: dict, objs: dict, stats: Stats):
    """All faces merged into one mesh (no materials)."""
    parts = element_parts(gelem, objs, stats)
    return _merge_parts(list(parts.values())) if parts else None
