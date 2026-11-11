"""Polygon triangulation with holes: Python port of mapbox/earcut (ISC licence,
https://github.com/mapbox/earcut) without the z-order hashing.

earcut(rings) -> list of (i, j, k) indices into the flattened point list of all rings.
"""
from __future__ import annotations

import math


class _Node:
    __slots__ = ("i", "x", "y", "prev", "next", "steiner")

    def __init__(self, i, x, y):
        self.i, self.x, self.y = i, x, y
        self.prev = self.next = None
        self.steiner = False


def earcut(rings):
    pts = [p for r in rings for p in r]
    data = [c for p in pts for c in p[:2]]
    hole_idx, n = [], len(rings[0])
    for r in rings[1:]:
        hole_idx.append(n)
        n += len(r)
    outer_len = len(rings[0]) * 2
    tris: list = []
    outer = _linked_list(data, 0, outer_len, True)
    if outer is None or outer.next is outer.prev:
        return tris
    if hole_idx:
        outer = _eliminate_holes(data, hole_idx, outer)
    _earcut_linked(outer, tris, 0)
    return tris


def _linked_list(data, start, end, clockwise):
    last = None
    if clockwise == (_signed_area(data, start, end) > 0):
        for i in range(start, end, 2):
            last = _insert(i // 2, data[i], data[i + 1], last)
    else:
        for i in range(end - 2, start - 1, -2):
            last = _insert(i // 2, data[i], data[i + 1], last)
    if last is not None and _equals(last, last.next):
        _remove(last)
        last = last.next
    return last


def _filter_points(start, end=None):
    if start is None:
        return start
    end = end or start
    p = start
    while True:
        again = False
        if not p.steiner and (_equals(p, p.next) or _area(p.prev, p, p.next) == 0):
            _remove(p)
            p = end = p.prev
            if p is p.next:
                break
            again = True
        else:
            p = p.next
        if not (again or p is not end):
            break
    return end


def _earcut_linked(ear, tris, pas):
    if ear is None:
        return
    stop = ear
    while ear.prev is not ear.next:
        prev, nxt = ear.prev, ear.next
        if _is_ear(ear):
            tris.append((prev.i, ear.i, nxt.i))
            _remove(ear)
            ear = nxt.next
            stop = nxt.next
            continue
        ear = nxt
        if ear is stop:
            if pas == 0:
                _earcut_linked(_filter_points(ear), tris, 1)
            elif pas == 1:
                ear = _cure_local_intersections(_filter_points(ear), tris)
                _earcut_linked(ear, tris, 2)
            elif pas == 2:
                _split_earcut(ear, tris)
            break


def _is_ear(ear):
    a, b, c = ear.prev, ear, ear.next
    if _area(a, b, c) >= 0:
        return False
    x0, x1 = min(a.x, b.x, c.x), max(a.x, b.x, c.x)
    y0, y1 = min(a.y, b.y, c.y), max(a.y, b.y, c.y)
    p = c.next
    while p is not a:
        if (x0 <= p.x <= x1 and y0 <= p.y <= y1
                and _point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y)
                and _area(p.prev, p, p.next) >= 0):
            return False
        p = p.next
    return True


def _cure_local_intersections(start, tris):
    p = start
    while True:
        a, b = p.prev, p.next.next
        if (not _equals(a, b) and _intersects(a, p, p.next, b)
                and _locally_inside(a, b) and _locally_inside(b, a)):
            tris.append((a.i, p.i, b.i))
            _remove(p)
            _remove(p.next)
            p = start = b
        p = p.next
        if p is start:
            break
    return _filter_points(p)


def _split_earcut(start, tris):
    a = start
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                c = _split_polygon(a, b)
                a = _filter_points(a, a.next)
                c = _filter_points(c, c.next)
                _earcut_linked(a, tris, 0)
                _earcut_linked(c, tris, 0)
                return
            b = b.next
        a = a.next
        if a is start:
            break


def _eliminate_holes(data, hole_idx, outer):
    queue = []
    for k, h in enumerate(hole_idx):
        start = h * 2
        end = hole_idx[k + 1] * 2 if k + 1 < len(hole_idx) else len(data)
        lst = _linked_list(data, start, end, False)
        if lst is None:
            continue
        if lst is lst.next:
            lst.steiner = True
        queue.append(_get_leftmost(lst))
    queue.sort(key=lambda nd: nd.x)
    for h in queue:
        outer = _eliminate_hole(h, outer)
    return outer


def _eliminate_hole(hole, outer):
    bridge = _find_hole_bridge(hole, outer)
    if bridge is None:
        return outer
    bridge_rev = _split_polygon(bridge, hole)
    _filter_points(bridge_rev, bridge_rev.next)
    return _filter_points(bridge, bridge.next)


def _find_hole_bridge(hole, outer):
    p = outer
    hx, hy = hole.x, hole.y
    qx = -math.inf
    m = None
    while True:
        if hy <= p.y and hy >= p.next.y and p.next.y != p.y:
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)
            if hx >= x > qx:
                qx = x
                m = p if p.x < p.next.x else p.next
                if x == hx:
                    return m
        p = p.next
        if p is outer:
            break
    if m is None:
        return None
    stop, mx, my = m, m.x, m.y
    tan_min = math.inf
    p = m
    while True:
        if (hx >= p.x >= mx and hx != p.x
                and _point_in_triangle(hx if hy < my else qx, hy, mx, my, qx if hy < my else hx, hy, p.x, p.y)):
            tan = abs(hy - p.y) / (hx - p.x)
            if _locally_inside(p, hole) and (tan < tan_min or (tan == tan_min and (
                    p.x > m.x or (p.x == m.x and _sector_contains_sector(m, p))))):
                m, tan_min = p, tan
        p = p.next
        if p is stop:
            break
    return m


def _sector_contains_sector(m, p):
    return _area(m.prev, m, p.prev) < 0 and _area(p.next, m, m.next) < 0


def _get_leftmost(start):
    p = left = start
    while True:
        if p.x < left.x or (p.x == left.x and p.y < left.y):
            left = p
        p = p.next
        if p is start:
            break
    return left


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) >= (ax - px) * (cy - py)
            and (ax - px) * (by - py) >= (bx - px) * (ay - py)
            and (bx - px) * (cy - py) >= (cx - px) * (by - py))


def _is_valid_diagonal(a, b):
    return (a.next.i != b.i and a.prev.i != b.i and not _intersects_polygon(a, b)
            and ((_locally_inside(a, b) and _locally_inside(b, a) and _middle_inside(a, b)
                  and (_area(a.prev, a, b.prev) or _area(a, b.prev, b)))
                 or (_equals(a, b) and _area(a.prev, a, a.next) > 0 and _area(b.prev, b, b.next) > 0)))


def _area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _equals(p1, p2):
    return p1.x == p2.x and p1.y == p2.y


def _sign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _on_segment(p, q, r):
    return min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.y, r.y) <= q.y <= max(p.y, r.y)


def _intersects(p1, q1, p2, q2):
    o1, o2 = _sign(_area(p1, q1, p2)), _sign(_area(p1, q1, q2))
    o3, o4 = _sign(_area(p2, q2, p1)), _sign(_area(p2, q2, q1))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _intersects_polygon(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i
                and _intersects(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            break
    return False


def _locally_inside(a, b):
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    p, inside = a, False
    px, py = (a.x + b.x) / 2, (a.y + b.y) / 2
    while True:
        if ((p.y > py) != (p.next.y > py) and p.next.y != p.y
                and px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            inside = not inside
        p = p.next
        if p is a:
            break
    return inside


def _split_polygon(a, b):
    a2, b2 = _Node(a.i, a.x, a.y), _Node(b.i, b.x, b.y)
    an, bp = a.next, b.prev
    a.next, b.prev = b, a
    a2.next, an.prev = an, a2
    b2.next, a2.prev = a2, b2
    bp.next, b2.prev = b2, bp
    return b2


def _insert(i, x, y, last):
    p = _Node(i, x, y)
    if last is None:
        p.prev = p.next = p
    else:
        p.next, p.prev = last.next, last
        last.next.prev = p
        last.next = p
    return p


def _remove(p):
    p.next.prev = p.prev
    p.prev.next = p.next


def _signed_area(data, start, end):
    s, j = 0.0, end - 2
    for i in range(start, end, 2):
        s += (data[j] - data[i]) * (data[i + 1] + data[j + 1])
        j = i
    return s
