"""fig1_icons: vector icons for Figure 1 (v2) as matplotlib Paths on a 24 x 24 grid (y pointing down).

Two sources, drawn in one consistent line style by fig1_v2.py:
  * Lucide icons (lucide-static@0.469.0, ISC licence; portions MIT from Feather), downloaded once from
    https://cdn.jsdelivr.net/npm/lucide-static@0.469.0/icons/<name>.svg into
    figures/assets/lucide-static-0.469.0/ (LICENSE and package.json kept alongside).
    The SVG files are only parsed as XML data here; nothing downloaded is executed.
  * Custom icons drawn for this figure on the same 24-unit grid (porous carbon particle,
    Ni-loaded particle, fixed-bed reactor tube) so that they match Lucide's geometry.
SVG path data are converted with svgpath2mpl (BSD licence, PyPI).
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from svgpath2mpl import parse_path

LUCIDE_VERSION = "0.469.0"
LUCIDE_DIR = Path(__file__).resolve().parent / "assets" / f"lucide-static-{LUCIDE_VERSION}" / "icons"
_SVG_NS = "{http://www.w3.org/2000/svg}"


def _f(v):
    return float(v) if v not in (None, "") else 0.0


def _circle_d(cx, cy, r):
    return f"M{cx - r},{cy}A{r},{r} 0 1 0 {cx + r},{cy}A{r},{r} 0 1 0 {cx - r},{cy}Z"


def _ellipse_d(cx, cy, rx, ry):
    return f"M{cx - rx},{cy}A{rx},{ry} 0 1 0 {cx + rx},{cy}A{rx},{ry} 0 1 0 {cx - rx},{cy}Z"


def _rect_d(x, y, w, h, rx=0.0, ry=None):
    ry = rx if ry is None else ry
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    if rx <= 0:
        return f"M{x},{y}H{x + w}V{y + h}H{x}Z"
    return (f"M{x + rx},{y}H{x + w - rx}A{rx},{ry} 0 0 1 {x + w},{y + ry}V{y + h - ry}"
            f"A{rx},{ry} 0 0 1 {x + w - rx},{y + h}H{x + rx}A{rx},{ry} 0 0 1 {x},{y + h - ry}"
            f"V{y + ry}A{rx},{ry} 0 0 1 {x + rx},{y}Z")


def _points_d(points, close):
    nums = [float(v) for v in points.replace(",", " ").split()]
    pts = list(zip(nums[0::2], nums[1::2]))
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    return d + ("Z" if close else "")


def element_d(tag, a):
    """SVG element (tag without namespace, attribute dict) -> path data string, or None."""
    if tag == "path":
        return a.get("d")
    if tag == "circle":
        return _circle_d(_f(a.get("cx")), _f(a.get("cy")), _f(a.get("r")))
    if tag == "ellipse":
        return _ellipse_d(_f(a.get("cx")), _f(a.get("cy")), _f(a.get("rx")), _f(a.get("ry")))
    if tag == "rect":
        rx = a.get("rx"); ry = a.get("ry")
        rx = _f(rx) if rx is not None else (_f(ry) if ry is not None else 0.0)
        ry = _f(ry) if ry is not None else rx
        return _rect_d(_f(a.get("x")), _f(a.get("y")), _f(a.get("width")), _f(a.get("height")), rx, ry)
    if tag == "line":
        return f"M{_f(a.get('x1'))},{_f(a.get('y1'))}L{_f(a.get('x2'))},{_f(a.get('y2'))}"
    if tag in ("polyline", "polygon"):
        return _points_d(a.get("points", ""), tag == "polygon")
    return None


def lucide(name):
    """Return [(Path, filled=False), ...] for a Lucide icon (24 x 24, stroke-only)."""
    root = ET.parse(LUCIDE_DIR / f"{name}.svg").getroot()
    out = []
    for el in root.iter():
        tag = el.tag.replace(_SVG_NS, "")
        d = element_d(tag, el.attrib)
        if d:
            out.append((parse_path(d), False))
    if not out:
        raise ValueError(f"no drawable elements in lucide icon {name!r}")
    return out


# ---------------- custom icons on the same 24-unit grid ----------------
# each entry: (svg path data, filled?)  -- filled shapes take the icon colour as fill, no stroke
_PORES = [(7.6, 9.4, 1.55), (13.6, 7.2, 1.25), (16.2, 12.4, 1.7), (11.4, 12.6, 1.05),
          (8.4, 15.6, 1.35), (13.4, 17.0, 1.2)]

CUSTOM = {
    # activated carbon particle: outline with open pores
    "porous_particle": [(_circle_d(12, 12, 9.2), False)] + [(_circle_d(x, y, r), False) for x, y, r in _PORES],
    # Ni impregnation: drop of Ni-salt solution above a porous particle; Ni sites as filled dots
    "ni_particle": [
        ("M12 1.2C12 1.2 9.6 4.1 9.6 5.7A2.4 2.4 0 0 0 14.4 5.7C14.4 4.1 12 1.2 12 1.2Z", False),
        (_circle_d(12, 15.6, 7.2), False),
    ] + [(_circle_d(x, y, 0.95), True) for x, y in [(9.0, 13.2), (13.6, 12.0), (15.6, 15.8),
                                                    (11.4, 16.4), (8.6, 18.0), (13.6, 19.6)]],
    # fixed-bed reactor: vertical tube between the two halves of a furnace, catalyst bed (filled) in the
    # hot zone; gas enters at the top and products leave at the bottom (chevrons give the flow direction)
    "fixed_bed": [
        ("M10.2 0.8V6.4M13.8 0.8V6.4", False),                           # tube above the furnace
        ("M10.2 17.6V23.2M13.8 17.6V23.2", False),                       # tube below the furnace
        (_rect_d(5.0, 6.4, 14.0, 11.2, 1.6), False),                    # furnace
        ("M10.2 6.4V17.6M13.8 6.4V17.6", False),                         # tube inside the furnace
        (_rect_d(10.2, 10.0, 3.6, 4.0), True),                           # catalyst bed
    ],
}


def custom(name):
    return [(parse_path(d), filled) for d, filled in CUSTOM[name]]


def icon_paths(name):
    return custom(name) if name in CUSTOM else lucide(name)
