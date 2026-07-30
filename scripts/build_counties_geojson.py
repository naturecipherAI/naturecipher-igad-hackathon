"""
Build dashboard/kenya-counties.geojson from geoBoundaries KEN ADM1.

Kenya's ADM1 units are the 47 counties. Source is public domain (RCMRD via
geoBoundaries). The raw release is ~8 MB, too heavy for a static dashboard, so
this simplifies with Douglas-Peucker and rounds coordinates before writing.

All 47 counties are kept. The 11 inside a forecast region carry region_id; the
other 36 carry null and render as uncovered, which shows the coverage gap
honestly instead of cropping it out of frame.

Source (not committed, ~8 MB):
    curl -L -o ken_adm1.geojson \
      https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson

Usage:
    python scripts/build_counties_geojson.py ken_adm1.geojson
"""

import json
import sys
from pathlib import Path

import yaml

TOLERANCE_DEG = 0.008
COORD_PRECISION = 4

# geoBoundaries labels some counties with a shortened name. Map the source's name
# onto the project's, rather than editing regions.yaml to match a source quirk.
NAME_ALIASES = {
    "tharaka": "tharaka nithi",
}


def _perp_distance(pt, start, end):
    if start == end:
        return ((pt[0] - start[0]) ** 2 + (pt[1] - start[1]) ** 2) ** 0.5
    dx, dy = end[0] - start[0], end[1] - start[1]
    num = abs(dy * pt[0] - dx * pt[1] + end[0] * start[1] - end[1] * start[0])
    return num / ((dx * dx + dy * dy) ** 0.5)


def simplify(points, tolerance):
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, index = 0.0, first
        for i in range(first + 1, last):
            d = _perp_distance(points[i], points[first], points[last])
            if d > max_dist:
                max_dist, index = d, i
        if max_dist > tolerance:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(points, keep) if k]


def round_ring(ring):
    return [[round(x, COORD_PRECISION), round(y, COORD_PRECISION)] for x, y in ring]


def process_ring(ring):
    simplified = simplify([tuple(c[:2]) for c in ring], TOLERANCE_DEG)
    # A polygon ring needs at least 4 positions with the first repeated last.
    if len(simplified) < 4:
        simplified = [tuple(c[:2]) for c in ring][:4]
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return round_ring(simplified)


def process_geometry(geom):
    if geom["type"] == "Polygon":
        rings = [process_ring(r) for r in geom["coordinates"]]
        return {"type": "Polygon", "coordinates": rings}
    if geom["type"] == "MultiPolygon":
        polys = []
        for poly in geom["coordinates"]:
            rings = [process_ring(r) for r in poly]
            area = abs(sum(
                rings[0][i][0] * rings[0][i + 1][1] - rings[0][i + 1][0] * rings[0][i][1]
                for i in range(len(rings[0]) - 1)
            )) / 2
            # Drop slivers left behind by simplification, keep real islands.
            if area > 0.0005:
                polys.append(rings)
        if not polys:
            polys = [[process_ring(geom["coordinates"][0][0])]]
        return {"type": "MultiPolygon", "coordinates": polys}
    raise ValueError("unexpected geometry type: " + geom["type"])


def normalise(name):
    key = name.lower().replace("-", " ").replace("'", "").strip()
    return NAME_ALIASES.get(key, key)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    regions = yaml.safe_load((root / "config/regions.yaml").read_text(encoding="utf-8"))["regions"]

    county_to_region = {}
    for region_id, cfg in regions.items():
        for county in cfg["counties"]:
            county_to_region[normalise(county)] = region_id

    src = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    features, matched = [], set()
    for feat in src["features"]:
        name = feat["properties"]["shapeName"]
        region_id = county_to_region.get(normalise(name))
        if region_id:
            matched.add(normalise(name))
        features.append({
            "type": "Feature",
            "properties": {"county": name, "region_id": region_id},
            "geometry": process_geometry(feat["geometry"]),
        })

    missing = set(county_to_region) - matched
    if missing:
        raise SystemExit(f"counties in regions.yaml with no boundary match: {sorted(missing)}")

    out = {
        "type": "FeatureCollection",
        "attribution": "geoBoundaries gbOpen KEN ADM1, source RCMRD, public domain",
        "features": features,
    }
    dest = root / "dashboard/kenya-counties.geojson"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    covered = sum(1 for f in features if f["properties"]["region_id"])
    print(f"wrote {dest}")
    print(f"  counties {len(features)}, covered {covered}, uncovered {len(features) - covered}")
    print(f"  size {dest.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
