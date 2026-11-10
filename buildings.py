#!/usr/bin/env python3
"""TessyNav building tiles builder v2 — buildings from Overture Maps (OpenStreetMap + Microsoft + Google + others),
v2: heights from OpenStreetMap where Overture has none (towers drawn as low blocks, like Pointe-Nord on Nuns' Island).

Overture keeps every OpenStreetMap building and fills the gaps with buildings detected from satellite imagery,
so areas that are nearly empty in OpenStreetMap (many suburbs) get their buildings. Runs on GitHub Actions
(see .github/workflows/buildings.yml) and publishes with GitHub Pages at https://tessynav.github.io/tessynav-buildings/

  python3 buildings.py download   download Overture's buildings for each region in regions.json
  python3 buildings.py osm        read the heights and floor counts of OpenStreetMap's buildings (osm.pbf)
  python3 buildings.py build      write the tiles into out/

Output:
  out/index.json          version, date, regions covered (inside a region, a missing pack means "no buildings")
  out/b/<px>_<py>.json    one pack = 4 x 4 cells of 0.005 degrees, the same grid as the road tiles
     {"v":1, "o":[west, south], "b":[[h, base, x0, y0, dx1, dy1, ...], ...]}
       h, base   height and base height in decimetres (base > 0: a part raised above the ground)
       x, y      the outline in millionths of a degree from the pack's corner, the first point absolute and the
                 others as steps from the previous one (small numbers, small files); no repeated last point
"""
import json, math, os, subprocess, sys, time

CELL = 0.005          # same cell size as the road tiles (TN.CELL in the app)
PACK = 4              # cells per pack side
Q = 1e6               # millionths of a degree (about 8 cm east-west, 11 cm north-south in Quebec)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')


def pack_of(lng, lat):
    return math.floor(math.floor(lng / CELL) / PACK), math.floor(math.floor(lat / CELL) / PACK)


def known_height(h, floors):
    """a real height in metres (Overture's height, or its number of floors), or None when it has neither"""
    if h is not None and h == h and 2 <= h <= 600:
        return float(h)
    if floors is not None and floors == floors and 1 <= floors <= 150:
        return 3.2 * floors + 1.0
    return None


def guess_height(area):
    return 6.0 if area < 120 else 8.0 if area < 400 else 10.0 if area < 2500 else 12.0


def height_of(h, floors, area):
    """the building's height in metres: Overture's height, or its number of floors, or a guess from its size"""
    k = known_height(h, floors)
    return k if k is not None else guess_height(area)


def parse_height(v):
    """an OpenStreetMap height tag in metres: "45", "45 m", "45.5", "150 ft", "150'" (None if unreadable)"""
    if not v:
        return None
    t = v.strip().lower().replace(',', '.')
    feet = t.endswith('ft') or t.endswith("'") or 'feet' in t
    num = ''
    for ch in t:
        if ch.isdigit() or ch == '.':
            num += ch
        elif num:
            break
    try:
        x = float(num)
    except ValueError:
        return None
    if feet:
        x *= 0.3048
    return x if 2 <= x <= 600 else None


OSMG = 0.001          # grid of the OpenStreetMap heights: about 80 x 110 m


def osm_heights(pbf):
    """OpenStreetMap buildings (and building parts) that have a height or a number of floors: a point inside each
    with its height, on a small grid, saved to data/osm_heights.json"""
    import osmium

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.grid, self.n = {}, 0

        def area(self, a):
            t = a.tags
            if 'building' not in t and 'building:part' not in t:
                return
            h = parse_height(t.get('height'))
            if h is None:
                lv = t.get('building:levels')
                try:
                    lv = float(lv) if lv is not None else None
                except ValueError:
                    lv = None
                if lv is not None and 1 <= lv <= 150:
                    h = 3.2 * lv + 1.0
            if h is None:
                return
            try:
                for ring in a.outer_rings():
                    pts = [(nd.lon, nd.lat) for nd in ring]
                    if len(pts) < 3:
                        continue
                    x = sum(p[0] for p in pts) / len(pts); y = sum(p[1] for p in pts) / len(pts)
                    k = '%d_%d' % (math.floor(x / OSMG), math.floor(y / OSMG))
                    self.grid.setdefault(k, []).append([round(x, 7), round(y, 7), round(h, 1)])
                    self.n += 1
            except Exception:
                pass

    h = H()
    h.apply_file(pbf, locations=True)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, 'osm_heights.json'), 'w') as fh:
        json.dump(h.grid, fh)
    print('OpenStreetMap buildings with a height or floors: %d' % h.n, flush=True)


def osm_lookup(grid, poly):
    """the tallest OpenStreetMap height whose building's middle lies inside this footprint (None if none)"""
    import shapely
    if not grid:
        return None
    x0, y0, x1, y1 = poly.bounds
    best = None
    for gx in range(math.floor(x0 / OSMG), math.floor(x1 / OSMG) + 1):
        for gy in range(math.floor(y0 / OSMG), math.floor(y1 / OSMG) + 1):
            for x, y, h in grid.get('%d_%d' % (gx, gy), ()):
                if x0 <= x <= x1 and y0 <= y <= y1 and shapely.contains_xy(poly, x, y):
                    if best is None or h > best:
                        best = h
    return best


def ring_area_m2(ring, lat):
    kx = 111320 * math.cos(math.radians(lat)); ky = 110540
    a = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0] * kx, ring[i][1] * ky
        x2, y2 = ring[i + 1][0] * kx, ring[i + 1][1] * ky
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def download(cfg):
    os.makedirs(DATA, exist_ok=True)
    for reg in cfg['regions']:
        w, s, e, n = reg['bbox']
        out = os.path.join(DATA, reg['name'].replace(' ', '_') + '.parquet')
        print('downloading', reg['name'], reg['bbox'], flush=True)
        t0 = time.time()
        subprocess.run(['overturemaps', 'download', '--bbox=%f,%f,%f,%f' % (w, s, e, n), '-f', 'geoparquet',
                        '--type=building', '-o', out], check=True)
        print('  done in %.0f s, %.0f MB' % (time.time() - t0, os.path.getsize(out) / 1e6), flush=True)


def build(cfg):
    import pyarrow.parquet as pq
    import shapely
    os.makedirs(os.path.join(OUT, 'b'), exist_ok=True)
    tmp = os.path.join(DATA, 'packs'); os.makedirs(tmp, exist_ok=True)
    total, kept, buf, nbuf = 0, 0, {}, 0
    # OpenStreetMap heights (step "osm"), used where Overture has none
    grid = {}
    gp = os.path.join(DATA, 'osm_heights.json')
    if os.path.exists(gp):
        with open(gp) as fh:
            grid = json.load(fh)
    from_osm = 0

    def flush():
        nonlocal buf, nbuf
        for k, rows in buf.items():
            with open(os.path.join(tmp, k + '.jsonl'), 'a') as fh:
                for r in rows:
                    fh.write(r + '\n')
        buf, nbuf = {}, 0

    for reg in cfg['regions']:
        path = os.path.join(DATA, reg['name'].replace(' ', '_') + '.parquet')
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        cols = [c for c in ('geometry', 'height', 'num_floors', 'min_height', 'min_floor') if c in names]
        W, S, E, N = reg['bbox']
        for batch in pf.iter_batches(batch_size=20000, columns=cols):
            d = batch.to_pydict()
            geoms = shapely.from_wkb(d['geometry'])
            for i, g in enumerate(geoms):
                total += 1
                if g is None or g.is_empty:
                    continue
                polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g] if g.geom_type == 'Polygon' else []
                for p in polys:
                    ring = list(p.exterior.coords)
                    if len(ring) < 4:
                        continue
                    c = p.representative_point()
                    if not (W <= c.x < E and S <= c.y < N):
                        continue          # (each building belongs to exactly one region: no doubles at the edges)
                    area = ring_area_m2(ring, c.y)
                    if area < 12:
                        continue          # sheds and noise
                    hm = known_height(d.get('height', [None] * len(geoms))[i], d.get('num_floors', [None] * len(geoms))[i])
                    if hm is None:
                        hm = osm_lookup(grid, p)
                        if hm is not None:
                            from_osm += 1
                        else:
                            hm = guess_height(area)
                    mh = d.get('min_height', [None] * len(geoms))[i]
                    mf = d.get('min_floor', [None] * len(geoms))[i]
                    base = float(mh) if mh is not None and mh == mh and 0 < mh < hm else (3.2 * mf if mf is not None and mf == mf and mf > 0 and 3.2 * mf < hm else 0.0)
                    px, py = pack_of(c.x, c.y)
                    ox, oy = px * PACK * CELL, py * PACK * CELL
                    pts = ring[:-1]
                    q = [(round((x - ox) * Q), round((y - oy) * Q)) for x, y in pts]
                    # drop points that land on the same spot after rounding
                    qq = [q[0]]
                    for t in q[1:]:
                        if t != qq[-1]:
                            qq.append(t)
                    if len(qq) < 3:
                        continue
                    flat = [round(hm * 10), round(base * 10), qq[0][0], qq[0][1]]
                    for j in range(1, len(qq)):
                        flat += [qq[j][0] - qq[j - 1][0], qq[j][1] - qq[j - 1][1]]
                    key = '%d_%d' % (px, py)
                    buf.setdefault(key, []).append(json.dumps(flat, separators=(',', ':')))
                    nbuf += 1; kept += 1
            if nbuf > 300000:
                flush()
            print('  %s: %d read, %d kept' % (reg['name'], total, kept), flush=True)
    flush()
    # one file per pack
    npk, size = 0, 0
    for f in os.listdir(tmp):
        key = f[:-6]
        px, py = map(int, key.split('_'))
        with open(os.path.join(tmp, f)) as fh:
            rows = fh.read().strip().split('\n')
        body = '{"v":1,"o":[%s,%s],"b":[%s]}' % (repr(round(px * PACK * CELL, 6)), repr(round(py * PACK * CELL, 6)), ','.join(rows))
        with open(os.path.join(OUT, 'b', key + '.json'), 'w') as out:
            out.write(body)
        npk += 1; size += len(body)
    idx = {'v': 1, 'date': time.strftime('%Y-%m-%d'), 'cell': CELL, 'pack': PACK, 'q': Q,
           'regions': [{'name': r['name'], 'bbox': r['bbox']} for r in cfg['regions']],
           'buildings': kept, 'packs': npk, 'source': 'Overture Maps Foundation (ODbL / CDLA Permissive 2.0)'}
    with open(os.path.join(OUT, 'index.json'), 'w') as fh:
        json.dump(idx, fh)
    print('buildings %d of %d, packs %d, %.0f MB; heights from OpenStreetMap where Overture had none: %d' % (kept, total, npk, size / 1e6, from_osm), flush=True)


if __name__ == '__main__':
    cfg = json.load(open(os.path.join(HERE, 'regions.json')))
    step = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if step == 'download':
        download(cfg)
    elif step == 'osm':
        osm_heights(sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, 'osm-heights.osm.pbf'))
    elif step == 'build':
        build(cfg)
    else:
        sys.exit('use: buildings.py download | osm [file.osm.pbf] | build')
