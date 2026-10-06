"""Pie-chart map with three zoom levels (cells -> states -> countries), switched client side via deck.gl min/max zoom."""
import numpy as np
import pandas as pd
import json

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def default_colors(managements):
    return {m: list(PALETTE[i % len(PALETTE)]) + [220] for i, m in enumerate(managements)}


def _pie_polygons(lon, lat, shares, radius_km):
    total = float(np.sum(shares))
    if total <= 0:
        return []
    lon_factor = np.cos(np.deg2rad(lat))
    start, out = 0.0, []
    for share in shares:
        frac = share / total
        if frac <= 0:
            out.append(None)
            continue
        end = start + frac * 360
        angles = np.deg2rad(np.linspace(start, end, max(2, int(frac * 40))))
        pts = [[lon, lat]] + [[lon + radius_km / 111 * np.cos(a) / lon_factor,
                               lat + radius_km / 111 * np.sin(a)] for a in angles] + [[lon, lat]]
        out.append(pts)
        start = end
    return out


def _radar_polygon(lon, lat, values, radius_km):
    lon_factor = np.cos(np.deg2rad(lat))
    angles = np.deg2rad(np.linspace(0, 360, len(values), endpoint=False))
    pts = [[lon + radius_km * v / 111 * np.cos(a) / lon_factor, lat + radius_km * v / 111 * np.sin(a)]
           for v, a in zip(values, angles)]
    return pts + [pts[0]]


def _records_for_level(df, managements, colors, es_list, radius_km, show_radar):
    slices, radars = [], []
    for _, row in df.iterrows():
        shares = row[managements].to_numpy(dtype=float)
        info = f"{row['name']} ({int(row['n_cells'])} cells)" if "n_cells" in row else row["name"]
        for m, share, poly in zip(managements, shares, _pie_polygons(row["lon"], row["lat"], shares, radius_km)):
            if poly is None:
                continue
            slices.append({"polygon": poly, "name": info, "slice": m, "value": f"{share:.1%}", "color": colors[m]})
        if show_radar:
            radars.append({"polygon": _radar_polygon(row["lon"], row["lat"], row[es_list].to_numpy(dtype=float), radius_km),
                           "name": info, "slice": "ES scores (" + ", ".join(es_list) + ")",
                           "value": ", ".join(f"{v:.2f}" for v in row[es_list])})
    return {"slices": slices, "radars": radars}


_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet"/>
<style>html,body{margin:0;height:100%;} #map{position:absolute;inset:0;}
#zoom{position:absolute;left:8px;bottom:8px;z-index:5;background:#fffc;padding:2px 6px;font:12px sans-serif;border-radius:3px;}</style>
</head><body><div id="map"></div><div id="zoom"></div><script>
const LEVELS = __LEVELS__;           // [{name, minZoom, maxZoom, slices, radars}]
const INITIAL = __VIEW__;
function makeLayers(zoom) {
  const layers = [];
  for (const lv of LEVELS) {
    const visible = zoom >= lv.minZoom && zoom < lv.maxZoom;
    if (!visible) continue;
    layers.push(new deck.PolygonLayer({id: 'pie_' + lv.name, data: lv.slices, getPolygon: d => d.polygon,
      getFillColor: d => d.color, pickable: true, autoHighlight: true}));
    if (lv.radars.length) layers.push(new deck.PolygonLayer({id: 'radar_' + lv.name, data: lv.radars,
      getPolygon: d => d.polygon, filled: false, stroked: true, getLineColor: [0,0,0], lineWidthUnits: 'pixels',
      getLineWidth: 1.5, pickable: true}));
  }
  return layers;
}
const zoomLabel = document.getElementById('zoom');
function show(z) { zoomLabel.textContent = 'zoom ' + z.toFixed(1); }
show(INITIAL.zoom);
const d = new deck.DeckGL({
  container: 'map', mapLib: maplibregl,
  mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  initialViewState: INITIAL, controller: true, layers: makeLayers(INITIAL.zoom),
  onViewStateChange: ({viewState}) => { show(viewState.zoom); d.setProps({layers: makeLayers(viewState.zoom)}); },
  getTooltip: ({object}) => object && (object.name + '\\n' + object.slice + ': ' + object.value),
});
</script></body></html>"""


def multilevel_html(cells, states, countries, managements, colors, es_list, cell_radius_km,
                    show_radar=True, zoom_cells=6.0, zoom_states=4.0):
    """Standalone deck.gl page (for st.components.v1.html). Zoom switching happens in the browser:
    cells at zoom >= zoom_cells, states in [zoom_states, zoom_cells), countries below zoom_states."""
    levels = [
        ("cell", zoom_cells, 30, cells, cell_radius_km),
        ("state", zoom_states, zoom_cells, states, 60),
        ("country", 0, zoom_states, countries, 250),
    ]
    payload = []
    for name, lo, hi, df, radius in levels:
        payload.append({"name": name, "minZoom": lo, "maxZoom": hi,
                        **_records_for_level(df, managements, colors, es_list, radius, show_radar)})
    view = {"latitude": float(cells["lat"].mean()), "longitude": float(cells["lon"].mean()),
            "zoom": zoom_cells + 1, "pitch": 0, "bearing": 0}
    return _HTML.replace("__LEVELS__", json.dumps(payload)).replace("__VIEW__", json.dumps(view))
