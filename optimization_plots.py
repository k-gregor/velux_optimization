import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import cartopy.crs as ccrs
from cartopy.io import shapereader
countries_polygons = gpd.read_file(shapereader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces'))

import colorsys


base_colors = {
    "spbau": (0/255, 0/255, 255/255),
    "tobd": (0/255, 128/255, 0/255),
    "tone": (255/255, 165/255, 0/255),
    "spbau_stop": (128/255, 0/255, 128/255)
}

# (value_factor, saturation_factor)
intensity = {
    "longrot": (1.0, 0.5),   # lighter → less saturated
    "manbau": (1.0, 1.0),    # base
    "shortrot": (0.7, 1.0)   # darker → lower value
}

def adjust_color(rgb, v_factor, s_factor):
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    v = min(v * v_factor, 1.0)
    s = min(s * s_factor, 1.0)

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), 220)

def get_color(name):
    if name == "spbau_stop":
        r, g, b = base_colors["spbau_stop"]
        return (int(r*255), int(g*255), int(b*255), 220)

    system, management = name.split("_")
    base = base_colors[system]
    v_factor, s_factor = intensity[management]

    return adjust_color(base, v_factor, s_factor)






def deck_plot(optimized_data, management_forms, management_colors, es, radius_km=5):

    import numpy as np

    def make_pie_polygons(lat, lon, values, es_values=None, radius_km=10):
        """
        Returns:
            pie_polygons: list of slice polygons
            radar_polygon: single polygon (or None)

        values: portfolio shares (pie)
        es_values: list of ES values (radar overlay)
        """

        # --- PIE PART (unchanged) ---
        total = sum(values)
        start_angle = 0
        pie_polygons = []

        lon_factor = np.cos(np.deg2rad(lat))

        for v in values:
            fraction = v / total if total != 0 else 0
            end_angle = start_angle + fraction * 360

            points = [[lon, lat]]
            angles = np.linspace(start_angle, end_angle, max(2, int(fraction * 30)))

            for angle in angles:
                rad = np.deg2rad(angle)
                dlon = (radius_km / 111) * np.cos(rad) / lon_factor
                dlat = (radius_km / 111) * np.sin(rad)
                points.append([lon + dlon, lat + dlat])

            points.append([lon, lat])
            pie_polygons.append(points)
            start_angle = end_angle

        # --- RADAR PART ---
        radar_polygon = None

        if es_values is not None and len(es_values) > 0:
            es_values = np.array(es_values, dtype=float)

            n = len(es_values)
            angles = np.linspace(0, 360, n, endpoint=False)

            radar_points = []

            for val, angle in zip(es_values, angles):
                rad = np.deg2rad(angle)

                # scale radius by ES value
                r = radius_km * val

                dlon = (r / 111) * np.cos(rad) / lon_factor
                dlat = (r / 111) * np.sin(rad)

                radar_points.append([lon + dlon, lat + dlat])

            # close polygon
            radar_points.append(radar_points[0])
            radar_polygon = radar_points

            return pie_polygons, radar_polygon

    poly_data = []
    radar_data = []

    for i, row in optimized_data.iterrows():
        slices, radar = make_pie_polygons(
            row["lat"],
            row["lon"],
            row[management_forms].values,
            row[es],   # ES values for radar
            radius_km
        )

        # --- Pie slices ---
        for j, poly in enumerate(slices):
            poly_data.append({
                "polygon": poly,
                "name": "Portfolio",
                "value": row[management_forms].values[j],
                "slice": management_forms[j],
                "color": management_colors[j]
            })

        # --- Radar polygon ---
        if radar is not None:
            radar_data.append({
                "polygon": radar,
                "name": "Ecosystem Services",
                "value": ", ".join([f"{v:.2f}" for v in row[es]]),
                "color": [0, 0, 0, 80]  # semi-transparent black
            })

    # DataFrames
    poly_df = pd.DataFrame(poly_data)
    radar_df = pd.DataFrame(radar_data)

    # --- Pie layer ---
    pie_layer = pdk.Layer(
        "PolygonLayer",
        data=poly_df,
        get_polygon="polygon",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True
    )

    radar_layer = pdk.Layer(
        "PolygonLayer",
        data=radar_df,
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[0, 0, 0],
        get_line_width=2,

        line_width_units="pixels",   # <-- KEY FIX
        line_width_min_pixels=1,     # ensures visibility

        pickable=True
    )

    # --- View ---
    view_state = pdk.ViewState(
        latitude=optimized_data["lat"].mean(),
        longitude=optimized_data["lon"].mean(),
        zoom=8
    )

    # --- Deck ---
    deck = pdk.Deck(
        layers=[pie_layer, radar_layer],  # <-- add radar here
        initial_view_state=view_state,
        tooltip={"text": "{name}\n{slice}: {value}"},
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )

    return deck


brandenburg_extent = [10.5, 15.5, 51, 54]


def draw_pie_inset(ax, ratios, x, y, management_colors, size=0.03):
    trans = ax.transData.transform((x, y))
    inv = ax.transAxes.inverted().transform(trans)

    pie_ax = ax.inset_axes([inv[0]-size/2, inv[1]-size/2, size, size])

    pie_ax.pie(
        ratios,
        colors=management_colors,
        wedgeprops=dict(width=0.75)  # 👈 donut!
    )

    pie_ax.set_aspect('equal')
    pie_ax.axis('off')


def plot_portfolios_on_hexagons(ds, management_options, management_colors):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_extent(brandenburg_extent, crs=ccrs.PlateCarree())
    ax.add_geometries(
        countries_polygons['geometry'], crs=ccrs.PlateCarree(),
        facecolor='none', edgecolor='0.2', linewidth=0.5
    )

    gpd.GeoDataFrame(ds, geometry='geometry').plot(ax=ax, alpha=0.3, edgecolor='1.0', linewidth=1.0)

    coords = ds.geometry.apply(lambda x: x.representative_point())

    for (_, row), point in zip(ds.iterrows(), coords):
        draw_pie_inset(ax, row[management_options], point.x, point.y, management_colors, size=0.07)