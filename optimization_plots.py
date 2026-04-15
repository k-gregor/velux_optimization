import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import cartopy.crs as ccrs
from cartopy.io import shapereader
countries_polygons = gpd.read_file(shapereader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces'))


def deck_plot(optimized_data, management_forms, management_colors):

    radius_km = 5  # radius of pies in km

    def make_pie_polygons(lat, lon, values, radius_km=5):
        """
        Returns a list of polygons (one per slice) around (lat, lon)
        Corrects for longitude scaling at the given latitude.
        """
        total = sum(values)
        start_angle = 0
        polygons = []

        # factor to correct lon distance at this latitude
        lon_factor = np.cos(np.deg2rad(lat))

        for i, v in enumerate(values):
            fraction = v / total if total != 0 else 0
            end_angle = start_angle + fraction * 360

            points = [[lon, lat]]  # center
            angles = np.linspace(start_angle, end_angle, max(2, int(fraction*30)))
            for angle in angles:
                rad = np.deg2rad(angle)
                dlon = (radius_km / 111) * np.cos(rad) / lon_factor  # scale longitude
                dlat = (radius_km / 111) * np.sin(rad)
                points.append([lon + dlon, lat + dlat])
            points.append([lon, lat])
            polygons.append(points)
            start_angle = end_angle

        return polygons

    # Flatten all slices into a DataFrame
    poly_data = []
    for i, row in optimized_data.iterrows():
        slices = make_pie_polygons(row["lat"], row["lon"], row[management_forms].values, radius_km)
        for j, poly in enumerate(slices):
            poly_data.append({
                "polygon": poly,
                "name": 'Portfolio',
                "value": row[management_forms].values[j],
                "slice": management_forms[j],
                "color": management_colors[j]
            })

    poly_df = pd.DataFrame(poly_data)

    # PolygonLayer
    layer = pdk.Layer(
        "PolygonLayer",
        data=poly_df,
        get_polygon="polygon",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True
    )

    # View centered over your data
    view_state = pdk.ViewState(
        latitude=optimized_data["lat"].mean(),
        longitude=optimized_data["lon"].mean(),
        zoom=8
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\n{slice}: {value}"}
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