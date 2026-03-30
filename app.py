import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

st.title("Brandenburg robust optimization")

optimizer_data = pd.read_csv('brandenburg_optimizer_data.csv')


param = st.slider("Importance of C sequestration", 1, 10, 5)




# Trigger computation
if st.button("Optimierung starten"):
    # Your Python computation here
    # Example: generate random data based on param
    n = 5
    df = pd.DataFrame({
        "lat": np.random.uniform(48.0, 48.5, n),
        "lon": np.random.uniform(11.0, 11.7, n),
        "high": np.random.randint(0, param*5, n),
        "low": np.random.randint(0, param*5, n),
        "medium": np.random.randint(0, param*5, n),
        "none": np.random.randint(0, param*5, n),
        "name": [f"Point {i}" for i in range(n)]
    })

    cols = ["high", "low", "medium", "none"]
    colors = [
        [255, 165, 0, 160],
        [50, 205, 50, 160],
        [0, 100, 0, 160],
        [128, 0, 128, 160]
    ]
    radius_km = 5

    def make_pie_polygons(lat, lon, values, radius_km=5):
        total = sum(values)
        start_angle = 0
        polygons = []
        lon_factor = np.cos(np.deg2rad(lat))
        for i, v in enumerate(values):
            fraction = v / total if total != 0 else 0
            end_angle = start_angle + fraction * 360
            points = [[lon, lat]]
            angles = np.linspace(start_angle, end_angle, max(2, int(fraction*30)))
            for angle in angles:
                rad = np.deg2rad(angle)
                dlon = (radius_km / 111) * np.cos(rad) / lon_factor
                dlat = (radius_km / 111) * np.sin(rad)
                points.append([lon + dlon, lat + dlat])
            points.append([lon, lat])
            polygons.append(points)
            start_angle = end_angle
        return polygons

    poly_data = []
    for i, row in df.iterrows():
        slices = make_pie_polygons(row["lat"], row["lon"], row[cols].values, radius_km)
        for j, poly in enumerate(slices):
            poly_data.append({
                "polygon": poly,
                "name": row["name"],
                "value": row[cols].values[j],
                "slice": cols[j],
                "color": colors[j]
            })

    poly_df = pd.DataFrame(poly_data)

    layer = pdk.Layer(
        "PolygonLayer",
        data=poly_df,
        get_polygon="polygon",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=8
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\n{slice}: {value}"},
        # map_style="mapbox://styles/mapbox/dark-v11"
    )

    st.pydeck_chart(deck)