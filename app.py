import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import optimization as opt
import geopandas as gpd
import optimization_plots


# Start by `conda activate velux_opt && streamlit run app.py`

st.title("Brandenburg robust optimization")
data_for_optimizer = pd.read_csv('brandenburg_optimizer_data_normalized.csv')
hexagons = gpd.read_file('hexagon_grid.shp').to_crs(epsg=4326)

# ['abovegroundCarbon', 'shannonIndex', 'evapotranspiration_mm', 'mean_soilwatercontent_mm', 'volumeHarvested']
param_c = st.slider("Importance of C sequestration", 0, 5, 3)
param_bio = st.slider("Importance of biodiversity", 0, 5, 3)
param_et = st.slider("Importance of evapotranspiration", 0, 5, 3)
param_water = st.slider("Importance of soil water", 0, 5, 3)
param_harv = st.slider("Importance of harvests", 0, 5, 3)

ILAND_CLIMATE_MODELS = ['ICHEC-EC-EARTH', 'MPI-M-MPI-ESM-LR', 'NCC-NorESM1-M']
ILAND_CLIMATE_SCENARIOS = ['rcp26', 'rcp45', 'rcp85']
ILAND_MANAGEMENTS = ["high-structure", "low-structure", "medium-structure", "no-mgmt"]
ILAND_ES = ['abovegroundCarbon', 'shannonIndex', 'evapotranspiration_mm', 'mean_soilwatercontent_mm', 'volumeHarvested']
ILAND_COLORS = [
    [255, 165, 0, 160],   # orange
    [50, 205, 50, 160],   # limegreen
    [0, 100, 0, 160],     # darkgreen
    [128, 0, 128, 160]    # purple
]

if "deck" not in st.session_state:
    st.session_state.deck = None

if st.button("Optimierung starten (~30s)"):
    weights = np.array([param_c, param_bio, param_et, param_water, param_harv], dtype=float)
    if weights.sum() > 0:
        weights /= weights.sum()

    optimized_data = data_for_optimizer.groupby(['rid', 'Germany_id'], group_keys=False).apply(
        lambda gc: opt.optimize_gridcell(
            gc, gc.name,
            location_names=['rid', 'Germany_id'],
            management_options=ILAND_MANAGEMENTS,
            climate_scenarios=ILAND_CLIMATE_SCENARIOS,
            es=ILAND_ES,
            scenario_columnname='RCPScenario',
            es_columnname='ES',
            es_weights=weights
        )
    )

    mean_portfolios = optimized_data.groupby('Germany_id').apply(opt.compute_mean_portfolios)

    final_data = mean_portfolios.reset_index().merge(
        hexagons,
        right_on="Germany_id",
        left_on="Germany_id"
    )

    final_data["lon"] = final_data.geometry.apply(lambda x: x.representative_point().x)
    final_data["lat"] = final_data.geometry.apply(lambda x: x.representative_point().y)

    st.session_state.deck = optimization_plots.deck_plot(
        final_data,
        management_forms=ILAND_MANAGEMENTS,
        management_colors=ILAND_COLORS
    )

# 👇 ALWAYS display if available
if st.session_state.deck is not None:
    st.pydeck_chart(st.session_state.deck)