import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import optimization as opt
import geopandas as gpd
import optimization_plots

# Start with:
# conda activate velux_opt && streamlit run app.py



ILAND_CLIMATE_MODELS = ['ICHEC-EC-EARTH', 'MPI-M-MPI-ESM-LR', 'NCC-NorESM1-M']
ILAND_CLIMATE_SCENARIOS = ['rcp26', 'rcp45', 'rcp85']
ILAND_MANAGEMENTS = ["high-structure", "low-structure", "medium-structure", "no-mgmt"]
ILAND_ES = ['abovegroundCarbon', 'shannonIndex', 'evapotranspiration_mm', 'mean_soilwatercontent_mm', 'volumeHarvested']
ILAND_COLORS = [
    [255, 165, 0, 160],
    [50, 205, 50, 160],
    [0, 100, 0, 160],
    [128, 0, 128, 160]
]

LPJ_ES = ['cpool', 'shannon', 'harvest', 'litter']
LPJ_CLIMATE_SCENARIOS = ['ssp126', 'ssp585']
LPJ_MANAGEMENT_OPTIONS = [
    'spbau_manbau', 'spbau_shortrot', 'spbau_longrot', 'spbau_stop',
    'tobd_manbau', 'tobd_shortrot', 'tobd_longrot',
    'tone_manbau', 'tone_shortrot', 'tone_longrot'
]
LPJ_MANAGEMENT_COLORS = [optimization_plots.get_color(m) for m in LPJ_MANAGEMENT_OPTIONS]



st.set_page_config(layout="wide")

st.title("Robust forest optimization")



# =========================
# CACHED DATA LOADING
# =========================
@st.cache_data
def load_iland_data():
    return pd.read_csv('brandenburg_optimizer_data_normalized.csv')

@st.cache_data
def load_lpj_data():
    return pd.read_csv('lpjguess_optimizer_data_normalized.csv', sep=';')

@st.cache_data
def load_hexagons():
    return gpd.read_file('hexagon_grid.shp').to_crs(epsg=4326)

iland_data_for_optimizer = load_iland_data()
lpj_data_for_optimizer = load_lpj_data()
hexagons = load_hexagons()




col_iland, col_lpj = st.columns(2)

with col_iland:
    st.subheader("iLand weights")
    param_c = st.slider("C sequestration", 0, 5, 3, key="iland_c")
    param_bio = st.slider("Species diversity", 0, 5, 3, key="iland_bio")
    param_et = st.slider("Evapotranspiration", 0, 5, 3, key="iland_et")
    param_water = st.slider("Soil water", 0, 5, 3, key="iland_water")
    param_harv = st.slider("Harvests", 0, 5, 3, key="iland_harv")

    st.subheader("iLand climate scenarios")
    iland_selected_scenarios = st.multiselect(
        "Select climate scenarios:",
        ILAND_CLIMATE_SCENARIOS,
        default=ILAND_CLIMATE_SCENARIOS  # optional default
    )

    iland_button = st.button("Start iLand optimization (~30s)")

with col_lpj:
    st.subheader("LPJ-GUESS weights")
    lpj_param_c = st.slider("C sequestration", 0, 5, 3, key="lpj_c")
    lpj_param_litter = st.slider("Litter", 0, 5, 3, key="lpj_litter")
    lpj_param_shannon = st.slider("Shannon", 0, 5, 3, key="lpj_shannon")
    lpj_param_harv = st.slider("Harvest", 0, 5, 3, key="lpj_harv")

    st.subheader("iLand climate scenarios")
    lpj_selected_scenarios = st.multiselect(
        "Select climate scenarios:",
        LPJ_CLIMATE_SCENARIOS,
        default=LPJ_CLIMATE_SCENARIOS  # optional default
    )

    lpj_button = st.button("Start LPJ optimization (~5s)")


st.divider()



if "deck_iland" not in st.session_state:
    st.session_state.deck_iland = None

if "deck_lpj" not in st.session_state:
    st.session_state.deck_lpj = None

# =========================
# BUTTONS
# =========================

# ---- iLand optimization ----
if iland_button:
    with st.spinner("Running iLand optimization..."):
        weights = np.array([param_c, param_bio, param_et, param_water, param_harv], dtype=float)
        if weights.sum() > 0:
            weights /= weights.sum()

        optimized_data = iland_data_for_optimizer.groupby(['rid', 'Germany_id'], group_keys=False).apply(
            lambda gc: opt.optimize_gridcell(
                gc, gc.name,
                location_names=['rid', 'Germany_id'],
                management_options=ILAND_MANAGEMENTS,
                climate_scenarios=iland_selected_scenarios,
                es=ILAND_ES,
                scenario_columnname='RCPScenario',
                es_columnname='ES',
                es_weights=weights
            )
        )

        mean_portfolios = optimized_data.groupby('Germany_id').apply(opt.compute_mean_portfolios)

        results = mean_portfolios.reset_index().merge(
            hexagons,
            on="Germany_id"
        )

        results["lon"] = results.geometry.apply(lambda x: x.representative_point().x)
        results["lat"] = results.geometry.apply(lambda x: x.representative_point().y)

        st.session_state.deck_iland = optimization_plots.deck_plot(
            results,
            management_forms=ILAND_MANAGEMENTS,
            management_colors=ILAND_COLORS
        )

# ---- LPJ optimization ----
if lpj_button:
    with st.spinner("Running LPJ optimization..."):

        # TODO should be done with a dictionary to make sure it matches the right one.
        lpj_weights = np.array([
            lpj_param_c,
            lpj_param_shannon,
            lpj_param_harv,
            lpj_param_litter
        ], dtype=float)

        if lpj_weights.sum() > 0:
            lpj_weights /= lpj_weights.sum()

        optimized_data = lpj_data_for_optimizer.groupby(['Lon', 'Lat'], group_keys=False).apply(
            lambda gc: opt.optimize_gridcell(
                gc.reset_index(),
                location=gc.name,
                management_options=LPJ_MANAGEMENT_OPTIONS,
                climate_scenarios=lpj_selected_scenarios    ,
                location_names=['Lon', 'Lat'],
                scenario_columnname='ssp',
                es_columnname='ES',
                es=LPJ_ES,
                es_weights=lpj_weights
            )
        )

        results = optimized_data.reset_index().rename(columns={'Lon': 'lon', 'Lat': 'lat'})

        st.session_state.deck_lpj = optimization_plots.deck_plot(
            results,
            management_forms=LPJ_MANAGEMENT_OPTIONS,
            management_colors=LPJ_MANAGEMENT_COLORS,
            radius_km=15
        )

# =========================
# DISPLAY
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.session_state.deck_iland is not None:
        st.subheader("iLand Optimization")
        st.pydeck_chart(st.session_state.deck_iland)

with col2:
    if st.session_state.deck_lpj is not None:
        st.subheader("LPJ-GUESS Optimization")
        st.pydeck_chart(st.session_state.deck_lpj)