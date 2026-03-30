import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import optimization as opt
import geopandas as gpd
import optimization_plots

st.title("Brandenburg robust optimization")
data_for_optimizer = pd.read_csv('brandenburd_optimizer_data.csv').set_index(['ES', 'Germany_id', 'rid', 'RCPScenario'])
hexagons = gpd.read_file('hexagon_grid.shp').to_crs(epsg=4326)

# ['abovegroundCarbon', 'shannonIndex', 'evapotranspiration_mm', 'mean_soilwatercontent_mm', 'volumeHarvested']
param_c = st.slider("Importance of C sequestration", 0, 5, 3)
param_bio = st.slider("Importance of biodiversity", 0, 5, 3)
param_et = st.slider("Importance of evapotranspiration", 0, 5, 3)
param_water = st.slider("Importance of soil water", 0, 5, 3)
param_harv = st.slider("Importance of harvests", 0, 5, 3)




# Trigger computation
if st.button("Optimierung starten (~30s)"):

    weights = [float(param_c), float(param_bio), float(param_et), float(param_water), float(param_harv)]
    sum_of_weights = np.sum(weights)
    if sum_of_weights > 0:
        weights /= sum_of_weights
    optimized_data = data_for_optimizer.groupby(['rid', 'Germany_id'], group_keys=False).apply(lambda gc: opt.optimize_gridcell(gc, gc.name, es_weights=weights))
    mean_portfolios = optimized_data.groupby('Germany_id').apply(opt.compute_mean_portfolios)
    #%%
    final_data = mean_portfolios.reset_index().merge(
        hexagons,
        right_on="Germany_id",
        left_on="Germany_id"
    )

    final_data["lon"] = final_data.geometry.apply(lambda x: x.representative_point().x)
    final_data["lat"] = final_data.geometry.apply(lambda x: x.representative_point().y)

    deck = optimization_plots.deck_plot(final_data)

    st.pydeck_chart(deck)