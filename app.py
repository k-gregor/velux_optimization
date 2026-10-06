import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import generic_optimization as go
import multilevel_plot as mp

# Start with:
# conda activate velux_opt && streamlit run app.py

st.set_page_config(layout="wide")
st.title("Robust forest optimization")

st.markdown(
    "Upload a CSV in long format: one row per **grid cell × climate scenario × ecosystem service**, "
    "with the coordinates, a scenario column, an ecosystem-service column and one column per management option."
)

st.markdown("Example (column names are arbitrary, you map them below; lower values are not better, higher is always better):")
st.code(
    "Lon;Lat;ssp;variable;spbau_manbau;spbau_shortrot;tobd_manbau\n"
    "-8.75;42.75;ssp126;cpool;9.24;9.24;9.24\n"
    "-8.75;42.75;ssp126;shannon;1.10;0.95;1.32\n"
    "-8.75;42.75;ssp585;cpool;8.90;8.71;9.01\n"
    "-8.75;42.75;ssp585;shannon;1.05;0.90;1.28\n"
    "-8.25;42.25;ssp126;cpool;16.93;16.64;17.31\n"
    "...",
    language="text",
)
st.caption("Every grid cell needs one row for each combination of climate scenario and ecosystem service. "
           "Your data does **not** need to be normalized: for each cell, scenario and service the values are "
           "automatically rescaled across the management columns to 0 (worst) – 1 (best). "
           "Already normalized data works too.")

uploaded = st.file_uploader("Data file (csv, ';' / ',' / tab separated)", type=["csv", "txt", "gz"])
if uploaded is None:
    st.stop()


@st.cache_data
def load(file):
    return go.read_table(file)


df = load(uploaded)
st.caption(f"{len(df):,} rows, {len(df.columns)} columns")
with st.expander("Preview"):
    st.dataframe(df.head(20))

# ---- column mapping ----
cols = list(df.columns)


def pick(label, candidates):
    guess = go.guess_column(cols, candidates)
    return st.selectbox(label, cols, index=cols.index(guess) if guess is not None else 0)


c1, c2, c3, c4 = st.columns(4)
with c1:
    lon_col = pick("Longitude column", ["lon", "longitude", "x"])
with c2:
    lat_col = pick("Latitude column", ["lat", "latitude", "y"])
with c3:
    scen_col = pick("Climate scenario column", ["scenario", "ssp", "rcp", "climate"])
with c4:
    es_col = pick("Ecosystem service column", ["es", "ecoservice", "variable", "service"])

used = {lon_col, lat_col, scen_col, es_col}
remaining = [c for c in cols if c not in used]
numeric_remaining = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
managements = st.multiselect("Management columns", remaining, default=numeric_remaining)
if len(managements) < 2 or len({lon_col, lat_col, scen_col, es_col}) < 4:
    st.info("Select distinct columns for the four roles and at least two management columns.")
    st.stop()

# ---- selections ----
all_scenarios = sorted(df[scen_col].astype(str).unique())
all_es = sorted(df[es_col].astype(str).unique())

# The form keeps widget changes local to the browser: nothing reruns until "Start optimization" is pressed.
with st.form("optimization_settings", border=False):
    st.subheader("Climate scenarios")
    scenarios = st.multiselect("Scenarios to be robust against", all_scenarios, default=all_scenarios)

    st.subheader("Ecosystem service weights")
    weight_cols = st.columns(min(len(all_es), 5))
    weights, lower_better = {}, []
    for i, e in enumerate(all_es):
        with weight_cols[i % len(weight_cols)]:
            weights[e] = st.slider(e, 0, 5, 3, key=f"w_{e}")
            if st.checkbox("lower is better", key=f"lb_{e}"):
                lower_better.append(e)

    start = st.form_submit_button("Start optimization")

radar = st.checkbox("Overlay ecosystem service scores (radar outline)", value=True)

# everything that influences the result; used to tell whether the displayed result is still up to date
settings = dict(file=uploaded.name, size=uploaded.size, cols=(lon_col, lat_col, scen_col, es_col),
                managements=managements, scenarios=scenarios, weights=weights, lower_better=lower_better)

if start:
    es_used = [e for e in all_es if weights[e] > 0]
    if not scenarios or not es_used:
        st.error("Select at least one scenario and give at least one ecosystem service a weight > 0.")
        st.stop()
    prepared = go.prepare_data(df, lon_col, lat_col, scen_col, es_col, managements, lower_is_better=lower_better)
    prepared, n_dropped = go.drop_incomplete_cells(prepared, scenarios, es_used)
    if n_dropped:
        st.warning(f"{n_dropped} grid cells lack some selected scenario/service combination and were skipped.")
    if prepared.empty:
        st.error("No complete grid cells left.")
        st.stop()

    bar = st.progress(0.0, text="Optimizing grid cells...")
    cells = go.run_optimization(prepared, managements, scenarios, {e: weights[e] for e in es_used},
                                progress=lambda f: bar.progress(f, text="Optimizing grid cells..."))
    bar.empty()
    with st.spinner("Assigning regions..."):
        cells = go.assign_regions(cells)
    st.session_state.upload_result = dict(cells=cells, managements=managements, es=es_used, settings=settings)

res = st.session_state.get("upload_result")
if res is None:
    st.stop()

if res["settings"] != settings:
    st.warning("The settings or the data have changed since this result was computed. "
               "Click \"Start optimization\" to update it.")

cells, managements, es_used = res["cells"], res["managements"], res["es"]
cells = cells.assign(name=cells["lon"].round(2).astype(str) + ", " + cells["lat"].round(2).astype(str))
value_cols = managements + es_used
states = go.aggregate(cells, value_cols, ["country", "state"])
countries = go.aggregate(cells, value_cols, "country")

# pie radius ~ 40% of the median distance to the nearest neighbouring cell (capped for isolated cells)
xy = cells[["lon", "lat"]].to_numpy() * [np.cos(np.deg2rad(cells["lat"].mean())), 1.0]
nn = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([0.5])
radius_km = 0.4 * float(np.median(nn)) * 111

st.subheader("Result")
st.caption("Zoom out to aggregate: grid cells → states/provinces → countries (mean portfolio of the cells in each region).")
html = mp.multilevel_html(cells, states, countries, managements, mp.default_colors(managements), es_used,
                          cell_radius_km=radius_km, show_radar=radar)
components.html(html, height=700)

legend = " &nbsp; ".join(
    f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>&#9632;</span> {m}" for m, c in mp.default_colors(managements).items())
st.markdown(legend, unsafe_allow_html=True)

with st.expander("Results table"):
    st.dataframe(cells.drop(columns="name"))
    st.download_button("Download cell results (csv)", cells.drop(columns="name").to_csv(index=False), "optimization_result.csv")
