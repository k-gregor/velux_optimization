"""
Generic (data-agnostic) optimization pipeline for uploaded data in long format:
one row per (Lat, Lon, climate scenario, ecosystem service) and one column per management option.
"""
import numpy as np
import pandas as pd
import geopandas as gpd

import optimization as opt


def guess_column(columns, candidates):
    """Returns the first column whose lower-case name contains one of the candidates (in order)."""
    lowered = {c: str(c).lower() for c in columns}
    for cand in candidates:
        for c, low in lowered.items():
            if low == cand:
                return c
    for cand in candidates:
        for c, low in lowered.items():
            if cand in low:
                return c
    return None


def read_table(file):
    # sep=None lets pandas sniff ',' vs ';' vs tab
    return pd.read_csv(file, sep=None, engine="python")


def prepare_data(df, lon, lat, scenario, es, managements, lower_is_better=()):
    """Selects and renames the relevant columns and normalizes every (cell, scenario, ES) row across the
    management options to [0, 1]. A cell is one Lon/Lat pair (the 'unit' column is its identifier).
    Services in `lower_is_better` are negated before normalizing, so that 1 is always the best management."""
    data = df[[lon, lat, scenario, es] + managements].copy()
    data.columns = ["Lon", "Lat", "scenario", "ES"] + managements
    data["unit"] = data["Lon"].astype(str) + "|" + data["Lat"].astype(str)
    data["scenario"] = data["scenario"].astype(str)
    data["ES"] = data["ES"].astype(str)
    data[managements] = data[managements].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=managements)
    flip = data["ES"].isin([str(e) for e in lower_is_better])
    data.loc[flip, managements] = -data.loc[flip, managements]
    data[managements] = opt.normalize(data[managements])
    return data


def drop_incomplete_cells(data, scenarios, es_list):
    """The LP needs every selected (scenario, ES) combination in each unit. Returns (complete data, #dropped units)."""
    data = data[data["scenario"].isin(scenarios) & data["ES"].isin(es_list)]
    data = data.drop_duplicates(subset=["unit", "scenario", "ES"])
    counts = data.groupby("unit").size()
    complete = counts[counts == len(scenarios) * len(es_list)].index
    return data[data["unit"].isin(complete)], len(counts) - len(complete)


def run_optimization(data, managements, scenarios, es_weights, progress=None):
    """
    :param data: output of prepare_data (already filtered by drop_incomplete_cells)
    :param es_weights: dict ES -> weight (>= 0, not all zero)
    :return: DataFrame with one row per location (Lon/Lat): lon, lat, portfolio shares per management, and the
             worst-case (over climate scenarios) weighted ES score per ES. Units sharing a location are averaged.
    """
    es_list = list(es_weights.keys())
    weights = np.array([es_weights[e] for e in es_list], dtype=float)
    weights /= weights.sum()

    portfolios = {}
    groups = data.groupby("unit", sort=False)
    n = groups.ngroups
    for i, (unit, gc) in enumerate(groups):
        portfolios[unit] = opt.optimize_gridcell(
            gc.copy(), unit,
            location_names=["unit"],
            management_options=managements,
            climate_scenarios=scenarios,
            es=es_list,
            scenario_columnname="scenario",
            es_columnname="ES",
            es_weights=weights,
        )
        if progress is not None and (i % 20 == 0 or i == n - 1):
            progress((i + 1) / n)

    port = pd.DataFrame.from_dict(portfolios, orient="index")
    port.index.name = "unit"

    # achieved score per unit/scenario/ES = sum_m share_m * normalized value_m; the worst scenario counts
    merged = data.merge(port.reset_index(), on="unit", suffixes=("", "_w"))
    merged["score"] = np.einsum("ij,ij->i", merged[managements].to_numpy(),
                                merged[[m + "_w" for m in managements]].to_numpy())
    scores = merged.groupby(["unit", "ES"])["score"].min().unstack("ES")[es_list]

    locations = data.drop_duplicates("unit").set_index("unit")[["Lon", "Lat"]]
    per_unit = locations.join(port).join(scores)
    result = per_unit.groupby(["Lon", "Lat"]).mean().reset_index().rename(columns={"Lon": "lon", "Lat": "lat"})
    result["n_units"] = per_unit.groupby(["Lon", "Lat"]).size().to_numpy()
    return result


# ---------------------------------------------------------------------------
# Regional aggregation
# ---------------------------------------------------------------------------
_regions_cache = {}


NATURAL_EARTH_ADMIN1 = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"


def _load_regions():
    """Natural Earth admin-1 (states/provinces) with country names; downloaded once per process."""
    if "gdf" not in _regions_cache:
        gdf = gpd.read_file(NATURAL_EARTH_ADMIN1, encoding="utf-8").set_crs(4326, allow_override=True)
        _regions_cache["gdf"] = gdf[["name", "admin", "geometry"]].rename(columns={"name": "state", "admin": "country"})
    return _regions_cache["gdf"]


def assign_regions(cells):
    """Adds 'state' and 'country' columns (Natural Earth admin-1 / admin-0). Cells that fall just outside every
    polygon (e.g. coastal cells of a coarse grid) get the nearest region within ~1 degree."""
    regions = _load_regions()
    pts = gpd.GeoDataFrame(cells[["lon", "lat"]].copy(),
                           geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs=4326)
    pts["_id"] = np.arange(len(pts))
    joined = gpd.sjoin(pts, regions, how="left", predicate="within").drop_duplicates("_id")
    missing = joined["state"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(pts[pts["_id"].isin(joined.loc[missing, "_id"])], regions,
                                 how="left", max_distance=1.0).drop_duplicates("_id")
        joined = joined.set_index("_id")
        near = near.set_index("_id")
        joined.loc[near.index, ["state", "country"]] = near[["state", "country"]].to_numpy()
        joined = joined.reset_index()
    out = cells.copy()
    out["state"] = joined["state"].fillna("Unknown").to_numpy()
    out["country"] = joined["country"].fillna("Unknown").to_numpy()
    return out


def aggregate(cells, value_cols, by):
    """Mean of portfolio shares / ES scores over all cells of each region. Mean of simplex points is on the simplex.
    Position = mean of the cell positions. Region names are put into 'name'."""
    keys = by if isinstance(by, list) else [by]
    agg = cells.groupby(keys)[value_cols + ["lon", "lat"]].mean()
    agg["n_cells"] = cells.groupby(keys).size()
    agg = agg.reset_index()
    agg["name"] = agg[keys].astype(str).agg(", ".join, axis=1)
    return agg
