"""Dynamic covariate series from Earth Engine.

Landsat 5/7/8/9 surface reflectance carries every survey round on one record, so the
2009, 2015 and 2018 LUCAS points are treated identically. Sentinel-2 begins only in
2015, and switching sensor partway through would let a sensor change masquerade as a
temporal signal.

Each observation gets a monthly series running back ``n_months`` from its sampling date,
which is the five-year window fixed in the project decisions.
"""
from __future__ import annotations
from typing import Sequence
import pandas as pd

EE_PROJECT = "ee-arsenicbd"

# Landsat Collection-2 Level-2 surface reflectance, harmonised band names.
_SENSORS = {
    "LANDSAT/LT05/C02/T1_L2": dict(bands=["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                                   start="1984-01-01", end="2012-05-05"),
    "LANDSAT/LE07/C02/T1_L2": dict(bands=["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                                   start="1999-01-01", end="2024-01-01"),
    "LANDSAT/LC08/C02/T1_L2": dict(bands=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                                   start="2013-04-01", end="2099-01-01"),
    "LANDSAT/LC09/C02/T1_L2": dict(bands=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                                   start="2021-11-01", end="2099-01-01"),
}
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]
# NDWI and MNDWI carry the water regime. In flooded rice the surface is under water for
# part of every year, and submergence is the mechanism that slows decomposition, so the
# flooding signal is data rather than noise and is given its own channels instead of
# being masked away.
INDICES = ["ndvi", "ndwi", "mndwi"]


def _indices(img):
    """NDVI (vegetation), NDWI (open water), MNDWI (water, less confused by built-up)."""
    return (img.normalizedDifference(["nir", "red"]).rename("ndvi")
            .addBands(img.normalizedDifference(["green", "nir"]).rename("ndwi"))
            .addBands(img.normalizedDifference(["green", "swir1"]).rename("mndwi")))


def init(project: str = EE_PROJECT):
    import ee
    ee.Initialize(project=project)
    return ee


def _masked_collection(ee, start: str, end: str, region):
    """Cloud- and shadow-masked Landsat, scaled to reflectance, bands renamed.

    Everything here runs server-side; nothing inside a mapped function may call
    ``getInfo``, so bands are renamed to the common set before masking and the mask then
    selects them by their fixed names.
    """
    def prep(img):
        qa = img.select("QA_PIXEL")
        # Collection-2 QA_PIXEL bits: 3 cloud, 4 cloud shadow, 5 snow
        clear = (qa.bitwiseAnd(1 << 3).eq(0)
                 .And(qa.bitwiseAnd(1 << 4).eq(0))
                 .And(qa.bitwiseAnd(1 << 5).eq(0)))
        sr = img.select(COMMON).multiply(2.75e-05).add(-0.2)
        return sr.updateMask(clear).copyProperties(img, ["system:time_start"])

    cols = []
    for cid, meta in _SENSORS.items():
        s0, e0 = max(start, meta["start"]), min(end, meta["end"])
        if s0 >= e0:          # sensor does not overlap the window; an empty range errors
            continue
        c = (ee.ImageCollection(cid)
             .filterDate(s0, e0)
             .filterBounds(region)
             .select(meta["bands"] + ["QA_PIXEL"], COMMON + ["QA_PIXEL"])
             .map(prep))
        cols.append(c)
    if not cols:
        raise ValueError(f"no Landsat sensor covers {start} to {end}")
    out = cols[0]
    for c in cols[1:]:
        out = out.merge(c)
    return out


def monthly_series(points: pd.DataFrame, end_date: str, n_months: int = 60,
                   scale: int = 30, project: str = EE_PROJECT,
                   id_col: str = "point_id", lat_col: str = "lat", lon_col: str = "lon"
                   ) -> pd.DataFrame:
    """Monthly median reflectance and NDVI at each point, for the n_months before end_date.

    Returns a long table: one row per point per month. Months with no clear observation
    come back as NaN and are handled downstream, not silently filled here.
    """
    ee = init(project)
    start = (pd.Timestamp(end_date) - pd.DateOffset(months=n_months)).strftime("%Y-%m-%d")
    feats = [ee.Feature(ee.Geometry.Point([float(r[lon_col]), float(r[lat_col])]),
                        {"point_id": str(r[id_col])}) for _, r in points.iterrows()]
    fc = ee.FeatureCollection(feats)
    coll = _masked_collection(ee, start, end_date, fc.geometry().bounds())

    months = pd.date_range(start, end_date, freq="MS")[:n_months]
    rows = []
    for m in months:
        m_end = (m + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        month_coll = coll.filterDate(m.strftime("%Y-%m-%d"), m_end)
        if month_coll.size().getInfo() == 0:      # no scene that month; leave the gap
            continue
        img = month_coll.median()
        idx = _indices(img)
        sampled = img.addBands(idx).reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=scale).getInfo()
        for f in sampled["features"]:
            p = f["properties"]
            rows.append({"point_id": p.get("point_id"), "month": m.strftime("%Y-%m"),
                         **{b: p.get(b) for b in COMMON + INDICES}})
    return pd.DataFrame(rows)


def export_year(points: "pd.DataFrame", year: int, description: str | None = None,
                scale: int = 30, project: str = EE_PROJECT, folder: str = "soc_article",
                id_col: str = "point_id", lat_col: str = "lat", lon_col: str = "lon"):
    """Export one calendar year of monthly covariates for every point, to Drive.

    Point-by-point sampling costs about three seconds per month, which is impossible for
    thousands of points, so the work goes to Earth Engine's batch queue instead. Each
    task carries twelve months of nine variables as 108 bands, sampled once per point.

    Splitting by calendar year rather than by survey window keeps each task small and
    lets the five-year window for any round be assembled locally afterwards, without
    exporting the overlapping years more than once.

    Returns the started task; poll it with ``task.status()``.
    """
    ee = init(project)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    feats = [ee.Feature(ee.Geometry.Point([float(r[lon_col]), float(r[lat_col])]),
                        {"point_id": str(r[id_col])}) for _, r in points.iterrows()]
    fc = ee.FeatureCollection(feats)
    coll = _masked_collection(ee, start, end, fc.geometry().bounds())

    stack = None
    for m in pd.date_range(start, end, freq="MS")[:12]:
        m_end = (m + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        img = coll.filterDate(m.strftime("%Y-%m-%d"), m_end).median()
        img = img.addBands(_indices(img)).select(COMMON + INDICES)
        img = img.rename([f"{b}_{m.strftime('%Y%m')}" for b in COMMON + INDICES])
        stack = img if stack is None else stack.addBands(img)

    task = ee.batch.Export.table.toDrive(
        collection=stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale),
        description=description or f"lucas_covariates_{year}",
        folder=folder, fileFormat="CSV")
    task.start()
    return task


def window_years(survey_year: int, n_years: int = 5) -> list[int]:
    """Calendar years of the covariate window for a survey.

    The window ends in the December *before* the survey year, so no month after the soil
    was sampled can enter the predictors. LUCAS does not publish sampling dates per
    point, and including the survey year would risk conditioning a forecast on weather
    that followed the measurement.
    """
    return list(range(survey_year - n_years, survey_year))
