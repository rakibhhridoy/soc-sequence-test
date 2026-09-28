# Data

Nothing here is version-controlled. This file records where each input comes from.

| Dataset | Use | Source |
|---|---|---|
| LUCAS topsoil 2009, 2015, 2018 | SOC target and soil properties, repeated at the same points | European Soil Data Centre (ESDAC), on request; see `LUCAS_HOWTO.md` |
| Landsat 5, 7, 8, 9 Collection-2 surface reflectance | monthly dynamic covariates | Google Earth Engine, via `scripts/export_covariates.py` |
| ERA5-Land monthly means | temperature, precipitation and soil water for the climate check | Copernicus Climate Data Store, via `scripts/extract_era5land.py` |

Raw downloads go in `raw/` and are never edited in place. Everything in `interim/` and
`processed/` is rebuilt by the scripts. The LUCAS licence does not permit redistribution,
so none of these files, and no table keyed to LUCAS point identifiers, is published.
