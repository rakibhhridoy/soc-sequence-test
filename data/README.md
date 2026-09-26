# Data

Nothing here is version-controlled. This file records where each input comes from and
how to rebuild it.

## Required

| Dataset | Use | Source | Status |
|---|---|---|---|
| LUCAS topsoil 2009 / 2015 / 2018 | SOC target, repeated at the same points | ESDAC, European Commission (registration needed) | to download |
| Sentinel-2 / Landsat composites | dynamic reflectance series | Google Earth Engine | to export |
| ERA5-Land, CHIRPS | dynamic climate series | Copernicus CDS; Climate Hazards Center | to export |
| SoilGrids 2.0 | static soil covariates where measured values are missing | ISRIC | to download |

## Do not use as a training target

`../../../SOC/data/SOC_Properties_40Years_1985_2025.csv` is a derived file, not a
record of measurements. `Generate_40Year_Data.py` interpolates the soil properties
between the 1985 and 2025 surveys, so consecutive years carry identical values, and its
`SOC_Satellite_Derived` column is a ridge-regression fit on the same satellite indices
that would serve as predictors. It is usable only as an illustration of that earlier
method, and a model trained on it would be learning the interpolation.

## Counts to establish first

The number of LUCAS points that genuinely repeat across rounds sets the ceiling on what
any model can be shown to do. Establish and record it before building anything.
