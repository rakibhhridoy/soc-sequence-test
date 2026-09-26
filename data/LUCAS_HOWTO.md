# Getting the LUCAS topsoil data

ESDAC blocks automated requests, so this could not be fetched and checked page by page.
The procedure below follows ESDAC's documented access route; expect small differences in
wording, and follow what the site actually says.

## 1. Register

1. Go to the European Soil Data Centre, `esdac.jrc.ec.europa.eu`.
2. Find the LUCAS topsoil pages. Each campaign has its own page, titled along the lines
   of "LUCAS 2009 TOPSOIL data", "LUCAS 2015 TOPSOIL data" and "LUCAS 2018 TOPSOIL data".
   A 2022 campaign may also be listed; take it if it is there, since a fourth time point
   materially strengthens the temporal validation.
3. Each page carries a request form rather than a direct download. Complete it with your
   name, institutional email, organisation and a one-line statement of purpose.

   Suggested purpose: *Research use. Evaluating a convolutional-recurrent deep learning
   model for forecasting change in soil organic carbon, using repeated LUCAS topsoil
   observations. Non-commercial; results to be published.*

4. A download link arrives by email, usually within a day or two.

Use your university address (`mdrakib-2019417268@swe.du.ac.bd`). Institutional addresses
are approved more readily than free webmail.

## 2. What to download

For each campaign, take the **topsoil point data** tables, not the interpolated raster
maps. The point tables carry one row per sampled location with its coordinates and
measured properties. The interpolated maps are model output and cannot serve as a
training target.

The needed columns are the point identifier, coordinates, survey year, organic carbon,
and the supporting properties: pH, nitrogen, clay, sand, silt, cation exchange capacity,
bulk density where present, and land cover or land use.

## 3. Where to put it

Save the files unchanged into `data/raw/lucas/`, one folder per campaign:

```
data/raw/lucas/2009/...
data/raw/lucas/2015/...
data/raw/lucas/2018/...
```

Do not edit them. Everything downstream is rebuilt from `data/raw/` by script.

## 4. The number that decides the project

The point identifier links a location across campaigns. The first task once the files
arrive is to count how many points carry a usable organic carbon value in **every**
campaign, restricted to cropland and grassland.

That count caps what any model can be shown to do. It gets established and recorded
before any model is written. If it turns out small, the plan changes and the paper says
so.

## 5. Citation

LUCAS requires acknowledgement of the European Commission's Joint Research Centre as the
source, and the campaign papers should be cited. Take the exact citation from each
dataset page when the download link arrives, and add it to `manuscript/refs.bib`
verified against CrossRef, as with every other reference in this project.
